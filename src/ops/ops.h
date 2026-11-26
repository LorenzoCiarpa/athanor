/* ops.h: the numeric operators of athanor.
 *
 * Every op has a CPU reference, <op>_ref, written to be obviously correct and
 * never optimized: scalar loops, arithmetic and accumulation in fp32 (float),
 * one rounding to the output dtype at the end. GPU kernels (<op>_gpu, step 7)
 * and optional fast CPU paths are tested against these functions, which in
 * turn are tested against the reference implementation on real weights
 * (tests/test_golden.c). The math of each op follows docs/MODELS.md and
 * docs/contract/<model>.md; nothing here knows which model calls it.
 *
 * Conventions shared by every _ref function:
 *
 *   - Tensors are contiguous, row-major and host-addressable (tier RAM or
 *     PINNED). Activations and weights are f32 or bf16, in any combination;
 *     the dtype of each output tensor is the dtype the result is rounded to.
 *     Token ids and positions are i32.
 *   - "Rows" are the leading dimensions flattened: an op over the last
 *     dimension of [T, heads, d] treats it as T * heads rows of length d.
 *   - Where the reference implementation rounds an intermediate value to the
 *     activation dtype and the contract records it, the op does the same and
 *     says so (rmsnorm_gated, rope). Every other intermediate stays in fp32.
 *   - Outputs never alias inputs, except where stated.
 *   - Errors: ATH_ERR_INVALID_ARG for missing tensors or inconsistent shapes,
 *     ATH_ERR_TYPE for unsupported dtypes, ATH_ERR_UNSUPPORTED for tensors
 *     that are not in host memory or limits stated below, ATH_ERR_RANGE for
 *     token ids outside the table. The message is logged at error level.
 *
 * GPU kernels (step 7). ath_<op>_gpu takes the device first, then the
 * arguments of ath_<op>_ref, with the same shapes, dtypes and math, and every
 * tensor in that device's memory (tier VRAM; ATH_ERR_UNSUPPORTED otherwise).
 *
 *   - Asynchronous: the work is queued on the device's compute stream
 *     (ATH_STREAM_COMPUTE) and the function returns. Ops queued one after
 *     the other run in that order; the host must synchronize the stream, or
 *     copy on it, before it reads an output.
 *   - The errors above are checked on the host before anything is queued.
 *     A fault while a kernel runs is reported by the next call that waits
 *     for the stream (src/backend/backend.h).
 *   - Results match the reference within tolerances, not to the bit: the
 *     kernels sum in another order, fuse multiply-adds, and the device's
 *     exp, log, sin and cos round differently from libm. tests/README.md,
 *     GPU kernels, has the tolerances and what was measured.
 *   - Limits beyond the reference's are stated per function.
 *   - In a build without a GPU backend (make) every _gpu function checks its
 *     arguments, then fails with ATH_ERR_UNSUPPORTED.
 *
 * One entry point per op (step 8), ath_<op>(d, ...) at the end of this
 * header, runs it where a forward pass runs: on device d when d is not NULL
 * (ath_<op>_gpu, every tensor in d's memory), on the CPU otherwise
 * (ath_<op>_ref, or the fast path where there is one). A family composes
 * its forward pass from these, once for both devices.
 */
#ifndef ATH_OPS_H
#define ATH_OPS_H

#include "backend/backend.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "core/threads.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Activation fused into an op. */
typedef enum {
    ATH_ACT_NONE,
    ATH_ACT_SILU      /* x * sigmoid(x) */
} ath_act;

/* ---- embed.c --------------------------------------------------------------- */

/* out[i, :] = table[ids[i], :].
 * table [V, H]; ids [n] i32, each in [0, V); out [n, H]. */
ath_status ath_embed_ref(const ath_tensor *table, const ath_tensor *ids, ath_tensor *out);
/* The ids are on the device, so they are not checked against the table: an
 * id outside [0, V) gives a row of NaN, and nothing is read outside the
 * table. ath_model_forward checks token ids before they reach the op. */
ath_status ath_embed_gpu(const ath_device *d, const ath_tensor *table, const ath_tensor *ids, ath_tensor *out);

/* ---- matmul.c -------------------------------------------------------------- */

/* out = x · wᵀ: out[r, m] = Σ_k x[r, k] w[m, k], summed in order of k.
 * x [..., K]; w [M, K] (the layout of the checkpoint); out [..., M] with the
 * leading dimensions of x. x and out are f32 or bf16; w is f32, bf16 or
 * Q8_0 (step 12). A Q8_0 weight is summed in the canonical order of
 * src/ops/q8_0.h instead (blocks of 32 terms, 16 partial sums, a fixed
 * tree), which every implementation follows, the GPU's included: the same
 * bits on both sides (invariant 7, step 12). */
ath_status ath_matmul_ref(const ath_tensor *x, const ath_tensor *w, ath_tensor *out);

/* The same contract and the same bits as ath_matmul_ref, faster: several
 * outputs at a time per thread, spread over the threads of pool (NULL: the
 * calling thread only). The CPU forward pass uses it; src/ops/README.md,
 * Fast CPU paths. */
ath_status ath_matmul_fast(const ath_tensor *x, const ath_tensor *w, ath_tensor *out, ath_pool *pool);

/* The GPU matmul: cuBLASLt when it has the dtypes (x and w bf16 with out bf16
 * or f32, or all three f32), accumulating in fp32 (never TF32); athanor's
 * own kernel, slower, for the other combinations (src/ops/cuda/matmul.cu). */
ath_status ath_matmul_gpu(const ath_device *d, const ath_tensor *x, const ath_tensor *w, ath_tensor *out);

/* Several weights that share their input (step 9): out[i] = x · w[i]ᵀ for
 * i < n <= ATH_MATMUL_N_MAX, with the results of n calls of ath_matmul,
 * bit for bit. On the GPU a few rows of x against bf16 weights take one
 * GEMV launch for all of them: the projections of one layer's input are
 * read in one pass. The CPU (in dispatch) runs ath_matmul_fast on each. */
#define ATH_MATMUL_N_MAX 4

/* GPU fusion of the decode step (step 9): out = out + x · wᵀ, the product
 * rounded to out's dtype first, with the bits of ath_matmul into a
 * temporary followed by ath_add; the residual stream takes each output
 * projection this way. It takes what the GEMV kernel takes
 * (ath_matmul_fits_gemv) and fails with ATH_ERR_UNSUPPORTED otherwise, where
 * the caller uses the two ops. (Normalizing x in the same launch was
 * measured and dropped: the blocks' prologue cost more than the norm
 * kernel it saved. Adding a prompt's product inside cuBLASLt, beta = 1, was
 * measured and dropped in step 10: one rounding instead of the reference's
 * two moved a greedy token above the noise in the gate of step 8.) */
bool       ath_matmul_fits_gemv(const ath_tensor *x, const ath_tensor *w);
/* GPU fusion of a SwiGLU MLP's first half for one row of x (step 9):
 * out = silu(x · wgᵀ) · (x · wuᵀ), each product rounded to out's dtype
 * first: the bits of ath_matmul_n into two tensors of out's dtype followed
 * by ath_swiglu, without writing the two. ATH_ERR_UNSUPPORTED where
 * ath_matmul_swiglu_fits is false. */
bool       ath_matmul_swiglu_fits(const ath_tensor *x, const ath_tensor *wg, const ath_tensor *wu);
ath_status ath_matmul_swiglu_gpu(const ath_device *d, const ath_tensor *x, const ath_tensor *wg, const ath_tensor *wu,
                                 ath_tensor *out);
ath_status ath_matmul_add_gpu(const ath_device *d, const ath_tensor *x, const ath_tensor *w, ath_tensor *out);
ath_status ath_matmul_n_gpu(const ath_device *d, const ath_tensor *x, int n, const ath_tensor *const *w,
                            ath_tensor *const *out);

/* ---- norm.c ---------------------------------------------------------------- */

/* RMSNorm over the last dimension:
 *   out = x / sqrt(mean(x²) + eps) · (w_offset + w)
 * w_offset 0 is the plain parameterization, 1 the zero-centered one (the
 * stored weight is w and the scale is 1 + w). x, out [..., D]; w [D]. */
ath_status ath_rmsnorm_ref(const ath_tensor *x, const ath_tensor *w, float w_offset, float eps,
                           ath_tensor *out);
/* GPU fusion of the residual add and the next norm (step 10): x ← x + y,
 * rounded to x's dtype, then out = rmsnorm(x); the bits of ath_add followed
 * by ath_rmsnorm, x read once. y like x. Rows of up to 1024 elements, a
 * multiple of 8, 16-byte aligned buffers (ath_add_rmsnorm_fits says);
 * ATH_ERR_UNSUPPORTED otherwise, where the caller uses the two ops. */
bool       ath_add_rmsnorm_fits(const ath_tensor *x, const ath_tensor *y, const ath_tensor *w);
ath_status ath_add_rmsnorm_gpu(const ath_device *d, ath_tensor *x, const ath_tensor *y, const ath_tensor *w,
                               float w_offset, float eps, ath_tensor *out);
ath_status ath_rmsnorm_gpu(const ath_device *d, const ath_tensor *x, const ath_tensor *w, float w_offset,
                           float eps, ath_tensor *out);

/* Gated RMSNorm over the last dimension, SiLU gate, plain weight:
 *   n   = x / sqrt(mean(x²) + eps), rounded to the dtype of x
 *   out = (w · n) · silu(z)
 * The rounding of n is the reference's (contract, Gated RMSNorm).
 * x, z, out [..., D]; w [D]. */
ath_status ath_rmsnorm_gated_ref(const ath_tensor *x, const ath_tensor *z, const ath_tensor *w,
                                 float eps, ath_tensor *out);
ath_status ath_rmsnorm_gated_gpu(const ath_device *d, const ath_tensor *x, const ath_tensor *z,
                                 const ath_tensor *w, float eps, ath_tensor *out);

/* L2 normalization over the last dimension: out = x / sqrt(Σ x² + eps).
 * x, out [..., D]. */
ath_status ath_l2norm_ref(const ath_tensor *x, float eps, ath_tensor *out);
ath_status ath_l2norm_gpu(const ath_device *d, const ath_tensor *x, float eps, ath_tensor *out);

/* ---- rope.c ---------------------------------------------------------------- */

/* Rotary position embedding, rotate-half convention, on the first rope_dim
 * dimensions of every head; the other dimensions are copied. For a head
 * vector x at position p, with h = rope_dim / 2 and i in [0, h):
 *   f_i = 1 / theta^(2i / rope_dim),  c = cos(p f_i),  s = sin(p f_i)
 *   out[i]     = x[i] c − x[i + h] s
 *   out[i + h] = x[i + h] c + x[i] s
 * c and s are computed in fp32 and rounded to the dtype of x, like the
 * reference's cos/sin tables (contract, Full-attention layer).
 * x, out [T, heads, head_dim]; pos [T] i32; rope_dim even, <= head_dim. */
ath_status ath_rope_ref(const ath_tensor *x, const ath_tensor *pos, int32_t rope_dim, float theta,
                        ath_tensor *out);
/* GPU: rope_dim <= ATH_ROPE_MAX_DIM. */
#define ATH_ROPE_MAX_DIM 512
ath_status ath_rope_gpu(const ath_device *d, const ath_tensor *x, const ath_tensor *pos, int32_t rope_dim,
                        float theta, ath_tensor *out);

/* ---- act.c ----------------------------------------------------------------- */

/* out = silu(gate) · up, elementwise. gate, up, out of the same shape. */
ath_status ath_swiglu_ref(const ath_tensor *gate, const ath_tensor *up, ath_tensor *out);
ath_status ath_swiglu_gpu(const ath_device *d, const ath_tensor *gate, const ath_tensor *up, ath_tensor *out);

/* out = x · sigmoid(gate), elementwise. x, gate, out of the same shape. */
ath_status ath_sigmoid_mul_ref(const ath_tensor *x, const ath_tensor *gate, ath_tensor *out);
ath_status ath_sigmoid_mul_gpu(const ath_device *d, const ath_tensor *x, const ath_tensor *gate, ath_tensor *out);

/* out = a + b, elementwise. a, b, out of the same shape; out may alias a or b. */
ath_status ath_add_ref(const ath_tensor *a, const ath_tensor *b, ath_tensor *out);
ath_status ath_add_gpu(const ath_device *d, const ath_tensor *a, const ath_tensor *b, ath_tensor *out);

/* ---- attn.c ---------------------------------------------------------------- */

/* Causal softmax attention with grouped-query heads.
 *   q [T, nh, d], k [S, nkv, d], v [S, nkv, dv], out [T, nh, dv], nh a
 *   multiple of nkv. Query t sits at position p_t and sees keys 0 … p_t:
 *   pos NULL: the T queries are the last T of the S positions, p_t = S − T + t
 *             (S >= T);
 *   pos i32 [T] (step 9): p_t = pos[t] < S, so k and v may be a whole cache
 *             of which each query reads the rows up to its position.
 *   Query head h reads key/value head h / (nh / nkv).
 *   out = Σ_j exp(s_j − max s) v_j / Σ_j exp(s_j − max s),  s_j = scale (q · k_j)
 * Scores, softmax and the weighted sum are fp32; dv <= ATH_ATTN_MAX_DV. */
#define ATH_ATTN_MAX_DV 1024
ath_status ath_attn_ref(const ath_tensor *q, const ath_tensor *k, const ath_tensor *v, const ath_tensor *pos,
                        float scale, ath_tensor *out);
/* GPU: the query/key head dim d is also at most ATH_ATTN_MAX_DV. The keys
 * are taken in pieces of ATH_ATTN_SPLIT, each reduced to a partial softmax
 * and combined in order; with a scratch arena, up to ATH_ATTN_SPLIT_MAX_T
 * queries spread their pieces over blocks (split-K, for decode over a long
 * cache), and the result has the same bits as without. Positions in device
 * memory cannot be checked by the host: a query whose position is outside
 * [0, S) gets a row of NaN. */
#define ATH_ATTN_SPLIT 64
#define ATH_ATTN_SPLIT_MAX_T 8
/* From ATH_ATTN_MMA_MIN_T queries on (prompts, step 10), with bf16 q, k, v
 * and out, d == dv a multiple of 16 up to 256, on an NVIDIA GPU of compute
 * capability 8 or more: FlashAttention on the tensor cores (attn_mma.cu).
 * Scores and sums are fp32 as above, the probabilities that multiply the
 * values are rounded to bf16 (tests/README.md, Prompt kernels); every call
 * of this kind gives a query the same bits whatever the other queries. */
#define ATH_ATTN_MMA_MIN_T 16
ath_status ath_attn_gpu(const ath_device *d, const ath_tensor *q, const ath_tensor *k, const ath_tensor *v,
                        const ath_tensor *pos, float scale, ath_dev_arena *scratch, ath_tensor *out);

/* ---- kv.c (step 14): the KV cache in int8 ---------------------------------------
 *
 * A cache [S, nkv, d] of ATH_DTYPE_I8 with a scale per group of g
 * consecutive values of a row (one token, one KV head) in scale
 * [S, nkv, d / g] f32; g follows from the two shapes (it divides d). A
 * group x is stored as q = rint(x / s), s = max|x| / 127 (all zero for a
 * zero group), and a value reads back as q · s rounded to the activation
 * dtype (bf16 or f32): the int8 cache is the activation-dtype cache those
 * values make, and attention over it (ath_attn_q8) gives, bit for bit, what
 * ath_attn gives over that cache. The same arithmetic on the CPU and the
 * GPU: the same bits. The Qwen models use g = 32 (a scale per row of 256
 * cost 2.4 times the bf16 noise in KL, almost all of it in the values:
 * tests/README.md, KV cache in int8). */

/* The rows of x [T, nkv, d] (f32 or bf16) into the rows pos[t] of cache and
 * scale. On the GPU pos is in device memory: a row whose position is
 * outside [0, S) is not written. */
ath_status ath_kv_store_ref(const ath_tensor *x, const ath_tensor *pos, ath_tensor *cache, ath_tensor *scale);
ath_status ath_kv_store_gpu(const ath_device *d, const ath_tensor *x, const ath_tensor *pos, ath_tensor *cache,
                            ath_tensor *scale);
/* Rows [0, n) of cache read back into out [n, nkv, d] (f32 or bf16: the
 * dtype they round to). */
ath_status ath_kv_load_ref(const ath_tensor *cache, const ath_tensor *scale, ath_tensor *out);
ath_status ath_kv_load_gpu(const ath_device *d, const ath_tensor *cache, const ath_tensor *scale, ath_tensor *out);
/* ath_attn over int8 keys k [S, nkv, d] and values v [S, nkv, dv] with
 * their scales ks [S, nkv, d / g], vs [S, nkv, dv / gv], read back in q's
 * dtype. On the GPU the kernels of short batches (decode), whatever the
 * number of queries; the fast ones when g and gv are multiples of 8. */
ath_status ath_attn_q8_ref(const ath_tensor *q, const ath_tensor *k, const ath_tensor *ks, const ath_tensor *v,
                           const ath_tensor *vs, const ath_tensor *pos, float scale, ath_tensor *out);
ath_status ath_attn_q8_gpu(const ath_device *d, const ath_tensor *q, const ath_tensor *k, const ath_tensor *ks,
                           const ath_tensor *v, const ath_tensor *vs, const ath_tensor *pos, float scale,
                           ath_dev_arena *scratch, ath_tensor *out);

/* ---- scatter.c (step 9) ---------------------------------------------------- */

/* dst[idx[i]] = src[i] for every row i: src [n, ...] and dst [R, ...] of the
 * same dtype with rows of the same number of elements, idx i32 [n] with
 * values in [0, R). Rows are copied byte for byte. The KV cache takes the
 * keys and values of new tokens at their positions this way. On the GPU idx
 * is in device memory and cannot be checked: a row whose index is outside
 * [0, R) is not written. */
ath_status ath_scatter_rows_ref(const ath_tensor *src, const ath_tensor *idx, ath_tensor *dst);
ath_status ath_scatter_rows_gpu(const ath_device *d, const ath_tensor *src, const ath_tensor *idx, ath_tensor *dst);

/* ---- conv.c ---------------------------------------------------------------- */

/* Causal depthwise convolution over time, with a running state:
 *   xx = [state; x]   (K − 1 + T rows)
 *   out[t, c] = act(Σ_{j<K} w[c, j] xx[t + j, c])
 * so the last tap w[c, K − 1] multiplies the current input. Afterwards the
 * state holds the last K − 1 rows of xx: the inputs the next call needs.
 * x, out [T, C]; w [C, K] or [C, 1, K] (the checkpoint layout); state
 * [K − 1, C], read and overwritten (zero it at the start of a sequence). */
ath_status ath_conv_ref(const ath_tensor *x, const ath_tensor *w, ath_tensor *state, ath_act act,
                        ath_tensor *out);
ath_status ath_conv_gpu(const ath_device *d, const ath_tensor *x, const ath_tensor *w, ath_tensor *state,
                        ath_act act, ath_tensor *out);

/* ---- gdn.c ----------------------------------------------------------------- */

/* Gated DeltaNet gates, per token and value head:
 *   g    = −exp(a_log) · softplus(a + dt_bias)     (α = exp(g) is the decay, in (0, 1])
 *   beta = sigmoid(b)                              (the write strength)
 * softplus(x) = log(1 + exp(x)), and x itself above 20 (torch's threshold).
 * a, b, g, beta [T, nv]; a_log, dt_bias [nv]. With a_given (step 12) a_log
 * holds A = −exp(A_log) itself, as llama.cpp's converter stores it in a
 * GGUF file (ssm_a), and g = A · softplus(a + dt_bias). */
ath_status ath_gdn_gates_ref(const ath_tensor *a, const ath_tensor *b, const ath_tensor *a_log, bool a_given,
                             const ath_tensor *dt_bias, ath_tensor *g, ath_tensor *beta);
ath_status ath_gdn_gates_gpu(const ath_device *d, const ath_tensor *a, const ath_tensor *b,
                             const ath_tensor *a_log, bool a_given, const ath_tensor *dt_bias, ath_tensor *g,
                             ath_tensor *beta);

/* Gated delta rule, recurrent form. For every token t and value head h, with
 * the state S [dk, dv] of that head:
 *   S ← exp(g_t) S
 *   u = Sᵀ k_t
 *   S ← S + k_t (beta_t (v_t − u))ᵀ
 *   out_t = Sᵀ (q_scale q_t)
 * where q_t and k_t are the query and key L2-normalized per head, as
 * ath_l2norm does with l2_eps, in fp32 (step 10: the op normalizes them, so
 * that the normalized values are never stored; l2_eps 0 leaves a unit
 * vector as it is). q, k [T, nk, dk]; v, out [T, nv, dv]; g, beta [T, nv];
 * state [nv, dk, dv] f32, read and updated in place (zero it at the start of
 * a sequence). Value head h reads query/key head h / (nv / nk) (the
 * reference's grouped order), or h mod nk with tiled (step 12: llama.cpp's
 * converter stores the value heads of a GGUF file in that order). */
ath_status ath_gdn_ref(const ath_tensor *q, const ath_tensor *k, const ath_tensor *v,
                       const ath_tensor *g, const ath_tensor *beta, float q_scale, float l2_eps, bool tiled,
                       ath_tensor *state, ath_tensor *out);
/* GPU: dk <= ATH_GDN_MAX_DK. The recurrent form, as the reference: one step
 * per token. With a scratch arena, from ATH_GDN_CHUNK_MIN_T tokens on, f32 g,
 * dk up to 128 and a multiple of 8, dv up to 128 and a multiple of 4
 * (prompts, step 10): the chunked form of the reference
 * (torch_chunk_gated_delta_rule, chunks of 64 tokens), which computes the
 * same recurrence in fp32 with another order of operations (tests/README.md,
 * Prompt kernels). Every call of this kind gives the same bits for a prompt
 * fed whole or in pieces whose lengths are multiples of 64. */
#define ATH_GDN_MAX_DK 256
#define ATH_GDN_CHUNK_MIN_T 64
ath_status ath_gdn_gpu(const ath_device *d, const ath_tensor *q, const ath_tensor *k, const ath_tensor *v,
                       const ath_tensor *g, const ath_tensor *beta, float q_scale, float l2_eps, bool tiled,
                       ath_tensor *state, ath_dev_arena *scratch, ath_tensor *out);

/* ---- moe.c (step 12) ---------------------------------------------------------
 *
 * A mixture-of-experts layer (docs/contract/qwen35-35b-a3b.md, MoE op
 * contract): the router picks k experts per token, each chosen expert runs
 * its SwiGLU MLP on the token, and the outputs are added with the router's
 * weights, plus the gated shared expert. Three ops: the router's choice, one
 * routed expert, and the weighted combination. */

/* The router's choice per token: p = softmax(logits) over the n experts in
 * fp32, the k largest (on ties the lower index), and their weights p / Σ p
 * over the k, stored in w's dtype. logits [T, n] f32; ids [T, k] i32, sorted
 * by expert index (the order in which the reference adds the experts, and
 * the order of ath_moe_combine); w [T, k] f32 or bf16. k <= ATH_MOE_MAX_K. */
#define ATH_MOE_MAX_K 16
ath_status ath_moe_route_ref(const ath_tensor *logits, ath_tensor *ids, ath_tensor *w);
ath_status ath_moe_route_gpu(const ath_device *d, const ath_tensor *logits, ath_tensor *ids, ath_tensor *w);

/* One routed expert on every row of x: y = down · h with h = silu(g) · u,
 * g = gate · x and u = up · x, every product an ath_matmul_ref (for Q8_0
 * weights the canonical order) rounded to y's dtype, silu by the canonical
 * exponential (ath_q8_silu, src/ops/q8_0.h) so that the CPU and the GPU give
 * the same bits (invariant 7, step 12), h rounded to y's dtype. x [T, H],
 * gate and up [I, H], down [H, I], y [T, H]; x and y f32 or bf16, the
 * weights f32, bf16 or Q8_0. */
ath_status ath_moe_expert_ref(const ath_tensor *x, const ath_tensor *gate, const ath_tensor *up,
                              const ath_tensor *down, ath_tensor *y);

/* The routed experts of one layer on the GPU: for every token t and choice
 * c, y[t, c] = the MLP of expert ids[t, c] on x[t], with
 * ath_moe_expert_ref's bits for Q8_0 weights, wherever that expert's
 * weights are on the device; the entries of the other experts are left as
 * they are (the CPU computes them: ath_moe_experts_cpu). tab: in device
 * memory, 3 · n pointers, each expert's gate [I, H], then each expert's up
 * [I, H], then each expert's down [H, I], all of dtype wdt (Q8_0 or
 * Q8_0_R16: the canonical order, the reference's bits; f32 or bf16: within
 * tolerance); NULL for an expert not on the device. x [T, H], ids [T, k],
 * h [T, k, I] working memory and y [T, k, H] of x's dtype. */
ath_status ath_moe_experts_gpu(const ath_device *d, const ath_tensor *x, const ath_tensor *ids,
                               const void *const *tab, ath_dtype wdt, int64_t n_experts, int64_t inter, ath_tensor *h,
                               ath_tensor *y);

/* ---- moe_cpu.c: experts in host memory (step 12) ----
 *
 * The routed experts of one MoE layer, wherever they are: each expert's
 * gate [I, H], up [I, H] and down [H, I] in device memory (dev_tab, read by
 * the kernels at run time, so that a captured graph sees experts move), and
 * in host memory (host_tab) for the ones the CPU computes. A pipeline
 * (src/infer/tier.c) fills it; the forward pass passes it to the ops. */
typedef struct {
    int64_t            n_experts, inter, hidden;   /* n, I, H                                  */
    ath_dtype          dev_dtype;                  /* the device copies' dtype                 */
    const void *const *dev_tab;                    /* device memory: 3n pointers, every gate,
                                                    * then every up, then every down; NULL where
                                                    * an expert is not on the device          */
    ath_dtype          host_dtype;                 /* ATH_DTYPE_Q8_0_R16 or the dev_dtype       */
    const void *const *host_tab;                   /* host memory: 3n pointers in the same order;
                                                    * NULL when every expert is on the device */
    const uint8_t     *on_dev;                     /* host memory: [n], 1 where dev_tab has it  */
    uint32_t          *uses;                       /* host memory: [n] choices counted by the
                                                    * CPU's pass (the placement's statistics),
                                                    * or NULL                                  */
} ath_moe_experts;

/* Q8_0 rows regrouped for the CPU (core/tensor.h, ATH_DTYPE_Q8_0_R16): the
 * same values. src Q8_0 [rows, K], dst Q8_0_R16 [rows, K], K a multiple of
 * 512; host memory. */
ath_status ath_q8_0_repack(const ath_tensor *src, ath_tensor *dst);

/* The routed experts of one layer that are not on the device (on_dev 0),
 * on the CPU: y[t, c] for every token t and choice c whose expert ids[t, c]
 * has on_dev 0, with ath_moe_expert_ref's bits (the canonical order: the
 * same bits as the GPU's for the experts there, invariant 7); the other
 * entries are left as they are. With host copies in Q8_0_R16 and a CPU with
 * AVX-512, 16 partial sums in one register; otherwise the same operations
 * one lane at a time. Counts every choice in ex->uses. x [T, H], ids
 * [T, k], y [T, k, H], host memory; work memory from a (reset before
 * returning); rows spread over the threads of pool. */
ath_status ath_moe_experts_cpu(const ath_tensor *x, const ath_tensor *ids, const ath_moe_experts *ex, ath_tensor *y,
                               ath_pool *pool, ath_arena *a);

/* Working state of the experts split between a GPU and the CPU: pinned
 * buffers for the tokens of up to max_tokens rows at a time (the CPU's
 * inputs and outputs), the events that order the side stream, the CPU's
 * working memory and the jobs it runs. One per device and model (the
 * engine's), shared by the MoE layers in turn. */
typedef struct ath_moe_ws ath_moe_ws;
ath_status ath_moe_ws_create(ath_moe_ws **ws, const ath_device *d, ath_pool *pool, int64_t max_tokens, int64_t k,
                             int64_t hidden, int64_t inter, ath_arena *a);
void       ath_moe_ws_free(ath_moe_ws *ws);

/* The routed experts of one layer on a GPU, wherever they are: y[t, c] for
 * every token and choice, the experts on the device computed there
 * (ath_moe_experts_gpu) while the others run on the CPU from their host
 * copies (ath_moe_experts_cpu, called from the stream: ath_host_call), and
 * the CPU's outputs merged into y. Everything is queued on the device's
 * streams and nothing waits on the host, so a decode step that contains it
 * can be captured in a graph and replayed: the CPU's share runs again at
 * every replay, for the experts away from the device at that time. The
 * bits do not depend on where an expert is (invariant 7). x [T, H], ids
 * [T, k] on the device; h [T, k, I] device working memory, y [T, k, H]. */
ath_status ath_moe_experts_split(const ath_device *d, const ath_tensor *x, const ath_tensor *ids,
                                 const ath_moe_experts *ex, ath_moe_ws *ws, ath_tensor *h, ath_tensor *y);

/* The layer's output: out[t] = Σ_j w[t, j] · y[t, j] + sigmoid(sg[t]) · ys[t],
 * the routed sum as a chain of fused multiply-adds in the order of j (the
 * experts by index, ath_moe_route), the shared term added last, rounded
 * once to out's dtype. y [T, k, H], w [T, k], ys [T, H], sg [T, 1] (the
 * shared expert's output and its gate's logit; both NULL without a shared
 * expert), out [T, H]; f32 or bf16. */
ath_status ath_moe_combine_ref(const ath_tensor *y, const ath_tensor *w, const ath_tensor *ys, const ath_tensor *sg,
                               ath_tensor *out);
ath_status ath_moe_combine_gpu(const ath_device *d, const ath_tensor *y, const ath_tensor *w, const ath_tensor *ys,
                               const ath_tensor *sg, ath_tensor *out);

/* ---- dispatch.c (step 8): one entry point per op ------------------------------
 *
 * d NULL: the CPU (ath_<op>_ref; ath_matmul_fast on the threads of pool).
 * d set: the GPU kernel on d, asynchronous on its compute stream. */
ath_status ath_embed(const ath_device *d, const ath_tensor *table, const ath_tensor *ids, ath_tensor *out);
ath_status ath_matmul(const ath_device *d, ath_pool *pool, const ath_tensor *x, const ath_tensor *w, ath_tensor *out);
ath_status ath_rmsnorm(const ath_device *d, const ath_tensor *x, const ath_tensor *w, float w_offset, float eps,
                       ath_tensor *out);
ath_status ath_rmsnorm_gated(const ath_device *d, const ath_tensor *x, const ath_tensor *z, const ath_tensor *w,
                             float eps, ath_tensor *out);
ath_status ath_l2norm(const ath_device *d, const ath_tensor *x, float eps, ath_tensor *out);
ath_status ath_rope(const ath_device *d, const ath_tensor *x, const ath_tensor *pos, int32_t rope_dim, float theta,
                    ath_tensor *out);
ath_status ath_swiglu(const ath_device *d, const ath_tensor *gate, const ath_tensor *up, ath_tensor *out);
ath_status ath_sigmoid_mul(const ath_device *d, const ath_tensor *x, const ath_tensor *gate, ath_tensor *out);
ath_status ath_add(const ath_device *d, const ath_tensor *a, const ath_tensor *b, ath_tensor *out);
ath_status ath_matmul_n(const ath_device *d, ath_pool *pool, const ath_tensor *x, int n, const ath_tensor *const *w,
                        ath_tensor *const *out);
ath_status ath_attn(const ath_device *d, const ath_tensor *q, const ath_tensor *k, const ath_tensor *v,
                    const ath_tensor *pos, float scale, ath_dev_arena *scratch, ath_tensor *out);
ath_status ath_kv_store(const ath_device *d, const ath_tensor *x, const ath_tensor *pos, ath_tensor *cache,
                        ath_tensor *scale);
ath_status ath_kv_load(const ath_device *d, const ath_tensor *cache, const ath_tensor *scale, ath_tensor *out);
ath_status ath_attn_q8(const ath_device *d, const ath_tensor *q, const ath_tensor *k, const ath_tensor *ks,
                       const ath_tensor *v, const ath_tensor *vs, const ath_tensor *pos, float scale,
                       ath_dev_arena *scratch, ath_tensor *out);
ath_status ath_conv(const ath_device *d, const ath_tensor *x, const ath_tensor *w, ath_tensor *state, ath_act act,
                    ath_tensor *out);
ath_status ath_scatter_rows(const ath_device *d, const ath_tensor *src, const ath_tensor *idx, ath_tensor *dst);
ath_status ath_gdn_gates(const ath_device *d, const ath_tensor *a, const ath_tensor *b, const ath_tensor *a_log,
                         bool a_given, const ath_tensor *dt_bias, ath_tensor *g, ath_tensor *beta);
ath_status ath_gdn(const ath_device *d, const ath_tensor *q, const ath_tensor *k, const ath_tensor *v,
                   const ath_tensor *g, const ath_tensor *beta, float q_scale, float l2_eps, bool tiled,
                   ath_tensor *state, ath_dev_arena *scratch, ath_tensor *out);

/* ---- sample.c (step 5) ----------------------------------------------------- */

/* How the next token is chosen from the logits. The filters apply in the
 * order of transformers' logits warpers: temperature, top-k, top-p, min-p. */
typedef struct {
    float   temperature;   /* 0: greedy, the largest logit (the lowest id of equals)   */
    int32_t top_k;         /* keep the k most likely tokens; 0: all                   */
    float   top_p;         /* keep the most likely tokens until their probability
                            * reaches top_p (at least one); 1: all                    */
    float   min_p;         /* drop tokens less likely than min_p times the most
                            * likely one; 0: none                                     */
    float   presence;      /* presence penalty (step 13): subtracted from the logit of
                            * every token generated so far, before the filters. The
                            * caller applies it (ath_presence_ref, ath_presence_gpu;
                            * the decoder does); ath_sample_* read the logits as given.
                            * 0: none                                                 */
} ath_sampler;

/* One token id from logits [vocab] or [1, vocab] (f32 or bf16). u is a
 * uniform number in [0, 1) from the caller's generator (core/rng.h): the
 * token is where u falls in the cumulative distribution of the kept tokens,
 * ordered from the most likely (the lowest id first among equals), so the
 * op itself is deterministic. Greedy ignores u. Working memory (two arrays
 * of vocab entries) comes from scratch and is released before returning.
 * ATH_ERR_INVALID_ARG for temperature < 0, top_k < 0, top_p outside (0, 1]
 * or min_p outside [0, 1]. */
ath_status ath_sample_ref(const ath_tensor *logits, const ath_sampler *p, double u, ath_arena *scratch,
                          int32_t *out);

/* GPU (step 9): the same choice from logits in device memory, with u (one
 * double) and the token (one int32) in device memory too, so that a decode
 * step samples without the host. Greedy, or top_k from 1 to
 * ATH_SAMPLE_GPU_MAX_K; with every token a candidate (top_k 0, or above the
 * maximum) ATH_ERR_UNSUPPORTED, and the caller samples on the host. u
 * cannot be checked: it must be in [0, 1). Working memory comes from the
 * device scratch arena and is released before returning. */
/* The presence penalty, as vLLM and OpenAI apply it: penalty is subtracted
 * once from the logit of every token marked in seen (one byte per entry of
 * the vocabulary; nonzero: generated earlier in this answer), however many
 * times it came. In place, on f32 logits [vocab] or [1, vocab]. One
 * subtraction per logit, so the GPU gives the same bits. */
ath_status ath_presence_ref(ath_tensor *logits, const uint8_t *seen, float penalty);
/* GPU: logits, seen and token in device memory. *token (the token a decode
 * step runs: the last one generated) is marked in seen first, then the
 * penalty applies as above. */
ath_status ath_presence_gpu(const ath_device *d, ath_tensor *logits, uint8_t *seen, const int32_t *token,
                            float penalty);

#define ATH_SAMPLE_GPU_MAX_K 64
bool       ath_sample_gpu_fits(const ath_sampler *p);
ath_status ath_sample_gpu(const ath_device *d, const ath_tensor *logits, const ath_sampler *p, const double *u,
                          ath_dev_arena *scratch, int32_t *out);

#ifdef __cplusplus
}
#endif

#endif /* ATH_OPS_H */
