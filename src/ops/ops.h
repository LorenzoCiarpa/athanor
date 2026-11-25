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
 * a, b, g, beta [T, nv]; a_log, dt_bias [nv]. */
ath_status ath_gdn_gates_ref(const ath_tensor *a, const ath_tensor *b, const ath_tensor *a_log,
                             const ath_tensor *dt_bias, ath_tensor *g, ath_tensor *beta);
ath_status ath_gdn_gates_gpu(const ath_device *d, const ath_tensor *a, const ath_tensor *b,
                             const ath_tensor *a_log, const ath_tensor *dt_bias, ath_tensor *g, ath_tensor *beta);

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
 * a sequence). Value head h reads query/key head h / (nv / nk). */
ath_status ath_gdn_ref(const ath_tensor *q, const ath_tensor *k, const ath_tensor *v,
                       const ath_tensor *g, const ath_tensor *beta, float q_scale, float l2_eps,
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
                       const ath_tensor *g, const ath_tensor *beta, float q_scale, float l2_eps, ath_tensor *state,
                       ath_dev_arena *scratch, ath_tensor *out);

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
ath_status ath_conv(const ath_device *d, const ath_tensor *x, const ath_tensor *w, ath_tensor *state, ath_act act,
                    ath_tensor *out);
ath_status ath_scatter_rows(const ath_device *d, const ath_tensor *src, const ath_tensor *idx, ath_tensor *dst);
ath_status ath_gdn_gates(const ath_device *d, const ath_tensor *a, const ath_tensor *b, const ath_tensor *a_log,
                         const ath_tensor *dt_bias, ath_tensor *g, ath_tensor *beta);
ath_status ath_gdn(const ath_device *d, const ath_tensor *q, const ath_tensor *k, const ath_tensor *v,
                   const ath_tensor *g, const ath_tensor *beta, float q_scale, float l2_eps, ath_tensor *state,
                   ath_dev_arena *scratch, ath_tensor *out);

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
#define ATH_SAMPLE_GPU_MAX_K 64
bool       ath_sample_gpu_fits(const ath_sampler *p);
ath_status ath_sample_gpu(const ath_device *d, const ath_tensor *logits, const ath_sampler *p, const double *u,
                          ath_dev_arena *scratch, int32_t *out);

#ifdef __cplusplus
}
#endif

#endif /* ATH_OPS_H */
