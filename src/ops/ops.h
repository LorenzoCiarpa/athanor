/* ops.h: the numeric operators of athanor.
 *
 * Every op has a CPU reference, <op>_ref, written to be obviously correct and
 * never optimized: scalar loops, arithmetic and accumulation in fp32 (float),
 * one rounding to the output dtype at the end. GPU kernels (step 7) and
 * optional fast CPU paths are tested against these functions, which in turn
 * are tested against the reference implementation on real weights
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
 * The device-dispatching entry points (one per op, choosing _ref, a fast CPU
 * path or a GPU kernel from where the tensors live) join this header with the
 * second implementation of an op, in step 7.
 */
#ifndef ATH_OPS_H
#define ATH_OPS_H

#include "core/tensor.h"
#include "core/threads.h"

/* Activation fused into an op. */
typedef enum {
    ATH_ACT_NONE,
    ATH_ACT_SILU      /* x * sigmoid(x) */
} ath_act;

/* ---- embed.c --------------------------------------------------------------- */

/* out[i, :] = table[ids[i], :].
 * table [V, H]; ids [n] i32, each in [0, V); out [n, H]. */
ath_status ath_embed_ref(const ath_tensor *table, const ath_tensor *ids, ath_tensor *out);

/* ---- matmul.c -------------------------------------------------------------- */

/* out = x · wᵀ: out[r, m] = Σ_k x[r, k] w[m, k], summed in order of k.
 * x [..., K]; w [M, K] (the layout of the checkpoint); out [..., M] with the
 * leading dimensions of x. */
ath_status ath_matmul_ref(const ath_tensor *x, const ath_tensor *w, ath_tensor *out);

/* The same contract and the same bits as ath_matmul_ref, faster: several
 * outputs at a time per thread, spread over the threads of pool (NULL: the
 * calling thread only). The CPU forward pass uses it; src/ops/README.md,
 * Fast CPU paths. */
ath_status ath_matmul_fast(const ath_tensor *x, const ath_tensor *w, ath_tensor *out, ath_pool *pool);

/* ---- norm.c ---------------------------------------------------------------- */

/* RMSNorm over the last dimension:
 *   out = x / sqrt(mean(x²) + eps) · (w_offset + w)
 * w_offset 0 is the plain parameterization, 1 the zero-centered one (the
 * stored weight is w and the scale is 1 + w). x, out [..., D]; w [D]. */
ath_status ath_rmsnorm_ref(const ath_tensor *x, const ath_tensor *w, float w_offset, float eps,
                           ath_tensor *out);

/* Gated RMSNorm over the last dimension, SiLU gate, plain weight:
 *   n   = x / sqrt(mean(x²) + eps), rounded to the dtype of x
 *   out = (w · n) · silu(z)
 * The rounding of n is the reference's (contract, Gated RMSNorm).
 * x, z, out [..., D]; w [D]. */
ath_status ath_rmsnorm_gated_ref(const ath_tensor *x, const ath_tensor *z, const ath_tensor *w,
                                 float eps, ath_tensor *out);

/* L2 normalization over the last dimension: out = x / sqrt(Σ x² + eps).
 * x, out [..., D]. */
ath_status ath_l2norm_ref(const ath_tensor *x, float eps, ath_tensor *out);

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

/* ---- act.c ----------------------------------------------------------------- */

/* out = silu(gate) · up, elementwise. gate, up, out of the same shape. */
ath_status ath_swiglu_ref(const ath_tensor *gate, const ath_tensor *up, ath_tensor *out);

/* out = x · sigmoid(gate), elementwise. x, gate, out of the same shape. */
ath_status ath_sigmoid_mul_ref(const ath_tensor *x, const ath_tensor *gate, ath_tensor *out);

/* out = a + b, elementwise. a, b, out of the same shape; out may alias a or b. */
ath_status ath_add_ref(const ath_tensor *a, const ath_tensor *b, ath_tensor *out);

/* ---- attn.c ---------------------------------------------------------------- */

/* Causal softmax attention with grouped-query heads.
 *   q [T, nh, d], k [S, nkv, d], v [S, nkv, dv], out [T, nh, dv], S >= T,
 *   nh a multiple of nkv. The T queries are the last T of the S positions:
 *   query t sits at position S − T + t and sees keys 0 … S − T + t.
 *   Query head h reads key/value head h / (nh / nkv).
 *   out = Σ_j exp(s_j − max s) v_j / Σ_j exp(s_j − max s),  s_j = scale (q · k_j)
 * Scores, softmax and the weighted sum are fp32; dv <= ATH_ATTN_MAX_DV. */
#define ATH_ATTN_MAX_DV 1024
ath_status ath_attn_ref(const ath_tensor *q, const ath_tensor *k, const ath_tensor *v, float scale,
                        ath_tensor *out);

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

/* ---- gdn.c ----------------------------------------------------------------- */

/* Gated DeltaNet gates, per token and value head:
 *   g    = −exp(a_log) · softplus(a + dt_bias)     (α = exp(g) is the decay, in (0, 1])
 *   beta = sigmoid(b)                              (the write strength)
 * softplus(x) = log(1 + exp(x)), and x itself above 20 (torch's threshold).
 * a, b, g, beta [T, nv]; a_log, dt_bias [nv]. */
ath_status ath_gdn_gates_ref(const ath_tensor *a, const ath_tensor *b, const ath_tensor *a_log,
                             const ath_tensor *dt_bias, ath_tensor *g, ath_tensor *beta);

/* Gated delta rule, recurrent form. For every token t and value head h, with
 * the state S [dk, dv] of that head:
 *   S ← exp(g_t) S
 *   u = Sᵀ k_t
 *   S ← S + k_t (beta_t (v_t − u))ᵀ
 *   out_t = Sᵀ (q_scale q_t)
 * q, k [T, nk, dk], already L2-normalized; v, out [T, nv, dv]; g, beta [T, nv];
 * state [nv, dk, dv] f32, read and updated in place (zero it at the start of
 * a sequence). Value head h reads query/key head h / (nv / nk). */
ath_status ath_gdn_ref(const ath_tensor *q, const ath_tensor *k, const ath_tensor *v,
                       const ath_tensor *g, const ath_tensor *beta, float q_scale,
                       ath_tensor *state, ath_tensor *out);

#endif /* ATH_OPS_H */
