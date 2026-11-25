/* conv.c: causal depthwise conv1d over time with a running state. See ops.h.
 *
 * Channel c of the output at token t is a K-tap filter over the inputs
 * t − K + 1 … t of the same channel. The state carries the K − 1 inputs
 * before the first token, so a sequence fed in pieces (prefill, then one
 * token at a time) gives the same outputs as fed at once. The reference keeps
 * K inputs per channel; the oldest never reaches an output (contract, Gated
 * DeltaNet layer).
 */
#include "ops/check.h"
#include "ops/launch.h"
#include "ops/ops.h"

#include <math.h>

static const char op[] = "conv";

/* Element (r, c) of xx = [state; x], the inputs seen by the filter. */
static float xx(const ath_tensor *state, const ath_tensor *x, int64_t r, int64_t c)
{
    const int64_t C = x->shape[1], past = state->shape[0];
    return r < past ? ld(state, r * C + c) : ld(x, (r - past) * C + c);
}

static ath_status args(const ath_tensor *x, const ath_tensor *w, const ath_tensor *state, ath_act act,
                       const ath_tensor *out, ops_where where)
{
    ATH_CHECK(ops_float(x, op, "x"));
    ATH_CHECK(ops_float(w, op, "w"));
    ATH_CHECK(ops_float(state, op, "state"));
    ATH_CHECK(ops_float(out, op, "out"));
    if (x->ndim != 2 || !ath_tensor_same_shape(x, out))
        return ops_fail(ATH_ERR_INVALID_ARG, op, "x and out must be [T, C]");
    const int64_t C = x->shape[1];
    const bool w2 = w->ndim == 2, w3 = w->ndim == 3 && w->shape[1] == 1;
    if (!(w2 || w3) || w->shape[0] != C)
        return ops_fail(ATH_ERR_INVALID_ARG, op, "w must be [%lld, K] or [%lld, 1, K]", (long long)C,
                        (long long)C);
    const int64_t K = w->shape[w->ndim - 1];
    if (K < 1 || state->ndim != 2 || state->shape[0] != K - 1 || state->shape[1] != C)
        return ops_fail(ATH_ERR_INVALID_ARG, op, "state must be [K - 1, C] = [%lld, %lld]",
                        (long long)(K - 1), (long long)C);
    if (act != ATH_ACT_NONE && act != ATH_ACT_SILU)
        return ops_fail(ATH_ERR_INVALID_ARG, op, "unknown activation %d", (int)act);
    return OPS_PLACE(op, where, x, w, state, out);
}

ath_status ath_conv_ref(const ath_tensor *x, const ath_tensor *w, ath_tensor *state, ath_act act,
                        ath_tensor *out)
{
    ATH_CHECK(args(x, w, state, act, out, OPS_HOST));
    const int64_t T = x->shape[0], C = x->shape[1], K = w->shape[w->ndim - 1];
    for (int64_t t = 0; t < T; t++)
        for (int64_t c = 0; c < C; c++) {
            float acc = 0.0f;
            for (int64_t j = 0; j < K; j++)
                acc += ld(w, c * K + j) * xx(state, x, t + j, c);
            if (act == ATH_ACT_SILU)
                acc = acc / (1.0f + expf(-acc));
            st(out, t * C + c, acc);
        }
    /* New state: rows T … T + K − 2 of xx. Ascending order is safe in place:
     * row T + i of xx is either a state row above i (not yet overwritten) or
     * an input row. */
    for (int64_t i = 0; i < K - 1; i++)
        for (int64_t c = 0; c < C; c++)
            st(state, i * C + c, xx(state, x, T + i, c));
    return ATH_OK;
}

ath_status ath_conv_gpu(const ath_device *d, const ath_tensor *x, const ath_tensor *w, ath_tensor *state,
                        ath_act act, ath_tensor *out)
{
    ATH_CHECK(args(x, w, state, act, out, OPS_GPU));
    return ath_conv_launch(d, x, w, state, act, out);
}
