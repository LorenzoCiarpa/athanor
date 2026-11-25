/* attn.c: causal softmax attention with grouped-query heads. See ops.h.
 *
 * Naive and exact: for each query and head, one pass for the largest score,
 * one pass that accumulates exp(s_j − max) and exp(s_j − max) · v_j, and a
 * final division. Scores are recomputed in the second pass instead of being
 * stored, so the function needs no memory beyond the dv accumulators.
 * The reference (eager_attention_forward) rounds the scores and the
 * probabilities to bf16; this function keeps both in fp32 (docs/MODELS.md:
 * softmax is an fp32 island).
 */
#include "ops/check.h"
#include "ops/launch.h"
#include "ops/ops.h"

#include <math.h>

static const char op[] = "attn";

static ath_status args(const ath_tensor *q, const ath_tensor *k, const ath_tensor *v, const ath_tensor *out,
                       ops_where where)
{
    ATH_CHECK(ops_float(q, op, "q"));
    ATH_CHECK(ops_float(k, op, "k"));
    ATH_CHECK(ops_float(v, op, "v"));
    ATH_CHECK(ops_float(out, op, "out"));
    if (q->ndim != 3 || k->ndim != 3 || v->ndim != 3 || out->ndim != 3)
        return ops_fail(ATH_ERR_INVALID_ARG, op, "q, k, v and out must be [tokens, heads, dim]");
    const int64_t T = q->shape[0], nh = q->shape[1], d = q->shape[2];
    const int64_t S = k->shape[0], nkv = k->shape[1], dv = v->shape[2];
    if (k->shape[2] != d || v->shape[0] != S || v->shape[1] != nkv || out->shape[0] != T ||
        out->shape[1] != nh || out->shape[2] != dv || S < T || nkv <= 0 || nh % nkv != 0)
        return ops_fail(ATH_ERR_INVALID_ARG, op,
                        "q [T, nh, d], k [S, nkv, d], v [S, nkv, dv], out [T, nh, dv], S >= T, "
                        "nh a multiple of nkv");
    if (dv > ATH_ATTN_MAX_DV)
        return ops_fail(ATH_ERR_UNSUPPORTED, op, "value head dim %lld > %d", (long long)dv,
                        ATH_ATTN_MAX_DV);
    return OPS_PLACE(op, where, q, k, v, out);
}

ath_status ath_attn_ref(const ath_tensor *q, const ath_tensor *k, const ath_tensor *v, float scale,
                        ath_tensor *out)
{
    ATH_CHECK(args(q, k, v, out, OPS_HOST));
    const int64_t T = q->shape[0], nh = q->shape[1], d = q->shape[2];
    const int64_t S = k->shape[0], nkv = k->shape[1], dv = v->shape[2];
    const int64_t group = nh / nkv;
    float acc[ATH_ATTN_MAX_DV];
    for (int64_t t = 0; t < T; t++) {
        const int64_t last = S - T + t;                 /* last key this query sees */
        for (int64_t h = 0; h < nh; h++) {
            const int64_t kh = h / group;
            const int64_t qb = (t * nh + h) * d;
            float max = -INFINITY;
            for (int64_t j = 0; j <= last; j++) {
                float s = 0.0f;
                for (int64_t i = 0; i < d; i++)
                    s += ld(q, qb + i) * ld(k, (j * nkv + kh) * d + i);
                s *= scale;
                if (s > max)
                    max = s;
            }
            float sum = 0.0f;
            for (int64_t i = 0; i < dv; i++)
                acc[i] = 0.0f;
            for (int64_t j = 0; j <= last; j++) {
                float s = 0.0f;
                for (int64_t i = 0; i < d; i++)
                    s += ld(q, qb + i) * ld(k, (j * nkv + kh) * d + i);
                float e = expf(s * scale - max);
                sum += e;
                for (int64_t i = 0; i < dv; i++)
                    acc[i] += e * ld(v, (j * nkv + kh) * dv + i);
            }
            for (int64_t i = 0; i < dv; i++)
                st(out, (t * nh + h) * dv + i, acc[i] / sum);
        }
    }
    return ATH_OK;
}

ath_status ath_attn_gpu(const ath_device *d, const ath_tensor *q, const ath_tensor *k, const ath_tensor *v,
                        float scale, ath_tensor *out)
{
    ATH_CHECK(args(q, k, v, out, OPS_GPU));
    if (q->shape[2] > ATH_ATTN_MAX_DV)
        return ops_fail(ATH_ERR_UNSUPPORTED, op, "head dim %lld > %d on the GPU", (long long)q->shape[2],
                        ATH_ATTN_MAX_DV);
    return ath_attn_launch(d, q, k, v, scale, out);
}
