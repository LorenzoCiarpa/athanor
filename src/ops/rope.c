/* rope.c: rotary position embedding, rotate-half convention. See ops.h.
 *
 * The frequency table is computed the way the reference computes it, in fp32:
 * f_i = 1 / theta^(2i / rope_dim) and the angle p · f_i, then cos and sin
 * rounded to the activation dtype. Dimension i of the rotary part pairs with
 * dimension i + rope_dim / 2 (rotate-half), not with i + 1 (interleaved).
 */
#include "ops/check.h"
#include "ops/launch.h"
#include "ops/ops.h"

#include <math.h>

static const char op[] = "rope";

/* Frequency i of the rotary part, as the reference computes it. The GPU
 * kernel receives these values from here instead of computing them with the
 * device's powf: the angle is p · f_i, so one ulp of difference in f_i
 * becomes p ulps in the angle, a visible error at long positions. */
static float freq(int32_t i, int32_t rope_dim, float theta)
{
    return 1.0f / powf(theta, (float)(2 * i) / (float)rope_dim);
}

static ath_status args(const ath_tensor *x, const ath_tensor *pos, int32_t rope_dim, float theta,
                       const ath_tensor *out, ops_where where)
{
    ATH_CHECK(ops_float(x, op, "x"));
    ATH_CHECK(ops_tensor(pos, ATH_DTYPE_I32, op, "pos"));
    ATH_CHECK(ops_float(out, op, "out"));
    if (x->ndim != 3 || !ath_tensor_same_shape(x, out) || pos->ndim != 1 ||
        pos->shape[0] != x->shape[0])
        return ops_fail(ATH_ERR_INVALID_ARG, op, "shapes must be x, out [T, heads, head_dim], pos [T]");
    const int64_t D = x->shape[2];
    if (rope_dim <= 0 || rope_dim % 2 != 0 || rope_dim > D || !(theta > 0.0f))
        return ops_fail(ATH_ERR_INVALID_ARG, op, "rope_dim %d must be even and in (0, %lld], theta > 0",
                        rope_dim, (long long)D);
    return OPS_PLACE(op, where, x, pos, out);
}

ath_status ath_rope_ref(const ath_tensor *x, const ath_tensor *pos, int32_t rope_dim, float theta,
                        ath_tensor *out)
{
    ATH_CHECK(args(x, pos, rope_dim, theta, out, OPS_HOST));
    const int64_t T = x->shape[0], heads = x->shape[1], D = x->shape[2];
    const int32_t half = rope_dim / 2;
    const int32_t *p = pos->data;
    for (int64_t t = 0; t < T; t++)
        for (int32_t i = 0; i < half; i++) {
            float angle = (float)p[t] * freq(i, rope_dim, theta);
            float c = ath_round_to(x->dtype, cosf(angle));
            float s = ath_round_to(x->dtype, sinf(angle));
            for (int64_t h = 0; h < heads; h++) {
                int64_t base = (t * heads + h) * D;
                float x0 = ld(x, base + i), x1 = ld(x, base + i + half);
                st(out, base + i, x0 * c - x1 * s);
                st(out, base + i + half, x1 * c + x0 * s);
            }
        }
    for (int64_t r = 0; r < T * heads; r++)
        for (int64_t i = rope_dim; i < D; i++)
            st(out, r * D + i, ld(x, r * D + i));
    return ATH_OK;
}

ath_status ath_rope_gpu(const ath_device *d, const ath_tensor *x, const ath_tensor *pos, int32_t rope_dim,
                        float theta, ath_tensor *out)
{
    ATH_CHECK(args(x, pos, rope_dim, theta, out, OPS_GPU));
    if (rope_dim > ATH_ROPE_MAX_DIM)
        return ops_fail(ATH_ERR_UNSUPPORTED, op, "rope_dim %d > %d on the GPU", rope_dim, ATH_ROPE_MAX_DIM);
    float f[ATH_ROPE_MAX_DIM / 2];
    for (int32_t i = 0; i < rope_dim / 2; i++)
        f[i] = freq(i, rope_dim, theta);
    return ath_rope_launch(d, x, pos, rope_dim, f, out);
}
