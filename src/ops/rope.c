/* rope.c: rotary position embedding, rotate-half convention. See ops.h.
 *
 * The frequency table is computed the way the reference computes it, in fp32:
 * f_i = 1 / theta^(2i / rope_dim) and the angle p · f_i, then cos and sin
 * rounded to the activation dtype. Dimension i of the rotary part pairs with
 * dimension i + rope_dim / 2 (rotate-half), not with i + 1 (interleaved).
 */
#include "ops/check.h"
#include "ops/ops.h"

#include <math.h>

ath_status ath_rope_ref(const ath_tensor *x, const ath_tensor *pos, int32_t rope_dim, float theta,
                        ath_tensor *out)
{
    static const char op[] = "rope";
    ATH_CHECK(ops_float(x, op, "x"));
    ATH_CHECK(ops_tensor(pos, ATH_DTYPE_I32, op, "pos"));
    ATH_CHECK(ops_float(out, op, "out"));
    if (x->ndim != 3 || !ath_tensor_same_shape(x, out) || pos->ndim != 1 ||
        pos->shape[0] != x->shape[0])
        return ops_fail(ATH_ERR_INVALID_ARG, op, "shapes must be x, out [T, heads, head_dim], pos [T]");
    const int64_t T = x->shape[0], heads = x->shape[1], D = x->shape[2];
    if (rope_dim <= 0 || rope_dim % 2 != 0 || rope_dim > D || !(theta > 0.0f))
        return ops_fail(ATH_ERR_INVALID_ARG, op, "rope_dim %d must be even and in (0, %lld], theta > 0",
                        rope_dim, (long long)D);

    const int32_t half = rope_dim / 2;
    const int32_t *p = pos->data;
    for (int64_t t = 0; t < T; t++)
        for (int32_t i = 0; i < half; i++) {
            float f = 1.0f / powf(theta, (float)(2 * i) / (float)rope_dim);
            float angle = (float)p[t] * f;
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
