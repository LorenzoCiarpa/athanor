/* norm.c: RMSNorm (plain and zero-centered), gated RMSNorm, L2 norm. See ops.h.
 *
 * All three normalize each row over the last dimension with the statistic
 * summed in fp32 in order, then scale, in the order the reference does it
 * (contract, sections RMSNorm and Gated RMSNorm):
 *
 *   rmsnorm        x · rsqrt(mean(x²) + eps) · (w_offset + w)
 *   rmsnorm_gated  round(x · rsqrt(mean(x²) + eps)) · w · silu(z)
 *   l2norm         x · rsqrt(Σ x² + eps)
 */
#include "ops/check.h"
#include "ops/launch.h"
#include "ops/ops.h"

#include <math.h>

/* 1 / sqrt(Σ x² / div + eps) over the row starting at element base. */
static float inv_rms(const ath_tensor *x, int64_t base, int64_t D, float div, float eps)
{
    float ss = 0.0f;
    for (int64_t i = 0; i < D; i++) {
        float v = ld(x, base + i);
        ss += v * v;
    }
    return 1.0f / sqrtf(ss / div + eps);
}

/* x and out of the same shape; w (if present) a vector over the last
 * dimension; z (if present) of the shape of x. */
static ath_status check_rows(const char *op, const ath_tensor *x, const ath_tensor *out, const ath_tensor *w,
                             const ath_tensor *z, ops_where where)
{
    ATH_CHECK(ops_float(x, op, "x"));
    ATH_CHECK(ops_float(out, op, "out"));
    if (!ath_tensor_same_shape(x, out) || x->ndim < 1)
        return ops_fail(ATH_ERR_INVALID_ARG, op, "x and out must have the same shape");
    if (w) {
        ATH_CHECK(ops_float(w, op, "w"));
        if (w->ndim != 1 || w->shape[0] != ops_last(x))
            return ops_fail(ATH_ERR_INVALID_ARG, op, "w must be [%lld]", (long long)ops_last(x));
    }
    if (z) {
        ATH_CHECK(ops_float(z, op, "z"));
        if (!ath_tensor_same_shape(x, z))
            return ops_fail(ATH_ERR_INVALID_ARG, op, "x and z must have the same shape");
    }
    ATH_CHECK(OPS_PLACE(op, where, x, out));
    if (w)
        ATH_CHECK(OPS_PLACE(op, where, w));
    if (z)
        ATH_CHECK(OPS_PLACE(op, where, z));
    return ATH_OK;
}

/* The weight and the gate are required where the op has them. */
static ath_status need(const char *op, const ath_tensor *t, const char *name)
{
    return t ? ATH_OK : ops_fail(ATH_ERR_INVALID_ARG, op, "%s: missing", name);
}

ath_status ath_rmsnorm_ref(const ath_tensor *x, const ath_tensor *w, float w_offset, float eps,
                           ath_tensor *out)
{
    ATH_CHECK(need("rmsnorm", w, "w"));
    ATH_CHECK(check_rows("rmsnorm", x, out, w, NULL, OPS_HOST));
    const int64_t D = ops_last(x), rows = ops_rows(x);
    for (int64_t r = 0; r < rows; r++) {
        float inv = inv_rms(x, r * D, D, (float)D, eps);
        for (int64_t i = 0; i < D; i++)
            st(out, r * D + i, ld(x, r * D + i) * inv * (w_offset + ld(w, i)));
    }
    return ATH_OK;
}

ath_status ath_rmsnorm_gated_ref(const ath_tensor *x, const ath_tensor *z, const ath_tensor *w,
                                 float eps, ath_tensor *out)
{
    static const char op[] = "rmsnorm_gated";
    ATH_CHECK(need(op, w, "w"));
    ATH_CHECK(need(op, z, "z"));
    ATH_CHECK(check_rows(op, x, out, w, z, OPS_HOST));
    const int64_t D = ops_last(x), rows = ops_rows(x);
    for (int64_t r = 0; r < rows; r++) {
        float inv = inv_rms(x, r * D, D, (float)D, eps);
        for (int64_t i = 0; i < D; i++) {
            int64_t e = r * D + i;
            float n = ath_round_to(x->dtype, ld(x, e) * inv);
            float g = ld(z, e);
            st(out, e, ld(w, i) * n * (g / (1.0f + expf(-g))));
        }
    }
    return ATH_OK;
}

ath_status ath_l2norm_ref(const ath_tensor *x, float eps, ath_tensor *out)
{
    ATH_CHECK(check_rows("l2norm", x, out, NULL, NULL, OPS_HOST));
    const int64_t D = ops_last(x), rows = ops_rows(x);
    for (int64_t r = 0; r < rows; r++) {
        float inv = inv_rms(x, r * D, D, 1.0f, eps);
        for (int64_t i = 0; i < D; i++)
            st(out, r * D + i, ld(x, r * D + i) * inv);
    }
    return ATH_OK;
}

ath_status ath_rmsnorm_gpu(const ath_device *d, const ath_tensor *x, const ath_tensor *w, float w_offset,
                           float eps, ath_tensor *out)
{
    ATH_CHECK(need("rmsnorm", w, "w"));
    ATH_CHECK(check_rows("rmsnorm", x, out, w, NULL, OPS_GPU));
    return ath_norm_launch(d, ATH_NORM_RMS, x, NULL, w, w_offset, eps, out);
}

ath_status ath_add_rmsnorm_gpu(const ath_device *d, ath_tensor *x, const ath_tensor *y, const ath_tensor *w,
                               float w_offset, float eps, ath_tensor *out)
{
    ATH_CHECK(need("add_rmsnorm", w, "w"));
    ATH_CHECK(need("add_rmsnorm", y, "y"));
    ATH_CHECK(check_rows("add_rmsnorm", x, out, w, NULL, OPS_GPU));
    if (y->dtype != x->dtype || !ath_tensor_same_shape(x, y) || !ath_tensor_is_contiguous(y) ||
        y->tier != ATH_TIER_VRAM)
        return ops_fail(ATH_ERR_INVALID_ARG, "add_rmsnorm", "y must be a device tensor like x");
    if (!ath_add_rmsnorm_fits(x, y, w))
        return ops_fail(ATH_ERR_UNSUPPORTED, "add_rmsnorm", "rows the warp kernel does not take");
    return ath_add_rmsnorm_launch(d, x, y, w, w_offset, eps, out);
}

ath_status ath_rmsnorm_gated_gpu(const ath_device *d, const ath_tensor *x, const ath_tensor *z,
                                 const ath_tensor *w, float eps, ath_tensor *out)
{
    ATH_CHECK(need("rmsnorm_gated", w, "w"));
    ATH_CHECK(need("rmsnorm_gated", z, "z"));
    ATH_CHECK(check_rows("rmsnorm_gated", x, out, w, z, OPS_GPU));
    return ath_norm_launch(d, ATH_NORM_RMS_GATED, x, z, w, 0.0f, eps, out);
}

ath_status ath_l2norm_gpu(const ath_device *d, const ath_tensor *x, float eps, ath_tensor *out)
{
    ATH_CHECK(check_rows("l2norm", x, out, NULL, NULL, OPS_GPU));
    return ath_norm_launch(d, ATH_NORM_L2, x, NULL, NULL, 0.0f, eps, out);
}
