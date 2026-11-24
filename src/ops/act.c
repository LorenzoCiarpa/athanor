/* act.c: elementwise activations, gates and the residual add. See ops.h.
 *
 * silu(x) = x / (1 + exp(−x)) and sigmoid(x) = 1 / (1 + exp(−x)), in fp32,
 * written the way torch's CPU kernels compute them.
 */
#include "ops/check.h"
#include "ops/ops.h"

#include <math.h>

static ath_status check_same(const char *op, const ath_tensor *a, const char *an,
                             const ath_tensor *b, const char *bn, const ath_tensor *out)
{
    ATH_CHECK(ops_float(a, op, an));
    ATH_CHECK(ops_float(b, op, bn));
    ATH_CHECK(ops_float(out, op, "out"));
    if (!ath_tensor_same_shape(a, b) || !ath_tensor_same_shape(a, out))
        return ops_fail(ATH_ERR_INVALID_ARG, op, "%s, %s and out must have the same shape", an, bn);
    return ATH_OK;
}

ath_status ath_swiglu_ref(const ath_tensor *gate, const ath_tensor *up, ath_tensor *out)
{
    ATH_CHECK(check_same("swiglu", gate, "gate", up, "up", out));
    const int64_t n = ath_tensor_numel(out);
    for (int64_t i = 0; i < n; i++) {
        float g = ld(gate, i);
        st(out, i, g / (1.0f + expf(-g)) * ld(up, i));
    }
    return ATH_OK;
}

ath_status ath_sigmoid_mul_ref(const ath_tensor *x, const ath_tensor *gate, ath_tensor *out)
{
    ATH_CHECK(check_same("sigmoid_mul", x, "x", gate, "gate", out));
    const int64_t n = ath_tensor_numel(out);
    for (int64_t i = 0; i < n; i++)
        st(out, i, ld(x, i) * (1.0f / (1.0f + expf(-ld(gate, i)))));
    return ATH_OK;
}

ath_status ath_add_ref(const ath_tensor *a, const ath_tensor *b, ath_tensor *out)
{
    ATH_CHECK(check_same("add", a, "a", b, "b", out));
    const int64_t n = ath_tensor_numel(out);
    for (int64_t i = 0; i < n; i++)
        st(out, i, ld(a, i) + ld(b, i));
    return ATH_OK;
}
