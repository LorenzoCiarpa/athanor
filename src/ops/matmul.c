/* matmul.c: dense matrix product against a weight in checkpoint layout. See ops.h.
 *
 * The weight is [M, K], one output feature per row, as nn.Linear stores it:
 * out = x · wᵀ. Each output element is one dot product accumulated in fp32 in
 * order of k. For bf16 inputs every product is exact in fp32 (8 + 8
 * significant bits), so the only rounding is in the running sum.
 */
#include "ops/check.h"
#include "ops/ops.h"

ath_status ath_matmul_ref(const ath_tensor *x, const ath_tensor *w, ath_tensor *out)
{
    static const char op[] = "matmul";
    ATH_CHECK(ops_float(x, op, "x"));
    ATH_CHECK(ops_float(w, op, "w"));
    ATH_CHECK(ops_float(out, op, "out"));
    if (x->ndim < 1 || w->ndim != 2 || out->ndim != x->ndim)
        return ops_fail(ATH_ERR_INVALID_ARG, op, "shapes must be x [..., K], w [M, K], out [..., M]");
    const int64_t K = ops_last(x), M = w->shape[0], N = ops_rows(x);
    if (w->shape[1] != K || ops_last(out) != M)
        return ops_fail(ATH_ERR_INVALID_ARG, op, "x [..., %lld], w [%lld, %lld], out [..., %lld]",
                        (long long)K, (long long)w->shape[0], (long long)w->shape[1],
                        (long long)ops_last(out));
    for (int32_t i = 0; i + 1 < x->ndim; i++)
        if (out->shape[i] != x->shape[i])
            return ops_fail(ATH_ERR_INVALID_ARG, op, "out must have the leading dimensions of x");

    for (int64_t r = 0; r < N; r++)
        for (int64_t m = 0; m < M; m++) {
            float acc = 0.0f;
            for (int64_t k = 0; k < K; k++)
                acc += ld(x, r * K + k) * ld(w, m * K + k);
            st(out, r * M + m, acc);
        }
    return ATH_OK;
}
