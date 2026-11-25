/* scatter.c: rows of a tensor written at given rows of another. See ops.h.
 *
 * The KV cache takes the keys and values of new tokens at their positions
 * this way (step 9): with the positions in a tensor, which on a GPU is in
 * device memory, the forward pass no longer computes cache addresses from
 * the number of tokens seen, and one decode step can be replayed. Rows are
 * copied byte for byte, whatever their dtype.
 */
#include "ops/check.h"
#include "ops/launch.h"
#include "ops/ops.h"

#include <string.h>

static const char op[] = "scatter_rows";

static ath_status args(const ath_tensor *src, const ath_tensor *idx, const ath_tensor *dst, ops_where where)
{
    ATH_CHECK(ops_tensor(idx, ATH_DTYPE_I32, op, "idx"));
    if (!src || !dst)
        return ops_fail(ATH_ERR_INVALID_ARG, op, "missing src or dst");
    ATH_CHECK(ops_tensor(src, src->dtype, op, "src"));
    ATH_CHECK(ops_tensor(dst, src->dtype, op, "dst"));
    if (src->ndim < 1 || dst->ndim < 1 || idx->ndim != 1 || idx->shape[0] != src->shape[0])
        return ops_fail(ATH_ERR_INVALID_ARG, op, "src [n, ...], idx [n], dst [R, ...]");
    const int64_t n = src->shape[0], R = dst->shape[0];
    const int64_t row = n ? ath_tensor_numel(src) / n : 0, drow = R ? ath_tensor_numel(dst) / R : 0;
    if (n > 0 && (R == 0 || row != drow))
        return ops_fail(ATH_ERR_INVALID_ARG, op, "rows of %lld and %lld elements", (long long)row, (long long)drow);
    return OPS_PLACE(op, where, src, idx, dst);
}

ath_status ath_scatter_rows_ref(const ath_tensor *src, const ath_tensor *idx, ath_tensor *dst)
{
    ATH_CHECK(args(src, idx, dst, OPS_HOST));
    const int64_t n = src->shape[0], R = dst->shape[0];
    const int32_t *ix = idx->data;
    for (int64_t i = 0; i < n; i++)
        if (ix[i] < 0 || ix[i] >= R)
            return ops_fail(ATH_ERR_RANGE, op, "idx[%lld] = %d outside [0, %lld)", (long long)i, ix[i],
                            (long long)R);
    const size_t row = n ? ath_tensor_nbytes(src) / (size_t)n : 0;
    for (int64_t i = 0; i < n; i++)
        memcpy((char *)dst->data + (size_t)ix[i] * row, (const char *)src->data + (size_t)i * row, row);
    return ATH_OK;
}

ath_status ath_scatter_rows_gpu(const ath_device *d, const ath_tensor *src, const ath_tensor *idx, ath_tensor *dst)
{
    ATH_CHECK(args(src, idx, dst, OPS_GPU));
    return ath_scatter_rows_launch(d, src, idx, dst);
}
