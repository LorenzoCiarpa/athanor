/* embed.c: token embedding lookup. See ops.h. */
#include "ops/check.h"
#include "ops/ops.h"

ath_status ath_embed_ref(const ath_tensor *table, const ath_tensor *ids, ath_tensor *out)
{
    static const char op[] = "embed";
    ATH_CHECK(ops_float(table, op, "table"));
    ATH_CHECK(ops_tensor(ids, ATH_DTYPE_I32, op, "ids"));
    ATH_CHECK(ops_float(out, op, "out"));
    if (table->ndim != 2 || ids->ndim != 1 || out->ndim != 2 || out->shape[0] != ids->shape[0] ||
        out->shape[1] != table->shape[1])
        return ops_fail(ATH_ERR_INVALID_ARG, op, "shapes must be table [V, H], ids [n], out [n, H]");

    const int64_t V = table->shape[0], H = table->shape[1], n = ids->shape[0];
    const int32_t *id = ids->data;
    for (int64_t i = 0; i < n; i++)
        if (id[i] < 0 || id[i] >= V)
            return ops_fail(ATH_ERR_RANGE, op, "ids[%lld] = %d outside [0, %lld)", (long long)i, id[i],
                            (long long)V);
    for (int64_t i = 0; i < n; i++)
        for (int64_t j = 0; j < H; j++)
            st(out, i * H + j, ld(table, (int64_t)id[i] * H + j));
    return ATH_OK;
}
