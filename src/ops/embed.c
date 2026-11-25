/* embed.c: token embedding lookup. See ops.h. */
#include "ops/check.h"
#include "ops/launch.h"
#include "ops/ops.h"
#include "ops/q8_0.h"

static const char op[] = "embed";

static ath_status args(const ath_tensor *table, const ath_tensor *ids, const ath_tensor *out, ops_where where)
{
    ATH_CHECK(table && table->dtype == ATH_DTYPE_Q8_0 ? ops_tensor(table, ATH_DTYPE_Q8_0, op, "table")
                                                       : ops_float(table, op, "table"));
    ATH_CHECK(ops_tensor(ids, ATH_DTYPE_I32, op, "ids"));
    ATH_CHECK(ops_float(out, op, "out"));
    if (table->ndim != 2 || ids->ndim != 1 || out->ndim != 2 || out->shape[0] != ids->shape[0] ||
        out->shape[1] != table->shape[1])
        return ops_fail(ATH_ERR_INVALID_ARG, op, "shapes must be table [V, H], ids [n], out [n, H]");
    return OPS_PLACE(op, where, table, ids, out);
}

ath_status ath_embed_ref(const ath_tensor *table, const ath_tensor *ids, ath_tensor *out)
{
    ATH_CHECK(args(table, ids, out, OPS_HOST));
    const int64_t V = table->shape[0], H = table->shape[1], n = ids->shape[0];
    const int32_t *id = ids->data;
    for (int64_t i = 0; i < n; i++)
        if (id[i] < 0 || id[i] >= V)
            return ops_fail(ATH_ERR_RANGE, op, "ids[%lld] = %d outside [0, %lld)", (long long)i, id[i],
                            (long long)V);
    if (table->dtype == ATH_DTYPE_Q8_0) {   /* step 12: d · q, exact in f32, then out's rounding */
        const size_t row = ath_dtype_bytes(ATH_DTYPE_Q8_0, H);
        for (int64_t i = 0; i < n; i++) {
            const uint8_t *r = (const uint8_t *)table->data + (size_t)id[i] * row;
            for (int64_t j = 0; j < H; j++) {
                const uint8_t *blk = r + j / ATH_Q8_0_BLOCK * ATH_Q8_0_BYTES;
                st(out, i * H + j, ath_q8_0_scale(blk) * (float)(int8_t)blk[2 + j % ATH_Q8_0_BLOCK]);
            }
        }
        return ATH_OK;
    }
    for (int64_t i = 0; i < n; i++)
        for (int64_t j = 0; j < H; j++)
            st(out, i * H + j, ld(table, (int64_t)id[i] * H + j));
    return ATH_OK;
}

ath_status ath_embed_gpu(const ath_device *d, const ath_tensor *table, const ath_tensor *ids, ath_tensor *out)
{
    ATH_CHECK(args(table, ids, out, OPS_GPU));
    return ath_embed_launch(d, table, ids, out);
}
