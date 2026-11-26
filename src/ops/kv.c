/* kv.c: the KV cache in int8 (step 14). See ops.h.
 *
 * A row of keys or values (one token, one KV head: 256 values in the Qwen
 * models) is stored as int8 with a scale of its own:
 *
 *   s = max|x| / 127,  q = rint(x · (1 / s))  (q = 0, s = 0 for a zero row)
 *
 * and read back as q · s rounded to the activation dtype. Each step is one
 * correctly rounded fp32 operation, written the same way in cuda/kv.cu, so
 * the CPU and the GPU store and read the same bits.
 */
#include "ops/check.h"
#include "ops/launch.h"
#include "ops/ops.h"

#include <math.h>

static const char op_store[] = "kv_store", op_load[] = "kv_load";

static ath_status cache_args(const char *op, const ath_tensor *cache, const ath_tensor *scale, ops_where where)
{
    ATH_CHECK(ops_tensor(cache, ATH_DTYPE_I8, op, "cache"));
    ATH_CHECK(ops_tensor(scale, ATH_DTYPE_F32, op, "scale"));
    if (cache->ndim != 3 || scale->ndim != 2 || scale->shape[0] != cache->shape[0] ||
        scale->shape[1] != cache->shape[1] || cache->shape[2] < 1)
        return ops_fail(ATH_ERR_INVALID_ARG, op, "cache [S, nkv, d] (i8) and scale [S, nkv] (f32)");
    return OPS_PLACE(op, where, cache, scale);
}

static ath_status store_args(const ath_tensor *x, const ath_tensor *pos, const ath_tensor *cache,
                             const ath_tensor *scale, ops_where where)
{
    ATH_CHECK(cache_args(op_store, cache, scale, where));
    ATH_CHECK(ops_float(x, op_store, "x"));
    ATH_CHECK(ops_tensor(pos, ATH_DTYPE_I32, op_store, "pos"));
    if (x->ndim != 3 || x->shape[1] != cache->shape[1] || x->shape[2] != cache->shape[2] || pos->ndim != 1 ||
        pos->shape[0] != x->shape[0])
        return ops_fail(ATH_ERR_INVALID_ARG, op_store, "x [T, nkv, d] like the cache's rows, pos [T]");
    for (int64_t t = 0; where == OPS_HOST && t < x->shape[0]; t++)
        if (((const int32_t *)pos->data)[t] < 0 || ((const int32_t *)pos->data)[t] >= cache->shape[0])
            return ops_fail(ATH_ERR_RANGE, op_store, "position %d outside the cache's %lld rows",
                            ((const int32_t *)pos->data)[t], (long long)cache->shape[0]);
    return OPS_PLACE(op_store, where, x, pos);
}

static ath_status load_args(const ath_tensor *cache, const ath_tensor *scale, const ath_tensor *out,
                            ops_where where)
{
    ATH_CHECK(cache_args(op_load, cache, scale, where));
    ATH_CHECK(ops_float(out, op_load, "out"));
    if (out->ndim != 3 || out->shape[0] > cache->shape[0] || out->shape[1] != cache->shape[1] ||
        out->shape[2] != cache->shape[2])
        return ops_fail(ATH_ERR_INVALID_ARG, op_load, "out [n, nkv, d] with n up to the cache's rows");
    return OPS_PLACE(op_load, where, out);
}

ath_status ath_kv_store_ref(const ath_tensor *x, const ath_tensor *pos, ath_tensor *cache, ath_tensor *scale)
{
    ATH_CHECK(store_args(x, pos, cache, scale, OPS_HOST));
    const int64_t T = x->shape[0], nkv = x->shape[1], d = x->shape[2];
    int8_t *c = cache->data;
    float *sc = scale->data;
    for (int64_t t = 0; t < T; t++)
        for (int64_t h = 0; h < nkv; h++) {
            const int64_t xi = (t * nkv + h) * d, row = (int64_t)((const int32_t *)pos->data)[t] * nkv + h;
            float amax = 0.0f;
            for (int64_t i = 0; i < d; i++)
                amax = fmaxf(amax, fabsf(ld(x, xi + i)));
            const float s = amax / 127.0f;
            const float inv = s > 0.0f ? 1.0f / s : 0.0f;
            for (int64_t i = 0; i < d; i++) {
                float q = rintf(ld(x, xi + i) * inv);
                q = q > 127.0f ? 127.0f : q < -127.0f ? -127.0f : q;
                c[row * d + i] = (int8_t)q;
            }
            sc[row] = s;
        }
    return ATH_OK;
}

ath_status ath_kv_load_ref(const ath_tensor *cache, const ath_tensor *scale, ath_tensor *out)
{
    ATH_CHECK(load_args(cache, scale, out, OPS_HOST));
    const int64_t rows = out->shape[0] * out->shape[1], d = out->shape[2];
    const int8_t *c = cache->data;
    const float *sc = scale->data;
    for (int64_t r = 0; r < rows; r++)
        for (int64_t i = 0; i < d; i++)
            st(out, r * d + i, (float)c[r * d + i] * sc[r]);
    return ATH_OK;
}

ath_status ath_kv_store_gpu(const ath_device *d, const ath_tensor *x, const ath_tensor *pos, ath_tensor *cache,
                            ath_tensor *scale)
{
    ATH_CHECK(store_args(x, pos, cache, scale, OPS_GPU));
    return ath_kv_store_launch(d, x, pos, cache, scale);
}

ath_status ath_kv_load_gpu(const ath_device *d, const ath_tensor *cache, const ath_tensor *scale, ath_tensor *out)
{
    ATH_CHECK(load_args(cache, scale, out, OPS_GPU));
    return ath_kv_load_launch(d, cache, scale, out);
}
