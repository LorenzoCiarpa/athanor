/* kv.cu: the KV cache in int8 on the GPU (step 14). See ops.h and kv.c.
 *
 * store  one warp per row (a token and a KV head): the largest magnitude by
 *        a warp reduction (fmaxf is exact, so the order does not matter),
 *        s = max / 127 and 1 / s as correctly rounded divisions, q = x · (1/s)
 *        rounded to the nearest even integer: kv.c's operations, the same
 *        bits.
 *  load  one thread per value: q · s rounded to the output's dtype.
 */
#include "ops/cuda/kernel.cuh"

#define KV_WARPS 4

static __global__ void __launch_bounds__(KV_WARPS * ATH_WARP)
kv_store(const void *x, int xdt, const int32_t *pos, int8_t *cache, float *scale, int64_t T, int64_t S, int64_t nkv,
         int64_t d)
{
    const int lane = threadIdx.x % ATH_WARP;
    const int64_t r = (int64_t)blockIdx.x * KV_WARPS + threadIdx.x / ATH_WARP;   /* row of x: t * nkv + h */
    if (r >= T * nkv)
        return;
    const int64_t t = r / nkv, h = r % nkv;
    const int64_t p = pos[t];
    if (p < 0 || p >= S)
        return;
    float amax = 0.0f;
    for (int64_t i = lane; i < d; i += ATH_WARP)
        amax = fmaxf(amax, fabsf(k_ld(x, xdt, r * d + i)));
    for (int m = ATH_WARP / 2; m > 0; m >>= 1)
        amax = fmaxf(amax, ath_shfl_xor(amax, m));
    const float s = __fdiv_rn(amax, 127.0f);
    const float inv = s > 0.0f ? __fdiv_rn(1.0f, s) : 0.0f;
    const int64_t row = p * nkv + h;
    for (int64_t i = lane; i < d; i += ATH_WARP) {
        int q = __float2int_rn(__fmul_rn(k_ld(x, xdt, r * d + i), inv));
        q = q > 127 ? 127 : q < -127 ? -127 : q;
        cache[row * d + i] = (int8_t)q;
    }
    if (lane == 0)
        scale[row] = s;
}

static __global__ void kv_load(const int8_t *cache, const float *scale, void *out, int odt, int64_t n, int64_t d)
{
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    k_st(out, odt, i, __fmul_rn((float)cache[i], scale[i / d]));
}

extern "C" ath_status ath_kv_store_launch(const ath_device *d, const ath_tensor *x, const ath_tensor *pos,
                                          ath_tensor *cache, ath_tensor *scale)
{
    const int64_t T = x->shape[0], nkv = x->shape[1], hd = x->shape[2], S = cache->shape[0];
    if (T * nkv == 0)
        return ATH_OK;
    ATH_CHECK(ath_gpu_use(d));
    const unsigned blocks = (unsigned)((T * nkv + KV_WARPS - 1) / KV_WARPS);
    kv_store<<<blocks, KV_WARPS * ATH_WARP, 0, k_stream(d)>>>(x->data, x->dtype, (const int32_t *)pos->data,
                                                               (int8_t *)cache->data, (float *)scale->data, T, S, nkv,
                                                               hd);
    return ath_gpu_launched("kv_store");
}

extern "C" ath_status ath_kv_load_launch(const ath_device *d, const ath_tensor *cache, const ath_tensor *scale,
                                         ath_tensor *out)
{
    const int64_t n = ath_tensor_numel(out), hd = out->shape[2];
    if (n == 0)
        return ATH_OK;
    ATH_CHECK(ath_gpu_use(d));
    kv_load<<<(unsigned)((n + 255) / 256), 256, 0, k_stream(d)>>>((const int8_t *)cache->data,
                                                                  (const float *)scale->data, out->data, out->dtype, n,
                                                                  hd);
    return ath_gpu_launched("kv_load");
}
