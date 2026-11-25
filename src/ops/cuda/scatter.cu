/* scatter.cu: rows written at given rows, on the GPU. See ops.h and
 * scatter.c.
 *
 * One thread per 4-byte word when the rows are made of whole words (always
 * for the KV cache), per byte otherwise. The indices are on the device: a
 * row whose index is outside the destination is not written.
 */
#include "ops/cuda/kernel.cuh"

static __global__ void scatter_words(const uint32_t *src, const int32_t *idx, uint32_t *dst, int64_t n, int64_t words,
                                     int64_t R)
{
    K_FOR(e, n * words) {
        const int32_t r = idx[e / words];
        if (r >= 0 && r < R)
            dst[(int64_t)r * words + e % words] = src[e];
    }
}

static __global__ void scatter_bytes(const uint8_t *src, const int32_t *idx, uint8_t *dst, int64_t n, int64_t bytes,
                                     int64_t R)
{
    K_FOR(e, n * bytes) {
        const int32_t r = idx[e / bytes];
        if (r >= 0 && r < R)
            dst[(int64_t)r * bytes + e % bytes] = src[e];
    }
}

extern "C" ath_status ath_scatter_rows_launch(const ath_device *d, const ath_tensor *src, const ath_tensor *idx,
                                              ath_tensor *dst)
{
    const int64_t n = src->shape[0], R = dst->shape[0];
    const int64_t row = n ? (int64_t)(ath_tensor_nbytes(src) / (size_t)n) : 0;
    if (n * row == 0)
        return ATH_OK;
    ATH_CHECK(ath_gpu_use(d));
    const int32_t *ix = (const int32_t *)idx->data;
    if (row % 4 == 0 && (uintptr_t)src->data % 4 == 0 && (uintptr_t)dst->data % 4 == 0)
        scatter_words<<<k_blocks(n * row / 4, 256), 256, 0, k_stream(d)>>>((const uint32_t *)src->data, ix,
                                                                         (uint32_t *)dst->data, n, row / 4, R);
    else
        scatter_bytes<<<k_blocks(n * row, 256), 256, 0, k_stream(d)>>>((const uint8_t *)src->data, ix,
                                                                     (uint8_t *)dst->data, n, row, R);
    return ath_gpu_launched("scatter_rows");
}
