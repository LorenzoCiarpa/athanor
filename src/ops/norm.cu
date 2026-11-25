/* norm.cu: RMSNorm, gated RMSNorm and the L2 norm on the GPU. See ops.h and
 * norm.c, whose expressions the kernel repeats in the same order.
 *
 * One block per row: the threads add the squares of their elements (every
 * 128th one), the block adds the partial sums (k_block_sum), and every
 * thread scales its elements. Rows are short (128 to a few thousand
 * elements), so one block per row keeps the whole row in one place.
 */
#include "ops/kernel.cuh"

#define NORM_THREADS 128

static __global__ void norm_kernel(int kind, const void *x, int xdt, const void *z, int zdt, const void *w, int wdt,
                                   float w_offset, float eps, void *out, int odt, int64_t rows, int64_t D)
{
    __shared__ float scratch[32];
    const float div = kind == ATH_NORM_L2 ? 1.0f : (float)D;
    for (int64_t r = blockIdx.x; r < rows; r += gridDim.x) {       /* the same count for every thread */
        const int64_t base = r * D;
        float ss = 0.0f;
        for (int64_t i = threadIdx.x; i < D; i += blockDim.x) {
            const float v = k_ld(x, xdt, base + i);
            ss += v * v;
        }
        ss = k_block_sum(ss, scratch);
        const float inv = 1.0f / sqrtf(ss / div + eps);
        for (int64_t i = threadIdx.x; i < D; i += blockDim.x) {
            const float v = k_ld(x, xdt, base + i);
            float y;
            if (kind == ATH_NORM_RMS) {
                y = v * inv * (w_offset + k_ld(w, wdt, i));
            } else if (kind == ATH_NORM_RMS_GATED) {
                const float n = k_round(xdt, v * inv);              /* the reference's rounding */
                y = k_ld(w, wdt, i) * n * k_silu(k_ld(z, zdt, base + i));
            } else {
                y = v * inv;
            }
            k_st(out, odt, base + i, y);
        }
    }
}

extern "C" ath_status ath_norm_launch(const ath_device *d, ath_norm_kind kind, const ath_tensor *x,
                                      const ath_tensor *z, const ath_tensor *w, float w_offset, float eps,
                                      ath_tensor *out)
{
    const int64_t D = x->ndim > 0 ? x->shape[x->ndim - 1] : 1;
    const int64_t rows = D > 0 ? ath_tensor_numel(x) / D : 0;
    if (rows == 0)
        return ATH_OK;
    ATH_CHECK(ath_gpu_use(d));
    const unsigned blocks = (unsigned)(rows < ((int64_t)1 << 20) ? rows : (int64_t)1 << 20);
    norm_kernel<<<blocks, NORM_THREADS, 0, k_stream(d)>>>((int)kind, x->data, x->dtype, z ? z->data : NULL,
                                                          z ? z->dtype : ATH_DTYPE_F32, w ? w->data : NULL,
                                                          w ? w->dtype : ATH_DTYPE_F32, w_offset, eps, out->data,
                                                          out->dtype, rows, D);
    return ath_gpu_launched("norm_kernel");
}
