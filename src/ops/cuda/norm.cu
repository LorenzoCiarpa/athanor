/* norm.cu: RMSNorm, gated RMSNorm and the L2 norm on the GPU. See ops.h and
 * norm.c, whose expressions the kernel repeats in the same order.
 *
 * Rows of up to NORM_WARP_MAX elements in 16-byte pieces (every row of the
 * 0.8B model) take one warp each (step 9): the lanes load the row, its
 * weights and its gate at once into registers, add their squares, the warp
 * adds the 32 sums (k_warp_sum), and the lanes scale the values they hold,
 * with no block barrier and no second read. Longer rows take one block each: the threads add the
 * squares of their elements (every 128th one), the block adds the partial
 * sums (k_block_sum), and every thread scales its elements.
 */
#include "ops/cuda/kernel.cuh"

#define NORM_THREADS 128
#define NORM_WARP_MAX 1024
#define NORM_PIECES (NORM_WARP_MAX / 8 / ATH_WARP)

/* out = the norm of x[i] = v for one element, as norm.c writes it. */
static __device__ __forceinline__ float norm_one(int kind, float v, float inv, const void *z, int zdt, const void *w,
                                                 int wdt, float w_offset, int xdt, int64_t row_base, int64_t i)
{
    if (kind == ATH_NORM_RMS)
        return v * inv * (w_offset + k_ld(w, wdt, i));
    if (kind == ATH_NORM_RMS_GATED) {
        const float n = k_round(xdt, v * inv);                      /* the reference's rounding */
        return k_ld(w, wdt, i) * n * k_silu(k_ld(z, zdt, row_base + i));
    }
    return v * inv;
}

/* One warp per row, the row in 16-byte pieces (8 elements): lane l holds
 * pieces l, l + 32, ..., loaded at once with the weights and the gate, so a
 * row costs one trip to memory. D % 8 == 0 and every buffer on a 16-byte
 * boundary (the launcher checks). */
static __global__ void norm_warp(int kind, const void *x, int xdt, const void *z, int zdt, const void *w, int wdt,
                                 float w_offset, float eps, void *out, int odt, int64_t rows, int64_t D)
{
    const int lane = threadIdx.x % ATH_WARP;
    const int64_t r = (int64_t)blockIdx.x * (NORM_THREADS / ATH_WARP) + threadIdx.x / ATH_WARP;
    if (r >= rows)
        return;
    const int64_t pieces = D / 8, base = r * D;
    float v[NORM_PIECES][8], wv[NORM_PIECES][8], zv[NORM_PIECES][8], ss = 0.0f;
#pragma unroll
    for (int j = 0; j < NORM_PIECES; j++) {
        const int64_t c = lane + (int64_t)j * ATH_WARP;
        if (c < pieces) {
            k_ld8(x, xdt, base / 8 + c, v[j]);
            if (kind != ATH_NORM_L2)
                k_ld8(w, wdt, c, wv[j]);
            if (kind == ATH_NORM_RMS_GATED)
                k_ld8(z, zdt, base / 8 + c, zv[j]);
        }
    }
#pragma unroll
    for (int j = 0; j < NORM_PIECES; j++)
        if (lane + (int64_t)j * ATH_WARP < pieces)
#pragma unroll
            for (int i = 0; i < 8; i++)
                ss += v[j][i] * v[j][i];
    ss = k_warp_sum(ss);
    const float inv = 1.0f / sqrtf(ss / (kind == ATH_NORM_L2 ? 1.0f : (float)D) + eps);
#pragma unroll
    for (int j = 0; j < NORM_PIECES; j++) {
        const int64_t c = lane + (int64_t)j * ATH_WARP;
        if (c >= pieces)
            continue;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            float y;
            if (kind == ATH_NORM_RMS)
                y = v[j][i] * inv * (w_offset + wv[j][i]);
            else if (kind == ATH_NORM_RMS_GATED)
                y = wv[j][i] * k_round(xdt, v[j][i] * inv) * k_silu(zv[j][i]);   /* as norm_one */
            else
                y = v[j][i] * inv;
            k_st(out, odt, base + 8 * c + i, y);
        }
    }
}

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
            k_st(out, odt, base + i, norm_one(kind, v, inv, z, zdt, w, wdt, w_offset, xdt, base, i));
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
    const bool aligned = (uintptr_t)x->data % 16 == 0 && (!w || (uintptr_t)w->data % 16 == 0) &&
                         (!z || (uintptr_t)z->data % 16 == 0);
    if (D <= NORM_WARP_MAX && D % 8 == 0 && aligned && rows <= ((int64_t)1 << 30)) {
        const int64_t warps = NORM_THREADS / ATH_WARP;
        norm_warp<<<(unsigned)((rows + warps - 1) / warps), NORM_THREADS, 0, k_stream(d)>>>(
            (int)kind, x->data, x->dtype, z ? z->data : NULL, z ? z->dtype : ATH_DTYPE_F32, w ? w->data : NULL,
            w ? w->dtype : ATH_DTYPE_F32, w_offset, eps, out->data, out->dtype, rows, D);
        return ath_gpu_launched("norm_warp");
    }
    const unsigned blocks = (unsigned)(rows < ((int64_t)1 << 20) ? rows : (int64_t)1 << 20);
    norm_kernel<<<blocks, NORM_THREADS, 0, k_stream(d)>>>((int)kind, x->data, x->dtype, z ? z->data : NULL,
                                                          z ? z->dtype : ATH_DTYPE_F32, w ? w->data : NULL,
                                                          w ? w->dtype : ATH_DTYPE_F32, w_offset, eps, out->data,
                                                          out->dtype, rows, D);
    return ath_gpu_launched("norm_kernel");
}
