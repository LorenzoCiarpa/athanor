/* norm.cu: RMSNorm, gated RMSNorm and the L2 norm on the GPU. See ops.h and
 * norm.c, whose expressions the kernel repeats in the same order.
 *
 * Rows of up to NORM_WARP_MAX elements in 16-byte pieces (every row of the
 * 0.8B model) take one warp each (step 9), or half or a quarter of one
 * (step 10): the lanes load the row, its
 * weights and its gate at once into registers, add their squares, the warp
 * adds the 32 sums (k_warp_sum), and the lanes scale the values they hold,
 * with no block barrier and no second read. Longer rows take one block each: the threads add the
 * squares of their elements (every 128th one), the block adds the partial
 * sums (k_block_sum), and every thread scales its elements.
 *
 * ath_add_rmsnorm (step 10) is the warp kernel with the residual added
 * first: x += y, rounded and stored as ath_add does, then the norm of what
 * was stored; the bits of the two ops, one read of x instead of two.
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
 * row costs one trip to memory. A row of at most 16 or 8 pieces takes 16 or
 * 8 lanes (sub), two or four rows to a warp (step 10: a row of 128 used half
 * the warp); the sum goes over the row's lanes only, the same additions as
 * over the whole warp, whose other lanes added zeros. D % 8 == 0 and every
 * buffer on a 16-byte boundary (the launcher checks). */
static __device__ __forceinline__ void norm_warp_body(int kind, int pmax, const void *x, int xdt, const void *z,
                                                      int zdt, const void *w, int wdt, float w_offset, float eps,
                                                      void *out, int odt, int64_t rows, int64_t D, int lsub,
                                                      const void *y, int ydt, void *xw)
{
    const int sub = 1 << lsub, lane = threadIdx.x & (sub - 1);       /* sub lanes per row */
    const int64_t r = ((int64_t)blockIdx.x * (NORM_THREADS / ATH_WARP) + threadIdx.x / ATH_WARP) * (ATH_WARP >> lsub) +
                      ((threadIdx.x & (ATH_WARP - 1)) >> lsub);
    const bool live = r < rows;                       /* every lane takes part in the sums */
    const int64_t pieces = live ? D / 8 : 0, base = r * D;
    float v[NORM_PIECES][8], wv[NORM_PIECES][8], zv[NORM_PIECES][8], ss = 0.0f;
#pragma unroll
    for (int j = 0; j < NORM_PIECES; j++) {
        const int64_t c = lane + (int64_t)j * sub;
        if (j < pmax && c < pieces) {
            k_ld8(x, xdt, base / 8 + c, v[j]);
            if (y) {
                /* x += y first, rounded to x's dtype as ath_add rounds it,
                 * and stored: the norm then reads what ath_add wrote */
                float yv[8];
                k_ld8(y, ydt, base / 8 + c, yv);
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    v[j][i] = k_round(xdt, v[j][i] + yv[i]);
                    k_st(xw, xdt, base + 8 * c + i, v[j][i]);
                }
            }
            if (kind != ATH_NORM_L2)
                k_ld8(w, wdt, c, wv[j]);
            if (kind == ATH_NORM_RMS_GATED)
                k_ld8(z, zdt, base / 8 + c, zv[j]);
        }
    }
#pragma unroll
    for (int j = 0; j < NORM_PIECES; j++)
        if (j < pmax && lane + (int64_t)j * sub < pieces)
#pragma unroll
            for (int i = 0; i < 8; i++)
                ss += v[j][i] * v[j][i];
#pragma unroll
    for (int m = ATH_WARP / 2; m > 0; m >>= 1)        /* the butterfly of k_warp_sum, within the row's lanes */
        if (m < sub)
            ss += ath_shfl_xor(ss, m);
    const float inv = 1.0f / sqrtf(ss / (kind == ATH_NORM_L2 ? 1.0f : (float)D) + eps);
#pragma unroll
    for (int j = 0; j < NORM_PIECES; j++) {
        const int64_t c = lane + (int64_t)j * sub;
        if (j >= pmax || c >= pieces)
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

/* One kernel per kind, the body inlined with the kind a constant: each
 * holds its own path only. The body with every kind's branches was 7,700
 * instructions, more than the instruction cache holds, and decode launches
 * the norms some 80 times per token between the GEMV kernels (step 10). */
static __global__ void norm_warp_rms(const void *x, int xdt, const void *w, int wdt, float w_offset, float eps,
                                     void *out, int odt, int64_t rows, int64_t D, int lsub, const void *y, int ydt,
                                     void *xw)
{
    norm_warp_body(ATH_NORM_RMS, NORM_PIECES, x, xdt, NULL, ATH_DTYPE_F32, w, wdt, w_offset, eps, out, odt, rows, D, lsub, y, ydt,
                   xw);
}

/* Gated rows of at most 32 pieces (256 elements: a value head), one piece
 * per lane; longer ones take norm_kernel. */
static __global__ void norm_warp_gated(const void *x, int xdt, const void *z, int zdt, const void *w, int wdt,
                                       float eps, void *out, int odt, int64_t rows, int64_t D, int lsub)
{
    norm_warp_body(ATH_NORM_RMS_GATED, 1, x, xdt, z, zdt, w, wdt, 0.0f, eps, out, odt, rows, D, lsub, NULL,
                   ATH_DTYPE_F32, NULL);
}

static __global__ void norm_warp_l2(const void *x, int xdt, float eps, void *out, int odt, int64_t rows, int64_t D,
                                    int lsub)
{
    norm_warp_body(ATH_NORM_L2, NORM_PIECES, x, xdt, NULL, ATH_DTYPE_F32, NULL, ATH_DTYPE_F32, 0.0f, eps, out, odt, rows, D, lsub,
                   NULL, ATH_DTYPE_F32, NULL);
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

/* The warp kernel over rows of x, after x += y where y is given (the
 * fusion of step 10); false where it does not take the shapes. */
static bool norm_warp_fits(const ath_tensor *x, const ath_tensor *z, const ath_tensor *w, const ath_tensor *y)
{
    const int64_t D = x->ndim > 0 ? x->shape[x->ndim - 1] : 1;
    const int64_t rows = D > 0 ? ath_tensor_numel(x) / D : 0;
    if (z && D > 8 * ATH_WARP)
        return false;                                  /* gated rows: one piece per lane (norm_warp_gated) */
    const bool aligned = (uintptr_t)x->data % 16 == 0 && (!w || (uintptr_t)w->data % 16 == 0) &&
                         (!z || (uintptr_t)z->data % 16 == 0) && (!y || (uintptr_t)y->data % 16 == 0);
    return D <= NORM_WARP_MAX && D % 8 == 0 && aligned && rows <= ((int64_t)1 << 30);
}

static ath_status norm_warp_launch(const ath_device *d, ath_norm_kind kind, ath_tensor *x, const ath_tensor *z,
                                   const ath_tensor *w, float w_offset, float eps, ath_tensor *out,
                                   const ath_tensor *y)
{
    const int64_t D = x->shape[x->ndim - 1], rows = ath_tensor_numel(x) / D;
    const int lsub = D / 8 > 16 ? 5 : D / 8 > 8 ? 4 : 3;           /* 32, 16 or 8 lanes per row */
    const int64_t per_block = (NORM_THREADS / ATH_WARP) * (ATH_WARP >> lsub);
    const unsigned blocks = (unsigned)((rows + per_block - 1) / per_block);
    if (kind == ATH_NORM_RMS)
        norm_warp_rms<<<blocks, NORM_THREADS, 0, k_stream(d)>>>(x->data, x->dtype, w->data, w->dtype, w_offset, eps,
                                                                out->data, out->dtype, rows, D, lsub,
                                                                y ? y->data : NULL, y ? y->dtype : ATH_DTYPE_F32,
                                                                y ? x->data : NULL);
    else if (kind == ATH_NORM_RMS_GATED)
        norm_warp_gated<<<blocks, NORM_THREADS, 0, k_stream(d)>>>(x->data, x->dtype, z->data, z->dtype, w->data,
                                                                  w->dtype, eps, out->data, out->dtype, rows, D, lsub);
    else
        norm_warp_l2<<<blocks, NORM_THREADS, 0, k_stream(d)>>>(x->data, x->dtype, eps, out->data, out->dtype, rows, D,
                                                               lsub);
    return ath_gpu_launched("norm_warp");
}

extern "C" bool ath_add_rmsnorm_fits(const ath_tensor *x, const ath_tensor *y, const ath_tensor *w)
{
    return norm_warp_fits(x, NULL, w, y);
}

extern "C" ath_status ath_add_rmsnorm_launch(const ath_device *d, ath_tensor *x, const ath_tensor *y,
                                             const ath_tensor *w, float w_offset, float eps, ath_tensor *out)
{
    if (ath_tensor_numel(x) == 0)
        return ATH_OK;
    ATH_CHECK(ath_gpu_use(d));
    return norm_warp_launch(d, ATH_NORM_RMS, x, NULL, w, w_offset, eps, out, y);
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
    if (norm_warp_fits(x, z, w, NULL))
        return norm_warp_launch(d, kind, (ath_tensor *)x, z, w, w_offset, eps, out, NULL);   /* x only read */
    const unsigned blocks = (unsigned)(rows < ((int64_t)1 << 20) ? rows : (int64_t)1 << 20);
    norm_kernel<<<blocks, NORM_THREADS, 0, k_stream(d)>>>((int)kind, x->data, x->dtype, z ? z->data : NULL,
                                                          z ? z->dtype : ATH_DTYPE_F32, w ? w->data : NULL,
                                                          w ? w->dtype : ATH_DTYPE_F32, w_offset, eps, out->data,
                                                          out->dtype, rows, D);
    return ath_gpu_launched("norm_kernel");
}
