/* moe.cu: the mixture-of-experts kernels (step 12). See ops.h and moe.c.
 *
 * route: one warp per token. The lanes take the logits in turn; the maximum
 * and the sum of exponentials are warp reductions; the k choices are k warp
 * arg-max reductions over what is left (the larger probability, on ties the
 * lower index, as the reference); lane 0 renormalizes in order of rank and
 * writes the choices sorted by expert index.
 *
 * combine: one thread per output element, the routed sum as a chain of fused
 * multiply-adds in the order of the choices, then the shared term, as
 * moe.c.
 *
 * experts: the routed experts whose weights are on the device. expert_up: a
 * half-warp per (token, choice, row i of the expert's width): g = gate_i · x
 * and u = up_i · x as canonical Q8_0 dot products (k_q8_dot), then
 * h_i = silu(g) · u with the canonical SiLU, each rounded as moe.c rounds
 * them. expert_down: a half-warp per (token, choice, output row): down_r · h.
 * An expert whose pointer in the table is NULL is not on the device: its
 * half-warps skip it (the CPU computes it, with the same bits), and a block
 * whose rows all belong to such experts leaves at once. Weights in f32 or
 * bf16 (a model whose experts are not Q8_0, which tiering does not place)
 * take a plain dot: lane j sums the terms k = j mod 16 in order, then the
 * tree; within tolerance of the reference, not its bits.
 */
#include "ops/cuda/q8_0.cuh"
#include "ops/ops.h"

static __global__ void route_kernel(const float *logits, int32_t *ids, void *w, int wdt, int64_t T, int64_t n,
                                    int k)
{
    const int lane = threadIdx.x % ATH_WARP;
    const int64_t t = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / ATH_WARP;
    if (t >= T)
        return;
    const float *l = logits + t * n;
    float mx = -INFINITY;
    for (int64_t e = lane; e < n; e += ATH_WARP)
        mx = fmaxf(mx, l[e]);
    for (int m = ATH_WARP / 2; m > 0; m >>= 1)
        mx = fmaxf(mx, ath_shfl_xor(mx, m));
    float sum = 0.0f;
    for (int64_t e = lane; e < n; e += ATH_WARP)
        sum += expf(l[e] - mx);
    sum = k_warp_sum(sum);

    int32_t pick[ATH_MOE_MAX_K];
    float p[ATH_MOE_MAX_K];
    for (int j = 0; j < k; j++) {
        float bv = -1.0f;
        int32_t bi = INT32_MAX;
        for (int64_t e = lane; e < n; e += ATH_WARP) {
            bool taken = false;
            for (int i = 0; i < j; i++)
                taken |= pick[i] == e;
            const float pe = expf(l[e] - mx) / sum;
            if (!taken && (pe > bv || (pe == bv && e < bi))) {
                bv = pe;
                bi = (int32_t)e;
            }
        }
        for (int m = ATH_WARP / 2; m > 0; m >>= 1) {
            const float ov = ath_shfl_xor(bv, m);
            const int32_t oi = ath_shfl_xor(bi, m);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
            }
        }
        pick[j] = bi;
        p[j] = bv;
    }
    if (lane != 0)
        return;
    float ps = 0.0f;
    for (int j = 0; j < k; j++)
        ps += p[j];
    for (int j = 0; j < k; j++) {
        int r = 0;
        for (int i = 0; i < k; i++)
            r += pick[i] < pick[j];
        ids[t * k + r] = pick[j];
        k_st(w, wdt, t * k + r, p[j] / ps);
    }
}

extern "C" ath_status ath_moe_route_launch(const ath_device *d, const ath_tensor *logits, ath_tensor *ids,
                                           ath_tensor *w)
{
    const int64_t T = logits->shape[0], n = logits->shape[1];
    const int k = (int)ids->shape[1];
    if (T == 0)
        return ATH_OK;
    ATH_CHECK(ath_gpu_use(d));
    route_kernel<<<k_blocks(T * ATH_WARP, 128), 128, 0, k_stream(d)>>>((const float *)logits->data,
                                                                       (int32_t *)ids->data, w->data, w->dtype, T, n,
                                                                       k);
    return ath_gpu_launched("route_kernel");
}

static __global__ void combine_kernel(const void *y, int ydt, const void *w, int wdt, const void *ys, int ysdt,
                                      const void *sg, int sgdt, void *out, int odt, int64_t T, int64_t k, int64_t H)
{
    K_FOR(e, T * H) {
        const int64_t t = e / H, h = e % H;
        float acc = 0.0f;
        for (int64_t j = 0; j < k; j++)
            acc = fmaf(k_ld(w, wdt, t * k + j), k_ld(y, ydt, (t * k + j) * H + h), acc);
        if (ys)
            acc = acc + k_ld(ys, ysdt, e) * (1.0f / (1.0f + expf(-k_ld(sg, sgdt, t))));
        k_st(out, odt, e, acc);
    }
}

extern "C" ath_status ath_moe_combine_launch(const ath_device *d, const ath_tensor *y, const ath_tensor *w,
                                             const ath_tensor *ys, const ath_tensor *sg, ath_tensor *out)
{
    const int64_t T = y->shape[0], k = y->shape[1], H = y->shape[2];
    if (T * H == 0)
        return ATH_OK;
    ATH_CHECK(ath_gpu_use(d));
    combine_kernel<<<k_blocks(T * H, 256), 256, 0, k_stream(d)>>>(
        y->data, y->dtype, w->data, w->dtype, ys ? ys->data : NULL, ys ? ys->dtype : 0, sg ? sg->data : NULL,
        sg ? sg->dtype : 0, out->data, out->dtype, T, k, H);
    return ath_gpu_launched("combine_kernel");
}

/* ---- the routed experts on the device -------------------------------------------- */

#define EX_WARPS 4                    /* 8 rows (half-warps) per block */

/* A dot product of a float row (f32 or bf16) with x, by the half-warp. */
static __device__ __forceinline__ float k_float_dot(const void *row, int wdt, const void *x, int xdt, int64_t xoff,
                                                    int64_t K, int j, bool live)
{
    float acc = 0.0f;
    for (int64_t c = j; live && c < K; c += 16)
        acc = fmaf(k_ld(row, wdt, c), k_ld(x, xdt, xoff + c), acc);
    return k_q8_tree(acc);
}

static __device__ __forceinline__ float k_row_dot(const void *w, int wdt, int64_t row, const void *x, int xdt,
                                                  int64_t xoff, int64_t K, unsigned char *region, int j, bool live)
{
    if (wdt == ATH_DTYPE_Q8_0)
        return k_q8_dot((const uint8_t *)w + (live ? row : 0) * (K / ATH_Q8_0_BLOCK) * ATH_Q8_0_BYTES, x, xdt, xoff,
                        K, region, j, live);
    const size_t es = wdt == ATH_DTYPE_BF16 ? 2 : 4;
    return k_float_dot((const char *)w + (size_t)(live ? row : 0) * (size_t)K * es, wdt, x, xdt, xoff, K, j, live);
}

/* h[t, c, i] = silu(gate_i · x_t) · (up_i · x_t) for the experts on the device. */
static __global__ void __launch_bounds__(EX_WARPS * ATH_WARP)
expert_up(const void *x, int xdt, const int32_t *ids, const void *const *tab, int wdt, int64_t n_experts, void *h,
          int hdt, int64_t T, int64_t k, int64_t I, int64_t H)
{
    __shared__ __align__(16) unsigned char stage[2 * EX_WARPS][Q8_REGION];
    const int j = threadIdx.x & 15, half = threadIdx.x >> 4;
    const int64_t o = (int64_t)blockIdx.x * (2 * EX_WARPS) + half;   /* (t, c, i) */
    const int64_t tc = o / I, i = o % I;
    const void *gw = NULL, *uw = NULL;
    if (o < T * k * I) {
        const int32_t e = ids[tc];
        if (e >= 0 && e < n_experts) {
            gw = tab[e];
            uw = tab[n_experts + e];
        }
    }
    const bool live = gw != NULL && uw != NULL;
    if (!__syncthreads_or(live))
        return;
    const int64_t t = tc / k;
    const float g = k_row_dot(gw, wdt, i, x, xdt, t * H, H, stage[half], j, live);
    const float u = k_row_dot(uw, wdt, i, x, xdt, t * H, H, stage[half], j, live);
    if (live && j == 0)
        k_st(h, hdt, o, ath_q8_silu(k_round(hdt, g)) * k_round(hdt, u));
}

/* y[t, c, r] = down_r · h[t, c] for the experts on the device. */
static __global__ void __launch_bounds__(EX_WARPS * ATH_WARP)
expert_down(const void *h, int hdt, const int32_t *ids, const void *const *tab, int wdt, int64_t n_experts, void *y,
            int ydt, int64_t T, int64_t k, int64_t I, int64_t H)
{
    __shared__ __align__(16) unsigned char stage[2 * EX_WARPS][Q8_REGION];
    const int j = threadIdx.x & 15, half = threadIdx.x >> 4;
    const int64_t o = (int64_t)blockIdx.x * (2 * EX_WARPS) + half;   /* (t, c, r) */
    const int64_t tc = o / H, r = o % H;
    const void *dw = NULL;
    if (o < T * k * H) {
        const int32_t e = ids[tc];
        if (e >= 0 && e < n_experts)
            dw = tab[2 * n_experts + e];
    }
    const bool live = dw != NULL;
    if (!__syncthreads_or(live))
        return;
    const float v = k_row_dot(dw, wdt, r, h, hdt, tc * I, I, stage[half], j, live);
    if (live && j == 0)
        k_st(y, ydt, o, v);
}

extern "C" ath_status ath_moe_experts_launch(const ath_device *d, const ath_tensor *x, const ath_tensor *ids,
                                             const void *const *tab, ath_dtype wdt, int64_t n_experts, int64_t inter,
                                             ath_tensor *h, ath_tensor *y)
{
    const int64_t T = x->shape[0], H = x->shape[1], k = ids->shape[1];
    if (T * k == 0)
        return ATH_OK;
    ATH_CHECK(ath_gpu_use(d));
    const int per = 2 * EX_WARPS;
    expert_up<<<(unsigned)((T * k * inter + per - 1) / per), EX_WARPS * ATH_WARP, 0, k_stream(d)>>>(
        x->data, x->dtype, (const int32_t *)ids->data, tab, wdt, n_experts, h->data, h->dtype, T, k, inter, H);
    ATH_CHECK(ath_gpu_launched("expert_up"));
    expert_down<<<(unsigned)((T * k * H + per - 1) / per), EX_WARPS * ATH_WARP, 0, k_stream(d)>>>(
        h->data, h->dtype, (const int32_t *)ids->data, tab, wdt, n_experts, y->data, y->dtype, T, k, inter, H);
    return ath_gpu_launched("expert_down");
}

/* The CPU's outputs (in pinned host memory, which the device reads over the
 * bus: unified addressing) copied into y where the expert is not on the
 * device: only those rows cross the bus. */
static __global__ void merge_kernel(const int32_t *ids, const void *const *tab, int64_t n_experts, const void *src,
                                    void *y, int64_t T, int64_t k, int64_t H, size_t es)
{
    K_FOR(e, T * k * H) {
        const int64_t tc = e / H;
        const int32_t x = ids[tc];
        if (x >= 0 && x < n_experts && tab[x] == NULL) {
            if (es == 2)
                ((uint16_t *)y)[e] = ((const uint16_t *)src)[e];
            else
                ((float *)y)[e] = ((const float *)src)[e];
        }
    }
}

extern "C" ath_status ath_moe_merge_launch(const ath_device *d, const ath_tensor *ids, const void *const *tab,
                                           int64_t n_experts, const void *src, ath_tensor *y)
{
    const int64_t T = y->shape[0], k = y->shape[1], H = y->shape[2];
    if (T * k * H == 0)
        return ATH_OK;
    ATH_CHECK(ath_gpu_use(d));
    merge_kernel<<<k_blocks(T * k * H, 256), 256, 0, k_stream(d)>>>((const int32_t *)ids->data, tab, n_experts, src,
                                                                    y->data, T, k, H, ath_dtype_size(y->dtype));
    return ath_gpu_launched("merge_kernel");
}
