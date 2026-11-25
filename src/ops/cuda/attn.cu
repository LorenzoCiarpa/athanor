/* attn.cu: causal softmax attention with grouped-query heads on the GPU.
 * See ops.h and attn.c.
 *
 * One block per query and head. The block walks the keys the query sees in
 * tiles of ATTN_TILE and keeps a running softmax, the recurrence at the
 * heart of FlashAttention, written for one query:
 *
 *   m  the largest score so far, l = Σ exp(s_j − m), acc = Σ exp(s_j − m) v_j
 *   for each tile: m' = max(m, scores of the tile)
 *                  l, acc scaled by exp(m − m'), the tile's terms added
 *   out = acc / l
 *
 * which equals the reference's two passes (first the maximum, then the
 * sums) up to rounding, without storing a score per key. Each warp computes
 * the scores of some keys of the tile (lanes split the head dimension, then
 * k_warp_sum); each thread owns dv / ATTN_THREADS output dimensions. Scores,
 * softmax and sums are fp32, as in the reference op.
 *
 * Parallelism is one block per (query, head): plenty for a prompt, a few
 * blocks for one decoded token. Splitting the keys of one query over
 * several blocks (split-K) is step 9; tiled prefill on tensor cores is
 * step 10.
 */
#include "ops/cuda/kernel.cuh"

#define ATTN_THREADS 128
#define ATTN_TILE 32
#define ATTN_PER_THREAD ((ATH_ATTN_MAX_DV + ATTN_THREADS - 1) / ATTN_THREADS)

static __global__ void __launch_bounds__(ATTN_THREADS)
attn_kernel(const void *q, int qdt, const void *k, int kdt, const void *v, int vdt, void *out, int odt, int64_t T,
            int64_t S, int64_t nh, int64_t nkv, int64_t d, int64_t dv, float scale)
{
    __shared__ float qs[ATH_ATTN_MAX_DV];   /* the query, as floats       */
    __shared__ float sc[ATTN_TILE];         /* the scores of one tile     */
    const int tid = threadIdx.x, lane = tid % ATH_WARP, warp = tid / ATH_WARP;
    const int64_t th = blockIdx.x;          /* t · nh + h                 */
    const int64_t t = th / nh, kh = th % nh / (nh / nkv);
    const int64_t last = S - T + t;         /* the last key this query sees */

    for (int64_t i = tid; i < d; i += ATTN_THREADS)
        qs[i] = k_ld(q, qdt, th * d + i);
    __syncthreads();

    float acc[ATTN_PER_THREAD];
#pragma unroll
    for (int r = 0; r < ATTN_PER_THREAD; r++)
        acc[r] = 0.0f;
    float m = -INFINITY, l = 0.0f;

    for (int64_t j0 = 0; j0 <= last; j0 += ATTN_TILE) {
        const int n = (int)(last + 1 - j0 < ATTN_TILE ? last + 1 - j0 : ATTN_TILE);
        for (int jj = warp; jj < n; jj += ATTN_THREADS / ATH_WARP) {
            const int64_t kb = ((j0 + jj) * nkv + kh) * d;
            float s = 0.0f;
            for (int64_t i = lane; i < d; i += ATH_WARP)
                s += qs[i] * k_ld(k, kdt, kb + i);
            s = k_warp_sum(s);
            if (lane == 0)
                sc[jj] = s * scale;
        }
        __syncthreads();

        float mt = m;
        for (int jj = 0; jj < n; jj++)
            mt = fmaxf(mt, sc[jj]);
        const float corr = expf(m - mt);    /* 0 on the first tile, where m = −inf */
        l *= corr;
#pragma unroll
        for (int r = 0; r < ATTN_PER_THREAD; r++)
            acc[r] *= corr;
        for (int jj = 0; jj < n; jj++) {
            const float p = expf(sc[jj] - mt);
            const int64_t vb = ((j0 + jj) * nkv + kh) * dv;
            l += p;
#pragma unroll
            for (int r = 0; r < ATTN_PER_THREAD; r++) {
                const int64_t i = tid + (int64_t)r * ATTN_THREADS;
                if (i < dv)
                    acc[r] += p * k_ld(v, vdt, vb + i);
            }
        }
        m = mt;
        __syncthreads();                    /* the next tile overwrites sc */
    }
#pragma unroll
    for (int r = 0; r < ATTN_PER_THREAD; r++) {
        const int64_t i = tid + (int64_t)r * ATTN_THREADS;
        if (i < dv)
            k_st(out, odt, th * dv + i, acc[r] / l);
    }
}

extern "C" ath_status ath_attn_launch(const ath_device *d, const ath_tensor *q, const ath_tensor *k,
                                      const ath_tensor *v, float scale, ath_tensor *out)
{
    const int64_t T = q->shape[0], nh = q->shape[1], hd = q->shape[2];
    const int64_t S = k->shape[0], nkv = k->shape[1], dv = v->shape[2];
    if (T * nh == 0 || dv == 0)
        return ATH_OK;
    if (T * nh > INT32_MAX) {
        ath_log(ATH_LOG_ERROR, "attn: %lld queries × heads, more than one launch holds", (long long)(T * nh));
        return ATH_ERR_UNSUPPORTED;
    }
    ATH_CHECK(ath_gpu_use(d));
    attn_kernel<<<(unsigned)(T * nh), ATTN_THREADS, 0, k_stream(d)>>>(q->data, q->dtype, k->data, k->dtype, v->data,
                                                                      v->dtype, out->data, out->dtype, T, S, nh, nkv,
                                                                      hd, dv, scale);
    return ath_gpu_launched("attn_kernel");
}
