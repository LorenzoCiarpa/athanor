/* attn.cu: causal softmax attention with grouped-query heads on the GPU.
 * See ops.h and attn.c.
 *
 * The keys a query sees are taken in pieces of ATH_ATTN_SPLIT (256). Each
 * piece is reduced to a partial softmax, the recurrence at the heart of
 * FlashAttention:
 *
 *   m  the largest score of the piece, l = Σ exp(s_j − m), acc = Σ exp(s_j − m) v_j
 *
 * computed over tiles of ATTN_TILE keys with a running maximum. The pieces
 * are then combined in order,
 *
 *   m' = max(m, m_p),  l = l e^(m − m') + l_p e^(m_p − m'),  acc likewise,
 *
 * and out = acc / l, which equals the reference's two passes (first the
 * maximum, then the sums) up to rounding. Two ways to run it (step 9):
 *
 *   rows   one block per query and head walks its pieces one after the
 *          other (prompts: plenty of blocks already);
 *   split  one block per piece and key head computes the partials of the
 *          query heads that read that key head, reading each key once, and
 *          a second kernel combines them (decode: one query would otherwise
 *          be a handful of blocks walking a long cache alone).
 *
 * Both call the same device functions with the same order of operations,
 * written with explicit rounding intrinsics (__fmaf_rn, __fmul_rn, ...) so
 * that the compiler cannot contract them differently in the two kernels:
 * a query gives the same bits either way, and decoding a token gives the
 * bits of the last row of a prompt that ends with it.
 *
 * The number of pieces comes from S, the rows of k and v, not from the
 * positions: with a whole cache as k and v and the positions in device
 * memory, the launch does not depend on how many tokens the sequence has,
 * which is what lets a decode step be captured once and replayed (step 9).
 * Pieces past a query's position are skipped. Scores, softmax and sums are
 * fp32, as in the reference op.
 */
#include "ops/cuda/kernel.cuh"

#define ATTN_THREADS 128
#define ATTN_TILE 32
#define ATTN_PER_THREAD ((ATH_ATTN_MAX_DV + ATTN_THREADS - 1) / ATTN_THREADS)
#define ATTN_MAX_GROUP 8         /* query heads one split block serves      */
#define ATTN_KREG (ATH_ATTN_MAX_DV / ATH_WARP)

/* The position of query t, or -1 if it is outside [0, S). */
static __device__ __forceinline__ int64_t attn_last(const int32_t *pos, int64_t S, int64_t T, int64_t t)
{
    const int64_t p = pos ? (int64_t)pos[t] : S - T + t;
    return p >= 0 && p < S ? p : -1;
}

/* The partial softmax of keys [j0, j1) of key head kh for G query heads,
 * whose queries are in qs [G][d] (shared). Each thread owns the output
 * elements tid, tid + ATTN_THREADS, ... of each head; m and l are the same
 * on every thread. sc is shared [G][ATTN_TILE]. */
static __device__ __forceinline__ void attn_piece(const float *qs, float *sc, int G, const void *k, int kdt,
                                                  const void *v, int vdt, int64_t nkv, int64_t kh, int64_t d,
                                                  int64_t dv, float scale, int64_t j0, int64_t j1,
                                                  float m[ATTN_MAX_GROUP], float l[ATTN_MAX_GROUP],
                                                  float acc[ATTN_MAX_GROUP][ATTN_PER_THREAD])
{
    const int tid = threadIdx.x, lane = tid % ATH_WARP, warp = tid / ATH_WARP;
#pragma unroll
    for (int g = 0; g < ATTN_MAX_GROUP; g++) {
        m[g] = -INFINITY;
        l[g] = 0.0f;
#pragma unroll
        for (int r = 0; r < ATTN_PER_THREAD; r++)
            acc[g][r] = 0.0f;
    }
    for (int64_t t0 = j0; t0 < j1; t0 += ATTN_TILE) {
        const int n = (int)(j1 - t0 < ATTN_TILE ? j1 - t0 : ATTN_TILE);
        /* scores: a warp per key, the lanes split the head dimension */
        for (int jj = warp; jj < n; jj += ATTN_THREADS / ATH_WARP) {
            const int64_t kb = ((t0 + jj) * nkv + kh) * d;
            float kr[ATTN_KREG];
#pragma unroll
            for (int i = 0; i < ATTN_KREG; i++)
                kr[i] = lane + (int64_t)i * ATH_WARP < d ? k_ld(k, kdt, kb + lane + (int64_t)i * ATH_WARP) : 0.0f;
            for (int g = 0; g < G; g++) {
                float s = 0.0f;
#pragma unroll
                for (int i = 0; i < ATTN_KREG; i++)
                    if (lane + (int64_t)i * ATH_WARP < d)
                        s = __fmaf_rn(qs[g * d + lane + i * ATH_WARP], kr[i], s);
                s = k_warp_sum(s);
                if (lane == 0)
                    sc[g * ATTN_TILE + jj] = __fmul_rn(s, scale);
            }
        }
        __syncthreads();
        /* the running softmax of the piece, per head */
#pragma unroll
        for (int g = 0; g < ATTN_MAX_GROUP; g++) {
            if (g >= G)
                break;
            float mt = m[g];
            for (int jj = 0; jj < n; jj++)
                mt = fmaxf(mt, sc[g * ATTN_TILE + jj]);
            const float corr = expf(__fsub_rn(m[g], mt));   /* 0 on the first tile, where m = −inf */
            l[g] = __fmul_rn(l[g], corr);
#pragma unroll
            for (int r = 0; r < ATTN_PER_THREAD; r++)
                acc[g][r] = __fmul_rn(acc[g][r], corr);
            m[g] = mt;
        }
        for (int jj = 0; jj < n; jj++) {
            const int64_t vb = ((t0 + jj) * nkv + kh) * dv;
            float vv[ATTN_PER_THREAD];
#pragma unroll
            for (int r = 0; r < ATTN_PER_THREAD; r++) {
                const int64_t i = tid + (int64_t)r * ATTN_THREADS;
                vv[r] = i < dv ? k_ld(v, vdt, vb + i) : 0.0f;
            }
#pragma unroll
            for (int g = 0; g < ATTN_MAX_GROUP; g++) {
                if (g >= G)
                    break;
                const float p = expf(__fsub_rn(sc[g * ATTN_TILE + jj], m[g]));
                l[g] = __fadd_rn(l[g], p);
#pragma unroll
                for (int r = 0; r < ATTN_PER_THREAD; r++)
                    acc[g][r] = __fmaf_rn(p, vv[r], acc[g][r]);
            }
        }
        __syncthreads();                    /* the next tile overwrites sc */
    }
}

/* (m, l, acc) ← (m, l, acc) combined with a piece's (mp, lp, accp). The
 * running state starts empty (m = −inf, l = 0) and every piece combined
 * has at least one key. */
static __device__ __forceinline__ void attn_combine(float *m, float *l, float acc[ATTN_PER_THREAD], float mp,
                                                    float lp, const float accp[ATTN_PER_THREAD])
{
    const float mn = fmaxf(*m, mp);
    const float a = expf(__fsub_rn(*m, mn)), b = expf(__fsub_rn(mp, mn));
    *l = __fadd_rn(__fmul_rn(*l, a), __fmul_rn(lp, b));
#pragma unroll
    for (int r = 0; r < ATTN_PER_THREAD; r++)
        acc[r] = __fadd_rn(__fmul_rn(acc[r], a), __fmul_rn(accp[r], b));
    *m = mn;
}

static __device__ __forceinline__ void attn_store(void *out, int odt, int64_t row, int64_t dv, float l,
                                                  const float acc[ATTN_PER_THREAD], bool valid)
{
#pragma unroll
    for (int r = 0; r < ATTN_PER_THREAD; r++) {
        const int64_t i = threadIdx.x + (int64_t)r * ATTN_THREADS;
        if (i < dv)
            k_st(out, odt, row * dv + i, valid ? __fdiv_rn(acc[r], l) : NAN);
    }
}

/* rows: one block per (query, head), its pieces in order. */
static __global__ void __launch_bounds__(ATTN_THREADS)
attn_rows(const void *q, int qdt, const void *k, int kdt, const void *v, int vdt, const int32_t *pos, void *out,
          int odt, int64_t T, int64_t S, int64_t nh, int64_t nkv, int64_t d, int64_t dv, float scale)
{
    extern __shared__ float sh[];
    float *qs = sh, *sc = sh + d;
    const int64_t th = blockIdx.x, t = th / nh, kh = th % nh / (nh / nkv);
    const int64_t last = attn_last(pos, S, T, t);
    for (int64_t i = threadIdx.x; i < d; i += ATTN_THREADS)
        qs[i] = k_ld(q, qdt, th * d + i);
    __syncthreads();
    float m = -INFINITY, l = 0.0f, acc[ATTN_PER_THREAD];
#pragma unroll
    for (int r = 0; r < ATTN_PER_THREAD; r++)
        acc[r] = 0.0f;
    for (int64_t j0 = 0; j0 <= last; j0 += ATH_ATTN_SPLIT) {
        const int64_t j1 = j0 + ATH_ATTN_SPLIT < last + 1 ? j0 + ATH_ATTN_SPLIT : last + 1;
        float mp[ATTN_MAX_GROUP], lp[ATTN_MAX_GROUP], ap[ATTN_MAX_GROUP][ATTN_PER_THREAD];
        attn_piece(qs, sc, 1, k, kdt, v, vdt, nkv, kh, d, dv, scale, j0, j1, mp, lp, ap);
        attn_combine(&m, &l, acc, mp[0], lp[0], ap[0]);
    }
    attn_store(out, odt, th, dv, l, acc, last >= 0);
}

/* split, first kernel: block (piece, group of G query heads, query) writes
 * the partials of those heads to part [T, nh, pieces, dv + 2] as m, l,
 * acc[0 .. dv). */
static __global__ void __launch_bounds__(ATTN_THREADS)
attn_split(const void *q, int qdt, const void *k, int kdt, const void *v, int vdt, const int32_t *pos,
           float *part, int64_t T, int64_t S, int64_t nh, int64_t nkv, int64_t d, int64_t dv, float scale, int G)
{
    extern __shared__ float sh[];
    float *qs = sh, *sc = sh + (int64_t)G * d;
    const int64_t pieces = gridDim.x, p = blockIdx.x, t = blockIdx.z, h0 = (int64_t)blockIdx.y * G;
    const int64_t kh = h0 / (nh / nkv);
    const int64_t last = attn_last(pos, S, T, t), j0 = p * ATH_ATTN_SPLIT;
    if (j0 > last)
        return;                          /* the combine reads pieces up to the position only */
    const int64_t j1 = j0 + ATH_ATTN_SPLIT < last + 1 ? j0 + ATH_ATTN_SPLIT : last + 1;
    for (int64_t i = threadIdx.x; i < G * d; i += ATTN_THREADS)
        qs[i] = k_ld(q, qdt, (t * nh + h0) * d + i);
    __syncthreads();
    float m[ATTN_MAX_GROUP], l[ATTN_MAX_GROUP], acc[ATTN_MAX_GROUP][ATTN_PER_THREAD];
    attn_piece(qs, sc, G, k, kdt, v, vdt, nkv, kh, d, dv, scale, j0, j1, m, l, acc);
#pragma unroll
    for (int g = 0; g < ATTN_MAX_GROUP; g++) {
        if (g >= G)
            break;
        float *o = part + ((t * nh + h0 + g) * pieces + p) * (dv + 2);
        if (threadIdx.x == 0)
            o[0] = m[g], o[1] = l[g];
#pragma unroll
        for (int r = 0; r < ATTN_PER_THREAD; r++) {
            const int64_t i = threadIdx.x + (int64_t)r * ATTN_THREADS;
            if (i < dv)
                o[2 + i] = acc[g][r];
        }
    }
}

/* split, second kernel: one block per (query, head) combines its pieces. */
static __global__ void __launch_bounds__(ATTN_THREADS)
attn_merge(const float *part, const int32_t *pos, void *out, int odt, int64_t T, int64_t S, int64_t nh,
           int64_t dv, int64_t pieces)
{
    const int64_t th = blockIdx.x, t = th / nh;
    const int64_t last = attn_last(pos, S, T, t);
    float m = -INFINITY, l = 0.0f, acc[ATTN_PER_THREAD];
#pragma unroll
    for (int r = 0; r < ATTN_PER_THREAD; r++)
        acc[r] = 0.0f;
    for (int64_t p = 0; p * ATH_ATTN_SPLIT <= last; p++) {
        const float *o = part + (th * pieces + p) * (dv + 2);
        float ap[ATTN_PER_THREAD];
#pragma unroll
        for (int r = 0; r < ATTN_PER_THREAD; r++) {
            const int64_t i = threadIdx.x + (int64_t)r * ATTN_THREADS;
            ap[r] = i < dv ? o[2 + i] : 0.0f;
        }
        attn_combine(&m, &l, acc, o[0], o[1], ap);
    }
    attn_store(out, odt, th, dv, l, acc, last >= 0);
}

extern "C" ath_status ath_attn_launch(const ath_device *d, const ath_tensor *q, const ath_tensor *k,
                                      const ath_tensor *v, const ath_tensor *pos, float scale, float *part,
                                      ath_tensor *out)
{
    const int64_t T = q->shape[0], nh = q->shape[1], hd = q->shape[2];
    const int64_t S = k->shape[0], nkv = k->shape[1], dv = v->shape[2];
    const int32_t *p = pos ? (const int32_t *)pos->data : NULL;
    if (T * nh == 0 || dv == 0)
        return ATH_OK;
    if (T * nh > INT32_MAX) {
        ath_log(ATH_LOG_ERROR, "attn: %lld queries × heads, more than one launch holds", (long long)(T * nh));
        return ATH_ERR_UNSUPPORTED;
    }
    ATH_CHECK(ath_gpu_use(d));
    if (!part) {
        const size_t smem = (size_t)(hd + ATTN_TILE) * sizeof(float);
        attn_rows<<<(unsigned)(T * nh), ATTN_THREADS, smem, k_stream(d)>>>(
            q->data, q->dtype, k->data, k->dtype, v->data, v->dtype, p, out->data, out->dtype, T, S, nh, nkv, hd, dv,
            scale);
        return ath_gpu_launched("attn_rows");
    }
    /* G: the largest divisor of the group size up to ATTN_MAX_GROUP */
    const int64_t group = nh / nkv;
    int G = (int)(group < ATTN_MAX_GROUP ? group : ATTN_MAX_GROUP);
    while (group % G)
        G--;
    const int64_t pieces = (S + ATH_ATTN_SPLIT - 1) / ATH_ATTN_SPLIT;
    const dim3 grid((unsigned)pieces, (unsigned)(nh / G), (unsigned)T);
    const size_t smem = (size_t)G * (hd + ATTN_TILE) * sizeof(float);
    attn_split<<<grid, ATTN_THREADS, smem, k_stream(d)>>>(q->data, q->dtype, k->data, k->dtype, v->data, v->dtype, p,
                                                         part, T, S, nh, nkv, hd, dv, scale, G);
    ATH_CHECK(ath_gpu_launched("attn_split"));
    attn_merge<<<(unsigned)(T * nh), ATTN_THREADS, 0, k_stream(d)>>>(part, p, out->data, out->dtype, T, S, nh, dv,
                                                                    pieces);
    return ath_gpu_launched("attn_merge");
}
