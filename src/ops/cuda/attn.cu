/* attn.cu: causal softmax attention with grouped-query heads on the GPU.
 * See ops.h and attn.c.
 *
 * The keys a query sees are taken in pieces of ATH_ATTN_SPLIT (64). Each
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
#define ATTN_MAX_GROUP 4         /* query heads one split block serves      */
#define ATTN_VKEYS 4             /* keys whose values are loaded at once    */
#define ATTN_KREG (ATH_ATTN_MAX_DV / ATH_WARP)

/* Keys or values as the kernels read them: f32 or bf16, or (step 14) an
 * int8 cache with a scale per row (one token, one KV head) read back as
 * q · s rounded to rnd, the queries' dtype; the arithmetic after the load
 * is the same either way, so attention over an int8 cache gives the bits it
 * gives over the cache those values make. */
typedef struct {
    const void  *p;
    int          dt;     /* ATH_DTYPE_F32, ATH_DTYPE_BF16 or ATH_DTYPE_I8 */
    const float *s;      /* I8: the row scales                            */
    int          rnd;    /* I8: the dtype the values round to             */
} kvsrc;

/* Element i of a [S, nkv, n] buffer whose rows have n elements. */
static __device__ __forceinline__ float kv_ld(kvsrc a, int64_t i, int64_t n)
{
    if (a.dt == ATH_DTYPE_I8)
        return k_round(a.rnd, __fmul_rn((float)((const int8_t *)a.p)[i], a.s[i / n]));
    return k_ld(a.p, a.dt, i);
}

/* Elements 8e … 8e + 7, in one row (n a multiple of 8). */
static __device__ __forceinline__ void kv_ld8(kvsrc a, int64_t e, int64_t n, float f[8])
{
    if (a.dt != ATH_DTYPE_I8) {
        k_ld8(a.p, a.dt, e, f);
        return;
    }
    const uint2 u = ((const uint2 *)a.p)[e];
    const float s = a.s[8 * e / n];
    const uint32_t w[2] = {u.x, u.y};
#pragma unroll
    for (int i = 0; i < 8; i++)
        f[i] = k_round(a.rnd, __fmul_rn((float)(int8_t)(w[i / 4] >> (8 * (i % 4))), s));
}

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
static __device__ __forceinline__ void attn_piece(const float *qs, float *sc, int G, kvsrc k, kvsrc v,
                                                  int64_t nkv, int64_t kh, int64_t d,
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
                kr[i] = lane + (int64_t)i * ATH_WARP < d ? kv_ld(k, kb + lane + (int64_t)i * ATH_WARP, d) : 0.0f;
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
        /* the values of ATTN_VKEYS keys are loaded before any is used, so
         * their loads are in flight together; the sums still go key by key */
        for (int j0 = 0; j0 < n; j0 += ATTN_VKEYS) {
            float vv[ATTN_VKEYS][ATTN_PER_THREAD];
#pragma unroll
            for (int q = 0; q < ATTN_VKEYS; q++) {
                const int64_t vb = ((t0 + j0 + q) * nkv + kh) * dv;
#pragma unroll
                for (int r = 0; r < ATTN_PER_THREAD; r++) {
                    const int64_t i = tid + (int64_t)r * ATTN_THREADS;
                    vv[q][r] = j0 + q < n && i < dv ? kv_ld(v, vb + i, dv) : 0.0f;
                }
            }
#pragma unroll
            for (int q = 0; q < ATTN_VKEYS; q++) {
                if (j0 + q >= n)
                    break;
#pragma unroll
                for (int g = 0; g < ATTN_MAX_GROUP; g++) {
                    if (g >= G)
                        break;
                    const float p = expf(__fsub_rn(sc[g * ATTN_TILE + j0 + q], m[g]));
                    l[g] = __fadd_rn(l[g], p);
#pragma unroll
                    for (int r = 0; r < ATTN_PER_THREAD; r++)
                        acc[g][r] = __fmaf_rn(p, vv[q][r], acc[g][r]);
                }
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
attn_rows(const void *q, int qdt, kvsrc k, kvsrc v, const int32_t *pos, void *out,
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
        attn_piece(qs, sc, 1, k, v, nkv, kh, d, dv, scale, j0, j1, mp, lp, ap);
        attn_combine(&m, &l, acc, mp[0], lp[0], ap[0]);
    }
    attn_store(out, odt, th, dv, l, acc, last >= 0);
}

/* split, first kernel: block (piece, group of G query heads, query) writes
 * the partials of those heads to part [T, nh, pieces, dv + 2] as m, l,
 * acc[0 .. dv). */
static __global__ void __launch_bounds__(ATTN_THREADS)
attn_split(const void *q, int qdt, kvsrc k, kvsrc v, const int32_t *pos,
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
    attn_piece(qs, sc, G, k, v, nkv, kh, d, dv, scale, j0, j1, m, l, acc);
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

/* ---- the fast path: head dims up to 256 in 16-byte pieces (step 9) ---------------
 *
 * The kernels above load one key, or one key's values, at a time and wait
 * for each: a tile costs some 16 trips to memory one after the other, which
 * at decode (a few blocks) is the whole time. Here warp w takes keys
 * 8w … 8w + 7 of a tile and lane l the 8 elements 8l … 8l + 7 of a head:
 * the warp loads its 8 keys at once, then its 8 value rows at once, one
 * trip each. The softmax of a tile runs head g on warp g and key j on lane
 * j (warp reductions for its maximum and sum), once, instead of every
 * thread walking the scores; each warp keeps the weighted sum of its own
 * keys, and the four are added in warp order at the end of the piece. Rows and split use these functions both,
 * so they keep giving the same bits. */
#define FAST_DIM 256
#define FAST_KEYS (ATTN_TILE / (ATTN_THREADS / ATH_WARP))
#define FAST_WARPS (ATTN_THREADS / ATH_WARP)

/* The largest of the 32 values of a warp, on every lane. */
static __device__ __forceinline__ float k_warp_max(float v)
{
    for (int m = ATH_WARP / 2; m > 0; m >>= 1)
        v = fmaxf(v, ath_shfl_xor(v, m));
    return v;
}

static __device__ __forceinline__ void fast_piece(const float *qs, float *sc, float *xch, float *st, int G, kvsrc k,
                                                  kvsrc v, int64_t nkv, int64_t kh, int64_t d,
                                                  int64_t dv, float scale, int64_t j0, int64_t j1,
                                                  float m[ATTN_MAX_GROUP], float l[ATTN_MAX_GROUP],
                                                  float accp[ATTN_MAX_GROUP][8])
{
    const int lane = threadIdx.x % ATH_WARP, warp = threadIdx.x / ATH_WARP;
    float accw[ATTN_MAX_GROUP][8];
#pragma unroll
    for (int g = 0; g < ATTN_MAX_GROUP; g++) {
        m[g] = -INFINITY;
        l[g] = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; i++)
            accw[g][i] = 0.0f;
    }
    const bool kq = lane < d / 8, vq = lane < dv / 8;
    for (int64_t t0 = j0; t0 < j1; t0 += ATTN_TILE) {
        const int n = (int)(j1 - t0 < ATTN_TILE ? j1 - t0 : ATTN_TILE);
        float kv[FAST_KEYS][8];
#pragma unroll
        for (int q = 0; q < FAST_KEYS; q++) {
            const int jj = warp * FAST_KEYS + q;
            if (jj < n && kq)
                kv_ld8(k, (((t0 + jj) * nkv + kh) * d) / 8 + lane, d, kv[q]);
            else
#pragma unroll
                for (int i = 0; i < 8; i++)
                    kv[q][i] = 0.0f;
        }
#pragma unroll
        for (int g = 0; g < ATTN_MAX_GROUP; g++)
#pragma unroll
            for (int q = 0; q < FAST_KEYS; q++) {
                if (g >= G)
                    break;
                float s = 0.0f;
#pragma unroll
                for (int i = 0; i < 8; i++)
                    s = kq ? __fmaf_rn(qs[g * d + 8 * lane + i], kv[q][i], s) : s;
                s = k_warp_sum(s);
                const int jj = warp * FAST_KEYS + q;
                if (lane == 0 && jj < n)
                    sc[g * ATTN_TILE + jj] = __fmul_rn(s, scale);
            }
        __syncthreads();
        /* the tile's softmax, head g on warp g, key j on lane j: the
         * probabilities replace the scores in sc, and st gets the new
         * maximum, the correction of the old one and the tile's sum */
        if (warp < G) {
            const int g = warp;
            const float sv = lane < n ? sc[g * ATTN_TILE + lane] : -INFINITY;
            const float mt = fmaxf(m[g], k_warp_max(sv));
            const float pv = lane < n ? expf(__fsub_rn(sv, mt)) : 0.0f;
            const float lt = k_warp_sum(pv);
            sc[g * ATTN_TILE + lane] = pv;
            if (lane == 0) {
                st[3 * g] = mt;
                st[3 * g + 1] = expf(__fsub_rn(m[g], mt));    /* 0 on the first tile, where m = −inf */
                st[3 * g + 2] = lt;
            }
        }
        __syncthreads();
#pragma unroll
        for (int g = 0; g < ATTN_MAX_GROUP; g++) {
            if (g >= G)
                break;
            const float corr = st[3 * g + 1];
            l[g] = __fadd_rn(__fmul_rn(l[g], corr), st[3 * g + 2]);
#pragma unroll
            for (int i = 0; i < 8; i++)
                accw[g][i] = __fmul_rn(accw[g][i], corr);
            m[g] = st[3 * g];
        }
#pragma unroll
        for (int q = 0; q < FAST_KEYS; q++) {
            const int jj = warp * FAST_KEYS + q;
            if (jj < n && vq)
                kv_ld8(v, (((t0 + jj) * nkv + kh) * dv) / 8 + lane, dv, kv[q]);
        }
#pragma unroll
        for (int q = 0; q < FAST_KEYS; q++) {
            const int jj = warp * FAST_KEYS + q;
            if (jj >= n)
                break;
#pragma unroll
            for (int g = 0; g < ATTN_MAX_GROUP; g++) {
                if (g >= G)
                    break;
                const float p = sc[g * ATTN_TILE + jj];          /* the probability */
#pragma unroll
                for (int i = 0; i < 8; i++)
                    accw[g][i] = __fmaf_rn(p, kv[q][i], accw[g][i]);
            }
        }
        __syncthreads();                    /* the next tile overwrites sc and st */
    }
    /* the four warps' sums, in warp order, on warp 0 */
#pragma unroll
    for (int g = 0; g < ATTN_MAX_GROUP; g++)
#pragma unroll
        for (int i = 0; i < 8; i++)
            if (g < G)
                xch[(warp * ATTN_MAX_GROUP + g) * FAST_DIM + 8 * lane + i] = accw[g][i];
    __syncthreads();
    if (warp == 0)
#pragma unroll
        for (int g = 0; g < ATTN_MAX_GROUP; g++)
#pragma unroll
            for (int i = 0; i < 8; i++) {
                if (g >= G)
                    break;
                float a = xch[g * FAST_DIM + 8 * lane + i];
                for (int w = 1; w < FAST_WARPS; w++)
                    a = __fadd_rn(a, xch[(w * ATTN_MAX_GROUP + g) * FAST_DIM + 8 * lane + i]);
                accp[g][i] = a;
            }
    __syncthreads();
}

static __global__ void __launch_bounds__(ATTN_THREADS)
fast_rows(const void *q, int qdt, kvsrc k, kvsrc v, const int32_t *pos, void *out,
          int odt, int64_t T, int64_t S, int64_t nh, int64_t nkv, int64_t d, int64_t dv, float scale)
{
    __shared__ float qs[FAST_DIM], sc[ATTN_TILE], st[3], xch[FAST_WARPS * ATTN_MAX_GROUP * FAST_DIM];
    const int lane = threadIdx.x % ATH_WARP, warp = threadIdx.x / ATH_WARP;
    const int64_t th = blockIdx.x, t = th / nh, kh = th % nh / (nh / nkv);
    const int64_t last = attn_last(pos, S, T, t);
#pragma unroll
    for (int r = 0; r < FAST_DIM / ATTN_THREADS; r++) {
        const int64_t i = threadIdx.x + (int64_t)r * ATTN_THREADS;
        if (i < d)
            qs[i] = k_ld(q, qdt, th * d + i);
    }
    __syncthreads();
    float m = -INFINITY, l = 0.0f, acc[ATTN_PER_THREAD];
#pragma unroll
    for (int r = 0; r < ATTN_PER_THREAD; r++)
        acc[r] = 0.0f;
    for (int64_t j0 = 0; j0 <= last; j0 += ATH_ATTN_SPLIT) {
        const int64_t j1 = j0 + ATH_ATTN_SPLIT < last + 1 ? j0 + ATH_ATTN_SPLIT : last + 1;
        float mp[ATTN_MAX_GROUP], lp[ATTN_MAX_GROUP], ap[ATTN_MAX_GROUP][8];
        fast_piece(qs, sc, xch, st, 1, k, v, nkv, kh, d, dv, scale, j0, j1, mp, lp, ap);
        if (warp == 0)
            attn_combine(&m, &l, acc, mp[0], lp[0], ap[0]);
    }
    if (warp == 0)
#pragma unroll
        for (int i = 0; i < 8; i++)
            if (8 * lane + i < dv)
                k_st(out, odt, th * dv + 8 * lane + i, last >= 0 ? __fdiv_rn(acc[i], l) : NAN);
}

static __global__ void __launch_bounds__(ATTN_THREADS)
fast_split(const void *q, int qdt, kvsrc k, kvsrc v, const int32_t *pos, float *part,
           int64_t T, int64_t S, int64_t nh, int64_t nkv, int64_t d, int64_t dv, float scale, int G)
{
    __shared__ float qs[ATTN_MAX_GROUP * FAST_DIM], sc[ATTN_MAX_GROUP * ATTN_TILE], st[3 * ATTN_MAX_GROUP],
        xch[FAST_WARPS * ATTN_MAX_GROUP * FAST_DIM];
    const int lane = threadIdx.x % ATH_WARP, warp = threadIdx.x / ATH_WARP;
    const int64_t pieces = gridDim.x, p = blockIdx.x, t = blockIdx.z, h0 = (int64_t)blockIdx.y * G;
    const int64_t kh = h0 / (nh / nkv);
    const int64_t last = attn_last(pos, S, T, t), j0 = p * ATH_ATTN_SPLIT;
    if (j0 > last)
        return;
    const int64_t j1 = j0 + ATH_ATTN_SPLIT < last + 1 ? j0 + ATH_ATTN_SPLIT : last + 1;
#pragma unroll
    for (int r = 0; r < ATTN_MAX_GROUP * FAST_DIM / ATTN_THREADS; r++) {   /* the loads at once */
        const int64_t i = threadIdx.x + (int64_t)r * ATTN_THREADS;
        if (i < G * d)
            qs[i] = k_ld(q, qdt, (t * nh + h0) * d + i);
    }
    __syncthreads();
    float m[ATTN_MAX_GROUP], l[ATTN_MAX_GROUP], ap[ATTN_MAX_GROUP][8];
    fast_piece(qs, sc, xch, st, G, k, v, nkv, kh, d, dv, scale, j0, j1, m, l, ap);
    if (warp != 0)
        return;
#pragma unroll
    for (int g = 0; g < ATTN_MAX_GROUP; g++) {
        if (g >= G)
            break;
        float *o = part + ((t * nh + h0 + g) * pieces + p) * (dv + 2);
        if (lane == 0)
            o[0] = m[g], o[1] = l[g];
#pragma unroll
        for (int i = 0; i < 8; i++)
            if (8 * lane + i < dv)
                o[2 + 8 * lane + i] = ap[g][i];
    }
}

/* The launches, over keys and values as kvsrc. */
static ath_status attn_run(const ath_device *d, const ath_tensor *q, kvsrc k, kvsrc v, int64_t S, int64_t nkv,
                           int64_t dv, bool aligned, const ath_tensor *pos, float scale, float *part, ath_tensor *out)
{
    const int64_t T = q->shape[0], nh = q->shape[1], hd = q->shape[2];
    const int32_t *p = pos ? (const int32_t *)pos->data : NULL;
    if (T * nh == 0 || dv == 0)
        return ATH_OK;
    if (T * nh > INT32_MAX) {
        ath_log(ATH_LOG_ERROR, "attn: %lld queries × heads, more than one launch holds", (long long)(T * nh));
        return ATH_ERR_UNSUPPORTED;
    }
    ATH_CHECK(ath_gpu_use(d));
    const bool fast = hd <= FAST_DIM && dv <= FAST_DIM && hd % 8 == 0 && dv % 8 == 0 && aligned;
    if (fast && !part) {
        fast_rows<<<(unsigned)(T * nh), ATTN_THREADS, 0, k_stream(d)>>>(q->data, q->dtype, k, v, p, out->data,
                                                                       out->dtype, T, S, nh, nkv, hd, dv, scale);
        return ath_gpu_launched("attn fast_rows");
    }
    if (!part) {
        const size_t smem = (size_t)(hd + ATTN_TILE) * sizeof(float);
        attn_rows<<<(unsigned)(T * nh), ATTN_THREADS, smem, k_stream(d)>>>(q->data, q->dtype, k, v, p, out->data,
                                                                          out->dtype, T, S, nh, nkv, hd, dv, scale);
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
    if (fast)
        fast_split<<<grid, ATTN_THREADS, 0, k_stream(d)>>>(q->data, q->dtype, k, v, p, part, T, S, nh, nkv, hd, dv,
                                                           scale, G);
    else
        attn_split<<<grid, ATTN_THREADS, smem, k_stream(d)>>>(q->data, q->dtype, k, v, p, part, T, S, nh, nkv, hd,
                                                             dv, scale, G);
    ATH_CHECK(ath_gpu_launched("attn_split"));
    attn_merge<<<(unsigned)(T * nh), ATTN_THREADS, 0, k_stream(d)>>>(part, p, out->data, out->dtype, T, S, nh, dv,
                                                                    pieces);
    return ath_gpu_launched("attn_merge");
}

extern "C" ath_status ath_attn_launch(const ath_device *d, const ath_tensor *q, const ath_tensor *k,
                                      const ath_tensor *v, const ath_tensor *pos, float scale, float *part,
                                      ath_tensor *out)
{
    const kvsrc ks = {k->data, k->dtype, NULL, 0}, vs = {v->data, v->dtype, NULL, 0};
    const bool aligned = (uintptr_t)k->data % 16 == 0 && (uintptr_t)v->data % 16 == 0;
    return attn_run(d, q, ks, vs, k->shape[0], k->shape[1], v->shape[2], aligned, pos, scale, part, out);
}

extern "C" ath_status ath_attn_q8_launch(const ath_device *d, const ath_tensor *q, const ath_tensor *k,
                                         const ath_tensor *ks, const ath_tensor *v, const ath_tensor *vs,
                                         const ath_tensor *pos, float scale, float *part, ath_tensor *out)
{
    const kvsrc kq = {k->data, ATH_DTYPE_I8, (const float *)ks->data, q->dtype};
    const kvsrc vq = {v->data, ATH_DTYPE_I8, (const float *)vs->data, q->dtype};
    const bool aligned = (uintptr_t)k->data % 8 == 0 && (uintptr_t)v->data % 8 == 0;
    return attn_run(d, q, kq, vq, k->shape[0], k->shape[1], v->shape[2], aligned, pos, scale, part, out);
}
