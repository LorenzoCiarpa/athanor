/* attn_mma.cu: causal attention for prompts on the tensor cores (step 10).
 * See ops.h and attn.c; attn.cu has the kernels for decode and the rest.
 *
 * FlashAttention's forward pass (Dao, 2023: FlashAttention-2). A block takes
 * FA_ROWS (64) queries of one head, 16 per warp, and walks the keys their
 * positions reach in tiles of FA_KEYS (64). Per tile, with the tile's keys
 * K, values V and the warp's queries Q:
 *
 *   S = scale · Q Kᵀ           mma: bf16 products, fp32 sums
 *   S_ij = −inf where key j is past query i's position
 *   m' = max(m, max_j S_ij)    the running maximum of each row
 *   P = exp(S − m')            rounded to bf16, the tensor cores' input
 *   l = l e^(m − m') + Σ_j P_ij,  O = O e^(m − m') + P V
 *
 * and at the end out = O / l. The reference op computes the same sums in
 * fp32 in two passes (attn.c); here the scores and sums are fp32 as well,
 * and the probabilities are bf16, as in FlashAttention: l adds the rounded
 * values, so the weights O / l applies are exactly the ones multiplied, and
 * the output is a weighted mean of the values with weights within a
 * relative 2^-8 of the reference's (tests/README.md, Prompt kernels).
 *
 * The tiles of keys start at multiples of FA_KEYS, whatever the queries'
 * rows: a query that sees nothing of a tile (a later row's key) gets −inf
 * scores, a factor e^0 = 1 and probabilities 0, which leave its sums as they
 * were. So a query's output depends on its own position only, and a prompt
 * fed in pieces gives the bits of the prompt at once, as long as every piece
 * takes this kernel (ATH_ATTN_MMA_MIN_T).
 *
 * Memory: the 64 queries stay in shared memory for the whole walk; one tile
 * of keys and one of values, loaded with cp.async so that the values of a
 * tile load while the scores are computed and the next keys while the
 * values are summed. Rows are 512 bytes (256 bf16) and their 16-byte chunks
 * are permuted by the row (swizzle), so that the eight rows ldmatrix reads
 * at one column sit in different banks. 96 KiB of shared memory, one block
 * per SM; the blocks with the longest rows start first.
 */
#include "ops/cuda/mma.cuh"

#define FA_WARPS 4
#define FA_ROWS (16 * FA_WARPS)                         /* queries per block   */
#define FA_KEYS 64                                       /* keys per tile       */
#define FA_DIM 256                                       /* largest head dim    */
#define FA_ROW_BYTES (FA_DIM * 2)
#define FA_SMEM ((FA_ROWS + 2 * FA_KEYS) * FA_ROW_BYTES) /* 96 KiB              */
#if FA_ROWS != FA_KEYS
#error "fa_load loads tiles of FA_KEYS rows, the queries' tile too"
#endif

#ifdef ATH_HAVE_MMA

/* Byte offset of 16-byte chunk ch of row r in a tile: the chunk index
 * permuted within each group of 8 by the row. */
static __device__ __forceinline__ uint32_t fa_sw(int r, int ch)
{
    return (uint32_t)(r * FA_ROW_BYTES + ((ch ^ (r & 7)) << 4));
}

/* Rows r0 … r0 + 63 of a [rows, stride] bf16 matrix into a tile, d elements
 * each (d / 8 chunks); rows after last are zeros (never read from memory). */
static __device__ __forceinline__ void fa_load(unsigned char *tile, const uint16_t *src, int64_t r0, int64_t last,
                                               int64_t stride, int nch)
{
    for (int e = threadIdx.x; e < FA_KEYS * nch; e += FA_WARPS * ATH_WARP) {
        const int r = e / nch, ch = e % nch;
        const bool ok = r0 + r <= last;
        k_cp16(tile + fa_sw(r, ch), ok ? src + (r0 + r) * stride + ch * 8 : src, ok);
    }
}

static __global__ void __launch_bounds__(FA_WARPS *ATH_WARP, 1)
attn_mma(const uint16_t *q, const uint16_t *k, const uint16_t *v, const int32_t *pos, uint16_t *out, int64_t T,
         int64_t S, int64_t nh, int64_t nkv, int d, float scale)
{
    extern __shared__ __align__(128) unsigned char fa_smem[];
    unsigned char *qs = fa_smem, *ks = qs + FA_ROWS * FA_ROW_BYTES, *vs = ks + FA_KEYS * FA_ROW_BYTES;
    __shared__ int32_t prow[FA_ROWS];
    const int lane = threadIdx.x % ATH_WARP, warp = threadIdx.x / ATH_WARP;
    const int g = lane / 4, c = 2 * (lane % 4), nch = d / 8;
    const int64_t t0 = (int64_t)(gridDim.x - 1 - blockIdx.x) * FA_ROWS;   /* longest rows first */
    const int64_t h = blockIdx.y, kh = h / (nh / nkv);

    /* positions of the block's queries (−1: none, or outside the keys) and
     * the last key any of them sees */
    if (threadIdx.x < FA_ROWS) {
        const int64_t t = t0 + threadIdx.x;
        int64_t p = -1;
        if (t < T)
            p = pos ? (int64_t)pos[t] : S - T + t;
        prow[threadIdx.x] = p >= 0 && p < S ? (int32_t)p : -1;
    }
    __syncthreads();
    int32_t kmax = -1;
    for (int r = 0; r < FA_ROWS; r++)
        kmax = max(kmax, prow[r]);
    const int ra = warp * 16 + g;                       /* this lane's rows: ra, ra + 8 */
    const int32_t pa = prow[ra], pb = prow[ra + 8];

    fa_load(qs, q + h * d, t0, T - 1, nh * d, nch);
    fa_load(ks, k + kh * d, 0, kmax, nkv * d, nch);
    k_cp_commit();

    float o[FA_DIM / 8][4];
#pragma unroll
    for (int n = 0; n < FA_DIM / 8; n++)
        o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};

    for (int64_t j0 = 0; j0 <= kmax; j0 += FA_KEYS) {
        k_cp_wait_all();
        __syncthreads();                                /* keys in; the values' tile is free */
        fa_load(vs, v + kh * d, j0, kmax, nkv * d, nch);
        k_cp_commit();

        /* S = Q Kᵀ for the warp's 16 queries and the tile's 64 keys */
        float s[FA_KEYS / 8][4];
#pragma unroll
        for (int n = 0; n < FA_KEYS / 8; n++)
            s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
#pragma unroll 4
        for (int kk = 0; kk < FA_DIM / 16; kk++) {
            if (kk >= d / 16)
                break;
            uint32_t a[4];
            k_ldsm4(a, qs + fa_sw(warp * 16 + (lane & 15), 2 * kk + (lane >> 4)));
#pragma unroll
            for (int p = 0; p < FA_KEYS / 16; p++) {
                uint32_t b[4];
                k_ldsm4(b, ks + fa_sw(16 * p + ((lane >> 4) << 3) + (lane & 7), 2 * kk + ((lane >> 3) & 1)));
                k_mma_bf16(s[2 * p], a, b[0], b[1]);
                k_mma_bf16(s[2 * p + 1], a, b[2], b[3]);
            }
        }

        /* scale, mask, the rows' maxima over the quad that shares them */
        float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int n = 0; n < FA_KEYS / 8; n++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int64_t j = j0 + 8 * n + c + (e & 1);
                const float x = j <= (e < 2 ? pa : pb) ? __fmul_rn(s[n][e], scale) : -INFINITY;
                s[n][e] = x;
                mx[e >> 1] = fmaxf(mx[e >> 1], x);
            }
        float base[2], corr[2];
#pragma unroll
        for (int i = 0; i < 2; i++) {
            mx[i] = fmaxf(mx[i], ath_shfl_xor(mx[i], 1));
            mx[i] = fmaxf(mx[i], ath_shfl_xor(mx[i], 2));
            const float mn = fmaxf(m[i], mx[i]);
            base[i] = mn == -INFINITY ? 0.0f : mn;       /* a row that sees nothing yet: all exp give 0 */
            corr[i] = expf(__fsub_rn(m[i], base[i]));    /* 0 on its first keys, 1 if the maximum stays */
            m[i] = mn;
        }

        /* probabilities rounded to bf16, packed as the A fragments of P V */
        uint32_t pf[FA_KEYS / 8][2];
        float ls[2] = {0.0f, 0.0f};
#pragma unroll
        for (int n = 0; n < FA_KEYS / 8; n++)
#pragma unroll
            for (int i = 0; i < 2; i++) {
                const uint32_t w = k_pack_bf16(expf(__fsub_rn(s[n][2 * i], base[i])),
                                               expf(__fsub_rn(s[n][2 * i + 1], base[i])));
                pf[n][i] = w;
                ls[i] = __fadd_rn(ls[i], __fadd_rn(k_bf16_to_f32((uint16_t)w), k_bf16_to_f32((uint16_t)(w >> 16))));
            }
#pragma unroll
        for (int i = 0; i < 2; i++)
            l[i] = __fadd_rn(__fmul_rn(l[i], corr[i]), ls[i]);
#pragma unroll
        for (int n = 0; n < FA_DIM / 8; n++) {
            o[n][0] = __fmul_rn(o[n][0], corr[0]);
            o[n][1] = __fmul_rn(o[n][1], corr[0]);
            o[n][2] = __fmul_rn(o[n][2], corr[1]);
            o[n][3] = __fmul_rn(o[n][3], corr[1]);
        }

        k_cp_wait_all();
        __syncthreads();                                /* values in; the keys' tile is free */
        if (j0 + FA_KEYS <= kmax) {
            fa_load(ks, k + kh * d, j0 + FA_KEYS, kmax, nkv * d, nch);
            k_cp_commit();
        }

        /* O += P V */
#pragma unroll
        for (int s2 = 0; s2 < FA_KEYS / 16; s2++) {
            const uint32_t a[4] = {pf[2 * s2][0], pf[2 * s2][1], pf[2 * s2 + 1][0], pf[2 * s2 + 1][1]};
#pragma unroll
            for (int np = 0; np < FA_DIM / 16; np++) {
                if (np < d / 16) {
                    uint32_t b[4];
                    k_ldsm4_t(b, vs + fa_sw(16 * s2 + ((lane >> 3) & 1) * 8 + (lane & 7), 2 * np + (lane >> 4)));
                    k_mma_bf16(o[2 * np], a, b[0], b[1]);
                    k_mma_bf16(o[2 * np + 1], a, b[2], b[3]);
                }
            }
        }
    }

    /* l over the quad, then out = O / l (NaN for a query without a valid
     * position, as the other kernels) */
#pragma unroll
    for (int i = 0; i < 2; i++) {
        l[i] = __fadd_rn(l[i], ath_shfl_xor(l[i], 1));
        l[i] = __fadd_rn(l[i], ath_shfl_xor(l[i], 2));
    }
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const int64_t t = t0 + ra + 8 * i;
        if (t >= T)
            continue;
        const bool valid = (i ? pb : pa) >= 0;
        uint32_t *row = (uint32_t *)(out + (t * nh + h) * d);
#pragma unroll
        for (int n = 0; n < FA_DIM / 8; n++)
            if (8 * n < d)
                row[(8 * n + c) / 2] = valid ? k_pack_bf16(__fdiv_rn(o[n][2 * i], l[i]), __fdiv_rn(o[n][2 * i + 1], l[i]))
                                             : k_pack_bf16(NAN, NAN);
    }
}

#endif /* ATH_HAVE_MMA */

extern "C" bool ath_attn_mma_fits(const ath_device *d, const ath_tensor *q, const ath_tensor *k, const ath_tensor *v,
                                  const ath_tensor *out)
{
#ifdef ATH_HAVE_MMA
    const int64_t hd = q->shape[2];
    return d->cc_major >= 8 && d->smem_per_block >= FA_SMEM && q->dtype == ATH_DTYPE_BF16 &&
           k->dtype == ATH_DTYPE_BF16 && v->dtype == ATH_DTYPE_BF16 && out->dtype == ATH_DTYPE_BF16 &&
           v->shape[2] == hd && hd % 16 == 0 && hd <= FA_DIM && (uintptr_t)q->data % 16 == 0 &&
           (uintptr_t)k->data % 16 == 0 && (uintptr_t)v->data % 16 == 0 && (uintptr_t)out->data % 4 == 0;
#else
    (void)d, (void)q, (void)k, (void)v, (void)out;
    return false;
#endif
}

extern "C" ath_status ath_attn_mma_launch(const ath_device *d, const ath_tensor *q, const ath_tensor *k,
                                          const ath_tensor *v, const ath_tensor *pos, float scale, ath_tensor *out)
{
#ifdef ATH_HAVE_MMA
    const int64_t T = q->shape[0], nh = q->shape[1], hd = q->shape[2];
    const int64_t S = k->shape[0], nkv = k->shape[1];
    if (T * nh == 0)
        return ATH_OK;
    const int64_t blocks = (T + FA_ROWS - 1) / FA_ROWS;
    if (blocks > INT32_MAX || nh > 65535) {
        ath_log(ATH_LOG_ERROR, "attn: %lld queries of %lld heads, more than one launch holds", (long long)T,
                (long long)nh);
        return ATH_ERR_UNSUPPORTED;
    }
    ATH_CHECK(ath_gpu_use(d));
    if (cudaFuncSetAttribute(attn_mma, cudaFuncAttributeMaxDynamicSharedMemorySize, FA_SMEM) != cudaSuccess)
        return ath_gpu_launched("attn_mma (shared memory)");
    attn_mma<<<dim3((unsigned)blocks, (unsigned)nh), FA_WARPS * ATH_WARP, FA_SMEM, k_stream(d)>>>(
        (const uint16_t *)q->data, (const uint16_t *)k->data, (const uint16_t *)v->data,
        pos ? (const int32_t *)pos->data : NULL, (uint16_t *)out->data, T, S, nh, nkv, (int)hd, scale);
    return ath_gpu_launched("attn_mma");
#else
    (void)d, (void)q, (void)k, (void)v, (void)pos, (void)scale, (void)out;
    return ATH_ERR_UNSUPPORTED;
#endif
}
