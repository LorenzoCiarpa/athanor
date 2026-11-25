/* matmul_q8_0.cu: the matmul with Q8_0 weights on the GPU (step 12). See
 * ops.h (ath_matmul) and src/ops/q8_0.h for the canonical order, which every
 * kernel here follows: their results are ath_matmul_ref's, bit for bit, for
 * any number of rows and any launch.
 *
 * q8_gemv: a few rows of x (decode). A block of Q8_WARPS warps computes
 * 2 · Q8_WARPS output columns, one per half-warp (q8_0.cuh). Per group of 16
 * blocks the half-warp stages the weights in shared memory, each lane runs
 * the chains of its block against the same 32 activations of every row of
 * x, and the next group's weights are already loading. It needs K a
 * multiple of 512 (whole groups) and 16-byte aligned weights and x; q8_any
 * takes the rest (one lane reads its own block, 2 bytes at a time: slower,
 * the same arithmetic).
 *
 * q8_tile: a prompt (more rows than Q8_ROWS). A block computes a tile of
 * TM output columns × TN rows of x. Per group of 16 blocks it stages the
 * weights of its TM columns (as q8_gemv) and the TN rows' 512 activations
 * in shared memory, transposed so that element 32 j + i of the group sits at
 * xs[i][t][j] (a padded row per i: the stores of consecutive i fall in
 * different banks, the loads of consecutive j too). Thread (j, column group,
 * row group) runs the chains of block j for 4 columns × 4 rows at once: per
 * element i, 4 activations from shared memory and 4 values from registers
 * give 16 fused multiply-adds. The canonical order holds per output, so the
 * bits are q8_gemv's and the reference's.
 */
#include "ops/cuda/q8_0.cuh"

#define Q8_ROWS 8       /* the most rows of x one q8_gemv launch takes */
#define Q8_WARPS 4      /* warps per block: 8 output columns           */

template <int ROWS>
static __global__ void __launch_bounds__(Q8_WARPS * ATH_WARP)
q8_gemv(const void *x, int xdt, const uint8_t *w, void *out, int odt, int N, int64_t M, int64_t K)
{
    __shared__ __align__(16) unsigned char stage[2 * Q8_WARPS][Q8_REGION];
    const int j = threadIdx.x & 15, half = threadIdx.x >> 4;
    const int64_t m = (int64_t)blockIdx.x * (2 * Q8_WARPS) + half;
    const bool live = m < M;
    const int64_t nb = K / ATH_Q8_0_BLOCK, groups = nb / 16;
    const uint4 *row = (const uint4 *)(w + (live ? m : 0) * nb * ATH_Q8_0_BYTES);
    unsigned char *region = stage[half];

    float P[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; r++)
        P[r] = 0.0f;
    q8_pieces next = k_q8_fetch(row, j, live && groups > 0);
    for (int64_t g = 0; g < groups; g++) {
        __syncwarp();                         /* the previous group is read */
        k_q8_stage(region, j, next);
        __syncwarp();
        if (g + 1 < groups)
            next = k_q8_fetch(row + (g + 1) * (Q8_GROUP_BYTES / 16), j, live);
        uint32_t nw[9];
        k_q8_words(region, j, nw);
        const float d = k_q8_scale(nw);
        const int64_t b = g * 16 + j;
#pragma unroll
        for (int r = 0; r < ROWS; r++) {
            if (r < N) {
                float xf[32];
                k_q8_x(x, xdt, (int64_t)r * K, b, xf);
                P[r] = fmaf(d, k_q8_chain(nw, xf), P[r]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < ROWS; r++) {
        if (r < N) {
            const float v = k_q8_tree(P[r]);
            if (live && j == 0)
                k_st(out, odt, (int64_t)r * M + m, v);
        }
    }
}

/* Any K that is a multiple of 32 and 2-byte aligned weights: lane j reads
 * its own blocks, 2 bytes at a time; activations one at a time. */
static __global__ void q8_any(const void *x, int xdt, const uint8_t *w, void *out, int odt, int64_t N, int64_t M,
                              int64_t K)
{
    const int j = threadIdx.x & 15, half = (threadIdx.x >> 4) & 1;
    const int64_t warps = (int64_t)gridDim.x * blockDim.x / ATH_WARP;
    const int64_t nb = K / ATH_Q8_0_BLOCK;
    /* a warp takes two outputs per round, both halves together, so that the
     * shuffles of the tree see every lane; a half past the end computes
     * nothing */
    for (int64_t base = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / ATH_WARP * 2; base < N * M;
         base += 2 * warps) {
        const int64_t o0 = base + half;
        const bool live = o0 < N * M;
        const int64_t o = live ? o0 : 0, r = o / M, m = o % M;
        const uint16_t *row = (const uint16_t *)(w + m * nb * ATH_Q8_0_BYTES);
        float P = 0.0f;
        for (int64_t b = j; live && b < nb; b += 16) {
            uint32_t nw[9];
            const uint16_t *blk = row + b * (ATH_Q8_0_BYTES / 2);
#pragma unroll
            for (int i = 0; i < 8; i++)
                nw[i] = (uint32_t)blk[2 * i] | (uint32_t)blk[2 * i + 1] << 16;
            nw[8] = blk[16];
            float xf[32];
#pragma unroll
            for (int i = 0; i < 32; i++)
                xf[i] = k_ld(x, xdt, r * K + b * ATH_Q8_0_BLOCK + i);
            P = fmaf(k_q8_scale(nw), k_q8_chain(nw, xf), P);
        }
        const float v = k_q8_tree(P);
        if (live && j == 0)
            k_st(out, odt, o, v);
    }
}

/* ---- prompts -------------------------------------------------------------------- */

#define TM 32                          /* output columns per block               */
#define TN 16                          /* rows of x per block                    */
#define TILE_THREADS (16 * (TM / 4) * (TN / 4))
#define XS_STRIDE (TN * 16 + 1)        /* floats per element i of a group        */
#define TILE_SMEM (TM * Q8_REGION + 32 * XS_STRIDE * 4)

static __global__ void __launch_bounds__(TILE_THREADS, 1)
q8_tile(const void *x, int xdt, const uint8_t *w, void *out, int odt, int64_t N, int64_t M, int64_t K)
{
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char *ws = smem;                                  /* TM regions of Q8_REGION */
    float *xs = (float *)(smem + TM * Q8_REGION);              /* [32][XS_STRIDE]         */
    const int tid = threadIdx.x, j = tid & 15;
    const int cg = (tid >> 4) % (TM / 4), rg = (tid >> 4) / (TM / 4);
    const int64_t m0 = (int64_t)blockIdx.x * TM, t0 = (int64_t)blockIdx.y * TN;
    const int64_t nb = K / ATH_Q8_0_BLOCK, groups = nb / 16;
    const size_t rowb = (size_t)nb * ATH_Q8_0_BYTES;

    float P[4][4];
#pragma unroll
    for (int c = 0; c < 4; c++)
#pragma unroll
        for (int r = 0; r < 4; r++)
            P[c][r] = 0.0f;

    for (int64_t g = 0; g < groups; g++) {
        __syncthreads();                                       /* the previous group is read */
        /* weights: TM columns × 34 pieces of 16 bytes */
        for (int p = tid; p < TM * 34; p += TILE_THREADS) {
            const int c = p / 34, piece = p % 34;
            const int64_t m = m0 + c;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (m < M)
                v = k_q8_ld((const uint4 *)(w + (size_t)m * rowb + (size_t)g * Q8_GROUP_BYTES) + piece);
            ((uint4 *)(ws + c * Q8_REGION))[piece] = v;
        }
        /* activations: TN rows × 512, 8 at a time, transposed */
        for (int p = tid; p < TN * 64; p += TILE_THREADS) {
            const int t = p / 64, e8 = p % 64;
            float f[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            if (t0 + t < N)
                k_ld8(x, xdt, ((t0 + t) * K + g * 512) / 8 + e8, f);
#pragma unroll
            for (int u = 0; u < 8; u++) {
                const int k = e8 * 8 + u;                      /* 32 jj + i */
                xs[(k & 31) * XS_STRIDE + t * 16 + (k >> 5)] = f[u];
            }
        }
        __syncthreads();

        uint32_t nw[4][9];
#pragma unroll
        for (int c = 0; c < 4; c++)
            k_q8_words(ws + (cg * 4 + c) * Q8_REGION, j, nw[c]);
        float s[4][4];
#pragma unroll
        for (int i = 0; i < ATH_Q8_0_BLOCK; i++) {
            float xv[4];
#pragma unroll
            for (int r = 0; r < 4; r++)
                xv[r] = xs[i * XS_STRIDE + (rg * 4 + r) * 16 + j];
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const float q = k_q8_val(nw[c], i);
#pragma unroll
                for (int r = 0; r < 4; r++)
                    s[c][r] = i == 0 ? q * xv[r] : fmaf(q, xv[r], s[c][r]);
            }
        }
#pragma unroll
        for (int c = 0; c < 4; c++) {
            const float d = k_q8_scale(nw[c]);
#pragma unroll
            for (int r = 0; r < 4; r++)
                P[c][r] = fmaf(d, s[c][r], P[c][r]);
        }
    }
#pragma unroll
    for (int c = 0; c < 4; c++)
#pragma unroll
        for (int r = 0; r < 4; r++) {
            const float v = k_q8_tree(P[c][r]);
            const int64_t m = m0 + cg * 4 + c, t = t0 + rg * 4 + r;
            if (j == 0 && m < M && t < N)
                k_st(out, odt, t * M + m, v);
        }
}

static bool gemv_fits(const ath_tensor *x, const ath_tensor *w, int64_t K)
{
    return K % 512 == 0 && (uintptr_t)w->data % 16 == 0 && (uintptr_t)x->data % 16 == 0;
}

extern "C" ath_status ath_matmul_q8_0_launch(const ath_device *d, const ath_tensor *x, const ath_tensor *w,
                                             ath_tensor *out)
{
    const int64_t K = x->shape[x->ndim - 1], M = w->shape[0];
    const int64_t N = K > 0 ? ath_tensor_numel(x) / K : ath_tensor_numel(out) / (M > 0 ? M : 1);
    if (N * M == 0)
        return ATH_OK;
    if (K == 0)
        return ath_memset(d, out->data, 0, ath_tensor_nbytes(out), ATH_STREAM_COMPUTE);
    ATH_CHECK(ath_gpu_use(d));
    const uint8_t *wd = (const uint8_t *)w->data;
    if (!gemv_fits(x, w, K)) {
        q8_any<<<k_blocks(N * M * 16, 256), 256, 0, k_stream(d)>>>(x->data, x->dtype, wd, out->data, out->dtype, N, M,
                                                                    K);
        return ath_gpu_launched("q8_any");
    }
    if (N > Q8_ROWS) {
        if (cudaFuncSetAttribute(q8_tile, cudaFuncAttributeMaxDynamicSharedMemorySize, TILE_SMEM) != cudaSuccess)
            return ath_gpu_launched("q8_tile (shared memory)");
        const dim3 grid((unsigned)((M + TM - 1) / TM), (unsigned)((N + TN - 1) / TN));
        q8_tile<<<grid, TILE_THREADS, TILE_SMEM, k_stream(d)>>>(x->data, x->dtype, wd, out->data, out->dtype, N, M, K);
        return ath_gpu_launched("q8_tile");
    }
    const unsigned blocks = (unsigned)((M + 2 * Q8_WARPS - 1) / (2 * Q8_WARPS));
    const size_t xes = ath_dtype_size(x->dtype), oes = ath_dtype_size(out->dtype);
    for (int64_t r0 = 0; r0 < N; r0 += Q8_ROWS) {
        const int n = (int)(N - r0 < Q8_ROWS ? N - r0 : Q8_ROWS);
        const void *xr = (const char *)x->data + (size_t)(r0 * K) * xes;
        void *orow = (char *)out->data + (size_t)(r0 * M) * oes;
        if (n == 1)
            q8_gemv<1><<<blocks, Q8_WARPS * ATH_WARP, 0, k_stream(d)>>>(xr, x->dtype, wd, orow, out->dtype, 1, M, K);
        else
            q8_gemv<Q8_ROWS><<<blocks, Q8_WARPS * ATH_WARP, 0, k_stream(d)>>>(xr, x->dtype, wd, orow, out->dtype, n, M,
                                                                            K);
        ATH_CHECK(ath_gpu_launched("q8_gemv"));
    }
    return ATH_OK;
}
