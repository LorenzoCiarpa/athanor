/* matmul.cu: out = x · wᵀ on the GPU. See ops.h and matmul.c.
 *
 * Dense matrix products go to the vendor's library, cuBLASLt (hipBLASLt on
 * AMD), as docs/PLAN.md decides: athanor writes its own kernels for GEMV,
 * quantized formats and fused ops, later (steps 9 and 10, phase 2), where a
 * benchmark shows a gain.
 *
 * Layouts. cuBLASLt counts in column-major order. A row-major matrix is the
 * column-major view of its transpose, so out [N, M] row-major is outᵀ
 * [M, N] column-major, and
 *
 *   outᵀ = w · xᵀ        A = w: column-major [K, M], transposed by the op
 *                        B = x: column-major [K, N], as it is
 *                        D = out: column-major [M, N], leading dimension M
 *
 * the "TN" product, which tensor cores read without any copy. Accumulation
 * is fp32 (CUBLAS_COMPUTE_32F; never the TF32 mode, which would round f32
 * inputs to 10 mantissa bits).
 *
 * cuBLASLt takes bf16 inputs with a bf16 or f32 output, or f32 throughout.
 * The other combinations of ath_matmul (x and w of different dtypes, or f32
 * inputs into a bf16 output) run on matmul_kernel below: one warp per
 * output, slower, for the exact (f32-activation) path that tests use.
 */
#include "ops/cuda/kernel.cuh"

#include <string.h>

/* ---- cuBLASLt --------------------------------------------------------------- */

/* The alignment cuBLASLt may assume for a matrix: the largest power of two,
 * up to 256, that divides its address and its row pitch in bytes. The
 * library's default assumes 256 and would choose kernels that need it. */
static uint32_t alignment(const void *p, int64_t pitch_bytes)
{
    uint64_t a = (uint64_t)(uintptr_t)p | (uint64_t)pitch_bytes | 256u;
    return (uint32_t)(a & (~a + 1));
}

static cudaDataType_t blas_type(ath_dtype dt) { return dt == ATH_DTYPE_BF16 ? CUDA_R_16BF : CUDA_R_32F; }

static ath_status blas_fail(cublasStatus_t s, const char *what)
{
    ath_log(ATH_LOG_ERROR, "matmul: %s: %s", what, cublasLtGetStatusString(s));
    return ATH_ERR_DEVICE;
}

static ath_status blas_matmul(const ath_device *d, const ath_tensor *x, const ath_tensor *w, ath_tensor *out,
                              int64_t N, int64_t M, int64_t K)
{
    cublasLtHandle_t h = (cublasLtHandle_t)d->blas;
    const cudaDataType_t in = blas_type(x->dtype), ot = blas_type(out->dtype);
    const int64_t ies = (int64_t)ath_dtype_size(x->dtype), oes = (int64_t)ath_dtype_size(out->dtype);
    cublasLtMatmulDesc_t op = NULL;
    cublasLtMatrixLayout_t la = NULL, lb = NULL, ld = NULL;
    cublasLtMatmulPreference_t pref = NULL;
    cublasLtMatmulHeuristicResult_t best;
    const char *what = "cublasLtMatmulDescCreate";
    int found = 0;

    cublasStatus_t s = cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (s == CUBLAS_STATUS_SUCCESS) {
        const cublasOperation_t t = CUBLAS_OP_T, n = CUBLAS_OP_N;
        what = "cublasLtMatmulDescSetAttribute";
        s = cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof t);
        if (s == CUBLAS_STATUS_SUCCESS)
            s = cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &n, sizeof n);
    }
    if (s == CUBLAS_STATUS_SUCCESS) {
        what = "cublasLtMatrixLayoutCreate";
        s = cublasLtMatrixLayoutCreate(&la, in, (uint64_t)K, (uint64_t)M, K);
        if (s == CUBLAS_STATUS_SUCCESS)
            s = cublasLtMatrixLayoutCreate(&lb, in, (uint64_t)K, (uint64_t)N, K);
        if (s == CUBLAS_STATUS_SUCCESS)
            s = cublasLtMatrixLayoutCreate(&ld, ot, (uint64_t)M, (uint64_t)N, M);
    }
    if (s == CUBLAS_STATUS_SUCCESS) {
        what = "cublasLtMatmulPreference";
        s = cublasLtMatmulPreferenceCreate(&pref);
        const uint64_t ws = d->blas_ws_bytes;
        if (s == CUBLAS_STATUS_SUCCESS)
            s = cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof ws);
#if !defined(__HIP_PLATFORM_AMD__)
        const uint32_t aa = alignment(w->data, K * ies), ab = alignment(x->data, K * ies);
        const uint32_t ad = alignment(out->data, M * oes);
        if (s == CUBLAS_STATUS_SUCCESS)
            s = cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &aa, sizeof aa);
        if (s == CUBLAS_STATUS_SUCCESS)
            s = cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &ab, sizeof ab);
        if (s == CUBLAS_STATUS_SUCCESS)
            s = cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &ad, sizeof ad);
        if (s == CUBLAS_STATUS_SUCCESS)
            s = cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &ad, sizeof ad);
#else
        (void)ies, (void)oes;
#endif
    }
    if (s == CUBLAS_STATUS_SUCCESS) {
        what = "cublasLtMatmulAlgoGetHeuristic";
        s = cublasLtMatmulAlgoGetHeuristic(h, op, la, lb, ld, ld, pref, 1, &best, &found);
    }
    if (s == CUBLAS_STATUS_SUCCESS && found > 0) {
        const float alpha = 1.0f, beta = 0.0f;
        what = "cublasLtMatmul";
        s = cublasLtMatmul(h, op, &alpha, w->data, la, x->data, lb, &beta, out->data, ld, out->data, ld, &best.algo,
                           d->blas_ws, d->blas_ws_bytes, k_stream(d));
    }
    if (pref)
        cublasLtMatmulPreferenceDestroy(pref);
    if (ld)
        cublasLtMatrixLayoutDestroy(ld);
    if (lb)
        cublasLtMatrixLayoutDestroy(lb);
    if (la)
        cublasLtMatrixLayoutDestroy(la);
    if (op)
        cublasLtMatmulDescDestroy(op);
    if (s != CUBLAS_STATUS_SUCCESS)
        return blas_fail(s, what);
    if (found == 0) {
        ath_log(ATH_LOG_ERROR, "matmul: cuBLASLt has no algorithm for [%lld, %lld] x [%lld, %lld]ᵀ (%s, %s)",
                (long long)N, (long long)K, (long long)M, (long long)K, ath_dtype_name(x->dtype),
                ath_dtype_name(out->dtype));
        return ATH_ERR_UNSUPPORTED;
    }
    return ATH_OK;
}

/* ---- athanor's kernel, for the other dtype combinations ------------------------ */

/* One warp per output: lane l adds the terms k = l, l + 32, ..., then the
 * warp adds its 32 partial sums. */
static __global__ void matmul_kernel(const void *x, int xdt, const void *w, int wdt, void *out, int odt, int64_t N,
                                     int64_t M, int64_t K)
{
    const int lane = threadIdx.x % ATH_WARP;
    const int64_t warps = (int64_t)gridDim.x * blockDim.x / ATH_WARP;
    for (int64_t o = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / ATH_WARP; o < N * M; o += warps) {
        const int64_t r = o / M, m = o % M;
        float acc = 0.0f;
        for (int64_t k = lane; k < K; k += ATH_WARP)
            acc += k_ld(x, xdt, r * K + k) * k_ld(w, wdt, m * K + k);
        acc = k_warp_sum(acc);
        if (lane == 0)
            k_st(out, odt, o, acc);
    }
}

/* ---- GEMV: a few rows of x against bf16 weights (step 9) ----------------------- */

/* Decode multiplies one row of activations (a few, for a few sequences) by
 * every weight matrix: each weight is read once per token and used for one
 * multiply-add per row, so the time is the time to read the weights. A
 * library GEMM tiles for reuse that is not there; this kernel only streams.
 *
 * One warp per output column m (row m of w), or S = 2 or 4 warps for a
 * long row, each a contiguous share of it (gemv_split): the lanes read the
 * row in 16-byte pieces (8 bf16), lane l the pieces l, l + 32, ..., four at
 * a time so that several loads are in flight, and multiply each piece with
 * the same 8 elements of every row of x (x is small and stays in the
 * caches). Each lane sums its products in fp32 in order of k, the warp adds
 * the 32 sums (k_warp_sum), and the S warps' sums are added in order: the
 * order is fixed by K alone, so the result does not depend on M, N or the
 * launch, and a row gives the same bits whether it comes alone or with
 * others. Several weights that share x (ath_matmul_n) are one launch. */
#define GEMV_ROWS 8      /* the most rows of x one launch takes        */
#define GEMV_WARPS 4     /* warps per block, one output column each    */
#define GEMV_UNROLL 4    /* 16-byte pieces each lane loads at a time   */

/* The weights of one launch, which share x: matrix j has M[j] rows (output
 * columns) and takes the warps first[j] … first[j + 1] − 1. */
typedef struct {
    const uint4 *w[ATH_MATMUL_N_MAX];
    void        *out[ATH_MATMUL_N_MAX];
    int          odt[ATH_MATMUL_N_MAX];
    int64_t      M[ATH_MATMUL_N_MAX];
    int64_t      first[ATH_MATMUL_N_MAX + 1];
    int          n;
    bool         add;   /* the product added to out, as ath_add does (step 9) */
} gemv_set;

/* The warps that share one output column: long rows are split so that every
 * lane has about four pieces to load, whatever K (step 9). It depends on K
 * only, so a column's bits do not depend on the other columns. */
static inline int gemv_split(int64_t K) { return K >= 3072 ? 4 : K >= 1536 ? 2 : 1; }

/* Output element e of matrix j: the sum rounded to the output's dtype, and
 * with set.add, then added to what is there and rounded again (ath_add). */
static __device__ __forceinline__ void gemv_store(const gemv_set *set, int j, int64_t e, float v)
{
    if (set->add)
        v = k_ld(set->out[j], set->odt[j], e) + k_round(set->odt[j], v);
    k_st(set->out[j], set->odt[j], e, v);
}

/* The kernel body for at most `rows` rows of x, a constant in each kernel
 * below, so that the decode kernel (one row) keeps one accumulator and its
 * registers stay few: the same arithmetic per row in both. */
static __device__ __forceinline__ void gemv_body(const void *x, int xdt, const gemv_set &set, int N, int64_t K, int S,
                                                 const int rows)
{
    __shared__ float part[GEMV_WARPS][GEMV_ROWS];
    const int lane = threadIdx.x % ATH_WARP, warp = threadIdx.x / ATH_WARP;
    const int64_t gm = (int64_t)blockIdx.x * (GEMV_WARPS / S) + warp / S;
    const int share = warp % S;                       /* this warp's part of the row */
    const bool live = gm < set.first[set.n];
    int j = 0;
    while (live && j + 1 < set.n && gm >= set.first[j + 1])
        j++;
    const int64_t m = live ? gm - set.first[j] : 0, M = set.M[j];
    const int64_t pieces = K / 8, per = (pieces + S - 1) / S;
    const int64_t c_begin = share * per, c_end = c_begin + per < pieces ? c_begin + per : pieces;
    const uint4 *row = set.w[j] + m * pieces;
    float acc[GEMV_ROWS];
#pragma unroll
    for (int r = 0; r < rows; r++)
        acc[r] = 0.0f;
    /* the first round of weights */
    const int64_t step = (int64_t)ATH_WARP * GEMV_UNROLL;
    uint4 wv[GEMV_UNROLL];
#pragma unroll
    for (int u = 0; u < GEMV_UNROLL; u++) {
        const int64_t c = c_begin + lane + (int64_t)u * ATH_WARP;
        wv[u] = live && c < c_end ? row[c] : make_uint4(0, 0, 0, 0);
    }
    for (int64_t c0 = c_begin + lane; live && c0 < c_end; c0 += step) {
        if (c0 != c_begin + lane)
#pragma unroll
            for (int u = 0; u < GEMV_UNROLL; u++) {
                const int64_t c = c0 + (int64_t)u * ATH_WARP;
                wv[u] = c < c_end ? row[c] : make_uint4(0, 0, 0, 0);
            }
#pragma unroll
        for (int u = 0; u < GEMV_UNROLL; u++) {
            const int64_t c = c0 + (int64_t)u * ATH_WARP;
            if (c >= c_end)
                break;
            float wf[8];
            k_unpack_bf16x8(wv[u], wf);
#pragma unroll
            for (int r = 0; r < rows; r++) {
                if (r < N) {
                    float xf[8];
                    k_ld8(x, xdt, (r * K) / 8 + c, xf);
#pragma unroll
                    for (int i = 0; i < 8; i++)
                        acc[r] += wf[i] * xf[i];
                }
            }
        }
    }
    /* each warp's sum, then the parts of a column in order */
#pragma unroll
    for (int r = 0; r < rows; r++) {
        if (r < N) {
            const float v = k_warp_sum(acc[r]);
            if (S == 1) {
                if (live && lane == 0)
                    gemv_store(&set, j, r * M + m, v);
            } else if (lane == 0) {
                part[warp][r] = v;
            }
        }
    }
    if (S == 1)
        return;
    __syncthreads();
    if (!live || share != 0 || lane >= N)
        return;
    float v = 0.0f;
    for (int p = 0; p < S; p++)
        v += part[warp + p][lane];
    gemv_store(&set, j, (int64_t)lane * M + m, v);
}

static __global__ void __launch_bounds__(GEMV_WARPS * ATH_WARP)
gemv_one(const void *x, int xdt, gemv_set set, int N, int64_t K, int S)
{
    gemv_body(x, xdt, set, N, K, S, 1);
}

static __global__ void __launch_bounds__(GEMV_WARPS * ATH_WARP)
gemv_few(const void *x, int xdt, gemv_set set, int N, int64_t K, int S)
{
    gemv_body(x, xdt, set, N, K, S, GEMV_ROWS);
}

/* The GEMV kernel takes the product: few rows, bf16 weights, rows of 16-byte
 * pieces on 16-byte boundaries. */
static bool gemv_fits(const ath_tensor *x, const ath_tensor *w, int64_t N, int64_t K)
{
    return N <= GEMV_ROWS && w->dtype == ATH_DTYPE_BF16 && K % 8 == 0 &&
           (x->dtype == ATH_DTYPE_BF16 || x->dtype == ATH_DTYPE_F32) && (uintptr_t)w->data % 16 == 0 &&
           (uintptr_t)x->data % 16 == 0;
}

static ath_status gemv_launch(const ath_device *d, const ath_tensor *x, const gemv_set *set, int64_t N, int64_t K)
{
    const int S = gemv_split(K), per_block = GEMV_WARPS / S;
    const int64_t blocks = (set->first[set->n] + per_block - 1) / per_block;
    if (N == 1)
        gemv_one<<<(unsigned)blocks, GEMV_WARPS * ATH_WARP, 0, k_stream(d)>>>(x->data, x->dtype, *set, 1, K, S);
    else
        gemv_few<<<(unsigned)blocks, GEMV_WARPS * ATH_WARP, 0, k_stream(d)>>>(x->data, x->dtype, *set, (int)N, K, S);
    return ath_gpu_launched("gemv");
}

static void gemv_add(gemv_set *set, const ath_tensor *w, ath_tensor *out)
{
    const int j = set->n++;
    set->w[j] = (const uint4 *)w->data;
    set->out[j] = out->data;
    set->odt[j] = out->dtype;
    set->M[j] = w->shape[0];
    set->first[j + 1] = set->first[j] + w->shape[0];
}

extern "C" ath_status ath_matmul_n_launch(const ath_device *d, const ath_tensor *x, int n, const ath_tensor *const *w,
                                          ath_tensor *const *out)
{
    const int64_t K = x->shape[x->ndim - 1];
    const int64_t N = K > 0 ? ath_tensor_numel(x) / K : 0;
    bool fit = N > 0;
    for (int i = 0; i < n; i++)
        fit = fit && w[i]->shape[0] > 0 && gemv_fits(x, w[i], N, K);
    if (!fit) {
        for (int i = 0; i < n; i++)
            ATH_CHECK(ath_matmul_launch(d, x, w[i], out[i]));
        return ATH_OK;
    }
    ATH_CHECK(ath_gpu_use(d));
    gemv_set set;
    memset(&set, 0, sizeof set);
    for (int i = 0; i < n; i++)
        gemv_add(&set, w[i], out[i]);
    return gemv_launch(d, x, &set, N, K);
}

extern "C" ath_status ath_matmul_launch(const ath_device *d, const ath_tensor *x, const ath_tensor *w,
                                        ath_tensor *out)
{
    const int64_t K = x->shape[x->ndim - 1], M = w->shape[0];
    const int64_t N = K > 0 ? ath_tensor_numel(x) / K : ath_tensor_numel(out) / (M > 0 ? M : 1);
    if (N * M == 0)
        return ATH_OK;
    if (K == 0)                                    /* empty sums: zeros, in either dtype */
        return ath_memset(d, out->data, 0, ath_tensor_nbytes(out), ATH_STREAM_COMPUTE);
    ATH_CHECK(ath_gpu_use(d));
    if (gemv_fits(x, w, N, K)) {
        gemv_set set;
        memset(&set, 0, sizeof set);
        gemv_add(&set, w, out);
        return gemv_launch(d, x, &set, N, K);
    }
    const bool bf16_in = x->dtype == ATH_DTYPE_BF16 && w->dtype == ATH_DTYPE_BF16;
    const bool f32_all = x->dtype == ATH_DTYPE_F32 && w->dtype == ATH_DTYPE_F32 && out->dtype == ATH_DTYPE_F32;
    if (bf16_in || f32_all)
        return blas_matmul(d, x, w, out, N, M, K);
    matmul_kernel<<<k_blocks(N * M * ATH_WARP, 256), 256, 0, k_stream(d)>>>(x->data, x->dtype, w->data, w->dtype,
                                                                            out->data, out->dtype, N, M, K);
    return ath_gpu_launched("matmul_kernel");
}

/* ---- fusions (step 9) ---------------------------------------------------------- */

extern "C" bool ath_gemv_fits(const ath_tensor *x, const ath_tensor *w)
{
    const int64_t K = x->shape[x->ndim - 1];
    return K > 0 && gemv_fits(x, w, ath_tensor_numel(x) / K, K);
}

extern "C" ath_status ath_matmul_add_launch(const ath_device *d, const ath_tensor *x, const ath_tensor *w,
                                            ath_tensor *out)
{
    const int64_t K = x->shape[x->ndim - 1], N = ath_tensor_numel(x) / K;
    ATH_CHECK(ath_gpu_use(d));
    gemv_set set;
    memset(&set, 0, sizeof set);
    gemv_add(&set, w, out);
    set.add = true;
    return gemv_launch(d, x, &set, N, K);
}
