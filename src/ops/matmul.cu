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
#include "ops/kernel.cuh"

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
    const bool bf16_in = x->dtype == ATH_DTYPE_BF16 && w->dtype == ATH_DTYPE_BF16;
    const bool f32_all = x->dtype == ATH_DTYPE_F32 && w->dtype == ATH_DTYPE_F32 && out->dtype == ATH_DTYPE_F32;
    if (bf16_in || f32_all)
        return blas_matmul(d, x, w, out, N, M, K);
    matmul_kernel<<<k_blocks(N * M * ATH_WARP, 256), 256, 0, k_stream(d)>>>(x->data, x->dtype, w->data, w->dtype,
                                                                            out->data, out->dtype, N, M, K);
    return ath_gpu_launched("matmul_kernel");
}
