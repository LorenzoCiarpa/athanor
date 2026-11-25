/* matmul.c: dense matrix product against a weight in checkpoint layout. See ops.h.
 *
 * The weight is [M, K], one output feature per row, as nn.Linear stores it:
 * out = x · wᵀ. Each output element is one dot product accumulated in fp32 in
 * order of k. For bf16 inputs every product is exact in fp32 (8 + 8
 * significant bits), so the only rounding is in the running sum.
 *
 * Two versions compute the same sums with the same bits: the reference, one
 * output at a time, and ath_matmul_fast below. The GPU version is in
 * cuda/matmul.cu.
 */
#include "ops/check.h"
#include "ops/launch.h"
#include "ops/ops.h"
#include "ops/q8_0.h"

static const char op[] = "matmul";

/* Arguments of every version: N rows of x, M outputs, K terms per sum. */
static ath_status check(const ath_tensor *x, const ath_tensor *w, const ath_tensor *out, ops_where where,
                        int64_t *N, int64_t *M, int64_t *K)
{
    ATH_CHECK(ops_float(x, op, "x"));
    ATH_CHECK(w && w->dtype == ATH_DTYPE_Q8_0 ? ops_tensor(w, ATH_DTYPE_Q8_0, op, "w") : ops_float(w, op, "w"));
    ATH_CHECK(ops_float(out, op, "out"));
    if (x->ndim < 1 || w->ndim != 2 || out->ndim != x->ndim)
        return ops_fail(ATH_ERR_INVALID_ARG, op, "shapes must be x [..., K], w [M, K], out [..., M]");
    *K = ops_last(x);
    *M = w->shape[0];
    *N = ops_rows(x);
    if (w->shape[1] != *K || ops_last(out) != *M)
        return ops_fail(ATH_ERR_INVALID_ARG, op, "x [..., %lld], w [%lld, %lld], out [..., %lld]",
                        (long long)*K, (long long)w->shape[0], (long long)w->shape[1],
                        (long long)ops_last(out));
    for (int32_t i = 0; i + 1 < x->ndim; i++)
        if (out->shape[i] != x->shape[i])
            return ops_fail(ATH_ERR_INVALID_ARG, op, "out must have the leading dimensions of x");
    return OPS_PLACE(op, where, x, w, out);
}

/* One output against a Q8_0 row, in the canonical order (q8_0.h). */
static float dot_q8_0_ref(const ath_tensor *x, int64_t r, const uint8_t *row, int64_t K)
{
    float P[ATH_Q8_0_LANES] = {0};
    for (int64_t b = 0; b < K / ATH_Q8_0_BLOCK; b++) {
        const uint8_t *blk = row + b * ATH_Q8_0_BYTES;
        const int8_t *q = (const int8_t *)(blk + 2);
        float s = (float)q[0] * ld(x, r * K + b * ATH_Q8_0_BLOCK);
        for (int i = 1; i < ATH_Q8_0_BLOCK; i++)
            s = fmaf((float)q[i], ld(x, r * K + b * ATH_Q8_0_BLOCK + i), s);
        P[b % ATH_Q8_0_LANES] = fmaf(ath_q8_0_scale(blk), s, P[b % ATH_Q8_0_LANES]);
    }
    return ath_q8_0_tree(P);
}

ath_status ath_matmul_ref(const ath_tensor *x, const ath_tensor *w, ath_tensor *out)
{
    int64_t N, M, K;
    ATH_CHECK(check(x, w, out, OPS_HOST, &N, &M, &K));

    if (w->dtype == ATH_DTYPE_Q8_0) {
        const size_t row = ath_dtype_bytes(ATH_DTYPE_Q8_0, K);
        for (int64_t r = 0; r < N; r++)
            for (int64_t m = 0; m < M; m++)
                st(out, r * M + m, dot_q8_0_ref(x, r, (const uint8_t *)w->data + (size_t)m * row, K));
        return ATH_OK;
    }
    for (int64_t r = 0; r < N; r++)
        for (int64_t m = 0; m < M; m++) {
            float acc = 0.0f;
            for (int64_t k = 0; k < K; k++)
                acc += ld(x, r * K + k) * ld(w, m * K + k);
            st(out, r * M + m, acc);
        }
    return ATH_OK;
}

/* ---- ath_matmul_fast ---------------------------------------------------------
 *
 * The reference is slow because it waits: each output is one chain of fp32
 * additions, and every addition needs the result of the previous one, so a
 * core finishes one addition every few cycles while it could start several
 * per cycle. Here a thread computes CHAINS outputs together. Each is still
 * its own chain in the order of k, with a rounded product and a rounded sum
 * per step (no fused multiply-add: GCC does not contract a·b + c in ISO C
 * mode, -std=c11), so the chains overlap in the pipeline and every output
 * keeps the reference's bits. Blocks of CHAINS outputs are spread over the
 * threads of the pool; each output is computed by one thread, in one order,
 * so the number of threads changes the time and nothing else.
 *
 * Every row of x passes over one block before the next block starts, so the
 * block's weights (CHAINS rows of K) stay in cache across the tokens of a
 * prompt. The weights keep the checkpoint layout: no copy, no repacking.
 * tests/test_ops.c and tests/test_golden.c compare the two versions bit for
 * bit, also with f32 inputs, where a contracted a·b + c would differ.
 */
enum { CHAINS = 8 };

/* Multiply-adds below which waking the pool costs more than it saves. */
#define MIN_WORK_PER_RUN 262144.0

typedef struct {
    const ath_tensor *x, *w, *out;
    int64_t           N, M, K;
    int64_t           blocks;     /* ceil(M / CHAINS) */
} job;

/* Outputs m .. m + CHAINS − 1 of every row of x, for weights in bf16 (the
 * checkpoint's dtype). The inner loop has no branch, so the compiler unrolls
 * it and keeps the CHAINS accumulators in registers; a generic element load
 * (ld, which branches on the dtype) would keep them in memory. */
static void block_bf16(const job *j, int64_t m)
{
    const int64_t K = j->K;
    const uint16_t *w = (const uint16_t *)j->w->data + m * K;
    for (int64_t r = 0; r < j->N; r++) {
        float acc[CHAINS] = {0};
        for (int64_t k = 0; k < K; k++) {
            const float xv = ld(j->x, r * K + k);
            for (int i = 0; i < CHAINS; i++)
                acc[i] += xv * ath_bf16_to_f32(w[i * K + k]);
        }
        for (int i = 0; i < CHAINS; i++)
            st(j->out, r * j->M + m + i, acc[i]);
    }
}

/* Outputs m0 .. m1 − 1 one at a time, as the reference: the last block when
 * M is not a multiple of CHAINS, and weights in f32 (no checkpoint stores a
 * matmul weight in f32; the tests do). */
static void one_at_a_time(const job *j, int64_t m0, int64_t m1)
{
    const int64_t K = j->K;
    for (int64_t r = 0; r < j->N; r++)
        for (int64_t m = m0; m < m1; m++) {
            float acc = 0.0f;
            for (int64_t k = 0; k < K; k++)
                acc += ld(j->x, r * K + k) * ld(j->w, m * K + k);
            st(j->out, r * j->M + m, acc);
        }
}

/* Outputs m0 .. m1 − 1 of every row of x against Q8_0 weights, in the
 * canonical order (q8_0.h), CHAINS outputs at a time so that their chains
 * overlap; each block of x is widened to f32 once per CHAINS outputs. fmaf
 * compiles to one instruction where the build targets FMA and to libm's
 * correctly rounded fmaf otherwise: the same bits either way. */
static void range_q8_0(const job *j, int64_t m0, int64_t m1)
{
    const int64_t K = j->K, nb = K / ATH_Q8_0_BLOCK;
    const size_t rowb = ath_dtype_bytes(ATH_DTYPE_Q8_0, K);
    for (int64_t r = 0; r < j->N; r++) {
        for (int64_t m = m0; m < m1; m += CHAINS) {
            const int c = m1 - m < CHAINS ? (int)(m1 - m) : CHAINS;
            float P[CHAINS][ATH_Q8_0_LANES] = {{0}};
            for (int64_t b = 0; b < nb; b++) {
                float xb[ATH_Q8_0_BLOCK];
                for (int e = 0; e < ATH_Q8_0_BLOCK; e++)
                    xb[e] = ld(j->x, r * K + b * ATH_Q8_0_BLOCK + e);
                for (int i = 0; i < c; i++) {
                    const uint8_t *blk = (const uint8_t *)j->w->data + (size_t)(m + i) * rowb + b * ATH_Q8_0_BYTES;
                    const int8_t *q = (const int8_t *)(blk + 2);
                    float s = (float)q[0] * xb[0];
                    for (int e = 1; e < ATH_Q8_0_BLOCK; e++)
                        s = fmaf((float)q[e], xb[e], s);
                    P[i][b % ATH_Q8_0_LANES] = fmaf(ath_q8_0_scale(blk), s, P[i][b % ATH_Q8_0_LANES]);
                }
            }
            for (int i = 0; i < c; i++)
                st(j->out, r * j->M + m + i, ath_q8_0_tree(P[i]));
        }
    }
}

/* Task t of n: a contiguous range of blocks. */
static void task(void *ctx, int32_t t, int32_t n)
{
    const job *j = ctx;
    if (j->w->dtype == ATH_DTYPE_Q8_0) {
        const int64_t b0 = j->blocks * t / n, b1 = j->blocks * (t + 1) / n;
        const int64_t m1 = b1 * CHAINS < j->M ? b1 * CHAINS : j->M;
        if (b0 * CHAINS < m1)
            range_q8_0(j, b0 * CHAINS, m1);
        return;
    }
    for (int64_t b = j->blocks * t / n; b < j->blocks * (t + 1) / n; b++) {
        const int64_t m = b * CHAINS;
        if (m + CHAINS <= j->M && j->w->dtype == ATH_DTYPE_BF16)
            block_bf16(j, m);
        else
            one_at_a_time(j, m, m + CHAINS <= j->M ? m + CHAINS : j->M);
    }
}

ath_status ath_matmul_fast(const ath_tensor *x, const ath_tensor *w, ath_tensor *out, ath_pool *pool)
{
    job j = {.x = x, .w = w, .out = out};
    ATH_CHECK(check(x, w, out, OPS_HOST, &j.N, &j.M, &j.K));
    j.blocks = (j.M + CHAINS - 1) / CHAINS;
    int64_t n = ath_pool_threads(pool);
    if (n > j.blocks)
        n = j.blocks;
    if (n > 1 && (double)j.N * (double)j.M * (double)j.K < MIN_WORK_PER_RUN)
        n = 1;
    ath_pool_run(pool, (int32_t)n, task, &j);
    return ATH_OK;
}

ath_status ath_matmul_gpu(const ath_device *d, const ath_tensor *x, const ath_tensor *w, ath_tensor *out)
{
    int64_t N, M, K;
    ATH_CHECK(check(x, w, out, OPS_GPU, &N, &M, &K));
    return ath_matmul_launch(d, x, w, out);
}

ath_status ath_matmul_n_gpu(const ath_device *d, const ath_tensor *x, int n, const ath_tensor *const *w,
                            ath_tensor *const *out)
{
    if (n < 1 || n > ATH_MATMUL_N_MAX || !w || !out)
        return ops_fail(ATH_ERR_INVALID_ARG, op, "1 to %d weights", ATH_MATMUL_N_MAX);
    for (int i = 0; i < n; i++) {
        int64_t N, M, K;
        ATH_CHECK(check(x, w[i], out[i], OPS_GPU, &N, &M, &K));
    }
    return ath_matmul_n_launch(d, x, n, w, out);
}

bool ath_matmul_fits_gemv(const ath_tensor *x, const ath_tensor *w)
{
    return x && w && x->data && w->data && x->ndim >= 1 && ath_gemv_fits(x, w);
}

ath_status ath_matmul_add_gpu(const ath_device *d, const ath_tensor *x, const ath_tensor *w, ath_tensor *out)
{
    int64_t N, M, K;
    ATH_CHECK(check(x, w, out, OPS_GPU, &N, &M, &K));
    if (!ath_gemv_fits(x, w))
        return ops_fail(ATH_ERR_UNSUPPORTED, op, "matmul_add: shapes the GEMV kernel does not take");
    return ath_matmul_add_launch(d, x, w, out);
}

bool ath_matmul_swiglu_fits(const ath_tensor *x, const ath_tensor *wg, const ath_tensor *wu)
{
    return x && wg && wu && x->data && wg->data && wu->data && x->ndim >= 1 && ath_gemv_swiglu_fits(x, wg, wu);
}

ath_status ath_matmul_swiglu_gpu(const ath_device *d, const ath_tensor *x, const ath_tensor *wg, const ath_tensor *wu,
                                 ath_tensor *out)
{
    int64_t N, M, K;
    ATH_CHECK(check(x, wg, out, OPS_GPU, &N, &M, &K));
    ATH_CHECK(check(x, wu, out, OPS_GPU, &N, &M, &K));
    if (!ath_gemv_swiglu_fits(x, wg, wu))
        return ops_fail(ATH_ERR_UNSUPPORTED, op, "matmul_swiglu: shapes the kernel does not take");
    return ath_matmul_swiglu_launch(d, x, wg, wu, out);
}
