/* sample.cu: the next token, chosen on the GPU (step 9). See ops.h and
 * sample.c.
 *
 * Decode on the GPU would otherwise copy the logits (a megabyte for a
 * vocabulary of 248k) to the host at every token and wait for them. Here the
 * choice is made where the logits are, and only the token id is written.
 *
 * The k candidates (k = 1 for greedy, top_k otherwise) are the first k in
 * the reference's order: larger logit first, then lower id. Two kernels:
 *
 *   select   each block holds SAMPLE_CHUNK logits in registers and takes its
 *            k best one after the other: the best that comes after the one
 *            taken before (a warp reduction, then one across the warps)
 *   merge    one block does the same over the blocks' candidates, then
 *            thread 0 runs the reference's arithmetic on the k it kept:
 *            probabilities exp((l − l_max) / T) in double, the top-p and
 *            min-p cuts, and the inverse transform of u
 *
 * Every step is the reference's, in the same order and in double, so the
 * token is the reference's; only exp on the device may differ from libm's
 * in the last bit of a double.
 */
#include "ops/cuda/kernel.cuh"

#define SAMPLE_THREADS 256
#define SAMPLE_PER_THREAD 4
#define SAMPLE_CHUNK (SAMPLE_THREADS * SAMPLE_PER_THREAD)
#define MERGE_THREADS 1024
#define MERGE_PER_THREAD 16       /* blocks × k candidates over 1024 threads */

typedef struct {
    float   l;
    int32_t id;
} cand;

/* a comes before b in the reference's order. */
static __device__ __forceinline__ bool before(cand a, cand b) { return a.l > b.l || (a.l == b.l && a.id < b.id); }

static __device__ __forceinline__ cand mkc(float l, int32_t id)
{
    cand c;
    c.l = l;
    c.id = id;
    return c;
}

#define NONE mkc(-INFINITY, INT32_MAX)

/* The first of the threads' candidates, on every thread of the block. */
static __device__ cand block_first(cand c, cand *sh)
{
    const int lane = threadIdx.x % ATH_WARP, warp = threadIdx.x / ATH_WARP, warps = blockDim.x / ATH_WARP;
    for (int m = ATH_WARP / 2; m > 0; m >>= 1) {
        const cand o = mkc(ath_shfl_xor(c.l, m), ath_shfl_xor(c.id, m));
        if (before(o, c))
            c = o;
    }
    __syncthreads();                       /* sh may still be read by the previous call */
    if (lane == 0)
        sh[warp] = c;
    __syncthreads();
    cand best = sh[0];
    for (int w = 1; w < warps; w++)
        if (before(sh[w], best))
            best = sh[w];
    return best;
}

/* Of this thread's candidates v[0 .. n), the first that comes after prev. */
static __device__ __forceinline__ cand first_after(const cand *v, int n, cand prev, bool any)
{
    cand best = NONE;
    for (int i = 0; i < n; i++)
        if ((!any || before(prev, v[i])) && before(v[i], best))
            best = v[i];
    return best;
}

static __global__ void __launch_bounds__(SAMPLE_THREADS)
sample_select(const void *logits, int ldt, int64_t V, int k, cand *out)
{
    __shared__ cand sh[SAMPLE_THREADS / ATH_WARP];
    cand v[SAMPLE_PER_THREAD];
    const int64_t base = (int64_t)blockIdx.x * SAMPLE_CHUNK;
#pragma unroll
    for (int i = 0; i < SAMPLE_PER_THREAD; i++) {
        const int64_t e = base + threadIdx.x + (int64_t)i * SAMPLE_THREADS;
        v[i] = e < V ? mkc(k_ld(logits, ldt, e), (int32_t)e) : NONE;
    }
    cand prev = NONE;
    for (int j = 0; j < k; j++) {
        prev = block_first(first_after(v, SAMPLE_PER_THREAD, prev, j > 0), sh);
        if (threadIdx.x == 0)
            out[(int64_t)blockIdx.x * k + j] = prev;
    }
}

static __global__ void __launch_bounds__(MERGE_THREADS)
sample_merge(const cand *in, int n, int k, float temperature, float top_p, float min_p, const double *u, int32_t *out)
{
    __shared__ cand sh[MERGE_THREADS / ATH_WARP];
    __shared__ cand c[ATH_SAMPLE_GPU_MAX_K];
    cand v[MERGE_PER_THREAD];
    int mine = 0;
    for (int i = threadIdx.x; i < n && mine < MERGE_PER_THREAD; i += MERGE_THREADS)
        v[mine++] = in[i];
    cand prev = NONE;
    for (int j = 0; j < k; j++) {
        prev = block_first(first_after(v, mine, prev, j > 0), sh);
        if (threadIdx.x == 0)
            c[j] = prev;
    }
    if (threadIdx.x != 0)
        return;
    if (k == 1) {                          /* greedy */
        *out = c[0].id;
        return;
    }
    /* sample.c, from "Relative probabilities": the same steps in double */
    double w[ATH_SAMPLE_GPU_MAX_K];
    const double T = temperature, top = c[0].l;
    double total = 0;
    for (int i = 0; i < k; i++) {
        w[i] = exp(((double)c[i].l - top) / T);
        total += w[i];
    }
    int keep = k;
    if (top_p < 1) {
        double cum = 0;
        for (int i = 0; i < k; i++) {
            cum += w[i];
            if (cum >= top_p * total) {
                keep = i + 1;
                break;
            }
        }
    }
    while (keep > 1 && w[keep - 1] < min_p * w[0])
        keep--;
    double kept = 0;
    for (int i = 0; i < keep; i++)
        kept += w[i];
    const double r = *u * kept;
    double cum = 0;
    int32_t id = c[keep - 1].id;
    for (int i = 0; i < keep; i++) {
        cum += w[i];
        if (cum > r) {
            id = c[i].id;
            break;
        }
    }
    *out = id;
}

extern "C" size_t ath_sample_launch_bytes(int64_t V, int32_t k)
{
    return (size_t)((V + SAMPLE_CHUNK - 1) / SAMPLE_CHUNK) * (size_t)k * sizeof(cand);
}

extern "C" ath_status ath_sample_launch(const ath_device *d, const ath_tensor *logits, const ath_sampler *p,
                                        int32_t k, const double *u, void *work, int32_t *out)
{
    const int64_t V = logits->shape[logits->ndim - 1];
    const int64_t blocks = (V + SAMPLE_CHUNK - 1) / SAMPLE_CHUNK;
    if (blocks * k > (int64_t)MERGE_THREADS * MERGE_PER_THREAD) {
        ath_log(ATH_LOG_ERROR, "sample: %lld candidates, more than the merge holds", (long long)(blocks * k));
        return ATH_ERR_UNSUPPORTED;
    }
    ATH_CHECK(ath_gpu_use(d));
    sample_select<<<(unsigned)blocks, SAMPLE_THREADS, 0, k_stream(d)>>>(logits->data, logits->dtype, V, k,
                                                                         (cand *)work);
    ATH_CHECK(ath_gpu_launched("sample_select"));
    sample_merge<<<1, MERGE_THREADS, 0, k_stream(d)>>>((const cand *)work, (int)(blocks * k), k, p->temperature,
                                                        p->top_p, p->min_p, u, out);
    return ath_gpu_launched("sample_merge");
}

/* ---- presence penalty (step 13) ----------------------------------------------
 *
 * One thread per logit. The thread of the step's token marks it in seen;
 * no other thread reads that byte, so marking and penalizing need no
 * ordering between threads. */
static __global__ void presence_kernel(float *logits, uint8_t *seen, const int32_t *token, int64_t V, float penalty)
{
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= V)
        return;
    bool s = seen[i] != 0;
    if (i == *token && !s) {
        seen[i] = 1;
        s = true;
    }
    if (s)
        logits[i] -= penalty;
}

extern "C" ath_status ath_presence_launch(const ath_device *d, ath_tensor *logits, uint8_t *seen, const int32_t *token,
                                          float penalty)
{
    const int64_t V = logits->shape[logits->ndim - 1];
    ATH_CHECK(ath_gpu_use(d));
    presence_kernel<<<(unsigned)((V + 255) / 256), 256, 0, k_stream(d)>>>((float *)logits->data, seen, token, V,
                                                                          penalty);
    return ath_gpu_launched("presence");
}
