/* gdn.cu: the Gated DeltaNet gates and the gated delta rule, recurrent
 * form, on the GPU. See ops.h and gdn.c.
 *
 * Gates: one thread per (token, value head), the reference's expressions.
 *
 * Recurrence: the columns of a head's state S [dk, dv] never interact
 * (gdn.c). A block takes 32 neighbouring columns of one head, a lane per
 * column, so that every row of the state it reads or writes is one
 * contiguous 128-byte access (step 9; a warp per column read rows 512 bytes
 * apart). The GDN_WARPS warps split the rows of the columns: each keeps its
 * rows in registers for all the tokens of the call, and the state goes
 * back to memory once at the end. Per token every warp decays its rows and
 * adds its part of u = Sᵀk, the parts are added in warp order, then each
 * warp writes β (v − u) along k into its rows and adds its part of the
 * output with the query, added in warp order again. The order of every sum
 * is fixed by dk, so feeding tokens in pieces gives the bits of feeding them
 * at once. Tokens run one after the other, as the recurrence requires;
 * the chunked form that parallelizes a long prompt is gdn_chunk.cu (step
 * 10). The query and key of each token are L2-normalized here (step 10):
 * sums of squares over the warps, added in warp order.
 */
#include "ops/cuda/kernel.cuh"

#define GDN_WARPS 4
#define GDN_ROWS (ATH_GDN_MAX_DK / GDN_WARPS)   /* state rows per warp */

/* torch.nn.functional.softplus with beta 1 and threshold 20. */
static __device__ __forceinline__ float k_softplus(float x)
{
    return x > 20.0f ? x : log1pf(expf(x));
}

static __global__ void gates_kernel(const void *a, int adt, const void *b, int bdt, const void *a_log, int ldt,
                                    bool a_given, const void *dt_bias, int ddt, void *g, int gdt, void *beta,
                                    int betadt, int64_t T, int64_t nv)
{
    K_FOR(e, T * nv) {
        const int64_t h = e % nv;
        const float A = a_given ? k_ld(a_log, ldt, h) : -expf(k_ld(a_log, ldt, h));
        k_st(g, gdt, e, A * k_softplus(k_ld(a, adt, e) + k_ld(dt_bias, ddt, h)));
        k_st(beta, betadt, e, 1.0f / (1.0f + expf(-k_ld(b, bdt, e))));
    }
}

extern "C" ath_status ath_gdn_gates_launch(const ath_device *d, const ath_tensor *a, const ath_tensor *b,
                                           const ath_tensor *a_log, bool a_given, const ath_tensor *dt_bias,
                                           ath_tensor *g, ath_tensor *beta)
{
    const int64_t T = a->shape[0], nv = a->shape[1];
    if (T * nv == 0)
        return ATH_OK;
    ATH_CHECK(ath_gpu_use(d));
    gates_kernel<<<k_blocks(T * nv, 256), 256, 0, k_stream(d)>>>(a->data, a->dtype, b->data, b->dtype, a_log->data,
                                                                 a_log->dtype, a_given, dt_bias->data, dt_bias->dtype, g->data,
                                                                 g->dtype, beta->data, beta->dtype, T, nv);
    return ath_gpu_launched("gates_kernel");
}

static __device__ __forceinline__ void
gdn_body(int rows_max, const void *q, int qdt, const void *k, int kdt, const void *v, int vdt, const void *g, int gdt,
         const void *beta, int bdt, float q_scale, float l2_eps, float *state, void *out, int odt, int64_t T,
         int64_t nk, bool tiled, int64_t dk, int64_t nv, int64_t dv, int64_t qs, int64_t ks, int64_t vs)
{
    __shared__ float kq[2][ATH_GDN_MAX_DK];         /* the token's k and q, as given  */
    __shared__ float part[GDN_WARPS][ATH_WARP];      /* the warps' partial sums      */
    __shared__ float l2[2][GDN_WARPS];               /* the warps' sums of k², q²     */
    const int lane = threadIdx.x % ATH_WARP, warp = threadIdx.x / ATH_WARP;
    const int64_t h = blockIdx.x, j = (int64_t)blockIdx.y * ATH_WARP + lane;   /* this lane's column */
    const bool live = j < dv;
    const int64_t kh = tiled ? h % nk : h / (nv / nk);   /* query/key head of value head h */
    float *S = state + h * dk * dv;

    /* rows warp, warp + GDN_WARPS, ... of the column, in registers for the
     * whole call; a row of the block's 32 columns is one 128-byte load */
    float s[GDN_ROWS];
#pragma unroll
    for (int r = 0; r < GDN_ROWS; r++) {
        if (r >= rows_max)
            break;
        const int64_t i = warp + (int64_t)r * GDN_WARPS;
        s[r] = live && i < dk ? S[i * dv + j] : 0.0f;
    }
    for (int64_t t = 0; t < T; t++) {
        const int64_t qb = t * qs + kh * dk, kb = t * ks + kh * dk, tv = t * nv + h;
        float sk = 0.0f, sq = 0.0f;
        for (int64_t i = threadIdx.x; i < dk; i += blockDim.x) {
            const float kv = k_ld(k, kdt, kb + i), qv = k_ld(q, qdt, qb + i);
            kq[0][i] = kv;
            kq[1][i] = qv;
            sk += kv * kv;
            sq += qv * qv;
        }
        sk = k_warp_sum(sk);
        sq = k_warp_sum(sq);
        if (lane == 0)
            l2[0][warp] = sk, l2[1][warp] = sq;
        const float alpha = expf(k_ld(g, gdt, tv)), b = k_ld(beta, bdt, tv);
        __syncthreads();
        sk = l2[0][0], sq = l2[1][0];
        for (int w = 1; w < GDN_WARPS; w++)
            sk += l2[0][w], sq += l2[1][w];
        const float ik = 1.0f / sqrtf(sk + l2_eps), iq = 1.0f / sqrtf(sq + l2_eps);
        /* u = Sᵀk over this warp's rows, then over the warps in order */
        float u = 0.0f;
#pragma unroll
        for (int r = 0; r < GDN_ROWS; r++) {
            if (r >= rows_max)
                break;
        if (r >= rows_max)
            break;
            const int64_t i = warp + (int64_t)r * GDN_WARPS;
            if (i < dk) {
                s[r] *= alpha;
                u += s[r] * (kq[0][i] * ik);
            }
        }
        part[warp][lane] = u;
        __syncthreads();
        u = part[0][lane];
        for (int w = 1; w < GDN_WARPS; w++)
            u += part[w][lane];
        const float delta = live ? (k_ld(v, vdt, t * vs + h * dv + j) - u) * b : 0.0f;
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < GDN_ROWS; r++) {
            if (r >= rows_max)
                break;
        if (r >= rows_max)
            break;
            const int64_t i = warp + (int64_t)r * GDN_WARPS;
            if (i < dk) {
                s[r] += (kq[0][i] * ik) * delta;
                o += s[r] * ((kq[1][i] * iq) * q_scale);
            }
        }
        __syncthreads();                             /* every warp has read part */
        part[warp][lane] = o;
        __syncthreads();
        if (warp == 0 && live) {
            o = part[0][lane];
            for (int w = 1; w < GDN_WARPS; w++)
                o += part[w][lane];
            k_st(out, odt, tv * dv + j, o);
        }
        __syncthreads();                             /* kq and part are rewritten next */
    }
#pragma unroll
    for (int r = 0; r < GDN_ROWS; r++) {
        if (r >= rows_max)
            break;
        const int64_t i = warp + (int64_t)r * GDN_WARPS;
        if (live && i < dk)
            S[i * dv + j] = s[r];
    }
}

/* The body with the rows a warp holds a constant: 32 for dk up to 128 (the
 * family's heads), 64 up to ATH_GDN_MAX_DK. Half the code and registers for
 * the common case; decode launches it 18 times per token (step 10). */
static __global__ void __launch_bounds__(GDN_WARPS *ATH_WARP)
gdn_kernel(const void *q, int qdt, const void *k, int kdt, const void *v, int vdt, const void *g, int gdt,
           const void *beta, int bdt, float q_scale, float l2_eps, float *state, void *out, int odt, int64_t T,
           int64_t nk, bool tiled, int64_t dk, int64_t nv, int64_t dv, int64_t qs, int64_t ks, int64_t vs)
{
    gdn_body(GDN_ROWS, q, qdt, k, kdt, v, vdt, g, gdt, beta, bdt, q_scale, l2_eps, state, out, odt, T, nk, tiled, dk,
             nv, dv, qs, ks, vs);
}

static __global__ void __launch_bounds__(GDN_WARPS *ATH_WARP)
gdn_kernel_128(const void *q, int qdt, const void *k, int kdt, const void *v, int vdt, const void *g, int gdt,
               const void *beta, int bdt, float q_scale, float l2_eps, float *state, void *out, int odt, int64_t T,
               int64_t nk, bool tiled, int64_t dk, int64_t nv, int64_t dv, int64_t qs, int64_t ks, int64_t vs)
{
    gdn_body(128 / GDN_WARPS, q, qdt, k, kdt, v, vdt, g, gdt, beta, bdt, q_scale, l2_eps, state, out, odt, T, nk,
             tiled, dk, nv, dv, qs, ks, vs);
}

extern "C" ath_status ath_gdn_launch(const ath_device *d, const ath_tensor *q, const ath_tensor *k,
                                     const ath_tensor *v, const ath_tensor *g, const ath_tensor *beta, float q_scale,
                                     float l2_eps, bool tiled, ath_tensor *state, ath_tensor *out)
{
    const int64_t T = q->shape[0], nk = q->shape[1], dk = q->shape[2], nv = v->shape[1], dv = v->shape[2];
    if (nv == 0 || dv == 0 || T == 0)
        return ATH_OK;
    const int64_t col_blocks = (dv + ATH_WARP - 1) / ATH_WARP;
    if (nv > INT32_MAX || col_blocks > 65535) {
        ath_log(ATH_LOG_ERROR, "gdn: %lld heads of %lld columns, more than one launch holds", (long long)nv,
                (long long)dv);
        return ATH_ERR_UNSUPPORTED;
    }
    ATH_CHECK(ath_gpu_use(d));
    const dim3 grid((unsigned)nv, (unsigned)col_blocks);
    (dk <= 128 ? gdn_kernel_128 : gdn_kernel)<<<grid, GDN_WARPS * ATH_WARP, 0, k_stream(d)>>>(
        q->data, q->dtype, k->data, k->dtype, v->data, v->dtype, g->data, g->dtype, beta->data, beta->dtype, q_scale,
        l2_eps, (float *)state->data, out->data, out->dtype, T, nk, tiled, dk, nv, dv, q->stride[0], k->stride[0],
        v->stride[0]);
    return ath_gpu_launched("gdn_kernel");
}
