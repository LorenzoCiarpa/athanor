/* gdn.cu: the Gated DeltaNet gates and the gated delta rule, recurrent
 * form, on the GPU. See ops.h and gdn.c.
 *
 * Gates: one thread per (token, value head), the reference's expressions.
 *
 * Recurrence: the columns of a head's state S [dk, dv] never interact
 * (gdn.c), so each column is one warp's job. Lane l keeps rows l, l + 32,
 * ... of the column in registers for all the tokens of the call and the
 * state goes back to memory once at the end; per token the warp decays the
 * column, reads u = Sᵀk (k_warp_sum), writes β (v − u) along k and reads the
 * output with the query (k_warp_sum again). A block holds GDN_WARPS
 * neighbouring columns of one head, so its loads of a state row are
 * contiguous. Tokens run one after the other, as the recurrence requires;
 * the chunked form that parallelizes a long prompt is step 10.
 */
#include "ops/kernel.cuh"

#define GDN_WARPS 8
#define GDN_ROWS (ATH_GDN_MAX_DK / ATH_WARP)   /* state rows per lane */

/* torch.nn.functional.softplus with beta 1 and threshold 20. */
static __device__ __forceinline__ float k_softplus(float x)
{
    return x > 20.0f ? x : log1pf(expf(x));
}

static __global__ void gates_kernel(const void *a, int adt, const void *b, int bdt, const void *a_log, int ldt,
                                    const void *dt_bias, int ddt, void *g, int gdt, void *beta, int betadt, int64_t T,
                                    int64_t nv)
{
    K_FOR(e, T * nv) {
        const int64_t h = e % nv;
        k_st(g, gdt, e, -expf(k_ld(a_log, ldt, h)) * k_softplus(k_ld(a, adt, e) + k_ld(dt_bias, ddt, h)));
        k_st(beta, betadt, e, 1.0f / (1.0f + expf(-k_ld(b, bdt, e))));
    }
}

extern "C" ath_status ath_gdn_gates_launch(const ath_device *d, const ath_tensor *a, const ath_tensor *b,
                                           const ath_tensor *a_log, const ath_tensor *dt_bias, ath_tensor *g,
                                           ath_tensor *beta)
{
    const int64_t T = a->shape[0], nv = a->shape[1];
    if (T * nv == 0)
        return ATH_OK;
    ATH_CHECK(ath_gpu_use(d));
    gates_kernel<<<k_blocks(T * nv, 256), 256, 0, k_stream(d)>>>(a->data, a->dtype, b->data, b->dtype, a_log->data,
                                                                 a_log->dtype, dt_bias->data, dt_bias->dtype, g->data,
                                                                 g->dtype, beta->data, beta->dtype, T, nv);
    return ath_gpu_launched("gates_kernel");
}

static __global__ void __launch_bounds__(GDN_WARPS *ATH_WARP)
gdn_kernel(const void *q, int qdt, const void *k, int kdt, const void *v, int vdt, const void *g, int gdt,
           const void *beta, int bdt, float q_scale, float *state, void *out, int odt, int64_t T, int64_t nk,
           int64_t dk, int64_t nv, int64_t dv)
{
    const int lane = threadIdx.x % ATH_WARP;
    const int64_t h = blockIdx.x;
    const int64_t j = (int64_t)blockIdx.y * GDN_WARPS + threadIdx.x / ATH_WARP;   /* this warp's column */
    if (j >= dv)
        return;                                    /* the whole warp: no block-wide barrier below */
    const int64_t kh = h / (nv / nk);              /* query/key head of value head h */
    float *S = state + h * dk * dv;

    float s[GDN_ROWS];
#pragma unroll
    for (int r = 0; r < GDN_ROWS; r++) {
        const int64_t i = lane + (int64_t)r * ATH_WARP;
        s[r] = i < dk ? S[i * dv + j] : 0.0f;
    }
    for (int64_t t = 0; t < T; t++) {
        const int64_t qk = (t * nk + kh) * dk, tv = t * nv + h;
        const float alpha = expf(k_ld(g, gdt, tv)), b = k_ld(beta, bdt, tv);
        float kk[GDN_ROWS], qq[GDN_ROWS], u = 0.0f;
#pragma unroll
        for (int r = 0; r < GDN_ROWS; r++) {
            const int64_t i = lane + (int64_t)r * ATH_WARP;
            kk[r] = qq[r] = 0.0f;
            if (i < dk) {
                kk[r] = k_ld(k, kdt, qk + i);
                qq[r] = k_ld(q, qdt, qk + i) * q_scale;
                s[r] *= alpha;
                u += s[r] * kk[r];
            }
        }
        u = k_warp_sum(u);
        const float delta = (k_ld(v, vdt, tv * dv + j) - u) * b;
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < GDN_ROWS; r++) {
            if (lane + (int64_t)r * ATH_WARP < dk) {
                s[r] += kk[r] * delta;
                o += s[r] * qq[r];
            }
        }
        o = k_warp_sum(o);
        if (lane == 0)
            k_st(out, odt, tv * dv + j, o);
    }
#pragma unroll
    for (int r = 0; r < GDN_ROWS; r++) {
        const int64_t i = lane + (int64_t)r * ATH_WARP;
        if (i < dk)
            S[i * dv + j] = s[r];
    }
}

extern "C" ath_status ath_gdn_launch(const ath_device *d, const ath_tensor *q, const ath_tensor *k,
                                     const ath_tensor *v, const ath_tensor *g, const ath_tensor *beta, float q_scale,
                                     ath_tensor *state, ath_tensor *out)
{
    const int64_t T = q->shape[0], nk = q->shape[1], dk = q->shape[2], nv = v->shape[1], dv = v->shape[2];
    if (nv == 0 || dv == 0 || T == 0)
        return ATH_OK;
    const int64_t col_blocks = (dv + GDN_WARPS - 1) / GDN_WARPS;
    if (nv > INT32_MAX || col_blocks > 65535) {
        ath_log(ATH_LOG_ERROR, "gdn: %lld heads of %lld columns, more than one launch holds", (long long)nv,
                (long long)dv);
        return ATH_ERR_UNSUPPORTED;
    }
    ATH_CHECK(ath_gpu_use(d));
    const dim3 grid((unsigned)nv, (unsigned)col_blocks);
    gdn_kernel<<<grid, GDN_WARPS * ATH_WARP, 0, k_stream(d)>>>(q->data, q->dtype, k->data, k->dtype, v->data, v->dtype,
                                                               g->data, g->dtype, beta->data, beta->dtype, q_scale,
                                                               (float *)state->data, out->data, out->dtype, T, nk, dk,
                                                               nv, dv);
    return ath_gpu_launched("gdn_kernel");
}
