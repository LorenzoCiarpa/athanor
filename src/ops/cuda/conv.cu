/* conv.cu: causal depthwise convolution with a running state, on the GPU.
 * See ops.h and conv.c.
 *
 * Two kernels on the compute stream, so the second starts after the first
 * has finished: one thread per output (t, c) reads its K inputs from
 * xx = [state; x], as the reference does; then one thread per channel
 * writes the new state (the last K − 1 rows of xx) in ascending order,
 * which is safe in place for the reason conv.c gives. Writing the state in
 * the first kernel would race with the threads still reading it.
 *
 * With one token (decode, step 9) the thread of channel c is the only one
 * that reads the state of c, so one kernel computes the output and then
 * moves the state: the same arithmetic, one launch instead of two.
 */
#include "ops/cuda/kernel.cuh"

/* Row r, channel c of xx = [state; x]. */
static __device__ __forceinline__ float k_xx(const void *state, int sdt, const void *x, int xdt, int64_t r,
                                             int64_t c, int64_t C, int64_t past)
{
    return r < past ? k_ld(state, sdt, r * C + c) : k_ld(x, xdt, (r - past) * C + c);
}

static __global__ void conv_kernel(const void *x, int xdt, const void *w, int wdt, const void *state, int sdt,
                                   int silu, void *out, int odt, int64_t T, int64_t C, int64_t K)
{
    K_FOR(e, T * C) {
        const int64_t t = e / C, c = e % C;
        float acc = 0.0f;
        for (int64_t j = 0; j < K; j++)
            acc += k_ld(w, wdt, c * K + j) * k_xx(state, sdt, x, xdt, t + j, c, C, K - 1);
        if (silu)
            acc = k_silu(acc);
        k_st(out, odt, e, acc);
    }
}

static __global__ void conv_state_kernel(const void *x, int xdt, void *state, int sdt, int64_t T, int64_t C,
                                         int64_t K)
{
    K_FOR(c, C) {
        for (int64_t i = 0; i < K - 1; i++)
            k_st(state, sdt, i * C + c, k_xx(state, sdt, x, xdt, T + i, c, C, K - 1));
    }
}

static __global__ void conv_one_kernel(const void *x, int xdt, const void *w, int wdt, void *state, int sdt,
                                       int silu, void *out, int odt, int64_t C, int64_t K)
{
    K_FOR(c, C) {
        float acc = 0.0f;
        for (int64_t j = 0; j < K; j++)
            acc += k_ld(w, wdt, c * K + j) * k_xx(state, sdt, x, xdt, j, c, C, K - 1);
        if (silu)
            acc = k_silu(acc);
        k_st(out, odt, c, acc);
        for (int64_t i = 0; i < K - 1; i++)
            k_st(state, sdt, i * C + c, k_xx(state, sdt, x, xdt, 1 + i, c, C, K - 1));
    }
}

extern "C" ath_status ath_conv_launch(const ath_device *d, const ath_tensor *x, const ath_tensor *w,
                                      ath_tensor *state, ath_act act, ath_tensor *out)
{
    const int64_t T = x->shape[0], C = x->shape[1], K = w->shape[w->ndim - 1];
    if (T * C == 0)
        return ATH_OK;
    ATH_CHECK(ath_gpu_use(d));
    if (T == 1) {
        conv_one_kernel<<<k_blocks(C, 256), 256, 0, k_stream(d)>>>(x->data, x->dtype, w->data, w->dtype, state->data,
                                                                   state->dtype, act == ATH_ACT_SILU, out->data,
                                                                   out->dtype, C, K);
        return ath_gpu_launched("conv_one_kernel");
    }
    conv_kernel<<<k_blocks(T * C, 256), 256, 0, k_stream(d)>>>(x->data, x->dtype, w->data, w->dtype, state->data,
                                                               state->dtype, act == ATH_ACT_SILU, out->data,
                                                               out->dtype, T, C, K);
    ATH_CHECK(ath_gpu_launched("conv_kernel"));
    if (K < 2)
        return ATH_OK;
    conv_state_kernel<<<k_blocks(C, 256), 256, 0, k_stream(d)>>>(x->data, x->dtype, state->data, state->dtype, T, C,
                                                                 K);
    return ath_gpu_launched("conv_state_kernel");
}
