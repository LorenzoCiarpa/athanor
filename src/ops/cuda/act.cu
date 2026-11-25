/* act.cu: the elementwise ops on the GPU: SwiGLU, the sigmoid gate, the
 * residual add. See ops.h and act.c, whose expressions the kernel repeats.
 *
 * One kernel for the three, one thread per element. out may alias a or b:
 * each thread reads its two inputs before it writes the same index.
 */
#include "ops/cuda/kernel.cuh"

static __global__ void elem_kernel(int kind, const void *a, int adt, const void *b, int bdt, void *out, int odt,
                                   int64_t n)
{
    K_FOR(e, n) {
        const float x = k_ld(a, adt, e), y = k_ld(b, bdt, e);
        float r;
        if (kind == ATH_ELEM_SWIGLU)
            r = k_silu(x) * y;                         /* silu(gate) · up */
        else if (kind == ATH_ELEM_SIGMOID_MUL)
            r = x * (1.0f / (1.0f + expf(-y)));        /* x · sigmoid(gate) */
        else
            r = x + y;
        k_st(out, odt, e, r);
    }
}

extern "C" ath_status ath_elem_launch(const ath_device *d, ath_elem_kind kind, const ath_tensor *a,
                                      const ath_tensor *b, ath_tensor *out)
{
    const int64_t n = ath_tensor_numel(out);
    if (n == 0)
        return ATH_OK;
    ATH_CHECK(ath_gpu_use(d));
    elem_kernel<<<k_blocks(n, 256), 256, 0, k_stream(d)>>>((int)kind, a->data, a->dtype, b->data, b->dtype, out->data,
                                                           out->dtype, n);
    return ath_gpu_launched("elem_kernel");
}
