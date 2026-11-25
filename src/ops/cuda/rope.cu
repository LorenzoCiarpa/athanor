/* rope.cu: rotary position embedding on the GPU. See ops.h and rope.c.
 *
 * One thread per pair (i, i + rope_dim / 2) of the rotary part, and one per
 * element copied unchanged after it. The frequencies come from the host
 * (rope.c computes them with the reference's expression and libm's powf),
 * passed by value in the kernel's arguments: the angle p · f_i would turn a
 * one-ulp difference of the device's powf into p ulps. cos and sin of the
 * angle are the device's, rounded to the activation dtype as the reference
 * rounds its tables.
 */
#include "ops/cuda/kernel.cuh"

typedef struct {
    float f[ATH_ROPE_MAX_DIM / 2];
} rope_freqs;

static __global__ void rope_kernel(const void *x, int xdt, const int32_t *pos, rope_freqs fr, int half, int64_t T,
                                   int64_t heads, int64_t D, void *out, int odt)
{
    const int64_t per_row = D - half;   /* half pairs, then D − rope_dim copies */
    K_FOR(e, T * heads * per_row) {
        const int64_t r = e / per_row, j = e % per_row, base = r * D;
        if (j < half) {
            const float angle = (float)pos[r / heads] * fr.f[j];
            const float c = k_round(xdt, cosf(angle)), s = k_round(xdt, sinf(angle));
            const float x0 = k_ld(x, xdt, base + j), x1 = k_ld(x, xdt, base + j + half);
            k_st(out, odt, base + j, x0 * c - x1 * s);
            k_st(out, odt, base + j + half, x1 * c + x0 * s);
        } else {
            const int64_t i = j + half;   /* in [rope_dim, D) */
            k_st(out, odt, base + i, k_ld(x, xdt, base + i));
        }
    }
}

extern "C" ath_status ath_rope_launch(const ath_device *d, const ath_tensor *x, const ath_tensor *pos,
                                      int32_t rope_dim, const float *freq, ath_tensor *out)
{
    const int64_t T = x->shape[0], heads = x->shape[1], D = x->shape[2];
    const int half = rope_dim / 2;
    const int64_t n = T * heads * (D - half);
    if (n == 0)
        return ATH_OK;
    rope_freqs fr;
    memset(&fr, 0, sizeof fr);
    memcpy(fr.f, freq, (size_t)half * sizeof(float));
    ATH_CHECK(ath_gpu_use(d));
    rope_kernel<<<k_blocks(n, 256), 256, 0, k_stream(d)>>>(x->data, x->dtype, (const int32_t *)pos->data, fr, half, T,
                                                           heads, D, out->data, out->dtype);
    return ath_gpu_launched("rope_kernel");
}
