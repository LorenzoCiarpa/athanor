/* q8_0.h: Q8_0 weights, the canonical dot product and the canonical
 * exponential (step 12). Private to src/ops/: C files and .cu files include
 * it, and both compile the same expressions.
 *
 * A Q8_0 row of K elements is K / 32 blocks of 34 bytes: an f16 scale d
 * (little-endian) and 32 int8 values q; element i of a block is d · q[i],
 * exact in f32 (11 significant bits times 8).
 *
 * The canonical dot product. Invariant 7, read as option (b) at step 12
 * (docs/STATUS.md): a routed expert computed on the CPU and on the GPU gives
 * the same bits, so every implementation of a Q8_0 dot product follows one
 * order of operations, the one below:
 *
 *   s_b    = q_b[0] x_b[0], then s_b = fma(q_b[i], x_b[i], s_b), i = 1 … 31
 *   P[j]   = 0 for j = 0 … 15; for b = 0, 1, …: P[b mod 16] = fma(d_b, s_b, P[b mod 16])
 *   result = the tree P[j] += P[j + h], h = 8, 4, 2, 1 (j < h); then P[0]
 *
 * Each step is one IEEE operation rounded to nearest (fma rounds once), so
 * the result is the same wherever it is computed: 16 lanes of an AVX-512
 * register on the CPU, 16 lanes of a warp on the GPU (the tree is the xor
 * butterfly of __shfl_xor, whose pairs are the tree's: lane j adds lane
 * j ^ h, and addition is commutative). With bf16 activations every product
 * q · x is exact (8 + 8 significant bits), so the fma inside a block only
 * rounds the running sum.
 *
 * The canonical exponential. SiLU inside an expert needs exp, and libm's
 * expf and CUDA's expf differ in the last bits. ath_q8_expf uses only
 * operations IEEE defines exactly (fma, multiply, add, floor, integer bit
 * assembly), so the CPU and the GPU agree bit for bit; its error is within
 * 2 ulp of exp on [-87, 88] (tests/test_ops.c measures it).
 */
#ifndef ATH_OPS_Q8_0_H
#define ATH_OPS_Q8_0_H

#include "core/tensor.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#ifdef __CUDACC__
#define ATH_HD __host__ __device__
#else
#define ATH_HD
#endif

#define ATH_Q8_0_LANES 16   /* partial sums of the canonical dot product */

/* f16 bits to f32, exactly (subnormals, infinities and NaN included). */
static inline ATH_HD float ath_f16_to_f32(uint16_t h)
{
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu, man = h & 0x3ffu, u;
    if (exp == 0x1fu) {
        u = sign | 0x7f800000u | (man << 13);                 /* inf, NaN */
    } else if (exp != 0) {
        u = sign | ((exp + 112u) << 23) | (man << 13);        /* normal: rebias 15 → 127 */
    } else if (man == 0) {
        u = sign;                                             /* ±0 */
    } else {                                                  /* subnormal: normalize */
        exp = 113u;
        while (!(man & 0x400u)) {
            man <<= 1;
            exp--;
        }
        u = sign | (exp << 23) | ((man & 0x3ffu) << 13);
    }
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

/* The scale of a block (its first two bytes). */
static inline ATH_HD float ath_q8_0_scale(const uint8_t *block)
{
    return ath_f16_to_f32((uint16_t)(block[0] | (uint16_t)block[1] << 8));
}

/* The tree over the 16 partial sums. */
static inline ATH_HD float ath_q8_0_tree(float P[ATH_Q8_0_LANES])
{
    for (int h = ATH_Q8_0_LANES / 2; h >= 1; h /= 2)
        for (int j = 0; j < h; j++)
            P[j] = P[j] + P[j + h];
    return P[0];
}

/* exp(x) by the canonical sequence: x = n ln 2 + r with n = floor(x log2 e
 * + 1/2) and |r| <= ln 2 / 2 (ln 2 split in two parts, so n ln 2 is exact to
 * 2^-32), exp(r) by its Taylor polynomial of degree 7 in Horner form, scaled
 * by 2^n assembled from bits. Inputs below -87 give exp(-87) (2^n stays a
 * normal number), above 88 exp(88); NaN gives NaN. */
static inline ATH_HD float ath_q8_expf(float x)
{
    if (x != x)
        return x;
    x = x < -87.0f ? -87.0f : (x > 88.0f ? 88.0f : x);
    const float n = floorf(fmaf(x, 1.44269502e+00f, 0.5f));
    float r = fmaf(n, -6.93145752e-01f, x);    /* ln 2 high part, 0x3f317200: n · it is exact */
    r = fmaf(n, -1.42860677e-06f, r);          /* ln 2 low part                               */
    float p = 1.98412701e-04f;                 /* 1/5040 */
    p = fmaf(p, r, 1.38888892e-03f);           /* 1/720  */
    p = fmaf(p, r, 8.33333377e-03f);           /* 1/120  */
    p = fmaf(p, r, 4.16666679e-02f);           /* 1/24   */
    p = fmaf(p, r, 1.66666672e-01f);           /* 1/6    */
    p = fmaf(p, r, 5.00000000e-01f);           /* 1/2    */
    p = fmaf(p, r, 1.0f);
    p = fmaf(p, r, 1.0f);
    const int e = (int)n;                      /* in [-126, 127] after the clamp */
    const uint32_t bits = (uint32_t)(e + 127) << 23;
    float s;
    memcpy(&s, &bits, sizeof s);
    return p * s;
}

/* SiLU by the canonical exponential: x / (1 + exp(-x)), one IEEE division. */
static inline ATH_HD float ath_q8_silu(float x)
{
    return x / (1.0f + ath_q8_expf(-x));
}

#endif /* ATH_OPS_Q8_0_H */
