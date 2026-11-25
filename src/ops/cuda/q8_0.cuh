/* q8_0.cuh: Q8_0 on the GPU, in the canonical order of src/ops/q8_0.h
 * (step 12). Private to the .cu files of src/ops/cuda/.
 *
 * A half-warp computes one dot product: lane j takes the blocks b with
 * b mod 16 = j, keeps its partial sum P[j], and the 16 partial sums meet in
 * the xor butterfly of k_q8_tree, which is the canonical tree. Inside a
 * block the lane runs the canonical chain over its 32 elements.
 *
 * Loading. The blocks are 34 bytes and only 2-byte aligned, and lane j of
 * a half-warp reads block 16 g + j: a lane reading its own block would
 * touch some 17 sectors per instruction. So the 16 lanes first copy the 16
 * blocks of a group (544 contiguous bytes, 34 pieces of 16 bytes, on
 * 16-byte boundaries when K is a multiple of 512) into shared memory with
 * coalesced loads, and each lane then reads its block there as 32-bit
 * words, shifted into place with funnel shifts (k_q8_words). The regions of
 * the two half-warps of a warp are Q8_REGION bytes apart, which puts their
 * words in disjoint banks.
 */
#ifndef ATH_OPS_Q8_0_CUH
#define ATH_OPS_Q8_0_CUH

#include "ops/cuda/kernel.cuh"
#include "ops/q8_0.h"

#define Q8_GROUP_BYTES (16 * ATH_Q8_0_BYTES)   /* 544: one block per lane of a half-warp */
#define Q8_REGION 576                           /* shared bytes per half-warp: 544, 8 of
                                                 * slack for the last lane's tenth word,
                                                 * and padding to change banks          */

/* The 9 words of block j of a staged group, block bytes 4 i … 4 i + 3 in
 * word i (bytes 34 and 35 of the last word are the next block's). */
static __device__ __forceinline__ void k_q8_words(const unsigned char *region, int j, uint32_t nw[9])
{
    const int at = j * ATH_Q8_0_BYTES, sh = at & 3;       /* 0 or 2 */
    const uint32_t *w = (const uint32_t *)(region + (at - sh));
    uint32_t raw[10];
#pragma unroll
    for (int i = 0; i < 10; i++)
        raw[i] = w[i];
#pragma unroll
    for (int i = 0; i < 9; i++)
        nw[i] = __funnelshift_r(raw[i], raw[i + 1], 8 * sh);
}

/* Value i of a block from its words: byte 2 + i, as a signed integer. */
static __device__ __forceinline__ float k_q8_val(const uint32_t nw[9], int i)
{
    return (float)(int8_t)(nw[(2 + i) >> 2] >> (8 * ((2 + i) & 3)));
}

static __device__ __forceinline__ float k_q8_scale(const uint32_t nw[9])
{
    return ath_f16_to_f32((uint16_t)(nw[0] & 0xffffu));
}

/* The canonical chain of one block against 32 activations. */
static __device__ __forceinline__ float k_q8_chain(const uint32_t nw[9], const float xf[32])
{
    float s = k_q8_val(nw, 0) * xf[0];
#pragma unroll
    for (int i = 1; i < ATH_Q8_0_BLOCK; i++)
        s = fmaf(k_q8_val(nw, i), xf[i], s);
    return s;
}

/* The canonical tree over the 16 lanes of a half-warp; every lane of the
 * half-warp gets the result. All 32 lanes of the warp must call it. */
static __device__ __forceinline__ float k_q8_tree(float v)
{
#pragma unroll
    for (int h = ATH_Q8_0_LANES / 2; h >= 1; h /= 2)
        v = v + ath_shfl_xor(v, h);
    return v;
}

/* Activations 32 b … 32 b + 31 of a row of an f32 or bf16 buffer (on a
 * 16-byte boundary). */
static __device__ __forceinline__ void k_q8_x(const void *x, int xdt, int64_t row_elem, int64_t b, float xf[32])
{
    const int64_t e8 = (row_elem + b * ATH_Q8_0_BLOCK) / 8;
#pragma unroll
    for (int c = 0; c < 4; c++)
        k_ld8(x, xdt, e8 + c, xf + 8 * c);
}

/* 16 bytes of weights, read once: past L1. */
static __device__ __forceinline__ uint4 k_q8_ld(const uint4 *p)
{
#if defined(__HIP_PLATFORM_AMD__)
    return __builtin_nontemporal_load(p);
#else
    uint4 r;
    asm("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
        : "l"(p));
    return r;
#endif
}

/* The 34 pieces of a group, lane j of the half-warp loading pieces j and
 * j + 16, lanes 0 and 1 also 32 and 33 (zeros where !live). */
typedef struct {
    uint4 a, b, c;
} q8_pieces;

static __device__ __forceinline__ q8_pieces k_q8_fetch(const uint4 *group, int j, bool live)
{
    q8_pieces p;
    const uint4 z = make_uint4(0, 0, 0, 0);
    p.a = live ? k_q8_ld(group + j) : z;
    p.b = live ? k_q8_ld(group + j + 16) : z;
    p.c = live && j < 2 ? k_q8_ld(group + j + 32) : z;
    return p;
}

static __device__ __forceinline__ void k_q8_stage(unsigned char *region, int j, const q8_pieces &p)
{
    uint4 *r = (uint4 *)region;
    r[j] = p.a;
    r[j + 16] = p.b;
    if (j < 2)
        r[j + 32] = p.c;
}

/* One canonical dot product of a Q8_0 row of K elements with the row of x
 * that starts at element xoff, by the half-warp of lane j; every lane of
 * the half-warp gets the result. region: this half-warp's Q8_REGION bytes of
 * shared memory. K a multiple of 512 with 16-byte aligned rows takes the
 * staged path (q8_gemv's, the next group loading while one computes);
 * other K read their blocks 2 bytes at a time. A half-warp with !live
 * computes nothing that matters but takes part in the warp's
 * synchronization and shuffles. */
static __device__ __forceinline__ float k_q8_dot(const uint8_t *row, const void *x, int xdt, int64_t xoff, int64_t K,
                                                 unsigned char *region, int j, bool live)
{
    const int64_t nb = K / ATH_Q8_0_BLOCK;
    float P = 0.0f;
    if (K % 512 == 0) {
        const uint4 *r = (const uint4 *)row;
        const int64_t groups = nb / 16;
        q8_pieces next = k_q8_fetch(r, j, live && groups > 0);
        for (int64_t g = 0; g < groups; g++) {
            __syncwarp();
            k_q8_stage(region, j, next);
            __syncwarp();
            if (g + 1 < groups)
                next = k_q8_fetch(r + (g + 1) * (Q8_GROUP_BYTES / 16), j, live);
            uint32_t nw[9];
            k_q8_words(region, j, nw);
            float xf[32];
            if (live)
                k_q8_x(x, xdt, xoff, g * 16 + j, xf);
            else
                for (int i = 0; i < 32; i++)
                    xf[i] = 0.0f;
            P = fmaf(k_q8_scale(nw), k_q8_chain(nw, xf), P);
        }
    } else {
        const uint16_t *r = (const uint16_t *)row;
        for (int64_t b = j; live && b < nb; b += 16) {
            uint32_t nw[9];
            const uint16_t *blk = r + b * (ATH_Q8_0_BYTES / 2);
#pragma unroll
            for (int i = 0; i < 8; i++)
                nw[i] = (uint32_t)blk[2 * i] | (uint32_t)blk[2 * i + 1] << 16;
            nw[8] = blk[16];
            float xf[32];
#pragma unroll
            for (int i = 0; i < 32; i++)
                xf[i] = k_ld(x, xdt, xoff + b * ATH_Q8_0_BLOCK + i);
            P = fmaf(k_q8_scale(nw), k_q8_chain(nw, xf), P);
        }
    }
    return k_q8_tree(P);
}

#endif /* ATH_OPS_Q8_0_CUH */
