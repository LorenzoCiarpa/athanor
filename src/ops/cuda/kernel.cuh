/* kernel.cuh: what the GPU kernels share. Private to the .cu
 * files of src/ops/cuda/.
 *
 * Elements. Kernels read and write f32 and bf16 through k_ld and k_st, the
 * device twins of ld and st (check.h). The dtype is an argument of the
 * kernel, the same for every thread of a launch, so the branch on it never
 * splits a warp. bf16 rounding is the host's (core/tensor.h), bit for bit:
 * to nearest, ties to even, every NaN to 0x7fc0.
 *
 * Reductions. k_warp_sum adds the 32 values of a warp with a butterfly
 * (__shfl_xor): every lane ends with the same bits, because each step adds
 * the same two numbers on both lanes of a pair and IEEE addition is
 * commutative. k_block_sum adds the warps' totals in warp order. The order
 * of a sum is fixed by the launch shape, so a kernel gives the same bits
 * every time it runs on the same shapes.
 */
#ifndef ATH_OPS_KERNEL_CUH
#define ATH_OPS_KERNEL_CUH

#include "backend/cuda/compat.h"
#include "backend/cuda/gpu.h"
#include "core/tensor.h"
#include "ops/launch.h"

#include <math.h>
#include <stdint.h>

static __device__ __forceinline__ float k_bf16_to_f32(uint16_t h)
{
    return __uint_as_float((uint32_t)h << 16);
}

static __device__ __forceinline__ uint16_t k_f32_to_bf16(float f)
{
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u)              /* NaN */
        return 0x7fc0u;
    u += 0x7fffu + ((u >> 16) & 1u);                  /* round half to even; overflow gives inf */
    return (uint16_t)(u >> 16);
}

/* Element i of a buffer of dtype dt (ATH_DTYPE_F32 or ATH_DTYPE_BF16). */
static __device__ __forceinline__ float k_ld(const void *p, int dt, int64_t i)
{
    if (dt == ATH_DTYPE_BF16)
        return k_bf16_to_f32(((const uint16_t *)p)[i]);
    return ((const float *)p)[i];
}

static __device__ __forceinline__ void k_st(void *p, int dt, int64_t i, float v)
{
    if (dt == ATH_DTYPE_BF16)
        ((uint16_t *)p)[i] = k_f32_to_bf16(v);
    else
        ((float *)p)[i] = v;
}

/* 8 bf16 of one 16-byte load, as floats (element 2i is the low half of
 * word i). */
static __device__ __forceinline__ void k_unpack_bf16x8(uint4 u, float f[8])
{
    const uint32_t v[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int i = 0; i < 4; i++) {
        f[2 * i] = __uint_as_float(v[i] << 16);
        f[2 * i + 1] = __uint_as_float(v[i] & 0xffff0000u);
    }
}

/* Elements 8e … 8e + 7 of a bf16 or f32 buffer on a 16-byte boundary, in
 * one or two 16-byte loads. */
static __device__ __forceinline__ void k_ld8(const void *p, int dt, int64_t e, float f[8])
{
    if (dt == ATH_DTYPE_BF16) {
        k_unpack_bf16x8(((const uint4 *)p)[e], f);
    } else {
        const float4 a = ((const float4 *)p)[2 * e], b = ((const float4 *)p)[2 * e + 1];
        f[0] = a.x, f[1] = a.y, f[2] = a.z, f[3] = a.w, f[4] = b.x, f[5] = b.y, f[6] = b.z, f[7] = b.w;
    }
}

/* Elements 4e … 4e + 3 of a bf16 or f32 buffer on an 8- or 16-byte boundary,
 * in one load. */
static __device__ __forceinline__ void k_ld4(const void *p, int dt, int64_t e, float f[4])
{
    if (dt == ATH_DTYPE_BF16) {
        const uint2 u = ((const uint2 *)p)[e];
        f[0] = __uint_as_float(u.x << 16), f[1] = __uint_as_float(u.x & 0xffff0000u);
        f[2] = __uint_as_float(u.y << 16), f[3] = __uint_as_float(u.y & 0xffff0000u);
    } else {
        const float4 a = ((const float4 *)p)[e];
        f[0] = a.x, f[1] = a.y, f[2] = a.z, f[3] = a.w;
    }
}

/* The value v takes once stored as dt. */
static __device__ __forceinline__ float k_round(int dt, float v)
{
    return dt == ATH_DTYPE_BF16 ? k_bf16_to_f32(k_f32_to_bf16(v)) : v;
}

/* The sum over the 32 lanes of a warp, on every lane. All lanes must call it. */
static __device__ __forceinline__ float k_warp_sum(float v)
{
    for (int m = ATH_WARP / 2; m > 0; m >>= 1)
        v += ath_shfl_xor(v, m);
    return v;
}

/* The sum over the threads of the block, on every thread. blockDim.x is a
 * multiple of 32; every thread must call it; scratch holds 32 floats. */
static __device__ __forceinline__ float k_block_sum(float v, float *scratch)
{
    const int lane = threadIdx.x % ATH_WARP, warp = threadIdx.x / ATH_WARP;
    v = k_warp_sum(v);
    __syncthreads();                     /* scratch may still be read by a previous call */
    if (lane == 0)
        scratch[warp] = v;
    __syncthreads();
    float total = 0.0f;
    for (int w = 0; w < (int)(blockDim.x / ATH_WARP); w++)
        total += scratch[w];
    return total;
}

/* x / (1 + exp(−x)), as act.c writes it. */
static __device__ __forceinline__ float k_silu(float x)
{
    return x / (1.0f + expf(-x));
}

/* Blocks of `threads` for a grid-stride loop over n items: enough to fill
 * the device, and the loop covers the rest. */
static inline unsigned k_blocks(int64_t n, int threads)
{
    const int64_t cap = (int64_t)1 << 16;
    int64_t b = (n + threads - 1) / threads;
    return (unsigned)(b < 1 ? 1 : b > cap ? cap : b);
}

/* The index of this thread in a grid-stride loop, and the stride. */
#define K_FOR(e, n)                                                                       \
    for (int64_t e = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; e < (n);             \
         e += (int64_t)gridDim.x * blockDim.x)

/* The compute stream of d, where every op runs. */
static inline cudaStream_t k_stream(const ath_device *d)
{
    return ath_gpu_stream(d, ATH_STREAM_COMPUTE);
}

#endif /* ATH_OPS_KERNEL_CUH */
