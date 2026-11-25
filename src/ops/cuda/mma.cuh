/* mma.cuh: tensor-core building blocks (step 10). Private to the .cu files
 * of src/ops/cuda/.
 *
 * NVIDIA's warp-level matrix instructions, written in PTX:
 *
 *   mma.sync   one warp multiplies a 16×16 tile of bf16 by a 16×8 tile of
 *              bf16 and adds the product to a 16×8 tile of fp32, on the
 *              tensor cores (products exact, sums in fp32)
 *   ldmatrix   loads 8×8 tiles of 16-bit elements from shared memory into
 *              the registers each lane must hold for mma, transposed or not
 *   cp.async   copies 16 bytes from global to shared memory without passing
 *              through registers, so that a tile can load while the tensor
 *              cores work on the previous one
 *
 * They exist from sm_80 (Ampere) on; ATH_HAVE_MMA says whether this build
 * has them. AMD GPUs have other matrix instructions (MFMA); under HIP the
 * kernels that use these are not compiled and the ops keep their other
 * kernels (step 11 decides the AMD form).
 *
 * Fragments of mma.m16n8k16 (PTX ISA, "Matrix fragments for mma.m16n8k16"),
 * for lane l, g = l / 4 and c = 2 (l % 4), two 16-bit elements per register:
 *
 *   A 16×16, a[0..3]   rows g, g + 8, g, g + 8; columns c, c + 1 (a[0], a[1])
 *                      and c + 8, c + 9 (a[2], a[3])
 *   B 16×8,  b[0..1]   rows (k) c, c + 1 (b[0]) and c + 8, c + 9 (b[1]); column g
 *   C 16×8,  c[0..3]   fp32: (g, c), (g, c + 1), (g + 8, c), (g + 8, c + 1)
 *
 * ldmatrix .x4 loads four 8×8 tiles; lanes 8i … 8i + 7 give the addresses of
 * the 8 rows (16 bytes each) of tile i, and register i receives tile i: lane
 * l gets row l / 4, elements 2 (l % 4) and 2 (l % 4) + 1, or with .trans the
 * same of the transposed tile.
 */
#ifndef ATH_OPS_MMA_CUH
#define ATH_OPS_MMA_CUH

#include "ops/cuda/kernel.cuh"

#if !defined(__HIP_PLATFORM_AMD__)
#define ATH_HAVE_MMA 1

/* The instructions need sm_80; a build for an older GPU compiles the
 * kernels to a trap, and the ops' fits functions keep them from running
 * there (compute capability below 8). */
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
#define ATH_MMA_ARCH 1
#endif

static __device__ __forceinline__ uint32_t k_smem(const void *p)
{
    return (uint32_t)__cvta_generic_to_shared(p);
}

/* 16 bytes from global memory g to shared memory s, asynchronously; zeros
 * instead when !ok (g must still be a valid address). */
static __device__ __forceinline__ void k_cp16(void *s, const void *g, bool ok)
{
#ifdef ATH_MMA_ARCH
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(k_smem(s)), "l"(g), "r"(ok ? 16 : 0));
#endif
}

/* Close the group of copies issued so far; wait until every group is done
 * (the caller then synchronizes the block before reading what others
 * copied). */
static __device__ __forceinline__ void k_cp_commit(void)
{
#ifdef ATH_MMA_ARCH
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

static __device__ __forceinline__ void k_cp_wait_all(void)
{
#ifdef ATH_MMA_ARCH
    asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}

static __device__ __forceinline__ void k_ldsm4(uint32_t r[4], const void *p)
{
#ifdef ATH_MMA_ARCH
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(k_smem(p)));
#endif
}

static __device__ __forceinline__ void k_ldsm4_t(uint32_t r[4], const void *p)
{
#ifdef ATH_MMA_ARCH
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(k_smem(p)));
#endif
}

/* c += a · b: A 16×16 bf16, B 16×8 bf16, C 16×8 fp32. */
static __device__ __forceinline__ void k_mma_bf16(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1)
{
#ifdef ATH_MMA_ARCH
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
                 "{%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    __trap();
#endif
}

/* Two floats rounded to bf16 as k_f32_to_bf16 does, lo in the low half. */
static __device__ __forceinline__ uint32_t k_pack_bf16(float lo, float hi)
{
    return (uint32_t)k_f32_to_bf16(lo) | (uint32_t)k_f32_to_bf16(hi) << 16;
}

#endif /* !__HIP_PLATFORM_AMD__ */

#endif /* ATH_OPS_MMA_CUH */
