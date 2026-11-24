/* compat.h: one source for CUDA and HIP.
 *
 * GPU code (src/backend/gpu.cu, and the .cu kernels of src/ops from step 7)
 * is written once against the CUDA runtime API and includes this header
 * instead of a vendor header. nvcc compiles it as CUDA. hipcc compiles the
 * same files for AMD with __HIP_PLATFORM_AMD__ defined, and the names below
 * map every CUDA call in use to its HIP twin, which has the same arguments.
 *
 * Only what the code uses is mapped: a new runtime call in GPU code needs
 * its line here, or the HIP build (step 11) stops with an undefined name.
 * Until step 11 the HIP half is written against the HIP documentation and
 * has not been compiled.
 */
#ifndef ATH_COMPAT_H
#define ATH_COMPAT_H

#if defined(__HIP_PLATFORM_AMD__)

#include <hip/hip_runtime.h>

#define ATH_GPU_BACKEND "hip"

typedef hipError_t     cudaError_t;
typedef hipStream_t    cudaStream_t;
typedef hipEvent_t     cudaEvent_t;
typedef hipDeviceProp_t cudaDeviceProp;

#define cudaSuccess                  hipSuccess
#define cudaErrorMemoryAllocation    hipErrorOutOfMemory
#define cudaErrorNoDevice            hipErrorNoDevice
#define cudaErrorInsufficientDriver  hipErrorInsufficientDriver
#define cudaMemcpyDefault            hipMemcpyDefault
#define cudaStreamNonBlocking        hipStreamNonBlocking
#define cudaEventDefault             hipEventDefault
#define cudaEventDisableTiming       hipEventDisableTiming
#define cudaHostAllocDefault         hipHostMallocDefault
#define cudaDevAttrMaxSharedMemoryPerBlockOptin hipDeviceAttributeSharedMemPerBlockOptin

#define cudaGetErrorString           hipGetErrorString
#define cudaGetLastError             hipGetLastError
#define cudaGetDeviceCount           hipGetDeviceCount
#define cudaGetDeviceProperties      hipGetDeviceProperties
#define cudaDeviceGetAttribute       hipDeviceGetAttribute
#define cudaSetDevice                hipSetDevice
#define cudaDriverGetVersion         hipDriverGetVersion
#define cudaRuntimeGetVersion        hipRuntimeGetVersion
#define cudaMemGetInfo               hipMemGetInfo
#define cudaMalloc                   hipMalloc
#define cudaFree                     hipFree
#define cudaHostAlloc                hipHostMalloc
#define cudaFreeHost                 hipHostFree
#define cudaMemcpyAsync              hipMemcpyAsync
#define cudaMemsetAsync              hipMemsetAsync
#define cudaStreamCreateWithFlags    hipStreamCreateWithFlags
#define cudaStreamDestroy            hipStreamDestroy
#define cudaStreamSynchronize        hipStreamSynchronize
#define cudaStreamWaitEvent          hipStreamWaitEvent
#define cudaDeviceSynchronize        hipDeviceSynchronize
#define cudaEventCreateWithFlags     hipEventCreateWithFlags
#define cudaEventDestroy             hipEventDestroy
#define cudaEventRecord              hipEventRecord
#define cudaEventSynchronize         hipEventSynchronize
#define cudaEventElapsedTime         hipEventElapsedTime

/* Threads per wave. RDNA runs kernels in wave32 mode, the size the kernels
 * are written for; CDNA (wave64) is not a target. */
#define ATH_WARP 32

#else

#include <cuda_runtime.h>

#define ATH_GPU_BACKEND "cuda"
#define ATH_WARP 32

#endif

#endif /* ATH_COMPAT_H */
