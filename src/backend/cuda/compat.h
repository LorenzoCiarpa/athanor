/* compat.h: one source for CUDA and HIP.
 *
 * GPU code (src/backend/cuda/gpu.cu, and the kernels in src/ops/cuda/)
 * is written once against the CUDA runtime API and cuBLASLt, and includes
 * this header instead of a vendor header. nvcc compiles it as CUDA. hipcc
 * compiles the same files for AMD with __HIP_PLATFORM_AMD__ defined, and the
 * names below map every CUDA call in use to its HIP twin (hipBLASLt for
 * cuBLASLt), which has the same arguments.
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
#include <hipblaslt/hipblaslt.h>

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
#define cudaDeviceGetPCIBusId        hipDeviceGetPCIBusId
#define cudaSetDevice                hipSetDevice
#define cudaDriverGetVersion         hipDriverGetVersion
#define cudaRuntimeGetVersion        hipRuntimeGetVersion
#define cudaMemGetInfo               hipMemGetInfo
#define cudaMalloc                   hipMalloc
#define cudaFree                     hipFree
#define cudaHostAlloc                hipHostMalloc
#define cudaFreeHost                 hipHostFree
#define cudaMemcpyAsync              hipMemcpyAsync
#define cudaMemcpy2DAsync            hipMemcpy2DAsync
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
#define cudaGraph_t                  hipGraph_t
#define cudaGraphExec_t              hipGraphExec_t
#define cudaStreamBeginCapture       hipStreamBeginCapture
#define cudaStreamEndCapture         hipStreamEndCapture
#define cudaStreamCaptureModeThreadLocal hipStreamCaptureModeThreadLocal
#define cudaGraphInstantiateWithFlags hipGraphInstantiateWithFlags
#define cudaGraphLaunch              hipGraphLaunch
#define cudaGraphExecDestroy         hipGraphExecDestroy
#define cudaGraphDestroy             hipGraphDestroy

/* the matmul library */
typedef hipblasLtHandle_t                 cublasLtHandle_t;
typedef hipblasStatus_t                   cublasStatus_t;
typedef hipblasOperation_t                cublasOperation_t;
typedef hipDataType                       cudaDataType_t;
typedef hipblasLtMatmulDesc_t             cublasLtMatmulDesc_t;
typedef hipblasLtMatrixLayout_t           cublasLtMatrixLayout_t;
typedef hipblasLtMatmulPreference_t       cublasLtMatmulPreference_t;
typedef hipblasLtMatmulHeuristicResult_t  cublasLtMatmulHeuristicResult_t;

#define CUBLAS_STATUS_SUCCESS                     HIPBLAS_STATUS_SUCCESS
#define CUBLAS_COMPUTE_32F                        HIPBLAS_COMPUTE_32F
#define CUBLAS_OP_N                               HIPBLAS_OP_N
#define CUBLAS_OP_T                               HIPBLAS_OP_T
#define CUDA_R_32F                                HIP_R_32F
#define CUDA_R_16BF                               HIP_R_16BF
#define CUBLASLT_MATMUL_DESC_TRANSA               HIPBLASLT_MATMUL_DESC_TRANSA
#define CUBLASLT_MATMUL_DESC_TRANSB               HIPBLASLT_MATMUL_DESC_TRANSB
#define CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES  HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES

#define cublasLtCreate                    hipblasLtCreate
#define cublasLtDestroy                   hipblasLtDestroy
#define cublasLtMatmulDescCreate          hipblasLtMatmulDescCreate
#define cublasLtMatmulDescDestroy         hipblasLtMatmulDescDestroy
#define cublasLtMatmulDescSetAttribute    hipblasLtMatmulDescSetAttribute
#define cublasLtMatrixLayoutCreate        hipblasLtMatrixLayoutCreate
#define cublasLtMatrixLayoutDestroy       hipblasLtMatrixLayoutDestroy
#define cublasLtMatmulPreferenceCreate    hipblasLtMatmulPreferenceCreate
#define cublasLtMatmulPreferenceDestroy   hipblasLtMatmulPreferenceDestroy
#define cublasLtMatmulPreferenceSetAttribute hipblasLtMatmulPreferenceSetAttribute
#define cublasLtMatmulAlgoGetHeuristic    hipblasLtMatmulAlgoGetHeuristic
#define cublasLtMatmul                    hipblasLtMatmul
#define cublasLtGetStatusString           hipblasStatusToString

/* Threads per wave. RDNA runs kernels in wave32 mode, the size the kernels
 * are written for; CDNA (wave64) is not a target. */
#define ATH_WARP 32

/* Exchange between the lanes of a wave (all lanes take part). */
#define ath_shfl_xor(v, mask) __shfl_xor((v), (mask))

#else

#include <cublasLt.h>
#include <cuda_runtime.h>

#define ATH_GPU_BACKEND "cuda"
#define ATH_WARP 32

/* Exchange between the lanes of a warp (all lanes take part). */
#define ath_shfl_xor(v, mask) __shfl_xor_sync(0xffffffffu, (v), (mask))

#endif

#endif /* ATH_COMPAT_H */
