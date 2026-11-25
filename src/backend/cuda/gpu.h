/* gpu.h: what GPU code shares: runtime errors mapped to a status, the
 * device made current, its streams, and the check after a kernel launch.
 *
 * Private to GPU code: included only by .cu files (src/backend/cuda/gpu.cu
 * and the kernels in src/ops/cuda/), never by C code above the backend,
 * which sees backend.h alone.
 */
#ifndef ATH_BACKEND_GPU_H
#define ATH_BACKEND_GPU_H

#include "backend/backend.h"
#include "backend/cuda/compat.h"

/* Log a runtime error and map it to a status. The runtime also remembers
 * the last error until it is read; reading it here keeps a handled error
 * from being reported again by an unrelated later call. */
static inline ath_status ath_gpu_fail(cudaError_t e, const char *what)
{
    cudaGetLastError();
    ath_log(ATH_LOG_ERROR, "gpu: %s: %s", what, cudaGetErrorString(e));
    return e == cudaErrorMemoryAllocation ? ATH_ERR_NOMEM : ATH_ERR_DEVICE;
}

#define ATH_GPU_CHECK(call)                                                   \
    do {                                                                      \
        cudaError_t gpu_e_ = (call);                                          \
        if (gpu_e_ != cudaSuccess)                                            \
            return ath_gpu_fail(gpu_e_, #call);                               \
    } while (0)

/* Make d the calling thread's current device: the runtime keeps one per
 * host thread, and every call works on it. */
static inline ath_status ath_gpu_use(const ath_device *d)
{
    if (!d || !d->stream[0])
        return ATH_ERR_INVALID_ARG;
    ATH_GPU_CHECK(cudaSetDevice(d->index));
    return ATH_OK;
}

static inline cudaStream_t ath_gpu_stream(const ath_device *d, ath_stream s)
{
    return (cudaStream_t)d->stream[s];
}

/* After a kernel launch: an invalid launch (configuration, resources) is
 * reported here. A fault while the kernel runs is reported by a later call
 * that waits for it (a synchronization or a copy to the host). */
static inline ath_status ath_gpu_launched(const char *kernel)
{
    cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? ATH_OK : ath_gpu_fail(e, kernel);
}

#endif /* ATH_BACKEND_GPU_H */
