/* gpu.cu: the GPU runtime behind backend.h, for CUDA (and HIP through
 * compat.h from step 11).
 *
 * Written in the C subset of CUDA C++ (AGENTS.md: no classes, templates or
 * exceptions; compiled with -fno-exceptions -fno-rtti), with C linkage for
 * every function of backend.h. Every call first selects the device the
 * handle names, because the runtime keeps a "current device" per host
 * thread: two devices used from one thread must not see each other's
 * state.
 */
#include "backend/backend.h"
#include "backend/compat.h"

#include <stdio.h>
#include <string.h>

/* Log a runtime error and map it to a status. The runtime also remembers
 * the last error until it is read; reading it here keeps a handled error
 * from being reported again by an unrelated later call. */
static ath_status fail(cudaError_t e, const char *what)
{
    cudaGetLastError();
    ath_log(ATH_LOG_ERROR, "gpu: %s: %s", what, cudaGetErrorString(e));
    return e == cudaErrorMemoryAllocation ? ATH_ERR_NOMEM : ATH_ERR_DEVICE;
}

#define GPU_CHECK(call)                                                       \
    do {                                                                      \
        cudaError_t gpu_e_ = (call);                                          \
        if (gpu_e_ != cudaSuccess)                                            \
            return fail(gpu_e_, #call);                                       \
    } while (0)

static ath_status use(const ath_device *d)
{
    if (!d || !d->stream[0])
        return ATH_ERR_INVALID_ARG;
    GPU_CHECK(cudaSetDevice(d->index));
    return ATH_OK;
}

static cudaStream_t stream_of(const ath_device *d, ath_stream s)
{
    return (cudaStream_t)d->stream[s];
}

extern "C" {

const char *ath_gpu_backend(void) { return ATH_GPU_BACKEND; }

int32_t ath_gpu_count(void)
{
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess) {
        cudaGetLastError();
        ath_log(e == cudaErrorNoDevice ? ATH_LOG_INFO : ATH_LOG_WARN, "gpu: no usable device: %s",
                cudaGetErrorString(e));
        return 0;
    }
    return n;
}

ath_status ath_device_open(ath_device *d, int32_t index)
{
    if (!d)
        return ATH_ERR_INVALID_ARG;
    memset(d, 0, sizeof *d);
    int32_t n = ath_gpu_count();
    if (index < 0 || index >= n) {
        ath_log(ATH_LOG_ERROR, "gpu: no device %d (%d available)", index, n);
        return ATH_ERR_INVALID_ARG;
    }
    GPU_CHECK(cudaSetDevice(index));
    cudaDeviceProp p;
    GPU_CHECK(cudaGetDeviceProperties(&p, index));
    int smem_optin = 0, drv = 0, rt = 0;
    GPU_CHECK(cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, index));
    GPU_CHECK(cudaDriverGetVersion(&drv));
    GPU_CHECK(cudaRuntimeGetVersion(&rt));

    d->index = index;
    snprintf(d->name, sizeof d->name, "%s", p.name);
    d->cc_major = p.major;
    d->cc_minor = p.minor;
    d->n_sm = p.multiProcessorCount;
    d->warp = p.warpSize;
    d->mem_total = p.totalGlobalMem;
    d->smem_per_block = (uint64_t)smem_optin;
    d->l2_bytes = (uint64_t)p.l2CacheSize;
    d->driver_version = drv;
    d->runtime_version = rt;
    if (d->warp != ATH_WARP) {
        ath_log(ATH_LOG_ERROR, "gpu: %s runs %d threads per warp; the kernels are written for %d", d->name,
                d->warp, ATH_WARP);
        return ATH_ERR_UNSUPPORTED;
    }
    for (int s = 0; s < ATH_STREAM_COUNT; s++) {
        cudaStream_t st;
        cudaError_t e = cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
        if (e != cudaSuccess) {
            ath_device_close(d);
            return fail(e, "cudaStreamCreateWithFlags");
        }
        d->stream[s] = st;
    }
    return ATH_OK;
}

void ath_device_close(ath_device *d)
{
    if (!d || !d->stream[0]) {
        if (d)
            memset(d, 0, sizeof *d);
        return;
    }
    if (cudaSetDevice(d->index) == cudaSuccess) {
        for (int s = 0; s < ATH_STREAM_COUNT; s++) {
            if (!d->stream[s])
                continue;
            cudaStreamSynchronize(stream_of(d, (ath_stream)s));
            cudaStreamDestroy(stream_of(d, (ath_stream)s));
        }
    }
    cudaGetLastError();
    memset(d, 0, sizeof *d);
}

ath_status ath_device_mem(const ath_device *d, uint64_t *free_bytes, uint64_t *total_bytes)
{
    ATH_CHECK(use(d));
    size_t f = 0, t = 0;
    GPU_CHECK(cudaMemGetInfo(&f, &t));
    if (free_bytes)
        *free_bytes = f;
    if (total_bytes)
        *total_bytes = t;
    return ATH_OK;
}

/* ---- memory ---------------------------------------------------------------- */

ath_status ath_dev_alloc(const ath_device *d, size_t bytes, void **ptr)
{
    if (!ptr)
        return ATH_ERR_INVALID_ARG;
    *ptr = NULL;
    ATH_CHECK(use(d));
    if (!bytes)
        return ATH_OK;
    cudaError_t e = cudaMalloc(ptr, bytes);
    if (e != cudaSuccess) {
        *ptr = NULL;
        cudaGetLastError();
        ath_log(e == cudaErrorMemoryAllocation ? ATH_LOG_WARN : ATH_LOG_ERROR, "gpu: cannot allocate %zu bytes on %s: %s",
                bytes, d->name, cudaGetErrorString(e));
        return e == cudaErrorMemoryAllocation ? ATH_ERR_NOMEM : ATH_ERR_DEVICE;
    }
    return ATH_OK;
}

void ath_dev_free(const ath_device *d, void *ptr)
{
    if (!ptr || use(d) != ATH_OK)
        return;
    cudaError_t e = cudaFree(ptr);
    if (e != cudaSuccess)
        fail(e, "cudaFree");
}

ath_status ath_pinned_alloc(const ath_device *d, size_t bytes, void **ptr)
{
    if (!ptr)
        return ATH_ERR_INVALID_ARG;
    *ptr = NULL;
    ATH_CHECK(use(d));
    if (!bytes)
        return ATH_OK;
    cudaError_t e = cudaHostAlloc(ptr, bytes, cudaHostAllocDefault);
    if (e != cudaSuccess) {
        *ptr = NULL;
        cudaGetLastError();
        ath_log(ATH_LOG_WARN, "gpu: cannot pin %zu bytes of host memory: %s", bytes, cudaGetErrorString(e));
        return e == cudaErrorMemoryAllocation ? ATH_ERR_NOMEM : ATH_ERR_DEVICE;
    }
    return ATH_OK;
}

void ath_pinned_free(const ath_device *d, void *ptr)
{
    if (!ptr || use(d) != ATH_OK)
        return;
    cudaError_t e = cudaFreeHost(ptr);
    if (e != cudaSuccess)
        fail(e, "cudaFreeHost");
}

/* ---- copies ----------------------------------------------------------------- */

static bool stream_ok(ath_stream s) { return (unsigned)s < ATH_STREAM_COUNT; }

ath_status ath_copy(const ath_device *d, void *dst, const void *src, size_t bytes, ath_stream s)
{
    if (!stream_ok(s) || (bytes && (!dst || !src)))
        return ATH_ERR_INVALID_ARG;
    ATH_CHECK(use(d));
    if (!bytes)
        return ATH_OK;
    GPU_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, stream_of(d, s)));
    return ATH_OK;
}

ath_status ath_memset(const ath_device *d, void *dst, int value, size_t bytes, ath_stream s)
{
    if (!stream_ok(s) || (bytes && !dst))
        return ATH_ERR_INVALID_ARG;
    ATH_CHECK(use(d));
    if (!bytes)
        return ATH_OK;
    GPU_CHECK(cudaMemsetAsync(dst, value, bytes, stream_of(d, s)));
    return ATH_OK;
}

ath_status ath_stream_sync(const ath_device *d, ath_stream s)
{
    if (!stream_ok(s))
        return ATH_ERR_INVALID_ARG;
    ATH_CHECK(use(d));
    GPU_CHECK(cudaStreamSynchronize(stream_of(d, s)));
    return ATH_OK;
}

ath_status ath_device_sync(const ath_device *d)
{
    ATH_CHECK(use(d));
    GPU_CHECK(cudaDeviceSynchronize());
    return ATH_OK;
}

/* ---- events ----------------------------------------------------------------- */

ath_status ath_event_create(const ath_device *d, ath_event *e, bool timing)
{
    if (!e)
        return ATH_ERR_INVALID_ARG;
    e->handle = NULL;
    e->timing = timing;
    ATH_CHECK(use(d));
    cudaEvent_t ev;
    GPU_CHECK(cudaEventCreateWithFlags(&ev, timing ? cudaEventDefault : cudaEventDisableTiming));
    e->handle = ev;
    return ATH_OK;
}

void ath_event_destroy(const ath_device *d, ath_event *e)
{
    if (!e || !e->handle || use(d) != ATH_OK)
        return;
    cudaEventDestroy((cudaEvent_t)e->handle);
    cudaGetLastError();
    e->handle = NULL;
}

ath_status ath_event_record(const ath_device *d, ath_event *e, ath_stream s)
{
    if (!e || !e->handle || !stream_ok(s))
        return ATH_ERR_INVALID_ARG;
    ATH_CHECK(use(d));
    GPU_CHECK(cudaEventRecord((cudaEvent_t)e->handle, stream_of(d, s)));
    return ATH_OK;
}

ath_status ath_stream_wait(const ath_device *d, ath_stream s, const ath_event *e)
{
    if (!e || !e->handle || !stream_ok(s))
        return ATH_ERR_INVALID_ARG;
    ATH_CHECK(use(d));
    GPU_CHECK(cudaStreamWaitEvent(stream_of(d, s), (cudaEvent_t)e->handle, 0));
    return ATH_OK;
}

ath_status ath_event_sync(const ath_device *d, const ath_event *e)
{
    if (!e || !e->handle)
        return ATH_ERR_INVALID_ARG;
    ATH_CHECK(use(d));
    GPU_CHECK(cudaEventSynchronize((cudaEvent_t)e->handle));
    return ATH_OK;
}

ath_status ath_event_elapsed(const ath_device *d, const ath_event *start, const ath_event *end, float *ms)
{
    if (!start || !end || !ms || !start->handle || !end->handle || !start->timing || !end->timing)
        return ATH_ERR_INVALID_ARG;
    ATH_CHECK(use(d));
    GPU_CHECK(cudaEventElapsedTime(ms, (cudaEvent_t)start->handle, (cudaEvent_t)end->handle));
    return ATH_OK;
}

} /* extern "C" */
