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
#include "backend/cuda/compat.h"
#include "backend/cuda/gpu.h"

#include <stdio.h>
#include <string.h>

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
    ATH_GPU_CHECK(cudaSetDevice(index));
    cudaDeviceProp p;
    ATH_GPU_CHECK(cudaGetDeviceProperties(&p, index));
    int smem_optin = 0, drv = 0, rt = 0;
    ATH_GPU_CHECK(cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, index));
    ATH_GPU_CHECK(cudaDriverGetVersion(&drv));
    ATH_GPU_CHECK(cudaRuntimeGetVersion(&rt));

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
    ATH_GPU_CHECK(cudaDeviceGetPCIBusId(d->pci_bus_id, (int)sizeof d->pci_bus_id, index));
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
            return ath_gpu_fail(e, "cudaStreamCreateWithFlags");
        }
        d->stream[s] = st;
    }
    /* the matmul library: a handle per device and its workspace */
    cublasLtHandle_t h;
    cublasStatus_t bs = cublasLtCreate(&h);
    if (bs != CUBLAS_STATUS_SUCCESS) {
        ath_log(ATH_LOG_ERROR, "gpu: cublasLtCreate: %s", cublasLtGetStatusString(bs));
        ath_device_close(d);
        return ATH_ERR_DEVICE;
    }
    d->blas = h;
#if defined(__HIP_PLATFORM_AMD__)
    int bv = 0;
    hipblasLtGetVersion(h, &bv);
    d->blas_version = bv;
#else
    d->blas_version = (int32_t)cublasLtGetVersion();
#endif
    cudaError_t e = cudaMalloc(&d->blas_ws, ATH_BLAS_WORKSPACE);
    if (e != cudaSuccess) {
        d->blas_ws = NULL;
        ath_device_close(d);
        return ath_gpu_fail(e, "cudaMalloc (matmul workspace)");
    }
    d->blas_ws_bytes = ATH_BLAS_WORKSPACE;
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
            cudaStreamSynchronize(ath_gpu_stream(d, (ath_stream)s));
            cudaStreamDestroy(ath_gpu_stream(d, (ath_stream)s));
        }
        if (d->blas_ws)
            cudaFree(d->blas_ws);
        if (d->blas)
            cublasLtDestroy((cublasLtHandle_t)d->blas);
    }
    cudaGetLastError();
    memset(d, 0, sizeof *d);
}

ath_status ath_device_mem(const ath_device *d, uint64_t *free_bytes, uint64_t *total_bytes)
{
    ATH_CHECK(ath_gpu_use(d));
    size_t f = 0, t = 0;
    ATH_GPU_CHECK(cudaMemGetInfo(&f, &t));
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
    ATH_CHECK(ath_gpu_use(d));
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
    if (!ptr || ath_gpu_use(d) != ATH_OK)
        return;
    cudaError_t e = cudaFree(ptr);
    if (e != cudaSuccess)
        ath_gpu_fail(e, "cudaFree");
}

ath_status ath_pinned_alloc(const ath_device *d, size_t bytes, void **ptr)
{
    if (!ptr)
        return ATH_ERR_INVALID_ARG;
    *ptr = NULL;
    ATH_CHECK(ath_gpu_use(d));
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
    if (!ptr || ath_gpu_use(d) != ATH_OK)
        return;
    cudaError_t e = cudaFreeHost(ptr);
    if (e != cudaSuccess)
        ath_gpu_fail(e, "cudaFreeHost");
}

/* ---- copies ----------------------------------------------------------------- */

static bool stream_ok(ath_stream s) { return (unsigned)s < ATH_STREAM_COUNT; }

ath_status ath_copy(const ath_device *d, void *dst, const void *src, size_t bytes, ath_stream s)
{
    if (!stream_ok(s) || (bytes && (!dst || !src)))
        return ATH_ERR_INVALID_ARG;
    ATH_CHECK(ath_gpu_use(d));
    if (!bytes)
        return ATH_OK;
    ATH_GPU_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, ath_gpu_stream(d, s)));
    return ATH_OK;
}

ath_status ath_copy_2d(const ath_device *d, void *dst, size_t dpitch, const void *src, size_t spitch, size_t width,
                       size_t rows, ath_stream s)
{
    if (!stream_ok(s) || dpitch < width || spitch < width || (width && rows && (!dst || !src)))
        return ATH_ERR_INVALID_ARG;
    ATH_CHECK(ath_gpu_use(d));
    if (!width || !rows)
        return ATH_OK;
    ATH_GPU_CHECK(cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, rows, cudaMemcpyDefault, ath_gpu_stream(d, s)));
    return ATH_OK;
}

ath_status ath_memset(const ath_device *d, void *dst, int value, size_t bytes, ath_stream s)
{
    if (!stream_ok(s) || (bytes && !dst))
        return ATH_ERR_INVALID_ARG;
    ATH_CHECK(ath_gpu_use(d));
    if (!bytes)
        return ATH_OK;
    ATH_GPU_CHECK(cudaMemsetAsync(dst, value, bytes, ath_gpu_stream(d, s)));
    return ATH_OK;
}

ath_status ath_stream_sync(const ath_device *d, ath_stream s)
{
    if (!stream_ok(s))
        return ATH_ERR_INVALID_ARG;
    ATH_CHECK(ath_gpu_use(d));
    ATH_GPU_CHECK(cudaStreamSynchronize(ath_gpu_stream(d, s)));
    return ATH_OK;
}

/* ---- graphs ------------------------------------------------------------------ */

ath_status ath_graph_begin(const ath_device *d, ath_stream s)
{
    if (!stream_ok(s))
        return ATH_ERR_INVALID_ARG;
    ATH_CHECK(ath_gpu_use(d));
    /* thread-local: only this thread's calls are checked against the capture */
    ATH_GPU_CHECK(cudaStreamBeginCapture(ath_gpu_stream(d, s), cudaStreamCaptureModeThreadLocal));
    return ATH_OK;
}

ath_status ath_graph_end(const ath_device *d, ath_stream s, ath_graph *g)
{
    if (!stream_ok(s) || !g)
        return ATH_ERR_INVALID_ARG;
    memset(g, 0, sizeof *g);
    ATH_CHECK(ath_gpu_use(d));
    cudaGraph_t graph = NULL;
    cudaError_t e = cudaStreamEndCapture(ath_gpu_stream(d, s), &graph);
    if (e != cudaSuccess) {
        if (graph)
            cudaGraphDestroy(graph);
        return ath_gpu_fail(e, "cudaStreamEndCapture");
    }
    if (!graph)
        return ATH_OK;
    cudaGraphExec_t exec = NULL;
    e = cudaGraphInstantiateWithFlags(&exec, graph, 0);
    if (e != cudaSuccess) {
        cudaGraphDestroy(graph);
        return ath_gpu_fail(e, "cudaGraphInstantiateWithFlags");
    }
    g->graph = graph;
    g->exec = exec;
    return ATH_OK;
}

ath_status ath_graph_launch(const ath_device *d, const ath_graph *g, ath_stream s)
{
    if (!stream_ok(s) || !g || !g->exec)
        return ATH_ERR_INVALID_ARG;
    ATH_CHECK(ath_gpu_use(d));
    ATH_GPU_CHECK(cudaGraphLaunch((cudaGraphExec_t)g->exec, ath_gpu_stream(d, s)));
    return ATH_OK;
}

void ath_graph_destroy(const ath_device *d, ath_graph *g)
{
    if (!g)
        return;
    if (d && (g->exec || g->graph) && ath_gpu_use(d) == ATH_OK) {
        if (g->exec)
            cudaGraphExecDestroy((cudaGraphExec_t)g->exec);
        if (g->graph)
            cudaGraphDestroy((cudaGraph_t)g->graph);
    }
    memset(g, 0, sizeof *g);
}

ath_status ath_device_sync(const ath_device *d)
{
    ATH_CHECK(ath_gpu_use(d));
    ATH_GPU_CHECK(cudaDeviceSynchronize());
    return ATH_OK;
}

/* ---- events ----------------------------------------------------------------- */

ath_status ath_event_create(const ath_device *d, ath_event *e, bool timing)
{
    if (!e)
        return ATH_ERR_INVALID_ARG;
    e->handle = NULL;
    e->timing = timing;
    ATH_CHECK(ath_gpu_use(d));
    cudaEvent_t ev;
    ATH_GPU_CHECK(cudaEventCreateWithFlags(&ev, timing ? cudaEventDefault : cudaEventDisableTiming));
    e->handle = ev;
    return ATH_OK;
}

void ath_event_destroy(const ath_device *d, ath_event *e)
{
    if (!e || !e->handle || ath_gpu_use(d) != ATH_OK)
        return;
    cudaEventDestroy((cudaEvent_t)e->handle);
    cudaGetLastError();
    e->handle = NULL;
}

ath_status ath_event_record(const ath_device *d, ath_event *e, ath_stream s)
{
    if (!e || !e->handle || !stream_ok(s))
        return ATH_ERR_INVALID_ARG;
    ATH_CHECK(ath_gpu_use(d));
    ATH_GPU_CHECK(cudaEventRecord((cudaEvent_t)e->handle, ath_gpu_stream(d, s)));
    return ATH_OK;
}

ath_status ath_stream_wait(const ath_device *d, ath_stream s, const ath_event *e)
{
    if (!e || !e->handle || !stream_ok(s))
        return ATH_ERR_INVALID_ARG;
    ATH_CHECK(ath_gpu_use(d));
    ATH_GPU_CHECK(cudaStreamWaitEvent(ath_gpu_stream(d, s), (cudaEvent_t)e->handle, 0));
    return ATH_OK;
}

ath_status ath_host_call(const ath_device *d, ath_stream s, ath_host_fn fn, void *ctx)
{
    if (!fn || !stream_ok(s))
        return ATH_ERR_INVALID_ARG;
    ATH_CHECK(ath_gpu_use(d));
    ATH_GPU_CHECK(cudaLaunchHostFunc(ath_gpu_stream(d, s), (cudaHostFn_t)fn, ctx));
    return ATH_OK;
}

ath_status ath_stream_capturing(const ath_device *d, ath_stream s, bool *capturing)
{
    if (!capturing || !stream_ok(s))
        return ATH_ERR_INVALID_ARG;
    ATH_CHECK(ath_gpu_use(d));
    cudaStreamCaptureStatus st;
    ATH_GPU_CHECK(cudaStreamIsCapturing(ath_gpu_stream(d, s), &st));
    *capturing = st != cudaStreamCaptureStatusNone;
    return ATH_OK;
}

ath_status ath_event_sync(const ath_device *d, const ath_event *e)
{
    if (!e || !e->handle)
        return ATH_ERR_INVALID_ARG;
    ATH_CHECK(ath_gpu_use(d));
    ATH_GPU_CHECK(cudaEventSynchronize((cudaEvent_t)e->handle));
    return ATH_OK;
}

ath_status ath_event_elapsed(const ath_device *d, const ath_event *start, const ath_event *end, float *ms)
{
    if (!start || !end || !ms || !start->handle || !end->handle || !start->timing || !end->timing)
        return ATH_ERR_INVALID_ARG;
    ATH_CHECK(ath_gpu_use(d));
    ATH_GPU_CHECK(cudaEventElapsedTime(ms, (cudaEvent_t)start->handle, (cudaEvent_t)end->handle));
    return ATH_OK;
}

} /* extern "C" */
