/* backend.h: the GPU runtime, as athanor uses it.
 *
 * One C interface over the vendor runtime (CUDA now, HIP from step 11
 * through compat.h): devices, device and pinned host memory, copies, streams
 * and events. Code above the backend never includes a vendor header; it
 * holds opaque handles and passes the device explicitly (no globals,
 * AGENTS.md).
 *
 * A build without a GPU backend (`make`, the default) links gpu_none.c:
 * ath_gpu_count() is 0 and every other function fails with
 * ATH_ERR_UNSUPPORTED, so the same code compiles everywhere and decides at
 * run time.
 *
 * Streams. Each open device has three streams, one per kind of work, so
 * that copies overlap computation (src/backend/README.md):
 *
 *   ATH_STREAM_COMPUTE  kernels (every GPU op of src/ops) and device-to-device copies
 *   ATH_STREAM_H2D      host-to-device copies (weights, prefetched experts)
 *   ATH_STREAM_D2H      device-to-host copies (logits, captured activations)
 *
 * Work on one stream runs in order; work on different streams is ordered
 * only by events (ath_event_record, then ath_stream_wait), never by
 * synchronizing the whole device. Every copy and memset here is
 * asynchronous with respect to the host: the host must not touch the
 * memory until the stream or an event recorded after the work completes.
 * Copies from pageable (non-pinned) host memory are a special case: the
 * runtime stages them, and they may complete before the call returns.
 * Pinned memory (ath_pinned_alloc) is what makes host copies truly
 * asynchronous and fast.
 *
 * Errors from the runtime become ATH_ERR_DEVICE (ATH_ERR_NOMEM when memory
 * runs out), with the runtime's message logged. A failed asynchronous
 * operation may be reported by a later call on the same device.
 */
#ifndef ATH_BACKEND_H
#define ATH_BACKEND_H

#include "core/arena.h"
#include "core/base.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ATH_STREAM_COMPUTE,
    ATH_STREAM_H2D,
    ATH_STREAM_D2H,
    ATH_STREAM_COUNT
} ath_stream;

/* An open device. The fields are filled by ath_device_open and read-only
 * afterwards; `stream` holds the runtime's stream handles. */
typedef struct {
    int32_t  index;                  /* runtime device index                          */
    char     name[256];
    int32_t  cc_major, cc_minor;     /* compute capability: 12.0 for sm_120           */
    int32_t  n_sm;                   /* multiprocessors                               */
    int32_t  warp;                   /* threads per warp (wave)                       */
    uint64_t mem_total;              /* device memory, bytes                          */
    uint64_t smem_per_block;         /* shared memory a block may use (opt-in max)    */
    uint64_t l2_bytes;
    int32_t  driver_version;         /* 13000 for CUDA 13.0                           */
    int32_t  runtime_version;        /* the runtime this binary was built with         */
    char     pci_bus_id[32];         /* domain:bus:device.function, "0000:01:00.0"     */
    int32_t  blas_version;           /* the matmul library's own version number        */
    void    *stream[ATH_STREAM_COUNT];
    /* The vendor's dense matmul library (cuBLASLt; hipBLASLt from step 11)
     * and its workspace, used by the GPU matmul (src/ops/cuda/matmul.cu). One
     * workspace per device is enough because ops run on ATH_STREAM_COMPUTE
     * only, one after the other. */
    void    *blas;
    void    *blas_ws;
    size_t   blas_ws_bytes;
} ath_device;

/* Workspace of the matmul library: what cuBLASLt's documentation
 * recommends for Hopper and later, so that it can choose any algorithm. */
#define ATH_BLAS_WORKSPACE ((size_t)32 << 20)

/* An event: a point in a stream's work, used to order streams and to time
 * work. */
typedef struct {
    void *handle;
    bool  timing;
} ath_event;

/* "cuda", "hip", or "none" for a build without a GPU backend. */
const char *ath_gpu_backend(void);

/* Number of usable devices; 0 without a GPU backend, without a driver or
 * without a device (the reason is logged at info level). */
int32_t ath_gpu_count(void);

/* Open device `index`: query its properties, create its streams, the matmul
 * library's handle and its workspace. ATH_ERR_INVALID_ARG for an index out
 * of range. */
ath_status ath_device_open(ath_device *d, int32_t index);

/* Wait for every stream, then destroy them. Safe on a zeroed device. */
void ath_device_close(ath_device *d);

/* Free and total device memory now, in bytes. */
ath_status ath_device_mem(const ath_device *d, uint64_t *free_bytes, uint64_t *total_bytes);

/* ---- memory ---------------------------------------------------------------
 *
 * Device memory and pinned host memory, at least 256-byte aligned. These are
 * the runtime's allocators: slow, meant for large blocks. Small and frequent
 * allocations go through an ath_dev_arena (below). ATH_ERR_NOMEM when the
 * memory is not available. Freeing NULL does nothing. */
ath_status ath_dev_alloc(const ath_device *d, size_t bytes, void **ptr);
void       ath_dev_free(const ath_device *d, void *ptr);
ath_status ath_pinned_alloc(const ath_device *d, size_t bytes, void **ptr);
void       ath_pinned_free(const ath_device *d, void *ptr);

/* ---- copies --------------------------------------------------------------
 *
 * Copy `bytes` from src to dst on stream s. Each pointer may be device,
 * pinned or ordinary host memory; the runtime infers the direction from
 * the addresses (unified virtual addressing). Asynchronous, as above. */
ath_status ath_copy(const ath_device *d, void *dst, const void *src, size_t bytes, ath_stream s);

/* Copy `rows` rows of `width` bytes, row r from src + r · spitch to
 * dst + r · dpitch, on stream s: a strided slice (columns of a matrix)
 * gathered into contiguous memory, or the other way round, in one call.
 * Pointers as for ath_copy; pitches at least width. */
ath_status ath_copy_2d(const ath_device *d, void *dst, size_t dpitch, const void *src, size_t spitch, size_t width,
                       size_t rows, ath_stream s);

/* Set `bytes` of device memory to the byte `value`, on stream s. */
ath_status ath_memset(const ath_device *d, void *dst, int value, size_t bytes, ath_stream s);

/* Wait on the host until all work queued on stream s, or on the whole
 * device, has finished. */
ath_status ath_stream_sync(const ath_device *d, ath_stream s);
ath_status ath_device_sync(const ath_device *d);

/* ---- events -------------------------------------------------------------
 *
 * timing: true to measure time with the event (slightly more expensive to
 * record; events used only for ordering should pass false). */
ath_status ath_event_create(const ath_device *d, ath_event *e, bool timing);
void       ath_event_destroy(const ath_device *d, ath_event *e);
/* Mark the point in stream s after all the work queued so far. */
ath_status ath_event_record(const ath_device *d, ath_event *e, ath_stream s);
/* Make stream s wait, on the device, for the work before event e. The host
 * does not wait. */
ath_status ath_stream_wait(const ath_device *d, ath_stream s, const ath_event *e);
/* Wait on the host for the work before event e. */
ath_status ath_event_sync(const ath_device *d, const ath_event *e);

/* ---- host work in a stream (step 12) ---------------------------------------
 * Run fn(ctx) on a thread of the runtime when the work queued on stream s
 * so far has finished; the work queued after it waits until fn returns. fn
 * must not call the runtime. It takes part in graphs like any other work
 * (a host node): a replay calls fn again with the same ctx, which must
 * therefore outlive the graph. The routed experts that are not in device
 * memory run this way on the CPU (src/ops/moe.c). */
typedef void (*ath_host_fn)(void *ctx);
ath_status ath_host_call(const ath_device *d, ath_stream s, ath_host_fn fn, void *ctx);
/* Whether stream s is recording a graph now (between ath_graph_begin and
 * ath_graph_end): work queued now will be replayed, so what it points to
 * must last as long as the graph. */
ath_status ath_stream_capturing(const ath_device *d, ath_stream s, bool *capturing);
/* Milliseconds between two completed timing events (start recorded before
 * end). ATH_ERR_INVALID_ARG for events created without timing. */
ath_status ath_event_elapsed(const ath_device *d, const ath_event *start, const ath_event *end, float *ms);

/* ---- graphs (step 9) --------------------------------------------------------
 *
 * A graph records the work queued on one stream between ath_graph_begin and
 * ath_graph_end, without running it; ath_graph_launch then runs all of it
 * with one call. A decode step of some 500 kernels costs one launch on the
 * host instead of 500. The recorded work keeps its arguments, addresses
 * included, so whatever changes from one replay to the next must be read
 * from memory by the work itself (a copy from pinned host memory, written
 * before the launch, or values left in device memory by the previous
 * replay). While a stream is being recorded nothing may wait for it, and
 * nothing is allocated or freed through the runtime. */
typedef struct {
    void *graph;   /* the recorded work        */
    void *exec;    /* the graph ready to run   */
} ath_graph;

/* Start recording the work queued on stream s. */
ath_status ath_graph_begin(const ath_device *d, ath_stream s);
/* Stop recording s and prepare the graph in g. Also after a failure while
 * recording, which leaves the stream usable again: then g stays empty and
 * the runtime's error is returned (or ATH_OK with an empty g if nothing
 * failed in the runtime; the caller knows its own failure). */
ath_status ath_graph_end(const ath_device *d, ath_stream s, ath_graph *g);
/* Run the recorded work on stream s, after the work queued there so far. */
ath_status ath_graph_launch(const ath_device *d, const ath_graph *g, ath_stream s);
/* Free g (empty or not). */
void ath_graph_destroy(const ath_device *d, ath_graph *g);

/* ---- device arena ----------------------------------------------------------
 *
 * Linear allocation in device memory, with the semantics of the core arenas
 * (src/core/arena.h): pointer bumping inside blocks, mark and reset,
 * everything freed together. Block bookkeeping lives on the host, in the
 * arena `host` given at init, since the host cannot write into device
 * memory. A request larger than the block size gets a block of its own. */
typedef struct ath_dev_block ath_dev_block;

typedef struct {
    const ath_device *dev;
    ath_arena        *host;          /* block records                            */
    ath_dev_block    *first, *cur;
    size_t            block_size;    /* minimum size of a new block, in bytes    */
} ath_dev_arena;

typedef struct {
    ath_dev_block *block;
    size_t         used;
} ath_dev_arena_pos;

#define ATH_DEV_ARENA_DEFAULT_BLOCK ((size_t)256 << 20)   /* 256 MiB */
#define ATH_DEV_ALIGN 256

/* block_size 0 selects ATH_DEV_ARENA_DEFAULT_BLOCK. */
void  ath_dev_arena_init(ath_dev_arena *a, const ath_device *d, ath_arena *host, size_t block_size);
/* Free every block's device memory. The host records stay in `host`. */
void  ath_dev_arena_free(ath_dev_arena *a);
/* size bytes of device memory aligned to align (a power of two up to
 * ATH_DEV_ALIGN; 0 means ATH_DEV_ALIGN). NULL on failure. */
void *ath_dev_arena_alloc(ath_dev_arena *a, size_t size, size_t align);
ath_dev_arena_pos ath_dev_arena_mark(const ath_dev_arena *a);
void              ath_dev_arena_reset(ath_dev_arena *a, ath_dev_arena_pos pos);
size_t            ath_dev_arena_used(const ath_dev_arena *a);
size_t            ath_dev_arena_reserved(const ath_dev_arena *a);

#ifdef __cplusplus
}
#endif

#endif /* ATH_BACKEND_H */
