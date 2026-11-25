/* gpu_none.c: backend.h in a build without a GPU backend (`make`).
 *
 * There are no devices: ath_gpu_count() is 0, opening a device fails with
 * ATH_ERR_UNSUPPORTED, and every function that needs a device does the
 * same. Code that uses the GPU when one is present compiles and links
 * unchanged, and decides at run time.
 */
#include "backend/backend.h"

#include <string.h>

const char *ath_gpu_backend(void) { return "none"; }

int32_t ath_gpu_count(void) { return 0; }

ath_status ath_device_open(ath_device *d, int32_t index)
{
    if (d)
        memset(d, 0, sizeof *d);
    ath_log(ATH_LOG_ERROR, "gpu: device %d: this build has no GPU backend (make cuda)", index);
    return ATH_ERR_UNSUPPORTED;
}

void ath_device_close(ath_device *d)
{
    if (d)
        memset(d, 0, sizeof *d);
}

ath_status ath_device_mem(const ath_device *d, uint64_t *free_bytes, uint64_t *total_bytes)
{
    (void)d, (void)free_bytes, (void)total_bytes;
    return ATH_ERR_UNSUPPORTED;
}

ath_status ath_dev_alloc(const ath_device *d, size_t bytes, void **ptr)
{
    (void)d, (void)bytes;
    if (ptr)
        *ptr = NULL;
    return ATH_ERR_UNSUPPORTED;
}

void ath_dev_free(const ath_device *d, void *ptr) { (void)d, (void)ptr; }

ath_status ath_pinned_alloc(const ath_device *d, size_t bytes, void **ptr)
{
    (void)d, (void)bytes;
    if (ptr)
        *ptr = NULL;
    return ATH_ERR_UNSUPPORTED;
}

void ath_pinned_free(const ath_device *d, void *ptr) { (void)d, (void)ptr; }

ath_status ath_copy(const ath_device *d, void *dst, const void *src, size_t bytes, ath_stream s)
{
    (void)d, (void)dst, (void)src, (void)bytes, (void)s;
    return ATH_ERR_UNSUPPORTED;
}

ath_status ath_copy_2d(const ath_device *d, void *dst, size_t dpitch, const void *src, size_t spitch, size_t width,
                       size_t rows, ath_stream s)
{
    (void)d, (void)dst, (void)dpitch, (void)src, (void)spitch, (void)width, (void)rows, (void)s;
    return ATH_ERR_UNSUPPORTED;
}

ath_status ath_memset(const ath_device *d, void *dst, int value, size_t bytes, ath_stream s)
{
    (void)d, (void)dst, (void)value, (void)bytes, (void)s;
    return ATH_ERR_UNSUPPORTED;
}

ath_status ath_stream_sync(const ath_device *d, ath_stream s)
{
    (void)d, (void)s;
    return ATH_ERR_UNSUPPORTED;
}

ath_status ath_graph_begin(const ath_device *d, ath_stream s)
{
    (void)d, (void)s;
    return ATH_ERR_UNSUPPORTED;
}

ath_status ath_graph_end(const ath_device *d, ath_stream s, ath_graph *g)
{
    (void)d, (void)s;
    if (g)
        memset(g, 0, sizeof *g);
    return ATH_ERR_UNSUPPORTED;
}

ath_status ath_graph_launch(const ath_device *d, const ath_graph *g, ath_stream s)
{
    (void)d, (void)g, (void)s;
    return ATH_ERR_UNSUPPORTED;
}

void ath_graph_destroy(const ath_device *d, ath_graph *g)
{
    (void)d;
    if (g)
        memset(g, 0, sizeof *g);
}

ath_status ath_device_sync(const ath_device *d)
{
    (void)d;
    return ATH_ERR_UNSUPPORTED;
}

ath_status ath_event_create(const ath_device *d, ath_event *e, bool timing)
{
    (void)d, (void)timing;
    if (e)
        e->handle = NULL;
    return ATH_ERR_UNSUPPORTED;
}

void ath_event_destroy(const ath_device *d, ath_event *e) { (void)d, (void)e; }

ath_status ath_event_record(const ath_device *d, ath_event *e, ath_stream s)
{
    (void)d, (void)e, (void)s;
    return ATH_ERR_UNSUPPORTED;
}

ath_status ath_stream_wait(const ath_device *d, ath_stream s, const ath_event *e)
{
    (void)d, (void)s, (void)e;
    return ATH_ERR_UNSUPPORTED;
}

ath_status ath_host_call(const ath_device *d, ath_stream s, ath_host_fn fn, void *ctx)
{
    (void)d, (void)s, (void)fn, (void)ctx;
    return ATH_ERR_UNSUPPORTED;
}

ath_status ath_stream_capturing(const ath_device *d, ath_stream s, bool *capturing)
{
    (void)d, (void)s, (void)capturing;
    return ATH_ERR_UNSUPPORTED;
}

ath_status ath_event_sync(const ath_device *d, const ath_event *e)
{
    (void)d, (void)e;
    return ATH_ERR_UNSUPPORTED;
}

ath_status ath_event_elapsed(const ath_device *d, const ath_event *start, const ath_event *end, float *ms)
{
    (void)d, (void)start, (void)end, (void)ms;
    return ATH_ERR_UNSUPPORTED;
}
