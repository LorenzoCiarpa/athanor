/* devices.c: `athanor devices [--bench]`: the GPUs this build can use.
 *
 * Lists every device with its properties, through the backend only
 * (src/backend/backend.h). With --bench it also times copies with events:
 * host to device and back from pinned and from ordinary memory, and device
 * to device. The numbers describe this machine for the record; they are
 * not a benchmark report (docs/BENCHMARKS.md).
 */
#include "commands.h"

#include "backend/backend.h"

#include <string.h>

void ath_cmd_devices_usage(FILE *out)
{
    fprintf(out,
            "usage: athanor devices [--bench]\n\n"
            "List the GPUs this build can use: name, compute capability, multiprocessors,\n"
            "memory, cache, driver and runtime versions. A build without a GPU backend\n"
            "(make, not make cuda) lists none.\n\n"
            "  --bench   also time 256 MiB copies with GPU events (best of 5): host to device\n"
            "            and back from pinned and from ordinary memory, device to device\n");
}

static double gib(uint64_t b) { return (double)b / 1073741824.0; }

/* Best time of `reps` copies of n bytes from src to dst, in milliseconds. */
static ath_status time_copy(const ath_device *d, void *dst, const void *src, size_t n, int reps, float *best)
{
    ath_event t0, t1;
    ATH_CHECK(ath_event_create(d, &t0, true));
    ath_status s = ath_event_create(d, &t1, true);
    *best = 0;
    for (int r = 0; s == ATH_OK && r < reps; r++) {
        float ms = 0;
        s = ath_event_record(d, &t0, ATH_STREAM_COMPUTE);
        if (s == ATH_OK)
            s = ath_copy(d, dst, src, n, ATH_STREAM_COMPUTE);
        if (s == ATH_OK)
            s = ath_event_record(d, &t1, ATH_STREAM_COMPUTE);
        if (s == ATH_OK)
            s = ath_event_sync(d, &t1);
        if (s == ATH_OK)
            s = ath_event_elapsed(d, &t0, &t1, &ms);
        if (s == ATH_OK && (r == 0 || ms < *best))
            *best = ms;
    }
    ath_event_destroy(d, &t0);
    ath_event_destroy(d, &t1);
    return s;
}

static ath_status bench(ath_arena *a, const ath_device *d)
{
    const size_t n = (size_t)256 << 20;
    void *dev_a = NULL, *dev_b = NULL, *pin = NULL;
    char *page = ath_arena_alloc(a, n, 4096);
    if (!page)
        return ATH_ERR_NOMEM;
    memset(page, 1, n);   /* touch it: no page faults inside the timing */
    ath_status s = ath_dev_alloc(d, n, &dev_a);
    if (s == ATH_OK)
        s = ath_dev_alloc(d, n, &dev_b);
    if (s == ATH_OK)
        s = ath_pinned_alloc(d, n, &pin);
    if (s == ATH_OK)
        memset(pin, 1, n);

    static const struct { const char *what; int dst, src; double factor; } runs[] = {
        {"host (pinned) to device", 0, 2, 1}, {"device to host (pinned)", 2, 0, 1},
        {"host (pageable) to device", 0, 3, 1}, {"device to host (pageable)", 3, 0, 1},
        {"device to device", 1, 0, 2},   /* reads and writes n bytes */
    };
    void *buf[4] = {dev_a, dev_b, pin, page};
    printf("  copies of 256 MiB, best of 5 (for the record, not a benchmark report):\n");
    for (size_t i = 0; s == ATH_OK && i < ATH_ARRAY_LEN(runs); i++) {
        float ms;
        s = time_copy(d, buf[runs[i].dst], buf[runs[i].src], n, 5, &ms);
        if (s == ATH_OK)
            printf("    %-28s %8.3f ms  %7.1f GB/s%s\n", runs[i].what, ms, runs[i].factor * n / (ms * 1e-3) / 1e9,
                   runs[i].factor > 1 ? " (read + write)" : "");
    }
    ath_dev_free(d, dev_a);
    ath_dev_free(d, dev_b);
    ath_pinned_free(d, pin);
    return s;
}

int ath_cmd_devices(ath_arena *a, int argc, char **argv)
{
    bool do_bench = false;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--bench")) {
            do_bench = true;
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            ath_cmd_devices_usage(stdout);
            return ATH_EXIT_OK;
        } else {
            fprintf(stderr, "athanor devices: unexpected argument %s\n\n", argv[i]);
            ath_cmd_devices_usage(stderr);
            return ATH_EXIT_USAGE;
        }
    }
    int32_t n = ath_gpu_count();
    printf("GPU backend: %s, %d device%s\n", ath_gpu_backend(), n, n == 1 ? "" : "s");
    if (!strcmp(ath_gpu_backend(), "none"))
        printf("(this build has no GPU backend: build with make cuda)\n");
    int rc = ATH_EXIT_OK;
    for (int32_t i = 0; i < n; i++) {
        ath_device d;
        ath_status s = ath_device_open(&d, i);
        if (s != ATH_OK) {
            fprintf(stderr, "athanor devices: device %d: %s\n", i, ath_status_str(s));
            rc = ATH_EXIT_FAILURE;
            continue;
        }
        uint64_t free_b = 0;
        ath_device_mem(&d, &free_b, NULL);
        printf("\n[%d] %s\n", i, d.name);
        printf("  compute capability %d.%d (sm_%d%d), %d multiprocessors, warp %d\n", d.cc_major, d.cc_minor,
               d.cc_major, d.cc_minor, d.n_sm, d.warp);
        printf("  memory %.1f GiB, %.1f GiB free; L2 %.0f MiB; shared memory per block up to %llu KiB\n",
               gib(d.mem_total), gib(free_b), (double)d.l2_bytes / 1048576.0,
               (unsigned long long)(d.smem_per_block / 1024));
        printf("  driver CUDA %d.%d, runtime %d.%d\n", d.driver_version / 1000, d.driver_version % 1000 / 10,
               d.runtime_version / 1000, d.runtime_version % 1000 / 10);
        if (do_bench) {
            ath_arena_pos pos = ath_arena_mark(a);
            s = bench(a, &d);
            ath_arena_reset(a, pos);
            if (s != ATH_OK) {
                fprintf(stderr, "athanor devices: device %d: bench failed: %s\n", i, ath_status_str(s));
                rc = ATH_EXIT_FAILURE;
            }
        }
        ath_device_close(&d);
    }
    return rc;
}
