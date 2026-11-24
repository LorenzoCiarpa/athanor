/* test_backend.c: the GPU runtime (src/backend/backend.h).
 *
 * In a build without a GPU backend (`make test`) the stub must refuse every
 * operation cleanly; the GPU tests are skipped. In the CUDA build
 * (`make cuda test`) they run on device 0: memory, copies in every
 * direction, memset, the three streams ordered by events, timing, the
 * failure of an impossible allocation, and the device arena. Data is
 * checked byte for byte.
 */
#include "test.h"

#include "backend/backend.h"

#include <string.h>

/* A pattern that differs at every byte offset and between seeds. */
static void fill(unsigned char *p, size_t n, uint32_t seed)
{
    uint32_t x = seed * 2654435761u + 1;
    for (size_t i = 0; i < n; i++) {
        x ^= x << 13, x ^= x >> 17, x ^= x << 5;
        p[i] = (unsigned char)x;
    }
}

static bool same(const unsigned char *a, const unsigned char *b, size_t n) { return memcmp(a, b, n) == 0; }

/* Open device 0, or skip the test when the build or the machine has none. */
#define OPEN_OR_SKIP(d)                                                       \
    do {                                                                      \
        if (ath_gpu_count() == 0)                                             \
            ATH_SKIP(strcmp(ath_gpu_backend(), "none") == 0                   \
                         ? "built without a GPU backend (make cuda test)"     \
                         : "no usable GPU device (ATH_LOG=info shows why)");  \
        ATH_REQUIRE_OK(ath_device_open(&(d), 0));                             \
    } while (0)

ATH_TEST(backend_without_gpu)
{
    if (strcmp(ath_gpu_backend(), "none") != 0)
        ATH_SKIP("this build has a GPU backend");
    ATH_EXPECT_EQ_INT(ath_gpu_count(), 0);
    ath_device d;
    ATH_EXPECT_STATUS(ath_device_open(&d, 0), ATH_ERR_UNSUPPORTED);
    void *p = (void *)1;
    ATH_EXPECT_STATUS(ath_dev_alloc(&d, 16, &p), ATH_ERR_UNSUPPORTED);
    ATH_EXPECT(p == NULL);
    ATH_EXPECT_STATUS(ath_pinned_alloc(&d, 16, &p), ATH_ERR_UNSUPPORTED);
    ath_event e;
    ATH_EXPECT_STATUS(ath_event_create(&d, &e, true), ATH_ERR_UNSUPPORTED);
    ath_dev_free(&d, NULL);
    ath_device_close(&d);
}

ATH_TEST(device_properties)
{
    ath_device d;
    OPEN_OR_SKIP(d);
    ATH_EXPECT(d.name[0] != '\0');
    ATH_EXPECT(d.cc_major >= 5);
    ATH_EXPECT(d.n_sm > 0);
    ATH_EXPECT_EQ_INT(d.warp, 32);
    ATH_EXPECT(d.mem_total > ((uint64_t)1 << 30));
    ATH_EXPECT(d.driver_version > 0 && d.runtime_version > 0);
    uint64_t free_b = 0, total_b = 0;
    ATH_EXPECT_OK(ath_device_mem(&d, &free_b, &total_b));
    ATH_EXPECT(free_b > 0 && free_b <= total_b && total_b == d.mem_total);
    printf("  %s: sm_%d%d, %d SMs, %.1f GiB (%.1f free), L2 %.0f MiB, %llu KiB shared per block, driver %d, "
           "runtime %d\n",
           d.name, d.cc_major, d.cc_minor, d.n_sm, d.mem_total / 1073741824.0, free_b / 1073741824.0,
           d.l2_bytes / 1048576.0, (unsigned long long)(d.smem_per_block / 1024), d.driver_version,
           d.runtime_version);

    ath_device bad;
    ATH_EXPECT_STATUS(ath_device_open(&bad, ath_gpu_count()), ATH_ERR_INVALID_ARG);
    ATH_EXPECT_STATUS(ath_device_open(&bad, -1), ATH_ERR_INVALID_ARG);
    ath_device_close(&bad);   /* zeroed by the failed open: a no-op */
    ath_device_close(&d);
    ath_device_close(&d);     /* twice is fine */
}

/* Round trips through device memory, from pinned and from ordinary host
 * memory, at odd sizes and offsets. */
ATH_TEST(copies_and_memset)
{
    ath_device d;
    OPEN_OR_SKIP(d);
    ath_arena a;
    ath_arena_init(&a, 0);
    static const size_t sizes[] = {1, 3, 255, 256, 4097, 1000003, (size_t)24 << 20};
    const size_t max = (size_t)24 << 20;
    unsigned char *pin_in, *pin_out, *dev_a, *dev_b;
    ATH_REQUIRE_OK(ath_pinned_alloc(&d, max + 64, (void **)&pin_in));
    ATH_REQUIRE_OK(ath_pinned_alloc(&d, max + 64, (void **)&pin_out));
    ATH_REQUIRE_OK(ath_dev_alloc(&d, max + 64, (void **)&dev_a));
    ATH_REQUIRE_OK(ath_dev_alloc(&d, max + 64, (void **)&dev_b));
    unsigned char *host_in = ath_arena_alloc(&a, max + 64, 64), *host_out = ath_arena_alloc(&a, max + 64, 64);
    ATH_REQUIRE(host_in && host_out);
    ATH_EXPECT(((uintptr_t)dev_a & 255) == 0);

    for (size_t k = 0; k < ATH_ARRAY_LEN(sizes); k++) {
        const size_t n = sizes[k], off = k % 7;   /* unaligned starts too */
        /* pinned → device → device → pinned, one stream */
        fill(pin_in + off, n, (uint32_t)k + 1);
        memset(pin_out, 0, n + off);
        ATH_EXPECT_OK(ath_copy(&d, dev_a + off, pin_in + off, n, ATH_STREAM_COMPUTE));
        ATH_EXPECT_OK(ath_copy(&d, dev_b, dev_a + off, n, ATH_STREAM_COMPUTE));
        ATH_EXPECT_OK(ath_copy(&d, pin_out + off, dev_b, n, ATH_STREAM_COMPUTE));
        ATH_EXPECT_OK(ath_stream_sync(&d, ATH_STREAM_COMPUTE));
        ATH_EXPECT(same(pin_in + off, pin_out + off, n));
        /* ordinary host memory both ways */
        fill(host_in, n, (uint32_t)k + 100);
        ATH_EXPECT_OK(ath_copy(&d, dev_a, host_in, n, ATH_STREAM_H2D));
        ATH_EXPECT_OK(ath_stream_sync(&d, ATH_STREAM_H2D));
        ATH_EXPECT_OK(ath_copy(&d, host_out, dev_a, n, ATH_STREAM_D2H));
        ATH_EXPECT_OK(ath_stream_sync(&d, ATH_STREAM_D2H));
        ATH_EXPECT(same(host_in, host_out, n));
    }

    /* memset: a range inside a known pattern, the bytes around it untouched */
    fill(pin_in, 4096, 7);
    ATH_EXPECT_OK(ath_copy(&d, dev_a, pin_in, 4096, ATH_STREAM_COMPUTE));
    ATH_EXPECT_OK(ath_memset(&d, dev_a + 100, 0xAB, 1000, ATH_STREAM_COMPUTE));
    ATH_EXPECT_OK(ath_copy(&d, pin_out, dev_a, 4096, ATH_STREAM_COMPUTE));
    ATH_EXPECT_OK(ath_device_sync(&d));
    bool ok = same(pin_out, pin_in, 100) && same(pin_out + 1100, pin_in + 1100, 4096 - 1100);
    for (int i = 100; i < 1100; i++)
        ok = ok && pin_out[i] == 0xAB;
    ATH_EXPECT(ok);

    /* arguments */
    ATH_EXPECT_OK(ath_copy(&d, dev_a, pin_in, 0, ATH_STREAM_COMPUTE));
    ATH_EXPECT_STATUS(ath_copy(&d, dev_a, pin_in, 1, ATH_STREAM_COUNT), ATH_ERR_INVALID_ARG);
    ATH_EXPECT_STATUS(ath_copy(&d, NULL, pin_in, 1, ATH_STREAM_COMPUTE), ATH_ERR_INVALID_ARG);
    ATH_EXPECT_STATUS(ath_memset(&d, NULL, 0, 1, ATH_STREAM_COMPUTE), ATH_ERR_INVALID_ARG);

    ath_dev_free(&d, dev_a);
    ath_dev_free(&d, dev_b);
    ath_pinned_free(&d, pin_in);
    ath_pinned_free(&d, pin_out);
    ath_device_close(&d);
    ath_arena_free(&a);
}

/* The three streams in a pipeline ordered only by events: upload on H2D,
 * a device copy on COMPUTE, download on D2H. Each round overwrites the
 * buffers of the previous one, so a missing dependency shows up as stale
 * data. */
ATH_TEST(streams_ordered_by_events)
{
    ath_device d;
    OPEN_OR_SKIP(d);
    const size_t n = (size_t)32 << 20;
    unsigned char *in, *out, *dev_a, *dev_b;
    ATH_REQUIRE_OK(ath_pinned_alloc(&d, n, (void **)&in));
    ATH_REQUIRE_OK(ath_pinned_alloc(&d, n, (void **)&out));
    ATH_REQUIRE_OK(ath_dev_alloc(&d, n, (void **)&dev_a));
    ATH_REQUIRE_OK(ath_dev_alloc(&d, n, (void **)&dev_b));
    ath_event uploaded, computed, downloaded;
    ATH_REQUIRE_OK(ath_event_create(&d, &uploaded, false));
    ATH_REQUIRE_OK(ath_event_create(&d, &computed, false));
    ATH_REQUIRE_OK(ath_event_create(&d, &downloaded, false));

    int bad = 0;
    for (uint32_t round = 0; round < 8; round++) {
        fill(in, n, round + 1);
        ATH_EXPECT_OK(ath_copy(&d, dev_a, in, n, ATH_STREAM_H2D));
        ATH_EXPECT_OK(ath_event_record(&d, &uploaded, ATH_STREAM_H2D));
        ATH_EXPECT_OK(ath_stream_wait(&d, ATH_STREAM_COMPUTE, &uploaded));
        ATH_EXPECT_OK(ath_copy(&d, dev_b, dev_a, n, ATH_STREAM_COMPUTE));
        ATH_EXPECT_OK(ath_event_record(&d, &computed, ATH_STREAM_COMPUTE));
        ATH_EXPECT_OK(ath_stream_wait(&d, ATH_STREAM_D2H, &computed));
        ATH_EXPECT_OK(ath_copy(&d, out, dev_b, n, ATH_STREAM_D2H));
        ATH_EXPECT_OK(ath_event_record(&d, &downloaded, ATH_STREAM_D2H));
        ATH_EXPECT_OK(ath_event_sync(&d, &downloaded));
        bad += !same(in, out, n);
    }
    ATH_EXPECT_EQ_INT(bad, 0);

    ath_event_destroy(&d, &uploaded);
    ath_event_destroy(&d, &computed);
    ath_event_destroy(&d, &downloaded);
    ath_dev_free(&d, dev_a);
    ath_dev_free(&d, dev_b);
    ath_pinned_free(&d, in);
    ath_pinned_free(&d, out);
    ath_device_close(&d);
}

/* Timing with events: a large device-to-device copy and an upload from
 * pinned memory take a positive time, and the rates they imply are
 * plausible for any GPU (a sanity check, not a measurement). */
ATH_TEST(event_timing)
{
    ath_device d;
    OPEN_OR_SKIP(d);
    const size_t n = (size_t)256 << 20;
    void *a, *b, *pin;
    ATH_REQUIRE_OK(ath_dev_alloc(&d, n, &a));
    ATH_REQUIRE_OK(ath_dev_alloc(&d, n, &b));
    ATH_REQUIRE_OK(ath_pinned_alloc(&d, n, &pin));
    memset(pin, 1, n);
    ath_event t0, t1, t2, plain;
    ATH_REQUIRE_OK(ath_event_create(&d, &t0, true));
    ATH_REQUIRE_OK(ath_event_create(&d, &t1, true));
    ATH_REQUIRE_OK(ath_event_create(&d, &t2, true));
    ATH_REQUIRE_OK(ath_event_create(&d, &plain, false));

    ATH_EXPECT_OK(ath_copy(&d, a, pin, n, ATH_STREAM_COMPUTE));   /* warm up */
    ATH_EXPECT_OK(ath_event_record(&d, &t0, ATH_STREAM_COMPUTE));
    ATH_EXPECT_OK(ath_copy(&d, b, a, n, ATH_STREAM_COMPUTE));
    ATH_EXPECT_OK(ath_event_record(&d, &t1, ATH_STREAM_COMPUTE));
    ATH_EXPECT_OK(ath_copy(&d, a, pin, n, ATH_STREAM_COMPUTE));
    ATH_EXPECT_OK(ath_event_record(&d, &t2, ATH_STREAM_COMPUTE));
    ATH_EXPECT_OK(ath_event_sync(&d, &t2));
    float d2d = 0, h2d = 0;
    ATH_EXPECT_OK(ath_event_elapsed(&d, &t0, &t1, &d2d));
    ATH_EXPECT_OK(ath_event_elapsed(&d, &t1, &t2, &h2d));
    /* a device copy reads and writes n bytes */
    double d2d_gbs = 2.0 * n / (d2d * 1e-3) / 1e9, h2d_gbs = (double)n / (h2d * 1e-3) / 1e9;
    printf("  256 MiB: device to device %.3f ms (%.0f GB/s read + write), pinned host to device %.3f ms "
           "(%.1f GB/s)\n",
           d2d, d2d_gbs, h2d, h2d_gbs);
    ATH_EXPECT(d2d > 0 && h2d > 0);
    ATH_EXPECT(d2d_gbs > 10 && d2d_gbs < 20000);
    ATH_EXPECT(h2d_gbs > 0.5 && h2d_gbs < 500);
    ATH_EXPECT_STATUS(ath_event_elapsed(&d, &t0, &plain, &d2d), ATH_ERR_INVALID_ARG);

    ath_event_destroy(&d, &t0);
    ath_event_destroy(&d, &t1);
    ath_event_destroy(&d, &t2);
    ath_event_destroy(&d, &plain);
    ath_dev_free(&d, a);
    ath_dev_free(&d, b);
    ath_pinned_free(&d, pin);
    ath_device_close(&d);
}

/* An allocation larger than the device fails with ATH_ERR_NOMEM, and the
 * device keeps working. */
ATH_TEST(impossible_allocation)
{
    ath_device d;
    OPEN_OR_SKIP(d);
    void *p = (void *)1;
    ATH_EXPECT_STATUS(ath_dev_alloc(&d, (size_t)d.mem_total * 2, &p), ATH_ERR_NOMEM);
    ATH_EXPECT(p == NULL);
    unsigned char in[64], out[64];
    fill(in, sizeof in, 3);
    void *dev;
    ATH_REQUIRE_OK(ath_dev_alloc(&d, sizeof in, &dev));
    ATH_EXPECT_OK(ath_copy(&d, dev, in, sizeof in, ATH_STREAM_COMPUTE));
    ATH_EXPECT_OK(ath_copy(&d, out, dev, sizeof out, ATH_STREAM_COMPUTE));
    ATH_EXPECT_OK(ath_stream_sync(&d, ATH_STREAM_COMPUTE));
    ATH_EXPECT(same(in, out, sizeof in));
    ATH_EXPECT_OK(ath_dev_alloc(&d, 0, &p));
    ATH_EXPECT(p == NULL);
    ath_dev_free(&d, dev);
    ath_device_close(&d);
}

ATH_TEST(device_arena)
{
    ath_device d;
    OPEN_OR_SKIP(d);
    ath_arena host;
    ath_arena_init(&host, 0);
    ath_dev_arena da;
    ath_dev_arena_init(&da, &d, &host, (size_t)1 << 20);   /* 1 MiB blocks */

    char *a = ath_dev_arena_alloc(&da, 100, 0);
    char *b = ath_dev_arena_alloc(&da, 100, 16);
    ATH_REQUIRE(a && b);
    ATH_EXPECT(((uintptr_t)a & 255) == 0 && ((uintptr_t)b & 15) == 0 && b >= a + 100);
    ATH_EXPECT(ath_dev_arena_alloc(&da, 8, 512) == NULL);   /* alignment above 256 */

    ath_dev_arena_pos mark = ath_dev_arena_mark(&da);
    char *c = ath_dev_arena_alloc(&da, (size_t)900 << 10, 0);   /* fills the first block */
    char *big = ath_dev_arena_alloc(&da, (size_t)3 << 20, 0);   /* a block of its own */
    ATH_REQUIRE(c && big);
    ATH_EXPECT(ath_dev_arena_reserved(&da) >= ((size_t)4 << 20));
    size_t used = ath_dev_arena_used(&da);
    ATH_EXPECT(used >= ((size_t)3 << 20) + ((size_t)900 << 10));

    /* the memory is usable: write through one allocation, read it back */
    unsigned char in[5000], out[5000];
    fill(in, sizeof in, 9);
    ATH_EXPECT_OK(ath_copy(&d, big + 12345, in, sizeof in, ATH_STREAM_COMPUTE));
    ATH_EXPECT_OK(ath_copy(&d, out, big + 12345, sizeof out, ATH_STREAM_COMPUTE));
    ATH_EXPECT_OK(ath_stream_sync(&d, ATH_STREAM_COMPUTE));
    ATH_EXPECT(same(in, out, sizeof in));

    /* reset: the same addresses come back, no new device memory */
    size_t reserved = ath_dev_arena_reserved(&da);
    ath_dev_arena_reset(&da, mark);
    ATH_EXPECT(ath_dev_arena_alloc(&da, (size_t)900 << 10, 0) == c);
    ATH_EXPECT(ath_dev_arena_alloc(&da, (size_t)3 << 20, 0) == big);
    ATH_EXPECT_EQ_U64(ath_dev_arena_reserved(&da), reserved);

    ath_dev_arena_free(&da);
    ATH_EXPECT_EQ_U64(ath_dev_arena_reserved(&da), 0);
    ath_arena_free(&host);
    ath_device_close(&d);
}

static const ath_test_case tests[] = {
    ATH_TEST_ENTRY(backend_without_gpu),
    ATH_TEST_ENTRY(device_properties),
    ATH_TEST_ENTRY(copies_and_memset),
    ATH_TEST_ENTRY(streams_ordered_by_events),
    ATH_TEST_ENTRY(event_timing),
    ATH_TEST_ENTRY(impossible_allocation),
    ATH_TEST_ENTRY(device_arena),
};

int main(int argc, char **argv)
{
    return ath_test_main(tests, ATH_ARRAY_LEN(tests), argc, argv);
}
