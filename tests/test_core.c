/* test_core.c: unit tests for src/core.
 *
 * Synthetic files are written into a fresh directory under $TMPDIR (or /tmp)
 * and removed afterwards. Tests on the real checkpoint read the directory in
 * $ATH_MODEL_DIR (default models/Qwen3.5-0.8B) and are skipped if it is absent.
 */
#include "test.h"

#include "core/arena.h"
#include "core/file.h"

#include <stdint.h>
#include <unistd.h>

/* ---- helpers: a scratch directory for synthetic files ---------------------- */

typedef struct {
    char path[512];
    char files[16][64];
    int  n_files;
} tmpdir;

static bool tmpdir_make(tmpdir *d)
{
    const char *base = getenv("TMPDIR");
    if (!base || !*base)
        base = "/tmp";
    snprintf(d->path, sizeof d->path, "%s/athanor-test-XXXXXX", base);
    d->n_files = 0;
    return mkdtemp(d->path) != NULL;
}

static bool tmpdir_write(tmpdir *d, const char *name, const void *data, size_t n)
{
    char p[1024];
    if (ath_path_join(p, sizeof p, d->path, name) != ATH_OK)
        return false;
    FILE *f = fopen(p, "wb");
    if (!f)
        return false;
    bool ok = fwrite(data, 1, n, f) == n;
    ok = (fclose(f) == 0) && ok;
    if (ok && d->n_files < (int)ATH_ARRAY_LEN(d->files)) {
        snprintf(d->files[d->n_files], sizeof d->files[0], "%s", name);
        d->n_files++;
    }
    return ok;
}

static void tmpdir_remove(tmpdir *d)
{
    char p[1024];
    for (int i = 0; i < d->n_files; i++)
        if (ath_path_join(p, sizeof p, d->path, d->files[i]) == ATH_OK)
            unlink(p);
    rmdir(d->path);
}

/* ---- base ----------------------------------------------------------------- */

ATH_TEST(status_strings)
{
    for (int i = 0; i < ATH_STATUS_COUNT; i++) {
        const char *s = ath_status_str((ath_status)i);
        ATH_EXPECT(s != NULL && s[0] != '\0');
        ATH_EXPECT(strcmp(s, "unknown status") != 0);
        for (int j = 0; j < i; j++)
            ATH_EXPECT(strcmp(s, ath_status_str((ath_status)j)) != 0);
    }
    ATH_EXPECT_STR_EQ(ath_status_str(ATH_STATUS_COUNT), "unknown status");
    ATH_EXPECT_STR_EQ(ath_status_str((ath_status)-1), "unknown status");
}

static ath_status fails_with(ath_status s) { return s; }

static ath_status check_chain(ath_status inner, int *reached)
{
    ATH_CHECK(fails_with(ATH_OK));
    ATH_CHECK(fails_with(inner));
    *reached = 1;
    return ATH_OK;
}

ATH_TEST(check_propagates)
{
    int reached = 0;
    ATH_EXPECT_OK(check_chain(ATH_OK, &reached));
    ATH_EXPECT(reached == 1);
    reached = 0;
    ATH_EXPECT_STATUS(check_chain(ATH_ERR_FORMAT, &reached), ATH_ERR_FORMAT);
    ATH_EXPECT(reached == 0);
}

ATH_TEST(log_levels)
{
    const char *saved = getenv("ATH_LOG");
    char keep[32] = "";
    if (saved)
        snprintf(keep, sizeof keep, "%s", saved);

    setenv("ATH_LOG", "none", 1);
    ATH_EXPECT(!ath_log_enabled(ATH_LOG_ERROR));
    setenv("ATH_LOG", "warn", 1);
    ATH_EXPECT(ath_log_enabled(ATH_LOG_ERROR));
    ATH_EXPECT(ath_log_enabled(ATH_LOG_WARN));
    ATH_EXPECT(!ath_log_enabled(ATH_LOG_INFO));
    setenv("ATH_LOG", "4", 1);
    ATH_EXPECT(ath_log_enabled(ATH_LOG_DEBUG));
    ATH_EXPECT(!ath_log_enabled(ATH_LOG_TRACE));
    setenv("ATH_LOG", "trace", 1);
    ATH_EXPECT(ath_log_enabled(ATH_LOG_TRACE));
    unsetenv("ATH_LOG");
    ATH_EXPECT(ath_log_enabled(ATH_LOG_WARN)); /* default */
    ATH_EXPECT(!ath_log_enabled(ATH_LOG_INFO));

    if (saved)
        setenv("ATH_LOG", keep, 1);
    else
        setenv("ATH_LOG", "none", 1);
}

/* ---- arena ---------------------------------------------------------------- */

ATH_TEST(arena_alignment)
{
    ath_arena a;
    ath_arena_init(&a, 4096);
    size_t aligns[] = {1, 2, 4, 8, 16, 64, 256, 4096};
    for (size_t i = 0; i < ATH_ARRAY_LEN(aligns); i++) {
        (void)ath_arena_alloc(&a, 3, 1); /* knock the offset off alignment */
        void *p = ath_arena_alloc(&a, 10, aligns[i]);
        ATH_REQUIRE(p != NULL);
        ATH_EXPECT((uintptr_t)p % aligns[i] == 0);
    }
    ATH_EXPECT(ath_arena_alloc(&a, 8, 0) == NULL);  /* not a power of two */
    ATH_EXPECT(ath_arena_alloc(&a, 8, 3) == NULL);
    ATH_EXPECT(ath_arena_alloc(&a, 0, 8) != NULL);  /* zero bytes is a valid request */
    ath_arena_free(&a);
}

ATH_TEST(arena_growth_keeps_data)
{
    ath_arena a;
    ath_arena_init(&a, 256);
    uint32_t *ptrs[200];
    for (uint32_t i = 0; i < 200; i++) {
        ptrs[i] = ath_arena_alloc(&a, 3 * sizeof(uint32_t), _Alignof(uint32_t));
        ATH_REQUIRE(ptrs[i] != NULL);
        ptrs[i][0] = i;
        ptrs[i][1] = i * 7u;
        ptrs[i][2] = ~i;
    }
    for (uint32_t i = 0; i < 200; i++)
        ATH_EXPECT(ptrs[i][0] == i && ptrs[i][1] == i * 7u && ptrs[i][2] == ~i);
    ATH_EXPECT(ath_arena_reserved(&a) >= 200 * 12);
    ATH_EXPECT(ath_arena_used(&a) >= 200 * 12);

    /* A request larger than the block size gets its own block. */
    unsigned char *big = ath_arena_alloc(&a, 10000, 64);
    ATH_REQUIRE(big != NULL);
    memset(big, 0xab, 10000);
    ATH_EXPECT(ptrs[199][0] == 199);
    ath_arena_free(&a);
    ATH_EXPECT(ath_arena_reserved(&a) == 0);
    ATH_EXPECT(ath_arena_used(&a) == 0);
}

ATH_TEST(arena_mark_reset)
{
    ath_arena a;
    ath_arena_init(&a, 128);
    ath_arena_pos empty = ath_arena_mark(&a);
    char *keep = ath_arena_strndup(&a, "persistent", 10);
    ATH_REQUIRE(keep != NULL);
    ATH_EXPECT_STR_EQ(keep, "persistent");

    ath_arena_pos m = ath_arena_mark(&a);
    size_t used_at_mark = ath_arena_used(&a);
    void *first_after = ath_arena_alloc(&a, 16, 16);
    for (int i = 0; i < 50; i++) /* spill into several new blocks */
        ATH_REQUIRE(ath_arena_alloc(&a, 100, 8) != NULL);
    size_t reserved = ath_arena_reserved(&a);

    ath_arena_reset(&a, m);
    ATH_EXPECT_EQ_U64(ath_arena_used(&a), used_at_mark);
    ATH_EXPECT_STR_EQ(keep, "persistent");
    void *again = ath_arena_alloc(&a, 16, 16);
    ATH_EXPECT(again == first_after); /* same position is handed out again */
    for (int i = 0; i < 50; i++)
        ATH_REQUIRE(ath_arena_alloc(&a, 100, 8) != NULL);
    ATH_EXPECT_EQ_U64(ath_arena_reserved(&a), reserved); /* blocks were reused */

    ath_arena_reset(&a, empty);
    ATH_EXPECT_EQ_U64(ath_arena_used(&a), 0);
    char *k2 = ath_arena_strndup(&a, "x", 1);
    ATH_EXPECT((void *)k2 == (void *)keep); /* back at the start of the first block */
    ATH_EXPECT_EQ_U64(ath_arena_reserved(&a), reserved);
    ath_arena_free(&a);
}

ATH_TEST(arena_zalloc_and_overflow)
{
    ath_arena a;
    ath_arena_init(&a, 0);
    unsigned char *z = ath_arena_zalloc(&a, 1000, 8);
    ATH_REQUIRE(z != NULL);
    int nonzero = 0;
    for (int i = 0; i < 1000; i++)
        nonzero += z[i] != 0;
    ATH_EXPECT(nonzero == 0);
    ATH_EXPECT(ath_arena_alloc(&a, SIZE_MAX, 1) == NULL);
    ATH_EXPECT(ath_arena_alloc(&a, SIZE_MAX - 8, 16) == NULL);
    ATH_EXPECT(ath_arena_strndup(&a, "", SIZE_MAX) == NULL);
    ath_arena_free(&a);
}

/* ---- file ----------------------------------------------------------------- */

ATH_TEST(file_mmap_and_pread)
{
    tmpdir d;
    ATH_REQUIRE(tmpdir_make(&d));
    unsigned char data[1000];
    for (int i = 0; i < 1000; i++)
        data[i] = (unsigned char)(i * 31 + 7);
    ATH_REQUIRE(tmpdir_write(&d, "data.bin", data, sizeof data));
    ATH_REQUIRE(tmpdir_write(&d, "empty.bin", "", 0));

    char p[1024];
    ATH_REQUIRE_OK(ath_path_join(p, sizeof p, d.path, "data.bin"));
    ATH_EXPECT(ath_file_exists(p));
    ath_file f;
    ATH_REQUIRE_OK(ath_file_open(&f, p));
    ATH_EXPECT_EQ_U64(f.size, 1000);
    ATH_REQUIRE(f.data != NULL);
    ATH_EXPECT(memcmp(f.data, data, 1000) == 0);

    unsigned char buf[100];
    ATH_EXPECT_OK(ath_file_pread(&f, buf, 100, 900));
    ATH_EXPECT(memcmp(buf, data + 900, 100) == 0);
    ATH_EXPECT_OK(ath_file_pread(&f, buf, 0, 1000));
    ATH_EXPECT_STATUS(ath_file_pread(&f, buf, 100, 901), ATH_ERR_RANGE);
    ATH_EXPECT_STATUS(ath_file_pread(&f, buf, 1, UINT64_MAX), ATH_ERR_RANGE);
    ath_file_close(&f);
    ath_file_close(&f); /* idempotent */
    ATH_EXPECT(f.fd == -1 && f.data == NULL);

    ATH_REQUIRE_OK(ath_path_join(p, sizeof p, d.path, "empty.bin"));
    ATH_REQUIRE_OK(ath_file_open(&f, p));
    ATH_EXPECT_EQ_U64(f.size, 0);
    ATH_EXPECT(f.data == NULL);
    ath_file_close(&f);

    ATH_REQUIRE_OK(ath_path_join(p, sizeof p, d.path, "missing.bin"));
    ATH_EXPECT(!ath_file_exists(p));
    ATH_EXPECT_STATUS(ath_file_open(&f, p), ATH_ERR_NOT_FOUND);
    ATH_EXPECT(f.fd == -1);
    ATH_EXPECT_STATUS(ath_file_open(&f, d.path), ATH_ERR_INVALID_ARG); /* a directory */
    ATH_EXPECT(!ath_file_exists(d.path));
    tmpdir_remove(&d);
}

ATH_TEST(path_join)
{
    char p[16];
    ATH_EXPECT_OK(ath_path_join(p, sizeof p, "a/b", "c.json"));
    ATH_EXPECT_STR_EQ(p, "a/b/c.json");
    ATH_EXPECT_OK(ath_path_join(p, sizeof p, "a/b/", "c.json"));
    ATH_EXPECT_STR_EQ(p, "a/b/c.json");
    ATH_EXPECT_STATUS(ath_path_join(p, sizeof p, "a/long/directory", "name"), ATH_ERR_RANGE);
}

/* ---- runner --------------------------------------------------------------- */

static const ath_test_case tests[] = {
    ATH_TEST_ENTRY(status_strings),
    ATH_TEST_ENTRY(check_propagates),
    ATH_TEST_ENTRY(log_levels),
    ATH_TEST_ENTRY(arena_alignment),
    ATH_TEST_ENTRY(arena_growth_keeps_data),
    ATH_TEST_ENTRY(arena_mark_reset),
    ATH_TEST_ENTRY(arena_zalloc_and_overflow),
    ATH_TEST_ENTRY(file_mmap_and_pread),
    ATH_TEST_ENTRY(path_join),
};

int main(int argc, char **argv)
{
    return ath_test_main(tests, ATH_ARRAY_LEN(tests), argc, argv);
}
