/* test_core.c: unit tests for src/core.
 *
 * Synthetic files are written into a fresh directory under $TMPDIR (or /tmp)
 * and removed afterwards. Tests on the real checkpoint read the directory in
 * $ATH_MODEL_DIR (default models/Qwen3.5-0.8B) and are skipped if it is absent.
 */
#include "test.h"

#include "core/arena.h"
#include "core/file.h"
#include "core/json.h"

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

/* The real checkpoint used by the tests that need one. These tests check
 * format-level invariants only, so any checkpoint directory works. */
static const char *model_dir(void)
{
    const char *d = getenv("ATH_MODEL_DIR");
    return (d && *d) ? d : "models/Qwen3.5-0.8B";
}

static bool model_file(char *out, size_t cap, const char *name)
{
    return ath_path_join(out, cap, model_dir(), name) == ATH_OK && ath_file_exists(out);
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

/* ---- json ----------------------------------------------------------------- */

static const ath_json *parse_ok(ath_test *t_, ath_arena *a, const char *text)
{
    const ath_json *root = NULL;
    ath_json_error err = {0};
    ath_status s = ath_json_parse(a, text, strlen(text), &root, &err);
    if (s != ATH_OK) {
        char msg[256];
        snprintf(msg, sizeof msg, "parse failed at %zu:%zu: %s", err.line, err.col, err.msg);
        ath_test_fail_(t_, __FILE__, __LINE__, msg);
        return NULL;
    }
    return root;
}

static ath_status parse_status(ath_arena *a, const char *text)
{
    const ath_json *root;
    return ath_json_parse(a, text, strlen(text), &root, NULL);
}

ATH_TEST(json_scalars_and_containers)
{
    ath_arena a;
    ath_arena_init(&a, 0);
    const ath_json *r = parse_ok(t_, &a,
        " {\"a\": 1, \"b\": [true, false, null, -2.5e3, \"x\"], \"c\": {}, \"d\": [], \"e\": {\"f\": {\"g\": 7}}}\n");
    ATH_REQUIRE(r != NULL);
    ATH_EXPECT(r->type == ATH_JSON_OBJECT);
    ATH_EXPECT_EQ_U64(ath_json_count(r), 5);

    int64_t i = 0;
    ATH_EXPECT_OK(ath_json_to_int(ath_json_get(r, "a"), &i));
    ATH_EXPECT_EQ_INT(i, 1);
    const ath_json *b = ath_json_get(r, "b");
    ATH_EXPECT_EQ_U64(ath_json_count(b), 5);
    bool flag = false;
    ATH_EXPECT_OK(ath_json_to_bool(ath_json_at(b, 0), &flag));
    ATH_EXPECT(flag);
    ATH_EXPECT_OK(ath_json_to_bool(ath_json_at(b, 1), &flag));
    ATH_EXPECT(!flag);
    ATH_EXPECT(ath_json_at(b, 2)->type == ATH_JSON_NULL);
    double d = 0;
    ATH_EXPECT_OK(ath_json_to_double(ath_json_at(b, 3), &d));
    ATH_EXPECT(d == -2500.0);
    const char *str;
    size_t len;
    ATH_EXPECT_OK(ath_json_to_str(ath_json_at(b, 4), &str, &len));
    ATH_EXPECT_STR_EQ(str, "x");
    ATH_EXPECT(ath_json_at(b, 5) == NULL);
    ATH_EXPECT(ath_json_get(r, "c")->type == ATH_JSON_OBJECT && ath_json_count(ath_json_get(r, "c")) == 0);
    ATH_EXPECT(ath_json_first(ath_json_get(r, "d")) == NULL);

    /* iteration in document order, keys recorded */
    const char *keys[] = {"a", "b", "c", "d", "e"};
    size_t k = 0;
    for (const ath_json *c = ath_json_first(r); c; c = c->next, k++)
        ATH_EXPECT_STR_EQ(c->key, keys[k]);
    ATH_EXPECT_EQ_U64(k, 5);

    /* dotted paths, including array indices */
    ATH_EXPECT_OK(ath_json_to_int(ath_json_path(r, "e.f.g"), &i));
    ATH_EXPECT_EQ_INT(i, 7);
    ATH_EXPECT_OK(ath_json_to_str(ath_json_path(r, "b.4"), &str, NULL));
    ATH_EXPECT_STR_EQ(str, "x");
    ATH_EXPECT(ath_json_path(r, "e.f.h") == NULL);
    ATH_EXPECT(ath_json_path(r, "b.9") == NULL);
    ATH_EXPECT(ath_json_path(r, "b.x") == NULL);
    ATH_EXPECT(ath_json_path(r, "a.b") == NULL);

    /* typed access errors */
    ATH_EXPECT_STATUS(ath_json_to_int(ath_json_get(r, "missing"), &i), ATH_ERR_NOT_FOUND);
    ATH_EXPECT_STATUS(ath_json_to_int(ath_json_get(r, "b"), &i), ATH_ERR_TYPE);
    ATH_EXPECT_STATUS(ath_json_to_int(ath_json_at(b, 3), &i), ATH_ERR_TYPE);
    ATH_EXPECT_STATUS(ath_json_to_str(ath_json_get(r, "a"), &str, NULL), ATH_ERR_TYPE);
    ATH_EXPECT_OK(ath_json_to_double(ath_json_get(r, "a"), &d)); /* int widens to double */
    ATH_EXPECT(d == 1.0);
    ATH_EXPECT(ath_json_get(b, "a") == NULL);  /* not an object */
    ATH_EXPECT(ath_json_at(r, 0) == NULL);     /* not an array  */
    ATH_EXPECT_STR_EQ(ath_json_type_name(ATH_JSON_OBJECT), "object");
    ath_arena_free(&a);
}

ATH_TEST(json_string_escapes)
{
    ath_arena a;
    ath_arena_init(&a, 0);
    const ath_json *r = parse_ok(t_, &a,
        "[\"q\\\"b\\\\s\\/b\\bf\\fn\\nr\\rt\\t\", \"\\u00e9\\u20AC\", \"\\ud83d\\ude00\","
        " \"a\\u0000b\", \"\xc3\xa9\xe2\x82\xac\xf0\x9f\x98\x80\", \"\"]");
    ATH_REQUIRE(r != NULL);
    const char *s;
    size_t len;
    ATH_EXPECT_OK(ath_json_to_str(ath_json_at(r, 0), &s, &len));
    ATH_EXPECT(len == 16 && memcmp(s, "q\"b\\s/b\bf\fn\nr\rt\t", 16) == 0);
    ATH_EXPECT_OK(ath_json_to_str(ath_json_at(r, 1), &s, &len));
    ATH_EXPECT_STR_EQ(s, "\xc3\xa9\xe2\x82\xac");       /* e-acute, euro sign */
    ATH_EXPECT_OK(ath_json_to_str(ath_json_at(r, 2), &s, &len));
    ATH_EXPECT_STR_EQ(s, "\xf0\x9f\x98\x80");           /* U+1F600 from a surrogate pair */
    ATH_EXPECT_OK(ath_json_to_str(ath_json_at(r, 3), &s, &len));
    ATH_EXPECT(len == 3 && s[0] == 'a' && s[1] == '\0' && s[2] == 'b' && s[3] == '\0');
    ATH_EXPECT_OK(ath_json_to_str(ath_json_at(r, 4), &s, &len));
    ATH_EXPECT(len == 9);                                 /* raw UTF-8 passes through */
    ATH_EXPECT_OK(ath_json_to_str(ath_json_at(r, 5), &s, &len));
    ATH_EXPECT(len == 0 && s[0] == '\0');

    /* keys are decoded too, and lookups compare decoded bytes */
    r = parse_ok(t_, &a, "{\"k\\u00e9y\": 1, \"dup\": 1, \"dup\": 2}");
    ATH_REQUIRE(r != NULL);
    ATH_EXPECT(ath_json_get(r, "k\xc3\xa9y") != NULL);
    int64_t i;
    ATH_EXPECT_OK(ath_json_to_int(ath_json_get(r, "dup"), &i));
    ATH_EXPECT_EQ_INT(i, 1); /* first duplicate wins */
    ath_arena_free(&a);
}

ATH_TEST(json_numbers)
{
    ath_arena a;
    ath_arena_init(&a, 0);
    const ath_json *r = parse_ok(t_, &a,
        "[0, -0, 9223372036854775807, -9223372036854775808, 9223372036854775808,"
        " 1.5e-3, 1E2, -0.0, 10000000, 1e-06, 123456789012345678901234567890]");
    ATH_REQUIRE(r != NULL);
    int64_t i;
    double d;
    ATH_EXPECT_OK(ath_json_to_int(ath_json_at(r, 0), &i));
    ATH_EXPECT_EQ_INT(i, 0);
    ATH_EXPECT_OK(ath_json_to_int(ath_json_at(r, 1), &i));
    ATH_EXPECT_EQ_INT(i, 0);
    ATH_EXPECT_OK(ath_json_to_int(ath_json_at(r, 2), &i));
    ATH_EXPECT(i == INT64_MAX);
    ATH_EXPECT_OK(ath_json_to_int(ath_json_at(r, 3), &i));
    ATH_EXPECT(i == INT64_MIN);
    ATH_EXPECT(ath_json_at(r, 4)->type == ATH_JSON_DOUBLE);  /* one past int64 */
    ATH_EXPECT_OK(ath_json_to_double(ath_json_at(r, 4), &d));
    ATH_EXPECT(d == 9223372036854775808.0);
    ATH_EXPECT_OK(ath_json_to_double(ath_json_at(r, 5), &d));
    ATH_EXPECT(d == 1.5e-3);
    ATH_EXPECT_OK(ath_json_to_double(ath_json_at(r, 6), &d));
    ATH_EXPECT(d == 100.0 && ath_json_at(r, 6)->type == ATH_JSON_DOUBLE);
    ATH_EXPECT_OK(ath_json_to_double(ath_json_at(r, 7), &d));
    ATH_EXPECT(d == 0.0 && signbit(d));
    ATH_EXPECT_OK(ath_json_to_int(ath_json_at(r, 8), &i));
    ATH_EXPECT_EQ_INT(i, 10000000);
    ATH_EXPECT_OK(ath_json_to_double(ath_json_at(r, 9), &d));
    ATH_EXPECT(d == 1e-06);
    ATH_EXPECT_OK(ath_json_to_double(ath_json_at(r, 10), &d));
    ATH_EXPECT_NEAR(d / 1.2345678901234568e29, 1.0, 1e-15);
    ath_arena_free(&a);
}

ATH_TEST(json_rejects_invalid)
{
    ath_arena a;
    ath_arena_init(&a, 0);
    const char *bad[] = {
        "", "   ", "{", "[", "[1,]", "{\"a\":1,}", "{\"a\" 1}", "{a:1}", "[1 2]",
        "01", "-", "1.", ".5", "1e", "1e+", "+1", "--1", "0x10", "NaN", "Infinity",
        "tru", "nul", "True", "\"abc", "\"a\\x\"", "\"a\\u12\"", "\"a\\u12G4\"",
        "\"\\udc00\"", "\"\\ud800\"", "\"\\ud800\\u0041\"", "\"\\ud800x\"",
        "\"tab\there\"", "\"nl\nhere\"", "\"\xff\"", "\"\xc0\x80\"", "\"\xed\xa0\x80\"",
        "\"\xf4\x90\x80\x80\"", "\"\xe2\x82\"", "[1] [2]", "{} x", "[\"a\"\"b\"]",
        "{\"a\":1 \"b\":2}", "[,1]", "{,}", "\"\\",
    };
    for (size_t k = 0; k < ATH_ARRAY_LEN(bad); k++) {
        if (parse_status(&a, bad[k]) != ATH_ERR_PARSE) {
            char msg[128];
            snprintf(msg, sizeof msg, "accepted invalid JSON #%zu: %s", k, bad[k]);
            ath_test_fail_(t_, __FILE__, __LINE__, msg);
        }
    }
    /* nesting beyond the limit is an error, at the limit it is not */
    char deep[2 * ATH_JSON_MAX_DEPTH + 8];
    size_t n = 0;
    for (int k = 0; k < ATH_JSON_MAX_DEPTH; k++)
        deep[n++] = '[';
    for (int k = 0; k < ATH_JSON_MAX_DEPTH; k++)
        deep[n++] = ']';
    deep[n] = '\0';
    ATH_EXPECT_OK(parse_status(&a, deep));
    memmove(deep + 1, deep, n + 1);
    deep[0] = '[';
    deep[n + 1] = ']';
    deep[n + 2] = '\0';
    ATH_EXPECT_STATUS(parse_status(&a, deep), ATH_ERR_PARSE);
    ath_arena_free(&a);
}

ATH_TEST(json_error_position_and_length)
{
    ath_arena a;
    ath_arena_init(&a, 0);
    const char *text = "{\n  \"a\": 1,\n  \"b\": tru\n}";
    const ath_json *r;
    ath_json_error err = {0};
    ATH_EXPECT_STATUS(ath_json_parse(&a, text, strlen(text), &r, &err), ATH_ERR_PARSE);
    ATH_EXPECT_EQ_U64(err.line, 3);
    ATH_EXPECT_EQ_U64(err.col, 8);
    ATH_EXPECT(err.msg != NULL);

    /* the input does not need a terminator: parse a prefix of a longer buffer */
    const char *buf = "[1,2]garbage";
    ATH_EXPECT_OK(ath_json_parse(&a, buf, 5, &r, NULL));
    ATH_EXPECT_EQ_U64(ath_json_count(r), 2);
    ATH_EXPECT_STATUS(ath_json_parse(&a, buf, 4, &r, NULL), ATH_ERR_PARSE);
    ath_arena_free(&a);
}

/* Parse a whole file out of its mmap, as the loaders do. */
static const ath_json *parse_file(ath_test *t_, ath_arena *a, ath_file *f, const char *path)
{
    if (ath_file_open(f, path) != ATH_OK) {
        ath_test_fail_(t_, __FILE__, __LINE__, path);
        return NULL;
    }
    const ath_json *root = NULL;
    ath_json_error err = {0};
    if (ath_json_parse(a, f->data, f->size, &root, &err) != ATH_OK) {
        char msg[1200];
        snprintf(msg, sizeof msg, "%s:%zu:%zu: %s", path, err.line, err.col, err.msg);
        ath_test_fail_(t_, __FILE__, __LINE__, msg);
    }
    return root;
}

ATH_TEST(json_real_model_files)
{
    char p[1024];
    if (!model_file(p, sizeof p, "config.json"))
        ATH_SKIP("no checkpoint in $ATH_MODEL_DIR");
    ath_arena a;
    ath_arena_init(&a, 0);
    ath_file f;

    const ath_json *cfg = parse_file(t_, &a, &f, p);
    ath_file_close(&f);
    ATH_REQUIRE(cfg != NULL && cfg->type == ATH_JSON_OBJECT);
    const char *s;
    ATH_EXPECT_OK(ath_json_to_str(ath_json_get(cfg, "model_type"), &s, NULL));

    if (model_file(p, sizeof p, "tokenizer_config.json")) {
        const ath_json *tc = parse_file(t_, &a, &f, p);
        ath_file_close(&f);
        ATH_REQUIRE(tc != NULL && tc->type == ATH_JSON_OBJECT);
        ATH_EXPECT(ath_json_count(tc) > 0);
    }
    if (model_file(p, sizeof p, "tokenizer.json")) {
        /* the largest JSON in a checkpoint: vocabulary and merges, MBs of escapes */
        const ath_json *tk = parse_file(t_, &a, &f, p);
        ath_file_close(&f);
        ATH_REQUIRE(tk != NULL);
        ATH_EXPECT(ath_json_count(ath_json_path(tk, "model.vocab")) > 1000);
    }
    ath_arena_free(&a);
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
    ATH_TEST_ENTRY(json_scalars_and_containers),
    ATH_TEST_ENTRY(json_string_escapes),
    ATH_TEST_ENTRY(json_numbers),
    ATH_TEST_ENTRY(json_rejects_invalid),
    ATH_TEST_ENTRY(json_error_position_and_length),
    ATH_TEST_ENTRY(json_real_model_files),
};

int main(int argc, char **argv)
{
    return ath_test_main(tests, ATH_ARRAY_LEN(tests), argc, argv);
}
