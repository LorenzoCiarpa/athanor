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
#include "core/safetensors.h"
#include "core/tensor.h"

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
    for (int i = 0; ok && i < d->n_files; i++)
        if (strcmp(d->files[i], name) == 0)
            return true;
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

/* ---- tensor --------------------------------------------------------------- */

ATH_TEST(dtype_table)
{
    ATH_EXPECT_EQ_U64(ath_dtype_size(ATH_DTYPE_F32), 4);
    ATH_EXPECT_EQ_U64(ath_dtype_size(ATH_DTYPE_F16), 2);
    ATH_EXPECT_EQ_U64(ath_dtype_size(ATH_DTYPE_BF16), 2);
    ATH_EXPECT_EQ_U64(ath_dtype_size(ATH_DTYPE_I32), 4);
    ATH_EXPECT_EQ_U64(ath_dtype_size(ATH_DTYPE_I64), 8);
    ATH_EXPECT_EQ_U64(ath_dtype_size(ATH_DTYPE_U8), 1);
    ATH_EXPECT_EQ_U64(ath_dtype_size(ATH_DTYPE_F8_E4M3), 1);
    ATH_EXPECT_EQ_U64(ath_dtype_size(ATH_DTYPE_F8_E5M2), 1);
    ATH_EXPECT_EQ_U64(ath_dtype_size(ATH_DTYPE_COUNT), 0);
    ATH_EXPECT_EQ_U64(ath_dtype_size(ATH_DTYPE_QUANT_FIRST), 0);
    ATH_EXPECT_STR_EQ(ath_dtype_name(ATH_DTYPE_BF16), "bf16");
    ATH_EXPECT_STR_EQ(ath_dtype_name(ATH_DTYPE_F8_E4M3), "f8_e4m3");
    ATH_EXPECT_STR_EQ(ath_dtype_name(ATH_DTYPE_QUANT_LAST), "?");
    for (int i = 0; i < ATH_DTYPE_COUNT; i++)
        ATH_EXPECT(ath_dtype_size((ath_dtype)i) > 0 && strcmp(ath_dtype_name((ath_dtype)i), "?") != 0);
    ATH_EXPECT_STR_EQ(ath_tier_name(ATH_TIER_VRAM), "vram");
}

ATH_TEST(tensor_init_and_views)
{
    float buf[2 * 3 * 4];
    for (int i = 0; i < 24; i++)
        buf[i] = (float)i;
    ath_tensor t;
    int64_t shape[] = {2, 3, 4};
    ATH_REQUIRE_OK(ath_tensor_init(&t, ATH_DTYPE_F32, 3, shape, buf, ATH_TIER_RAM));
    ATH_EXPECT_EQ_INT(t.stride[0], 12);
    ATH_EXPECT_EQ_INT(t.stride[1], 4);
    ATH_EXPECT_EQ_INT(t.stride[2], 1);
    ATH_EXPECT_EQ_INT(ath_tensor_numel(&t), 24);
    ATH_EXPECT_EQ_U64(ath_tensor_nbytes(&t), 96);
    ATH_EXPECT(ath_tensor_is_contiguous(&t));

    /* rows 1..2 of the middle dimension: shares memory, not contiguous */
    ath_tensor v;
    ATH_REQUIRE_OK(ath_tensor_slice(&t, 1, 1, 2, &v));
    ATH_EXPECT_EQ_INT(v.shape[1], 2);
    ATH_EXPECT(v.data == (void *)(buf + 4));
    ATH_EXPECT(!ath_tensor_is_contiguous(&v));
    /* element [1][1][3] of the view is element [1][2][3] of t */
    const float *vd = v.data;
    ATH_EXPECT(vd[1 * v.stride[0] + 1 * v.stride[1] + 3 * v.stride[2]] == 23.0f);

    /* outer slice stays contiguous */
    ATH_REQUIRE_OK(ath_tensor_slice(&t, 0, 1, 1, &v));
    ATH_EXPECT(v.data == (void *)(buf + 12) && ath_tensor_is_contiguous(&v));
    ATH_EXPECT_STATUS(ath_tensor_slice(&t, 0, 1, 2, &v), ATH_ERR_RANGE);
    ATH_EXPECT_STATUS(ath_tensor_slice(&t, 3, 0, 1, &v), ATH_ERR_INVALID_ARG);
    ATH_EXPECT_OK(ath_tensor_slice(&t, 2, 4, 0, &v)); /* empty slice at the end */
    ATH_EXPECT_EQ_INT(ath_tensor_numel(&v), 0);

    /* reshape a contiguous tensor; refuse a strided one */
    int64_t flat[] = {6, 4};
    ath_tensor r;
    ATH_REQUIRE_OK(ath_tensor_reshape(&t, 2, flat, &r));
    ATH_EXPECT(r.data == (void *)buf && r.stride[0] == 4);
    int64_t wrong[] = {5, 4};
    ATH_EXPECT_STATUS(ath_tensor_reshape(&t, 2, wrong, &r), ATH_ERR_INVALID_ARG);
    ATH_REQUIRE_OK(ath_tensor_slice(&t, 2, 0, 2, &v));
    ATH_EXPECT_STATUS(ath_tensor_reshape(&v, 2, flat, &r), ATH_ERR_INVALID_ARG);

    /* scalar, empty and invalid shapes */
    ATH_REQUIRE_OK(ath_tensor_init(&t, ATH_DTYPE_BF16, 0, NULL, NULL, ATH_TIER_RAM));
    ATH_EXPECT_EQ_INT(ath_tensor_numel(&t), 1);
    ATH_EXPECT_EQ_U64(ath_tensor_nbytes(&t), 2);
    int64_t empty[] = {0, 5};
    ATH_REQUIRE_OK(ath_tensor_init(&t, ATH_DTYPE_F32, 2, empty, NULL, ATH_TIER_RAM));
    ATH_EXPECT_EQ_U64(ath_tensor_nbytes(&t), 0);
    int64_t neg[] = {2, -1};
    ATH_EXPECT_STATUS(ath_tensor_init(&t, ATH_DTYPE_F32, 2, neg, NULL, ATH_TIER_RAM), ATH_ERR_RANGE);
    int64_t huge[] = {INT64_MAX / 2, 3};
    ATH_EXPECT_STATUS(ath_tensor_init(&t, ATH_DTYPE_U8, 2, huge, NULL, ATH_TIER_RAM), ATH_ERR_RANGE);
    int64_t huge0[] = {0, INT64_MAX / 2, 3};
    ATH_EXPECT_STATUS(ath_tensor_init(&t, ATH_DTYPE_U8, 3, huge0, NULL, ATH_TIER_RAM), ATH_ERR_RANGE);
    int64_t big[] = {INT64_MAX / 4};
    ATH_EXPECT_STATUS(ath_tensor_init(&t, ATH_DTYPE_I64, 1, big, NULL, ATH_TIER_RAM), ATH_ERR_RANGE);
    int64_t nine[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    ATH_EXPECT_STATUS(ath_tensor_init(&t, ATH_DTYPE_F32, 9, nine, NULL, ATH_TIER_RAM), ATH_ERR_RANGE);
    ATH_EXPECT_STATUS(ath_tensor_init(&t, ATH_DTYPE_QUANT_FIRST, 1, nine, NULL, ATH_TIER_RAM),
                      ATH_ERR_UNSUPPORTED);
}

/* ---- safetensors ---------------------------------------------------------- */

/* Write a safetensors file: 8-byte little-endian header length, the header
 * padded with spaces to a multiple of 8 (as the reference writer does), data. */
static bool write_st(tmpdir *d, const char *name, const char *header, const void *data, size_t n)
{
    unsigned char buf[4096];
    size_t hl = strlen(header), padded = (hl + 7) & ~(size_t)7;
    if (8 + padded + n > sizeof buf)
        return false;
    for (int i = 0; i < 8; i++)
        buf[i] = (unsigned char)((uint64_t)padded >> (8 * i));
    memcpy(buf + 8, header, hl);
    memset(buf + 8 + hl, ' ', padded - hl);
    if (n)
        memcpy(buf + 8 + padded, data, n);
    return tmpdir_write(d, name, buf, 8 + padded + n);
}

ATH_TEST(safetensors_name_order)
{
    ATH_EXPECT(ath_safetensors_name_cmp("layers.2.w", "layers.10.w") < 0);
    ATH_EXPECT(ath_safetensors_name_cmp("layers.10.w", "layers.2.w") > 0);
    ATH_EXPECT(ath_safetensors_name_cmp("layers.10.a", "layers.10.b") < 0);
    ATH_EXPECT(ath_safetensors_name_cmp("a", "a") == 0);
    ATH_EXPECT(ath_safetensors_name_cmp("a", "a.b") < 0);
    ATH_EXPECT(ath_safetensors_name_cmp("x9", "x10") < 0);
    ATH_EXPECT(ath_safetensors_name_cmp("x010", "x9") > 0);
    ATH_EXPECT(ath_safetensors_name_cmp("x01", "x1") != 0);   /* total order */
    ATH_EXPECT(ath_safetensors_name_cmp("x01", "x1") == -ath_safetensors_name_cmp("x1", "x01"));
    ATH_EXPECT(ath_safetensors_name_cmp("model-00002-of-00010", "model-00010-of-00010") < 0);
}

ATH_TEST(safetensors_single_file)
{
    tmpdir d;
    ATH_REQUIRE(tmpdir_make(&d));
    unsigned char data[43];
    float f[6] = {0, 1, 2, 3, 4, 5};
    uint16_t bf[4] = {0x3f80, 0x4000, 0xbf80, 0x0000}; /* 1, 2, -1, 0 in bf16 */
    int64_t scalar = -42;
    memcpy(data, f, 24);
    memcpy(data + 24, bf, 8);
    memcpy(data + 32, &scalar, 8);
    data[40] = 7; data[41] = 8; data[42] = 9;
    const char *header =
        "{\"__metadata__\":{\"format\":\"pt\"},"
        "\"layers.10.w\":{\"dtype\":\"F32\",\"shape\":[2,3],\"data_offsets\":[0,24]},"
        "\"layers.2.w\":{\"dtype\":\"BF16\",\"shape\":[4],\"data_offsets\":[24,32]},"
        "\"emb\":{\"dtype\":\"I64\",\"shape\":[],\"data_offsets\":[32,40]},"
        "\"empty\":{\"dtype\":\"U8\",\"shape\":[0,3],\"data_offsets\":[40,40]},"
        "\"b\":{\"dtype\":\"U8\",\"shape\":[3],\"data_offsets\":[40,43]}}";
    ATH_REQUIRE(write_st(&d, "model.safetensors", header, data, sizeof data));

    ath_arena a;
    ath_arena_init(&a, 0);
    ath_safetensors st;
    ATH_REQUIRE_OK(ath_safetensors_open(&st, &a, d.path));
    ATH_EXPECT_EQ_U64(st.n_shards, 1);
    ATH_EXPECT_EQ_U64(st.n_tensors, 5);
    ATH_EXPECT_EQ_U64(st.tensor_bytes, 43);
    ATH_EXPECT_EQ_U64(st.data_bytes, 43);
    ATH_EXPECT_EQ_INT(st.index_total_size, -1);
    ATH_EXPECT_EQ_U64(st.shards[0].header_bytes % 8, 0);
    ATH_EXPECT_EQ_U64(st.shards[0].data_offset, 8 + st.shards[0].header_bytes);
    const char *fmt;
    ATH_EXPECT_OK(ath_json_to_str(ath_json_path(st.shards[0].header, "__metadata__.format"), &fmt, NULL));
    ATH_EXPECT_STR_EQ(fmt, "pt");

    const char *order[] = {"b", "emb", "empty", "layers.2.w", "layers.10.w"};
    for (uint32_t i = 0; i < st.n_tensors && i < 5; i++)
        ATH_EXPECT_STR_EQ(st.tensors[i].name, order[i]);

    const ath_safetensors_entry *e = ath_safetensors_find(&st, "layers.10.w");
    ATH_REQUIRE(e != NULL);
    ATH_EXPECT(e->t.dtype == ATH_DTYPE_F32 && e->t.ndim == 2);
    ATH_EXPECT(e->t.shape[0] == 2 && e->t.shape[1] == 3);
    ATH_EXPECT(e->t.tier == ATH_TIER_RAM);
    /* zero-copy: the view points into the mapping, at data_offset + begin */
    ATH_EXPECT((const char *)e->t.data == (const char *)st.shards[0].file.data + st.shards[0].data_offset);
    float row1[3];
    memcpy(row1, (const float *)e->t.data + e->t.stride[0], sizeof row1);
    ATH_EXPECT(row1[0] == 3.0f && row1[2] == 5.0f);

    e = ath_safetensors_find(&st, "layers.2.w");
    ATH_REQUIRE(e != NULL);
    ATH_EXPECT(e->t.dtype == ATH_DTYPE_BF16 && e->begin == 24 && e->end == 32);
    ATH_EXPECT(((const uint16_t *)e->t.data)[2] == 0xbf80);
    e = ath_safetensors_find(&st, "emb");
    ATH_REQUIRE(e != NULL);
    ATH_EXPECT(e->t.ndim == 0 && ath_tensor_numel(&e->t) == 1);
    int64_t got;
    memcpy(&got, e->t.data, 8);
    ATH_EXPECT_EQ_INT(got, -42);
    e = ath_safetensors_find(&st, "empty");
    ATH_REQUIRE(e != NULL);
    ATH_EXPECT(ath_tensor_numel(&e->t) == 0);
    ATH_EXPECT(ath_safetensors_find(&st, "layers.3.w") == NULL);
    ATH_EXPECT(ath_safetensors_find(&st, "") == NULL);
    ath_safetensors_close(&st);

    /* the same file opened directly */
    char p[1024];
    ATH_REQUIRE_OK(ath_path_join(p, sizeof p, d.path, "model.safetensors"));
    ATH_REQUIRE_OK(ath_safetensors_open_file(&st, &a, p));
    ATH_EXPECT_EQ_U64(st.n_tensors, 5);
    ath_safetensors_close(&st);
    ath_arena_free(&a);
    tmpdir_remove(&d);
}

ATH_TEST(safetensors_rejects_bad_headers)
{
    tmpdir d;
    ATH_REQUIRE(tmpdir_make(&d));
    ath_arena a;
    ath_arena_init(&a, 0);
    char p[1024];
    ATH_REQUIRE_OK(ath_path_join(p, sizeof p, d.path, "bad.safetensors"));
    static const unsigned char zeros[64];
    ath_safetensors st;

    static const struct {
        const char *header;
        size_t      data_bytes;
        ath_status  want;
    } cases[] = {
        {"[1,2]", 0, ATH_ERR_FORMAT},                                   /* not an object  */
        {"{\"a\":", 0, ATH_ERR_PARSE},                                  /* not JSON       */
        {"{\"a\":{\"dtype\":\"F64\",\"shape\":[1],\"data_offsets\":[0,8]}}", 8, ATH_ERR_UNSUPPORTED},
        {"{\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,4]}}", 4, ATH_ERR_FORMAT},
        {"{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}", 2, ATH_ERR_FORMAT},
        {"{\"a\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[1,0]}}", 1, ATH_ERR_FORMAT},
        {"{\"a\":{\"dtype\":\"U8\",\"shape\":[8],\"data_offsets\":[0,8]},"
         "\"b\":{\"dtype\":\"U8\",\"shape\":[8],\"data_offsets\":[4,12]}}", 12, ATH_ERR_FORMAT},
        {"{\"a\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[0,4]},"
         "\"b\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[8,12]}}", 12, ATH_ERR_FORMAT},
        {"{\"a\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[0,4]}}", 8, ATH_ERR_FORMAT},
        {"{\"a\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[0,4]},"
         "\"a\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[4,8]}}", 8, ATH_ERR_FORMAT},
        {"{\"a\":{\"dtype\":\"U8\",\"shape\":[-4],\"data_offsets\":[0,4]}}", 4, ATH_ERR_FORMAT},
        {"{\"a\":{\"dtype\":\"U8\",\"shape\":[4]}}", 4, ATH_ERR_FORMAT},
        {"{\"a\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[0]}}", 4, ATH_ERR_FORMAT},
        {"{\"a\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[0,-4]}}", 4, ATH_ERR_FORMAT},
        {"{\"a\":{\"shape\":[4],\"data_offsets\":[0,4]}}", 4, ATH_ERR_FORMAT},
        {"{\"a\":{\"dtype\":\"U8\",\"shape\":4,\"data_offsets\":[0,4]}}", 4, ATH_ERR_FORMAT},
        {"{\"a\":{\"dtype\":\"U8\",\"shape\":[1,1,1,1,1,1,1,1,1],\"data_offsets\":[0,1]}}", 1,
         ATH_ERR_UNSUPPORTED},
        {"{\"a\":[1]}", 0, ATH_ERR_FORMAT},
        {"{\"\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]}}", 1, ATH_ERR_FORMAT},
        {"{\"a\\u0000b\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]}}", 1, ATH_ERR_FORMAT},
    };
    for (size_t k = 0; k < ATH_ARRAY_LEN(cases); k++) {
        ATH_REQUIRE(write_st(&d, "bad.safetensors", cases[k].header, zeros, cases[k].data_bytes));
        ath_status got = ath_safetensors_open_file(&st, &a, p);
        if (got != cases[k].want) {
            char msg[512];
            snprintf(msg, sizeof msg, "case %zu: got '%s', want '%s': %s", k, ath_status_str(got),
                     ath_status_str(cases[k].want), cases[k].header);
            ath_test_fail_(t_, __FILE__, __LINE__, msg);
        }
        if (got == ATH_OK)
            ath_safetensors_close(&st);
    }

    /* broken framing */
    ATH_REQUIRE(tmpdir_write(&d, "bad.safetensors", "\x02\x00\x00", 3));
    ATH_EXPECT_STATUS(ath_safetensors_open_file(&st, &a, p), ATH_ERR_FORMAT);
    ATH_REQUIRE(tmpdir_write(&d, "bad.safetensors", "\xff\x00\x00\x00\x00\x00\x00\x00{}", 10));
    ATH_EXPECT_STATUS(ath_safetensors_open_file(&st, &a, p), ATH_ERR_FORMAT);
    ATH_REQUIRE(tmpdir_write(&d, "bad.safetensors", "\x02\x00\x00\x00\x00\x00\x00\x00{}", 10));
    ATH_EXPECT_OK(ath_safetensors_open_file(&st, &a, p)); /* an empty checkpoint is valid */
    ATH_EXPECT_EQ_U64(st.n_tensors, 0);
    ath_safetensors_close(&st);
    /* a directory without a checkpoint */
    ATH_EXPECT_STATUS(ath_safetensors_open(&st, &a, d.path), ATH_ERR_NOT_FOUND);
    ath_arena_free(&a);
    tmpdir_remove(&d);
}

ATH_TEST(safetensors_sharded)
{
    tmpdir d;
    ATH_REQUIRE(tmpdir_make(&d));
    unsigned char data[16];
    for (int i = 0; i < 16; i++)
        data[i] = (unsigned char)i;
    ATH_REQUIRE(write_st(&d, "model-00001-of-00002.safetensors",
        "{\"layers.0.w\":{\"dtype\":\"U8\",\"shape\":[2,2],\"data_offsets\":[0,4]},"
        "\"embed\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[4,8]}}", data, 8));
    ATH_REQUIRE(write_st(&d, "model-00002-of-00002.safetensors",
        "{\"layers.1.w\":{\"dtype\":\"U8\",\"shape\":[2,2],\"data_offsets\":[0,4]},"
        "\"head\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[4,8]}}", data + 8, 8));
    const char *index =
        "{\"metadata\":{\"total_size\":16},\"weight_map\":{"
        "\"head\":\"model-00002-of-00002.safetensors\","
        "\"layers.1.w\":\"model-00002-of-00002.safetensors\","
        "\"layers.0.w\":\"model-00001-of-00002.safetensors\","
        "\"embed\":\"model-00001-of-00002.safetensors\"}}";
    ATH_REQUIRE(tmpdir_write(&d, "model.safetensors.index.json", index, strlen(index)));

    ath_arena a;
    ath_arena_init(&a, 0);
    ath_safetensors st;
    ATH_REQUIRE_OK(ath_safetensors_open(&st, &a, d.path));
    ATH_EXPECT_EQ_U64(st.n_shards, 2);
    ATH_EXPECT(strstr(st.shards[0].path, "00001-of") != NULL); /* sorted by name */
    ATH_EXPECT_EQ_U64(st.n_tensors, 4);
    ATH_EXPECT_EQ_U64(st.tensor_bytes, 16);
    ATH_EXPECT_EQ_U64(st.data_bytes, 16);
    ATH_EXPECT_EQ_INT(st.index_total_size, 16);
    const ath_safetensors_entry *e = ath_safetensors_find(&st, "layers.1.w");
    ATH_REQUIRE(e != NULL);
    ATH_EXPECT_EQ_U64(e->shard, 1);
    ATH_EXPECT(((const unsigned char *)e->t.data)[3] == 11);
    e = ath_safetensors_find(&st, "embed");
    ATH_REQUIRE(e != NULL);
    ATH_EXPECT(e->shard == 0 && ((const unsigned char *)e->t.data)[0] == 4);
    ath_safetensors_close(&st);

    /* the index and the shards must agree */
    static const struct {
        const char *index;
        ath_status  want;
    } bad[] = {
        {"{\"weight_map\":{\"head\":\"model-00002-of-00002.safetensors\","   /* omits layers.1.w */
         "\"layers.0.w\":\"model-00001-of-00002.safetensors\","
         "\"embed\":\"model-00001-of-00002.safetensors\"}}", ATH_ERR_FORMAT},
        {"{\"weight_map\":{\"head\":\"model-00001-of-00002.safetensors\","   /* wrong shard */
         "\"layers.1.w\":\"model-00002-of-00002.safetensors\","
         "\"layers.0.w\":\"model-00001-of-00002.safetensors\","
         "\"embed\":\"model-00001-of-00002.safetensors\"}}", ATH_ERR_FORMAT},
        {"{\"weight_map\":{\"head\":\"model-00002-of-00002.safetensors\","   /* unknown tensor */
         "\"layers.1.w\":\"model-00002-of-00002.safetensors\","
         "\"layers.0.w\":\"model-00001-of-00002.safetensors\","
         "\"embed\":\"model-00001-of-00002.safetensors\","
         "\"ghost\":\"model-00001-of-00002.safetensors\"}}", ATH_ERR_FORMAT},
        {"{\"weight_map\":{\"head\":\"model-00002-of-00002.safetensors\","   /* listed twice */
         "\"head\":\"model-00002-of-00002.safetensors\","
         "\"layers.1.w\":\"model-00002-of-00002.safetensors\","
         "\"layers.0.w\":\"model-00001-of-00002.safetensors\"}}", ATH_ERR_FORMAT},
        {"{\"weight_map\":{\"head\":\"../model-00002-of-00002.safetensors\"}}", ATH_ERR_FORMAT},
        {"{\"weight_map\":{\"head\":\"model-00003-of-00003.safetensors\"}}", ATH_ERR_NOT_FOUND},
        {"{\"weight_map\":[]}", ATH_ERR_FORMAT},
        {"{\"weight_map\":{\"head\":3}}", ATH_ERR_FORMAT},
        {"{\"weight_map\":", ATH_ERR_PARSE},
    };
    for (size_t k = 0; k < ATH_ARRAY_LEN(bad); k++) {
        ATH_REQUIRE(tmpdir_write(&d, "model.safetensors.index.json", bad[k].index, strlen(bad[k].index)));
        ath_status got = ath_safetensors_open(&st, &a, d.path);
        if (got != bad[k].want) {
            char msg[512];
            snprintf(msg, sizeof msg, "index case %zu: got '%s', want '%s'", k,
                     ath_status_str(got), ath_status_str(bad[k].want));
            ath_test_fail_(t_, __FILE__, __LINE__, msg);
        }
        if (got == ATH_OK)
            ath_safetensors_close(&st);
    }
    ath_arena_free(&a);
    tmpdir_remove(&d);
}

ATH_TEST(safetensors_real_model)
{
    char p[1024];
    if (!model_file(p, sizeof p, "config.json"))
        ATH_SKIP("no checkpoint in $ATH_MODEL_DIR");
    ath_arena a;
    ath_arena_init(&a, 0);
    ath_safetensors st;
    ATH_REQUIRE_OK(ath_safetensors_open(&st, &a, model_dir()));
    ATH_EXPECT(st.n_tensors > 0);
    /* acceptance criterion 1 of step 1: tensor bytes == data sections */
    ATH_EXPECT_EQ_U64(st.tensor_bytes, st.data_bytes);
    if (st.index_total_size >= 0)
        ATH_EXPECT_EQ_U64((uint64_t)st.index_total_size, st.tensor_bytes);

    uint64_t file_bytes = 0;
    for (uint32_t s = 0; s < st.n_shards; s++)
        file_bytes += st.shards[s].file.size - st.shards[s].data_offset;
    ATH_EXPECT_EQ_U64(file_bytes, st.data_bytes);

    for (uint32_t i = 0; i < st.n_tensors; i++) {
        const ath_safetensors_entry *e = &st.tensors[i];
        const ath_safetensors_shard *sh = &st.shards[e->shard];
        const char *lo = (const char *)sh->file.data + sh->data_offset;
        const char *hi = (const char *)sh->file.data + sh->file.size;
        const char *at = e->t.data;
        ATH_EXPECT(at >= lo && at + ath_tensor_nbytes(&e->t) <= hi);
        ATH_EXPECT((uintptr_t)at % ath_dtype_size(e->t.dtype) == 0);
        ATH_EXPECT(ath_safetensors_find(&st, e->name) == e);
        if (i > 0)
            ATH_EXPECT(ath_safetensors_name_cmp(st.tensors[i - 1].name, e->name) < 0);
    }
    ath_safetensors_close(&st);
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
    ATH_TEST_ENTRY(dtype_table),
    ATH_TEST_ENTRY(tensor_init_and_views),
    ATH_TEST_ENTRY(safetensors_name_order),
    ATH_TEST_ENTRY(safetensors_single_file),
    ATH_TEST_ENTRY(safetensors_rejects_bad_headers),
    ATH_TEST_ENTRY(safetensors_sharded),
    ATH_TEST_ENTRY(safetensors_real_model),
};

int main(int argc, char **argv)
{
    return ath_test_main(tests, ATH_ARRAY_LEN(tests), argc, argv);
}
