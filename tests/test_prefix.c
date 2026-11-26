/* test_prefix.c: snapshots of a sequence at prefixes of its tokens (step 14).
 *
 * On a tiny random model, on the CPU: a state saved trimmed and loaded into
 * another (of another capacity) goes on with the same bits, with the KV
 * cache in bf16 and in int8; the store finds the longest prefix, keeps its
 * budget in host memory, writes to disk what must outlive it (not the
 * superseded turns, yes the branch points), reads it back after a reopen
 * with the same continuation, and refuses a damaged, truncated or foreign
 * file. The engine's snapshots of device states: test_engine.
 */
#include "test.h"
#include "tiny.h"
#include "util.h"

#include "infer/prefix.h"
#include "model/model.h"

#include <dirent.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    tmpdir    d;
    ath_arena a;
    ath_model m;
} tiny_model;

static ath_status tiny_open(tiny_model *t, uint64_t seed)
{
    ath_arena_init(&t->a, 0);
    return open_tiny(&t->a, &t->m, &t->d, &DENSE, (ckpt){.seed = seed, .real_dtypes = true}, NULL);
}

static void tiny_close(tiny_model *t)
{
    ath_model_close(&t->m);
    tmpdir_remove(&t->d);
    ath_arena_free(&t->a);
}

/* A fixed pseudo-random sequence of tokens. */
static void tokens(int32_t *ids, int n, uint32_t seed)
{
    uint32_t x = seed * 2654435761u + 1;
    for (int i = 0; i < n; i++) {
        x ^= x << 13, x ^= x >> 17, x ^= x << 5;
        ids[i] = (int32_t)(x % V);
    }
}

/* ids[from, to) into s: the first `whole` of them in one call, then one at a
 * time; the logits of each one-token call into lg[(i - whole) * V] (lg may
 * be NULL). */
static ath_status feed(const ath_model *m, ath_state *s, const int32_t *ids, int from, int whole, int to, float *lg,
                       ath_arena *scratch)
{
    if (whole > from)
        ATH_CHECK(ath_model_forward(m, &(ath_batch){.tokens = ids + from, .n_tokens = whole - from}, s, NULL, scratch,
                                    NULL));
    for (int i = whole > from ? whole : from; i < to; i++) {
        ath_tensor l;
        if (lg)
            ath_tensor_init(&l, ATH_DTYPE_F32, 2, (int64_t[]){1, V}, lg + (size_t)(i - whole) * V, ATH_TIER_RAM);
        ATH_CHECK(ath_model_forward(m, &(ath_batch){.tokens = ids + i, .n_tokens = 1, .logits = lg ? &l : NULL}, s,
                                    NULL, scratch, NULL));
    }
    return ATH_OK;
}

/* The snapshot's size from the specs, as model.h describes it. */
static size_t expected_bytes(const ath_model *m, ath_dtype act, ath_dtype kv, int32_t rows)
{
    size_t total = 0;
    for (uint32_t i = 0; i < m->n_states; i++) {
        const ath_state_spec *sp = &m->states[i];
        if (sp->kind == ATH_STATE_KV)
            total += kv == ATH_DTYPE_I8 ? (size_t)rows * (size_t)(sp->per_token + 4 * (sp->per_token / sp->group))
                                        : (size_t)rows * (size_t)sp->per_token * ath_dtype_size(act);
        else
            total += (size_t)sp->fixed * ath_dtype_size(sp->kind == ATH_STATE_RECURRENT ? ATH_DTYPE_F32 : act);
    }
    return total;
}

/* A state saved after 20 tokens, loaded into a larger one, goes on as the
 * first does: the logits of 6 more tokens, bit for bit. */
ATH_TEST(prefix_snapshot_roundtrip)
{
    enum { N = 20, MORE = 6 };
    tiny_model t;
    ATH_REQUIRE_OK(tiny_open(&t, 3));
    ath_arena scratch;
    ath_arena_init(&scratch, 0);
    int32_t ids[N + MORE];
    tokens(ids, N + MORE, 1);
    for (int k = 0; k < 2; k++) {
        const ath_dtype kv = k ? ATH_DTYPE_I8 : ATH_DTYPE_BF16;
        ath_state a, b, small;
        ATH_REQUIRE_OK(ath_state_init_kv(&a, &t.m, ATH_DTYPE_BF16, kv, 40, &t.a));
        ATH_REQUIRE_OK(ath_state_init_kv(&b, &t.m, ATH_DTYPE_BF16, kv, 64, &t.a));
        ATH_REQUIRE_OK(ath_state_init_kv(&small, &t.m, ATH_DTYPE_BF16, kv, 10, &t.a));
        ATH_REQUIRE_OK(feed(&t.m, &a, ids, 0, N, N, NULL, &scratch));
        const size_t bytes = ath_state_snap_bytes(&t.m, &a);
        ATH_EXPECT_EQ_INT((int64_t)bytes, (int64_t)expected_bytes(&t.m, ATH_DTYPE_BF16, kv, N));
        char *snap = malloc(bytes);
        ATH_REQUIRE(snap);
        ATH_EXPECT_STATUS(ath_state_snap_save(&a, &t.m, snap, bytes - 1), ATH_ERR_INVALID_ARG);
        ATH_REQUIRE_OK(ath_state_snap_save(&a, &t.m, snap, bytes));
        ATH_EXPECT_STATUS(ath_state_snap_load(&small, &t.m, N, 0, snap, bytes), ATH_ERR_RANGE);
        ATH_EXPECT_STATUS(ath_state_snap_load(&b, &t.m, N - 1, 0, snap, bytes), ATH_ERR_INVALID_ARG);
        ATH_REQUIRE_OK(ath_state_snap_load(&b, &t.m, N, 0, snap, bytes));
        ATH_EXPECT_EQ_INT(b.n_past, N);
        float la[MORE * V], lb[MORE * V];
        ATH_REQUIRE_OK(feed(&t.m, &a, ids, N, N, N + MORE, la, &scratch));
        ATH_REQUIRE_OK(feed(&t.m, &b, ids, N, N, N + MORE, lb, &scratch));
        printf("    %s KV: %zu bytes after %d tokens; %d more tokens %s\n", k ? "int8" : "bf16", bytes, N, MORE,
               memcmp(la, lb, sizeof la) ? "DIFFER" : "with the same logits, bit for bit");
        ATH_EXPECT(memcmp(la, lb, sizeof la) == 0);
        free(snap);
    }
    ath_arena_free(&scratch);
    tiny_close(&t);
}

/* ---- the store ------------------------------------------------------------------ */

/* The snapshot files of a folder: their number, and the first one's name. */
static int snap_files(const char *dir, char first[600])
{
    DIR *d = opendir(dir);
    int n = 0;
    for (const struct dirent *de; d && (de = readdir(d));)
        if (strlen(de->d_name) > 5 && !strcmp(de->d_name + strlen(de->d_name) - 5, ".snap")) {
            if (!n && first)
                snprintf(first, 600, "%s/%s", dir, de->d_name);
            n++;
        }
    if (d)
        closedir(d);
    return n;
}

static void remove_folder(const char *dir)
{
    DIR *d = opendir(dir);
    char p[1200];
    for (const struct dirent *de; d && (de = readdir(d));)
        if (strcmp(de->d_name, ".") && strcmp(de->d_name, "..") && snprintf(p, sizeof p, "%s/%s", dir, de->d_name) > 0)
            unlink(p);
    if (d)
        closedir(d);
    rmdir(dir);
}

static ath_prefix_stats stats(const ath_prefix_store *ps)
{
    ath_prefix_stats s;
    ath_prefix_get_stats(ps, &s);
    return s;
}

/* The longest prefix, the minimum length, a prefix saved twice, the budget
 * in host memory (the least recently used leaves), and a loaded snapshot
 * going on as the sequence it was taken from. */
ATH_TEST(prefix_store_memory)
{
    enum { N = 40 };
    tiny_model t;
    ATH_REQUIRE_OK(tiny_open(&t, 5));
    ath_arena scratch;
    ath_arena_init(&scratch, 0);
    int32_t p[N], q[N];
    tokens(p, N, 2);
    tokens(q, N, 3);
    ath_state s, r;
    ATH_REQUIRE_OK(ath_state_init(&s, &t.m, ATH_DTYPE_BF16, N, &t.a));
    ATH_REQUIRE_OK(ath_state_init(&r, &t.m, ATH_DTYPE_BF16, N, &t.a));
    /* room for the snapshots of 12 and 16 tokens, not for a third */
    const uint64_t budget = expected_bytes(&t.m, ATH_DTYPE_BF16, ATH_DTYPE_BF16, 12) +
                            expected_bytes(&t.m, ATH_DTYPE_BF16, ATH_DTYPE_BF16, 16) + 8;
    ath_prefix_store *ps;
    ATH_REQUIRE_OK(ath_prefix_open(&ps, &t.m, ATH_DTYPE_BF16, ATH_DTYPE_BF16,
                                   &(ath_prefix_opts){.ram_bytes = budget, .min_tokens = 6}));
    ATH_REQUIRE_OK(feed(&t.m, &s, p, 0, 4, 4, NULL, &scratch));
    ATH_REQUIRE_OK(ath_prefix_save(ps, p, &s));                       /* 4 < 6: not stored */
    ATH_EXPECT_EQ_INT(stats(ps).ram_count, 0);
    ATH_REQUIRE_OK(feed(&t.m, &s, p, 4, 8, 8, NULL, &scratch));
    ATH_REQUIRE_OK(ath_prefix_save(ps, p, &s));                       /* p[0, 8) */
    ATH_REQUIRE_OK(ath_prefix_save(ps, p, &s));                       /* again: one entry */
    ATH_EXPECT_EQ_INT(stats(ps).ram_count, 1);
    ATH_REQUIRE_OK(feed(&t.m, &s, p, 8, 12, 12, NULL, &scratch));
    ATH_REQUIRE_OK(ath_prefix_save(ps, p, &s));                       /* p[0, 12) */
    ATH_EXPECT_EQ_INT(ath_prefix_find(ps, p, N), 12);
    ATH_EXPECT_EQ_INT(ath_prefix_find(ps, p, 12), 8);                 /* shorter than n */
    ATH_EXPECT_EQ_INT(ath_prefix_find(ps, q, N), 0);
    int32_t branch[N];
    memcpy(branch, p, sizeof branch);
    branch[10] = (branch[10] + 1) % V;
    ATH_EXPECT_EQ_INT(ath_prefix_find(ps, branch, N), 8);
    /* a third sequence: the least recently used (p[0, 8)) leaves host memory */
    ATH_REQUIRE_OK(ath_state_reset(&r, &t.m));
    ATH_REQUIRE_OK(feed(&t.m, &r, q, 0, 16, 16, NULL, &scratch));
    ATH_REQUIRE_OK(ath_prefix_save(ps, q, &r));
    ATH_EXPECT_EQ_INT(stats(ps).ram_count, 2);
    ATH_EXPECT(stats(ps).ram_bytes <= budget);
    ATH_EXPECT_EQ_INT(ath_prefix_find(ps, branch, N), 0);
    ATH_EXPECT_EQ_INT(ath_prefix_find(ps, q, N), 16);
    /* p[0, 12) loaded into r goes on as s does */
    ATH_REQUIRE_OK(ath_prefix_load(ps, p, 12, N, &r));
    ATH_EXPECT_EQ_INT(r.n_past, 12);
    float ls[8 * V], lr[8 * V];
    ATH_REQUIRE_OK(feed(&t.m, &s, p, 12, 12, 20, ls, &scratch));
    ATH_REQUIRE_OK(feed(&t.m, &r, p, 12, 12, 20, lr, &scratch));
    ATH_EXPECT(memcmp(ls, lr, sizeof ls) == 0);
    ATH_EXPECT(!stats(ps).load_disk && stats(ps).load_tokens == 12);
    ATH_EXPECT_STATUS(ath_prefix_load(ps, p, 9, N, &r), ATH_ERR_NOT_FOUND);
    ath_prefix_close(ps);
    ath_arena_free(&scratch);
    tiny_close(&t);
}

/* Snapshots on disk: written when the store closes, the superseded turns
 * left out and a branch point kept; read back after a reopen with the
 * continuation of the sequence they were taken from; the folder's budget;
 * a damaged, a truncated and a foreign file refused. */
ATH_TEST(prefix_store_disk)
{
    enum { N = 48 };
    tiny_model t, other;
    ATH_REQUIRE_OK(tiny_open(&t, 7));
    ATH_REQUIRE_OK(tiny_open(&other, 8));
    ath_arena scratch;
    ath_arena_init(&scratch, 0);
    tmpdir sd;
    ATH_REQUIRE(tmpdir_make(&sd));
    char dir[600];
    snprintf(dir, sizeof dir, "%s/snapshots", sd.path);   /* the store creates it */
    int32_t p[N];
    tokens(p, N, 4);
    ath_state s, r;
    ATH_REQUIRE_OK(ath_state_init(&s, &t.m, ATH_DTYPE_BF16, N, &t.a));
    ATH_REQUIRE_OK(ath_state_init(&r, &t.m, ATH_DTYPE_BF16, N, &t.a));
    const ath_prefix_opts po = {.dir = dir, .min_tokens = 8};

    /* turn after turn of one conversation: only the last goes to disk */
    ath_prefix_store *ps;
    ATH_REQUIRE_OK(ath_prefix_open(&ps, &t.m, ATH_DTYPE_BF16, ATH_DTYPE_BF16, &po));
    ATH_REQUIRE_OK(feed(&t.m, &s, p, 0, 16, 16, NULL, &scratch));
    ATH_REQUIRE_OK(ath_prefix_save(ps, p, &s));
    ATH_REQUIRE_OK(feed(&t.m, &s, p, 16, 24, 24, NULL, &scratch));
    ATH_REQUIRE_OK(ath_prefix_save(ps, p, &s));
    ATH_EXPECT_EQ_INT(snap_files(dir, NULL), 0);   /* nothing written before it must be */
    ath_prefix_close(ps);
    ATH_EXPECT_EQ_INT(snap_files(dir, NULL), 1);

    /* reopened: p[0, 24) from disk goes on as s does */
    ATH_REQUIRE_OK(ath_prefix_open(&ps, &t.m, ATH_DTYPE_BF16, ATH_DTYPE_BF16, &po));
    ATH_EXPECT_EQ_INT(stats(ps).disk_count, 1);
    ATH_EXPECT_EQ_INT(ath_prefix_find(ps, p, N), 24);
    ATH_EXPECT_EQ_INT(ath_prefix_find(ps, p, 20), 0);   /* p[0, 16) was not written */
    ATH_REQUIRE_OK(ath_prefix_load(ps, p, 24, N, &r));
    ATH_EXPECT(stats(ps).load_disk && stats(ps).load_tokens == 24);
    float ls[8 * V], lr[8 * V];
    ATH_REQUIRE_OK(feed(&t.m, &s, p, 24, 24, 32, ls, &scratch));
    ATH_REQUIRE_OK(feed(&t.m, &r, p, 24, 24, 32, lr, &scratch));
    ATH_EXPECT(memcmp(ls, lr, sizeof ls) == 0);
    printf("    from disk after a reopen: 24 tokens, %llu bytes, then 8 tokens %s\n",
           (unsigned long long)stats(ps).load_bytes, memcmp(ls, lr, sizeof ls) ? "DIFFER" : "with the same logits");

    /* a branch point: p[0, 12) continued two ways is written with the rest */
    int32_t b2[N];
    memcpy(b2, p, sizeof b2);
    b2[12] = (b2[12] + 1) % V;
    ATH_REQUIRE_OK(ath_state_reset(&r, &t.m));
    ATH_REQUIRE_OK(feed(&t.m, &r, p, 0, 12, 12, NULL, &scratch));
    ATH_REQUIRE_OK(ath_prefix_save(ps, p, &r));
    ATH_REQUIRE_OK(ath_prefix_load(ps, b2, 12, N, &r));   /* continued with b2[12] (p[12] already seen) */
    ATH_REQUIRE_OK(feed(&t.m, &r, b2, 12, 20, 20, NULL, &scratch));
    ATH_REQUIRE_OK(ath_prefix_save(ps, b2, &r));          /* b2[0, 20) */
    ath_prefix_close(ps);
    ATH_EXPECT_EQ_INT(snap_files(dir, NULL), 3);          /* p[0, 24), p[0, 12), b2[0, 20) */

    /* the folder's budget at open: the most recently used fits */
    const uint64_t one = 128 + 4 * 20 + expected_bytes(&t.m, ATH_DTYPE_BF16, ATH_DTYPE_BF16, 20);
    ath_prefix_opts tight = po;
    tight.disk_bytes = one;
    ATH_REQUIRE_OK(ath_prefix_open(&ps, &t.m, ATH_DTYPE_BF16, ATH_DTYPE_BF16, &tight));
    ATH_EXPECT_EQ_INT(stats(ps).disk_count, 1);
    ATH_EXPECT_EQ_INT(ath_prefix_find(ps, b2, N), 20);
    ath_prefix_close(ps);
    ATH_EXPECT_EQ_INT(snap_files(dir, NULL), 1);

    /* foreign: another model, other dtypes */
    ATH_REQUIRE_OK(ath_prefix_open(&ps, &other.m, ATH_DTYPE_BF16, ATH_DTYPE_BF16, &po));
    ATH_EXPECT_EQ_INT(stats(ps).disk_count, 0);
    ath_prefix_close(ps);
    ATH_REQUIRE_OK(ath_prefix_open(&ps, &t.m, ATH_DTYPE_BF16, ATH_DTYPE_I8, &po));
    ATH_EXPECT_EQ_INT(stats(ps).disk_count, 0);
    ath_prefix_close(ps);
    ATH_EXPECT_EQ_INT(snap_files(dir, NULL), 1);           /* left where they are */

    /* damaged: the header checks out, the payload does not; the load
     * refuses it, the state is untouched, the store forgets it */
    char path[600];
    ATH_REQUIRE(snap_files(dir, path) == 1);
    int fd = open(path, O_RDWR);
    ATH_REQUIRE(fd >= 0);
    const off_t end = lseek(fd, 0, SEEK_END);
    unsigned char c;
    ATH_REQUIRE(pread(fd, &c, 1, end - 1) == 1);
    c ^= 0x40;
    ATH_REQUIRE(pwrite(fd, &c, 1, end - 1) == 1);
    close(fd);
    ATH_REQUIRE_OK(ath_prefix_open(&ps, &t.m, ATH_DTYPE_BF16, ATH_DTYPE_BF16, &po));
    ATH_EXPECT_EQ_INT(ath_prefix_find(ps, b2, N), 20);
    ATH_REQUIRE_OK(ath_state_reset(&r, &t.m));
    ATH_EXPECT_STATUS(ath_prefix_load(ps, b2, 20, N, &r), ATH_ERR_NOT_FOUND);
    ATH_EXPECT_EQ_INT(r.n_past, 0);
    ATH_EXPECT_EQ_INT(ath_prefix_find(ps, b2, N), 0);
    ath_prefix_close(ps);
    ATH_EXPECT_EQ_INT(snap_files(dir, NULL), 0);

    /* truncated: not used at open */
    ATH_REQUIRE_OK(ath_prefix_open(&ps, &t.m, ATH_DTYPE_BF16, ATH_DTYPE_BF16, &po));
    ATH_REQUIRE_OK(ath_prefix_save(ps, p, &s));           /* s holds p[0, 32) */
    ath_prefix_close(ps);
    ATH_REQUIRE(snap_files(dir, path) == 1);
    ATH_REQUIRE(truncate(path, 200) == 0);
    ATH_REQUIRE_OK(ath_prefix_open(&ps, &t.m, ATH_DTYPE_BF16, ATH_DTYPE_BF16, &po));
    ATH_EXPECT_EQ_INT(stats(ps).disk_count, 0);
    ath_prefix_close(ps);

    remove_folder(dir);
    rmdir(sd.path);
    ath_arena_free(&scratch);
    tiny_close(&other);
    tiny_close(&t);
}

static const ath_test_case tests[] = {
    ATH_TEST_ENTRY(prefix_snapshot_roundtrip),
    ATH_TEST_ENTRY(prefix_store_memory),
    ATH_TEST_ENTRY(prefix_store_disk),
};

int main(int argc, char **argv)
{
    return ath_test_main(tests, ATH_ARRAY_LEN(tests), argc, argv);
}
