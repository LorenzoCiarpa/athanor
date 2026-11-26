/* prefix.c: snapshots of a sequence at prefixes of its tokens (step 14).
 * See prefix.h.
 *
 * The store is a short list of entries (a few per conversation at most),
 * scanned linearly. An entry that goes away is marked dead and the list is
 * compacted at the end of each call, so no call holds a pointer across a
 * move. Snapshot payloads are malloc'd: they are large, few, and freed one
 * at a time, outside any forward pass.
 */
#include "infer/prefix.h"

#include "core/buf.h"
#include "core/file.h"
#include "core/json.h"
#include "core/sha256.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define MAGIC    "ATHSNAP1"
#define VERSION  1u
#define HEADER   128u          /* bytes of a file's header        */
#define NAME_LEN 37            /* 32 hex digits and ".snap"        */

typedef struct {
    int32_t *ids;
    int32_t  n, n_draft;
    uint64_t bytes;            /* payload                                            */
    void    *data;             /* the payload in host memory, or NULL                */
    bool     on_disk;
    bool     dead;
    uint64_t used;             /* clock of the last save or load                     */
    int32_t  next[2];          /* tokens seen after the prefix, up to two different  */
    int32_t  n_next;
} entry;

struct ath_prefix_store {
    const ath_model *m;
    ath_dtype        act, kv;
    uint8_t          id[32];   /* the model and the dtypes                           */
    bool             has_dir;
    char             dir[ATH_PATH_MAX];
    uint64_t         ram_budget, disk_budget, ram_used, disk_used;
    int32_t          min_tokens;
    entry           *e;
    uint32_t         n, cap;
    uint64_t         clock;
    uint64_t         mtime;    /* the latest file time given or seen, in ns        */
    ath_prefix_stats st;
};

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

/* ---- XXH64 (Yann Collet's 64-bit hash): the payload's check, at memory speed ---- */

static const uint64_t P1 = 0x9E3779B185EBCA87ull, P2 = 0xC2B2AE3D27D4EB4Full, P3 = 0x165667B19E3779F9ull,
                      P4 = 0x85EBCA77C2B2AE63ull, P5 = 0x27D4EB2F165667C5ull;

static inline uint64_t rotl64(uint64_t x, int r)
{
    return (x << r) | (x >> (64 - r));
}

static inline uint64_t rd64(const uint8_t *p)
{
    uint64_t v;
    memcpy(&v, p, 8);
    return v;
}

static inline uint32_t rd32(const uint8_t *p)
{
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

static inline uint64_t xround(uint64_t acc, uint64_t in)
{
    acc += in * P2;
    return rotl64(acc, 31) * P1;
}

static inline uint64_t xmerge(uint64_t h, uint64_t v)
{
    h ^= xround(0, v);
    return h * P1 + P4;
}

static uint64_t xxh64(const void *data, size_t len)
{
    const uint8_t *p = data, *end = p + len;
    uint64_t h;
    if (len >= 32) {
        uint64_t v1 = P1 + P2, v2 = P2, v3 = 0, v4 = 0 - P1;
        do {
            v1 = xround(v1, rd64(p));
            v2 = xround(v2, rd64(p + 8));
            v3 = xround(v3, rd64(p + 16));
            v4 = xround(v4, rd64(p + 24));
            p += 32;
        } while ((size_t)(end - p) >= 32);
        h = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);
        h = xmerge(xmerge(xmerge(xmerge(h, v1), v2), v3), v4);
    } else {
        h = P5;
    }
    h += (uint64_t)len;
    for (; end - p >= 8; p += 8)
        h = rotl64(h ^ xround(0, rd64(p)), 27) * P1 + P4;
    if (end - p >= 4) {
        h = rotl64(h ^ (uint64_t)rd32(p) * P1, 23) * P2 + P3;
        p += 4;
    }
    for (; p < end; p++)
        h = rotl64(h ^ (uint64_t)*p * P5, 11) * P1;
    h ^= h >> 33;
    h *= P2;
    h ^= h >> 29;
    h *= P3;
    return h ^ (h >> 32);
}

/* ---- identity, names, headers ------------------------------------------------------ */

static void model_id(const ath_model *m, ath_dtype act, ath_dtype kv, uint8_t out[32])
{
    ath_sha256 c;
    ath_sha256_init(&c);
    static const char tag[] = "athanor prefix snapshots 1";
    ath_sha256_update(&c, tag, sizeof tag);
    const char *fam = m->family && m->family->name ? m->family->name : "";
    ath_sha256_update(&c, fam, strlen(fam) + 1);
    ath_arena a;
    ath_arena_init(&a, 0);
    ath_buf cfg;
    ath_buf_init(&cfg, &a);
    if (m->config && ath_json_write(m->config, &cfg) == ATH_OK)
        ath_sha256_update(&c, cfg.data, cfg.len);
    ath_arena_free(&a);
    for (uint32_t i = 0; i < m->ckpt.n_tensors; i++) {
        const ath_ckpt_tensor *t = &m->ckpt.tensors[i];
        ath_sha256_update(&c, t->name, strlen(t->name) + 1);
        const int64_t meta[3] = {t->t.dtype, t->t.ndim, (int64_t)t->bytes};
        ath_sha256_update(&c, meta, sizeof meta);
        ath_sha256_update(&c, t->t.shape, (size_t)t->t.ndim * sizeof t->t.shape[0]);
        const size_t edge = t->bytes < 4096 ? (size_t)t->bytes : 4096;
        if (t->t.data && edge) {
            ath_sha256_update(&c, t->t.data, edge);
            ath_sha256_update(&c, (const char *)t->t.data + t->bytes - edge, edge);
        }
    }
    for (uint32_t i = 0; i < m->n_states; i++) {
        const ath_state_spec *sp = &m->states[i];
        const int64_t v[5] = {sp->kind, sp->layer, sp->fixed, sp->per_token, sp->group};
        ath_sha256_update(&c, v, sizeof v);
    }
    const uint32_t dt[2] = {(uint32_t)act, (uint32_t)kv};
    ath_sha256_update(&c, dt, sizeof dt);
    ath_sha256_final(&c, out);
}

/* The file of the snapshot of ids[0, n): 32 hex digits of the hash of the
 * store's identity and the tokens, ".snap". */
static void file_name(const ath_prefix_store *ps, const int32_t *ids, int32_t n, char out[NAME_LEN + 1])
{
    ath_sha256 c;
    uint8_t d[32];
    char hex[65];
    ath_sha256_init(&c);
    ath_sha256_update(&c, ps->id, sizeof ps->id);
    ath_sha256_update(&c, &n, sizeof n);
    ath_sha256_update(&c, ids, (size_t)n * sizeof *ids);
    ath_sha256_final(&c, d);
    ath_sha256_hex(d, hex);
    memcpy(out, hex, 32);
    memcpy(out + 32, ".snap", 6);
}

static bool path_of(const ath_prefix_store *ps, const entry *e, char path[ATH_PATH_MAX])
{
    char name[NAME_LEN + 1];
    file_name(ps, e->ids, e->n, name);
    return ath_path_join(path, ATH_PATH_MAX, ps->dir, name) == ATH_OK;
}

static uint64_t file_bytes(const entry *e)
{
    return HEADER + (uint64_t)e->n * sizeof(int32_t) + e->bytes;
}

typedef struct {
    int32_t  n, n_draft;
    uint64_t bytes, hash, ids_hash;
} header;

/* Layout (little-endian): magic [0, 8), version and header size [8, 16),
 * identity [16, 48), act and kv dtypes [48, 56), n and n_draft [56, 64),
 * payload bytes [64, 72), payload hash [72, 80), tokens' hash [80, 88),
 * zeros, the hash of bytes [0, 120) at [120, 128). */
static void put_header(const ath_prefix_store *ps, const header *h, uint8_t b[HEADER])
{
    memset(b, 0, HEADER);
    memcpy(b, MAGIC, 8);
    const uint32_t v[4] = {VERSION, HEADER, (uint32_t)ps->act, (uint32_t)ps->kv};
    memcpy(b + 8, v, 8);
    memcpy(b + 16, ps->id, 32);
    memcpy(b + 48, v + 2, 8);
    memcpy(b + 56, &h->n, 4);
    memcpy(b + 60, &h->n_draft, 4);
    memcpy(b + 64, &h->bytes, 8);
    memcpy(b + 72, &h->hash, 8);
    memcpy(b + 80, &h->ids_hash, 8);
    const uint64_t self = xxh64(b, HEADER - 8);
    memcpy(b + HEADER - 8, &self, 8);
}

/* NULL if b is a header of this store, else why not. */
static const char *get_header(const ath_prefix_store *ps, const uint8_t b[HEADER], header *h)
{
    uint64_t self;
    uint32_t v[2];
    memcpy(&self, b + HEADER - 8, 8);
    memcpy(v, b + 8, 8);
    if (memcmp(b, MAGIC, 8))
        return "not a snapshot";
    if (self != xxh64(b, HEADER - 8))
        return "damaged header";
    if (v[0] != VERSION || v[1] != HEADER)
        return "another version";
    if (memcmp(b + 16, ps->id, 32))
        return "another model or other dtypes";
    memcpy(&h->n, b + 56, 4);
    memcpy(&h->n_draft, b + 60, 4);
    memcpy(&h->bytes, b + 64, 8);
    memcpy(&h->hash, b + 72, 8);
    memcpy(&h->ids_hash, b + 80, 8);
    if (h->n < 1 || h->n_draft < 0 || h->bytes > (1ull << 48))
        return "impossible sizes";
    return NULL;
}

/* ---- files ------------------------------------------------------------------------- */

static bool write_all(int fd, const void *p, size_t n)
{
    const char *c = p;
    while (n) {
        const ssize_t w = write(fd, c, n > (1u << 30) ? (1u << 30) : n);
        if (w < 0 && errno == EINTR)
            continue;
        if (w <= 0)
            return false;
        c += w;
        n -= (size_t)w;
    }
    return true;
}

static bool read_at(int fd, void *p, size_t n, uint64_t off)
{
    char *c = p;
    while (n) {
        const ssize_t r = pread(fd, c, n > (1u << 30) ? (1u << 30) : n, (off_t)off);
        if (r < 0 && errno == EINTR)
            continue;
        if (r <= 0)
            return false;
        c += r;
        n -= (size_t)r;
        off += (uint64_t)r;
    }
    return true;
}

/* Mark path used now: its time, later than every other file's of the
 * store (the kernel's clock is too coarse to order files written in a
 * row), gives the order of use after a restart. */
static void touch(ath_prefix_store *ps, const char *path)
{
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    uint64_t ns = (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
    if (ns <= ps->mtime)
        ns = ps->mtime + 1000;
    ps->mtime = ns;
    const struct timespec ts[2] = {{.tv_sec = (time_t)(ns / 1000000000ull), .tv_nsec = (long)(ns % 1000000000ull)},
                                   {.tv_sec = (time_t)(ns / 1000000000ull), .tv_nsec = (long)(ns % 1000000000ull)}};
    utimensat(AT_FDCWD, path, ts, 0);
}

/* Write e's file: a temporary name, synced, renamed, the folder synced. */
static bool write_file(ath_prefix_store *ps, entry *e)
{
    char path[ATH_PATH_MAX], tmp[ATH_PATH_MAX];
    if (!path_of(ps, e, path) || snprintf(tmp, sizeof tmp, "%s.%d.tmp", path, (int)getpid()) >= (int)sizeof tmp)
        return false;
    const header h = {e->n, e->n_draft, e->bytes, xxh64(e->data, e->bytes),
                      xxh64(e->ids, (size_t)e->n * sizeof *e->ids)};
    uint8_t hb[HEADER];
    put_header(ps, &h, hb);
    const int fd = open(tmp, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    bool ok = fd >= 0 && write_all(fd, hb, HEADER) && write_all(fd, e->ids, (size_t)e->n * sizeof *e->ids) &&
              write_all(fd, e->data, e->bytes) && fsync(fd) == 0;
    if (fd >= 0)
        ok = close(fd) == 0 && ok;
    ok = ok && rename(tmp, path) == 0;
    if (!ok) {
        ath_log(ATH_LOG_WARN, "prefix: cannot write %s: %s", path, strerror(errno));
        unlink(tmp);
        return false;
    }
    const int dfd = open(ps->dir, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (dfd >= 0) {
        fsync(dfd);
        close(dfd);
    }
    touch(ps, path);
    e->on_disk = true;
    ps->disk_used += file_bytes(e);
    return true;
}

/* e's payload from its file, checked against the header's hash. */
static ath_status read_file(ath_prefix_store *ps, const entry *e, void **out)
{
    char path[ATH_PATH_MAX];
    if (!path_of(ps, e, path))
        return ATH_ERR_RANGE;
    const int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0)
        return ATH_ERR_NOT_FOUND;
    uint8_t hb[HEADER];
    header h;
    void *buf = NULL;
    ath_status st = ATH_ERR_FORMAT;
    if (read_at(fd, hb, HEADER, 0) && !get_header(ps, hb, &h) && h.n == e->n && h.bytes == e->bytes) {
        if (!(buf = malloc(h.bytes ? h.bytes : 1)))
            st = ATH_ERR_NOMEM;
        else if (read_at(fd, buf, h.bytes, HEADER + (uint64_t)h.n * sizeof(int32_t)) && xxh64(buf, h.bytes) == h.hash)
            st = ATH_OK;
    }
    close(fd);
    if (st != ATH_OK) {
        free(buf);
        return st;
    }
    touch(ps, path);
    *out = buf;
    return ATH_OK;
}

/* ---- entries ----------------------------------------------------------------------- */

static bool begins(const entry *e, const int32_t *ids, int32_t n)
{
    return e->n <= n && memcmp(e->ids, ids, (size_t)e->n * sizeof *ids) == 0;
}

static entry *exact(ath_prefix_store *ps, const int32_t *ids, int32_t n)
{
    for (uint32_t i = 0; i < ps->n; i++)
        if (!ps->e[i].dead && ps->e[i].n == n && begins(&ps->e[i], ids, n))
            return &ps->e[i];
    return NULL;
}

static void note_next(entry *e, int32_t tok)
{
    for (int32_t i = 0; i < e->n_next; i++)
        if (e->next[i] == tok)
            return;
    if (e->n_next < 2)
        e->next[e->n_next++] = tok;
}

/* A longer snapshot begins with e's tokens, and they were only ever
 * followed one way. */
static bool superseded(const ath_prefix_store *ps, const entry *e)
{
    if (e->n_next >= 2)
        return false;
    for (uint32_t i = 0; i < ps->n; i++) {
        const entry *f = &ps->e[i];
        if (!f->dead && f != e && f->n > e->n && begins(e, f->ids, f->n))
            return true;
    }
    return false;
}

static void forget(ath_prefix_store *ps, entry *e)
{
    if (e->data)
        ps->ram_used -= e->bytes;
    free(e->data);
    free(e->ids);
    *e = (entry){.dead = true};
}

static void compact(ath_prefix_store *ps)
{
    uint32_t k = 0;
    for (uint32_t i = 0; i < ps->n; i++)
        if (!ps->e[i].dead)
            ps->e[k++] = ps->e[i];
    ps->n = k;
}

static void unlink_file(ath_prefix_store *ps, entry *e)
{
    char path[ATH_PATH_MAX];
    if (path_of(ps, e, path))
        unlink(path);
    ps->disk_used -= file_bytes(e);
    e->on_disk = false;
}

/* Room in the folder for need more bytes: delete the files of snapshots
 * used before `before`, the least recently used first, until it fits.
 * false if it does not (newer files are never deleted for older ones). */
static bool make_room(ath_prefix_store *ps, uint64_t need, uint64_t before)
{
    while (ps->disk_used + need > ps->disk_budget) {
        entry *v = NULL;
        for (uint32_t i = 0; i < ps->n; i++) {
            entry *e = &ps->e[i];
            if (!e->dead && e->on_disk && e->used < before && (!v || e->used < v->used))
                v = e;
        }
        if (!v)
            return false;
        unlink_file(ps, v);
        if (!v->data)
            forget(ps, v);
    }
    return true;
}

static bool to_disk(ath_prefix_store *ps, entry *e)
{
    return make_room(ps, file_bytes(e), e->used) && write_file(ps, e);
}

/* Take e's payload out of host memory: to disk first unless superseded. */
static void drop_ram(ath_prefix_store *ps, entry *e)
{
    if (ps->has_dir && !e->on_disk && !superseded(ps, e))
        to_disk(ps, e);
    free(e->data);
    e->data = NULL;
    ps->ram_used -= e->bytes;
    if (!e->on_disk)
        forget(ps, e);
}

/* Free the least recently used payloads until host memory fits its
 * budget, keep's last. */
static void ram_trim(ath_prefix_store *ps, entry *keep)
{
    while (ps->ram_used > ps->ram_budget) {
        entry *v = NULL;
        for (uint32_t i = 0; i < ps->n; i++) {
            entry *e = &ps->e[i];
            if (!e->dead && e->data && e != keep && (!v || e->used < v->used))
                v = e;
        }
        if (!v)
            break;
        drop_ram(ps, v);
    }
    if (ps->ram_used > ps->ram_budget && keep->data)
        drop_ram(ps, keep);
}

static entry *add(ath_prefix_store *ps)
{
    if (ps->n == ps->cap) {
        const uint32_t cap = ps->cap ? 2 * ps->cap : 16;
        entry *e = realloc(ps->e, cap * sizeof *e);
        if (!e)
            return NULL;
        ps->e = e;
        ps->cap = cap;
    }
    ps->e[ps->n] = (entry){0};
    return &ps->e[ps->n++];
}

/* ---- the folder ---------------------------------------------------------------------- */

static int by_used(const void *a, const void *b)
{
    const entry *x = a, *y = b;
    return x->used < y->used ? -1 : x->used > y->used;
}

/* Know the snapshots of this store in the folder: their headers and tokens. */
static ath_status scan(ath_prefix_store *ps)
{
    DIR *d = opendir(ps->dir);
    if (!d) {
        ath_log(ATH_LOG_ERROR, "prefix: cannot open the folder %s: %s", ps->dir, strerror(errno));
        return ATH_ERR_IO;
    }
    const struct dirent *de;
    uint32_t used = 0, skipped = 0;
    while ((de = readdir(d))) {
        if (strlen(de->d_name) != NAME_LEN || strcmp(de->d_name + 32, ".snap"))
            continue;
        char path[ATH_PATH_MAX];
        if (ath_path_join(path, sizeof path, ps->dir, de->d_name) != ATH_OK)
            continue;
        const int fd = open(path, O_RDONLY | O_CLOEXEC);
        struct stat sb;
        uint8_t hb[HEADER];
        header h;
        const char *why = NULL;
        int32_t *ids = NULL;
        if (fd < 0 || fstat(fd, &sb) != 0 || !read_at(fd, hb, HEADER, 0))
            why = "unreadable";
        else if (!(why = get_header(ps, hb, &h))) {
            const entry tmp = {.n = h.n, .bytes = h.bytes};
            if ((uint64_t)sb.st_size != file_bytes(&tmp))
                why = "truncated";
            else if (!(ids = malloc((size_t)h.n * sizeof *ids)))
                why = "out of memory";
            else if (!read_at(fd, ids, (size_t)h.n * sizeof *ids, HEADER) ||
                     xxh64(ids, (size_t)h.n * sizeof *ids) != h.ids_hash)
                why = "damaged tokens";
            if (!why) {
                char name[NAME_LEN + 1];
                file_name(ps, ids, h.n, name);
                if (strcmp(name, de->d_name))
                    why = "renamed";
            }
        }
        if (fd >= 0)
            close(fd);
        entry *e = why ? NULL : add(ps);
        if (!e) {
            free(ids);
            if (why && strcmp(why, "another model or other dtypes"))
                ath_log(ATH_LOG_WARN, "prefix: %s not used: %s", path, why ? why : "out of memory");
            skipped++;
            continue;
        }
        *e = (entry){.ids = ids, .n = h.n, .n_draft = h.n_draft, .bytes = h.bytes, .on_disk = true,
                     .used = (uint64_t)sb.st_mtim.tv_sec * 1000000000ull + (uint64_t)sb.st_mtim.tv_nsec};
        if (e->used > ps->mtime)
            ps->mtime = e->used;
        ps->disk_used += file_bytes(e);
        used++;
    }
    closedir(d);
    /* the files' times give the order of use; then the clock goes on from it */
    qsort(ps->e, ps->n, sizeof *ps->e, by_used);
    for (uint32_t i = 0; i < ps->n; i++)
        ps->e[i].used = ++ps->clock;
    /* how the prefixes on disk continue one another */
    for (uint32_t i = 0; i < ps->n; i++)
        for (uint32_t j = 0; j < ps->n; j++)
            if (ps->e[j].n > ps->e[i].n && begins(&ps->e[i], ps->e[j].ids, ps->e[j].n))
                note_next(&ps->e[i], ps->e[j].ids[ps->e[i].n]);
    make_room(ps, 0, UINT64_MAX);
    compact(ps);
    ath_log(ATH_LOG_INFO, "prefix: %u snapshots in %s (%.2f GiB)%s", used, ps->dir,
            (double)ps->disk_used / 1073741824.0, skipped ? "; others not this model's or not whole" : "");
    return ATH_OK;
}

/* ---- the store --------------------------------------------------------------------- */

ath_status ath_prefix_open(ath_prefix_store **out, const ath_model *m, ath_dtype act, ath_dtype kv,
                           const ath_prefix_opts *o)
{
    if (!out || !m)
        return ATH_ERR_INVALID_ARG;
    *out = NULL;
    const ath_prefix_opts def = {0};
    if (!o)
        o = &def;
    ath_prefix_store *ps = calloc(1, sizeof *ps);
    if (!ps)
        return ATH_ERR_NOMEM;
    ps->m = m;
    ps->act = act;
    ps->kv = kv;
    ps->ram_budget = o->ram_bytes ? o->ram_bytes : ATH_PREFIX_RAM_DEFAULT;
    ps->disk_budget = o->disk_bytes ? o->disk_bytes : ATH_PREFIX_DISK_DEFAULT;
    ps->min_tokens = o->min_tokens > 0 ? o->min_tokens : ATH_PREFIX_MIN_TOKENS;
    if (o->dir && o->dir[0]) {
        if (strlen(o->dir) >= sizeof ps->dir) {
            free(ps);
            return ATH_ERR_RANGE;
        }
        strcpy(ps->dir, o->dir);
        ps->has_dir = true;
        if (mkdir(ps->dir, 0755) != 0 && errno != EEXIST) {
            ath_log(ATH_LOG_ERROR, "prefix: cannot create the folder %s: %s", ps->dir, strerror(errno));
            free(ps);
            return ATH_ERR_IO;
        }
        model_id(m, act, kv, ps->id);
        const ath_status st = scan(ps);
        if (st != ATH_OK) {
            ath_prefix_close(ps);
            return st;
        }
    }
    *out = ps;
    return ATH_OK;
}

void ath_prefix_close(ath_prefix_store *ps)
{
    if (!ps)
        return;
    /* what must outlive the process, the least recently used first, so the
     * files' times keep the order of use: a newer snapshot makes room by
     * dropping older ones, never the other way round */
    uint32_t *order = ps->has_dir && ps->n ? malloc(ps->n * sizeof *order) : NULL;
    uint32_t k = 0;
    for (uint32_t i = 0; order && i < ps->n; i++) {
        const entry *e = &ps->e[i];
        if (!e->dead && e->data && !e->on_disk && !superseded(ps, e))
            order[k++] = i;
    }
    for (uint32_t a = 1; a < k; a++)   /* by use, oldest first (a few entries) */
        for (uint32_t b = a; b > 0 && ps->e[order[b]].used < ps->e[order[b - 1]].used; b--) {
            const uint32_t t = order[b];
            order[b] = order[b - 1];
            order[b - 1] = t;
        }
    for (uint32_t i = 0; i < k; i++)
        to_disk(ps, &ps->e[order[i]]);
    free(order);
    for (uint32_t i = 0; i < ps->n; i++)
        if (!ps->e[i].dead)
            forget(ps, &ps->e[i]);
    free(ps->e);
    free(ps);
}

int32_t ath_prefix_find(ath_prefix_store *ps, const int32_t *ids, int32_t n)
{
    int32_t best = 0;
    for (uint32_t i = 0; ps && ids && i < ps->n; i++) {
        const entry *e = &ps->e[i];
        if (!e->dead && e->n < n && e->n > best && begins(e, ids, n))
            best = e->n;
    }
    return best;
}

ath_status ath_prefix_load(ath_prefix_store *ps, const int32_t *ids, int32_t len, int32_t n, ath_state *s)
{
    if (!ps || !ids || !s || len < 1)
        return ATH_ERR_INVALID_ARG;
    entry *e = exact(ps, ids, len);
    if (!e)
        return ATH_ERR_NOT_FOUND;
    const double t0 = now();
    void *data = e->data;
    const bool disk = !data;
    if (disk) {
        const ath_status st = read_file(ps, e, &data);
        if (st != ATH_OK) {
            ath_log(ATH_LOG_WARN, "prefix: the snapshot of %d tokens does not read back whole (%s); forgotten", e->n,
                    ath_status_str(st));
            unlink_file(ps, e);
            forget(ps, e);
            compact(ps);
            return ATH_ERR_NOT_FOUND;
        }
    }
    const ath_status st = ath_state_snap_load(s, ps->m, e->n, e->n_draft, data, e->bytes);
    if (disk && st == ATH_OK) {   /* kept in host memory too, as far as the budget goes */
        e->data = data;
        ps->ram_used += e->bytes;
    } else if (disk) {
        free(data);
    }
    if (st != ATH_OK)
        return st;
    if (len < n)
        note_next(e, ids[len]);
    e->used = ++ps->clock;
    ps->st.load_tokens = e->n;
    ps->st.load_bytes = e->bytes;
    ps->st.load_s = now() - t0;
    ps->st.load_disk = disk;
    if (disk)
        ram_trim(ps, e);
    compact(ps);
    return ATH_OK;
}

ath_status ath_prefix_save(ath_prefix_store *ps, const int32_t *ids, const ath_state *s)
{
    if (!ps || !ids || !s)
        return ATH_ERR_INVALID_ARG;
    const int32_t n = s->n_past;
    if (n < ps->min_tokens)
        return ATH_OK;
    entry *same = exact(ps, ids, n);
    if (same) {
        same->used = ++ps->clock;
        return ATH_OK;
    }
    const double t0 = now();
    const size_t bytes = ath_state_snap_bytes(ps->m, s);
    void *data = malloc(bytes ? bytes : 1);
    int32_t *cp = malloc((size_t)n * sizeof *cp);
    ath_status st = data && cp ? ath_state_snap_save(s, ps->m, data, bytes) : ATH_ERR_NOMEM;
    entry *e = st == ATH_OK ? add(ps) : NULL;
    if (!e) {
        free(data);
        free(cp);
        return st != ATH_OK ? st : ATH_ERR_NOMEM;
    }
    memcpy(cp, ids, (size_t)n * sizeof *cp);
    *e = (entry){.ids = cp, .n = n, .n_draft = s->n_draft, .bytes = bytes, .data = data, .used = ++ps->clock};
    ps->ram_used += bytes;
    /* how this prefix and the others continue one another */
    for (uint32_t i = 0; i < ps->n; i++) {
        entry *f = &ps->e[i];
        if (f->dead || f == e)
            continue;
        if (f->n > n && begins(e, f->ids, f->n))
            note_next(e, f->ids[n]);
        else if (f->n < n && begins(f, ids, n))
            note_next(f, ids[f->n]);
    }
    ps->st.save_tokens = n;
    ps->st.save_bytes = bytes;
    ps->st.save_s = now() - t0;
    ram_trim(ps, e);
    compact(ps);
    return ATH_OK;
}

void ath_prefix_get_stats(const ath_prefix_store *ps, ath_prefix_stats *out)
{
    if (!out)
        return;
    *out = ps ? ps->st : (ath_prefix_stats){0};
    for (uint32_t i = 0; ps && i < ps->n; i++) {
        const entry *e = &ps->e[i];
        if (e->dead)
            continue;
        out->ram_count += e->data != NULL;
        out->disk_count += e->on_disk;
    }
    if (ps) {
        out->ram_bytes = ps->ram_used;
        out->disk_bytes = ps->disk_used;
    }
}
