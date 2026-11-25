/* safetensors.c: single-file and sharded safetensors readers. See safetensors.h. */
#include "core/safetensors.h"

#include <string.h>

static bool is_digit(unsigned char c) { return c >= '0' && c <= '9'; }

int ath_safetensors_name_cmp(const char *a, const char *b)
{
    const unsigned char *x = (const unsigned char *)a, *y = (const unsigned char *)b;
    while (*x && *y) {
        if (is_digit(*x) && is_digit(*y)) {
            /* Compare digit runs as numbers: drop leading zeros, then the
             * longer run is larger, then compare digit by digit. */
            while (*x == '0' && is_digit(x[1]))
                x++;
            while (*y == '0' && is_digit(y[1]))
                y++;
            const unsigned char *xe = x, *ye = y;
            while (is_digit(*xe))
                xe++;
            while (is_digit(*ye))
                ye++;
            size_t lx = (size_t)(xe - x), ly = (size_t)(ye - y);
            if (lx != ly)
                return lx < ly ? -1 : 1;
            int c = memcmp(x, y, lx);
            if (c)
                return c < 0 ? -1 : 1;
            x = xe;
            y = ye;
            continue;
        }
        if (*x != *y)
            return *x < *y ? -1 : 1;
        x++;
        y++;
    }
    if (*x || *y)
        return *x ? 1 : -1;
    /* Equal as numbers ("01" and "1"): fall back to bytes for a total order. */
    int c = strcmp(a, b);
    return c < 0 ? -1 : (c > 0 ? 1 : 0);
}

static int cmp_entry_name(const void *a, const void *b, void *ctx)
{
    (void)ctx;
    return ath_safetensors_name_cmp(((const ath_safetensors_entry *)a)->name,
                                    ((const ath_safetensors_entry *)b)->name);
}

static int cmp_str_ptr(const void *a, const void *b, void *ctx)
{
    (void)ctx;
    return ath_safetensors_name_cmp(*(const char *const *)a, *(const char *const *)b);
}

/* Order entry indices by (shard, begin, end) to check data coverage. */
static int cmp_entry_offset(const void *a, const void *b, void *ctx)
{
    const ath_safetensors_entry *e = ctx;
    const ath_safetensors_entry *x = &e[*(const uint32_t *)a], *y = &e[*(const uint32_t *)b];
    if (x->shard != y->shard)
        return x->shard < y->shard ? -1 : 1;
    if (x->begin != y->begin)
        return x->begin < y->begin ? -1 : 1;
    if (x->end != y->end)
        return x->end < y->end ? -1 : 1;
    return 0;
}

/* ---- dtypes ------------------------------------------------------------------ */

static const struct {
    const char *name;
    ath_dtype   dtype;
} st_dtypes[] = {
    {"F32", ATH_DTYPE_F32},         {"F16", ATH_DTYPE_F16},
    {"BF16", ATH_DTYPE_BF16},       {"I32", ATH_DTYPE_I32},
    {"I64", ATH_DTYPE_I64},         {"U8", ATH_DTYPE_U8},
    {"F8_E4M3", ATH_DTYPE_F8_E4M3}, {"F8_E5M2", ATH_DTYPE_F8_E5M2},
};

static bool dtype_from_str(const char *s, ath_dtype *out)
{
    for (size_t i = 0; i < ATH_ARRAY_LEN(st_dtypes); i++) {
        if (strcmp(s, st_dtypes[i].name) == 0) {
            *out = st_dtypes[i].dtype;
            return true;
        }
    }
    return false;
}

/* ---- one shard --------------------------------------------------------------- */

static bool is_metadata(const ath_json *member)
{
    return member->key_len == 12 && memcmp(member->key, "__metadata__", 12) == 0;
}

/* Map the file and parse its header. The caller closes sh->file on failure. */
static ath_status open_shard(ath_safetensors_shard *sh, ath_arena *a, const char *path)
{
    sh->path = ath_arena_strndup(a, path, strlen(path));
    if (!sh->path)
        return ATH_ERR_NOMEM;
    ATH_CHECK(ath_file_open(&sh->file, path));

    const unsigned char *p = sh->file.data;
    if (sh->file.size < 8) {
        ath_log(ATH_LOG_ERROR, "%s: file too small for a safetensors header", path);
        return ATH_ERR_FORMAT;
    }
    uint64_t n = 0;
    for (int i = 7; i >= 0; i--) /* little-endian, independent of the host */
        n = (n << 8) | p[i];
    if (n > sh->file.size - 8) {
        ath_log(ATH_LOG_ERROR, "%s: header length %llu exceeds the file size %zu", path,
                (unsigned long long)n, sh->file.size);
        return ATH_ERR_FORMAT;
    }
    sh->header_bytes = n;
    sh->data_offset = 8 + n;
    sh->data_bytes = sh->file.size - sh->data_offset;

    ath_json_error err;
    ath_status s = ath_json_parse(a, (const char *)p + 8, (size_t)n, &sh->header, &err);
    if (s != ATH_OK) {
        ath_log(ATH_LOG_ERROR, "%s: header JSON, line %zu column %zu: %s", path, err.line,
                err.col, err.msg);
        return s;
    }
    if (sh->header->type != ATH_JSON_OBJECT) {
        ath_log(ATH_LOG_ERROR, "%s: header is not a JSON object", path);
        return ATH_ERR_FORMAT;
    }
    return ATH_OK;
}

static ath_status read_u64(const ath_json *n, uint64_t *out)
{
    int64_t v;
    ATH_CHECK(ath_json_to_int(n, &v));
    if (v < 0)
        return ATH_ERR_RANGE;
    *out = (uint64_t)v;
    return ATH_OK;
}

/* Turn one header member into an entry with a zero-copy view. */
static ath_status fill_entry(ath_safetensors_entry *e, const ath_safetensors_shard *sh,
                             uint32_t shard, const ath_json *m)
{
    const char *path = sh->path;
    e->name = m->key;
    e->shard = shard;
    if (strlen(m->key) != m->key_len || m->key_len == 0) {
        ath_log(ATH_LOG_ERROR, "%s: empty tensor name or name with a NUL byte", path);
        return ATH_ERR_FORMAT;
    }
    if (m->type != ATH_JSON_OBJECT) {
        ath_log(ATH_LOG_ERROR, "%s: %s: entry is not an object", path, m->key);
        return ATH_ERR_FORMAT;
    }

    const char *dts;
    ath_dtype dtype;
    if (ath_json_to_str(ath_json_get(m, "dtype"), &dts, NULL) != ATH_OK) {
        ath_log(ATH_LOG_ERROR, "%s: %s: missing dtype", path, m->key);
        return ATH_ERR_FORMAT;
    }
    if (!dtype_from_str(dts, &dtype)) {
        ath_log(ATH_LOG_ERROR, "%s: %s: dtype %s is not supported", path, m->key, dts);
        return ATH_ERR_UNSUPPORTED;
    }

    const ath_json *shape = ath_json_get(m, "shape");
    if (!shape || shape->type != ATH_JSON_ARRAY) {
        ath_log(ATH_LOG_ERROR, "%s: %s: missing shape", path, m->key);
        return ATH_ERR_FORMAT;
    }
    size_t ndim = ath_json_count(shape);
    if (ndim > ATH_MAX_DIMS) {
        ath_log(ATH_LOG_ERROR, "%s: %s: %zu dimensions, at most %d supported", path, m->key,
                ndim, ATH_MAX_DIMS);
        return ATH_ERR_UNSUPPORTED;
    }
    int64_t dims[ATH_MAX_DIMS];
    size_t k = 0;
    for (const ath_json *d = ath_json_first(shape); d; d = d->next, k++) {
        if (ath_json_to_int(d, &dims[k]) != ATH_OK || dims[k] < 0) {
            ath_log(ATH_LOG_ERROR, "%s: %s: shape entries must be non-negative integers", path,
                    m->key);
            return ATH_ERR_FORMAT;
        }
    }

    const ath_json *offs = ath_json_get(m, "data_offsets");
    if (ath_json_count(offs) != 2 || offs->type != ATH_JSON_ARRAY ||
        read_u64(ath_json_at(offs, 0), &e->begin) != ATH_OK ||
        read_u64(ath_json_at(offs, 1), &e->end) != ATH_OK) {
        ath_log(ATH_LOG_ERROR, "%s: %s: data_offsets must be two non-negative integers", path,
                m->key);
        return ATH_ERR_FORMAT;
    }
    if (e->begin > e->end || e->end > sh->data_bytes) {
        ath_log(ATH_LOG_ERROR, "%s: %s: data_offsets [%llu, %llu] outside the data section (%llu bytes)",
                path, m->key, (unsigned long long)e->begin, (unsigned long long)e->end,
                (unsigned long long)sh->data_bytes);
        return ATH_ERR_FORMAT;
    }

    void *data = (char *)sh->file.data + sh->data_offset + e->begin;
    if (ath_tensor_init(&e->t, dtype, (int32_t)ndim, dims, data, ATH_TIER_RAM) != ATH_OK) {
        ath_log(ATH_LOG_ERROR, "%s: %s: shape too large", path, m->key);
        return ATH_ERR_FORMAT;
    }
    if ((uint64_t)ath_tensor_nbytes(&e->t) != e->end - e->begin) {
        ath_log(ATH_LOG_ERROR, "%s: %s: shape and dtype give %zu bytes, data_offsets give %llu",
                path, m->key, ath_tensor_nbytes(&e->t), (unsigned long long)(e->end - e->begin));
        return ATH_ERR_FORMAT;
    }
    /* The format does not require alignment, but kernels read elements in
     * place: a misaligned tensor would need a copy before use. */
    if ((sh->data_offset + e->begin) % ath_dtype_size(dtype) != 0)
        ath_log(ATH_LOG_WARN, "%s: %s: data not aligned to its element size", path, m->key);
    return ATH_OK;
}

/* ---- all shards -------------------------------------------------------------- */

/* Build, sort and validate the tensor table from the parsed shard headers. */
static ath_status load_entries(ath_safetensors *st, ath_arena *a)
{
    uint64_t total = 0;
    for (uint32_t s = 0; s < st->n_shards; s++)
        for (const ath_json *m = ath_json_first(st->shards[s].header); m; m = m->next)
            total += !is_metadata(m);
    if (total > UINT32_MAX)
        return ATH_ERR_UNSUPPORTED;

    st->tensors = ath_arena_alloc(a, (size_t)total * sizeof *st->tensors,
                                  _Alignof(ath_safetensors_entry));
    if (!st->tensors)
        return ATH_ERR_NOMEM;
    uint32_t n = 0;
    for (uint32_t s = 0; s < st->n_shards; s++) {
        for (const ath_json *m = ath_json_first(st->shards[s].header); m; m = m->next) {
            if (is_metadata(m))
                continue;
            ATH_CHECK(fill_entry(&st->tensors[n], &st->shards[s], s, m));
            st->tensor_bytes += st->tensors[n].end - st->tensors[n].begin;
            n++;
        }
        st->data_bytes += st->shards[s].data_bytes;
    }
    st->n_tensors = n;

    ath_arena_pos scratch = ath_arena_mark(a);
    ath_status status = ATH_OK;
    void *tmp = ath_arena_alloc(a, sizeof(ath_safetensors_entry), _Alignof(ath_safetensors_entry));
    uint32_t *order = ath_arena_alloc(a, (size_t)n * sizeof *order + 1, _Alignof(uint32_t));
    if (!tmp || !order) {
        status = ATH_ERR_NOMEM;
        goto done;
    }

    ath_sort(st->tensors, n, sizeof *st->tensors, cmp_entry_name, NULL, tmp);
    for (uint32_t i = 1; i < n; i++) {
        if (strcmp(st->tensors[i - 1].name, st->tensors[i].name) == 0) {
            ath_log(ATH_LOG_ERROR, "tensor %s appears more than once", st->tensors[i].name);
            status = ATH_ERR_FORMAT;
            goto done;
        }
    }

    /* Coverage: walking each shard's tensors by offset, every tensor must
     * start where the previous one ended, and the last must end at the end
     * of the data section. This also rules out overlapping tensors. */
    for (uint32_t i = 0; i < n; i++)
        order[i] = i;
    ath_sort(order, n, sizeof *order, cmp_entry_offset, st->tensors, tmp);
    uint32_t i = 0;
    for (uint32_t s = 0; s < st->n_shards; s++) {
        uint64_t expect = 0;
        for (; i < n && st->tensors[order[i]].shard == s; i++) {
            const ath_safetensors_entry *e = &st->tensors[order[i]];
            if (e->begin != expect) {
                ath_log(ATH_LOG_ERROR, "%s: %s starts at %llu, expected %llu (%s)",
                        st->shards[s].path, e->name, (unsigned long long)e->begin,
                        (unsigned long long)expect,
                        e->begin < expect ? "overlap" : "gap in the data section");
                status = ATH_ERR_FORMAT;
                goto done;
            }
            expect = e->end;
        }
        if (expect != st->shards[s].data_bytes) {
            ath_log(ATH_LOG_ERROR, "%s: tensors cover %llu of %llu data bytes",
                    st->shards[s].path, (unsigned long long)expect,
                    (unsigned long long)st->shards[s].data_bytes);
            status = ATH_ERR_FORMAT;
            goto done;
        }
    }
done:
    ath_arena_reset(a, scratch);
    return status;
}

static ath_status open_single(ath_safetensors *st, ath_arena *a, const char *path)
{
    st->shards = ath_arena_zalloc(a, sizeof *st->shards, _Alignof(ath_safetensors_shard));
    if (!st->shards)
        return ATH_ERR_NOMEM;
    st->shards[0].file.fd = -1;
    st->n_shards = 1;
    ATH_CHECK(open_shard(&st->shards[0], a, path));
    return load_entries(st, a);
}

static ath_status open_index(ath_safetensors *st, ath_arena *a, const char *dir,
                             const char *index_path)
{
    /* The index is small: parse it, keep the tree in the arena, unmap it. */
    ath_file f;
    ATH_CHECK(ath_file_open(&f, index_path));
    const ath_json *idx;
    ath_json_error err;
    ath_status s = ath_json_parse(a, f.data, f.size, &idx, &err);
    ath_file_close(&f);
    if (s != ATH_OK) {
        ath_log(ATH_LOG_ERROR, "%s: line %zu column %zu: %s", index_path, err.line, err.col,
                err.msg);
        return s;
    }
    const ath_json *wmap = ath_json_get(idx, "weight_map");
    if (!wmap || wmap->type != ATH_JSON_OBJECT) {
        ath_log(ATH_LOG_ERROR, "%s: missing weight_map object", index_path);
        return ATH_ERR_FORMAT;
    }
    int64_t total_size;
    if (ath_json_to_int(ath_json_path(idx, "metadata.total_size"), &total_size) == ATH_OK)
        st->index_total_size = total_size;

    /* Distinct shard file names, in natural order. */
    size_t n_map = ath_json_count(wmap);
    const char **names = ath_arena_alloc(a, (n_map + 1) * sizeof *names, _Alignof(const char *));
    if (!names)
        return ATH_ERR_NOMEM;
    uint32_t n_shards = 0;
    for (const ath_json *m = ath_json_first(wmap); m; m = m->next) {
        const char *file;
        size_t len;
        if (ath_json_to_str(m, &file, &len) != ATH_OK || len == 0 || strlen(file) != len ||
            strchr(file, '/')) {
            ath_log(ATH_LOG_ERROR, "%s: %s: shard must be a plain file name", index_path, m->key);
            return ATH_ERR_FORMAT;
        }
        uint32_t k = 0;
        while (k < n_shards && strcmp(names[k], file) != 0)
            k++;
        if (k == n_shards)
            names[n_shards++] = file;
    }
    const char *tmp;
    ath_sort(names, n_shards, sizeof *names, cmp_str_ptr, NULL, &tmp);

    st->shards = ath_arena_zalloc(a, (n_shards + 1) * sizeof *st->shards,
                                  _Alignof(ath_safetensors_shard));
    if (!st->shards)
        return ATH_ERR_NOMEM;
    for (uint32_t k = 0; k < n_shards; k++)
        st->shards[k].file.fd = -1;
    st->n_shards = n_shards;
    for (uint32_t k = 0; k < n_shards; k++) {
        char path[ATH_PATH_MAX];
        ATH_CHECK(ath_path_join(path, sizeof path, dir, names[k]));
        ATH_CHECK(open_shard(&st->shards[k], a, path));
    }
    ATH_CHECK(load_entries(st, a));

    /* The index and the shard headers must describe the same tensors:
     * every weight_map entry resolves to a tensor in the named shard, no
     * tensor is claimed twice, and no shard holds a tensor the index omits. */
    ath_arena_pos scratch = ath_arena_mark(a);
    unsigned char *seen = ath_arena_zalloc(a, (size_t)st->n_tensors + 1, 1);
    if (!seen)
        return ATH_ERR_NOMEM;
    ath_status status = ATH_OK;
    for (const ath_json *m = ath_json_first(wmap); m && status == ATH_OK; m = m->next) {
        const ath_safetensors_entry *e = ath_safetensors_find(st, m->key);
        const char *base = e ? strrchr(st->shards[e->shard].path, '/') : NULL;
        base = base ? base + 1 : (e ? st->shards[e->shard].path : NULL);
        if (!e) {
            ath_log(ATH_LOG_ERROR, "%s lists %s, which no shard contains", index_path, m->key);
            status = ATH_ERR_FORMAT;
        } else if (strcmp(base, m->u.str.ptr) != 0) {
            ath_log(ATH_LOG_ERROR, "%s puts %s in %s, but it is in %s", index_path, m->key,
                    m->u.str.ptr, base);
            status = ATH_ERR_FORMAT;
        } else if (seen[e - st->tensors]++) {
            ath_log(ATH_LOG_ERROR, "%s lists %s more than once", index_path, m->key);
            status = ATH_ERR_FORMAT;
        }
    }
    for (uint32_t i = 0; i < st->n_tensors && status == ATH_OK; i++) {
        if (!seen[i]) {
            ath_log(ATH_LOG_ERROR, "%s: tensor %s is missing from %s",
                    st->shards[st->tensors[i].shard].path, st->tensors[i].name, index_path);
            status = ATH_ERR_FORMAT;
        }
    }
    ath_arena_reset(a, scratch);
    ATH_CHECK(status);

    if (st->index_total_size >= 0 && (uint64_t)st->index_total_size != st->tensor_bytes)
        ath_log(ATH_LOG_WARN, "%s: metadata.total_size is %lld, tensors hold %llu bytes",
                index_path, (long long)st->index_total_size,
                (unsigned long long)st->tensor_bytes);
    return ATH_OK;
}

/* ---- public ------------------------------------------------------------------ */

static void reset(ath_safetensors *st)
{
    memset(st, 0, sizeof *st);
    st->index_total_size = -1;
}

ath_status ath_safetensors_open_file(ath_safetensors *st, ath_arena *a, const char *path)
{
    reset(st);
    ath_status s = open_single(st, a, path);
    if (s != ATH_OK)
        ath_safetensors_close(st);
    return s;
}

ath_status ath_safetensors_open(ath_safetensors *st, ath_arena *a, const char *dir)
{
    reset(st);
    char path[ATH_PATH_MAX];
    ATH_CHECK(ath_path_join(path, sizeof path, dir, "model.safetensors.index.json"));
    ath_status s;
    if (ath_file_exists(path)) {
        s = open_index(st, a, dir, path);
    } else {
        ATH_CHECK(ath_path_join(path, sizeof path, dir, "model.safetensors"));
        s = open_single(st, a, path);
    }
    if (s != ATH_OK)
        ath_safetensors_close(st);
    return s;
}

const ath_safetensors_entry *ath_safetensors_find(const ath_safetensors *st, const char *name)
{
    uint32_t lo = 0, hi = st->n_tensors;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        int c = ath_safetensors_name_cmp(st->tensors[mid].name, name);
        if (c == 0)
            return &st->tensors[mid];
        if (c < 0)
            lo = mid + 1;
        else
            hi = mid;
    }
    return NULL;
}

void ath_safetensors_close(ath_safetensors *st)
{
    for (uint32_t s = 0; s < st->n_shards; s++)
        ath_file_close(&st->shards[s].file);
}
