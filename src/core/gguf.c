/* gguf.c: GGUF reader. See gguf.h. */
#include "core/gguf.h"

#include "core/safetensors.h"

#include <string.h>

#define GGUF_MAGIC 0x46554747u   /* "GGUF" read as a little-endian u32 */

/* ---- a bounded cursor over the mapping ----------------------------------------- */

typedef struct {
    const unsigned char *p;      /* start of the file             */
    uint64_t             size;
    uint64_t             at;     /* next byte to read             */
    const char          *path;
    bool                 bad;    /* a read ran past the end       */
} cursor;

static bool take(cursor *c, uint64_t n, const unsigned char **out)
{
    if (c->bad || n > c->size - c->at) {
        c->bad = true;
        return false;
    }
    *out = c->p + c->at;
    c->at += n;
    return true;
}

static uint64_t rd_le(cursor *c, unsigned n)
{
    const unsigned char *b;
    if (!take(c, n, &b))
        return 0;
    uint64_t v = 0;
    for (unsigned i = n; i-- > 0;)  /* little-endian, independent of the host */
        v = (v << 8) | b[i];
    return v;
}

static uint32_t rd_u32(cursor *c) { return (uint32_t)rd_le(c, 4); }
static uint64_t rd_u64(cursor *c) { return rd_le(c, 8); }

/* A string: its bytes in the mapping and their count. */
static bool rd_str(cursor *c, const char **ptr, uint64_t *len)
{
    *len = rd_u64(c);
    const unsigned char *b;
    if (!take(c, *len, &b))
        return false;
    *ptr = (const char *)b;
    return true;
}

static ath_status copy_str(ath_arena *a, const char *p, uint64_t n, const char **out)
{
    if (n > SIZE_MAX - 1)
        return ATH_ERR_RANGE;
    char *s = ath_arena_strndup(a, p, (size_t)n);
    if (!s)
        return ATH_ERR_NOMEM;
    *out = s;
    return ATH_OK;
}

/* Bytes of one scalar of type t; 0 for strings, arrays and unknown types. */
static uint64_t scalar_bytes(uint32_t t)
{
    switch (t) {
    case ATH_GGUF_U8: case ATH_GGUF_I8: case ATH_GGUF_BOOL: return 1;
    case ATH_GGUF_U16: case ATH_GGUF_I16: return 2;
    case ATH_GGUF_U32: case ATH_GGUF_I32: case ATH_GGUF_F32: return 4;
    case ATH_GGUF_U64: case ATH_GGUF_I64: case ATH_GGUF_F64: return 8;
    }
    return 0;
}

static ath_status read_scalar(cursor *c, uint32_t t, ath_gguf_kv *kv)
{
    const uint64_t n = scalar_bytes(t);
    const uint64_t raw = rd_le(c, (unsigned)n);
    switch (t) {
    case ATH_GGUF_U8: case ATH_GGUF_U16: case ATH_GGUF_U32: case ATH_GGUF_U64:
        kv->v.u = raw;
        break;
    case ATH_GGUF_BOOL:
        if (raw > 1)
            return ATH_ERR_FORMAT;
        kv->v.u = raw;
        break;
    case ATH_GGUF_I8:  kv->v.i = (int8_t)raw; break;
    case ATH_GGUF_I16: kv->v.i = (int16_t)raw; break;
    case ATH_GGUF_I32: kv->v.i = (int32_t)raw; break;
    case ATH_GGUF_I64: kv->v.i = (int64_t)raw; break;
    case ATH_GGUF_F32: {
        uint32_t u = (uint32_t)raw;
        float f;
        memcpy(&f, &u, sizeof f);
        kv->v.f = f;
        break;
    }
    case ATH_GGUF_F64: {
        double f;
        memcpy(&f, &raw, sizeof f);
        kv->v.f = f;
        break;
    }
    }
    return ATH_OK;
}

/* Skip an array's elements, checking that they lie inside the file. Arrays
 * of arrays are allowed by the format; athanor needs none and refuses them. */
static ath_status skip_array(cursor *c, uint32_t et, uint64_t count)
{
    if (et == ATH_GGUF_STRING) {
        for (uint64_t i = 0; i < count && !c->bad; i++) {
            const char *p;
            uint64_t len;
            rd_str(c, &p, &len);
        }
        return ATH_OK;
    }
    const uint64_t eb = scalar_bytes(et);
    if (eb == 0)
        return et == ATH_GGUF_ARRAY ? ATH_ERR_UNSUPPORTED : ATH_ERR_FORMAT;
    if (count > UINT64_MAX / eb) {
        c->bad = true;
        return ATH_OK;
    }
    const unsigned char *b;
    take(c, count * eb, &b);
    return ATH_OK;
}

static ath_status read_kv(cursor *c, ath_arena *a, ath_gguf_kv *kv)
{
    const char *key;
    uint64_t klen;
    if (!rd_str(c, &key, &klen))
        return ATH_ERR_FORMAT;
    if (klen == 0 || memchr(key, 0, (size_t)klen)) {
        ath_log(ATH_LOG_ERROR, "%s: empty metadata key or key with a NUL byte", c->path);
        return ATH_ERR_FORMAT;
    }
    ATH_CHECK(copy_str(a, key, klen, &kv->key));
    const uint32_t t = rd_u32(c);
    kv->type = (ath_gguf_type)t;
    if (t == ATH_GGUF_STRING) {
        const char *p;
        uint64_t len;
        if (!rd_str(c, &p, &len))
            return ATH_ERR_FORMAT;
        kv->v.str.len = (size_t)len;
        return copy_str(a, p, len, &kv->v.str.ptr);
    }
    if (t == ATH_GGUF_ARRAY) {
        kv->v.arr.type = (ath_gguf_type)rd_u32(c);
        kv->v.arr.count = rd_u64(c);
        kv->v.arr.data = c->p + c->at;
        ath_status s = skip_array(c, kv->v.arr.type, kv->v.arr.count);
        if (s != ATH_OK)
            ath_log(ATH_LOG_ERROR, "%s: %s: array of type %u is not supported", c->path, kv->key,
                    (unsigned)kv->v.arr.type);
        return s;
    }
    if (scalar_bytes(t) == 0) {
        ath_log(ATH_LOG_ERROR, "%s: %s: unknown value type %u", c->path, kv->key, (unsigned)t);
        return ATH_ERR_UNSUPPORTED;
    }
    ath_status s = read_scalar(c, t, kv);
    if (s != ATH_OK)
        ath_log(ATH_LOG_ERROR, "%s: %s: invalid bool", c->path, kv->key);
    return s;
}

/* ---- tensors ------------------------------------------------------------------- */

static bool tensor_dtype(uint32_t ggml_type, ath_dtype *out)
{
    switch (ggml_type) {
    case 0:  *out = ATH_DTYPE_F32; return true;
    case 1:  *out = ATH_DTYPE_F16; return true;
    case 8:  *out = ATH_DTYPE_Q8_0; return true;
    case 30: *out = ATH_DTYPE_BF16; return true;
    }
    return false;
}

static ath_status read_tensor(cursor *c, ath_arena *a, ath_gguf_tensor *t)
{
    const char *name;
    uint64_t nlen;
    if (!rd_str(c, &name, &nlen))
        return ATH_ERR_FORMAT;
    if (nlen == 0 || memchr(name, 0, (size_t)nlen)) {
        ath_log(ATH_LOG_ERROR, "%s: empty tensor name or name with a NUL byte", c->path);
        return ATH_ERR_FORMAT;
    }
    ATH_CHECK(copy_str(a, name, nlen, &t->name));
    const uint32_t nd = rd_u32(c);
    if (c->bad)
        return ATH_ERR_FORMAT;
    if (nd > ATH_MAX_DIMS) {
        ath_log(ATH_LOG_ERROR, "%s: %s: %u dimensions, at most %d supported", c->path, t->name, nd,
                ATH_MAX_DIMS);
        return ATH_ERR_UNSUPPORTED;
    }
    int64_t shape[ATH_MAX_DIMS];
    for (uint32_t i = 0; i < nd; i++) {
        uint64_t ne = rd_u64(c);
        if (ne > (uint64_t)INT64_MAX) {
            ath_log(ATH_LOG_ERROR, "%s: %s: dimension too large", c->path, t->name);
            return ATH_ERR_FORMAT;
        }
        shape[nd - 1 - i] = (int64_t)ne;   /* ne[0] is the fastest: last in row-major */
    }
    t->ggml_type = rd_u32(c);
    t->offset = rd_u64(c);
    if (c->bad)
        return ATH_ERR_FORMAT;
    ath_dtype dt;
    if (!tensor_dtype(t->ggml_type, &dt)) {
        ath_log(ATH_LOG_ERROR, "%s: %s: ggml type %u is not supported (F32, F16, BF16, Q8_0)", c->path,
                t->name, (unsigned)t->ggml_type);
        return ATH_ERR_UNSUPPORTED;
    }
    ath_status s = ath_tensor_init(&t->t, dt, (int32_t)nd, shape, NULL, ATH_TIER_RAM);
    if (s != ATH_OK) {
        ath_log(ATH_LOG_ERROR, "%s: %s: shape too large, or rows not whole blocks of %s", c->path, t->name,
                ath_dtype_name(dt));
        return ATH_ERR_FORMAT;
    }
    t->bytes = ath_tensor_nbytes(&t->t);
    return ATH_OK;
}

static int cmp_name(const void *a, const void *b, void *ctx)
{
    (void)ctx;
    return ath_safetensors_name_cmp(((const ath_gguf_tensor *)a)->name, ((const ath_gguf_tensor *)b)->name);
}

static int cmp_offset(const void *a, const void *b, void *ctx)
{
    const ath_gguf_tensor *t = ctx;
    const ath_gguf_tensor *x = &t[*(const uint32_t *)a], *y = &t[*(const uint32_t *)b];
    if (x->offset != y->offset)
        return x->offset < y->offset ? -1 : 1;
    return x->bytes < y->bytes ? -1 : (x->bytes > y->bytes);
}

/* Place the tensors in the data section and check the layout. */
static ath_status place_tensors(ath_gguf *g, ath_arena *a)
{
    const uint64_t data_bytes = g->file.size - g->data_offset;
    ath_status status = ATH_OK;
    ath_arena_pos mark = ath_arena_mark(a);
    uint32_t *order = ath_arena_alloc(a, (size_t)g->n_tensors * sizeof *order + 1, _Alignof(uint32_t));
    uint32_t tmp;
    if (!order)
        return ATH_ERR_NOMEM;
    for (uint32_t i = 0; i < g->n_tensors; i++)
        order[i] = i;
    ath_sort(order, g->n_tensors, sizeof *order, cmp_offset, g->tensors, &tmp);

    uint64_t end = 0;
    for (uint32_t k = 0; k < g->n_tensors && status == ATH_OK; k++) {
        ath_gguf_tensor *t = &g->tensors[order[k]];
        if (t->offset % g->alignment) {
            ath_log(ATH_LOG_ERROR, "%s: %s: offset %llu not a multiple of the alignment %llu", g->path, t->name,
                    (unsigned long long)t->offset, (unsigned long long)g->alignment);
            status = ATH_ERR_FORMAT;
        } else if (t->offset < end) {
            ath_log(ATH_LOG_ERROR, "%s: %s overlaps the previous tensor", g->path, t->name);
            status = ATH_ERR_FORMAT;
        } else if (t->offset - end >= g->alignment) {
            ath_log(ATH_LOG_ERROR, "%s: %s: gap of %llu bytes before it in the data section", g->path, t->name,
                    (unsigned long long)(t->offset - end));
            status = ATH_ERR_FORMAT;
        } else if (t->offset > data_bytes || t->bytes > data_bytes - t->offset) {
            ath_log(ATH_LOG_ERROR, "%s: %s: [%llu, %llu) outside the data section (%llu bytes)", g->path, t->name,
                    (unsigned long long)t->offset, (unsigned long long)(t->offset + t->bytes),
                    (unsigned long long)data_bytes);
            status = ATH_ERR_FORMAT;
        } else {
            t->t.data = (char *)g->file.data + g->data_offset + t->offset;
            end = t->offset + t->bytes;
            g->tensor_bytes += t->bytes;
        }
    }
    if (status == ATH_OK && data_bytes - end >= g->alignment) {
        ath_log(ATH_LOG_ERROR, "%s: %llu bytes after the last tensor", g->path, (unsigned long long)(data_bytes - end));
        status = ATH_ERR_FORMAT;
    }
    ath_arena_reset(a, mark);
    return status;
}

/* ---- the file ------------------------------------------------------------------ */

static ath_status parse(ath_gguf *g, ath_arena *a)
{
    cursor c = {.p = g->file.data, .size = g->file.size, .path = g->path};
    const uint32_t magic = rd_u32(&c);
    g->version = rd_u32(&c);
    const uint64_t n_tensors = rd_u64(&c);
    g->n_kv = rd_u64(&c);
    if (c.bad || magic != GGUF_MAGIC) {
        ath_log(ATH_LOG_ERROR, "%s: not a GGUF file", g->path);
        return ATH_ERR_FORMAT;
    }
    if (g->version < 2 || g->version > 3) {
        ath_log(ATH_LOG_ERROR, "%s: GGUF version %u; athanor reads versions 2 and 3", g->path, g->version);
        return ATH_ERR_UNSUPPORTED;
    }
    /* Every key/value takes at least 12 bytes and every tensor info 24: a
     * count larger than the file allows is malformed (and would overflow
     * the allocations below). */
    if (g->n_kv > g->file.size / 12 || n_tensors > g->file.size / 24 || n_tensors > UINT32_MAX) {
        ath_log(ATH_LOG_ERROR, "%s: counts larger than the file", g->path);
        return ATH_ERR_FORMAT;
    }

    g->kv = ath_arena_zalloc(a, (size_t)g->n_kv * sizeof *g->kv + 1, _Alignof(ath_gguf_kv));
    g->tensors = ath_arena_zalloc(a, (size_t)n_tensors * sizeof *g->tensors + 1, _Alignof(ath_gguf_tensor));
    if (!g->kv || !g->tensors)
        return ATH_ERR_NOMEM;
    for (uint64_t i = 0; i < g->n_kv; i++) {
        ath_status s = read_kv(&c, a, &g->kv[i]);
        if (c.bad) {
            ath_log(ATH_LOG_ERROR, "%s: metadata runs past the end of the file", g->path);
            return ATH_ERR_FORMAT;
        }
        ATH_CHECK(s);
        for (uint64_t j = 0; j < i; j++) {
            if (strcmp(g->kv[j].key, g->kv[i].key) == 0) {
                ath_log(ATH_LOG_ERROR, "%s: metadata key %s appears more than once", g->path, g->kv[i].key);
                return ATH_ERR_FORMAT;
            }
        }
    }

    g->alignment = 32;
    const ath_gguf_kv *al = ath_gguf_get(g, "general.alignment");
    if (al) {
        if (al->type != ATH_GGUF_U32 || al->v.u == 0 || (al->v.u & (al->v.u - 1))) {
            ath_log(ATH_LOG_ERROR, "%s: general.alignment must be a power of two (u32)", g->path);
            return ATH_ERR_FORMAT;
        }
        g->alignment = al->v.u;
    }

    g->n_tensors = (uint32_t)n_tensors;
    for (uint32_t i = 0; i < g->n_tensors; i++) {
        ath_status s = read_tensor(&c, a, &g->tensors[i]);
        if (c.bad) {
            ath_log(ATH_LOG_ERROR, "%s: tensor table runs past the end of the file", g->path);
            return ATH_ERR_FORMAT;
        }
        ATH_CHECK(s);
    }
    const uint64_t pad = (g->alignment - c.at % g->alignment) % g->alignment;
    if (pad > g->file.size - c.at) {
        ath_log(ATH_LOG_ERROR, "%s: no data section", g->path);
        return ATH_ERR_FORMAT;
    }
    g->data_offset = c.at + pad;

    ath_gguf_tensor tmp;
    ath_sort(g->tensors, g->n_tensors, sizeof *g->tensors, cmp_name, NULL, &tmp);
    for (uint32_t i = 1; i < g->n_tensors; i++) {
        if (strcmp(g->tensors[i - 1].name, g->tensors[i].name) == 0) {
            ath_log(ATH_LOG_ERROR, "%s: tensor %s appears more than once", g->path, g->tensors[i].name);
            return ATH_ERR_FORMAT;
        }
    }
    return place_tensors(g, a);
}

ath_status ath_gguf_open(ath_gguf *g, ath_arena *a, const char *path)
{
    memset(g, 0, sizeof *g);
    g->file.fd = -1;
    g->path = ath_arena_strndup(a, path, strlen(path));
    if (!g->path)
        return ATH_ERR_NOMEM;
    ath_status s = ath_file_open(&g->file, path);
    if (s == ATH_OK)
        s = parse(g, a);
    if (s != ATH_OK)
        ath_gguf_close(g);
    return s;
}

const ath_gguf_kv *ath_gguf_get(const ath_gguf *g, const char *key)
{
    for (uint64_t i = 0; i < g->n_kv; i++)
        if (g->kv[i].key && strcmp(g->kv[i].key, key) == 0)
            return &g->kv[i];
    return NULL;
}

const ath_gguf_tensor *ath_gguf_find(const ath_gguf *g, const char *name)
{
    uint32_t lo = 0, hi = g->n_tensors;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        int c = ath_safetensors_name_cmp(g->tensors[mid].name, name);
        if (c == 0)
            return &g->tensors[mid];
        if (c < 0)
            lo = mid + 1;
        else
            hi = mid;
    }
    return NULL;
}

void ath_gguf_close(ath_gguf *g)
{
    ath_file_close(&g->file);
}
