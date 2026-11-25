/* ckpt.c: safetensors or GGUF behind one table. See ckpt.h. */
#include "core/ckpt.h"

#include <dirent.h>
#include <string.h>

const char *ath_ckpt_format_name(ath_ckpt_format f)
{
    return f == ATH_CKPT_GGUF ? "gguf" : "safetensors";
}

static bool ends_with(const char *s, const char *suffix)
{
    size_t n = strlen(s), m = strlen(suffix);
    return n >= m && strcmp(s + n - m, suffix) == 0;
}

/* The one .gguf file of dir, mmproj-*.gguf aside. */
static ath_status find_gguf(ath_arena *a, const char *dir, const char **out)
{
    DIR *d = opendir(dir);
    if (!d)
        return ATH_ERR_NOT_FOUND;
    const char *found = NULL;
    ath_status s = ATH_OK;
    for (struct dirent *e; (e = readdir(d)) != NULL;) {
        if (!ends_with(e->d_name, ".gguf") || strncmp(e->d_name, "mmproj-", 7) == 0)
            continue;
        if (found) {
            ath_log(ATH_LOG_ERROR, "%s: several .gguf files (%s, %s); athanor reads one", dir, found, e->d_name);
            s = ATH_ERR_FORMAT;
            break;
        }
        found = ath_arena_strndup(a, e->d_name, strlen(e->d_name));
        if (!found) {
            s = ATH_ERR_NOMEM;
            break;
        }
    }
    closedir(d);
    ATH_CHECK(s);
    if (!found)
        return ATH_ERR_NOT_FOUND;
    char path[ATH_PATH_MAX];
    ATH_CHECK(ath_path_join(path, sizeof path, dir, found));
    *out = ath_arena_strndup(a, path, strlen(path));
    return *out ? ATH_OK : ATH_ERR_NOMEM;
}

static ath_status open_safetensors(ath_ckpt *c, ath_arena *a, const char *dir)
{
    ATH_CHECK(ath_safetensors_open(&c->st, a, dir));
    c->format = ATH_CKPT_SAFETENSORS;
    c->n_tensors = c->st.n_tensors;
    c->tensor_bytes = c->st.tensor_bytes;
    c->tensors = ath_arena_alloc(a, (size_t)c->n_tensors * sizeof *c->tensors + 1, _Alignof(ath_ckpt_tensor));
    if (!c->tensors)
        return ATH_ERR_NOMEM;
    for (uint32_t i = 0; i < c->n_tensors; i++) {
        const ath_safetensors_entry *e = &c->st.tensors[i];
        c->tensors[i] = (ath_ckpt_tensor){.name = e->name, .t = e->t, .bytes = e->end - e->begin,
                                          .shard = e->shard, .offset = c->st.shards[e->shard].data_offset + e->begin};
    }
    return ATH_OK;
}

static ath_status open_gguf(ath_ckpt *c, ath_arena *a, const char *path)
{
    ATH_CHECK(ath_gguf_open(&c->gguf, a, path));
    c->format = ATH_CKPT_GGUF;
    c->path = c->gguf.path;
    c->n_tensors = c->gguf.n_tensors;
    c->tensor_bytes = c->gguf.tensor_bytes;
    c->tensors = ath_arena_alloc(a, (size_t)c->n_tensors * sizeof *c->tensors + 1, _Alignof(ath_ckpt_tensor));
    if (!c->tensors)
        return ATH_ERR_NOMEM;
    for (uint32_t i = 0; i < c->n_tensors; i++) {
        const ath_gguf_tensor *e = &c->gguf.tensors[i];
        c->tensors[i] = (ath_ckpt_tensor){.name = e->name, .t = e->t, .bytes = e->bytes,
                                          .offset = c->gguf.data_offset + e->offset};
    }
    return ATH_OK;
}

ath_status ath_ckpt_open(ath_ckpt *c, ath_arena *a, const char *dir)
{
    memset(c, 0, sizeof *c);
    c->gguf.file.fd = -1;
    c->path = ath_arena_strndup(a, dir, strlen(dir));
    if (!c->path)
        return ATH_ERR_NOMEM;
    char p1[ATH_PATH_MAX], p2[ATH_PATH_MAX];
    ATH_CHECK(ath_path_join(p1, sizeof p1, dir, "model.safetensors.index.json"));
    ATH_CHECK(ath_path_join(p2, sizeof p2, dir, "model.safetensors"));
    ath_status s;
    if (ath_file_exists(p1) || ath_file_exists(p2)) {
        s = open_safetensors(c, a, dir);
    } else {
        const char *g = NULL;
        s = find_gguf(a, dir, &g);
        if (s == ATH_ERR_NOT_FOUND)
            ath_log(ATH_LOG_ERROR, "%s: no model.safetensors, model.safetensors.index.json or .gguf file", dir);
        if (s == ATH_OK)
            s = open_gguf(c, a, g);
    }
    if (s != ATH_OK)
        ath_ckpt_close(c);
    return s;
}

const ath_ckpt_tensor *ath_ckpt_find(const ath_ckpt *c, const char *name)
{
    uint32_t lo = 0, hi = c->n_tensors;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        int r = ath_safetensors_name_cmp(c->tensors[mid].name, name);
        if (r == 0)
            return &c->tensors[mid];
        if (r < 0)
            lo = mid + 1;
        else
            hi = mid;
    }
    return NULL;
}

void ath_ckpt_close(ath_ckpt *c)
{
    if (c->format == ATH_CKPT_GGUF)
        ath_gguf_close(&c->gguf);
    else
        ath_safetensors_close(&c->st);
}
