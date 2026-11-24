/* inspect.c: `athanor inspect <model_dir> [--roles] [--json] [--filter <substring>]`.
 *
 * Without --roles: a summary of config.json, the shard files, one row per
 * tensor (name, dtype, shape, bytes) and totals per dtype, then the check
 * that the tensors account for every byte of the shards' data sections. This
 * works on any safetensors checkpoint and uses generic Hugging Face keys:
 * multimodal checkpoints keep the language model's keys under "text_config",
 * so each key is looked up at the top level first and then there.
 *
 * With --roles: the checkpoint is loaded through the model contract
 * (src/model/model.h) by the family that recognizes it, and the output is
 * what pipelines see: dimensions, the layer schedule, one row per param with
 * role, layer, expert and flags, totals per role, the ignored groups and the
 * coverage check. Nothing here knows a family: it prints the contract.
 *
 * Only headers are read: tensor data stays untouched in the mapping.
 */
#include "commands.h"

#include "core/file.h"
#include "core/json.h"
#include "core/safetensors.h"
#include "model/model.h"

#include <string.h>

void ath_cmd_inspect_usage(FILE *out)
{
    fprintf(out,
            "usage: athanor inspect <model_dir> [--roles] [--json] [--filter <substring>]\n\n"
            "Describe a checkpoint without loading it: config summary, shard files,\n"
            "every tensor with dtype, shape and size, totals per dtype, and a check\n"
            "that the tensors cover the data sections of the shard files exactly.\n\n"
            "  --roles              load the checkpoint through its model family and show\n"
            "                       what pipelines see: layer schedule, every param with\n"
            "                       role, layer, expert and flags, totals per role, the\n"
            "                       tensors left unbound on purpose\n"
            "  --json               machine-readable output on stdout\n"
            "  --filter <substring> list and total only tensors (params) whose name contains it\n");
}

typedef struct {
    const char *dir;
    bool        json;
    bool        roles;
    const char *filter;
} options;

typedef struct {
    uint32_t tensors;
    uint64_t params;
    uint64_t bytes;
} dtype_total;

/* ---- config ------------------------------------------------------------------ */

static const ath_json *cfg_get(const ath_json *cfg, const char *key)
{
    const ath_json *v = ath_json_get(cfg, key);
    return v ? v : ath_json_get(ath_json_get(cfg, "text_config"), key);
}

/* config.json is optional for inspect: its absence is reported, not fatal. */
static ath_status load_config(ath_arena *a, const char *dir, const ath_json **cfg)
{
    char path[ATH_PATH_MAX];
    *cfg = NULL;
    ATH_CHECK(ath_path_join(path, sizeof path, dir, "config.json"));
    if (!ath_file_exists(path))
        return ATH_OK;
    ath_file f;
    ATH_CHECK(ath_file_open(&f, path));
    ath_json_error err;
    ath_status s = ath_json_parse(a, f.data, f.size, cfg, &err);
    ath_file_close(&f);
    if (s != ATH_OK)
        ath_log(ATH_LOG_ERROR, "%s: line %zu column %zu: %s", path, err.line, err.col, err.msg);
    return s;
}

/* ---- formatting helpers ------------------------------------------------------ */

static void fmt_shape(char *out, size_t cap, const ath_tensor *t)
{
    size_t n = 0;
    n += (size_t)snprintf(out + n, cap - n, "[");
    for (int32_t i = 0; i < t->ndim && n < cap; i++)
        n += (size_t)snprintf(out + n, cap - n, i ? ", %lld" : "%lld", (long long)t->shape[i]);
    if (n < cap)
        snprintf(out + n, cap - n, "]");
}

static void fmt_bytes(char *out, size_t cap, uint64_t b)
{
    static const char *const units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double v = (double)b;
    size_t u = 0;
    while (v >= 1024.0 && u + 1 < ATH_ARRAY_LEN(units)) {
        v /= 1024.0;
        u++;
    }
    if (u == 0)
        snprintf(out, cap, "%llu B", (unsigned long long)b);
    else
        snprintf(out, cap, "%.2f %s", v, units[u]);
}

static const char *base_name(const char *path)
{
    const char *s = strrchr(path, '/');
    return s ? s + 1 : path;
}

static void json_str(FILE *o, const char *s, size_t len)
{
    fputc('"', o);
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '"' || c == '\\')
            fprintf(o, "\\%c", c);
        else if (c == '\n')
            fputs("\\n", o);
        else if (c == '\t')
            fputs("\\t", o);
        else if (c < 0x20)
            fprintf(o, "\\u%04x", c);
        else
            fputc(c, o);
    }
    fputc('"', o);
}

static void json_cstr(FILE *o, const char *s) { json_str(o, s, strlen(s)); }

static void json_value(FILE *o, const ath_json *v)
{
    if (!v) {
        fputs("null", o);
        return;
    }
    switch (v->type) {
    case ATH_JSON_NULL:   fputs("null", o); break;
    case ATH_JSON_BOOL:   fputs(v->u.b ? "true" : "false", o); break;
    case ATH_JSON_INT:    fprintf(o, "%lld", (long long)v->u.i); break;
    case ATH_JSON_DOUBLE:
        if (v->u.d == v->u.d && v->u.d - v->u.d == 0) /* finite */
            fprintf(o, "%.17g", v->u.d);
        else
            fputs("null", o);
        break;
    case ATH_JSON_STRING: json_str(o, v->u.str.ptr, v->u.str.len); break;
    case ATH_JSON_ARRAY:
    case ATH_JSON_OBJECT: {
        bool obj = v->type == ATH_JSON_OBJECT;
        fputc(obj ? '{' : '[', o);
        for (const ath_json *c = ath_json_first(v); c; c = c->next) {
            if (c != ath_json_first(v))
                fputs(", ", o);
            if (obj) {
                json_str(o, c->key, c->key_len);
                fputs(": ", o);
            }
            json_value(o, c);
        }
        fputc(obj ? '}' : ']', o);
        break;
    }
    }
}

/* Short text form of a config value for the human-readable summary. */
static void text_value(FILE *o, const ath_json *v)
{
    if (!v) {
        fputs("?", o);
        return;
    }
    if (v->type == ATH_JSON_STRING) {
        fputs(v->u.str.ptr, o);
        return;
    }
    if (v->type == ATH_JSON_ARRAY && ath_json_count(v) == 1) {
        text_value(o, ath_json_first(v));
        return;
    }
    json_value(o, v);
}

static bool selected(const options *opt, const char *name)
{
    return !opt->filter || strstr(name, opt->filter) != NULL;
}

/* ---- output ------------------------------------------------------------------ */

static const char *const cfg_keys[] = {"num_hidden_layers", "hidden_size", "vocab_size"};

static void print_text(const options *opt, const ath_json *cfg, const ath_safetensors *st,
                       const dtype_total *tot, const dtype_total *all)
{
    FILE *o = stdout;
    char b1[64], b2[64];

    printf("model    %s\n", opt->dir);
    if (cfg) {
        const ath_json *text_type = ath_json_path(cfg, "text_config.model_type");
        printf("config   ");
        text_value(o, ath_json_get(cfg, "architectures"));
        printf(", model_type ");
        text_value(o, ath_json_get(cfg, "model_type"));
        if (text_type) {
            printf(" (text: ");
            text_value(o, text_type);
            printf(")");
        }
        printf("\n         layers ");
        text_value(o, cfg_get(cfg, cfg_keys[0]));
        printf(", hidden size ");
        text_value(o, cfg_get(cfg, cfg_keys[1]));
        printf(", vocab ");
        text_value(o, cfg_get(cfg, cfg_keys[2]));
        printf("\n");
    } else {
        printf("config   (no config.json)\n");
    }
    for (uint32_t s = 0; s < st->n_shards; s++) {
        fmt_bytes(b1, sizeof b1, st->shards[s].data_bytes);
        printf("%s %s: header %llu B, data %s\n", s ? "        " : "shards  ",
               base_name(st->shards[s].path), (unsigned long long)st->shards[s].header_bytes, b1);
    }

    int wname = 4, wshape = 5;
    char shape[256];
    for (uint32_t i = 0; i < st->n_tensors; i++) {
        const ath_safetensors_entry *e = &st->tensors[i];
        if (!selected(opt, e->name))
            continue;
        int l = (int)strlen(e->name);
        wname = l > wname ? l : wname;
        fmt_shape(shape, sizeof shape, &e->t);
        l = (int)strlen(shape);
        wshape = l > wshape ? l : wshape;
    }
    printf("\n%-*s  %-7s  %-*s  %12s\n", wname, "name", "dtype", wshape, "shape", "bytes");
    uint32_t shown = 0;
    for (uint32_t i = 0; i < st->n_tensors; i++) {
        const ath_safetensors_entry *e = &st->tensors[i];
        if (!selected(opt, e->name))
            continue;
        fmt_shape(shape, sizeof shape, &e->t);
        printf("%-*s  %-7s  %-*s  %12llu\n", wname, e->name, ath_dtype_name(e->t.dtype), wshape,
               shape, (unsigned long long)(e->end - e->begin));
        shown++;
    }

    if (opt->filter)
        printf("\nfilter '%s': %u of %u tensors\n", opt->filter, shown, st->n_tensors);
    printf("\n%-7s  %8s  %14s  %14s\n", "dtype", "tensors", "params", "bytes");
    for (int d = 0; d < ATH_DTYPE_COUNT; d++) {
        if (!tot[d].tensors)
            continue;
        fmt_bytes(b1, sizeof b1, tot[d].bytes);
        printf("%-7s  %8u  %14llu  %14llu  %s\n", ath_dtype_name((ath_dtype)d), tot[d].tensors,
               (unsigned long long)tot[d].params, (unsigned long long)tot[d].bytes, b1);
    }
    fmt_bytes(b1, sizeof b1, all->bytes);
    printf("%-7s  %8u  %14llu  %14llu  %s\n", opt->filter ? "matched" : "total", all->tensors,
           (unsigned long long)all->params, (unsigned long long)all->bytes, b1);

    fmt_bytes(b1, sizeof b1, st->tensor_bytes);
    fmt_bytes(b2, sizeof b2, st->data_bytes);
    printf("\ncheck    tensor bytes %llu (%s) %s data sections %llu (%s)\n",
           (unsigned long long)st->tensor_bytes, b1,
           st->tensor_bytes == st->data_bytes ? "==" : "!=",
           (unsigned long long)st->data_bytes, b2);
    if (st->index_total_size >= 0)
        printf("         index metadata.total_size %lld %s tensor bytes\n",
               (long long)st->index_total_size,
               (uint64_t)st->index_total_size == st->tensor_bytes ? "==" : "!=");
}

static void print_json(const options *opt, const ath_json *cfg, const ath_safetensors *st,
                       const dtype_total *tot, const dtype_total *all)
{
    FILE *o = stdout;
    fputs("{\n  \"model_dir\": ", o);
    json_cstr(o, opt->dir);
    fputs(",\n  \"config\": ", o);
    if (cfg) {
        fputs("{\"architectures\": ", o);
        json_value(o, ath_json_get(cfg, "architectures"));
        fputs(", \"model_type\": ", o);
        json_value(o, ath_json_get(cfg, "model_type"));
        fputs(", \"text_model_type\": ", o);
        json_value(o, ath_json_path(cfg, "text_config.model_type"));
        for (size_t k = 0; k < ATH_ARRAY_LEN(cfg_keys); k++) {
            fprintf(o, ", \"%s\": ", cfg_keys[k]);
            json_value(o, cfg_get(cfg, cfg_keys[k]));
        }
        fputs("}", o);
    } else {
        fputs("null", o);
    }

    fputs(",\n  \"shards\": [", o);
    for (uint32_t s = 0; s < st->n_shards; s++) {
        fputs(s ? ",\n    {\"file\": " : "\n    {\"file\": ", o);
        json_cstr(o, base_name(st->shards[s].path));
        fprintf(o, ", \"header_bytes\": %llu, \"data_bytes\": %llu}",
                (unsigned long long)st->shards[s].header_bytes,
                (unsigned long long)st->shards[s].data_bytes);
    }
    fputs("\n  ],\n  \"filter\": ", o);
    if (opt->filter)
        json_cstr(o, opt->filter);
    else
        fputs("null", o);

    fputs(",\n  \"tensors\": [", o);
    bool first = true;
    for (uint32_t i = 0; i < st->n_tensors; i++) {
        const ath_safetensors_entry *e = &st->tensors[i];
        if (!selected(opt, e->name))
            continue;
        fputs(first ? "\n    {\"name\": " : ",\n    {\"name\": ", o);
        first = false;
        json_cstr(o, e->name);
        fprintf(o, ", \"dtype\": \"%s\", \"shape\": [", ath_dtype_name(e->t.dtype));
        for (int32_t d = 0; d < e->t.ndim; d++)
            fprintf(o, d ? ", %lld" : "%lld", (long long)e->t.shape[d]);
        fprintf(o, "], \"bytes\": %llu, \"shard\": %u, \"offset\": %llu}",
                (unsigned long long)(e->end - e->begin), e->shard,
                (unsigned long long)(st->shards[e->shard].data_offset + e->begin));
    }
    fputs("\n  ],\n  \"totals\": [", o);
    first = true;
    for (int d = 0; d < ATH_DTYPE_COUNT; d++) {
        if (!tot[d].tensors)
            continue;
        fprintf(o, "%s\n    {\"dtype\": \"%s\", \"tensors\": %u, \"params\": %llu, \"bytes\": %llu}",
                first ? "" : ",", ath_dtype_name((ath_dtype)d), tot[d].tensors,
                (unsigned long long)tot[d].params, (unsigned long long)tot[d].bytes);
        first = false;
    }
    fprintf(o, "\n  ],\n  \"selected\": {\"tensors\": %u, \"params\": %llu, \"bytes\": %llu},\n",
            all->tensors, (unsigned long long)all->params, (unsigned long long)all->bytes);
    fprintf(o, "  \"tensor_bytes\": %llu,\n  \"data_bytes\": %llu,\n",
            (unsigned long long)st->tensor_bytes, (unsigned long long)st->data_bytes);
    if (st->index_total_size >= 0)
        fprintf(o, "  \"index_total_size\": %lld,\n", (long long)st->index_total_size);
    else
        fputs("  \"index_total_size\": null,\n", o);
    fprintf(o, "  \"consistent\": %s\n}\n", st->tensor_bytes == st->data_bytes ? "true" : "false");
}

/* ---- --roles: the model contract ------------------------------------------------ */

static void fmt_flags(char *out, size_t cap, uint32_t flags)
{
    static const struct { uint32_t bit; const char *name; } names[] = {
        {ATH_P_WRITES_RESIDUAL, "residual"},
        {ATH_P_SENSITIVE, "sensitive"},
        {ATH_P_SHARED_EXPERT, "shared"},
    };
    size_t n = 0;
    out[0] = '\0';
    for (size_t i = 0; i < ATH_ARRAY_LEN(names) && n < cap; i++)
        if (flags & names[i].bit)
            n += (size_t)snprintf(out + n, cap - n, "%s%s", n ? "," : "", names[i].name);
    if (!n)
        snprintf(out, cap, "-");
}

static void fmt_index(char *out, size_t cap, int32_t v)
{
    if (v < 0)
        snprintf(out, cap, "-");
    else
        snprintf(out, cap, "%d", v);
}

typedef struct {
    uint32_t params;
    uint64_t bytes;
} param_total;

/* Totals per layer (index 0 holds the global params, layer -1) and per role,
 * over the params selected by --filter. */
typedef struct {
    param_total *layer;   /* [n_layers + 1] */
    param_total  role[ATH_ROLE_COUNT];
    param_total  all;
    uint32_t     ignored_tensors;
} roles_totals;

static ath_status roles_totals_compute(ath_arena *a, const options *opt, const ath_model *m,
                                       roles_totals *t)
{
    memset(t, 0, sizeof *t);
    t->layer = ath_arena_zalloc(a, ((size_t)m->n_layers + 1) * sizeof *t->layer, _Alignof(param_total));
    if (!t->layer)
        return ATH_ERR_NOMEM;
    for (uint32_t i = 0; i < m->n_params; i++) {
        const ath_param *p = &m->params[i];
        if (!selected(opt, p->name))
            continue;
        uint64_t b = ath_tensor_nbytes(&p->t);
        param_total *bins[] = {&t->layer[p->layer + 1], &t->role[p->role], &t->all};
        for (size_t k = 0; k < ATH_ARRAY_LEN(bins); k++) {
            bins[k]->params++;
            bins[k]->bytes += b;
        }
    }
    for (uint32_t g = 0; g < m->n_ignored; g++)
        t->ignored_tensors += m->ignored[g].n_tensors;
    return ATH_OK;
}

/* "18 gated_deltanet + 6 attention": counts of a per-layer label, in order
 * of first appearance. */
static void print_label_counts(const ath_model *m, bool mixer)
{
    const char *seen[16];
    size_t      n_seen = 0;
    for (int32_t i = 0; i < m->n_layers; i++) {
        const char *l = mixer ? m->layers[i].mixer : m->layers[i].ffn;
        bool known = false;
        for (size_t k = 0; k < n_seen && !known; k++)
            known = strcmp(seen[k], l) == 0;
        if (known || n_seen == ATH_ARRAY_LEN(seen))
            continue;
        seen[n_seen++] = l;
        int32_t count = 0;
        for (int32_t j = i; j < m->n_layers; j++)
            count += strcmp(mixer ? m->layers[j].mixer : m->layers[j].ffn, l) == 0;
        printf("%s%d %s", n_seen > 1 ? " + " : "", count, l);
    }
}

static void print_roles_text(const options *opt, const ath_model *m, const roles_totals *t)
{
    char b1[64], b2[64], b3[64], shape[256], flags[64];

    printf("model    %s\n", opt->dir);
    printf("family   %s: %d layers, hidden size %d, vocab %d, %s embeddings\n", m->family->name,
           m->n_layers, m->hidden_size, m->vocab_size, m->tie_embeddings ? "tied" : "untied");
    if (m->n_experts)
        printf("experts  %d routed per MoE layer, %d active per token\n", m->n_experts, m->n_experts_active);
    printf("schedule ");
    print_label_counts(m, true);
    printf("; ");
    print_label_counts(m, false);
    printf("\n");

    printf("\n%5s  %-16s  %-4s  %7s  %7s  %14s\n", "layer", "mixer", "ffn", "experts", "params", "bytes");
    for (int32_t i = 0; i < m->n_layers; i++) {
        const ath_layer *l = &m->layers[i];
        fmt_index(b2, sizeof b2, l->n_experts ? l->n_experts : -1);
        fmt_bytes(b1, sizeof b1, t->layer[i + 1].bytes);
        printf("%5d  %-16s  %-4s  %7s  %7u  %14llu  %s\n", i, l->mixer, l->ffn, b2, t->layer[i + 1].params,
               (unsigned long long)t->layer[i + 1].bytes, b1);
    }
    fmt_bytes(b1, sizeof b1, t->layer[0].bytes);
    printf("%5s  %-16s  %-4s  %7s  %7u  %14llu  %s\n", "-", "(global)", "", "", t->layer[0].params,
           (unsigned long long)t->layer[0].bytes, b1);

    int wname = 4, wshape = 5;
    for (uint32_t i = 0; i < m->n_params; i++) {
        const ath_param *p = &m->params[i];
        if (!selected(opt, p->name))
            continue;
        int l = (int)strlen(p->name);
        wname = l > wname ? l : wname;
        fmt_shape(shape, sizeof shape, &p->t);
        l = (int)strlen(shape);
        wshape = l > wshape ? l : wshape;
    }
    printf("\n%-*s  %-10s  %5s  %6s  %-18s  %-5s  %-*s  %12s\n", wname, "name", "role", "layer", "expert",
           "flags", "dtype", wshape, "shape", "bytes");
    for (uint32_t i = 0; i < m->n_params; i++) {
        const ath_param *p = &m->params[i];
        if (!selected(opt, p->name))
            continue;
        fmt_index(b1, sizeof b1, p->layer);
        fmt_index(b2, sizeof b2, p->expert);
        fmt_flags(flags, sizeof flags, p->flags);
        fmt_shape(shape, sizeof shape, &p->t);
        printf("%-*s  %-10s  %5s  %6s  %-18s  %-5s  %-*s  %12llu\n", wname, p->name, ath_role_name(p->role),
               b1, b2, flags, ath_dtype_name(p->t.dtype), wshape, shape,
               (unsigned long long)ath_tensor_nbytes(&p->t));
    }

    if (opt->filter)
        printf("\nfilter '%s': %u of %u params\n", opt->filter, t->all.params, m->n_params);
    printf("\n%-10s  %8s  %14s\n", "role", "params", "bytes");
    for (int r = 0; r < ATH_ROLE_COUNT; r++) {
        if (!t->role[r].params)
            continue;
        fmt_bytes(b1, sizeof b1, t->role[r].bytes);
        printf("%-10s  %8u  %14llu  %s\n", ath_role_name((ath_role)r), t->role[r].params,
               (unsigned long long)t->role[r].bytes, b1);
    }
    fmt_bytes(b1, sizeof b1, t->all.bytes);
    printf("%-10s  %8u  %14llu  %s\n", opt->filter ? "matched" : "total", t->all.params,
           (unsigned long long)t->all.bytes, b1);

    printf("\n");
    bool first = true;
    for (uint32_t g = 0; g < m->n_ignored; g++) {
        const ath_ignored *ig = &m->ignored[g];
        if (!ig->n_tensors)
            continue;
        fmt_bytes(b1, sizeof b1, ig->bytes);
        printf("%s %-16s %5u tensors  %10s  %s\n", first ? "ignored " : "        ", ig->prefix, ig->n_tensors,
               b1, ig->reason);
        first = false;
    }
    if (first)
        printf("ignored  none\n");
    uint32_t unbound = m->ckpt.n_tensors - m->n_bound - t->ignored_tensors;
    fmt_bytes(b3, sizeof b3, m->ckpt.tensor_bytes);
    printf("check    %u checkpoint tensors (%s): %u bound to %u params, %u ignored, %u unbound\n",
           m->ckpt.n_tensors, b3, m->n_bound, m->n_params, t->ignored_tensors, unbound);
}

static void print_roles_json(const options *opt, const ath_model *m, const roles_totals *t)
{
    FILE *o = stdout;
    char  flags[64];
    fputs("{\n  \"model_dir\": ", o);
    json_cstr(o, opt->dir);
    fputs(",\n  \"family\": ", o);
    json_cstr(o, m->family->name);
    fprintf(o, ",\n  \"n_layers\": %d,\n  \"hidden_size\": %d,\n  \"vocab_size\": %d,\n"
               "  \"tie_embeddings\": %s,\n  \"n_experts\": %d,\n  \"n_experts_active\": %d,\n",
            m->n_layers, m->hidden_size, m->vocab_size, m->tie_embeddings ? "true" : "false",
            m->n_experts, m->n_experts_active);
    fputs("  \"filter\": ", o);
    if (opt->filter)
        json_cstr(o, opt->filter);
    else
        fputs("null", o);

    fputs(",\n  \"layers\": [", o);
    for (int32_t i = 0; i < m->n_layers; i++) {
        const ath_layer *l = &m->layers[i];
        fprintf(o, "%s\n    {\"layer\": %d, \"mixer\": ", i ? "," : "", i);
        json_cstr(o, l->mixer);
        fputs(", \"ffn\": ", o);
        json_cstr(o, l->ffn);
        fprintf(o, ", \"n_experts\": %d, \"params\": %u, \"bytes\": %llu}", l->n_experts,
                t->layer[i + 1].params, (unsigned long long)t->layer[i + 1].bytes);
    }

    fputs("\n  ],\n  \"params\": [", o);
    bool first = true;
    for (uint32_t i = 0; i < m->n_params; i++) {
        const ath_param *p = &m->params[i];
        if (!selected(opt, p->name))
            continue;
        fputs(first ? "\n    {\"name\": " : ",\n    {\"name\": ", o);
        first = false;
        json_cstr(o, p->name);
        fmt_flags(flags, sizeof flags, p->flags);
        fprintf(o, ", \"role\": \"%s\", \"layer\": %d, \"expert\": %d, \"flags\": [", ath_role_name(p->role),
                p->layer, p->expert);
        if (p->flags)
            for (char *f = flags; f;) {
                char *comma = strchr(f, ',');
                fprintf(o, "%s\"%.*s\"", f == flags ? "" : ", ", comma ? (int)(comma - f) : (int)strlen(f), f);
                f = comma ? comma + 1 : NULL;
            }
        fprintf(o, "], \"dtype\": \"%s\", \"shape\": [", ath_dtype_name(p->t.dtype));
        for (int32_t d = 0; d < p->t.ndim; d++)
            fprintf(o, d ? ", %lld" : "%lld", (long long)p->t.shape[d]);
        fprintf(o, "], \"bytes\": %llu}", (unsigned long long)ath_tensor_nbytes(&p->t));
    }

    fputs("\n  ],\n  \"roles\": [", o);
    first = true;
    for (int r = 0; r < ATH_ROLE_COUNT; r++) {
        if (!t->role[r].params)
            continue;
        fprintf(o, "%s\n    {\"role\": \"%s\", \"params\": %u, \"bytes\": %llu}", first ? "" : ",",
                ath_role_name((ath_role)r), t->role[r].params, (unsigned long long)t->role[r].bytes);
        first = false;
    }
    fputs("\n  ],\n  \"ignored\": [", o);
    for (uint32_t g = 0; g < m->n_ignored; g++) {
        const ath_ignored *ig = &m->ignored[g];
        fputs(g ? ",\n    {\"prefix\": " : "\n    {\"prefix\": ", o);
        json_cstr(o, ig->prefix);
        fputs(", \"reason\": ", o);
        json_cstr(o, ig->reason);
        fprintf(o, ", \"tensors\": %u, \"bytes\": %llu}", ig->n_tensors, (unsigned long long)ig->bytes);
    }
    fprintf(o, "\n  ],\n  \"selected\": {\"params\": %u, \"bytes\": %llu},\n", t->all.params,
            (unsigned long long)t->all.bytes);
    fprintf(o, "  \"checkpoint_tensors\": %u,\n  \"bound_tensors\": %u,\n  \"ignored_tensors\": %u,\n"
               "  \"n_params\": %u\n}\n",
            m->ckpt.n_tensors, m->n_bound, t->ignored_tensors, m->n_params);
}

/* Loading fails, with the reason logged by the library, on any tensor the
 * family cannot account for: that is the point of the check. */
static int inspect_roles(ath_arena *a, const options *opt)
{
    ath_model m;
    ath_status s = ath_model_open(&m, a, opt->dir);
    if (s != ATH_OK) {
        fprintf(stderr, "athanor inspect: cannot load %s through a model family: %s\n", opt->dir,
                ath_status_str(s));
        return ATH_EXIT_FAILURE;
    }
    roles_totals t;
    s = roles_totals_compute(a, opt, &m, &t);
    if (s == ATH_OK) {
        if (opt->json)
            print_roles_json(opt, &m, &t);
        else
            print_roles_text(opt, &m, &t);
    }
    ath_model_close(&m);
    return s == ATH_OK ? ATH_EXIT_OK : ATH_EXIT_FAILURE;
}

/* ---- entry point --------------------------------------------------------------- */

int ath_cmd_inspect(ath_arena *a, int argc, char **argv)
{
    options opt = {0};
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            ath_cmd_inspect_usage(stdout);
            return ATH_EXIT_OK;
        } else if (strcmp(argv[i], "--json") == 0) {
            opt.json = true;
        } else if (strcmp(argv[i], "--roles") == 0) {
            opt.roles = true;
        } else if (strcmp(argv[i], "--filter") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "athanor inspect: --filter needs a value\n");
                return ATH_EXIT_USAGE;
            }
            opt.filter = argv[i];
        } else if (strncmp(argv[i], "--filter=", 9) == 0) {
            opt.filter = argv[i] + 9;
        } else if (argv[i][0] == '-' && argv[i][1] != '\0') {
            fprintf(stderr, "athanor inspect: unknown option '%s'\n", argv[i]);
            ath_cmd_inspect_usage(stderr);
            return ATH_EXIT_USAGE;
        } else if (!opt.dir) {
            opt.dir = argv[i];
        } else {
            fprintf(stderr, "athanor inspect: more than one model directory\n");
            return ATH_EXIT_USAGE;
        }
    }
    if (!opt.dir) {
        ath_cmd_inspect_usage(stderr);
        return ATH_EXIT_USAGE;
    }
    if (opt.roles)
        return inspect_roles(a, &opt);

    const ath_json *cfg;
    ath_status s = load_config(a, opt.dir, &cfg);
    if (s != ATH_OK) {
        fprintf(stderr, "athanor inspect: cannot read config.json: %s\n", ath_status_str(s));
        return ATH_EXIT_FAILURE;
    }
    ath_safetensors st;
    s = ath_safetensors_open(&st, a, opt.dir);
    if (s != ATH_OK) {
        fprintf(stderr, "athanor inspect: cannot open the checkpoint in %s: %s\n", opt.dir,
                ath_status_str(s));
        return ATH_EXIT_FAILURE;
    }

    dtype_total tot[ATH_DTYPE_COUNT] = {{0}};
    dtype_total all = {0};
    for (uint32_t i = 0; i < st.n_tensors; i++) {
        const ath_safetensors_entry *e = &st.tensors[i];
        if (!selected(&opt, e->name))
            continue;
        dtype_total *t = &tot[e->t.dtype];
        t->tensors++;
        t->params += (uint64_t)ath_tensor_numel(&e->t);
        t->bytes += e->end - e->begin;
        all.tensors++;
        all.params += (uint64_t)ath_tensor_numel(&e->t);
        all.bytes += e->end - e->begin;
    }

    if (opt.json)
        print_json(&opt, cfg, &st, tot, &all);
    else
        print_text(&opt, cfg, &st, tot, &all);

    int rc = st.tensor_bytes == st.data_bytes ? ATH_EXIT_OK : ATH_EXIT_FAILURE;
    ath_safetensors_close(&st);
    return rc;
}
