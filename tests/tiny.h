/* tiny.h: tiny synthetic Qwen3.5 checkpoints for the test suites.
 *
 * A config.json and one model.safetensors with every tensor a tiny config
 * implies (hidden 8, vocabulary 16, four layers: three Gated DeltaNet and one
 * full attention by default), written into a fresh directory under $TMPDIR.
 * Data is zero-filled (binding looks at names, dtypes and shapes only) or
 * pseudo-random (the forward pass). Mutations leave out, misshape or retype
 * tensors for the failure tests. Header-only, like test.h and util.h.
 *
 * The dimensions are small but not degenerate: grouped-query attention (two
 * query heads per KV head), more value heads than key heads in the Gated
 * DeltaNet (two per key head), partial RoPE.
 */
#ifndef ATH_TEST_TINY_H
#define ATH_TEST_TINY_H

#include "util.h"

#include "model/model.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- tiny Qwen3.5 configs ------------------------------------------------------ */

enum {
    H = 8, V = 16, I = 12,                /* hidden, vocabulary, MLP width        */
    NH = 2, NKV = 1, HD = 4,              /* attention heads, KV heads, head dim  */
    LK = 2, LV = 4, DK = 4, DV = 4, K = 4, /* Gated DeltaNet heads, dims, kernel  */
    E = 4, TOPK = 2, IE = 6, IS = 6       /* experts, top-k, expert widths        */
};

typedef struct {
    bool        moe;
    bool        vl;          /* vision-language layout: text_config, model.language_model. */
    bool        tie;
    int         layers;
    const char *types;       /* one char per layer, 'l' linear or 'f' full; NULL omits layer_types */
    const char *first;       /* raw JSON members put first in the text config (first key wins)    */
    const char *drop;        /* text-config key left out                                         */
} tiny;

static const tiny DENSE = {.vl = true, .tie = true, .layers = 4, .types = "lllf"};
static const tiny MOE = {.moe = true, .vl = true, .tie = false, .layers = 4, .types = "lllf"};

typedef struct {
    char  *buf;
    size_t len, cap;
} sbuf;

static inline void put(sbuf *s, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static inline void put(sbuf *s, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(s->buf + s->len, s->cap - s->len, fmt, ap);
    va_end(ap);
    if (n > 0)
        s->len = s->len + (size_t)n < s->cap ? s->len + (size_t)n : s->cap - 1;
}

/* Text-config member "key": value, unless the test drops that key. */
static inline void member(sbuf *s, const tiny *c, const char *key, const char *value)
{
    if (c->drop && strcmp(c->drop, key) == 0)
        return;
    put(s, "\"%s\": %s, ", key, value);
}

static inline void config_json(sbuf *s, const tiny *c)
{
    char v[256];
    const char *type = c->moe ? (c->vl ? "qwen3_5_moe" : "qwen3_5_moe_text") : (c->vl ? "qwen3_5" : "qwen3_5_text");
    put(s, "{\"architectures\": [\"Tiny\"], \"model_type\": \"%s\", \"tie_word_embeddings\": %s, ", type,
        c->tie ? "true" : "false");
    if (c->vl)
        put(s, "\"text_config\": {\"model_type\": \"%s_text\", ", type);
    if (c->first)
        put(s, "%s, ", c->first);
    snprintf(v, sizeof v, "%d", c->layers);
    member(s, c, "num_hidden_layers", v);
    member(s, c, "hidden_size", "8");
    member(s, c, "vocab_size", "16");
    member(s, c, "rms_norm_eps", "1e-06");
    member(s, c, "hidden_act", "\"silu\"");
    member(s, c, "num_attention_heads", "2");
    member(s, c, "num_key_value_heads", "1");
    member(s, c, "head_dim", "4");
    member(s, c, "attention_bias", "false");
    member(s, c, "attn_output_gate", "true");
    member(s, c, "rope_parameters",
           "{\"rope_type\": \"default\", \"rope_theta\": 10000000, \"partial_rotary_factor\": 0.5, "
           "\"mrope_section\": [1, 0, 0], \"mrope_interleaved\": true}");
    member(s, c, "linear_num_key_heads", "2");
    member(s, c, "linear_num_value_heads", "4");
    member(s, c, "linear_key_head_dim", "4");
    member(s, c, "linear_value_head_dim", "4");
    member(s, c, "linear_conv_kernel_dim", "4");
    member(s, c, "mlp_only_layers", "[]");
    if (c->moe) {
        member(s, c, "num_experts", "4");
        member(s, c, "num_experts_per_tok", "2");
        member(s, c, "moe_intermediate_size", "6");
        member(s, c, "shared_expert_intermediate_size", "6");
    } else {
        member(s, c, "intermediate_size", "12");
    }
    if (c->types) {
        size_t n = 0;
        n += (size_t)snprintf(v + n, sizeof v - n, "[");
        for (int i = 0; c->types[i]; i++)
            n += (size_t)snprintf(v + n, sizeof v - n, "%s\"%s\"", i ? ", " : "",
                                  c->types[i] == 'f' ? "full_attention" : "linear_attention");
        snprintf(v + n, sizeof v - n, "]");
        member(s, c, "layer_types", v);
    }
    put(s, "\"max_position_embeddings\": 64}");
    if (c->vl)
        put(s, "}");
}

/* ---- tiny safetensors checkpoints ---------------------------------------------- */

typedef struct {
    sbuf        hdr;
    uint64_t    data;        /* bytes of the data section so far      */
    int         n;
    /* mutations */
    const char *skip;        /* tensors whose name contains it are left out          */
    const char *misshape;    /* tensors whose name contains it get one extra row     */
    const char *as_i32;      /* tensors whose name contains it are written as I32    */
    /* contents */
    uint64_t    seed;        /* 0: zero-filled data; otherwise pseudo-random values
                              * in [-0.5, 0.5), the same for the same seed          */
    bool        real_dtypes; /* A_log and linear_attn.norm.weight in F32, as in the
                              * published checkpoints                              */
    struct { uint64_t offset, n; bool f32; } region[128];   /* float tensors written */
    int         n_region;
} ckpt;

static inline void tensor(ckpt *k, const char *name, int ndim, int64_t a, int64_t b, int64_t c)
{
    if (k->skip && strstr(name, k->skip))
        return;
    bool i32 = k->as_i32 && strstr(name, k->as_i32);
    bool f32 = !i32 && k->real_dtypes && (strstr(name, "A_log") || strstr(name, "linear_attn.norm.weight"));
    int64_t d[3] = {a, b, c};
    if (k->misshape && strstr(name, k->misshape))
        d[0]++;
    uint64_t n = 1;
    for (int i = 0; i < ndim; i++)
        n *= (uint64_t)d[i];
    uint64_t bytes = n * (i32 || f32 ? 4 : 2);
    if (!i32 && k->n_region < (int)ATH_ARRAY_LEN(k->region)) {
        k->region[k->n_region].offset = k->data;
        k->region[k->n_region].n = n;
        k->region[k->n_region].f32 = f32;
        k->n_region++;
    }
    put(&k->hdr, "%s\"%s\": {\"dtype\": \"%s\", \"shape\": [", k->n ? ", " : "", name,
        i32 ? "I32" : f32 ? "F32" : "BF16");
    for (int i = 0; i < ndim; i++)
        put(&k->hdr, i ? ", %lld" : "%lld", (long long)d[i]);
    put(&k->hdr, "], \"data_offsets\": [%llu, %llu]}", (unsigned long long)k->data,
        (unsigned long long)(k->data + bytes));
    k->data += bytes;
    k->n++;
}

static inline void layer_tensor(ckpt *k, const char *prefix, int l, const char *suffix, int ndim, int64_t a,
                         int64_t b, int64_t c)
{
    char name[160];
    snprintf(name, sizeof name, "%slayers.%d.%s", prefix, l, suffix);
    tensor(k, name, ndim, a, b, c);
}

/* Every tensor a tiny config implies, plus the vision and MTP tensors of a
 * vision-language checkpoint. */
static inline void model_tensors(ckpt *k, const tiny *c)
{
    const char *p = c->vl ? "model.language_model." : "model.";
    char name[160];
    snprintf(name, sizeof name, "%sembed_tokens.weight", p);
    tensor(k, name, 2, V, H, 0);
    snprintf(name, sizeof name, "%snorm.weight", p);
    tensor(k, name, 1, H, 0, 0);
    if (!c->tie)
        tensor(k, "lm_head.weight", 2, V, H, 0);
    for (int l = 0; l < c->layers; l++) {
        layer_tensor(k, p, l, "input_layernorm.weight", 1, H, 0, 0);
        layer_tensor(k, p, l, "post_attention_layernorm.weight", 1, H, 0, 0);
        if (c->types[l] == 'f') {
            layer_tensor(k, p, l, "self_attn.q_proj.weight", 2, 2 * NH * HD, H, 0);
            layer_tensor(k, p, l, "self_attn.k_proj.weight", 2, NKV * HD, H, 0);
            layer_tensor(k, p, l, "self_attn.v_proj.weight", 2, NKV * HD, H, 0);
            layer_tensor(k, p, l, "self_attn.q_norm.weight", 1, HD, 0, 0);
            layer_tensor(k, p, l, "self_attn.k_norm.weight", 1, HD, 0, 0);
            layer_tensor(k, p, l, "self_attn.o_proj.weight", 2, H, NH * HD, 0);
        } else {
            const int conv = 2 * LK * DK + LV * DV;
            layer_tensor(k, p, l, "linear_attn.in_proj_qkv.weight", 2, conv, H, 0);
            layer_tensor(k, p, l, "linear_attn.in_proj_z.weight", 2, LV * DV, H, 0);
            layer_tensor(k, p, l, "linear_attn.in_proj_a.weight", 2, LV, H, 0);
            layer_tensor(k, p, l, "linear_attn.in_proj_b.weight", 2, LV, H, 0);
            layer_tensor(k, p, l, "linear_attn.conv1d.weight", 3, conv, 1, K);
            layer_tensor(k, p, l, "linear_attn.A_log", 1, LV, 0, 0);
            layer_tensor(k, p, l, "linear_attn.dt_bias", 1, LV, 0, 0);
            layer_tensor(k, p, l, "linear_attn.norm.weight", 1, DV, 0, 0);
            layer_tensor(k, p, l, "linear_attn.out_proj.weight", 2, H, LV * DV, 0);
        }
        if (c->moe) {
            layer_tensor(k, p, l, "mlp.gate.weight", 2, E, H, 0);
            layer_tensor(k, p, l, "mlp.experts.gate_up_proj", 3, E, 2 * IE, H);
            layer_tensor(k, p, l, "mlp.experts.down_proj", 3, E, H, IE);
            layer_tensor(k, p, l, "mlp.shared_expert.gate_proj.weight", 2, IS, H, 0);
            layer_tensor(k, p, l, "mlp.shared_expert.up_proj.weight", 2, IS, H, 0);
            layer_tensor(k, p, l, "mlp.shared_expert.down_proj.weight", 2, H, IS, 0);
            layer_tensor(k, p, l, "mlp.shared_expert_gate.weight", 2, 1, H, 0);
        } else {
            layer_tensor(k, p, l, "mlp.gate_proj.weight", 2, I, H, 0);
            layer_tensor(k, p, l, "mlp.up_proj.weight", 2, I, H, 0);
            layer_tensor(k, p, l, "mlp.down_proj.weight", 2, H, I, 0);
        }
    }
    if (c->vl) {
        tensor(k, "model.visual.blocks.0.attn.qkv.weight", 2, 6, 4, 0);
        tensor(k, "model.visual.merger.linear_fc2.weight", 2, H, 4, 0);
        tensor(k, "mtp.fc.weight", 2, H, 2 * H, 0);
        tensor(k, "mtp.norm.weight", 1, H, 0, 0);
    }
}

/* Pseudo-random contents (k->seed != 0): uniform in [-0.5, 0.5), from a
 * 64-bit xorshift generator, so a test sees the same weights on every run. */
static inline void fill_random(const ckpt *k, unsigned char *data)
{
    uint64_t x = k->seed * 0x9E3779B97F4A7C15ull + 1;
    for (int r = 0; r < k->n_region; r++)
        for (uint64_t i = 0; i < k->region[r].n; i++) {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            float v = (float)(x >> 40) / (float)(1 << 24) - 0.5f;
            if (k->region[r].f32) {
                memcpy(data + k->region[r].offset + 4 * i, &v, 4);
            } else {
                uint32_t u;
                memcpy(&u, &v, 4);
                uint16_t h = (uint16_t)(u >> 16);    /* truncation is fine for test data */
                memcpy(data + k->region[r].offset + 2 * i, &h, 2);
            }
        }
}

/* Write config.json and model.safetensors (header padded to 8 bytes, data
 * zero-filled or pseudo-random) into d. extra, if not NULL, is one more
 * tensor name. */
static inline bool write_model(tmpdir *d, const tiny *c, ckpt *k, const char *extra)
{
    char  hdr[1 << 15], cfg[8192];
    k->hdr = (sbuf){hdr, 0, sizeof hdr};
    k->data = 0;
    k->n = 0;
    k->n_region = 0;
    put(&k->hdr, "{");
    model_tensors(k, c);
    if (extra)
        tensor(k, extra, 1, H, 0, 0);
    put(&k->hdr, "}");
    if (k->hdr.len + 1 >= k->hdr.cap)
        return false;

    size_t padded = (k->hdr.len + 7) & ~(size_t)7, total = 8 + padded + (size_t)k->data;
    unsigned char *file = calloc(1, total);
    if (!file)
        return false;
    for (int i = 0; i < 8; i++)
        file[i] = (unsigned char)((uint64_t)padded >> (8 * i));
    memcpy(file + 8, hdr, k->hdr.len);
    memset(file + 8 + k->hdr.len, ' ', padded - k->hdr.len);
    if (k->seed)
        fill_random(k, file + 8 + padded);
    bool ok = tmpdir_write(d, "model.safetensors", file, total);
    free(file);

    sbuf s = {cfg, 0, sizeof cfg};
    config_json(&s, c);
    return ok && tmpdir_write(d, "config.json", cfg, s.len);
}

/* Open a tiny model in a fresh directory; returns the status of ath_model_open. */
static inline ath_status open_tiny(ath_arena *a, ath_model *m, tmpdir *d, const tiny *c, ckpt k, const char *extra)
{
    if (!tmpdir_make(d) || !write_model(d, c, &k, extra))
        return ATH_ERR_IO;
    return ath_model_open(m, a, d->path);
}

#endif /* ATH_TEST_TINY_H */
