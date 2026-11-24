/* test_model.c: unit tests for the model contract (src/model) and the qwen35
 * family (src/models/qwen35).
 *
 * Synthetic checkpoints use tiny dimensions (hidden 8, vocabulary 16) and
 * zero-filled data: binding looks at names, dtypes and shapes only. They are
 * written into a fresh directory under $TMPDIR and removed afterwards. The
 * real-checkpoint test reads $ATH_MODEL_DIR and is skipped if it is absent.
 */
#include "test.h"
#include "util.h"

#include "model/model.h"
#include "models/qwen35/qwen35.h"

#include <stdarg.h>
#include <stdint.h>

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

static void put(sbuf *s, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static void put(sbuf *s, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(s->buf + s->len, s->cap - s->len, fmt, ap);
    va_end(ap);
    if (n > 0)
        s->len = s->len + (size_t)n < s->cap ? s->len + (size_t)n : s->cap - 1;
}

/* Text-config member "key": value, unless the test drops that key. */
static void member(sbuf *s, const tiny *c, const char *key, const char *value)
{
    if (c->drop && strcmp(c->drop, key) == 0)
        return;
    put(s, "\"%s\": %s, ", key, value);
}

static void config_json(sbuf *s, const tiny *c)
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

static ath_status parse_config(ath_arena *a, const tiny *c, ath_qwen35_config *out)
{
    char text[8192];
    sbuf s = {text, 0, sizeof text};
    config_json(&s, c);
    const ath_json *root;
    ath_status st = ath_json_parse(a, text, s.len, &root, NULL);
    return st ? st : ath_qwen35_config_parse(out, a, root);
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
} ckpt;

static void tensor(ckpt *k, const char *name, int ndim, int64_t a, int64_t b, int64_t c)
{
    if (k->skip && strstr(name, k->skip))
        return;
    bool i32 = k->as_i32 && strstr(name, k->as_i32);
    int64_t d[3] = {a, b, c};
    if (k->misshape && strstr(name, k->misshape))
        d[0]++;
    uint64_t bytes = i32 ? 4 : 2;
    for (int i = 0; i < ndim; i++)
        bytes *= (uint64_t)d[i];
    put(&k->hdr, "%s\"%s\": {\"dtype\": \"%s\", \"shape\": [", k->n ? ", " : "", name, i32 ? "I32" : "BF16");
    for (int i = 0; i < ndim; i++)
        put(&k->hdr, i ? ", %lld" : "%lld", (long long)d[i]);
    put(&k->hdr, "], \"data_offsets\": [%llu, %llu]}", (unsigned long long)k->data,
        (unsigned long long)(k->data + bytes));
    k->data += bytes;
    k->n++;
}

static void layer_tensor(ckpt *k, const char *prefix, int l, const char *suffix, int ndim, int64_t a,
                         int64_t b, int64_t c)
{
    char name[160];
    snprintf(name, sizeof name, "%slayers.%d.%s", prefix, l, suffix);
    tensor(k, name, ndim, a, b, c);
}

/* Every tensor a tiny config implies, plus the vision and MTP tensors of a
 * vision-language checkpoint. */
static void model_tensors(ckpt *k, const tiny *c)
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

/* Write config.json and model.safetensors (header padded to 8 bytes, data
 * zero-filled) into d. extra, if not NULL, is one more tensor name. */
static bool write_model(tmpdir *d, const tiny *c, ckpt *k, const char *extra)
{
    char  hdr[1 << 15], cfg[8192];
    k->hdr = (sbuf){hdr, 0, sizeof hdr};
    k->data = 0;
    k->n = 0;
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
    bool ok = tmpdir_write(d, "model.safetensors", file, total);
    free(file);

    sbuf s = {cfg, 0, sizeof cfg};
    config_json(&s, c);
    return ok && tmpdir_write(d, "config.json", cfg, s.len);
}

/* Open a tiny model in a fresh directory; returns the status of ath_model_open. */
static ath_status open_tiny(ath_arena *a, ath_model *m, tmpdir *d, const tiny *c, ckpt k, const char *extra)
{
    if (!tmpdir_make(d) || !write_model(d, c, &k, extra))
        return ATH_ERR_IO;
    return ath_model_open(m, a, d->path);
}

static uint32_t count_role(const ath_model *m, ath_role r)
{
    uint32_t n = 0;
    for (uint32_t i = 0; i < m->n_params; i++)
        n += m->params[i].role == r;
    return n;
}

static uint32_t count_flag(const ath_model *m, uint32_t flag)
{
    uint32_t n = 0;
    for (uint32_t i = 0; i < m->n_params; i++)
        n += (m->params[i].flags & flag) != 0;
    return n;
}

static const ath_param *find_param(const ath_model *m, const char *name, int32_t expert)
{
    for (uint32_t i = 0; i < m->n_params; i++)
        if (strcmp(m->params[i].name, name) == 0 && m->params[i].expert == expert)
            return &m->params[i];
    return NULL;
}

/* ---- contract ------------------------------------------------------------------ */

ATH_TEST(role_names)
{
    for (int r = 0; r < ATH_ROLE_COUNT; r++) {
        ATH_EXPECT(strcmp(ath_role_name((ath_role)r), "?") != 0);
        for (int q = 0; q < r; q++)
            ATH_EXPECT(strcmp(ath_role_name((ath_role)r), ath_role_name((ath_role)q)) != 0);
    }
    ATH_EXPECT_STR_EQ(ath_role_name(ATH_ROLE_MIX_IN), "mix_in");
    ATH_EXPECT_STR_EQ(ath_role_name(ATH_ROLE_COUNT), "?");
}

ATH_TEST(registry_lists_families)
{
    ATH_REQUIRE(ath_family_count() >= 1);
    bool found = false;
    for (size_t i = 0; i < ath_family_count(); i++)
        found |= ath_family_at(i) == &ath_family_qwen35;
    ATH_EXPECT(found);
    ATH_EXPECT(ath_family_at(ath_family_count()) == NULL);
}

/* ---- qwen35 config ------------------------------------------------------------- */

ATH_TEST(qwen35_config_dense)
{
    ath_arena a;
    ath_arena_init(&a, 0);
    ath_qwen35_config c;
    ATH_REQUIRE_OK(parse_config(&a, &DENSE, &c));
    ATH_EXPECT(!c.moe);
    ATH_EXPECT(c.tie_embeddings);
    ATH_EXPECT_STR_EQ(c.prefix, "model.language_model.");
    ATH_EXPECT_EQ_INT(c.n_layers, 4);
    ATH_EXPECT_EQ_INT(c.hidden, H);
    ATH_EXPECT_EQ_INT(c.vocab, V);
    ATH_EXPECT_EQ_INT(c.intermediate, I);
    ATH_EXPECT_EQ_INT(c.rope_dim, 2);   /* int(4 * 0.5) */
    ATH_EXPECT_NEAR(c.rope_theta, 1e7, 0);
    ATH_EXPECT_NEAR(c.rms_eps, 1e-6, 0);
    ATH_EXPECT_EQ_INT(c.lin_v_heads, LV);
    ATH_EXPECT_EQ_INT(c.mixer[0], ATH_QWEN35_GATED_DELTANET);
    ATH_EXPECT_EQ_INT(c.mixer[3], ATH_QWEN35_ATTENTION);

    tiny text = DENSE;
    text.vl = false;
    ATH_REQUIRE_OK(parse_config(&a, &text, &c));
    ATH_EXPECT_STR_EQ(c.prefix, "model.");

    ATH_REQUIRE_OK(parse_config(&a, &MOE, &c));
    ATH_EXPECT(c.moe && !c.tie_embeddings);
    ATH_EXPECT_EQ_INT(c.n_experts, E);
    ATH_EXPECT_EQ_INT(c.n_experts_active, TOPK);
    ATH_EXPECT_EQ_INT(c.expert_inter, IE);
    ATH_EXPECT_EQ_INT(c.shared_inter, IS);
    ATH_EXPECT_EQ_INT(c.intermediate, 0);
    ath_arena_free(&a);
}

ATH_TEST(qwen35_config_schedule_fallback)
{
    ath_arena a;
    ath_arena_init(&a, 0);
    ath_qwen35_config c;
    tiny t = DENSE;
    t.layers = 8;
    t.types = NULL;   /* no layer_types: full attention every 4th layer by default */
    ATH_REQUIRE_OK(parse_config(&a, &t, &c));
    for (int i = 0; i < 8; i++)
        ATH_EXPECT_EQ_INT(c.mixer[i], (i + 1) % 4 ? ATH_QWEN35_GATED_DELTANET : ATH_QWEN35_ATTENTION);
    t.first = "\"full_attention_interval\": 2";
    ATH_REQUIRE_OK(parse_config(&a, &t, &c));
    for (int i = 0; i < 8; i++)
        ATH_EXPECT_EQ_INT(c.mixer[i], i % 2 ? ATH_QWEN35_ATTENTION : ATH_QWEN35_GATED_DELTANET);
    ath_arena_free(&a);
}

ATH_TEST(qwen35_config_rejects)
{
    static const struct {
        bool        moe;
        const char *first, *drop;
        ath_status  want;
    } cases[] = {
        {false, NULL, "hidden_size", ATH_ERR_NOT_FOUND},
        {false, NULL, "rope_parameters", ATH_ERR_NOT_FOUND},
        {false, "\"hidden_size\": \"8\"", NULL, ATH_ERR_TYPE},
        {false, "\"hidden_size\": 0", NULL, ATH_ERR_FORMAT},
        {false, "\"hidden_size\": 4294967296", NULL, ATH_ERR_FORMAT},
        {false, "\"rms_norm_eps\": 0", NULL, ATH_ERR_FORMAT},
        {false, "\"hidden_act\": \"gelu\"", NULL, ATH_ERR_UNSUPPORTED},
        {false, "\"attention_bias\": true", NULL, ATH_ERR_UNSUPPORTED},
        {false, "\"attn_output_gate\": false", NULL, ATH_ERR_UNSUPPORTED},
        {false, "\"rope_parameters\": {\"rope_type\": \"yarn\", \"rope_theta\": 1e6}", NULL, ATH_ERR_UNSUPPORTED},
        {false, "\"rope_parameters\": {\"rope_type\": \"default\", \"rope_theta\": 1e6, \"partial_rotary_factor\": 0.3}",
         NULL, ATH_ERR_FORMAT},
        {false, "\"rope_parameters\": {\"rope_type\": \"default\", \"rope_theta\": 1e6, \"partial_rotary_factor\": 2}",
         NULL, ATH_ERR_FORMAT},
        {false, "\"num_attention_heads\": 3, \"num_key_value_heads\": 2", NULL, ATH_ERR_FORMAT},
        {false, "\"linear_num_value_heads\": 3", NULL, ATH_ERR_FORMAT},
        {false, "\"layer_types\": [\"linear_attention\"]", NULL, ATH_ERR_FORMAT},
        {false, "\"layer_types\": [\"linear_attention\", \"sliding_attention\", \"full_attention\", \"full_attention\"]",
         NULL, ATH_ERR_UNSUPPORTED},
        {false, "\"mlp_only_layers\": [0]", NULL, ATH_ERR_UNSUPPORTED},
        {false, NULL, "intermediate_size", ATH_ERR_NOT_FOUND},
        {true, NULL, "num_experts", ATH_ERR_NOT_FOUND},
        {true, "\"num_experts_per_tok\": 5", NULL, ATH_ERR_FORMAT},
    };
    ath_arena a;
    ath_arena_init(&a, 0);
    for (size_t i = 0; i < ATH_ARRAY_LEN(cases); i++) {
        tiny t = cases[i].moe ? MOE : DENSE;
        t.first = cases[i].first;
        t.drop = cases[i].drop;
        ath_qwen35_config c;
        ath_status s = parse_config(&a, &t, &c);
        if (s != cases[i].want)
            printf("    case %zu: %s / drop %s\n", i, cases[i].first ? cases[i].first : "-",
                   cases[i].drop ? cases[i].drop : "-");
        ATH_EXPECT_STATUS(s, cases[i].want);
    }

    /* not a Qwen3.5 config, or a vision-language one without text_config */
    static const char *const bad[] = {
        "{\"model_type\": \"llama\"}",
        "{\"model_type\": \"qwen3_5\", \"hidden_size\": 8}",
    };
    const ath_json *root;
    ath_qwen35_config c;
    ATH_REQUIRE_OK(ath_json_parse(&a, bad[0], strlen(bad[0]), &root, NULL));
    ATH_EXPECT(!ath_qwen35_match(root));
    ATH_EXPECT_STATUS(ath_qwen35_config_parse(&c, &a, root), ATH_ERR_UNSUPPORTED);
    ATH_REQUIRE_OK(ath_json_parse(&a, bad[1], strlen(bad[1]), &root, NULL));
    ATH_EXPECT(ath_qwen35_match(root));
    ATH_EXPECT_STATUS(ath_qwen35_config_parse(&c, &a, root), ATH_ERR_FORMAT);
    ath_arena_free(&a);
}

/* ---- qwen35 binding ------------------------------------------------------------ */

ATH_TEST(qwen35_bind_dense)
{
    ath_arena a;
    ath_arena_init(&a, 0);
    tmpdir d;
    ath_model m;
    ATH_REQUIRE_OK(open_tiny(&a, &m, &d, &DENSE, (ckpt){.skip = NULL}, NULL));

    /* embed + final norm + 3 x Gated DeltaNet (2 + 9 + 3) + 1 x attention (2 + 6 + 3) */
    ATH_EXPECT_EQ_INT(m.n_params, 2 + 3 * 14 + 11);
    ATH_EXPECT_EQ_INT(m.n_bound, m.n_params);
    ATH_EXPECT_EQ_INT(m.ckpt.n_tensors, m.n_bound + 4);
    ATH_EXPECT_STR_EQ(m.family->name, "qwen35");
    ATH_EXPECT_EQ_INT(m.n_layers, 4);
    ATH_EXPECT_EQ_INT(m.hidden_size, H);
    ATH_EXPECT_EQ_INT(m.vocab_size, V);
    ATH_EXPECT(m.tie_embeddings);
    ATH_EXPECT_EQ_INT(m.n_experts, 0);
    ATH_EXPECT_STR_EQ(m.layers[0].mixer, "gated_deltanet");
    ATH_EXPECT_STR_EQ(m.layers[3].mixer, "attention");
    ATH_EXPECT_STR_EQ(m.layers[3].ffn, "mlp");

    ATH_REQUIRE(m.n_ignored == 2);
    ATH_EXPECT_STR_EQ(m.ignored[0].prefix, "model.visual.");
    ATH_EXPECT_EQ_INT(m.ignored[0].n_tensors, 2);
    ATH_EXPECT_EQ_INT(m.ignored[0].bytes, (6 * 4 + H * 4) * 2);
    ATH_EXPECT_EQ_INT(m.ignored[1].n_tensors, 2);

    /* forward order: embedding first, final norm last (tied: no head) */
    ATH_EXPECT_EQ_INT(m.params[0].role, ATH_ROLE_EMBED);
    ATH_EXPECT_EQ_INT(m.params[0].flags, ATH_P_WRITES_RESIDUAL | ATH_P_SENSITIVE);
    ATH_EXPECT_EQ_INT(m.params[m.n_params - 1].role, ATH_ROLE_NORM);
    ATH_EXPECT_EQ_INT(m.params[m.n_params - 1].layer, -1);
    ATH_EXPECT_EQ_INT(count_role(&m, ATH_ROLE_LM_HEAD), 0);
    int32_t prev = -1;
    for (uint32_t i = 0; i + 1 < m.n_params; i++) {
        ATH_EXPECT(m.params[i].layer >= prev);   /* layers are contiguous and in order */
        prev = m.params[i].layer;
        ATH_EXPECT_EQ_INT(m.params[i].expert, -1);
    }

    /* residual writers: embedding, one mixer output and one MLP down per layer */
    ATH_EXPECT_EQ_INT(count_flag(&m, ATH_P_WRITES_RESIDUAL), 1 + 4 + 4);
    ATH_EXPECT_EQ_INT(count_role(&m, ATH_ROLE_MIX_OUT), 4);
    ATH_EXPECT_EQ_INT(count_role(&m, ATH_ROLE_GATE), 2 * 3);
    ATH_EXPECT_EQ_INT(count_role(&m, ATH_ROLE_CONV), 3);
    ATH_EXPECT_EQ_INT(count_role(&m, ATH_ROLE_OTHER), 2 * 3);

    const ath_param *q = find_param(&m, "model.language_model.layers.3.self_attn.q_proj.weight", -1);
    ATH_REQUIRE(q);
    ATH_EXPECT_EQ_INT(q->role, ATH_ROLE_MIX_IN);
    ATH_EXPECT_EQ_INT(q->layer, 3);
    ATH_EXPECT_EQ_INT(q->t.shape[0], 2 * NH * HD);
    const ath_param *o = find_param(&m, "model.language_model.layers.1.linear_attn.out_proj.weight", -1);
    ATH_REQUIRE(o);
    ATH_EXPECT_EQ_INT(o->flags, ATH_P_WRITES_RESIDUAL);
    const ath_param *e = find_param(&m, "model.language_model.embed_tokens.weight", -1);
    ATH_REQUIRE(e);
    ATH_EXPECT(e->t.data == ath_safetensors_find(&m.ckpt, e->name)->t.data);   /* zero-copy */

    ath_model_close(&m);
    tmpdir_remove(&d);
    ath_arena_free(&a);
}

ATH_TEST(qwen35_bind_moe)
{
    ath_arena a;
    ath_arena_init(&a, 0);
    tmpdir d;
    ath_model m;
    ATH_REQUIRE_OK(open_tiny(&a, &m, &d, &MOE, (ckpt){.skip = NULL}, NULL));

    /* per layer: 2 norms, mixer, router, 2 views per expert, 3 shared + gate */
    uint32_t moe = 1 + 2 * E + 4;
    ATH_EXPECT_EQ_INT(m.n_params, 3 + 3 * (2 + 9 + moe) + (2 + 6 + moe));
    ATH_EXPECT_EQ_INT(m.n_experts, E);
    ATH_EXPECT_EQ_INT(m.n_experts_active, TOPK);
    ATH_EXPECT_STR_EQ(m.layers[0].ffn, "moe");
    ATH_EXPECT_EQ_INT(m.layers[0].n_experts, E);
    ATH_EXPECT(!m.tie_embeddings);
    ATH_EXPECT_EQ_INT(count_role(&m, ATH_ROLE_LM_HEAD), 1);
    ATH_EXPECT_EQ_INT(m.params[m.n_params - 1].role, ATH_ROLE_LM_HEAD);
    ATH_EXPECT_EQ_INT(m.params[0].flags, ATH_P_WRITES_RESIDUAL);   /* untied: not sensitive */
    ATH_EXPECT_EQ_INT(count_role(&m, ATH_ROLE_ROUTER), 4);
    ATH_EXPECT_EQ_INT(count_role(&m, ATH_ROLE_EXPERT_IN), 4 * E);
    ATH_EXPECT_EQ_INT(count_role(&m, ATH_ROLE_EXPERT_OUT), 4 * E);
    ATH_EXPECT_EQ_INT(count_flag(&m, ATH_P_SHARED_EXPERT), 4 * 4);
    /* residual writers: embedding, mixer output, expert downs, shared down */
    ATH_EXPECT_EQ_INT(count_flag(&m, ATH_P_WRITES_RESIDUAL), 1 + 4 * (1 + E + 1));

    /* fused experts: one contiguous [rows, cols] view per expert, in place */
    const char *gu = "model.language_model.layers.2.mlp.experts.gate_up_proj";
    const ath_safetensors_entry *full = ath_safetensors_find(&m.ckpt, gu);
    ATH_REQUIRE(full);
    for (int32_t x = 0; x < E; x++) {
        const ath_param *p = find_param(&m, gu, x);
        ATH_REQUIRE(p);
        ATH_EXPECT_EQ_INT(p->role, ATH_ROLE_EXPERT_IN);
        ATH_EXPECT_EQ_INT(p->layer, 2);
        ATH_EXPECT_EQ_INT(p->t.ndim, 2);
        ATH_EXPECT_EQ_INT(p->t.shape[0], 2 * IE);
        ATH_EXPECT_EQ_INT(p->t.shape[1], H);
        ATH_EXPECT(ath_tensor_is_contiguous(&p->t));
        ATH_EXPECT((const char *)p->t.data == (const char *)full->t.data + (size_t)x * 2 * IE * H * 2);
    }
    const ath_param *down = find_param(&m, "model.language_model.layers.0.mlp.experts.down_proj", E - 1);
    ATH_REQUIRE(down);
    ATH_EXPECT_EQ_INT(down->flags, ATH_P_WRITES_RESIDUAL);
    ATH_EXPECT_EQ_INT(down->t.shape[0], H);
    ATH_EXPECT_EQ_INT(down->t.shape[1], IE);
    const ath_param *sg = find_param(&m, "model.language_model.layers.0.mlp.shared_expert_gate.weight", -1);
    ATH_REQUIRE(sg);
    ATH_EXPECT_EQ_INT(sg->role, ATH_ROLE_GATE);
    ATH_EXPECT_EQ_INT(sg->flags, ATH_P_SHARED_EXPERT | ATH_P_SENSITIVE);

    ath_model_close(&m);
    tmpdir_remove(&d);
    ath_arena_free(&a);
}

ATH_TEST(qwen35_bind_text_only)
{
    ath_arena a;
    ath_arena_init(&a, 0);
    tmpdir d;
    ath_model m;
    tiny t = DENSE;
    t.vl = false;
    ATH_REQUIRE_OK(open_tiny(&a, &m, &d, &t, (ckpt){.skip = NULL}, NULL));
    ATH_EXPECT_EQ_INT(m.n_params, 2 + 3 * 14 + 11);
    ATH_EXPECT_EQ_INT(m.n_bound, m.ckpt.n_tensors);
    ATH_EXPECT_EQ_INT(m.ignored[0].n_tensors + m.ignored[1].n_tensors, 0);
    ATH_EXPECT(find_param(&m, "model.layers.0.linear_attn.A_log", -1) != NULL);
    ath_model_close(&m);
    tmpdir_remove(&d);
    ath_arena_free(&a);
}

ATH_TEST(qwen35_bind_rejects)
{
    static const struct {
        const char *what;
        bool        moe;
        bool        tie_flip;  /* flip tie_word_embeddings in config.json only */
        ckpt        k;
        const char *extra;
        ath_status  want;
    } cases[] = {
        {"missing tensor", false, false, {.skip = "layers.2.linear_attn.dt_bias"}, NULL, ATH_ERR_NOT_FOUND},
        {"missing expert tensor", true, false, {.skip = "layers.1.mlp.experts.down_proj"}, NULL, ATH_ERR_NOT_FOUND},
        {"unexpected tensor", false, false, {.skip = NULL}, "model.language_model.layers.0.mlp.bias", ATH_ERR_FORMAT},
        {"unexpected global", false, false, {.skip = NULL}, "model.language_model.rotary_emb.inv_freq", ATH_ERR_FORMAT},
        {"misshaped", false, false, {.misshape = "layers.3.self_attn.k_proj"}, NULL, ATH_ERR_FORMAT},
        {"misshaped experts", true, false, {.misshape = "layers.0.mlp.experts.gate_up_proj"}, NULL, ATH_ERR_FORMAT},
        {"integer dtype", false, false, {.as_i32 = "layers.0.input_layernorm"}, NULL, ATH_ERR_FORMAT},
        {"tied with lm_head", true, true, {.skip = NULL}, NULL, ATH_ERR_FORMAT},
        {"untied without lm_head", false, true, {.skip = NULL}, NULL, ATH_ERR_NOT_FOUND},
    };
    for (size_t i = 0; i < ATH_ARRAY_LEN(cases); i++) {
        ath_arena a;
        ath_arena_init(&a, 0);
        tmpdir d;
        ath_model m;
        tiny t = cases[i].moe ? MOE : DENSE;
        /* the checkpoint follows t; the config gets the flipped tie value */
        ath_status s;
        if (cases[i].tie_flip) {
            ckpt k = cases[i].k;
            s = ATH_ERR_IO;
            if (tmpdir_make(&d) && write_model(&d, &t, &k, cases[i].extra)) {
                tiny cfg = t;
                cfg.tie = !t.tie;
                char text[8192];
                sbuf sb = {text, 0, sizeof text};
                config_json(&sb, &cfg);
                if (tmpdir_write(&d, "config.json", text, sb.len))
                    s = ath_model_open(&m, &a, d.path);
            }
        } else {
            s = open_tiny(&a, &m, &d, &t, cases[i].k, cases[i].extra);
        }
        if (s != cases[i].want)
            printf("    case '%s'\n", cases[i].what);
        ATH_EXPECT_STATUS(s, cases[i].want);
        if (s == ATH_OK)
            ath_model_close(&m);
        else
            ATH_EXPECT(m.n_params == 0 && m.family == NULL);   /* cleared on failure */
        tmpdir_remove(&d);
        ath_arena_free(&a);
    }
}

ATH_TEST(model_open_errors)
{
    ath_arena a;
    ath_arena_init(&a, 0);
    tmpdir d;
    ath_model m;
    ATH_REQUIRE(tmpdir_make(&d));
    ATH_EXPECT_STATUS(ath_model_open(&m, &a, d.path), ATH_ERR_NOT_FOUND);   /* no config.json */
    const char *other = "{\"model_type\": \"llama\"}";
    ATH_REQUIRE(tmpdir_write(&d, "config.json", other, strlen(other)));
    ATH_EXPECT_STATUS(ath_model_open(&m, &a, d.path), ATH_ERR_UNSUPPORTED);
    const char *broken = "{\"model_type\": ";
    ATH_REQUIRE(tmpdir_write(&d, "config.json", broken, strlen(broken)));
    ATH_EXPECT_STATUS(ath_model_open(&m, &a, d.path), ATH_ERR_PARSE);
    tmpdir_remove(&d);
    ath_arena_free(&a);
}

/* ---- the real checkpoint ------------------------------------------------------- */

ATH_TEST(qwen35_real_model)
{
    char p[1024];
    if (!model_file(p, sizeof p, "config.json"))
        ATH_SKIP("no checkpoint in $ATH_MODEL_DIR");
    ath_arena a;
    ath_arena_init(&a, 0);
    ath_model m;
    ATH_REQUIRE_OK(ath_model_open(&m, &a, model_dir()));

    uint32_t ignored = 0;
    for (uint32_t g = 0; g < m.n_ignored; g++)
        ignored += m.ignored[g].n_tensors;
    ATH_EXPECT_EQ_INT(m.n_bound + ignored, m.ckpt.n_tensors);

    /* the schedule follows layer_types (or its fallback) exactly */
    const ath_json *types = ath_json_path(m.config, "text_config.layer_types");
    if (!types)
        types = ath_json_get(m.config, "layer_types");
    if (types) {
        ATH_EXPECT_EQ_INT(ath_json_count(types), m.n_layers);
        int32_t i = 0;
        for (const ath_json *t = ath_json_first(types); t && i < m.n_layers; t = t->next, i++) {
            bool full = strcmp(t->u.str.ptr, "full_attention") == 0;
            ATH_EXPECT_STR_EQ(m.layers[i].mixer, full ? "attention" : "gated_deltanet");
        }
    }

    /* one residual-writing mixer output per layer; a dense model adds one MLP
     * down per layer; plus the embedding */
    ATH_EXPECT_EQ_INT(count_role(&m, ATH_ROLE_MIX_OUT), m.n_layers);
    ATH_EXPECT_EQ_INT(count_role(&m, ATH_ROLE_EMBED), 1);
    ATH_EXPECT_EQ_INT(count_role(&m, ATH_ROLE_LM_HEAD), m.tie_embeddings ? 0 : 1);
    if (m.n_experts == 0)
        ATH_EXPECT_EQ_INT(count_flag(&m, ATH_P_WRITES_RESIDUAL), 1 + 2 * m.n_layers);
    for (uint32_t i = 0; i < m.n_params; i++) {
        ATH_EXPECT(m.params[i].layer >= -1 && m.params[i].layer < m.n_layers);
        ATH_EXPECT(m.params[i].t.data != NULL);
    }
    ath_model_close(&m);
    ath_arena_free(&a);
}

/* ---- runner -------------------------------------------------------------------- */

static const ath_test_case tests[] = {
    ATH_TEST_ENTRY(role_names),
    ATH_TEST_ENTRY(registry_lists_families),
    ATH_TEST_ENTRY(qwen35_config_dense),
    ATH_TEST_ENTRY(qwen35_config_schedule_fallback),
    ATH_TEST_ENTRY(qwen35_config_rejects),
    ATH_TEST_ENTRY(qwen35_bind_dense),
    ATH_TEST_ENTRY(qwen35_bind_moe),
    ATH_TEST_ENTRY(qwen35_bind_text_only),
    ATH_TEST_ENTRY(qwen35_bind_rejects),
    ATH_TEST_ENTRY(model_open_errors),
    ATH_TEST_ENTRY(qwen35_real_model),
};

int main(int argc, char **argv)
{
    return ath_test_main(tests, ATH_ARRAY_LEN(tests), argc, argv);
}
