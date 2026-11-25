/* test_model.c: unit tests for the model contract (src/model) and the qwen35
 * family (src/models/qwen35).
 *
 * Synthetic checkpoints (tests/tiny.h) use tiny dimensions (hidden 8,
 * vocabulary 16) and zero-filled data: binding looks at names, dtypes and
 * shapes only. They are written into a fresh directory under $TMPDIR and
 * removed afterwards. The
 * real-checkpoint test reads $ATH_MODEL_DIR and is skipped if it is absent.
 */
#include "test.h"
#include "tiny.h"
#include "util.h"

#include "model/model.h"
#include "models/qwen35/qwen35.h"

#include <stdarg.h>
#include <stdint.h>

/* ---- helpers ------------------------------------------------------------------ */

static ath_status parse_config(ath_arena *a, const tiny *c, ath_qwen35_config *out)
{
    char text[8192];
    sbuf s = {text, 0, sizeof text};
    config_json(&s, c);
    const ath_json *root;
    ath_status st = ath_json_parse(a, text, s.len, &root, NULL);
    return st ? st : ath_qwen35_config_parse(out, a, root);
}

/* Params of role r that the text path uses (ATH_P_UNUSED ones are counted
 * with count_flag). */
/* Params of the main path (not unused, not the draft head) with role r. */
static uint32_t count_role(const ath_model *m, ath_role r)
{
    uint32_t n = 0;
    for (uint32_t i = 0; i < m->n_params; i++)
        n += m->params[i].role == r && !(m->params[i].flags & (ATH_P_UNUSED | ATH_P_DRAFT));
    return n;
}

/* Params with flag; the draft head's only when flag is ATH_P_DRAFT. */
static uint32_t count_flag(const ath_model *m, uint32_t flag)
{
    uint32_t n = 0;
    for (uint32_t i = 0; i < m->n_params; i++)
        n += (m->params[i].flags & flag) != 0 && (flag == ATH_P_DRAFT || !(m->params[i].flags & ATH_P_DRAFT));
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

    /* embed + final norm + 3 x Gated DeltaNet (2 + 9 + 3) + 1 x attention (2 + 6 + 3),
     * then the 4 tensors of the ignored groups, bound as unused */
    const uint32_t used = 2 + 3 * 14 + 11;
    ATH_EXPECT_EQ_INT(m.n_params, used + 4);
    ATH_EXPECT_EQ_INT(m.n_bound, m.n_params);
    ATH_EXPECT_EQ_INT(m.ckpt.n_tensors, m.n_bound);
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

    /* the ignored groups' tensors: whole, unused, role other, no layer, last */
    ATH_EXPECT_EQ_INT(count_flag(&m, ATH_P_UNUSED), 4);
    for (uint32_t i = used; i < m.n_params; i++) {
        const ath_param *p = &m.params[i];
        ATH_EXPECT_EQ_INT(p->flags, ATH_P_UNUSED);
        ATH_EXPECT_EQ_INT(p->role, ATH_ROLE_OTHER);
        ATH_EXPECT_EQ_INT(p->layer, -1);
        ATH_EXPECT(strncmp(p->name, "model.visual.", 13) == 0 || strncmp(p->name, "mtp.", 4) == 0);
        ATH_EXPECT(p->t.data == ath_ckpt_find(&m.ckpt, p->name)->t.data);
    }

    /* forward order: embedding first, final norm last (tied: no head) */
    ATH_EXPECT_EQ_INT(m.params[0].role, ATH_ROLE_EMBED);
    ATH_EXPECT_EQ_INT(m.params[0].flags, ATH_P_WRITES_RESIDUAL | ATH_P_SENSITIVE);
    ATH_EXPECT_EQ_INT(m.params[used - 1].role, ATH_ROLE_NORM);
    ATH_EXPECT_EQ_INT(m.params[used - 1].layer, -1);
    ATH_EXPECT_EQ_INT(count_role(&m, ATH_ROLE_LM_HEAD), 0);
    int32_t prev = -1;
    for (uint32_t i = 0; i + 1 < used; i++) {
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
    ATH_EXPECT(e->t.data == ath_ckpt_find(&m.ckpt, e->name)->t.data);   /* zero-copy */

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

    /* per layer: 2 norms, mixer, router, 3 views per expert (gate and up
     * halves of gate_up_proj, down: step 12), 3 shared + gate */
    uint32_t moe = 1 + 3 * E + 4;
    const uint32_t used = 3 + 3 * (2 + 9 + moe) + (2 + 6 + moe);
    ATH_EXPECT_EQ_INT(m.n_params, used + count_flag(&m, ATH_P_UNUSED));
    ATH_EXPECT_EQ_INT(m.n_bound, m.ckpt.n_tensors);
    ATH_EXPECT_EQ_INT(m.n_experts, E);
    ATH_EXPECT_EQ_INT(m.n_experts_active, TOPK);
    ATH_EXPECT_STR_EQ(m.layers[0].ffn, "moe");
    ATH_EXPECT_EQ_INT(m.layers[0].n_experts, E);
    ATH_EXPECT(!m.tie_embeddings);
    ATH_EXPECT_EQ_INT(count_role(&m, ATH_ROLE_LM_HEAD), 1);
    ATH_EXPECT_EQ_INT(m.params[used - 1].role, ATH_ROLE_LM_HEAD);
    ATH_EXPECT_EQ_INT(m.params[0].flags, ATH_P_WRITES_RESIDUAL);   /* untied: not sensitive */
    ATH_EXPECT_EQ_INT(count_role(&m, ATH_ROLE_ROUTER), 4);
    ATH_EXPECT_EQ_INT(count_role(&m, ATH_ROLE_EXPERT_IN), 4 * 2 * E);
    ATH_EXPECT_EQ_INT(count_role(&m, ATH_ROLE_EXPERT_OUT), 4 * E);
    ATH_EXPECT_EQ_INT(count_flag(&m, ATH_P_SHARED_EXPERT), 4 * 4);
    /* residual writers: embedding, mixer output, expert downs, shared down */
    ATH_EXPECT_EQ_INT(count_flag(&m, ATH_P_WRITES_RESIDUAL), 1 + 4 * (1 + E + 1));

    /* fused experts: per expert two contiguous [IE, H] views, the gate rows
     * then the up rows of its slice, in place */
    const char *gu = "model.language_model.layers.2.mlp.experts.gate_up_proj";
    const ath_ckpt_tensor *full = ath_ckpt_find(&m.ckpt, gu);
    ATH_REQUIRE(full);
    for (int32_t x = 0; x < E; x++) {
        int seen = 0;
        for (uint32_t i = 0; i < m.n_params; i++) {
            const ath_param *p = &m.params[i];
            if (strcmp(p->name, gu) != 0 || p->expert != x)
                continue;
            ATH_EXPECT_EQ_INT(p->role, ATH_ROLE_EXPERT_IN);
            ATH_EXPECT_EQ_INT(p->layer, 2);
            ATH_EXPECT_EQ_INT(p->t.ndim, 2);
            ATH_EXPECT_EQ_INT(p->t.shape[0], IE);
            ATH_EXPECT_EQ_INT(p->t.shape[1], H);
            ATH_EXPECT(ath_tensor_is_contiguous(&p->t));
            ATH_EXPECT((const char *)p->t.data ==
                       (const char *)full->t.data + ((size_t)x * 2 + (size_t)seen) * IE * H * 2);
            seen++;
        }
        ATH_EXPECT_EQ_INT(seen, 2);
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
    ATH_EXPECT_EQ_INT(count_flag(&m, ATH_P_UNUSED), 0);
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

    /* every tensor bound; the ignored groups are exactly the unused params */
    uint32_t ignored = 0;
    for (uint32_t g = 0; g < m.n_ignored; g++)
        ignored += m.ignored[g].n_tensors;
    ATH_EXPECT_EQ_INT(m.n_bound, m.ckpt.n_tensors);
    ATH_EXPECT_EQ_INT(count_flag(&m, ATH_P_UNUSED), ignored);
    ATH_EXPECT(ignored > 0);

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
        const bool draft = m.params[i].flags & ATH_P_DRAFT;
        ATH_EXPECT(draft ? m.params[i].layer >= m.n_layers && m.params[i].layer < m.n_layers + m.n_draft_layers
                         : m.params[i].layer >= -1 && m.params[i].layer < m.n_layers);
        ATH_EXPECT(m.params[i].t.data != NULL);
    }
    /* the MTP head, as the draft head (step 13): fc, three norms, and one
     * full-attention layer (2 norms, 6 attention params, 3 MLP params) */
    if (m.n_draft_layers) {
        ATH_EXPECT_EQ_INT(m.n_draft_layers, 1);
        ATH_EXPECT_EQ_INT(count_flag(&m, ATH_P_DRAFT), m.n_experts ? 4 + 8 + 5 + 3 * m.n_experts : 4 + 11);
        ATH_EXPECT_STR_EQ(m.layers[m.n_layers].mixer, "attention");
    }
    ath_model_close(&m);
    ath_arena_free(&a);
}

/* ---- runner -------------------------------------------------------------------- */

/* Qwen3.6-35B-A3B from its Q8_0 GGUF (step 12), when the file is there
 * (ATH_MOE_DIR, default models/Qwen3.6-35B-A3B): every tensor bound, the
 * per-expert views of the fused GGUF tensors, the MTP block as the draft
 * head (step 13). */
ATH_TEST(qwen36_gguf_binds)
{
    const char *dir = getenv("ATH_MOE_DIR") ? getenv("ATH_MOE_DIR") : "models/Qwen3.6-35B-A3B";
    char p[1024];
    if (ath_path_join(p, sizeof p, dir, "Qwen3.6-35B-A3B-Q8_0.gguf") != ATH_OK || !ath_file_exists(p))
        ATH_SKIP("no Qwen3.6-35B-A3B GGUF");
    ath_arena a;
    ath_arena_init(&a, 0);
    ath_model m;
    ATH_REQUIRE_OK(ath_model_open(&m, &a, dir));
    ATH_EXPECT(m.ckpt.format == ATH_CKPT_GGUF);
    ATH_EXPECT_EQ_INT(m.n_bound, m.ckpt.n_tensors);
    ATH_EXPECT_EQ_INT(m.n_experts, 256);
    ATH_EXPECT_EQ_INT(m.n_experts_active, 8);
    ATH_EXPECT_EQ_INT(count_role(&m, ATH_ROLE_EXPERT_IN), 40 * 2 * 256);
    ATH_EXPECT_EQ_INT(count_role(&m, ATH_ROLE_EXPERT_OUT), 40 * 256);
    ATH_EXPECT_EQ_INT(count_role(&m, ATH_ROLE_ROUTER), 40);
    ATH_EXPECT_EQ_INT(count_flag(&m, ATH_P_UNUSED), 0);
    ATH_EXPECT_EQ_INT(m.n_ignored, 0);
    ATH_EXPECT_EQ_INT(m.n_draft_layers, 1);
    /* blk.40: eh_proj, enorm, hnorm, shared_head_norm; attention; MoE */
    ATH_EXPECT_EQ_INT(count_flag(&m, ATH_P_DRAFT), 4 + 8 + 5 + 3 * 256);
    const ath_param *fc = find_param(&m, "blk.40.nextn.eh_proj.weight", -1);
    ATH_EXPECT(fc && fc->t.shape[0] == 2048 && fc->t.shape[1] == 4096 && (fc->flags & ATH_P_DRAFT));
    const ath_param *e = find_param(&m, "blk.7.ffn_down_exps.weight", 255);
    ATH_REQUIRE(e);
    ATH_EXPECT(e->t.dtype == ATH_DTYPE_Q8_0 && e->t.shape[0] == 2048 && e->t.shape[1] == 512);
    const ath_ckpt_tensor *full = ath_ckpt_find(&m.ckpt, "blk.7.ffn_down_exps.weight");
    ATH_REQUIRE(full);
    ATH_EXPECT((const char *)e->t.data == (const char *)full->t.data + (size_t)255 * 2048 * 512 / 32 * 34);
    const ath_param *conv = find_param(&m, "blk.0.ssm_conv1d.weight", -1);
    ATH_EXPECT(conv && conv->t.ndim == 3 && conv->t.shape[1] == 1 && conv->t.shape[2] == 4);
    ath_model_close(&m);
    ath_arena_free(&a);
}

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
    ATH_TEST_ENTRY(qwen36_gguf_binds),
};

int main(int argc, char **argv)
{
    return ath_test_main(tests, ATH_ARRAY_LEN(tests), argc, argv);
}
