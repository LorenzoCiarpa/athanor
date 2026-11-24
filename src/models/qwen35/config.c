/* config.c: config.json of a Qwen3.5 checkpoint → ath_qwen35_config.
 *
 * Vision-language checkpoints (model_type qwen3_5 / qwen3_5_moe) keep the
 * language model's keys under "text_config" and its tensors under
 * "model.language_model."; text-only ones (qwen3_5_text / qwen3_5_moe_text)
 * keep both at the top level and under "model.". The reference
 * implementation is transformers 5.24.0 (docs/contract/qwen35-0.8b.md).
 *
 * Keys the reference reads but athanor cannot honor are rejected instead of
 * ignored: a checkpoint that needs them would otherwise load and compute
 * something else.
 */
#include "qwen35.h"

#include <stdio.h>
#include <string.h>

static const char *const model_types[] = {"qwen3_5", "qwen3_5_moe", "qwen3_5_text", "qwen3_5_moe_text"};

static const char *model_type(const ath_json *config)
{
    const char *s = NULL;
    ath_json_to_str(ath_json_get(config, "model_type"), &s, NULL);
    return s;
}

bool ath_qwen35_match(const ath_json *config)
{
    const char *t = model_type(config);
    for (size_t i = 0; t && i < ATH_ARRAY_LEN(model_types); i++)
        if (strcmp(t, model_types[i]) == 0)
            return true;
    return false;
}

/* ---- typed reads with messages ------------------------------------------------ */

typedef struct {
    const ath_json *obj;     /* object holding the language model's keys */
    const char     *where;   /* "text_config." or "" for messages        */
} keys;

static const ath_json *get(const keys *k, const char *key, bool required)
{
    const ath_json *v = ath_json_get(k->obj, key);
    if (!v && required)
        ath_log(ATH_LOG_ERROR, "config.json: %s%s is missing", k->where, key);
    return v;
}

static ath_status get_int(const keys *k, const char *key, int32_t min, int32_t *out)
{
    const ath_json *v = get(k, key, true);
    if (!v)
        return ATH_ERR_NOT_FOUND;
    int64_t x;
    if (ath_json_to_int(v, &x) != ATH_OK) {
        ath_log(ATH_LOG_ERROR, "config.json: %s%s is not an integer", k->where, key);
        return ATH_ERR_TYPE;
    }
    if (x < min || x > INT32_MAX) {
        ath_log(ATH_LOG_ERROR, "config.json: %s%s = %lld is out of range", k->where, key, (long long)x);
        return ATH_ERR_FORMAT;
    }
    *out = (int32_t)x;
    return ATH_OK;
}

static ath_status get_positive_double(const keys *k, const char *key, double *out)
{
    const ath_json *v = get(k, key, true);
    if (!v)
        return ATH_ERR_NOT_FOUND;
    if (ath_json_to_double(v, out) != ATH_OK) {
        ath_log(ATH_LOG_ERROR, "config.json: %s%s is not a number", k->where, key);
        return ATH_ERR_TYPE;
    }
    if (!(*out > 0)) {
        ath_log(ATH_LOG_ERROR, "config.json: %s%s must be positive", k->where, key);
        return ATH_ERR_FORMAT;
    }
    return ATH_OK;
}

/* A string key that must have exactly the value the reference assumes. */
static ath_status expect_str(const keys *k, const char *key, const char *want)
{
    const ath_json *v = get(k, key, true);
    if (!v)
        return ATH_ERR_NOT_FOUND;
    const char *s;
    if (ath_json_to_str(v, &s, NULL) != ATH_OK) {
        ath_log(ATH_LOG_ERROR, "config.json: %s%s is not a string", k->where, key);
        return ATH_ERR_TYPE;
    }
    if (strcmp(s, want) != 0) {
        ath_log(ATH_LOG_ERROR, "config.json: %s%s = '%s' is not supported (only '%s')", k->where, key, s,
                want);
        return ATH_ERR_UNSUPPORTED;
    }
    return ATH_OK;
}

/* An optional boolean key that, when present, must have the value the
 * reference implementation hard-codes. */
static ath_status expect_bool_if_present(const keys *k, const char *key, bool want)
{
    const ath_json *v = get(k, key, false);
    if (!v)
        return ATH_OK;
    bool b;
    if (ath_json_to_bool(v, &b) != ATH_OK) {
        ath_log(ATH_LOG_ERROR, "config.json: %s%s is not a boolean", k->where, key);
        return ATH_ERR_TYPE;
    }
    if (b != want) {
        ath_log(ATH_LOG_ERROR, "config.json: %s%s = %s is not supported", k->where, key,
                b ? "true" : "false");
        return ATH_ERR_UNSUPPORTED;
    }
    return ATH_OK;
}

/* ---- parts -------------------------------------------------------------------- */

/* The layer schedule: layer_types if present, otherwise the reference's
 * fallback, full attention every full_attention_interval layers (default 4)
 * counting from 1. */
static ath_status parse_schedule(ath_qwen35_config *c, ath_arena *a, const keys *k)
{
    c->mixer = ath_arena_alloc(a, (size_t)c->n_layers * sizeof *c->mixer, _Alignof(ath_qwen35_mixer));
    if (!c->mixer)
        return ATH_ERR_NOMEM;

    const ath_json *types = get(k, "layer_types", false);
    if (!types) {
        int32_t interval = 4;
        if (get(k, "full_attention_interval", false))
            ATH_CHECK(get_int(k, "full_attention_interval", 1, &interval));
        for (int32_t i = 0; i < c->n_layers; i++)
            c->mixer[i] = (i + 1) % interval ? ATH_QWEN35_GATED_DELTANET : ATH_QWEN35_ATTENTION;
        return ATH_OK;
    }
    if (types->type != ATH_JSON_ARRAY || ath_json_count(types) != (size_t)c->n_layers) {
        ath_log(ATH_LOG_ERROR, "config.json: %slayer_types must be an array of num_hidden_layers (%d) strings",
                k->where, c->n_layers);
        return ATH_ERR_FORMAT;
    }
    int32_t i = 0;
    for (const ath_json *t = ath_json_first(types); t; t = t->next, i++) {
        const char *s = NULL;
        ath_json_to_str(t, &s, NULL);
        if (s && strcmp(s, "linear_attention") == 0) {
            c->mixer[i] = ATH_QWEN35_GATED_DELTANET;
        } else if (s && strcmp(s, "full_attention") == 0) {
            c->mixer[i] = ATH_QWEN35_ATTENTION;
        } else {
            ath_log(ATH_LOG_ERROR, "config.json: %slayer_types[%d] is not 'linear_attention' or 'full_attention'",
                    k->where, i);
            return ATH_ERR_UNSUPPORTED;
        }
    }
    return ATH_OK;
}

/* rope_parameters: plain RoPE on the first int(head_dim * partial_rotary_factor)
 * dimensions. mrope_section only matters for image and video positions; for
 * text it is the identity (contract, Full-attention layer). */
static ath_status parse_rope(ath_qwen35_config *c, const keys *k)
{
    const ath_json *rp = get(k, "rope_parameters", true);
    if (!rp)
        return ATH_ERR_NOT_FOUND;
    if (rp->type != ATH_JSON_OBJECT) {
        ath_log(ATH_LOG_ERROR, "config.json: %srope_parameters is not an object", k->where);
        return ATH_ERR_TYPE;
    }
    char where[64];
    snprintf(where, sizeof where, "%srope_parameters.", k->where);
    keys r = {rp, where};
    ATH_CHECK(expect_str(&r, "rope_type", "default"));
    ATH_CHECK(get_positive_double(&r, "rope_theta", &c->rope_theta));
    double factor = 0.25; /* the reference's default for configs without the key */
    if (get(&r, "partial_rotary_factor", false))
        ATH_CHECK(get_positive_double(&r, "partial_rotary_factor", &factor));
    if (factor > 1.0) {
        ath_log(ATH_LOG_ERROR, "config.json: %spartial_rotary_factor %g is above 1", where, factor);
        return ATH_ERR_FORMAT;
    }
    c->rope_dim = (int32_t)(c->head_dim * factor); /* int() in the reference: truncation */
    if (c->rope_dim < 2 || c->rope_dim % 2) {
        ath_log(ATH_LOG_ERROR, "config.json: rotary dimension %d (head_dim %d x %g) must be even and positive",
                c->rope_dim, c->head_dim, factor);
        return ATH_ERR_FORMAT;
    }
    return ATH_OK;
}

static ath_status parse_ffn(ath_qwen35_config *c, const keys *k)
{
    const ath_json *only = get(k, "mlp_only_layers", false);
    if (only && ath_json_count(only) != 0) {
        /* The reference ignores this key and builds a MoE block in every layer. */
        ath_log(ATH_LOG_ERROR, "config.json: %smlp_only_layers must be empty", k->where);
        return ATH_ERR_UNSUPPORTED;
    }
    if (!c->moe)
        return get_int(k, "intermediate_size", 1, &c->intermediate);
    ATH_CHECK(get_int(k, "num_experts", 1, &c->n_experts));
    ATH_CHECK(get_int(k, "num_experts_per_tok", 1, &c->n_experts_active));
    ATH_CHECK(get_int(k, "moe_intermediate_size", 1, &c->expert_inter));
    ATH_CHECK(get_int(k, "shared_expert_intermediate_size", 1, &c->shared_inter));
    if (c->n_experts_active > c->n_experts) {
        ath_log(ATH_LOG_ERROR, "config.json: num_experts_per_tok %d exceeds num_experts %d",
                c->n_experts_active, c->n_experts);
        return ATH_ERR_FORMAT;
    }
    return ATH_OK;
}

/* tie_word_embeddings: the top-level key governs the vision-language model
 * (its _tied_weights_keys); text-only configs have only one level. Default
 * false, as in the reference config classes. */
static ath_status parse_tie(ath_qwen35_config *c, const ath_json *config, const keys *k)
{
    const ath_json *v = ath_json_get(config, "tie_word_embeddings");
    if (!v)
        v = ath_json_get(k->obj, "tie_word_embeddings");
    c->tie_embeddings = false;
    if (v && ath_json_to_bool(v, &c->tie_embeddings) != ATH_OK) {
        ath_log(ATH_LOG_ERROR, "config.json: tie_word_embeddings is not a boolean");
        return ATH_ERR_TYPE;
    }
    return ATH_OK;
}

/* ---- entry point -------------------------------------------------------------- */

ath_status ath_qwen35_config_parse(ath_qwen35_config *c, ath_arena *a, const ath_json *config)
{
    memset(c, 0, sizeof *c);
    if (!ath_qwen35_match(config)) {
        ath_log(ATH_LOG_ERROR, "config.json: model_type is not a Qwen3.5 type");
        return ATH_ERR_UNSUPPORTED;
    }
    const char *type = model_type(config);
    c->moe = strcmp(type, "qwen3_5_moe") == 0 || strcmp(type, "qwen3_5_moe_text") == 0;
    bool vl = strcmp(type, "qwen3_5") == 0 || strcmp(type, "qwen3_5_moe") == 0;

    keys k = {config, ""};
    if (vl) {
        k.obj = ath_json_get(config, "text_config");
        k.where = "text_config.";
        if (!k.obj || k.obj->type != ATH_JSON_OBJECT) {
            ath_log(ATH_LOG_ERROR, "config.json: model_type %s needs a text_config object", type);
            return ATH_ERR_FORMAT;
        }
    }
    c->prefix = vl ? "model.language_model." : "model.";

    ATH_CHECK(get_int(&k, "num_hidden_layers", 1, &c->n_layers));
    ATH_CHECK(get_int(&k, "hidden_size", 1, &c->hidden));
    ATH_CHECK(get_int(&k, "vocab_size", 1, &c->vocab));
    ATH_CHECK(get_positive_double(&k, "rms_norm_eps", &c->rms_eps));
    ATH_CHECK(expect_str(&k, "hidden_act", "silu"));

    ATH_CHECK(get_int(&k, "num_attention_heads", 1, &c->n_heads));
    ATH_CHECK(get_int(&k, "num_key_value_heads", 1, &c->n_kv_heads));
    ATH_CHECK(get_int(&k, "head_dim", 1, &c->head_dim));
    ATH_CHECK(expect_bool_if_present(&k, "attention_bias", false));
    ATH_CHECK(expect_bool_if_present(&k, "attn_output_gate", true));
    ATH_CHECK(parse_rope(c, &k));

    ATH_CHECK(get_int(&k, "linear_num_key_heads", 1, &c->lin_k_heads));
    ATH_CHECK(get_int(&k, "linear_num_value_heads", 1, &c->lin_v_heads));
    ATH_CHECK(get_int(&k, "linear_key_head_dim", 1, &c->lin_k_dim));
    ATH_CHECK(get_int(&k, "linear_value_head_dim", 1, &c->lin_v_dim));
    ATH_CHECK(get_int(&k, "linear_conv_kernel_dim", 1, &c->conv_kernel));

    if (c->n_heads % c->n_kv_heads) {
        ath_log(ATH_LOG_ERROR, "config.json: num_attention_heads %d is not a multiple of num_key_value_heads %d",
                c->n_heads, c->n_kv_heads);
        return ATH_ERR_FORMAT;
    }
    if (c->lin_v_heads % c->lin_k_heads) {
        ath_log(ATH_LOG_ERROR,
                "config.json: linear_num_value_heads %d is not a multiple of linear_num_key_heads %d",
                c->lin_v_heads, c->lin_k_heads);
        return ATH_ERR_FORMAT;
    }

    ATH_CHECK(parse_ffn(c, &k));
    ATH_CHECK(parse_tie(c, config, &k));
    return parse_schedule(c, a, &k);
}
