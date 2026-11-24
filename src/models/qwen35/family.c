/* family.c: the qwen35 entry of the family registry. load() parses the
 * config, fills the model's dimensions and layer schedule, and binds the
 * tensors (weights.c). The Makefile lists this family in FAMILIES. */
#include "qwen35.h"

static ath_status load(ath_model *m, ath_arena *a)
{
    ath_qwen35_config *c = ath_arena_alloc(a, sizeof *c, _Alignof(ath_qwen35_config));
    if (!c)
        return ATH_ERR_NOMEM;
    ATH_CHECK(ath_qwen35_config_parse(c, a, m->config));

    m->family_data = c;
    m->hidden_size = c->hidden;
    m->vocab_size = c->vocab;
    m->tie_embeddings = c->tie_embeddings;
    m->n_experts = c->n_experts;
    m->n_experts_active = c->n_experts_active;

    m->n_layers = c->n_layers;
    m->layers = ath_arena_alloc(a, (size_t)c->n_layers * sizeof *m->layers, _Alignof(ath_layer));
    if (!m->layers)
        return ATH_ERR_NOMEM;
    for (int32_t i = 0; i < c->n_layers; i++) {
        m->layers[i].mixer = c->mixer[i] == ATH_QWEN35_ATTENTION ? "attention" : "gated_deltanet";
        m->layers[i].ffn = c->moe ? "moe" : "mlp";
        m->layers[i].n_experts = c->n_experts;
    }
    return ath_qwen35_bind(m, a, c);
}

const ath_family ath_family_qwen35 = {
    .name = "qwen35",
    .match = ath_qwen35_match,
    .load = load,
};
