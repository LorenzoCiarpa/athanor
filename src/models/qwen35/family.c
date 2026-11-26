/* family.c: the qwen35 entry of the family registry. load() parses the
 * config, fills the model's dimensions and layer schedule, binds the tensors
 * (weights.c) and declares the per-sequence state; forward() is in
 * forward.c, the text functions in text.c. The Makefile lists this family
 * in FAMILIES.
 */
#include "qwen35.h"

/* The state one sequence needs (contract, State per layer): per attention
 * layer the keys and values of every token; per Gated DeltaNet layer the last
 * K − 1 inputs of the convolution and the fp32 recurrent state. */
static ath_status declare_states(ath_model *m, ath_arena *a, ath_qwen35_model *q)
{
    const ath_qwen35_config *c = &q->c;
    uint32_t n = 0;
    for (int32_t l = 0; l < c->n_layers; l++)
        n += c->mixer[l] == ATH_QWEN35_ATTENTION ? 1 : 2;
    n += q->n_draft ? (uint32_t)q->n_draft + 1 : 0;
    m->states = ath_arena_alloc(a, (n ? n : 1) * sizeof *m->states, _Alignof(ath_state_spec));
    if (!m->states)
        return ATH_ERR_NOMEM;

    const int64_t kv = (int64_t)c->n_kv_heads * c->head_dim;
    const int64_t conv_dim = 2 * (int64_t)c->lin_k_heads * c->lin_k_dim + (int64_t)c->lin_v_heads * c->lin_v_dim;
    m->n_states = 0;
    for (int32_t l = 0; l < c->n_layers; l++) {
        ath_qwen35_layer *L = &q->layer[l];
        if (c->mixer[l] == ATH_QWEN35_ATTENTION) {
            L->kv = (int32_t)m->n_states;
            m->states[m->n_states++] = (ath_state_spec){
                .kind = ATH_STATE_KV, .layer = l, .per_token = 2 * kv, .row = c->head_dim};
        } else {
            L->conv_state = (int32_t)m->n_states;
            m->states[m->n_states++] = (ath_state_spec){
                .kind = ATH_STATE_CONV, .layer = l, .fixed = (int64_t)(c->conv_kernel - 1) * conv_dim};
            L->recurrent = (int32_t)m->n_states;
            m->states[m->n_states++] = (ath_state_spec){
                .kind = ATH_STATE_RECURRENT, .layer = l,
                .fixed = (int64_t)c->lin_v_heads * c->lin_k_dim * c->lin_v_dim};
        }
    }
    /* the draft head (step 13): its layers' keys and values, and its input
     * waiting for the next token (the main model's last output) */
    for (int32_t i = 0; i < q->n_draft; i++) {
        q->draft[i].kv = (int32_t)m->n_states;
        m->states[m->n_states++] = (ath_state_spec){
            .kind = ATH_STATE_KV, .layer = c->n_layers + i, .per_token = 2 * kv, .row = c->head_dim};
    }
    if (q->n_draft) {
        q->d_hidden = (int32_t)m->n_states;
        m->states[m->n_states++] = (ath_state_spec){.kind = ATH_STATE_HIDDEN, .layer = c->n_layers, .fixed = c->hidden};
    }
    return ATH_OK;
}

static ath_status load(ath_model *m, ath_arena *a)
{
    ath_qwen35_model *q = ath_arena_zalloc(a, sizeof *q, _Alignof(ath_qwen35_model));
    if (!q)
        return ATH_ERR_NOMEM;
    ath_qwen35_config *c = &q->c;
    ATH_CHECK(ath_qwen35_config_parse(c, a, m->config));

    m->family_data = q;
    m->hidden_size = c->hidden;
    m->vocab_size = c->vocab;
    m->tie_embeddings = c->tie_embeddings;
    m->n_experts = c->n_experts;
    m->n_experts_active = c->n_experts_active;

    m->n_layers = c->n_layers;
    const int32_t nl = c->n_layers + c->n_draft;   /* the draft head's layers after the main ones */
    m->layers = ath_arena_alloc(a, (size_t)nl * sizeof *m->layers, _Alignof(ath_layer));
    q->layer = ath_arena_zalloc(a, (size_t)c->n_layers * sizeof *q->layer, _Alignof(ath_qwen35_layer));
    q->draft = ath_arena_zalloc(a, (size_t)(c->n_draft ? c->n_draft : 1) * sizeof *q->draft, _Alignof(ath_qwen35_layer));
    if (!m->layers || !q->layer || !q->draft)
        return ATH_ERR_NOMEM;
    for (int32_t i = 0; i < nl; i++) {
        const bool draft = i >= c->n_layers;
        m->layers[i].mixer = draft || c->mixer[i] == ATH_QWEN35_ATTENTION ? "attention" : "gated_deltanet";
        m->layers[i].ffn = c->moe ? "moe" : "mlp";
        m->layers[i].n_experts = c->n_experts;
        ath_qwen35_layer *L = draft ? &q->draft[i - c->n_layers] : &q->layer[i];
        L->kv = L->conv_state = L->recurrent = -1;
    }
    q->d_hidden = -1;
    ATH_CHECK(ath_qwen35_bind(m, a, q));
    m->n_draft_layers = q->n_draft;
    return declare_states(m, a, q);
}

const ath_family ath_family_qwen35 = {
    .name = "qwen35",
    .match = ath_qwen35_match,
    .load = load,
    .forward = ath_qwen35_forward,
    .draft = ath_qwen35_draft,
    .pretokenize_regex = ath_qwen35_pretokenize_regex,
    .pretokenize = ath_qwen35_pretokenize,
    .chat = ath_qwen35_chat,
    .parse = ath_qwen35_parse,
};
