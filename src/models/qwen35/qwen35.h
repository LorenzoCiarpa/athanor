/* qwen35.h: declarations shared by the files of the Qwen hybrid family
 * (Qwen3.5 dense and MoE). Private to src/models/qwen35/: nothing outside
 * this directory includes it, except the family's own tests.
 *
 * The op contract these dimensions feed is docs/contract/qwen35-0.8b.md
 * (dense) and docs/contract/qwen35-35b-a3b.md (MoE).
 */
#ifndef ATH_QWEN35_H
#define ATH_QWEN35_H

#include "model/model.h"

typedef enum {
    ATH_QWEN35_GATED_DELTANET,   /* layer_types "linear_attention" */
    ATH_QWEN35_ATTENTION         /* layer_types "full_attention"   */
} ath_qwen35_mixer;

/* config.json, text part, as the family uses it. Every field comes from the
 * config; the only defaults are the reference's own for keys that older
 * configs omit (partial_rotary_factor, full_attention_interval). */
typedef struct {
    bool        moe;              /* model_type qwen3_5_moe or qwen3_5_moe_text           */
    bool        tie_embeddings;
    const char *prefix;           /* tensor prefix of the language model: "model.language_model."
                                   * (vision-language checkpoint) or "model." (text-only) */
    int32_t     n_layers;
    int32_t     hidden;
    int32_t     vocab;
    double      rms_eps;

    /* full attention */
    int32_t     n_heads;
    int32_t     n_kv_heads;
    int32_t     head_dim;
    int32_t     rope_dim;         /* int(head_dim * partial_rotary_factor)                */
    double      rope_theta;

    /* Gated DeltaNet */
    int32_t     lin_k_heads;
    int32_t     lin_v_heads;
    int32_t     lin_k_dim;
    int32_t     lin_v_dim;
    int32_t     conv_kernel;

    /* dense MLP (0 for MoE) */
    int32_t     intermediate;

    /* MoE (0 for dense) */
    int32_t     n_experts;
    int32_t     n_experts_active;
    int32_t     expert_inter;     /* moe_intermediate_size            */
    int32_t     shared_inter;     /* shared_expert_intermediate_size  */

    ath_qwen35_mixer *mixer;      /* [n_layers], the layer schedule   */
} ath_qwen35_config;

/* The params of one decoder layer, by what the forward pass does with them,
 * and the indices of its state in m->states. weights.c fills the params while
 * binding, family.c the state indices. Entries a layer does not have are NULL
 * (params) or -1 (state); MoE layers leave the dense MLP entries NULL. */
typedef struct {
    const ath_param *input_norm, *post_norm;
    /* full attention */
    const ath_param *q_proj, *k_proj, *v_proj, *q_norm, *k_norm, *o_proj;
    /* Gated DeltaNet */
    const ath_param *qkv_proj, *z_proj, *a_proj, *b_proj, *conv, *a_log, *dt_bias, *norm, *out_proj;
    /* dense MLP */
    const ath_param *gate_proj, *up_proj, *down_proj;
    int32_t          kv, conv_state, recurrent;
} ath_qwen35_layer;

/* family_data of a qwen35 model. */
typedef struct {
    ath_qwen35_config c;
    ath_qwen35_layer *layer;    /* [n_layers]                                        */
    const ath_param  *embed;
    const ath_param  *norm;     /* final norm                                        */
    const ath_param  *head;     /* output projection: lm_head, or embed when tied    */
} ath_qwen35_model;

/* config.c */
bool       ath_qwen35_match(const ath_json *config);
/* ATH_ERR_NOT_FOUND for a missing key, ATH_ERR_TYPE for a key of the wrong
 * type, ATH_ERR_FORMAT for inconsistent values, ATH_ERR_UNSUPPORTED for
 * values the reference implementation does not handle. mixer is allocated
 * in a. */
ath_status ath_qwen35_config_parse(ath_qwen35_config *c, ath_arena *a, const ath_json *config);

/* weights.c: fill m->params and m->ignored from m->ckpt, and the params of
 * q (q->c is parsed, q->layer allocated). The tensors of the ignored groups
 * come last, as params flagged ATH_P_UNUSED. */
ath_status ath_qwen35_bind(ath_model *m, ath_arena *a, ath_qwen35_model *q);

/* forward.c: the family's forward (model.h, ath_family.forward). */
ath_status ath_qwen35_forward(const ath_model *m, const ath_batch *b, ath_state *s, const ath_hooks *h,
                              ath_arena *scratch, const ath_exec *x);

/* text.c: pre-tokenization, chat template and output parser (model.h,
 * ath_family). */
extern const char ath_qwen35_pretokenize_regex[];
ath_status ath_qwen35_pretokenize(const char *text, size_t n, ath_spans *out);
ath_status ath_qwen35_chat(const ath_msg *msgs, int32_t n_msgs, const ath_tool_def *tools, int32_t n_tools,
                           uint32_t opts, ath_buf *out);
ath_status ath_qwen35_parse(ath_parse_state *st, const char *piece, size_t n, ath_out_event *events, int32_t cap,
                            int32_t *n_events);

extern const ath_family ath_family_qwen35;

#endif /* ATH_QWEN35_H */
