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

/* The KV cache in int8 (step 14): a scale per 32 values of a key or value
 * row (8 per head of 256), or per row when 32 does not divide it (the tiny
 * test models). One scale per row of 256 cost 2.4 times the bf16 noise in
 * KL, almost all of it in the values; groups of 32 cost 1.2 times
 * (tests/README.md, KV cache in int8). */
#define ATH_QWEN35_KV_GROUP 32

static inline int64_t ath_qwen35_kv_group(int64_t head_dim)
{
    return head_dim % ATH_QWEN35_KV_GROUP == 0 ? ATH_QWEN35_KV_GROUP : head_dim;
}

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

    /* multi-token prediction head (step 13): mtp_num_hidden_layers, full
     * attention and the MLP or MoE of the main layers; 0 without one */
    int32_t     n_draft;
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
    /* MoE (step 12): the router, every routed expert's gate, up and down
     * projections ([n_experts] each, views of the checkpoint's fused
     * tensors), the shared expert and its gate */
    const ath_param  *router;
    const ath_param **exp_gate, **exp_up, **exp_down;
    const ath_param  *sh_gate, *sh_up, *sh_down, *sh_gate_w;
    int32_t          kv, conv_state, recurrent;
} ath_qwen35_layer;

/* family_data of a qwen35 model. */
typedef struct {
    ath_qwen35_config c;
    /* The checkpoint is a GGUF file (step 12): llama.cpp's converter stored
     * the zero-centered norm weights as 1 + w (norm_offset 0 instead of 1),
     * A = −exp(A_log) instead of A_log (a_given), and the Gated DeltaNet's
     * value heads in tiled order (contract qwen36-35b-a3b.md, GGUF). */
    bool              gguf;
    float             norm_offset;
    ath_qwen35_layer *layer;    /* [n_layers]                                        */
    const ath_param  *embed;
    const ath_param  *norm;     /* final norm                                        */
    const ath_param  *head;     /* output projection: lm_head, or embed when tied    */
    /* The draft head (multi-token prediction, step 13), when the checkpoint
     * has it (n_draft > 0): x = fc([norm_e(embed(t + 1)); norm_h(h_t)]), its
     * layers, then norm and the shared output projection. The embedding and
     * head are the main model's. */
    int32_t           n_draft;
    ath_qwen35_layer *draft;    /* [n_draft], params of layers n_layers + i          */
    const ath_param  *d_fc, *d_enorm, *d_hnorm, *d_norm;
    int32_t           d_hidden; /* the state's ATH_STATE_HIDDEN buffer               */
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
/* forward.c: the draft head (model.h, ath_family.draft). */
ath_status ath_qwen35_draft(const ath_model *m, const ath_draft *d, ath_state *s, ath_arena *scratch,
                            const ath_exec *x);

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
