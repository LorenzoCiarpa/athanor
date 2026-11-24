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

/* config.c */
bool       ath_qwen35_match(const ath_json *config);
/* ATH_ERR_NOT_FOUND for a missing key, ATH_ERR_TYPE for a key of the wrong
 * type, ATH_ERR_FORMAT for inconsistent values, ATH_ERR_UNSUPPORTED for
 * values the reference implementation does not handle. mixer is allocated
 * in a. */
ath_status ath_qwen35_config_parse(ath_qwen35_config *c, ath_arena *a, const ath_json *config);

/* weights.c: fill m->params and m->ignored from m->ckpt. */
ath_status ath_qwen35_bind(ath_model *m, ath_arena *a, const ath_qwen35_config *c);

extern const ath_family ath_family_qwen35;

#endif /* ATH_QWEN35_H */
