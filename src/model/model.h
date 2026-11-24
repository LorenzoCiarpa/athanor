/* model.h: the contract between pipelines and model families.
 *
 * Pipelines (infer, quant, ablit, train) and tools see a model only through
 * this header. A family (src/models/<family>/) turns a checkpoint into an
 * ath_model: every checkpoint tensor either becomes one or more params, each
 * with a role, a layer, an expert index and flags, or belongs to a group that
 * the family deliberately leaves unbound and says why (an ignored group).
 * Pipelines decide what to do by role and flags, never by tensor name or by
 * family:
 *
 *   quantization   formats by role, higher precision for ATH_P_SENSITIVE
 *   abliteration   orthogonalizes every param flagged ATH_P_WRITES_RESIDUAL
 *   training       LoRA targets by role, hot/cold blocks by layer
 *   MoE placement  experts by layer and expert
 *
 * Step 2 defined roles, params, the layer schedule and loading. Step 4 adds
 * the forward pass: the per-sequence state a model needs (state specs, so
 * that pipelines allocate it without knowing the layer types), batches and
 * hooks. Pre-tokenization and chat templates (step 5), the matmul hook
 * (phase 2) and the backward pass (phase 4) join the contract when their
 * first user lands; docs/ARCHITECTURE.md shows the target.
 */
#ifndef ATH_MODEL_H
#define ATH_MODEL_H

#include "core/arena.h"
#include "core/json.h"
#include "core/safetensors.h"
#include "core/tensor.h"

/* What a param is for. Matmul roles name the direction relative to the
 * residual stream: _IN reads from it, _OUT writes back into it. */
typedef enum {
    ATH_ROLE_EMBED,       /* token embedding table [vocab, hidden]; with tie_embeddings
                           * it is also the output projection                          */
    ATH_ROLE_LM_HEAD,     /* output projection [vocab, hidden], untied models only      */
    ATH_ROLE_NORM,        /* normalization weight; its parameterization is family math  */
    ATH_ROLE_MIX_IN,      /* token mixer (attention or linear attention) input matmuls  */
    ATH_ROLE_MIX_OUT,     /* token mixer output matmul                                  */
    ATH_ROLE_MLP_IN,      /* dense MLP or shared expert input matmuls (gate, up)        */
    ATH_ROLE_MLP_OUT,     /* dense MLP or shared expert output matmul (down)            */
    ATH_ROLE_ROUTER,      /* MoE router [n_experts, hidden]                             */
    ATH_ROLE_EXPERT_IN,   /* routed expert input matmul, one param per expert           */
    ATH_ROLE_EXPERT_OUT,  /* routed expert output matmul, one param per expert          */
    ATH_ROLE_CONV,        /* short causal convolution kernel                            */
    ATH_ROLE_GATE,        /* small matmul that produces per-head or per-token gates     */
    ATH_ROLE_OTHER,       /* any other parameter (decay constants, biases of gates)     */
    ATH_ROLE_COUNT
} ath_role;

/* Lower-case name ("mix_in"); "?" for values out of range. */
const char *ath_role_name(ath_role r);

#define ATH_P_WRITES_RESIDUAL 1u  /* its output is added to the residual stream            */
#define ATH_P_SENSITIVE       2u  /* quantization keeps it at higher precision by default  */
#define ATH_P_SHARED_EXPERT   4u  /* part of the always-active shared expert of a MoE layer */

typedef struct {
    ath_tensor  t;        /* contiguous view of checkpoint memory, possibly a slice of a
                           * larger tensor (one expert of a fused expert tensor)       */
    ath_role    role;
    int32_t     layer;    /* -1 for global params (embedding, final norm, output head) */
    int32_t     expert;   /* routed expert index, -1 if not a routed expert           */
    uint32_t    flags;    /* ATH_P_*                                                  */
    const char *name;     /* checkpoint tensor the view belongs to                    */
} ath_param;

/* One entry of the layer schedule. The labels are for people (inspect,
 * logs); pipelines never branch on them: they use roles, flags and, from
 * step 4, state specs. */
typedef struct {
    const char *mixer;      /* token mixer, e.g. "attention", "gated_deltanet" */
    const char *ffn;        /* channel mixer, "mlp" or "moe"                   */
    int32_t     n_experts;  /* routed experts in this layer, 0 if dense        */
} ath_layer;

/* Checkpoint tensors a family leaves unbound on purpose. The family sets
 * prefix and reason; ath_model_open counts the matching tensors. */
typedef struct {
    const char *prefix;
    const char *reason;
    uint32_t    n_tensors;
    uint64_t    bytes;
} ath_ignored;

/* ---- per-sequence state (step 4) ---------------------------------------------
 *
 * A model lists the state one sequence needs as specs; pipelines allocate it
 * and decide where it lives by kind and size alone. The element type follows
 * from the kind and from the activation dtype the state is created with. */
typedef enum {
    ATH_STATE_KV,          /* keys and values of an attention layer, per token: grows
                            * with the sequence; activation dtype                    */
    ATH_STATE_CONV,        /* the last inputs of a short causal convolution; fixed
                            * size; activation dtype                                 */
    ATH_STATE_RECURRENT,   /* the recurrent state of a linear-attention layer; fixed
                            * size; always fp32                                      */
    ATH_STATE_KIND_COUNT
} ath_state_kind;

/* Lower-case name ("kv"); "?" for values out of range. */
const char *ath_state_kind_name(ath_state_kind k);

typedef struct {
    ath_state_kind kind;
    int32_t        layer;
    int64_t        fixed;       /* elements independent of the sequence length (CONV, RECURRENT) */
    int64_t        per_token;   /* elements per token (KV: keys and values)                      */
} ath_state_spec;

typedef struct ath_family ath_family;

typedef struct {
    const ath_family *family;
    const ath_json   *config;            /* parsed config.json                         */
    ath_safetensors   ckpt;              /* mapped checkpoint; params view its memory  */

    ath_param        *params;            /* global params and each layer's params, in
                                          * forward order                             */
    uint32_t          n_params;
    ath_layer        *layers;
    int32_t           n_layers;
    ath_ignored      *ignored;
    uint32_t          n_ignored;
    uint32_t          n_bound;           /* checkpoint tensors viewed by some param    */

    int32_t           hidden_size;
    int32_t           vocab_size;
    bool              tie_embeddings;    /* the EMBED param is also the output
                                          * projection; there is no LM_HEAD param    */
    int32_t           n_experts;         /* routed experts per MoE layer, 0 if dense   */
    int32_t           n_experts_active;  /* experts selected per token, 0 if dense     */

    ath_state_spec   *states;            /* per-sequence state, in layer order         */
    uint32_t          n_states;

    void             *family_data;       /* family-private (parsed config); opaque     */
} ath_model;

/* The state of one sequence, in host memory (ath_state_init). The engine of
 * step 8 keeps the same layout on the GPU. */
typedef struct {
    ath_dtype act;          /* activation dtype: ATH_DTYPE_BF16 (the checkpoint's, as the
                             * reference runs) or ATH_DTYPE_F32 (an exact forward pass)   */
    int32_t   capacity;     /* tokens the KV buffers hold                                 */
    int32_t   n_past;       /* tokens processed so far: the position of the next token    */
    void    **buf;          /* [n_states]: buf[i] holds m->states[i]                      */
} ath_state;

/* Tokens appended to the sequence of a state by one forward call. */
typedef struct {
    const int32_t *tokens;  /* [n_tokens] token ids                                      */
    int32_t        n_tokens;
    ath_tensor    *logits;  /* NULL (no output), or f32 [n_tokens, vocab_size] (every
                             * token) or [1, vocab_size] (the last token only)            */
} ath_batch;

/* Callbacks from inside the forward pass. Every member may be NULL; the
 * tensors are valid during the call only and must not be written. */
typedef struct {
    /* The residual stream entering layer `layer` (0 … n_layers − 1); with
     * layer == n_layers, the stream entering the final norm. x is
     * [n_tokens, hidden_size] in the activation dtype. Abliteration captures
     * it (phase 3); the forward tests compare it with the reference. */
    void (*on_residual)(void *u, int32_t layer, const ath_tensor *x);
    void  *u;
} ath_hooks;

struct ath_family {
    const char *name;
    /* true if this family handles the checkpoint described by config. */
    bool       (*match)(const ath_json *config);
    /* Fill m from m->config and m->ckpt (already open): dimensions, layers,
     * params and ignored groups. Everything is allocated in a. Fails with
     * ATH_ERR_UNSUPPORTED for configs the family does not handle,
     * ATH_ERR_NOT_FOUND for a missing tensor and ATH_ERR_FORMAT for a tensor
     * whose dtype or shape does not follow from the config. */
    ath_status (*load)(ath_model *m, ath_arena *a);
    /* Run b->tokens at positions s->n_past, s->n_past + 1, ...: read and update
     * the state buffers, call the hooks, write the logits. Called only through
     * ath_model_forward, which has checked every argument and advances
     * s->n_past afterwards. Working memory comes from scratch. NULL if the
     * family cannot run models yet. */
    ath_status (*forward)(const ath_model *m, const ath_batch *b, ath_state *s, const ath_hooks *h,
                          ath_arena *scratch);
};

/* Open the checkpoint in dir: parse config.json, pick the first compiled-in
 * family whose match accepts it, map the safetensors files, load, and check
 * coverage: every checkpoint tensor is viewed by at least one param or
 * belongs to exactly one ignored group, never both, and every param lies
 * inside the tensor it names. ATH_ERR_UNSUPPORTED if no family matches,
 * ATH_ERR_FORMAT for an unbound tensor. On failure everything is closed.
 * Tables live in a, which must outlive m. */
ath_status ath_model_open(ath_model *m, ath_arena *a, const char *dir);

/* Unmap the checkpoint. Param views become invalid. */
void ath_model_close(ath_model *m);

/* ---- running a model (state.c) ----------------------------------------------- */

/* Bytes of state for one sequence of up to capacity tokens, with activation
 * dtype act (ATH_DTYPE_BF16 or ATH_DTYPE_F32). ATH_ERR_RANGE if it does not
 * fit in size_t. */
ath_status ath_state_bytes(const ath_model *m, ath_dtype act, int32_t capacity, size_t *bytes);

/* Allocate the state of one sequence in a and reset it. capacity >= 1. */
ath_status ath_state_init(ath_state *s, const ath_model *m, ath_dtype act, int32_t capacity,
                          ath_arena *a);

/* Start a new sequence: n_past = 0, conv and recurrent state zeroed. (The
 * KV cache is not cleared: positions at or past n_past are never read.) */
void ath_state_reset(ath_state *s, const ath_model *m);

/* Run one batch through the model. Checks the arguments (ATH_ERR_INVALID_ARG,
 * ATH_ERR_TYPE; ATH_ERR_RANGE for a token id outside the vocabulary or a
 * batch that does not fit in the state; ATH_ERR_UNSUPPORTED if the family
 * has no forward pass), calls the family's forward and, on success, advances
 * s->n_past by b->n_tokens. On failure the state is undefined until
 * ath_state_reset. Working memory is allocated in scratch and released
 * before returning; h may be NULL. */
ath_status ath_model_forward(const ath_model *m, const ath_batch *b, ath_state *s, const ath_hooks *h,
                             ath_arena *scratch);

/* Families compiled into this binary (Makefile FAMILIES), in order. */
size_t            ath_family_count(void);
const ath_family *ath_family_at(size_t i);

#endif /* ATH_MODEL_H */
