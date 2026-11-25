/* model.h: the contract between pipelines and model families.
 *
 * Pipelines (infer, quant, ablit, train) and tools see a model only through
 * this header. A family (src/models/<family>/) turns a checkpoint into an
 * ath_model: every checkpoint tensor becomes one or more params, each with a
 * role, a layer, an expert index and flags. Tensors the text path does not
 * use (a vision encoder, for example) are params too, flagged ATH_P_UNUSED,
 * and belong to a group that the family declares and explains (an ignored
 * group): bound and ignored explicitly, never silently. Pipelines decide
 * what to do by role and flags, never by tensor name or by family:
 *
 *   quantization   formats by role, higher precision for ATH_P_SENSITIVE
 *   abliteration   orthogonalizes every param flagged ATH_P_WRITES_RESIDUAL
 *   training       LoRA targets by role, hot/cold blocks by layer
 *   MoE placement  experts by layer and expert
 *
 * Step 2 defined roles, params, the layer schedule and loading. Step 4 adds
 * the forward pass: the per-sequence state a model needs (state specs, so
 * that pipelines allocate it without knowing the layer types), batches and
 * hooks. Step 5 adds text: the family's pre-tokenization, its chat template
 * (with tool definitions and the thinking switch) and the parser of its
 * output (thinking, tool calls). The matmul hook (phase 2) and the backward
 * pass (phase 4) join the contract when their first user lands;
 * docs/ARCHITECTURE.md shows the target.
 */
#ifndef ATH_MODEL_H
#define ATH_MODEL_H

#include "backend/backend.h"
#include "core/arena.h"
#include "core/buf.h"
#include "core/json.h"
#include "core/ckpt.h"
#include "core/tensor.h"
#include "core/threads.h"
#include "ops/ops.h"
#include "text/bpe.h"

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
    ATH_ROLE_EXPERT_IN,   /* routed expert input matmul(s), per expert one param per
                           * checkpoint tensor (gate and up fused or apart)            */
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
#define ATH_P_UNUSED          8u  /* present in the checkpoint, unused by the text path
                                   * (e.g. vision encoder): the forward pass never reads it,
                                   * pipelines on the text path skip it, an export copies it
                                   * unchanged. Role other, layer -1, expert -1; it belongs
                                   * to exactly one ignored group                          */

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

/* Checkpoint tensors the text path does not use, grouped by name prefix.
 * The family sets prefix and reason and binds every tensor of the group as a
 * param flagged ATH_P_UNUSED; ath_model_open counts the matching tensors and
 * checks that the group and the unused params coincide. */
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

/* ---- text (step 5) ------------------------------------------------------------
 *
 * A conversation goes in as messages and tool definitions; the family
 * renders it into the exact prompt text of the model's chat template. The
 * model's output comes back as text pieces (decoded tokens), and the family
 * parses them into events: answer text, thinking, and tool calls with their
 * arguments as a JSON object. athanor never runs a tool (AGENTS.md,
 * invariant 10): a harness reads the calls through the API and sends the
 * results back as messages with role ATH_MSG_TOOL. */

typedef struct {
    const char *name;
    const char *description;   /* NULL: none                                     */
    const char *json_schema;   /* the parameters, a JSON Schema object as raw
                                * JSON text; NULL: none                         */
} ath_tool_def;

typedef struct {
    const char *id;            /* the harness's id of the call; may be NULL      */
    const char *name;
    const char *arguments;     /* a JSON object as raw text; NULL: none          */
} ath_tool_call;

typedef enum { ATH_MSG_SYSTEM, ATH_MSG_USER, ATH_MSG_ASSISTANT, ATH_MSG_TOOL } ath_msg_role;

typedef struct {
    ath_msg_role         role;
    const char          *content;        /* UTF-8, NUL-terminated; NULL as empty        */
    const char          *reasoning;      /* assistant: its thinking, when given apart
                                          * from content; NULL: none given             */
    const ath_tool_call *tool_calls;     /* assistant: the calls it made                */
    int32_t              n_tool_calls;
    const char          *tool_call_id;   /* tool: the id of the call this answers       */
} ath_msg;

#define ATH_CHAT_GENERATE 1u   /* end with the assistant's header: the model writes next */
#define ATH_CHAT_THINKING 2u   /* the model thinks before it answers                       */

typedef enum {
    ATH_OUT_TEXT,              /* answer text                                              */
    ATH_OUT_THINKING,          /* thinking text                                            */
    ATH_OUT_TOOL_CALL_BEGIN,   /* a tool call starts; tool_name is set                     */
    ATH_OUT_TOOL_CALL_ARGS,    /* bytes of its arguments; the bytes of all the ARGS events
                                * of one call form a JSON object                           */
    ATH_OUT_TOOL_CALL_END
} ath_out_kind;

typedef struct {
    ath_out_kind kind;
    const char  *data;         /* text, thinking or argument bytes                         */
    size_t       len;
    const char  *tool_name;    /* TOOL_CALL_BEGIN: the function's name, NUL-terminated     */
} ath_out_event;

/* The parser of one model output. Zero it, set the fields above priv, and
 * pass it to every parse call of that output. */
typedef struct {
    uint32_t            opts;      /* the ATH_CHAT_* options the prompt was rendered with */
    const ath_tool_def *tools;     /* the prompt's tools: their schemas give the types of
                                    * the arguments                                       */
    int32_t             n_tools;
    ath_arena          *a;         /* the parser's memory and the bytes of its events      */
    void               *priv;      /* the family's state, NULL before the first call       */
} ath_parse_state;

typedef struct ath_family ath_family;

typedef struct {
    const ath_family *family;
    const ath_json   *config;            /* parsed config.json                         */
    ath_ckpt          ckpt;              /* mapped checkpoint (safetensors or GGUF);
                                          * params view its memory                   */

    ath_param        *params;            /* global params and each layer's params, in
                                          * forward order; then the ATH_P_UNUSED ones */
    uint32_t          n_params;
    ath_layer        *layers;
    int32_t           n_layers;
    ath_ignored      *ignored;
    uint32_t          n_ignored;
    uint32_t          n_bound;           /* checkpoint tensors viewed by some param:
                                          * all of them once the model is open       */

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

/* The state of one sequence: in host memory (ath_state_init) or in the
 * memory of a device (ath_state_init_gpu, step 8), with the same layout. */
typedef struct {
    ath_dtype         act;      /* activation dtype: ATH_DTYPE_BF16 (the checkpoint's, as the
                                 * reference runs) or ATH_DTYPE_F32 (an exact forward pass)   */
    int32_t           capacity; /* tokens the KV buffers hold                                 */
    int32_t           n_past;   /* tokens processed so far: the position of the next token    */
    void            **buf;      /* [n_states]: buf[i] holds m->states[i] (the pointer array
                                 * itself is always in host memory)                           */
    const ath_device *dev;      /* NULL: the buffers are in host memory; otherwise on dev     */
} ath_state;

/* Where a forward pass runs (step 8). On the CPU every member but pool is
 * NULL, and a NULL ath_exec means the CPU on the calling thread. On a GPU,
 * dev, weights and scratch are set, and the state must be on dev
 * (ath_state_init_gpu): src/infer/engine.h builds it. */
typedef struct {
    ath_pool         *pool;      /* CPU threads for the fast paths; NULL: the calling thread     */
    const ath_device *dev;       /* the device that runs the pass; NULL: the CPU                 */
    const ath_tensor *weights;   /* dev: [n_params], each param's copy in dev's memory (ignored
                                  * params: an empty descriptor; a routed expert's params may be
                                  * empty too, see experts); NULL: the params themselves         */
    ath_dev_arena    *scratch;   /* dev: working memory, released after every call               */
    /* MoE on a GPU (step 12): per layer [n_layers] the routed experts wherever they are
     * (ops.h, ath_moe_experts: device and host copies; NULL entries for dense layers), and
     * the working state of the CPU's share. The engine sets them (src/infer/tier.c); a
     * family reads its experts there, never in weights. A routed expert's params, in
     * param order, are its gate, up and down projections: the tables list them so. */
    const ath_moe_experts *experts;
    ath_moe_ws            *moe_ws;
} ath_exec;

/* Tokens appended to the sequence of a state by one forward call. */
typedef struct {
    const int32_t *tokens;  /* [n_tokens] token ids                                      */
    int32_t        n_tokens;
    ath_tensor    *logits;  /* NULL (no output), or f32 [n_tokens, vocab_size] (every
                             * token) or [1, vocab_size] (the last token only)            */
    const int32_t *pos;     /* ATH_BATCH_DEVICE: [n_tokens] positions; NULL otherwise    */
    uint32_t       flags;   /* ATH_BATCH_*                                               */
} ath_batch;

/* A batch whose tokens, positions and logits are in the memory of the
 * device the pass runs on (step 9). The pass then copies nothing between
 * the host and the device and does not wait for the device: everything it
 * does is queued on the compute stream, so the call can be captured in a
 * graph and replayed (backend.h, ath_graph). The positions must equal
 * n_past, n_past + 1, ... when the work runs, and the token ids must be in
 * the vocabulary: the host cannot read them to check. */
#define ATH_BATCH_DEVICE 1u

/* Callbacks from inside the forward pass. Every member may be NULL; the
 * tensors are valid during the call only and must not be written. */
typedef struct {
    /* The residual stream entering layer `layer` (0 … n_layers − 1); with
     * layer == n_layers, the stream entering the final norm. x is
     * [n_tokens, hidden_size] in the activation dtype, where the pass runs:
     * in device memory (tier VRAM) on a GPU, where work on the compute
     * stream may still be writing it, so a hook that reads it copies it on
     * that stream and waits. Abliteration captures it (phase 3); the forward
     * tests compare it with the reference. */
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
     * s->n_past afterwards. x says where it runs (ath_exec; never NULL
     * here): on the CPU, working memory comes from scratch and CPU work may
     * be spread over the threads of x->pool, which never changes a result;
     * on a GPU, the ops run on x->dev with the params' copies x->weights,
     * device working memory comes from x->scratch, and the logits are copied
     * to the batch's host tensor before returning. NULL if the family cannot
     * run models yet. */
    ath_status (*forward)(const ath_model *m, const ath_batch *b, ath_state *s, const ath_hooks *h,
                          ath_arena *scratch, const ath_exec *x);

    /* ---- text (step 5); NULL if the family has none ---- */

    /* The Split regex of tokenizer.json that pretokenize implements; the
     * tokenizer refuses a file with another one. */
    const char *pretokenize_regex;
    /* Split text into the pieces BPE merges within (src/text/bpe.h). */
    ath_status (*pretokenize)(const char *text, size_t n, ath_spans *out);
    /* Append the prompt for msgs (and the tool definitions, if n_tools > 0)
     * to out, byte for byte as the model's chat template renders it.
     * ATH_ERR_INVALID_ARG where the template raises an error (no messages,
     * a system message after the first, no user message) or for arguments
     * that are not a JSON object. */
    ath_status (*chat)(const ath_msg *msgs, int32_t n_msgs, const ath_tool_def *tools, int32_t n_tools,
                       uint32_t opts, ath_buf *out);
    /* Parse the next piece of output (the decoded bytes of one or more
     * tokens, split anywhere) into events, at most cap. piece == NULL ends
     * the output: whatever was held back is decided. Markup split across
     * pieces is held back until it is complete; malformed markup comes out
     * as text. Event bytes are valid until the next call. cap >= n + 8 is
     * always enough; otherwise ATH_ERR_RANGE, and the state is undefined. */
    ath_status (*parse)(ath_parse_state *st, const char *piece, size_t n, ath_out_event *events, int32_t cap,
                        int32_t *n_events);
};

/* Open the checkpoint in dir: parse config.json, pick the first compiled-in
 * family whose match accepts it, map the weights (safetensors or one GGUF
 * file: core/ckpt.h), load, and check
 * coverage: every checkpoint tensor is viewed by at least one param; the
 * tensors of each ignored group are viewed by ATH_P_UNUSED params only, and
 * every ATH_P_UNUSED param views a tensor of exactly one group; every param
 * lies inside the tensor it names. ATH_ERR_UNSUPPORTED if no family matches,
 * ATH_ERR_FORMAT for an unbound tensor or a broken group. On failure
 * everything is closed. Tables live in a, which must outlive m. */
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

/* The same in the memory of device d (step 8): the buffers from mem, the
 * pointer array from a. */
ath_status ath_state_init_gpu(ath_state *s, const ath_model *m, ath_dtype act, int32_t capacity,
                              const ath_device *d, ath_dev_arena *mem, ath_arena *a);

/* Start a new sequence: n_past = 0, conv and recurrent state zeroed. (The
 * KV cache is not cleared: positions at or past n_past are never read.) On
 * a device the zeroing is queued on its compute stream, before any later
 * forward pass. */
ath_status ath_state_reset(ath_state *s, const ath_model *m);

/* Copy the state of one sequence into another state of the same model,
 * activation dtype and capacity: every buffer and n_past. A copy taken at
 * a point the conversation will come back to (the end of a turn's stable
 * prefix) lets a pipeline resume there, since a recurrent state cannot be
 * rolled back. ATH_ERR_INVALID_ARG if the two states differ in dtype,
 * capacity or device. On a device the copy is queued on its compute
 * stream, in order with the forward passes. */
ath_status ath_state_copy(ath_state *dst, const ath_state *src, const ath_model *m);

/* Run one batch through the model. Checks the arguments (ATH_ERR_INVALID_ARG,
 * ATH_ERR_TYPE; ATH_ERR_RANGE for a token id outside the vocabulary or a
 * batch that does not fit in the state; ATH_ERR_UNSUPPORTED if the family
 * has no forward pass), calls the family's forward and, on success, advances
 * s->n_past by b->n_tokens. On failure the state is undefined until
 * ath_state_reset. Working memory is allocated in scratch (and on a GPU in
 * x->scratch) and released before returning; h may be NULL.
 *
 * x says where the pass runs (ath_exec): NULL or a CPU exec for a state in
 * host memory, where x->pool lends threads to the CPU work and the results
 * are the same, bit for bit, with any number of threads; a GPU exec for a
 * state on that device (ATH_ERR_INVALID_ARG otherwise). On a GPU the
 * logits, in host memory, are ready when the call returns; a batch without
 * logits returns as soon as its work is queued. A device batch
 * (ATH_BATCH_DEVICE, GPU only) has its tokens, positions and logits on the
 * device and returns as soon as its work is queued; its token ids and
 * positions are the caller's to guarantee. */
ath_status ath_model_forward(const ath_model *m, const ath_batch *b, ath_state *s, const ath_hooks *h,
                             ath_arena *scratch, const ath_exec *x);

/* ---- text (text.c) ------------------------------------------------------------- */

/* Load the tokenizer of the checkpoint in dir (tokenizer.json) with the
 * family's pre-tokenization. ATH_ERR_UNSUPPORTED if the family has none. */
ath_status ath_model_tokenizer(const ath_model *m, const char *dir, ath_tokenizer *t, ath_arena *a);

/* The family's chat and parse (ath_family), after checking the arguments;
 * ATH_ERR_UNSUPPORTED if the family has none. */
ath_status ath_model_chat(const ath_model *m, const ath_msg *msgs, int32_t n_msgs, const ath_tool_def *tools,
                          int32_t n_tools, uint32_t opts, ath_buf *out);
ath_status ath_model_parse(const ath_model *m, ath_parse_state *st, const char *piece, size_t n,
                           ath_out_event *events, int32_t cap, int32_t *n_events);

/* Families compiled into this binary (Makefile FAMILIES), in order. */
size_t            ath_family_count(void);
const ath_family *ath_family_at(size_t i);

#endif /* ATH_MODEL_H */
