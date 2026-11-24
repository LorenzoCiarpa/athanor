# Architecture

athanor is organized in layers. Each layer may depend only on the layers below it. Only one layer, `src/models/`, knows anything about a specific model family; everything above it talks to models through a narrow contract.

## Scope: engine versus harness

athanor is an engine plus a server, not an agent harness (decided 2026-11-24, CHANGE-001). It owns everything that depends on the model or on the computation: weights, memory placement, KV and recurrent state (including on disk), prompt rendering, tool-call formatting and parsing, and the OpenAI- and Anthropic-compatible APIs. It never executes tools, runs an agent loop, keeps long-term memory, compacts conversations or ships a UI. Any harness (Claude Code, OpenCode, Codex CLI, a custom one) connects through the API. AGENTS.md, invariant 10.

| Function | Layer | In athanor |
| --- | --- | --- |
| VRAM/RAM/disk placement; KV and recurrent state; on-disk prefix snapshots | engine (`src/infer`) | yes |
| Prompt rendering, tool definitions in the template | family (`text.c`) | yes |
| Parsing text, thinking and tool calls from output | family (`text.c`) | yes |
| OpenAI/Anthropic APIs, streaming, tool-call conversion | server (`src/net`) | yes |
| Tool execution, agent loop, permissions | harness | no |
| Long-term memory, knowledge base, context compaction | harness | no |
| Desktop app, GUI | product | no |

Tool calling is model-specific: how tool definitions enter the prompt and how a tool call, or a thinking block, looks in the output is fixed by the model's chat template. So rendering and parsing belong to the family, and converting the parsed events into API formats belongs to `src/net`.

## Layers

| Layer | Directory | Knows about families? | Role |
| --- | --- | --- | --- |
| Tool | `tools/` | no | the single `athanor` binary and its subcommands |
| Pipelines | `src/infer`, `src/quant`, `src/ablit`, `src/train`, `src/net` | no | the four stages and the HTTP server |
| Contract | `src/model/` | no | `model.h`: the interface every family implements; the static registry |
| Families | `src/models/<family>/` | **yes** | config parsing, tensor naming, forward/backward composition, text handling |
| Ops | `src/ops/` | no | numeric operators, each with a CPU reference and a GPU kernel |
| Text | `src/text/` | no | generic BPE engine and Unicode tables |
| Backend | `src/backend/` | no | CPU and GPU (CUDA and HIP from one source) runtime |
| Core | `src/core/` | no | arenas, tensors, files, JSON, safetensors, threads |

Dependency direction: `tools → pipelines → contract → families → ops/text → backend → core`.

## The contract

`src/model/model.h` is the most important file in the project. Pipelines see a model only through it. This is the header as of step 5 (comments shortened); members marked *later* join it when their first user lands.

```c
typedef enum {
    ATH_ROLE_EMBED, ATH_ROLE_LM_HEAD, ATH_ROLE_NORM,
    ATH_ROLE_MIX_IN, ATH_ROLE_MIX_OUT,        /* token-mixing block: attention or linear attention */
    ATH_ROLE_MLP_IN, ATH_ROLE_MLP_OUT,        /* dense MLP and shared expert                       */
    ATH_ROLE_ROUTER, ATH_ROLE_EXPERT_IN, ATH_ROLE_EXPERT_OUT,
    ATH_ROLE_CONV, ATH_ROLE_GATE, ATH_ROLE_OTHER,
    ATH_ROLE_COUNT
} ath_role;

#define ATH_P_WRITES_RESIDUAL 1u   /* abliteration projects the direction out of these */
#define ATH_P_SENSITIVE       2u   /* quantization keeps these at higher precision by default */
#define ATH_P_SHARED_EXPERT   4u
#define ATH_P_UNUSED          8u   /* present in the checkpoint, unused by the text path (e.g. vision encoder) */

typedef struct {
    ath_tensor t;          /* contiguous view of checkpoint memory (one expert of a fused tensor is a slice) */
    ath_role   role;
    int32_t    layer;      /* -1 for global tensors */
    int32_t    expert;     /* -1 if not an expert   */
    uint32_t   flags;
    const char *name;      /* original checkpoint name */
} ath_param;

typedef struct { const char *mixer, *ffn; int32_t n_experts; } ath_layer;   /* schedule, labels for people */
typedef struct { const char *prefix, *reason; uint32_t n_tensors; uint64_t bytes; } ath_ignored;  /* ATH_P_UNUSED params */

/* per-sequence state (step 4): sizes in elements; KV and CONV hold activations
 * (the state's activation dtype), RECURRENT is always fp32 */
typedef enum { ATH_STATE_KV, ATH_STATE_CONV, ATH_STATE_RECURRENT, ATH_STATE_KIND_COUNT } ath_state_kind;
typedef struct {
    ath_state_kind kind;
    int32_t        layer;
    int64_t        fixed;       /* CONV, RECURRENT        */
    int64_t        per_token;   /* KV: keys and values    */
} ath_state_spec;

typedef struct {
    const ath_family *family;
    const ath_json   *config;
    ath_safetensors   ckpt;                    /* params view its memory               */
    ath_param        *params;   uint32_t n_params;   /* forward order                  */
    ath_layer        *layers;   int32_t  n_layers;
    ath_ignored      *ignored;  uint32_t n_ignored;  /* tensors left unbound on purpose */
    uint32_t          n_bound;
    int32_t           hidden_size, vocab_size;
    bool              tie_embeddings;          /* EMBED is also the output projection   */
    int32_t           n_experts, n_experts_active;
    ath_state_spec   *states;   uint32_t n_states;   /* per-sequence state, layer order */
    void             *family_data;             /* family-private, opaque to pipelines   */
} ath_model;

typedef struct { ath_dtype act; int32_t capacity, n_past; void **buf; } ath_state;   /* one sequence */
typedef struct { const int32_t *tokens; int32_t n_tokens; ath_tensor *logits; } ath_batch;
typedef struct {
    void (*on_residual)(void *u, int32_t layer, const ath_tensor *x);   /* stream entering a layer */
    /* later (phase 2): on_matmul_in */
    void *u;
} ath_hooks;

/* text (step 5) */
typedef struct { const char *name, *description, *json_schema; } ath_tool_def;     /* schema: raw JSON */
typedef struct { const char *id, *name, *arguments; } ath_tool_call;               /* arguments: raw JSON object */
typedef enum { ATH_MSG_SYSTEM, ATH_MSG_USER, ATH_MSG_ASSISTANT, ATH_MSG_TOOL } ath_msg_role;
typedef struct {
    ath_msg_role role;
    const char *content, *reasoning;                 /* reasoning: thinking given apart, or NULL */
    const ath_tool_call *tool_calls; int32_t n_tool_calls;   /* assistant */
    const char *tool_call_id;                        /* tool */
} ath_msg;
#define ATH_CHAT_GENERATE 1u   /* end with the assistant's header */
#define ATH_CHAT_THINKING 2u
typedef enum {
    ATH_OUT_TEXT, ATH_OUT_THINKING,
    ATH_OUT_TOOL_CALL_BEGIN, ATH_OUT_TOOL_CALL_ARGS, ATH_OUT_TOOL_CALL_END
} ath_out_kind;
typedef struct { ath_out_kind kind; const char *data; size_t len; const char *tool_name; } ath_out_event;
typedef struct {
    uint32_t opts; const ath_tool_def *tools; int32_t n_tools;   /* set by the caller */
    ath_arena *a;                                                 /* parser memory, event bytes */
    void *priv;                                                   /* the family's state */
} ath_parse_state;

struct ath_family {
    const char  *name;
    bool        (*match)  (const ath_json *config);
    ath_status  (*load)   (ath_model *m, ath_arena *a);      /* reads m->config, m->ckpt */
    ath_status  (*forward)(const ath_model *m, const ath_batch *b, ath_state *s,
                           const ath_hooks *h, ath_arena *scratch, ath_pool *pool);
    const char  *pretokenize_regex;                          /* the tokenizer.json Split regex it implements */
    ath_status  (*pretokenize)(const char *text, size_t n, ath_spans *out);
    ath_status  (*chat) (const ath_msg *msgs, int32_t n_msgs, const ath_tool_def *tools, int32_t n_tools,
                         uint32_t opts, ath_buf *out);
    ath_status  (*parse)(ath_parse_state *st, const char *piece, size_t n,   /* piece NULL: end of output */
                         ath_out_event *events, int32_t cap, int32_t *n_events);
    /* later: backward (phase 4) */
};

ath_status ath_model_open(ath_model *m, ath_arena *a, const char *dir);
void       ath_model_close(ath_model *m);
ath_status ath_state_bytes(const ath_model *m, ath_dtype act, int32_t capacity, size_t *bytes);
ath_status ath_state_init(ath_state *s, const ath_model *m, ath_dtype act, int32_t capacity, ath_arena *a);
void       ath_state_reset(ath_state *s, const ath_model *m);
ath_status ath_state_copy(ath_state *dst, const ath_state *src, const ath_model *m);
ath_status ath_model_forward(const ath_model *m, const ath_batch *b, ath_state *s,
                             const ath_hooks *h, ath_arena *scratch, ath_pool *pool);
ath_status ath_model_tokenizer(const ath_model *m, const char *dir, ath_tokenizer *t, ath_arena *a);
ath_status ath_model_chat(const ath_model *m, const ath_msg *msgs, int32_t n_msgs,
                          const ath_tool_def *tools, int32_t n_tools, uint32_t opts, ath_buf *out);
ath_status ath_model_parse(const ath_model *m, ath_parse_state *st, const char *piece, size_t n,
                           ath_out_event *events, int32_t cap, int32_t *n_events);
```

Members still to come, as designed:

```c
typedef struct {
    void (*on_residual) (void *u, int32_t layer, const ath_tensor *x);             /* abliteration capture (step 4) */
    void (*on_matmul_in)(void *u, const ath_param *w, const ath_tensor *x);        /* quant calibration (phase 2)   */
    void *u;
} ath_hooks;

    ath_status   (*backward)   (ath_model *m, ath_batch *b, ath_grads *g);     /* NULL until phase 4 */
```

`load` takes the open checkpoint (`ath_safetensors`) rather than the `ath_weights` abstraction of the first design: there is one weight source until the `.ath` container lands in phase 2, which will introduce the abstraction with its second user.

Since CHANGE-001 (2026-11-24) every checkpoint tensor is a param. Tensors the text path does not use, the vision encoder of a vision-language checkpoint and, until step 13, the MTP head, are bound as whole-tensor params flagged `ATH_P_UNUSED` (role other, layer −1, after all the other params), and each belongs to an ignored group that the family declares with a prefix and a reason. Pipelines on the text path skip them; an export (quantized container, abliterated checkpoint) copies them unchanged, so the result keeps its vision encoder.

Step 4 departs from the first design of the state in three places. A spec counts elements, not bytes, because the element type depends on the activation dtype the state is created with (bf16, as the reference runs, or f32 for an exact forward pass); the kind fixes the rest (KV and conv state hold activations, the recurrent state is fp32). `forward` takes the model as `const` and a scratch arena for its working tensors, which the contract releases after every call. A batch is one sequence; batches of several sequences arrive with the engine (step 8). Since task 4.7, `forward` also takes the caller's thread pool (`ath_pool`, `src/core/threads.h`; NULL runs on the calling thread). It is passed explicitly because the code has no globals. The CPU matmuls spread over it with the same bits as on one thread.

### Text

The text members joined the contract in step 5, as CHANGE-001 (2026-11-24) specified them, with the details step 5 settled. `ath_spans` and `ath_tokenizer` are the generic byte-level BPE engine of `src/text/bpe.h`; `ath_buf` is a growable buffer in an arena (`src/core/buf.h`).

- `pretokenize` splits text into the pieces BPE merges within: the family's tokenizer regex, written by hand. `pretokenize_regex` is that regex as `tokenizer.json` writes it; `ath_model_tokenizer` loads the tokenizer with the family's split and refuses a file whose regex differs.
- `chat` renders a conversation, with the tool definitions and the thinking switch, into the exact prompt text of the model's chat template, and fails (`ATH_ERR_INVALID_ARG`) where the template raises an error. Tool definitions and arguments are raw JSON; the family serializes them as the template does.
- `parse` turns the decoded output, piece by piece as it streams, into events: text, thinking, and tool calls (name, then argument bytes that form one JSON object, then end). Markup split across pieces is held back until it is complete; malformed markup comes out as text; a NULL piece ends the output and decides what was held back. The caller zeroes an `ath_parse_state`, sets the prompt's options, its tools (their schemas type the arguments) and an arena; the family keeps its state behind `priv`. Event bytes are valid until the next call.
- `ath_state_copy` copies a sequence's state. A pipeline keeps a copy where a conversation will come back to, because a recurrent state cannot be rolled back: `athanor chat` keeps one at the end of each prompt's last message, since the template renders a past assistant turn differently from the prompt that produced it.

### State and the forward pass

A model lists the state one sequence needs as specs: for Qwen3.5, a KV cache per attention layer and a conv state plus a recurrent state per Gated DeltaNet layer. `ath_state_init` allocates one buffer per spec in host memory and `ath_state_reset` starts a new sequence; a pipeline never needs to know which layer types produced the specs. `ath_model_forward` is the only way into a family's `forward`: it checks the batch (token ids in the vocabulary, room in the state, logits f32 `[n_tokens, vocab]` or `[1, vocab]`), runs the family's forward at positions `n_past …`, releases the scratch memory and advances `n_past`. Prefill, a prompt in pieces and decode take the same path, and give the same logits and state bit for bit (`tests/test_forward.c`). `on_residual` sees the residual stream entering every layer and the final norm; abliteration will capture it, and the forward tests compare it with the reference.

### Loading and the coverage rule

`ath_model_open(dir)` parses `config.json`, asks each compiled-in family whether it `match`es, maps the checkpoint, and calls the family's `load`. Then it checks, independently of the family, that every checkpoint tensor is viewed by at least one param, that the tensors of each ignored group are viewed by `ATH_P_UNUSED` params only and belong to no other group, that every `ATH_P_UNUSED` param views a tensor of some group, and that every param is a contiguous view inside the tensor it names. A tensor nobody accounts for fails the load (`ATH_ERR_FORMAT`): a checkpoint that is not what the family thinks it is must not run, and no tensor is ignored silently. Families report missing tensors (`ATH_ERR_NOT_FOUND`) and shapes or dtypes that do not follow from the config (`ATH_ERR_FORMAT`). `athanor inspect <dir> --roles` prints the result.

The family list is compiled in from the Makefile: `FAMILIES=qwen35` generates `build/gen/families.h` with one `ATH_FAMILY(qwen35)` line, and each family defines `const ath_family ath_family_<name>`. `registry.c` names no family.

### How each pipeline uses it

| Pipeline | Uses |
| --- | --- |
| infer | `states` (the state specs) to allocate KV, recurrent and conv state without knowing the block types; `ath_model_forward`; params with `layer` and `expert` to place MoE experts across tiers |
| quant | params by `role` and `flags` to choose formats; `on_matmul_in` to collect activation statistics for importance-weighted RTN, GPTQ and AWQ; `ATH_P_UNUSED` params copied unchanged into the container |
| ablit | `on_residual` to capture activations; every param with `ATH_P_WRITES_RESIDUAL` is orthogonalized; `ATH_P_UNUSED` params exported unchanged |
| train | `backward`; LoRA targets by `role`; hot/cold blocks by `layer`; `ATH_P_UNUSED` params frozen and never loaded |
| net | `chat` with the request's tool definitions to render the prompt; `parse` to turn the output into text, thinking and tool-call events, which `api.c` converts to OpenAI and Anthropic formats |
| `athanor chat` (step 5) | `ath_model_tokenizer`, `ath_model_chat`, `ath_model_forward`, `ath_sample_ref`, `ath_model_parse`; `ath_state_copy` to resume a conversation |

## Common versus family-specific

| Common (written once) | Family-specific (`src/models/<family>/`) |
| --- | --- |
| core, backend, all ops (kernels parameterized by shape and format) | `config.c`: `config.json` → dimensions and layer schedule |
| BPE engine and Unicode tables | `weights.c`: checkpoint names → params with roles and flags |
| inference engine: states, scheduler, sampling, MoE tiering, speculative decoding | `forward.c` and later `backward.c`: composition of ops per block |
| quantization formats, algorithms, recipes driven by roles | `text.c`: pre-tokenization split, chat template with tool definitions and thinking switch, output parser (thinking blocks, tool calls): tool rendering and output parsing are family-specific |
| abliteration driven by hooks and flags | default quantization recipe (which tensors are sensitive) |
| training: optimizers, LoRA, hot/cold layout, planner, data | optional fused kernels in `fused.cu`, admitted only with a benchmark |
| server (API formats: conversion of output events to OpenAI `tool_calls` and Anthropic `tool_use`), tools, test harness | golden test configuration |

## Adding models

| Case | Example | What you add | What changes in common code |
| --- | --- | --- | --- |
| New size | Qwen3.5-27B | golden test only | nothing |
| New version, same blocks | Qwen3.6-35B-A3B | possibly name or flag handling in `config.c`, `weights.c` | nothing |
| New version, new blocks | Qwen3.8-Flash-Next | new ops (sparse attention, 4-branch gated residual, n-gram table) and an updated forward | `src/ops/` by addition; maybe a new state kind |
| New family | a family with MLA or sliding-window attention | `src/models/<family>/` plus missing ops | `src/ops/` by addition; `src/text/` only if the tokenizer is not byte-level BPE |
| New weight format | a new FP4 variant | format, matmul kernels, algorithm | `src/quant/`, `src/ops/matmul` |
| New hardware | Apple Metal | backend and op ports | `src/backend/`, `src/ops/` |

Common code grows only by addition. Pipelines never change to accommodate a family; if they would, the contract is missing a concept.

## Directory tree

```
athanor/
├── Makefile                     targets: cpu (default), cuda, hip, test, clean; FAMILIES=qwen35
├── include/athanor.h            public C API
├── src/
│   ├── core/      base.c arena.c buf.c tensor.c file.c json.c safetensors.c threads.c rng.c (each with its .h)
│   ├── backend/   backend.h compat.h cpu.c gpu.cu
│   ├── ops/       ops.h, check.h (private); per op a .c reference and a .cu kernel:
│   │              embed matmul norm rope act attn conv gdn (step 3), sample (step 5), moe quant
│   ├── text/      bpe.c unicode.c, unicode_tables.h (generated, committed)
│   ├── model/     model.h registry.c state.c text.c
│   ├── models/
│   │   └── qwen35/  qwen35.h family.c config.c weights.c forward.c text.c [backward.c] [fused.cu]
│   ├── infer/     engine.c state.c sched.c tier.c spec.c
│   ├── quant/     formats.c rtn.c gptq.c awq.c recipe.c container.c gguf.c eval.c
│   ├── ablit/     capture.c direction.c edit.c search.c eval.c
│   ├── train/     loop.c optim.c lora.c hotcold.c data.c planner.c ckpt.c
│   └── net/       http.c api.c
├── tools/         athanor.c (dispatch), one file per subcommand (inspect.c, chat.c, ...), dev/ (development helpers)
├── tests/         test.h (runner) util.h tiny.h test_core.c test_model.c test_ops.c test_golden.c test_forward.c
│                  test_text.c ...; golden/ (dump.py, text.py, requirements.txt; data/ generated, not committed)
└── docs/          ARCHITECTURE.md PLAN.md STATUS.md status_extended.md MODELS.md BENCHMARKS.md
                   CHANGE-*.md (specification changes, applied as recorded in STATUS.md)
                   contract/ program/ proposals/ results/
```

## What athanor writes itself instead of using libraries

| Usually a library | In athanor |
| --- | --- |
| JSON | read-only parser into an arena |
| safetensors, GGUF | direct readers; both are a header plus offsets |
| tokenizer regex | a hand-written split function per family; Unicode category and normalization tables generated once into `src/text/unicode_tables.h` |
| chat template engine (Jinja) | the family's template rendered by hand in C, compared byte for byte with the reference's rendering |
| HTTP server with streaming | minimal HTTP/1.1 with server-sent events over POSIX sockets, localhost only |
| thread pool, async I/O | pthreads and `pread`; io_uring through raw syscalls later |
| test framework | a minimal C runner |
| dense GEMM | cuBLASLt / hipBLASLt behind `ops/matmul` (part of the toolkits); own kernels for GEMV, quantized formats and fused ops |

## Rules that keep the codebase small

1. Families are selected at compile time (`make FAMILIES=qwen35`) and registered in a static table; no plugins.
2. No op without its CPU reference and test.
3. Fused kernels live in the family folder, need a benchmark that shows the gain, and keep the op composition as fallback and reference.
4. No model names outside `src/models/`.
5. A pipeline that needs new information extends the contract, never works around it.
