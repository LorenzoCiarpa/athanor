# Architecture

athanor is organized in layers. Each layer may depend only on the layers below it. Only one layer, `src/models/`, knows anything about a specific model family; everything above it talks to models through a narrow contract.

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

`src/model/model.h` is the most important file in the project. Pipelines see a model only through it. This is the header as of step 2 (comments shortened); members marked *later* join it when their first user lands.

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

typedef struct {
    ath_tensor t;          /* contiguous view of checkpoint memory (one expert of a fused tensor is a slice) */
    ath_role   role;
    int32_t    layer;      /* -1 for global tensors */
    int32_t    expert;     /* -1 if not an expert   */
    uint32_t   flags;
    const char *name;      /* original checkpoint name */
} ath_param;

typedef struct { const char *mixer, *ffn; int32_t n_experts; } ath_layer;   /* schedule, labels for people */
typedef struct { const char *prefix, *reason; uint32_t n_tensors; uint64_t bytes; } ath_ignored;

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
    void             *family_data;             /* family-private, opaque to pipelines   */
    /* later (step 4): ath_state_spec *state_specs; uint32_t n_states; */
} ath_model;

struct ath_family {
    const char  *name;
    bool        (*match)      (const ath_json *config);
    ath_status  (*load)       (ath_model *m, ath_arena *a);      /* reads m->config, m->ckpt */
    /* later: forward (step 4), pretokenize and chat (step 5), backward (phase 4) */
};

ath_status ath_model_open(ath_model *m, ath_arena *a, const char *dir);
void       ath_model_close(ath_model *m);
```

Members still to come, as designed:

```c
typedef enum { ATH_STATE_KV, ATH_STATE_RECURRENT, ATH_STATE_CONV } ath_state_kind;

typedef struct {
    ath_state_kind kind;
    int32_t layer;
    size_t  fixed_bytes;       /* recurrent and conv state */
    size_t  bytes_per_token;   /* KV cache                 */
} ath_state_spec;

typedef struct {
    void (*on_residual) (void *u, int32_t layer, const ath_tensor *x);             /* abliteration capture */
    void (*on_matmul_in)(void *u, const ath_param *w, const ath_tensor *x);        /* quant calibration    */
    void *u;
} ath_hooks;

    ath_status   (*forward)    (ath_model *m, ath_batch *b, ath_state *s, const ath_hooks *h);
    ath_status   (*backward)   (ath_model *m, ath_batch *b, ath_grads *g);     /* NULL until phase 4 */
    ath_status   (*pretokenize)(const char *text, size_t n, ath_spans *out);
    ath_status   (*chat)       (const ath_msg *msgs, int32_t n, ath_buf *out);
```

`load` takes the open checkpoint (`ath_safetensors`) rather than the `ath_weights` abstraction of the first design: there is one weight source until the `.ath` container lands in phase 2, which will introduce the abstraction with its second user.

### Loading and the coverage rule

`ath_model_open(dir)` parses `config.json`, asks each compiled-in family whether it `match`es, maps the checkpoint, and calls the family's `load`. Then it checks, independently of the family, that every checkpoint tensor is viewed by at least one param or falls in exactly one ignored group, never both, and that every param is a contiguous view inside the tensor it names. A tensor nobody accounts for fails the load (`ATH_ERR_FORMAT`): a checkpoint that is not what the family thinks it is must not run. Families report missing tensors (`ATH_ERR_NOT_FOUND`) and shapes or dtypes that do not follow from the config (`ATH_ERR_FORMAT`). `athanor inspect <dir> --roles` prints the result.

The family list is compiled in from the Makefile: `FAMILIES=qwen35` generates `build/gen/families.h` with one `ATH_FAMILY(qwen35)` line, and each family defines `const ath_family ath_family_<name>`. `registry.c` names no family.

### How each pipeline uses it

| Pipeline | Uses |
| --- | --- |
| infer | `state_specs` to allocate KV, recurrent and conv state without knowing the block types; `forward`; params with `layer` and `expert` to place MoE experts across tiers |
| quant | params by `role` and `flags` to choose formats; `on_matmul_in` to collect activation statistics for importance-weighted RTN, GPTQ and AWQ |
| ablit | `on_residual` to capture activations; every param with `ATH_P_WRITES_RESIDUAL` is orthogonalized |
| train | `backward`; LoRA targets by `role`; hot/cold blocks by `layer` |

## Common versus family-specific

| Common (written once) | Family-specific (`src/models/<family>/`) |
| --- | --- |
| core, backend, all ops (kernels parameterized by shape and format) | `config.c`: `config.json` → dimensions and layer schedule |
| BPE engine and Unicode tables | `weights.c`: checkpoint names → params with roles and flags |
| inference engine: states, scheduler, sampling, MoE tiering, speculative decoding | `forward.c` and later `backward.c`: composition of ops per block |
| quantization formats, algorithms, recipes driven by roles | `text.c`: pre-tokenization split and chat template |
| abliteration driven by hooks and flags | default quantization recipe (which tensors are sensitive) |
| training: optimizers, LoRA, hot/cold layout, planner, data | optional fused kernels in `fused.cu`, admitted only with a benchmark |
| server, tools, test harness | golden test configuration |

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
│   ├── core/      base.c arena.c tensor.c file.c json.c safetensors.c threads.c rng.c (each with its .h)
│   ├── backend/   backend.h compat.h cpu.c gpu.cu
│   ├── ops/       ops.h; per op a .c reference and a .cu kernel:
│   │              norm rope act embed matmul attn gdn conv moe sample quant
│   ├── text/      bpe.c unicode.h (generated, committed)
│   ├── model/     model.h registry.c
│   ├── models/
│   │   └── qwen35/  qwen35.h family.c config.c weights.c forward.c text.c [backward.c] [fused.cu]
│   ├── infer/     engine.c state.c sched.c tier.c spec.c
│   ├── quant/     formats.c rtn.c gptq.c awq.c recipe.c container.c gguf.c eval.c
│   ├── ablit/     capture.c direction.c edit.c search.c eval.c
│   ├── train/     loop.c optim.c lora.c hotcold.c data.c planner.c ckpt.c
│   └── net/       http.c api.c
├── tools/         athanor.c (dispatch), one file per subcommand (inspect.c, ...), dev/ (development helpers)
├── tests/         test.h (runner) util.h test_core.c test_model.c test_ops.c ... golden/dump.py
└── docs/          ARCHITECTURE.md PLAN.md STATUS.md MODELS.md BENCHMARKS.md contract/ program/ results/
```

## What athanor writes itself instead of using libraries

| Usually a library | In athanor |
| --- | --- |
| JSON | read-only parser into an arena |
| safetensors, GGUF | direct readers; both are a header plus offsets |
| tokenizer regex | a hand-written split function per family; Unicode category tables generated once into `src/text/unicode.h` |
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
