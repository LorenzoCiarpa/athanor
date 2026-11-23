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

`src/model/model.h` is the most important file in the project. Pipelines see a model only through it.

```c
typedef enum {
    ATH_ROLE_EMBED, ATH_ROLE_LM_HEAD, ATH_ROLE_NORM,
    ATH_ROLE_MIX_IN, ATH_ROLE_MIX_OUT,        /* token-mixing block: attention or linear attention */
    ATH_ROLE_MLP_IN, ATH_ROLE_MLP_OUT,
    ATH_ROLE_ROUTER, ATH_ROLE_EXPERT_IN, ATH_ROLE_EXPERT_OUT,
    ATH_ROLE_CONV, ATH_ROLE_GATE, ATH_ROLE_OTHER
} ath_role;

#define ATH_P_WRITES_RESIDUAL 1u   /* abliteration projects the direction out of these */
#define ATH_P_SENSITIVE       2u   /* quantization keeps these at higher precision by default */
#define ATH_P_SHARED_EXPERT   4u

typedef struct {
    ath_tensor t;
    ath_role   role;
    int32_t    layer;      /* -1 for global tensors */
    int32_t    expert;     /* -1 if not an expert   */
    uint32_t   flags;
    const char *name;      /* original checkpoint name */
} ath_param;

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

typedef struct ath_family {
    const char  *name;
    int          (*match)      (const ath_json *config);
    ath_status   (*load)       (ath_model *m, const ath_json *config, ath_weights *w);
    ath_status   (*forward)    (ath_model *m, ath_batch *b, ath_state *s, const ath_hooks *h);
    ath_status   (*backward)   (ath_model *m, ath_batch *b, ath_grads *g);     /* NULL until phase 4 */
    ath_status   (*pretokenize)(const char *text, size_t n, ath_spans *out);
    ath_status   (*chat)       (const ath_msg *msgs, int32_t n, ath_buf *out);
} ath_family;

/* ath_model exposes: params[] and n_params, state_specs[] and n_states,
   n_layers, hidden size, vocab size, MoE info (experts per layer, top-k). */
```

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
│   ├── core/      base.h arena.c tensor.c file.c json.c safetensors.c threads.c rng.c
│   ├── backend/   backend.h compat.h cpu.c gpu.cu
│   ├── ops/       ops.h; per op a .c reference and a .cu kernel:
│   │              norm rope act embed matmul attn gdn conv moe sample quant
│   ├── text/      bpe.c unicode.h (generated, committed)
│   ├── model/     model.h registry.c
│   ├── models/
│   │   └── qwen35/  config.c weights.c forward.c text.c [backward.c] [fused.cu]
│   ├── infer/     engine.c state.c sched.c tier.c spec.c
│   ├── quant/     formats.c rtn.c gptq.c awq.c recipe.c container.c gguf.c eval.c
│   ├── ablit/     capture.c direction.c edit.c search.c eval.c
│   ├── train/     loop.c optim.c lora.c hotcold.c data.c planner.c ckpt.c
│   └── net/       http.c api.c
├── tools/         athanor.c (dispatch), dev/ (Python helpers, development only)
├── tests/         test_core.c test_ops.c test_model.c ... golden/dump.py
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
