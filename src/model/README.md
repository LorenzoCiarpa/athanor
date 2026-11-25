# src/model

The contract between pipelines and model families, and the static registry. Not to be confused with `src/models/` (the families that implement it) or `docs/contract/` (the op contract of each checkpoint, a document): `docs/ARCHITECTURE.md`, Three names. Introduced in step 2; the forward pass and the per-sequence state joined in step 4, text (tokenizer, chat template, output parser) in step 5, the GPU (`ath_exec`, states on a device) in step 8.

| File | Content |
| --- | --- |
| `model.h` | `ath_role` and `ath_role_name`, param flags, `ath_param`, the layer schedule `ath_layer`, ignored groups `ath_ignored` (their tensors are params flagged `ATH_P_UNUSED`), state kinds and specs (`ath_state_spec`), `ath_model`, `ath_state`, `ath_exec`, `ath_batch`, `ath_hooks`, the text types (`ath_tool_def`, `ath_tool_call`, `ath_msg`, `ATH_CHAT_*`, `ath_out_event`, `ath_parse_state`), `ath_family` (`match`, `load`, `forward`, `pretokenize_regex`, `pretokenize`, `chat`, `parse`); the functions of `registry.c`, `state.c` and `text.c` |
| `registry.c` | the compiled-in families (from the Makefile `FAMILIES` list through the generated `families.h`); `ath_model_open(dir)` reads `config.json`, picks the family whose `match` accepts it, maps the checkpoint, calls `load`, then runs the coverage check |
| `state.c` | `ath_state_bytes`, `ath_state_init`, `ath_state_init_gpu` (step 8), `ath_state_reset`: the state of one sequence from the model's specs, in host memory or on a device; `ath_model_forward`: argument checks (also that the pass runs where the state is), the family's `forward` with the caller's `ath_exec`, scratch memory released (host and device), `n_past` advanced; `ath_state_copy` (step 5; on a device, step 8) |
| `text.c` | `ath_model_tokenizer` (the checkpoint's `tokenizer.json` with the family's pre-tokenization), `ath_model_chat`, `ath_model_parse`: argument checks, then the family's function (step 5) |

The full definition and the reasoning behind it are in `docs/ARCHITECTURE.md`. Any change to `model.h` must be reflected there and must keep every existing family compiling.

## Loading a model

```c
ath_arena a;
ath_arena_init(&a, 0);
ath_model m;
ATH_CHECK(ath_model_open(&m, &a, "models/Qwen3.5-0.8B"));
for (uint32_t i = 0; i < m.n_params; i++) {
    const ath_param *p = &m.params[i];
    if (p->flags & ATH_P_WRITES_RESIDUAL)
        ;   /* p->t views the checkpoint mapping: role, layer and expert say what it is */
}
ath_model_close(&m);
ath_arena_free(&a);
```

`athanor inspect <dir> --roles` prints the same information.

## Running a model

```c
ath_state s;
ATH_CHECK(ath_state_init(&s, &m, ATH_DTYPE_BF16, 4096, &a));   /* up to 4096 tokens; f32 for an exact run */
ath_arena scratch;
ath_arena_init(&scratch, 0);
ath_pool pool;                                                   /* threads for the matmuls */
ATH_CHECK(ath_pool_init(&pool, ath_pool_default_threads(), &a));

float *out = ...;                                                /* [1, vocab_size] */
ath_tensor logits;
ATH_CHECK(ath_tensor_init(&logits, ATH_DTYPE_F32, 2, (int64_t[]){1, m.vocab_size}, out, ATH_TIER_RAM));
const ath_exec cpu = {.pool = &pool};
ATH_CHECK(ath_model_forward(&m, &(ath_batch){.tokens = prompt, .n_tokens = n_prompt, .logits = &logits}, &s, NULL, &scratch, &cpu));
ATH_CHECK(ath_model_forward(&m, &(ath_batch){.tokens = &next, .n_tokens = 1, .logits = &logits}, &s, NULL, &scratch, &cpu));
ATH_CHECK(ath_state_reset(&s, &m));                              /* a new sequence */
ath_pool_free(&pool);
```

On a GPU (step 8) the engine (`src/infer/engine.h`) holds the device, the params' copies on it and the `ath_exec` for them; the state is allocated on the same device, and the calls are the same:

```c
ath_engine e;
ATH_CHECK(ath_engine_open(&e, &m, 0, &a));                       /* device 0, the weights copied once */
ATH_CHECK(ath_engine_state(&e, &s, ATH_DTYPE_BF16, 4096, &a));
ATH_CHECK(ath_model_forward(&m, &(ath_batch){.tokens = prompt, .n_tokens = n_prompt, .logits = &logits}, &s, NULL, &scratch, &e.exec));
ath_engine_close(&e);
```

The logits tensor has one row per token or one row for the last token. `n_past` counts the tokens the state has seen; a batch that does not fit in `capacity` fails with `ATH_ERR_RANGE` and leaves the state as it was. Any other failure inside the family leaves the state undefined until `ath_state_reset`. Every working tensor of a call comes from the scratch arena and is released before it returns, so the arena's blocks are reused from call to call. The pool lends its threads to the CPU work. A NULL pool (or a NULL exec) runs everything on the calling thread, and the logits and the state are the same bits with any number of threads. On a GPU the logits are in host memory when the call returns; a batch without logits returns as soon as its work is queued on the device, and later calls on the same state come after it.

### Device batches (step 9)

A batch with `flags = ATH_BATCH_DEVICE` has its tokens, its positions (`pos`, `n_past + i` when the work runs) and its logits in the memory of the device the pass runs on. The pass then copies nothing between host and device and does not wait, so the call can be recorded in a graph and replayed: the decoder of `src/infer/` does that, one launch per token. `ath_model_forward` cannot read the ids or the positions, so it does not check them; positions are refused on any other batch. Batch initializers are designated (`.tokens = ...`), since the struct gained members.

## Text

```c
ath_tokenizer tok;
ATH_CHECK(ath_model_tokenizer(&m, dir, &tok, &a));          /* tokenizer.json + the family's split */
ath_msg msgs[] = {{.role = ATH_MSG_USER, .content = "Why is the sky blue?"}};
ath_buf prompt;
ath_buf_init(&prompt, &a);
ATH_CHECK(ath_model_chat(&m, msgs, 1, NULL, 0, ATH_CHAT_GENERATE, &prompt));   /* the template, byte for byte */
ath_tokens ids;
ath_tokens_init(&ids, &a);
ATH_CHECK(ath_tokenizer_encode(&tok, prompt.data, prompt.len, &ids, &scratch));
/* ... ath_model_forward on ids, then per token: sample, decode, parse ... */
ath_parse_state ps = {.opts = ATH_CHAT_GENERATE, .a = &a};
ath_out_event ev[64];
int32_t n_ev;
ATH_CHECK(ath_model_parse(&m, &ps, piece, piece_len, ev, 64, &n_ev));   /* text, thinking, tool calls */
ATH_CHECK(ath_model_parse(&m, &ps, NULL, 0, ev, 64, &n_ev));            /* end of the output */
```

Tools are `ath_tool_def`s (name, description, the parameters schema as raw JSON); an assistant message carries the calls it made (`ath_tool_call`, arguments as a raw JSON object), and a tool's result comes back as a message with role `ATH_MSG_TOOL`. athanor renders and parses tool calls; it never runs one (AGENTS.md, invariant 10). `tools/chat.c` is a complete user of this section.

`ath_state_copy` copies the state of one sequence into another of the same capacity, activation dtype and device. A recurrent state cannot be rolled back, so a pipeline that will come back to a point of a conversation keeps a copy there.

## State specs

`m.states` lists what one sequence needs, in layer order: `ATH_STATE_KV` (keys and values per token: `per_token` elements), `ATH_STATE_CONV` (the last inputs of a short convolution: `fixed` elements), `ATH_STATE_RECURRENT` (a linear-attention state: `fixed` elements). KV and conv state hold activations and take the state's activation dtype; the recurrent state is always fp32. `ath_state_bytes` sums them, each buffer rounded up to 64 bytes. For Qwen3.5-0.8B in bf16: 6 KV, 18 conv and 18 recurrent buffers, 18.6 MiB fixed plus 12 KiB per token.

`on_residual` in `ath_hooks` sees the residual stream entering every layer (`layer` 0 … n_layers − 1) and the final norm (`layer` = n_layers), as `[n_tokens, hidden_size]` in the activation dtype.

## The coverage rule

After the family's `load`, the registry checks that every checkpoint tensor is viewed by at least one param, and that every param is a contiguous view inside the tensor it names. Tensors the text path does not use (the vision encoder of a vision-language checkpoint, a head for a later step) are bound too, as params flagged `ATH_P_UNUSED` (role other, layer −1), and belong to an ignored group the family declares with a prefix and a reason. The registry checks that each group and the unused params coincide: a tensor of a group is viewed by unused params only and is in no other group, and every unused param views a tensor of some group. So no tensor is ignored silently: it is either used or bound as unused with a reason. Anything else fails with `ATH_ERR_FORMAT` and names the tensor. Families only have to look for the tensors they need; the registry catches the ones they did not expect. (Until 2026-11-24, CHANGE-001, the tensors of an ignored group were left unbound.)

## Why roles and flags

They let quantization, abliteration, training and MoE placement work on any family without knowing its internals: formats are chosen by role, abliteration edits every param flagged `ATH_P_WRITES_RESIDUAL`, LoRA targets are chosen by role, experts are placed by `layer` and `expert`.

| Role | Meaning |
| --- | --- |
| `embed`, `lm_head` | token embedding table; output projection when not tied (`tie_embeddings`) |
| `norm` | normalization weight (its parameterization is family math) |
| `mix_in`, `mix_out` | token mixer (attention, linear attention) matmuls from and back to the residual stream |
| `mlp_in`, `mlp_out` | dense MLP and shared-expert matmuls |
| `router`, `expert_in`, `expert_out` | MoE router and routed experts, one param per expert |
| `conv` | short causal convolution kernel |
| `gate` | small matmul producing per-head or per-token gates |
| `other` | everything else (decay constants, gate biases) |

The labels in `ath_layer` (`mixer`, `ffn`) are for people. Pipelines never branch on them; per-layer state comes as state specs (step 4).
