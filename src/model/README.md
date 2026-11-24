# src/model

The contract between pipelines and model families, and the static registry. Introduced in step 2; the forward pass and the per-sequence state joined in step 4.

| File | Content |
| --- | --- |
| `model.h` | `ath_role` and `ath_role_name`, param flags, `ath_param`, the layer schedule `ath_layer`, ignored groups `ath_ignored`, state kinds and specs (`ath_state_spec`), `ath_model`, `ath_state`, `ath_batch`, `ath_hooks`, `ath_family` (`match`, `load`, `forward`); the functions of `registry.c` and `state.c` |
| `registry.c` | the compiled-in families (from the Makefile `FAMILIES` list through the generated `families.h`); `ath_model_open(dir)` reads `config.json`, picks the family whose `match` accepts it, maps the checkpoint, calls `load`, then runs the coverage check |
| `state.c` | `ath_state_bytes`, `ath_state_init`, `ath_state_reset`: the state of one sequence from the model's specs, in host memory; `ath_model_forward`: argument checks, the family's `forward`, scratch memory released, `n_past` advanced |

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

float *out = ...;                                                /* [1, vocab_size] */
ath_tensor logits;
ATH_CHECK(ath_tensor_init(&logits, ATH_DTYPE_F32, 2, (int64_t[]){1, m.vocab_size}, out, ATH_TIER_RAM));
ATH_CHECK(ath_model_forward(&m, &(ath_batch){prompt, n_prompt, &logits}, &s, NULL, &scratch));   /* prefill */
ATH_CHECK(ath_model_forward(&m, &(ath_batch){&next, 1, &logits}, &s, NULL, &scratch));          /* decode */
ath_state_reset(&s, &m);                                         /* a new sequence */
```

The logits tensor has one row per token or one row for the last token. `n_past` counts the tokens the state has seen; a batch that does not fit in `capacity` fails with `ATH_ERR_RANGE` and leaves the state as it was. Any other failure inside the family leaves the state undefined until `ath_state_reset`. Every working tensor of a call comes from the scratch arena and is released before it returns, so the arena's blocks are reused from call to call.

## State specs

`m.states` lists what one sequence needs, in layer order: `ATH_STATE_KV` (keys and values per token: `per_token` elements), `ATH_STATE_CONV` (the last inputs of a short convolution: `fixed` elements), `ATH_STATE_RECURRENT` (a linear-attention state: `fixed` elements). KV and conv state hold activations and take the state's activation dtype; the recurrent state is always fp32. `ath_state_bytes` sums them, each buffer rounded up to 64 bytes. For Qwen3.5-0.8B in bf16: 6 KV, 18 conv and 18 recurrent buffers, 18.6 MiB fixed plus 12 KiB per token.

`on_residual` in `ath_hooks` sees the residual stream entering every layer (`layer` 0 … n_layers − 1) and the final norm (`layer` = n_layers), as `[n_tokens, hidden_size]` in the activation dtype.

## The coverage rule

After the family's `load`, the registry checks that every checkpoint tensor is either viewed by at least one param or matched by exactly one ignored group (never both), and that every param is a contiguous view inside the tensor it names. Anything else fails with `ATH_ERR_FORMAT` and names the tensor. Families only have to look for the tensors they need; the registry catches the ones they did not expect.

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
