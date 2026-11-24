# src/model

The contract between pipelines and model families, and the static registry. Introduced in step 2.

| File | Content |
| --- | --- |
| `model.h` | `ath_role` and `ath_role_name`, param flags, `ath_param`, the layer schedule `ath_layer`, ignored groups `ath_ignored`, `ath_model`, `ath_family`; `ath_model_open` / `ath_model_close` |
| `registry.c` | the compiled-in families (from the Makefile `FAMILIES` list through the generated `families.h`); `ath_model_open(dir)` reads `config.json`, picks the family whose `match` accepts it, maps the checkpoint, calls `load`, then runs the coverage check |

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

The labels in `ath_layer` (`mixer`, `ffn`) are for people. Pipelines never branch on them; per-layer state arrives as state specs in step 4.
