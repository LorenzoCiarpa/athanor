# src/model

The contract between pipelines and model families, and the static registry. Introduced in step 2.

| File | Content |
| --- | --- |
| `model.h` | `ath_role`, param flags, `ath_param`, `ath_state_spec`, `ath_hooks`, `ath_family`, `ath_model` |
| `registry.c` | static table of compiled-in families; `ath_model_open(dir)` reads `config.json` and picks the family whose `match` accepts it |

The full definition and the reasoning behind it are in `docs/ARCHITECTURE.md`. Any change to `model.h` must be reflected there and must keep every existing family compiling.

## Why roles and flags

They let quantization, abliteration, training and MoE placement work on any family without knowing its internals: formats are chosen by role, abliteration edits every param flagged `ATH_P_WRITES_RESIDUAL`, LoRA targets are chosen by role, experts are placed by `layer` and `expert`.
