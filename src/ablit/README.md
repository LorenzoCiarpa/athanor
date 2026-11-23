# src/ablit

Abliteration: estimating the residual-stream direction that mediates refusals and projecting it out of the weights that write to the residual stream (Arditi et al., 2024), with automatic parameter search. Phase 3.

| File | Content |
| --- | --- |
| `capture.c` | runs two prompt sets through the engine and records the residual stream via `on_residual`, streamed to disk |
| `direction.c` | difference of means per layer; projected variant; multiple directions via SVD; layer selection |
| `edit.c` | orthogonalizes every param flagged `ATH_P_WRITES_RESIDUAL` (attention and Gated DeltaNet outputs, MLP and every expert's down projection), optional norm preservation, re-quantization through `src/quant` |
| `search.c` | multi-objective search over layer range, per-layer strength and direction index, minimizing refusal rate and KL on harmless prompts; outputs the Pareto front |
| `eval.c` | refusal rate, KL, capability suite with a math subset (the capability most often damaged) |

## Generic by construction

The module never names a block type. It captures through hooks and edits by flag, so a new family is supported as soon as its `weights.c` flags the right params.

## Responsible use

Abliteration removes a safety behaviour. Edited models remain subject to the original model license and to the law of the user's jurisdiction. Export writes a model card section stating that the model was abliterated and with which parameters. The repository ships only prompt sets from published research.
