# src/models/qwen35

The Qwen hybrid family (Qwen3.5, Qwen3.6, Qwen3.8 dense and MoE). The only directory in athanor that knows this family exists.

| File | Content | Step |
| --- | --- | --- |
| `qwen35.h` | `ath_qwen35_config` and the functions shared by the files below; private to this directory (and the family's tests) | 2 |
| `family.c` | `ath_family_qwen35`: `load` = parse the config, fill dimensions and the layer schedule, bind | 2 |
| `config.c` | `config.json` → dimensions, layer schedule (Gated DeltaNet / full attention), MLP or MoE parameters; `match` on `model_type`; rejects keys the reference does not support | 2 |
| `weights.c` | checkpoint tensor names → `ath_param` with role, layer, expert, flags; dtype and shape checked against the config; fused experts become one view per expert; vision encoder and MTP head are ignored groups | 2 |
| `forward.c` | per-block composition of ops for prefill and decode; calls hooks at residual boundaries and before matmuls | 4 |
| `text.c` | Qwen pre-tokenization split and chat template | 5 |
| `backward.c` | per-block backward | T1 |
| `fused.cu` | optional fused kernels, admitted only with a benchmark | 9+ |

## Layout reminder

Three Gated DeltaNet layers for every full-attention layer; SwiGLU MLP or MoE (routed plus shared experts) after each mixing block; MTP head for speculative decoding. Exact op math, dtypes and **verify** items: `docs/MODELS.md` and `docs/contract/`.

## Supported checkpoints

Text path of Qwen3.5 dense (0.8B, 2B, 4B, 9B, 27B) and MoE (35B-A3B, 122B-A10B), Qwen3.6 (27B, 35B-A3B), Qwen3.8-27B. Qwen3.8-Flash-Next needs additional ops and is a separate milestone. The vision encoder is out of scope for now.

## Checkpoint layouts

| `model_type` | Keys | Tensor prefix | Notes |
| --- | --- | --- | --- |
| `qwen3_5`, `qwen3_5_moe` | under `text_config` | `model.language_model.` | vision-language checkpoints as published; `model.visual.` and `mtp.` are ignored groups |
| `qwen3_5_text`, `qwen3_5_moe_text` | top level | `model.` | text-only checkpoints saved by `Qwen3_5ForCausalLM` |

`lm_head.weight` is a top-level tensor in both layouts and exists only when `tie_word_embeddings` is false. MoE checkpoints store each layer's routed experts fused (`mlp.experts.gate_up_proj` [E, 2·I, H], `mlp.experts.down_proj` [E, H, I]); per-expert tensors (as in the MTP head) are not accepted for the language model. Roles and flags of every tensor: `docs/contract/qwen35-0.8b.md` and `docs/contract/qwen35-35b-a3b.md`, section Parameter binding.
