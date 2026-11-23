# src/models/qwen35

The Qwen hybrid family (Qwen3.5, Qwen3.6, Qwen3.8 dense and MoE). The only directory in athanor that knows this family exists.

| File | Content | Step |
| --- | --- | --- |
| `config.c` | `config.json` → dimensions, layer schedule (Gated DeltaNet / full attention / MLP / MoE), MTP presence | 2 |
| `weights.c` | checkpoint tensor names → `ath_param` with role, layer, expert, flags; fails on unbound or missing tensors | 2 |
| `forward.c` | per-block composition of ops for prefill and decode; calls hooks at residual boundaries and before matmuls | 4 |
| `text.c` | Qwen pre-tokenization split and chat template | 5 |
| `backward.c` | per-block backward | T1 |
| `fused.cu` | optional fused kernels, admitted only with a benchmark | 9+ |

## Layout reminder

Three Gated DeltaNet layers for every full-attention layer; SwiGLU MLP or MoE (routed plus shared experts) after each mixing block; MTP head for speculative decoding. Exact op math, dtypes and **verify** items: `docs/MODELS.md` and `docs/contract/`.

## Supported checkpoints

Text path of Qwen3.5 dense (0.8B, 2B, 4B, 9B, 27B) and MoE (35B-A3B, 122B-A10B), Qwen3.6 (27B, 35B-A3B), Qwen3.8-27B. Qwen3.8-Flash-Next needs additional ops and is a separate milestone. The vision encoder is out of scope for now.
