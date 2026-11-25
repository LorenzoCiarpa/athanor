# src/models/qwen35

The Qwen hybrid family (Qwen3.5, Qwen3.6, Qwen3.8 dense and MoE). The only directory in athanor that knows this family exists.

| File | Content | Step |
| --- | --- | --- |
| `qwen35.h` | `ath_qwen35_config`; `ath_qwen35_layer` (the params of a layer by what the forward pass does with them, and its state indices) and `ath_qwen35_model` (the family data: config, layer table, embedding, final norm, output head); the functions shared by the files below; private to this directory (and the family's tests) | 2, 4 |
| `family.c` | `ath_family_qwen35`: `load` = parse the config, fill dimensions and the layer schedule, bind, declare the state specs (KV per attention layer, conv and recurrent per Gated DeltaNet layer); `forward` | 2, 4 |
| `config.c` | `config.json` → dimensions, layer schedule (Gated DeltaNet / full attention), MLP or MoE parameters; `match` on `model_type`; rejects keys the reference does not support | 2 |
| `weights.c` | checkpoint tensor names → `ath_param` with role, layer, expert, flags; dtype and shape checked against the config; fused experts become one view per expert; vision encoder and MTP head are ignored groups, their tensors bound last as params flagged `ATH_P_UNUSED`; every param recorded in the layer table for the forward pass | 2, 4 |
| `forward.c` | the forward pass from the ops' entry points (`src/ops/dispatch.c`), one composition for both devices: on the CPU the reference ops with `ath_matmul_fast` (the same bits as `ath_matmul_ref`, on the caller's pool), on a GPU the kernels of step 7, with the params' device copies and device scratch memory (step 8): embedding, per layer the Gated DeltaNet or attention mixer and the MLP with their norms and residual additions, final norm, fp32 logits; one path for prefill, pieces and decode; KV, conv and recurrent state in the buffers of the state specs; activations in the state's dtype; calls `on_residual` at every layer boundary. MoE layers fail with `ATH_ERR_UNSUPPORTED` until step 12 | 4, 8 |
| `text.c` | pre-tokenization (the `tokenizer.json` regex by hand, `ath_qwen35_pretokenize_regex`), chat template with tool definitions and thinking switch (`chat_template.jinja` by hand), incremental output parser (thinking blocks, tool calls with arguments typed by the tool's schema) | 5 |
| `backward.c` | per-block backward | T1 |
| `fused.cu` | optional fused kernels, admitted only with a benchmark | 9+ |

The exact markup of the chat template (roles, tool definitions, tool calls and results, thinking blocks) must be taken from the reference chat template shipped with the checkpoint (`tokenizer_config.json` or `chat_template.jinja`) and recorded in `docs/contract/`, never guessed; the rendered prompts are compared byte for byte with the reference's (`tests/test_text.c`).

## Forward pass

`forward.c` follows the contract of `docs/contract/qwen35-0.8b.md` op by op. Activations are stored in the state's dtype: bf16 rounds each op's output once, f32 keeps everything in fp32. The values the reference keeps in fp32 whatever the activation dtype stay fp32: the L2-normalized q and k, the decay `g`, the recurrent state and the logits. Strided slices (query and gate interleaved per head in `q_proj`, q/k/v inside the convolution output) are copied out into contiguous tensors, because the ops take contiguous tensors only; the copies are exact. Values go straight into the KV cache from `v_proj`, keys after QK-norm and RoPE. Working tensors come from the scratch arena and are released after every layer: one layer's worth for the tokens of the call. On a GPU (step 8) they come from the device scratch arena of the `ath_exec`; a weight is read through its param's index into the device copies; the strided slices are one `ath_copy_2d` each; token ids and positions are copied to the device at the start and the logits back to the host at the end, which is the pass's only wait.

| State buffer | Layout | Qwen3.5-0.8B, bf16 |
| --- | --- | --- |
| KV | `[capacity, n_kv_heads, head_dim]` keys, then the same for values | 2 KiB per token per attention layer |
| conv | `[K − 1, conv_dim]`, the last K − 1 inputs of the convolution | 36 KiB per Gated DeltaNet layer |
| recurrent | `[n_v_heads, k_dim, v_dim]` fp32 | 1 MiB per Gated DeltaNet layer |

## Layout reminder

Three Gated DeltaNet layers for every full-attention layer; SwiGLU MLP or MoE (routed plus shared experts) after each mixing block; MTP head for speculative decoding. Exact op math, dtypes and **verify** items: `docs/MODELS.md` and `docs/contract/`.

## Supported checkpoints

Text path of Qwen3.5 dense (0.8B, 2B, 4B, 9B, 27B) and MoE (35B-A3B, 122B-A10B), Qwen3.6 (27B, 35B-A3B), Qwen3.8-27B. Qwen3.8-Flash-Next needs additional ops and is a separate milestone. The vision encoder is out of scope for now.

## Checkpoint layouts

| `model_type` | Keys | Tensor prefix | Notes |
| --- | --- | --- | --- |
| `qwen3_5`, `qwen3_5_moe` | under `text_config` | `model.language_model.` | vision-language checkpoints as published; `model.visual.` and `mtp.` are ignored groups (params flagged `ATH_P_UNUSED`) |
| `qwen3_5_text`, `qwen3_5_moe_text` | top level | `model.` | text-only checkpoints saved by `Qwen3_5ForCausalLM` |

`lm_head.weight` is a top-level tensor in both layouts and exists only when `tie_word_embeddings` is false. MoE checkpoints store each layer's routed experts fused (`mlp.experts.gate_up_proj` [E, 2·I, H], `mlp.experts.down_proj` [E, H, I]); per-expert tensors (as in the MTP head) are not accepted for the language model. Roles and flags of every tensor: `docs/contract/qwen35-0.8b.md` and `docs/contract/qwen35-35b-a3b.md`, section Parameter binding.
