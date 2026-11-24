# Contract: Qwen3.5-35B-A3B

The MoE sibling of [Qwen3.5-0.8B](qwen35-0.8b.md), recorded in step 2 so that the family's MoE parsing and binding rest on a real checkpoint rather than on guesses. Only `config.json`, the index and the 14 shard headers were downloaded (HTTP range requests, about 220 KB); the weights arrive with step 12. Everything not listed here (norms, RoPE, attention, Gated DeltaNet, residual stream) follows the 0.8B contract with this model's dimensions.

## Checkpoint

| Item | Value |
| --- | --- |
| Repository | [`Qwen/Qwen3.5-35B-A3B`](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) |
| Revision | `2508b2437e382c605eb0fc52b83f5545960f6b4a` |
| Read | 2026-11-24: `config.json` and `model.safetensors.index.json` with `hf download`, shard headers with HTTP range requests |
| Layout | 14 shards, 1,811 tensors, `metadata.total_size` 71,903,655,008 bytes |
| Reference implementation | transformers `5.24.0`, `models/qwen3_5_moe/modeling_qwen3_5_moe.py` sha256 `aa3caec05ddf172964b6cfe9b19d333dde892ee99b75bc629c4c18169860f5d6`, `configuration_qwen3_5_moe.py` sha256 `0ca59bbae621185f4846e5413ad11bab129c807cf5cdd47ab6070bec920d7327` |

| File | sha256 |
| --- | --- |
| `config.json` | `5e4d7f74fec2f360eb9cfbfcd6ec0c4c76e684d3a11caaed259d9fd9bfbc7944` |
| `model.safetensors.index.json` | `d8d0b7ca4e61ae107e3e87a3ff21136b3ac7c789e64bb24267227ca804e04205` |

Config (`text_config`, model_type `qwen3_5_moe_text`): 40 layers (30 Gated DeltaNet, 10 full attention, interval 4), `hidden_size` 2048, 16 attention heads, 2 KV heads, `head_dim` 256, Gated DeltaNet with 16 key heads and **32 value heads** of 128, `num_experts` 256, `num_experts_per_tok` 8, `moe_intermediate_size` 512, `shared_expert_intermediate_size` 512, `mlp_only_layers` [], vocabulary 248,320. `tie_word_embeddings` is false at the top level.

## Tensors that differ from the dense model

Language-model tensors are under `model.language_model.`, except `lm_head.weight` at the top level. All bf16 except `A_log` and `linear_attn.norm.weight` (f32), as in the dense model.

| Tensor | shape | Note |
| --- | --- | --- |
| `lm_head.weight` | [248320, 2048] | untied output projection |
| `layers.N.mlp.gate.weight` | [256, 2048] | router |
| `layers.N.mlp.experts.gate_up_proj` | [256, 1024, 2048] | all routed experts in one tensor, no `.weight` suffix |
| `layers.N.mlp.experts.down_proj` | [256, 2048, 512] | same |
| `layers.N.mlp.shared_expert.{gate,up}_proj.weight` | [512, 2048] | |
| `layers.N.mlp.shared_expert.down_proj.weight` | [2048, 512] | |
| `layers.N.mlp.shared_expert_gate.weight` | [1, 2048] | |
| `layers.N.linear_attn.A_log`, `dt_bias` | [32] | indexed by value head (32), not key head (16) |
| `layers.N.linear_attn.in_proj_a.weight`, `in_proj_b.weight` | [32, 2048] | same |
| `layers.N.linear_attn.in_proj_qkv.weight` | [8192, 2048] | 2 × 16 × 128 + 32 × 128 |

Every layer is MoE. The MTP head (`mtp.`, 785 tensors) stores its experts one tensor per expert (`mtp.layers.0.mlp.experts.E.gate_proj.weight` [512, 2048]); it is ignored until step 13. The vision encoder is under `model.visual.`.

## MoE op contract

`Qwen3_5MoeSparseMoeBlock`, `Qwen3_5MoeTopKRouter`, `Qwen3_5MoeExperts` (lines 837–934 of the MoE modeling file). Resolves the router and shared-expert **verify** items of `docs/MODELS.md`.

| Step | Contract |
| --- | --- |
| Router | `logits = x · gateᵀ` [T, 256]; `p = softmax(logits)` in fp32 over all experts; the top `num_experts_per_tok` (8) by `p`; their weights are renormalized to sum to 1 (always: the reference has no `norm_topk_prob` switch) and cast to the activation dtype. |
| Routed expert `e` | `gate_up_proj[e]` is [2·I, H]: rows `[0, I)` are the gate projection and rows `[I, 2I)` the up projection (`chunk(2)` on the output). `out_e = down_proj[e] · (silu(gate) · up)`, `down_proj[e]` [H, I]. |
| Combination | `Σ_k w_k · out_{e_k}` over the selected experts, accumulated in the activation dtype. |
| Shared expert | SwiGLU MLP of width `shared_expert_intermediate_size`, always active, scaled per token by `sigmoid(x · shared_expert_gateᵀ)` (a scalar), then added to the routed sum. |
| Layer output | the MoE block replaces the MLP in `y = h + moe(rmsnorm(h; post_attention_layernorm))`. |
| Config | `mlp_only_layers` exists in the config but the reference ignores it (every layer builds the MoE block); athanor rejects a non-empty list. `router_aux_loss_coef` matters for training only. |

## Parameter binding

In addition to the dense roles ([0.8B contract](qwen35-0.8b.md#parameter-binding)):

| Tensor | Role | Expert | Flags |
| --- | --- | --- | --- |
| `lm_head.weight` | lm_head | −1 | sensitive |
| `embed_tokens.weight` | embed | −1 | residual (not sensitive: untied) |
| `mlp.gate.weight` | router | −1 | sensitive |
| `mlp.experts.gate_up_proj` | expert_in, one param per expert: a [2·I, H] view of slice `e` | 0 … 255 | |
| `mlp.experts.down_proj` | expert_out, one param per expert: a [H, I] view of slice `e` | 0 … 255 | residual |
| `mlp.shared_expert.gate_proj.weight`, `up_proj.weight` | mlp_in | −1 | shared |
| `mlp.shared_expert.down_proj.weight` | mlp_out | −1 | shared, residual |
| `mlp.shared_expert_gate.weight` | gate | −1 | shared, sensitive |

Per-expert views share the checkpoint tensor's memory and name; nothing is copied. Placement (step 12) moves experts one by one through these params.
