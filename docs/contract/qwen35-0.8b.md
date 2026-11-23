# Contract: Qwen3.5-0.8B

Op contract for the first athanor model. Step 1 records the checkpoint and its tensors exactly as `athanor inspect` reports them. The op contract proper (dtype boundaries, normalization, RoPE, gating, state layout, the resolution of every **verify** item in `docs/MODELS.md`) is written in step 2 from the pinned reference implementation.

## Checkpoint

| Item | Value |
| --- | --- |
| Repository | [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) |
| Revision | `2fc06364715b967f1860aea9cf38778875588b17` |
| Downloaded | 2026-11-23 with `hf download Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B` (huggingface_hub 2.4.0) |
| Layout | one shard plus `model.safetensors.index.json` |
| Reference implementation | not pinned yet (step 2); `config.json` was written by transformers `4.57.0.dev0` |

| File | sha256 |
| --- | --- |
| `config.json` | `b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204` |
| `model.safetensors.index.json` | `d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e` |
| `model.safetensors-00001-of-00001.safetensors` | `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696` |
| `tokenizer.json` | `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42` |
| `tokenizer_config.json` | `49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c` |

## What `inspect` reports

```
config   Qwen3_5ForConditionalGeneration, model_type qwen3_5 (text: qwen3_5_text)
         layers 24, hidden size 1024, vocab 248320
shards   model.safetensors-00001-of-00001.safetensors: header 59840 B, data 1.63 GiB

dtype     tensors          params           bytes
f32            36            2592           10368  10.12 KiB
bf16          452       873436192      1746872384  1.63 GiB
total         488       873438784      1746882752  1.63 GiB

check    tensor bytes 1746882752 (1.63 GiB) == data sections 1746882752 (1.63 GiB)
         index metadata.total_size 1746882752 == tensor bytes
```

The header is 59,840 bytes (5 bytes of space padding), the data section starts at byte 59,848, and every tensor is aligned to its element size. The shard has no `__metadata__`.

| Group | Prefix | Tensors | Bytes | In scope |
| --- | --- | --- | --- | --- |
| Language model | `model.language_model.` | 320 | 1,504,791,232 | yes (752,393,024 parameters) |
| MTP head | `mtp.` | 15 | 40,905,728 | step 13 |
| Vision encoder | `model.visual.` | 153 | 201,185,792 | later milestone |

There is no `lm_head` tensor: `tie_word_embeddings` is `true`, so the output projection reuses `embed_tokens.weight`.

## Tensors by layer type

Layer schedule from the tensor names: layers 3, 7, 11, 15, 19 and 23 have `self_attn.*` (full attention); the other 18 have `linear_attn.*` (Gated DeltaNet). This matches `text_config.layer_types` and `full_attention_interval: 4` exactly. All layers of the same type have identical tensor names, dtypes and shapes.

### Global (`model.language_model.`)

| Tensor | dtype | shape | bytes |
| --- | --- | --- | --- |
| `embed_tokens.weight` | bf16 | [248320, 1024] | 508,559,360 |
| `norm.weight` | bf16 | [1024] | 2,048 |

### Gated DeltaNet layer (`model.language_model.layers.N.`, 18 layers)

14 tensors, 43,111,008 bytes per layer.

| Tensor | dtype | shape | bytes |
| --- | --- | --- | --- |
| `input_layernorm.weight` | bf16 | [1024] | 2,048 |
| `linear_attn.A_log` | f32 | [16] | 64 |
| `linear_attn.conv1d.weight` | bf16 | [6144, 1, 4] | 49,152 |
| `linear_attn.dt_bias` | bf16 | [16] | 32 |
| `linear_attn.in_proj_a.weight` | bf16 | [16, 1024] | 32,768 |
| `linear_attn.in_proj_b.weight` | bf16 | [16, 1024] | 32,768 |
| `linear_attn.in_proj_qkv.weight` | bf16 | [6144, 1024] | 12,582,912 |
| `linear_attn.in_proj_z.weight` | bf16 | [2048, 1024] | 4,194,304 |
| `linear_attn.norm.weight` | f32 | [128] | 512 |
| `linear_attn.out_proj.weight` | bf16 | [1024, 2048] | 4,194,304 |
| `mlp.down_proj.weight` | bf16 | [1024, 3584] | 7,340,032 |
| `mlp.gate_proj.weight` | bf16 | [3584, 1024] | 7,340,032 |
| `mlp.up_proj.weight` | bf16 | [3584, 1024] | 7,340,032 |
| `post_attention_layernorm.weight` | bf16 | [1024] | 2,048 |

The 36 f32 tensors of the checkpoint are exactly `A_log` and `linear_attn.norm.weight` of these 18 layers.

### Full-attention layer (`model.language_model.layers.N.`, 6 layers)

11 tensors, 36,705,280 bytes per layer.

| Tensor | dtype | shape | bytes |
| --- | --- | --- | --- |
| `input_layernorm.weight` | bf16 | [1024] | 2,048 |
| `mlp.down_proj.weight` | bf16 | [1024, 3584] | 7,340,032 |
| `mlp.gate_proj.weight` | bf16 | [3584, 1024] | 7,340,032 |
| `mlp.up_proj.weight` | bf16 | [3584, 1024] | 7,340,032 |
| `post_attention_layernorm.weight` | bf16 | [1024] | 2,048 |
| `self_attn.k_norm.weight` | bf16 | [256] | 512 |
| `self_attn.k_proj.weight` | bf16 | [512, 1024] | 1,048,576 |
| `self_attn.o_proj.weight` | bf16 | [1024, 2048] | 4,194,304 |
| `self_attn.q_norm.weight` | bf16 | [256] | 512 |
| `self_attn.q_proj.weight` | bf16 | [4096, 1024] | 8,388,608 |
| `self_attn.v_proj.weight` | bf16 | [512, 1024] | 1,048,576 |

### MTP head (`mtp.`)

| Tensor | dtype | shape | bytes |
| --- | --- | --- | --- |
| `fc.weight` | bf16 | [1024, 2048] | 4,194,304 |
| `pre_fc_norm_embedding.weight` | bf16 | [1024] | 2,048 |
| `pre_fc_norm_hidden.weight` | bf16 | [1024] | 2,048 |
| `layers.0.*` | bf16 | same 11 tensors and shapes as a full-attention layer | 36,705,280 |
| `norm.weight` | bf16 | [1024] | 2,048 |

`mtp_num_hidden_layers` is 1 and `mtp_use_dedicated_embeddings` is `false`.

### Vision encoder (`model.visual.`, out of scope)

12 blocks (`blocks.0` to `blocks.11`), each with `attn.qkv` [2304, 768], `attn.proj` [768, 768], `mlp.linear_fc1` [3072, 768], `mlp.linear_fc2` [768, 3072], `norm1`, `norm2` (all with bias); plus `patch_embed.proj` [768, 3, 2, 16, 16], `pos_embed` [2304, 768] and `merger.{norm, linear_fc1 [3072, 3072], linear_fc2 [1024, 3072]}`. All bf16.

## Shape arithmetic against `config.json`

Facts that tie each shape to config fields. They constrain the contract but are not the contract: the layout inside each tensor (order of q, k, v; interleaving of gates; which head count indexes `A_log`) is confirmed against the reference in step 2.

| Shape | Config fields |
| --- | --- |
| `in_proj_qkv` rows 6144 | 16 key heads × 128 + 16 key heads × 128 + 16 value heads × 128 |
| `in_proj_z` rows 2048, `out_proj` cols 2048 | 16 value heads × 128 (`linear_value_head_dim`) |
| `in_proj_a`, `in_proj_b` rows, `A_log`, `dt_bias` | 16: `linear_num_value_heads` and `linear_num_key_heads` are both 16 here, so this checkpoint cannot tell which one applies (**verify**) |
| `conv1d.weight` [6144, 1, 4] | depthwise over the 6144 q/k/v channels, kernel `linear_conv_kernel_dim` 4 |
| `linear_attn.norm.weight` [128] | `linear_value_head_dim` |
| `q_proj` rows 4096 | 2 × 8 heads × 256 (`head_dim`): twice the query width, consistent with `attn_output_gate: true` (**verify** layout) |
| `k_proj`, `v_proj` rows 512 | 2 KV heads × 256 |
| `o_proj` cols 2048 | 8 heads × 256 |
| `q_norm`, `k_norm` [256] | `head_dim`: per-head norms |
| `mlp.*` 3584 | `intermediate_size` |

## Observations for step 2 (evidence, not contract)

Weight statistics measured on this checkpoint, to be checked against the reference implementation when the RMSNorm parameterization **verify** item is resolved:

| Tensor | mean | min | max |
| --- | --- | --- | --- |
| `layers.0.input_layernorm.weight` | +0.24 | −0.11 | +1.00 |
| `layers.0.post_attention_layernorm.weight` | +0.09 | −0.36 | +0.28 |
| `norm.weight` | +3.31 | −0.78 | +5.31 |
| `layers.3.self_attn.q_norm.weight` | +0.43 | −0.21 | +1.23 |
| `layers.3.self_attn.k_norm.weight` | +0.42 | −0.90 | +1.63 |
| `layers.0.linear_attn.norm.weight` (f32) | +0.95 | +0.52 | +1.06 |

The decoder-layer norms and `q_norm`/`k_norm` center near zero with negative entries, as a zero-centered `(1 + w)` parameterization would; the Gated DeltaNet output norm centers near one, as a plain `w` would. Other config values the contract must cover: `rms_norm_eps` 1e-6, `rope_theta` 10,000,000, `partial_rotary_factor` 0.25, `mrope_section` [11, 11, 10] with `mrope_interleaved: true`, `mamba_ssm_dtype` float32.
