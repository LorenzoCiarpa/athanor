# Contract: Qwen3.5-0.8B

Op contract for the first athanor model. Step 1 recorded the checkpoint and its tensors exactly as `athanor inspect` reports them. Step 2 pinned the reference implementation and wrote the op contract from it ([Op contract](#op-contract)): dtype boundaries, normalization, RoPE, gating, state layout, and the resolution of the **verify** items of `docs/MODELS.md` that concern this model. [Parameter binding](#parameter-binding) records the role and flags of every tensor.

## Checkpoint

| Item | Value |
| --- | --- |
| Repository | [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) |
| Revision | `2fc06364715b967f1860aea9cf38778875588b17` |
| Downloaded | 2026-11-23 with `hf download Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B` (huggingface_hub 2.4.0) |
| Layout | one shard plus `model.safetensors.index.json` |
| Reference implementation | transformers `5.24.0` (pinned in step 2, see below); `config.json` was written by transformers `4.57.0.dev0` |

| File | sha256 |
| --- | --- |
| `config.json` | `b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204` |
| `model.safetensors.index.json` | `d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e` |
| `model.safetensors-00001-of-00001.safetensors` | `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696` |
| `tokenizer.json` | `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42` |
| `tokenizer_config.json` | `49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c` |

## Reference implementation

Hugging Face `transformers` **5.24.0**, the latest release on 2026-11-24, read from the PyPI wheel. athanor never imports or runs it; the golden harness does (`tests/golden/`, development only, with torch 2.15.0 CPU). The `qwen3_5` modeling file is the source of truth for every item below; line numbers refer to that file.

| Item | Value |
| --- | --- |
| Wheel | `transformers-5.24.0-py3-none-any.whl`, sha256 `ff5ebedf9359f94bb63772b6380eba6e7104d010cbbb2cb82c788817ea7486f6` |
| `models/qwen3_5/modeling_qwen3_5.py` | sha256 `c4d847b7e809ed2a49611e5afcea8bd0db39de597b67b57ce9753fb65f550de2` |
| `models/qwen3_5/configuration_qwen3_5.py` | sha256 `8124bf8c60552159339ea35d6f1e5ca46743473e83bd68a4311684c9e4f68178` |
| `conversion_mapping.py` | sha256 `6e4085f1c941c55b4d12cdddcbb6134a683fe3639af853988bcb1c026ac2d88c` |
| Classes | `Qwen3_5ForConditionalGeneration` (the checkpoint's architecture, names under `model.language_model.`); `Qwen3_5ForCausalLM` (text path; loads the same checkpoint through the `qwen3_5_text` prefix mapping and drops `mtp.*` and `model.visual.*`) |

To fetch the same files: `pip download transformers==5.24.0 --no-deps`, check the wheel hash, unzip `transformers/models/qwen3_5/*`.

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

`mtp_num_hidden_layers` is 1 and `mtp_use_dedicated_embeddings` is `false`: the head uses the main model's embedding and output head. Since step 13 it is athanor's draft head (op contract below).

### Vision encoder (`model.visual.`, out of scope)

12 blocks (`blocks.0` to `blocks.11`), each with `attn.qkv` [2304, 768], `attn.proj` [768, 768], `mlp.linear_fc1` [3072, 768], `mlp.linear_fc2` [768, 3072], `norm1`, `norm2` (all with bias); plus `patch_embed.proj` [768, 3, 2, 16, 16], `pos_embed` [2304, 768] and `merger.{norm, linear_fc1 [3072, 3072], linear_fc2 [1024, 3072]}`. All bf16.

## Shape arithmetic against `config.json`

Facts that tie each shape to config fields. The layout inside each tensor (order of q, k, v; interleaving of gates; which head count indexes `A_log`) was confirmed against the reference in step 2 and is part of the [Op contract](#op-contract).

| Shape | Config fields |
| --- | --- |
| `in_proj_qkv` rows 6144 | 16 key heads × 128 + 16 key heads × 128 + 16 value heads × 128 |
| `in_proj_z` rows 2048, `out_proj` cols 2048 | 16 value heads × 128 (`linear_value_head_dim`) |
| `in_proj_a`, `in_proj_b` rows, `A_log`, `dt_bias` | 16 = `linear_num_value_heads` (resolved in step 2: the reference sizes them by value heads, and Qwen3.5-35B-A3B, where the two counts differ, has `A_log` [32] with 32 value and 16 key heads) |
| `conv1d.weight` [6144, 1, 4] | depthwise over the 6144 q/k/v channels, kernel `linear_conv_kernel_dim` 4 |
| `linear_attn.norm.weight` [128] | `linear_value_head_dim` |
| `q_proj` rows 4096 | 2 × 8 heads × 256 (`head_dim`): query and output gate, interleaved per head (resolved in step 2, see [Full-attention layer](#full-attention-layer)) |
| `k_proj`, `v_proj` rows 512 | 2 KV heads × 256 |
| `o_proj` cols 2048 | 8 heads × 256 |
| `q_norm`, `k_norm` [256] | `head_dim`: per-head norms |
| `mlp.*` 3584 | `intermediate_size` |

## Observations for step 2 (evidence, not contract)

Weight statistics measured on this checkpoint in step 1. Step 2 confirmed the reading below against the reference: see [RMSNorm](#rmsnorm-zero-centered) and [Gated RMSNorm](#gated-rmsnorm-plain-weight).

| Tensor | mean | min | max |
| --- | --- | --- | --- |
| `layers.0.input_layernorm.weight` | +0.24 | −0.11 | +1.00 |
| `layers.0.post_attention_layernorm.weight` | +0.09 | −0.36 | +0.28 |
| `norm.weight` | +3.31 | −0.78 | +5.31 |
| `layers.3.self_attn.q_norm.weight` | +0.43 | −0.21 | +1.23 |
| `layers.3.self_attn.k_norm.weight` | +0.42 | −0.90 | +1.63 |
| `layers.0.linear_attn.norm.weight` (f32) | +0.95 | +0.52 | +1.06 |

The decoder-layer norms and `q_norm`/`k_norm` center near zero with negative entries, as a zero-centered `(1 + w)` parameterization would; the Gated DeltaNet output norm centers near one, as a plain `w` would. Other config values the contract must cover: `rms_norm_eps` 1e-6, `rope_theta` 10,000,000, `partial_rotary_factor` 0.25, `mrope_section` [11, 11, 10] with `mrope_interleaved: true`, `mamba_ssm_dtype` float32.

## Op contract

Confirmed against transformers 5.24.0 (`modeling_qwen3_5.py`). Notation: `T` tokens, `H` = `hidden_size` 1024, activations in the checkpoint dtype (bf16) unless an fp32 island is stated. Every value written as a config key is read from `config.json` at load time.

### Decoder layer and residual stream

```
h = x + mixer(rmsnorm(x; input_layernorm))          mixer: Gated DeltaNet or full attention, per layer_types
y = h + mlp(rmsnorm(h; post_attention_layernorm))
logits = rmsnorm(x_last; norm) · embed_tokensᵀ      tie_word_embeddings: true, no lm_head tensor
```

- The residual additions happen in the activation dtype (bf16); the reference keeps no fp32 copy of the residual stream across layers.
- The layer schedule is `text_config.layer_types` (`linear_attention` → Gated DeltaNet, `full_attention`). When the key is absent the reference derives it from `full_attention_interval` (default 4): layer `i` is full attention iff `(i + 1) % interval == 0`.
- The reference computes logits with `lm_head` in the activation dtype (bf16). `docs/MODELS.md` asks for fp32 logits in athanor: a deliberate difference, measured in step 4. On the prefill of the two forward prompts of the golden data, computing the reference's own logits in fp32 from its bf16 final hidden state lowers its mean KL against an fp32 run of the model from 5.4e-4 to 2.2e-4 (prose) and from 5.1e-4 to 3.6e-4 (chat): the bf16 rounding of the logits is about half of the reference's distance from the exact model on prose.

### RMSNorm (zero-centered)

Resolves the RMSNorm **verify** item for `input_layernorm`, `post_attention_layernorm`, the final `norm`, `q_norm` and `k_norm` (`Qwen3_5RMSNorm`, lines 835–852):

```
y = x̂ · (1 + w),   x̂ = x / sqrt(mean(x²) + eps),   eps = rms_norm_eps (1e-6)
```

Everything is computed in fp32 (x upcast, weight upcast) and rounded to bf16 once at the end. The checkpoint stores `w`, which centers near zero (step 1 observations); the reference initializes it to zeros.

### Gated RMSNorm (plain weight)

The Gated DeltaNet output norm `linear_attn.norm` (`Qwen3_5RMSNormGated`, lines 204–220) uses a plain weight and is applied per value head over `linear_value_head_dim` (128):

```
n = x / sqrt(mean(x²) + eps)          fp32, then rounded to bf16
y = (w · n) · silu(z)                 w plain (f32 in the checkpoint), silu(z) in fp32, result rounded to bf16
```

`eps` is `rms_norm_eps`. The order of operations matters for bit-level agreement: normalize, round to bf16, scale by the weight, multiply by the fp32 gate, round.

### Full-attention layer

`Qwen3_5Attention` (lines 742–815). Heads: `num_attention_heads` 8, `num_key_value_heads` 2, `head_dim` 256.

| Step | Contract |
| --- | --- |
| Q and gate | `q_proj` gives [T, 8 × 2 × 256]. Per head `h`, columns `[512h, 512h + 256)` are the query and `[512h + 256, 512h + 512)` the output gate (the projection is viewed as [T, heads, 2 × head_dim] and split in halves). Resolves the output-gate **verify** item. |
| K, V | `k_proj`, `v_proj` give [T, 2 × 256]. |
| QK-norm | zero-centered RMSNorm per head over `head_dim`, on q and k, before RoPE. |
| RoPE | rotate-half on the first `rope_dim = int(head_dim × partial_rotary_factor)` = 64 dimensions of each head; the other 192 pass through unchanged. `inv_freq[i] = rope_theta^(−2i / rope_dim)` for `i` in [0, 32), `rope_theta` 10,000,000; cos and sin computed in fp32 and cast to bf16; dimension `j` pairs with `j + 32` inside the rotary part. `rope_type` must be `default`. Resolves the RoPE **verify** item. |
| M-RoPE | `mrope_section` [11, 11, 10] with `mrope_interleaved` mixes three position streams (time, height, width) across frequencies. For text the three streams are the same token index (`Qwen3_5TextModel.forward`, `compute_3d_position_ids` without image or video grids), so the interleaving is the identity and RoPE is the plain 1D form above. |
| GQA | KV head `j` serves query heads `[4j, 4j + 4)` (`repeat_kv`). |
| Scores | `q · kᵀ · head_dim^−0.5`, causal mask, softmax in fp32, probabilities cast to bf16, times v. In the eager path `q · kᵀ` is a bf16 matmul, so the scores are rounded to bf16 before the softmax (`head_dim^−0.5` = 1/16 is a power of two and adds no rounding). athanor keeps scores and probabilities in fp32 (`ath_attn_ref`); step 3 measured the difference on this checkpoint: up to 9.8% of an output element (relative to its magnitude plus the tensor RMS), all of it from the score rounding (`tests/README.md`). |
| Output | `o = attn · sigmoid(gate)` on the flattened [T, 8 × 256] output, then `o_proj`. |
| Config | `attention_bias` false: no bias tensors. `attn_output_gate` is not read by the reference: the gate is always present. |

### Gated DeltaNet layer

`Qwen3_5GatedDeltaNet` (lines 500–656) and `torch_recurrent_gated_delta_rule` (lines 429–497). Dimensions: `nk` = `linear_num_key_heads` 16, `nv` = `linear_num_value_heads` 16, `dk` = `linear_key_head_dim` 128, `dv` = `linear_value_head_dim` 128, `K` = `linear_conv_kernel_dim` 4; `key_dim = nk·dk` 2048, `value_dim = nv·dv` 2048, `conv_dim = 2·key_dim + value_dim` 6144.

| Step | Contract |
| --- | --- |
| Projections | `qkv = in_proj_qkv(x)` [T, conv_dim] with channels q `[0, key_dim)`, k `[key_dim, 2·key_dim)`, v `[2·key_dim, conv_dim)`, each head-major (head `i` at `[i·d, (i+1)·d)`). `z = in_proj_z(x)` [T, value_dim], `b = in_proj_b(x)` [T, nv], `a = in_proj_a(x)` [T, nv]. |
| Convolution | causal depthwise conv1d over time on all `conv_dim` channels of `qkv` (before the split), weight [conv_dim, 1, K], no bias, left padding `K − 1`, then SiLU; computed in the weight dtype (bf16). The output at token `t` depends on inputs `t − K + 1 … t`, so decode needs the previous `K − 1` inputs per channel as conv state. |
| L2 norm | q and k per head in fp32: `x · rsqrt(Σx² + 1e-6)`. The epsilon is the fixed 1e-6 of `l2norm`, not `rms_norm_eps`. Resolves the L2-norm **verify** item. |
| Query scale | q is multiplied by `dk^−0.5` after the L2 norm. |
| Gates | `β = sigmoid(b)` (in bf16, then used in fp32); `g = −exp(A_log) · softplus(a + dt_bias)` in fp32; decay `α = exp(g)` ∈ (0, 1]. `A_log`, `dt_bias`, `in_proj_a` and `in_proj_b` are indexed by **value** head. |
| Head grouping | when `nv > nk`, q/k head `i` serves value heads `[i·r, (i+1)·r)` with `r = nv / nk` (`repeat_interleave`). Here `r` = 1. |
| Recurrence | per value head, state `S` [dk, dv] in fp32, zero at the start of a sequence: `S ← α_t·S`, `u = Sᵀk_t`, `S ← S + k_t·(β_t·(v_t − u))ᵀ`, `o_t = Sᵀq_t`. Equivalently `S_t = α_t·(I − β_t k_t k_tᵀ)·S_{t−1} + β_t·k_t v_tᵀ`; `docs/MODELS.md` writes the transposed state. |
| Chunked form | `torch_chunk_gated_delta_rule` (chunk 64) computes the same recurrence for prefill; it is the reference for the step 10 kernels, not a different contract. Implemented in step 10 (`src/ops/cuda/gdn_chunk.cu`, fp32 as the reference's function): the same algebra reorganized (`docs/program/gdn-gpu-kernel.md`), checked against the recurrent reference on random inputs and on the golden records (`tests/README.md`, Prompt kernels). |
| Output | `o` [T, nv, dv] cast to bf16, gated RMSNorm per value head with `z` (above), flattened to [T, value_dim], then `out_proj`. Resolves the output **verify** item. |
| State per layer | recurrent `nv × dk × dv` fp32 (1 MiB here) plus conv `(K − 1) × conv_dim` values per sequence. `mamba_ssm_dtype: float32` in the config matches the fp32 recurrent state. |

### MLP

`down_proj(silu(gate_proj(x)) · up_proj(x))` (`Qwen3_5MLP`), `hidden_act` silu, width `intermediate_size` 3584, in bf16.

### Checkpoint dtypes

`A_log` and `linear_attn.norm.weight` are f32 in the checkpoint; every other tensor is bf16. The reference upcasts `A_log` itself (`A_log.float()`). athanor uses the checkpoint values as stored.

Resolved in step 3: transformers 5.24.0 rounds both tensors to bf16 on load, with `dtype="auto"` as well as `dtype=torch.bfloat16` (the model has no `_keep_in_fp32_modules`). The forward code works unchanged with f32 values: `A_log.float()` is a no-op, and in the gated RMSNorm an f32 weight promotes `w · n` to fp32 before the gate. The golden data restores the stored f32 values after loading (`tests/golden/README.md`), so the reference and athanor compute with the same weights. On the golden prompt, against an fp32 run of the whole model, the bf16 reference has a mean KL of 5.4e-4 (max 1.4e-3) with the stored values and 5.7e-4 (max 2.9e-3) with the default rounding.

### End-to-end (step 4)

athanor's forward pass (`src/models/qwen35/forward.c`) composes the ops above in this order. Measured by `tests/test_forward.c` against the forward records of the golden data (two prompts, 75 + 8 and 28 + 8 positions, prefill and greedy decode; `tests/README.md`, Forward pass):

| Run | Against | Result |
| --- | --- | --- |
| athanor, f32 activations | reference loaded in fp32 | mean KL 4e-11 (prose) and 5e-11 (chat); residual stream within 4.7e-5 at every layer, logits within 2.7e-5 |
| athanor, bf16 activations | reference loaded in fp32 | mean KL 3.0e-4 over all 119 positions (prose 1.7e-4, chat 6.0e-4) |
| reference, bf16 (eager) | reference loaded in fp32 | mean KL 5.0e-4 (prose 5.3e-4, chat 4.4e-4) |

The chat prompt's first position (`<|im_start|>`) is ill-conditioned. In layer 15 (full attention) the attention output nearly cancels the residual stream (RMS 0.53 and 0.27 give a sum of RMS 0.13), so the relative error of one bf16 rounding at the layer input becomes about 10% at its output, and the KL at that position depends on which way the roundings fall. The reference itself goes from a KL of 2.7e-4 there to 5.5e-3 when `A_log` and `linear_attn.norm.weight` are rounded to bf16 as transformers does by default; athanor's is 7.3e-3. Without that position, athanor's mean KL on chat is 4.1e-4 against the reference's 4.5e-4.

### MTP head: the draft head (step 13)

transformers 5.24.0 has no implementation of the MTP head: both `Qwen3_5ForConditionalGeneration` and `Qwen3_5ForCausalLM` drop `mtp.*` on load (`_keys_to_ignore_on_load_unexpected`). The reference is vLLM 0.34.0 (`model_executor/models/qwen3_5_mtp.py`, `Qwen3_5MultiTokenPredictor`; the target's hidden states from `qwen3_next.py`, `Qwen3NextModel.forward`, and `v1/worker/gpu/spec_decode/utils.py`, `get_drafter_hidden_states`).

For position p of the main model, with t the token after it and h the main model's output there **after the final norm** (the input of `lm_head`, which vLLM hands to the drafter):

```
e = rmsnorm(embed(t), pre_fc_norm_embedding)          zero-centered, as every norm here
g = rmsnorm(h, pre_fc_norm_hidden)
x = fc([e; g])                                         fc.weight [hidden, 2 hidden]: e first
x = decoder_layer(x)                                   a full-attention layer with its MLP (MoE on MoE
                                                       models), its own KV cache, RoPE at position p
logits = rmsnorm(x, norm) · lm_headᵀ                   the guess at the token after t
```

The head's cache holds one row per position of the sequence: athanor runs it over every prompt position (with the next prompt token) and over every position decoding keeps, as vLLM's drafter does. In athanor: `ath_model_draft` (model.h), `ath_qwen35_draft` (forward.c); the state adds the head's KV cache (`ATH_STATE_KV`, layer 24) and its input waiting for the next token (`ATH_STATE_HIDDEN`, `hidden_size` elements).

The guess never decides a token, only how many tokens a pass keeps (`src/infer/spec.h`), so its arithmetic follows the main layers' code without a tolerance of its own. **Verified** by its effect: on a coding prompt, greedy, the main model's next token equals the guess at 86% of the passes (Qwen3.6-35B-A3B: 98%; on a story 58% and 75%); and speculative decoding gives plain decoding's tokens, bit for bit (`tests/test_spec.c`).

## Tokenizer (step 5)

Source: the checkpoint's `tokenizer.json` (sha256 `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`), as Hugging Face tokenizers 0.23.3 runs it.

| Item | Value |
| --- | --- |
| Model | byte-level BPE, 248,044 tokens, 247,587 merges ("a b" strings); no unknown token, no byte fallback, `ignore_merges` false |
| Added tokens | 26, ids 248,044 to 248,069, matched in the raw text before anything else (leftmost, then longest); none strips spaces or is normalized. Special: `<\|endoftext\|>` (248044, padding), `<\|im_start\|>` (248045), `<\|im_end\|>` (248046, end of a turn), the vision markers. Not special but still single tokens: `<tool_call>`, `</tool_call>`, `<tool_response>`, `</tool_response>`, `<think>` (248068), `</think>` (248069), the FIM and repository markers |
| Normalizer | NFC. The reference applies Unicode 9.0 normalization data (tokenizers' `unicode-normalization-alignments`), so characters added later neither compose nor reorder: 21 compositions (Todhri, Tulu-Tigalari, Gurung Khema, Kirat Rai, Dives Akuru) and 29 combining classes are left out of athanor's tables to match (`tools/dev/gen_unicode.py`) |
| Pre-tokenizer | Split, behavior Isolated, regex `(?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d)\|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+\|\p{N}\| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*\|\s*[\r\n]+\|\s+(?!\S)\|\s+`, then ByteLevel without prefix space or regex. The regex engine is Oniguruma with Unicode 16.0 classes; `\s` is U+0009 to U+000D, U+0085, Zs, Zl, Zp; `(?i:)` folds U+017F (ſ) to s |
| Decoder | ByteLevel: the bytes of the tokens, added tokens as their text |
| No BOS | `add_bos_token` false; a prompt starts with `<\|im_start\|>` |

Verified on 28.2 million tokens of text, including every code point (`tests/test_text.c`, `golden_tokenizer_corpus`).

## Chat template (step 5)

Source: the checkpoint's `chat_template.jinja` (sha256 `273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80`), rendered by transformers 5.24.0 (Jinja with `trim_blocks` and `lstrip_blocks`; `tojson` is Python's `json.dumps(ensure_ascii=False)`, without HTML escaping). Text content only; images and videos are not supported. `src/models/qwen35/text.c` renders it by hand; `golden_chat_template` compares 17 conversations byte for byte.

```
[tools present]
<|im_start|>system
# Tools

You have access to the following functions:

<tools>
{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}     one line per tool
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix: ...   fixed instructions
                                                                    (then "\n\n" + system content, if not empty)
<|im_end|>
[no tools, first message system]
<|im_start|>system
{system content}<|im_end|>
[each message]
user        <|im_start|>user\n{content}<|im_end|>\n
assistant   <|im_start|>assistant\n[<think>\n{reasoning}\n</think>\n\n]{content}{tool calls}<|im_end|>\n
tool        (first of a run: <|im_start|>user)\n<tool_response>\n{content}\n</tool_response>(last of a run: <|im_end|>\n)
[generation prompt]
<|im_start|>assistant\n<think>\n                    thinking on
<|im_start|>assistant\n<think>\n\n</think>\n\n    thinking off
```

- Every content is trimmed with Python's `str.strip()` (Unicode white space, including U+001C to U+001F).
- The thinking block of an assistant turn is rendered only for turns after the last user message that is not just a `<tool_response>`. Its reasoning is `reasoning_content` when given, otherwise the text between `<think>` and `</think>` in the content (newlines around it stripped), and the content keeps what follows `</think>`.
- Tool calls of an assistant turn: `\n\n` before the first if the content is not empty, `\n` between calls; each is `<tool_call>\n<function=NAME>\n` + per argument `<parameter=KEY>\nVALUE\n</parameter>\n` + `</function>\n</tool_call>`. VALUE is `str()` of the argument in Python (a string as is, `True`, `None`, numbers as Python prints them), or `tojson` for objects and lists.
- Errors: no messages; a system message after the first; no user message that is not only a tool response.

## Output markup (step 5)

What the model writes after the generation prompt, as `parse` reads it (`parse_fixtures`):

| Markup | Event |
| --- | --- |
| text up to `</think>` when the prompt opened thinking, or between `<think>` and `</think>` | thinking; the newlines next to the tags are formatting and dropped |
| `<tool_call>\n<function=NAME>\n<parameter=P>\nVALUE\n</parameter>\n...</function>\n</tool_call>` | a tool call: begin (NAME), arguments as one JSON object, end. VALUE becomes JSON by P's type in the tool's schema (string, integer, number, boolean, object, array; `null`), a string when it does not parse as that type; newlines before, between and after calls are formatting |
| anything else | text; `<tool_call>` without a well-formed `<function=...>` header is text |

The turn ends with `<|im_end|>` (id 248046), which the caller recognizes as a token, before parsing.

Recommended sampling (model card): thinking 0.6 / 0.95 / 20 (temperature, top-p, top-k) for precise tasks, or 1.0 / 0.95 / 20 with presence penalty 1.5; without thinking 0.7 / 0.8 / 20 with presence penalty 1.5, or 1.0 / 1.0 / 20 with 2.0. The presence penalty is not implemented yet (`docs/STATUS.md`, Open questions).

## Parameter binding

Step 2 binds every language-model tensor to an `ath_param` (`src/model/model.h`) with a role, a layer, an expert index and flags; `athanor inspect --roles` prints the result. Flags: `residual` = `ATH_P_WRITES_RESIDUAL` (abliteration projects the direction out of these), `sensitive` = `ATH_P_SENSITIVE` (quantization keeps these at higher precision by default), `shared` = `ATH_P_SHARED_EXPERT`. The sensitive set is a default to be measured in phase 2: the output projection (here the tied embedding) and the small matmuls that produce per-head gates.

| Tensor (under `model.language_model.`) | Role | Flags |
| --- | --- | --- |
| `embed_tokens.weight` | embed (also the output projection: tied) | residual, sensitive |
| `layers.N.input_layernorm.weight`, `layers.N.post_attention_layernorm.weight`, `norm.weight` | norm | |
| `linear_attn.in_proj_qkv.weight`, `linear_attn.in_proj_z.weight` | mix_in | |
| `linear_attn.in_proj_a.weight`, `linear_attn.in_proj_b.weight` | gate | sensitive |
| `linear_attn.conv1d.weight` | conv | |
| `linear_attn.A_log`, `linear_attn.dt_bias` | other | |
| `linear_attn.norm.weight` | norm | |
| `linear_attn.out_proj.weight` | mix_out | residual |
| `self_attn.q_proj.weight`, `self_attn.k_proj.weight`, `self_attn.v_proj.weight` | mix_in | |
| `self_attn.q_norm.weight`, `self_attn.k_norm.weight` | norm | |
| `self_attn.o_proj.weight` | mix_out | residual |
| `mlp.gate_proj.weight`, `mlp.up_proj.weight` | mlp_in | |
| `mlp.down_proj.weight` | mlp_out | residual |

The MTP head's 15 tensors are the draft head since step 13: flagged `draft` (`ATH_P_DRAFT`), after the main params, `fc` as mix_in, the norms as norm, `layers.0.*` with the roles of a full-attention layer, every one with layer 24 (the head's layer after the 24 main ones; `m->n_draft_layers` is 1). The other 153 tensors are the declared ignored group the reference text model drops: `model.visual.` (vision encoder, text path only); before step 13 `mtp.` was one too. Since CHANGE-001 (2026-11-24) each of them is bound as a whole-tensor param flagged `unused` (`ATH_P_UNUSED`, role other, layer −1, after all the other params), so `inspect --roles` lists them and reports their count and size, and an export can carry them; before, they were left unbound. Any other tensor, a missing tensor, or a shape that does not follow from `config.json` makes loading fail. MoE checkpoints add the roles router, expert_in and expert_out: see `docs/contract/qwen35-35b-a3b.md`.
