# Contract: Qwen3.6-35B-A3B (Q8_0 GGUF)

The model of step 12's gate. Its math is that of [Qwen3.5-35B-A3B](qwen35-35b-a3b.md): the same architecture class, the same dimensions, and the MoE block contract recorded there. Everything else follows the [0.8B contract](qwen35-0.8b.md). The weights are not the Hugging Face checkpoint. They are Unsloth's Q8_0 GGUF, the file Strata runs (`docs/STATUS.md`, step 12 decisions). This file therefore records what llama.cpp's converter did to the tensors, and how athanor reads them back.

## Checkpoint

| Item | Value |
| --- | --- |
| Configuration and tokenizer | [`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), revision `345d467b117d1fd5eda9b879a5988de9ae5f47a7`: `config.json` (sha256 `93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99`), `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `generation_config.json`, `vocab.json`, `merges.txt` |
| Weights | [`unsloth/Qwen3.6-35B-A3B-MTP-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF), revision `5bc3e238d916f48a861bac2f8a1990a0e9b7e98d`, `Qwen3.6-35B-A3B-Q8_0.gguf`, 37,801,097,504 bytes, sha256 `c1283d8b80c3e38b2735ddbc9766d3b3126f44d6c484be419d4e101d09a76131` |
| GGUF | version 3, 55 metadata keys, `general.architecture` `qwen35moe`, `qwen35moe.block_count` 41 (40 layers and the MTP block), 753 tensors: 443 Q8_0, 308 F32, 2 BF16; data section 37,790,106,112 bytes from offset 10,991,392, alignment 32 |
| Model directory | `models/Qwen3.6-35B-A3B/`: the GGUF beside the Hugging Face files (the safetensors index is not there, so `ath_ckpt` reads the GGUF: `src/core/ckpt.h`) |
| Reference implementation | transformers `5.24.0`, `models/qwen3_5_moe/modeling_qwen3_5_moe.py` sha256 `aa3caec05ddf172964b6cfe9b19d333dde892ee99b75bc629c4c18169860f5d6`: the file of the 3.5 contract. The token mixers are the dense model's classes; only the MLP is the MoE block (diffed 2026-11-25) |
| Converter | llama.cpp `5220a2d`, `conversion/qwen.py` sha256 `2f62b62fb1527c7693a8e178ca39fae6f04578dd089fb609725e1fce99b5c0cf` (classes `Qwen3NextModel`, `_LinearAttentionVReorderBase`, `Qwen3_5MoeTextModel`) and `conversion/base.py`. Unsloth's file is older (May 2026). The tensors it contains match this converter's MoE path, which merges nothing (see GGUF names) |

The configuration (`text_config`) has the dimensions of Qwen3.5-35B-A3B:
- 40 layers, 30 Gated DeltaNet and 10 full attention, interval 4;
- hidden size 2048;
- 16 attention heads, 2 KV heads, `head_dim` 256;
- Gated DeltaNet: 16 key heads and 32 value heads of 128;
- 256 experts, 8 per token, `moe_intermediate_size` 512, `shared_expert_intermediate_size` 512;
- vocabulary 248,320, untied.

Compared with 3.5, the 3.6 file adds `bos_token_id`, `output_router_logits`, `pad_token_id`, a top-level `partial_rotary_factor` (the same 0.25 as in `rope_parameters`) and `tie_word_embeddings: false`, and drops `mlp_only_layers`. None of these changes the math.

## GGUF names and the converter's transforms

Verified against the converter's source (`conversion/qwen.py`, `conversion/base.py`), and for the dense names on Qwen3.5-0.8B's GGUF, read both ways (Verification).

| GGUF tensor | Hugging Face tensor | What the converter did | How athanor reads it |
| --- | --- | --- | --- |
| `token_embd.weight` [V, H] | `embed_tokens.weight` | Q8_0 | embedding lookup on Q8_0 (`ath_embed`) |
| `output.weight` [V, H] | `lm_head.weight` | Q8_0; absent when tied | output head (Q8_0 matmul) |
| `output_norm.weight`, `blk.N.attn_norm.weight`, `blk.N.post_attention_norm.weight`, `blk.N.attn_q_norm.weight`, `blk.N.attn_k_norm.weight` | `norm.weight`, `input_layernorm.weight`, `post_attention_layernorm.weight`, `q_norm.weight`, `k_norm.weight` | stores **1 + w**: every `norm.weight` except the Gated DeltaNet's (`Qwen3NextModel.modify_tensors`), computed in f32 after the bf16 weight was widened (`base.py`, line 1578), so exact; F32 | the zero-centered norms with offset 0 instead of 1 (`ath_qwen35_model.norm_offset`): (0 + (1 + w)) is what the reference's (1 + w) gives, bit for bit |
| `blk.N.ssm_norm.weight` [dv] | `linear_attn.norm.weight` | unchanged (plain weight), F32 | as before |
| `blk.N.ssm_a` [nv] | `linear_attn.A_log` | stores **A = −exp(A_log)**, torch's exp in f32 | the gates take A as it is (`ath_gdn_gates`, `a_given`): g = A · softplus(a + dt_bias) |
| `blk.N.ssm_dt.bias` [nv] | `linear_attn.dt_bias` | renamed | as before |
| `blk.N.ssm_conv1d.weight` [C, K] | `linear_attn.conv1d.weight` [C, 1, K] | squeezed | bound as a [C, 1, K] view |
| `blk.N.attn_qkv.weight`, `blk.N.attn_gate.weight` | `in_proj_qkv.weight`, `in_proj_z.weight` | the value rows reordered (below); the dense converter merges the two into `attn_qkv_gate` [conv_dim + value_dim, H] | separate, or row slices of the merged tensor |
| `blk.N.ssm_beta.weight`, `blk.N.ssm_alpha.weight` [nv, H] | `in_proj_b.weight`, `in_proj_a.weight` | reordered; the dense converter merges them into `ssm_beta_alpha` [2 nv, H], beta first | separate, or row slices |
| `blk.N.ssm_out.weight` [H, value_dim] | `linear_attn.out_proj.weight` | the value columns reordered | as before |
| `blk.N.attn_q.weight` [2 q, H] | `self_attn.q_proj.weight` | unchanged: query and output gate interleaved per head | as before |
| `blk.N.attn_k`, `attn_v`, `attn_output` | `k_proj`, `v_proj`, `o_proj` | unchanged | as before |
| `blk.N.ffn_gate_inp.weight` [E, H] | `mlp.gate.weight` (router) | F32 (llama.cpp does not quantize the router) | router matmul on f32 weights |
| `blk.N.ffn_gate_exps.weight`, `ffn_up_exps.weight` [E, I, H] | `mlp.experts.gate_up_proj` [E, 2I, H] | split into gate and up (`Qwen2MoeModel.modify_tensors`); Q8_0 | per expert a [I, H] view of each; the Hugging Face fused tensor is bound as the same two views, rows [0, I) and [I, 2I) |
| `blk.N.ffn_down_exps.weight` [E, H, I] | `mlp.experts.down_proj` | Q8_0 | per expert a [H, I] view |
| `blk.N.ffn_gate_shexp`, `ffn_up_shexp` [Is, H], `ffn_down_shexp` [H, Is] | `mlp.shared_expert.*` | Q8_0 | as before |
| `blk.N.ffn_gate_inp_shexp.weight` [H] | `mlp.shared_expert_gate.weight` [1, H] | squeezed; F32 | bound as a [1, H] view |
| `blk.40.*` (20 tensors, `nextn.*` among them) | `mtp.*` | the MTP head as one more block | an ignored group (`blk.40.`), bound as `unused` params until step 13 |

**Value heads in tiled order.** The converter reorders the Gated DeltaNet's value heads when the key heads are fewer (16 key heads, 32 value heads here; `_LinearAttentionVReorderBase`, for ggml's broadcast). The reference groups them by key head: value head h reads key head h / 2. The GGUF tiles them: value head h reads key head h mod 16.

The permutation is applied consistently to every tensor indexed by value head:
- the V rows of `attn_qkv`;
- the rows of `attn_gate`, `ssm_beta` and `ssm_alpha`;
- `ssm_a` and `ssm_dt.bias`;
- the V channels of `ssm_conv1d`;
- the columns of `ssm_out`.

So the layer computes the same function with its value heads permuted. Only the pairing of a value head with its key head changes. athanor passes `tiled` to the gated delta rule (`ath_gdn`): key head h mod nk instead of h / (nv / nk). With as many value heads as key heads (the 0.8B) the two orders are the same.

**Quantization.** Q8_0: blocks of 32 weights, an f16 scale d and 32 int8 values, each weight d · q (llama.cpp's `quantize_row_q8_0`: d = max|x| / 127, q = round(x / d)). The file was quantized from the bf16 checkpoint. Q8_0 ignores the importance matrix named in the metadata (`quantize.imatrix.*`), which only lower formats use. Q8_0 tensors: every matmul and the embedding. F32 tensors: norms, router, shared-expert gate, convolution, `ssm_a`, `ssm_dt`. BF16: the MTP block's router and shared gate.

## Verification

- **Binding.** `athanor inspect models/Qwen3.6-35B-A3B --roles`: all 753 tensors bound, 40 MoE layers with 256 experts and 8 active. Per Gated DeltaNet layer 784 params, per attention layer 781 (two norms, the mixer, the router, 3 × 256 expert views, 4 shared). `tests/test_model.c` (`qwen36_gguf_binds`, when the file is there).
- **The transforms**, on the dense model, whose safetensors athanor already runs: Qwen3.5-0.8B's bf16 GGUF, written in step 9 by the converter at `5220a2d` (`convert_hf_to_gguf.py --outtype bf16`). It uses every transform above except the value reorder (16 and 16 heads), with qkv/z and beta/alpha merged. `tests/test_forward.c` (`gguf_matches_safetensors`) runs both on 12 tokens:
  - CPU, bf16 activations: the same logits, bit for bit;
  - CPU, f32 activations: within 2.4e-5;
  - GPU, f32: within 6.0e-6;
  - GPU, bf16: up to 0.116 apart, and the greedy token is the same at every position.
  
  The one value that differs is A: the converter computed −exp(A_log) with torch, athanor with libm's or CUDA's expf. The reference computes it with torch, as the converter did.
- **The value reorder, and everything else**, by the MoE model's forward records against the reference (`tests/golden/moe.py`, step 12). The script reads the GGUF and undoes every transform: it subtracts 1 from the norms, takes log(−A), permutes the value heads back to grouped, and dequantizes Q8_0 exactly in f32. It runs transformers' modules one layer at a time, in fp32 and in bf16, on 4 prompts (234 positions).
  - athanor on the GPU with f32 activations: KL 1e-12, logits within 9.6e-6, residual stream within 3.5e-5. Every transform reads back right.
  - With bf16 activations: pooled KL 1.811e-3 against the bf16 reference's 2.970e-3, and the same greedy token wherever the top-2 margin exceeds the bf16 noise.

  The bf16 reference rounds the dequantized weights to bf16; athanor keeps them exact.
