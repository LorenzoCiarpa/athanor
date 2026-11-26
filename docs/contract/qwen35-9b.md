# Contract: Qwen3.5-9B (Q8_0 GGUF)

The model of step 14's gate: a long context on a model that only just fits the GPU. Its math is the [0.8B contract](qwen35-0.8b.md)'s: the same classes (`Qwen3_5*`, transformers 5.24.0), a dense MLP, larger dimensions. Its bf16 weights (17.9 GB) do not fit the 16 GB GPU, so athanor runs Unsloth's Q8_0 GGUF (`docs/STATUS.md`, step 14 decisions). How llama.cpp's converter stores the tensors, and how athanor reads them back, is recorded in the [Qwen3.6-35B-A3B contract](qwen36-35b-a3b.md); this file lists what differs.

## Checkpoint

| Item | Value |
| --- | --- |
| Configuration and tokenizer | [`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B), revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`: `config.json` (sha256 `d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05`), `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `vocab.json`, `merges.txt` |
| Weights | [`unsloth/Qwen3.5-9B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF), revision `3885219b6810b007914f3a7950a8d1b469d598a5`, `Qwen3.5-9B-Q8_0.gguf`, 9,527,502,048 bytes, sha256 `809626574d0cb43d4becfa56169980da2bb448f2299270f7be443cb89d0a6ae4` |
| GGUF | version 3, 46 metadata keys, `general.architecture` `qwen35`, 427 tensors: 250 Q8_0, 177 F32; data section 8.86 GiB from offset 10,968,288, alignment 32 |
| Model directory | `models/Qwen3.5-9B/`: the GGUF beside the Hugging Face files (no safetensors, so `ath_ckpt` reads the GGUF) |
| Reference implementation | transformers `5.24.0`, `models/qwen3_5/modeling_qwen3_5.py`, the file of the 0.8B contract (sha256 `c4d847b7e809ed2a49611e5afcea8bd0db39de597b67b57ce9753fb65f550de2`) |

The configuration (`text_config`):
- 32 layers, 24 Gated DeltaNet and 8 full attention (`full_attention_interval` 4: layers 3, 7, ..., 31);
- hidden size 4096, MLP intermediate size 12,288;
- 16 attention heads, 4 KV heads, `head_dim` 256; partial rotary factor 0.25, `rope_theta` 10,000,000, interleaved mRoPE sections [11, 11, 10];
- Gated DeltaNet: 16 key heads and 32 value heads of 128, convolution kernel 4;
- vocabulary 248,320, untied; `max_position_embeddings` 262,144; `rms_norm_eps` 1e-6.

`mtp_num_hidden_layers` is 1, but this GGUF has no MTP block (no `blk.32`): the draft head of step 13 is not available with this file. `chat` and `bench` refuse `--mtp` (the model has no draft head); `serve` says so and decodes without it.

## GGUF names

Every name of the 35B's table applies (the norms stored as 1 + w, `ssm_a` as A = −exp(A_log), the convolution squeezed, the value heads in tiled order: 16 key heads, 32 value heads), except the MoE ones. In their place, the dense MLP:

| GGUF tensor | Hugging Face tensor | Stored |
| --- | --- | --- |
| `blk.N.ffn_gate.weight` [12288, 4096] | `mlp.gate_proj.weight` | Q8_0, unchanged |
| `blk.N.ffn_up.weight` [12288, 4096] | `mlp.up_proj.weight` | Q8_0, unchanged |
| `blk.N.ffn_down.weight` [4096, 12288] | `mlp.down_proj.weight` | Q8_0, unchanged |

`attn_qkv` and `attn_gate` come as two tensors, as do `ssm_beta` and `ssm_alpha` (this file's converter merged neither pair). `token_embd.weight` and `output.weight` are both Q8_0, 1.08 GB each. The small tensors (norms, `ssm_a`, `ssm_dt.bias`, `ssm_conv1d`) are F32.

## State of one sequence

| Part | Size |
| --- | --- |
| KV cache, bf16 (8 layers × keys and values × 4 heads × 256) | 32 KiB per token: 4 GiB at 131,072 tokens |
| KV cache, int8 with a scale per row (step 14) | 16.25 KiB per token: 2.03 GiB at 131,072 tokens |
| Gated DeltaNet recurrent state (24 layers × 32 heads × 128 × 128, f32) | 48 MiB, fixed |
| Convolution state (24 layers × 3 × 8192, activation dtype) | 1.1 MiB in bf16, fixed |

## Verification

`tests/golden/gguf.py` builds forward records from the GGUF, dequantized exactly, with the converter's transforms undone. The reference runs one layer at a time in fp32 and in bf16, as `moe.py` does for the 35B. `test_forward`'s `forward_golden` replays the records with `ATH_MODEL_DIR=models/Qwen3.5-9B ATH_GOLDEN_DIR=tests/golden/data/qwen35-9b`.

Measured 2026-11-26:
- bf16: pooled KL 1.283e-4 on the CPU and 1.864e-4 on the GPU, against the bf16 reference's 3.829e-4 (ratios 0.33 and 0.49). 0 greedy differences at the 206 positions above the noise (0.238).
- f32: KL about 1e-11. Logits within 1.6e-5 (CPU) and 1.1e-4 (GPU); residual stream within 2.0e-5 and 8.5e-5.

On this model, two positions of the chat prompt are ill-conditioned: the newline after `<|im_end|>` and the `<|im_start|>` after it. On both devices their rounding error grows by a factor of 4 to 20 through the layers, so the f32 tolerances are 5e-4 for the logits and 4e-4 for the residual stream, four times the worst measured value. `tests/README.md` (Forward records of Qwen3.5-9B) has the analysis.
