# tests/golden

Golden data: what the pinned reference implementation computes, op by op, on the real weights. `tests/test_golden.c` replays it through athanor's `_ref` functions. Development only: Python is never needed to build, test without golden data, or run athanor.

| File | Content |
| --- | --- |
| `dump.py` | runs the reference and writes the golden data |
| `requirements.in` | the two pinned packages (transformers 5.24.0, torch 2.15.0 CPU build) |
| `requirements.txt` | the full lock, compiled from `requirements.in` with uv |
| `data/<model>/` | `manifest.json` and `golden.safetensors`; ignored by git, regenerated locally |

## Making the data

```
uv venv --python 3.14 tests/golden/.venv
VIRTUAL_ENV=tests/golden/.venv uv pip install -r tests/golden/requirements.txt --index-strategy unsafe-best-match
make golden                       # tests/golden/.venv/bin/python tests/golden/dump.py models/Qwen3.5-0.8B
make test                         # test_golden replays it; skipped while data/ is missing
```

`dump.py` refuses to run unless transformers is 5.24.0, torch is 2.15.0+cpu, `modeling_qwen3_5.py` has the sha256 recorded in `docs/contract/qwen35-0.8b.md`, and none of `fla`, `causal_conv1d`, `kernels` is installed (any of them would replace the torch functions with fused kernels). It takes about 4 seconds and writes 434 MiB for Qwen3.5-0.8B; the numbers are reproducible run to run (one CPU thread).

## What is recorded

The reference runs on the CPU in bf16 over a fixed 75-token prompt, long enough for the reference's chunked Gated DeltaNet to cross a chunk boundary (64 tokens), then one decode step with the greedy next token and the cache. Every op of all 24 layers, in both phases, becomes a record:

| Op | Captured from | Inputs | Outputs |
| --- | --- | --- | --- |
| `embed` | `embed_tokens` | `ids` | `out` |
| `rmsnorm` | every `Qwen3_5RMSNorm` (layer norms, `q_norm`, `k_norm`, final norm) | `x` | `out`, `out32` |
| `matmul` | every `nn.Linear`, including the tied `lm_head` | `x` (rows 0 and T − 1 in prefill) | `out`, `out32` |
| `rope` | `apply_rotary_pos_emb`, once for q and once for k | `x`, `pos` | `out`, `out32` |
| `attn` | `eager_attention_forward` | `q`, `k`, `v` (after RoPE, the whole cache in decode) | `out`, `out32` |
| `sigmoid_mul` | the output gate of the attention layer | `x`, `gate` | `out`, `out32` |
| `conv` | `causal_conv1d_fn` (prefill), `causal_conv1d_update` (decode) | `x`, `state_in` (decode) | `out`, `out32`, `state_out` |
| `l2norm` | `l2norm` on q and k inside the Gated DeltaNet function | `x` | `out` (f32) |
| `gdn_gates` | `in_proj_a`, `in_proj_b` outputs and the gates passed to the recurrence | `a`, `b` | `g` (f32), `beta`, `beta32` |
| `gdn` | `torch_chunk_gated_delta_rule` (prefill), `torch_recurrent_gated_delta_rule` (decode) | `q`, `k`, `v`, `g`, `beta`, `state_in` (decode) | `out`, `out32`, `state_out`, `state_out32` |
| `rmsnorm_gated` | `Qwen3_5RMSNormGated` | `x`, `z` | `out`, `out32` |
| `swiglu` | the MLP activation | `gate`, `up` | `out`, `out32` |
| `add` | both residual additions of each layer | `a`, `b` | `out`, `out32` |

Records are captured with forward hooks on modules and by wrapping the module-level functions the forward code calls (each wrapper calls the original). Two outputs per op:

- `out`: the reference's own result, in its own dtype, with its own intermediate roundings.
- `out32`: the same reference function recomputed in fp32 from the same inputs (weights widened, fp32 RoPE tables). For the Gated DeltaNet the fp32 rerun uses the recurrent form, athanor's algorithm, also for the chunked prefill. No bf16 rounding is left in `out32`, so it checks the math of the `_ref` functions tightly.

Each record carries the checkpoint names of its weights and its parameters (eps, RoPE theta and dimension, attention scale, Gated DeltaNet query scale and L2-norm epsilon), read from the reference modules. `manifest.json` lists the records, the prompt and its tokens, the versions, the CPU, the sha256 of `config.json` and of the data file.

## Two deliberate choices

- **Eager attention.** The default `sdpa` fuses the softmax and both matmuls; the contract was written from `eager_attention_forward`, which is the readable path. The eager path rounds `q·kᵀ` to bf16 before the softmax; `dump.py` checks that keeping those scores in fp32 brings its output within 2 bf16 ulps of the fp32 rerun, which is why the native tolerance of `attn` is loose and its exact one is not (`tests/README.md`).
- **f32 tensors as stored.** `from_pretrained` rounds the 36 f32 tensors of the checkpoint (`linear_attn.A_log`, `linear_attn.norm.weight`) to bf16, with `dtype="auto"` as well as `dtype=torch.bfloat16`: the model has no `_keep_in_fp32_modules`. `dump.py` puts the stored f32 values back after loading; the forward code is unchanged (it upcasts `A_log` and promotes the norm weight). athanor uses the stored values, so both sides compute with the same weights. Measured on the prompt: against an fp32 run of the whole model, the bf16 reference with restored tensors has a mean KL of 5.4e-4 (max 1.4e-3); with the default rounding, 5.7e-4 (max 2.9e-3).
