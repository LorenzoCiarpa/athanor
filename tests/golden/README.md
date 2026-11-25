# tests/golden

Golden data: what the pinned reference implementation computes, op by op and for the whole forward pass, on the real weights. `tests/test_golden.c` replays the ops through athanor's `_ref` functions; `tests/test_forward.c` runs the forward pass on the same tokens. Development only: Python is never needed to build, test without golden data, or run athanor.

| File | Content |
| --- | --- |
| `dump.py` | runs the reference and writes the golden data of the ops and the forward pass |
| `text.py` | the text golden data: tokenizer corpus, chat template renderings, a greedy chat generation (`tests/README.md`, Text) |
| `generate.py` | the generations of the step 8 gate: 50 chat prompts, 256 greedy tokens each by the fp32 reference with its KV and recurrent cache, the top-2 margin of every step and the bf16 noise of each prompt (`tests/README.md`, Engine) |
| `requirements.in` | the two pinned packages (transformers 5.24.0, torch 2.15.0 CPU build) |
| `requirements.txt` | the full lock, compiled from `requirements.in` with uv |
| `data/<model>/` | `manifest.json` and `golden.safetensors` (`dump.py`), `text.json` and `text.safetensors` (`text.py`), `generate.json` (`generate.py`); ignored by git, regenerated locally |

## Making the data

```
uv venv --python 3.14 tests/golden/.venv
VIRTUAL_ENV=tests/golden/.venv uv pip install -r tests/golden/requirements.txt --index-strategy unsafe-best-match
make golden                       # dump.py, then text.py, on models/Qwen3.5-0.8B
make golden-generate              # generate.py: about 20 minutes on 8 threads
make test                         # test_golden, test_forward, test_text and test_engine use it; skipped while data/ is missing
```

`dump.py` refuses to run unless transformers is 5.24.0, torch is 2.15.0+cpu, `modeling_qwen3_5.py` has the sha256 recorded in `docs/contract/qwen35-0.8b.md`, and none of `fla`, `causal_conv1d`, `kernels` is installed (any of them would replace the torch functions with fused kernels). It takes about 9 seconds and writes 620 MiB for Qwen3.5-0.8B (434 MiB of op records, the rest forward records); the numbers are reproducible run to run (one CPU thread). `text.py` takes about 35 seconds and writes 163 MiB: the token ids of the corpus (most of it the sweep over every code point), the rendered prompts, and a generation that runs the fp32 and bf16 reference; it uses `dump.py`'s environment checks and its loading of the reference (`check_text_path`, `restore_f32`). `generate.py` (step 8) has its own target because it takes about 20 minutes: 12,800 decode steps of the fp32 reference on the CPU (8 threads, about 11 tokens/s; the reference's Gated DeltaNet and convolution run their torch fallbacks), then one bf16 pass over each prompt's tokens; it writes `generate.json` (about 0.5 MiB). Decoding does not stop at the end-of-turn token, so every prompt has 256 tokens on both sides.

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

Each record carries the checkpoint names of its weights and its parameters (eps, RoPE theta and dimension, attention scale, Gated DeltaNet query scale and L2-norm epsilon), read from the reference modules. `manifest.json` lists the records, the prompt and its tokens, the versions, the reference class, the CPU, the sha256 of `config.json` and of the data file. The checkpoint is a vision-language model; `dump.py` runs its text path only (`AutoModelForCausalLM` gives `Qwen3_5ForCausalLM`, without the vision encoder), stops if it loads anything else, and records the class as `reference.class`.

## Forward records

For the end-to-end comparison of step 4, `dump.py` then runs fixed prompts through the whole model twice:

| Prompt | Tokens | Decode |
| --- | --- | --- |
| `prose`: the op prompt | 75 | 8 greedy steps |
| `chat`: a user question in the checkpoint's chat template, with the generation prompt (empty thinking block) written out | 28 | 8 greedy steps |

- the bf16 reference (the same model as the op records: eager attention, f32 tensors restored) prefills the prompt, then decodes 8 tokens one at a time with its cache, each the greedy choice of the previous step;
- the reference loaded in fp32 (every weight widened exactly; the f32 tensors as stored) runs the same tokens the same way.

Per prompt, `fwd.<name>.tokens` (the prompt and the 8 decoded tokens), `.residual` and `.residual32` (the residual stream entering every decoder layer and the final norm, `[25, positions, 1024]`, from the bf16 and the fp32 run), `.logits` and `.logits32` (`[positions, 248320]`). The manifest lists the prompts under `forward`, with the KL of the bf16 reference against the fp32 run (mean and maximum over positions; `test_forward` recomputes the mean and checks it) and the number of positions where their greedy choices differ.

## Two deliberate choices

- **Eager attention.** The default `sdpa` fuses the softmax and both matmuls; the contract was written from `eager_attention_forward`, which is the readable path. The eager path rounds `q·kᵀ` to bf16 before the softmax; `dump.py` checks that keeping those scores in fp32 brings its output within 2 bf16 ulps of the fp32 rerun, which is why the native tolerance of `attn` is loose and its exact one is not (`tests/README.md`).
- **f32 tensors as stored.** `from_pretrained` rounds the 36 f32 tensors of the checkpoint (`linear_attn.A_log`, `linear_attn.norm.weight`) to bf16, with `dtype="auto"` as well as `dtype=torch.bfloat16`: the model has no `_keep_in_fp32_modules`. `dump.py` puts the stored f32 values back after loading; the forward code is unchanged (it upcasts `A_log` and promotes the norm weight). athanor uses the stored values, so both sides compute with the same weights. Measured on the prompt: against an fp32 run of the whole model, the bf16 reference with restored tensors has a mean KL of 5.4e-4 (max 1.4e-3); with the default rounding, 5.7e-4 (max 2.9e-3).
