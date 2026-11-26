# tests

Correctness is what lets everything else be fast.

| Level | Checks | When |
| --- | --- | --- |
| Unit | core modules; each op's `_ref` on hand-computed cases | every commit |
| Golden | each op of each layer vs tensors dumped from the reference implementation on real weights; the whole forward pass (residual stream at every layer, logits, KL) on fixed prompts | every commit (requires `tests/golden/data`) |
| Cross-backend | GPU kernels and `_fast` paths vs `_ref`, on random and on real inputs | every commit on GPU machines |
| End-to-end | greedy generation identical to reference for N tokens; KL on a fixed corpus | nightly |
| Gradient | each backward vs finite differences and reference autograd | phase 4 |
| Performance | bench subset; fails on regressions beyond a threshold | nightly, self-hosted |

## Golden data

`tests/golden/dump.py` (development only) runs the pinned reference implementation over a fixed prompt and one decode step and records every op it computes (the forward records of step 4 are described under [Forward pass](#forward-pass)): inputs, its output in its own dtype (`out`) and the same function recomputed in fp32 (`out32`), with the checkpoint names of the weights and the op parameters. The data lives in `tests/golden/data/` (ignored by git; `make golden` regenerates it in seconds). `tests/golden/README.md` has the environment, the record list and the two deliberate choices (eager attention, f32 tensors kept as stored).

`test_golden` replays each record through the `_ref` function twice:

- **native**: inputs as recorded (bf16), output in the recorded dtype, compared with `out`. It measures how far athanor is from what the reference computes, roundings included.
- **exact**: inputs widened to f32, output in f32, compared with `out32`. No bf16 rounding is left on either side: this is the check of the math.

## Tolerances

Every comparison reduces to one number, `score = max_i |got_i − want_i| / (|want_i| + rms(want))`: a relative error for elements of typical size or larger and an error relative to the tensor's RMS near zero, where a relative error means nothing. One bf16 ulp is at most 2^-7 (0.0078) relative.

athanor computes in fp32 and rounds each output once. A native tolerance follows from how many times the reference rounds to bf16 along the way: once (1 ulp: the two fp32 values may straddle a rounding boundary), two or three times (2 ulps: half an ulp per extra rounding). Exact tolerances are 1e-5, raised to keep at least a factor of 4 above the worst error measured where a long reduction or the recurrent state brings it close: `matmul` and `gdn` out 2e-5, `attn` 5e-5, the `gdn` state 1e-4. Measured values: Qwen3.5-0.8B, golden data of 2026-11-24 (transformers 5.24.0, torch 2.15.0, 906 records), worst over prefill and decode.

| Op | Output | Native tolerance | Measured | Exact tolerance | Measured | Why |
| --- | --- | --- | --- | --- | --- | --- |
| `embed` | out | 0 | 0 | — | — | a copy |
| `add` | out | 0 | 0 | 0 | 0 | one rounding of the same fp32 sum |
| `conv` | state_out | 0 | 0 | — | — | the state holds inputs |
| `matmul` | out | 1 ulp | 5.2e-3 | 2e-5 | 4.2e-6 | fp32 sum in another order, one rounding |
| `rmsnorm` | out | 1 ulp | 5.5e-3 | 1e-5 | 2.6e-6 | all fp32, one rounding |
| `rmsnorm_gated` | out | 1 ulp | 5.1e-3 | 1e-5 | 7.8e-7 | same intermediate rounding as the reference, then one |
| `gdn_gates` | beta | 1 ulp | 0 | 1e-5 | 1.1e-7 | sigmoid in fp32, one rounding |
| `gdn` | out | 1 ulp | 4.6e-3 | 2e-5 | 4.7e-6 | prefill: the reference uses the chunked form, same sums in another order |
| `swiglu` | out | 2 ulps | 7.6e-3 | 1e-5 | 2.0e-7 | the reference rounds silu, then the product |
| `sigmoid_mul` | out | 2 ulps | 7.3e-3 | 1e-5 | 2.1e-7 | the reference rounds the sigmoid, then the product |
| `conv` | out | 2 ulps | 8.4e-3 | 1e-5 | 1.6e-7 | the reference rounds the convolution, then silu |
| `rope` | out | 2 ulps | 9.4e-3 | 1e-5 | 2.8e-7 | the reference rounds both products and their sum |
| `l2norm` | out | 1e-5 | 6.7e-7 | 1e-5 | 6.7e-7 | f32 on both sides |
| `gdn_gates` | g | 1e-5 | 1.1e-7 | 1e-5 | 1.1e-7 | f32 on both sides |
| `gdn` | state_out | 1e-4 | 1.4e-5 | 1e-4 | 1.0e-5 | f32 state accumulating 75 tokens of updates |
| `attn` | out | 0.2 | 9.8e-2 | 5e-5 | 9.7e-6 | the eager reference rounds `q·kᵀ` to bf16 before the softmax: up to 2^-8 of a score, 0.06 at the largest score of the prompt (16.3), up to 6% on a probability; athanor keeps scores in fp32 |

A score above its tolerance fails the test and names the record and the element. `ATH_GOLDEN_WORST=1 ./build/test_golden` also prints the worst record of every row. Quantized formats (phase 2) will be compared through dequantized values, and with KL at model level.

## GPU kernels

Every GPU kernel (`ath_<op>_gpu`, step 7) is compared with its CPU reference twice: on random inputs in `test_gpu_ops`, and on the real inputs of every golden record in `test_golden`, which replays each record through the kernel as well as through `_ref` when the build has a GPU. The comparison is GPU against `_ref` on the same inputs, not against the reference implementation: the references are already held to that (Tolerances), and what is measured here is the kernel alone.

The measure is the score of the golden comparisons. f32 outputs are held to a tolerance per op, at least four times the worst error measured; bf16 outputs to one bf16 ulp (score at most 2^-7), two where the op rounds an intermediate value (rope's cos and sin tables, rmsnorm_gated's normalized value), since a flip there moves the output by up to 2^-8. The bf16 distance in ulps is reported too: near zero, a distance of many ulps is a tiny error relative to the tensor, which is why the score decides.

The kernels do not give the reference's bits, and are not meant to: they sum in another order (a warp's 32 partial sums, cuBLASLt's tiles), fuse multiply-adds, and the device's `expf`, `log1pf`, `cosf`, `sinf` round differently from libm. Copies are exact: `embed`, the conv state, and the residual `add` (one rounding of the same fp32 sum).

| Op | Output | Tolerance (f32) | Random inputs, worst (f32) | Real inputs, worst (exact) | Real inputs, worst (native, bf16) |
| --- | --- | --- | --- | --- | --- |
| `embed` | out | 0 | 0 | 0 | 0 |
| `add` | out | 0 | 0 | 0 | 0 |
| `conv` | state_out | 0 | 0 | 0 | 0 |
| `matmul` | out | 2e-5 | 1.9e-6 | 4.1e-6 | 7.4e-3 (1 ulp) |
| `rmsnorm` | out | 1.5e-5 (5e-6 random) | 1.2e-6 | 2.6e-6 | 5.5e-3 (1 ulp) |
| `rmsnorm_gated` | out | 1e-5 (5e-6 random) | 2.8e-7 | 7.8e-7 | 5.1e-3 (2 ulps) |
| `l2norm` | out | 1e-5 (5e-6 random) | 1.3e-7 | 6.1e-7 | f32 |
| `rope` | out | 1e-5 (1e-6 random) | 1.4e-7 | 3.2e-7 | 0 |
| `swiglu`, `sigmoid_mul` | out | 1e-5 (1e-6 random) | 1.1e-7 | 2.1e-7 | 0 |
| `conv` | out | 1e-5 (1e-6 random) | 1.7e-7 | 1.7e-7 | 4.3e-3 (1 ulp) |
| `gdn_gates` | g, beta | 1e-5 (1e-6 random) | 1.3e-7 | 1.9e-7 | 0 |
| `gdn` | out | 3e-5 (1e-5 random) | 8.8e-7 | 5.9e-6 | 4.7e-3 (1 ulp) |
| `gdn` | state_out | 5e-5 (1e-5 random) | 4.1e-7 | 1.2e-5 | f32 |
| `attn` | out | 5e-5 (1e-5 random) | 1.0e-6 | 1.1e-5 | 3.7e-3 (1 ulp) |

Measured 2026-11-25 on the RTX 5080 (CUDA 13.5, cuBLASLt 13.10), golden data of 2026-11-24. The real inputs are larger than the random ones where it matters: matmuls with K up to 3584, the recurrent state over 75 tokens, attention over 75 keys.

`test_gpu_ops` also checks, bit for bit, the properties the engine relies on: the convolution and the recurrence fed in pieces (4 + 1 + 5 and 5 + 1 + 6 tokens) equal the same tokens at once, and attention for the last token alone equals the last row of the prompt. The kernels sum in an order fixed by the shapes, so these hold on the GPU as on the CPU. The matmul is the exception: cuBLASLt chooses its algorithm from the number of rows, so a row of a prompt and the same row decoded alone may differ in the last bits (step 8 measures what this does to a whole forward pass).

Also covered: three dtype configurations per op (f32 throughout, bf16 throughout, bf16 in with f32 out), shapes that are not multiples of a warp or a block, a matmul operand at an address cuBLASLt cannot assume aligned, every matmul dtype combination (cuBLASLt for bf16·bf16 and f32·f32, athanor's own kernel for the rest), an embedding id outside the table (a row of NaN), the in-place residual add, and the argument checks: tensors on the wrong side, shapes, dtypes, and the kernels' own limits (`ATH_ATTN_MAX_DV` for the head dim, `ATH_GDN_MAX_DK`, `ATH_ROPE_MAX_DIM`). compute-sanitizer finds nothing in `test_gpu_ops`: memcheck with leak check, racecheck and initcheck report 0 errors.

```
ATH_GPU_OPS_DETAIL=1 ./build/cuda/test_gpu_ops     # every comparison with its measured error
CUDA_HOME=<toolkit> $CUDA_HOME/bin/compute-sanitizer --tool memcheck --leak-check full ./build/cuda/test_gpu_ops
```


### Decode kernels (step 9)

`test_gpu_ops` adds: `gpu_gemv` (every row count from 1 to 9, K from one piece to 3584, odd M, bf16 and f32 into bf16 and f32, a row alone against the same row among others bit for bit, an x off a 16-byte boundary that goes to the other kernel), `gpu_matmul_n` (several weights in one launch: the bits of one call each, for few rows and for many), `gpu_matmul_swiglu` (gate, up and SwiGLU in one launch: the bits of the three ops), `gpu_scatter_rows` (bit for bit), `gpu_sample` (360 cases against `ath_sample_ref`: vocabularies of 7 to 248,320, logits with and without ties, six samplers, ten values of u; 0 differ), and in `gpu_attn` positions over a noisy cache, rows against split-K bit for bit, decode against the prompt's last row bit for bit, NaN for a position past the cache. The tolerances are those of the step 7 kernels (2e-5 for f32 matmul outputs, one bf16 ulp; 1e-5 for attention), and the measured errors stay inside them: GEMV worst 1e-7 in f32 on the random cases; on the golden records attention 1.04e-5 (prefill) and 4.2e-6 (decode), Gated DeltaNet output 4.7e-6 and state 1.46e-5, matmul 4.2e-6.

### Prompt kernels (step 10)

The kernels of prompts (`src/ops/README.md`, Prompt kernels) are compared with their references like the others; where they compute differently, the tolerance says why.

- **Attention on the tensor cores** (`gpu_attn_mma`). The probabilities are rounded to bf16 before they multiply the values, a relative error of at most u = 2^-8 per weight. With weights w_j that add up to one (the kernel sums the rounded values), the output out = Σ w_j v_j moves by at most u/(1 − u) Σ w_j |v_j − out| ≤ 2^-8/(1 − 2^-8) (A + |out|), A the same attention over |v|. The test runs the reference twice, on v and on |v|, and checks every element against that bound plus the rounding of the output to bf16 (2^-8 |got|): the worst element is at 0.28 to 0.48 of its bound over five shapes (d 64, 128, 256; groups of 1, 3, 4 query heads per key head; a prompt after earlier positions, positions anywhere in a cache of 1000). The scale is 0.25, larger than a model's, so that the softmax is peaked and a key taken or missed by mistake would show. Pieces of 16, 74 and 60 queries over the whole cache give the bits of the 150 queries at once; a position outside the cache gives a row of NaN and leaves the other rows' bits. On the golden records (`test_golden`, native) the prompt's attention scores 8.05e-3 (1.03 ulp), so the native tolerance is 2 ulps for `attn`; the exact runs (f32) take the other kernels and stay at 1.04e-5.
- **Chunked Gated DeltaNet** (`gpu_gdn_chunk`): f32 against the recurrent reference, out within 1e-4 (6.7e-6 worst) and state within 1e-4 (4.6e-6 worst), on one chunk, a partial last chunk with the 0.8B head counts, 2100 tokens (more than one round), dims that are not multiples of 32 (which take the recurrent kernel: dk must be a multiple of 8) and dk ≠ dv; bf16 outputs within 2 ulps. 256 tokens at once against 128 + 64 + 64: the bits. On the golden records the prompt's recurrence (75 tokens, chunked) scores 6.0e-6 on the output and 1.35e-5 on the state, inside step 7's 3e-5 and 5e-5. q and k come not normalized: the op normalizes them (the references with `ath_l2norm_ref`'s expression, so the CPU's bits are those of the separate op before).
- **Fusions with the bits of their ops**: `gpu_add_rmsnorm` (the residual add in the norm's launch: x and out bit for bit against `ath_add_gpu` then `ath_rmsnorm_gpu`, bf16 and f32, rows of 1024, 128 and 64); the convolution in tiles of tokens keeps step 7's comparisons and its pieces test bit for bit; the norm with two or four rows to a warp keeps step 9's.

compute-sanitizer: memcheck with leak check on `gpu_attn_mma`, `gpu_gdn_chunk`, `gpu_add_rmsnorm`, `gpu_conv`, on `test_engine` (weights, state, pieces, decoder) and on a prompt of 512 tokens of the real model with 8 decoded tokens; racecheck on the attention, the chunked recurrence and the norms; initcheck on the attention and the chunked recurrence: 0 errors, 0 hazards, 0 bytes leaked.

### Long caches (step 14)

`gpu_attn_long`: the attention kernels over caches of 131,072 keys and more, the length of step 14's gate.
- Decode and split-K, with queries at positions 100, 65,536, 131,071 and 131,199 of a cache of 131,200 keys, are checked against the reference with the tolerance of short caches. Measured: score 4.6e-4 in bf16 (13 of 8192 values one ulp or more apart: the outputs are small, the mean of many keys); split-K gives the bits of rows.
- The tensor-core prompt kernel, with the last 16 positions of a 131,072-key cache, stays within its bound: 0.02 of it.
- An int8 cache of 131,200 keys (groups of 32) gives the bits of attention over what it reads back as, for one and four queries, rows and split-K.

RoPE at long positions was already checked (`gpu_rope`, positions up to 262,143: the frequencies are the reference's, computed on the host). The whole model at 128k: `tests/test_long.c` (Long context, below).

### Q8_0 (step 12)

No tolerance: every Q8_0 kernel follows the canonical order of `src/ops/q8_0.h`, so the comparisons are bit for bit. `test_ops`: `q8_0_f16_and_exp` (f16 conversion of normal, subnormal, infinite and NaN values; `ath_q8_expf` within 2 ulp of exp in double on [-87, 88), measured 0.89; the clamps; SiLU), `matmul_q8_0` (a hand case where the canonical sum is exact; six random shapes from K = 32 to 4096, x and out in f32 and bf16: the reference within K · 2^-24 Σ|w x| of the exact sum, worst 0.02 of that bound, and `ath_matmul_fast` on one and on three threads the reference's bits). `test_gpu_ops`: `gpu_matmul_q8_0` (seven shapes: one row, a few, 8, 21 (the prompt kernel), K = 96 and 160 (the kernel for other K), x off a 16-byte boundary, f32 and bf16 into both: the reference's bits; the Q8_0 embedding lookup bit for bit).

### GGUF (step 12)

`test_core`: `gguf_reads`, `gguf_rejects_bad_files` (files built in memory), `gguf_real_model` (the Qwen3.6-35B-A3B file when present). `test_model`: `qwen36_gguf_binds` (every tensor bound, per-expert views at the right offsets, the MTP block ignored; skipped without the file; `ATH_MOE_DIR` overrides `models/Qwen3.6-35B-A3B`). `test_forward`: `gguf_matches_safetensors`, Qwen3.5-0.8B from llama.cpp's bf16 GGUF against its safetensors (`docs/contract/qwen36-35b-a3b.md`, Verification): CPU bf16 the same bits, CPU f32 within 1e-4 (2.4e-5), GPU f32 within 1e-4 (6.0e-6), GPU bf16 the same greedy token wherever the top-2 margin exceeds 0.5 (logits up to 0.116 apart: CUDA's expf in A = −exp(A_log) moves some decays by an ulp, the converter's torch exp others). It needs a directory with the 0.8B's `config.json` and tokenizer beside the GGUF of step 9 (`convert_hf_to_gguf.py models/Qwen3.5-0.8B --outtype bf16`), by default `models/Qwen3.5-0.8B-gguf` (`ATH_GGUF_DIR`): for example symbolic links to `../Qwen3.5-0.8B/{config.json,tokenizer.json,tokenizer_config.json,vocab.json,merges.txt,chat_template.jinja}` and to the `.gguf` file.

### MoE (step 12)

The routed expert path follows one canonical order on the CPU and the GPU, so its comparisons are bit for bit; the router and the combination use each device's exp and are compared within tolerances. `test_ops`: `moe_route_hand` (the top 2 of 5 by hand, a tie), `moe_expert_and_combine` (an expert within 3.6e-7 of Σ|terms| of its definition in double; the combination by hand), `moe_experts_cpu` (AVX-512 and one lane: the reference's bits). `test_gpu_ops`: `gpu_moe` (the router: the same choices, weights within 4e-3, a bf16 ulp; the device's experts: the reference's bits; the combination within 1e-6 and an ulp), `gpu_moe_split` (the GPU and the CPU together, in a graph replayed after experts moved: the reference's bits). `test_engine`: `engine_moe_placement` (invariant 7 on Qwen3.6-35B-A3B: the same logits, bit for bit, with 73, 0 and 24 experts per layer in device memory and the slots changing between decoder steps; skipped without the GGUF).

## Forward pass

`test_forward` reaches the forward pass only through `src/model/model.h`, like a pipeline.

On tiny random checkpoints (`tests/tiny.h`): state specs and their sizes, every argument check of `ath_model_forward`, the hooks, and the property every pipeline relies on: a sequence run all at once, in pieces (5 + 1 + 6 tokens) or one token at a time gives the same logits and the same state, bit for bit, with bf16 and with f32 activations; the logits of the last token alone equal the last row; a reset state runs the sequence again from scratch.

On the real checkpoint, against the forward records of the golden data: two prompts (the op prompt, 75 tokens, and a chat-formatted one, 28 tokens), each followed by 8 greedy decode steps of the bf16 reference, and the same tokens through the reference loaded in fp32. athanor runs each prompt twice, the prompt in one call and then one token at a time, in four parallel threads that share one pool for the matmuls (about 11 s; 36 s before `ath_matmul_fast`, task 4.7, which changed no number below):

| Run | Check | Measured (2026-11-24) |
| --- | --- | --- |
| bf16 activations | **gate**: mean KL against the fp32 reference over every position of every prompt no larger than the bf16 reference's | 3.0e-4 against 5.0e-4 (ratio 0.60) |
| bf16 activations | **gate**: the greedy token equals the fp32 reference's wherever the fp32 top-2 margin exceeds the bf16 noise | 0 differences at 104 such positions; noise 0.21 |
| f32 activations | residual stream entering every layer and the final norm within 2e-4 of the fp32 reference | 4.7e-5 |
| f32 activations | logits within 1e-4 of the fp32 reference | 2.7e-5 |

- **KL.** KL(p₃₂ ‖ p) per position, in double precision, from the logits of the fp32 reference and of the run under test. The bf16 reference is held to the same measure; `test_forward` checks that its KL equals the one `dump.py` computed with torch.
- **Bf16 noise of a greedy decision.** For the two tokens the fp32 reference ranks first, the largest change of their logit difference between the fp32 and the bf16 reference over all positions. Where the fp32 margin is smaller than that, rounding alone can flip the choice, and the reference itself flips it twice on the prose prompt.
- **f32 tolerances.** Both sides compute in fp32 but sum in different orders (athanor's dot products in order of k, torch's blocked ones; the reference's prefill uses the chunked Gated DeltaNet). The errors of single ops are 1e-7 to 1e-5 (Tolerances, exact) and grow through 24 layers; the tolerances are about four times the worst measured value, as for the exact op tolerances.
- **Why the gate pools the prompts.** A mean over the positions of one short prompt is dominated by its worst-conditioned positions. On the chat prompt, position 0 (`<|im_start|>`) goes through a near-cancellation in layer 15 that turns one bf16 rounding into a 10% error, and its KL ranges from 2.7e-4 (the eager reference) to 5.5e-3 (the same reference with `A_log` rounded to bf16, the transformers default) and 7.3e-3 (athanor). Per prompt, athanor's mean KL is 1.7e-4 against 5.3e-4 on prose and 6.0e-4 against 4.4e-4 on chat (4.1e-4 against 4.5e-4 without position 0). The reference with sdpa attention, the transformers default, would fail a per-prompt comparison with the eager reference on chat too: 7.5e-4 against 5.1e-4 on the prefill. `docs/contract/qwen35-0.8b.md`, End-to-end, has the analysis.

The bf16 run also reports the residual stream at every layer against both references, as a diagnostic, not gated: against the fp32 run it is 2e-2 to 3e-2 after the first layer and at most 0.08 (prose) and 0.18 (chat), where the bf16 reference's own distance is at most 0.14 and 0.15.

```
./build/test_forward forward_golden                  # the real-checkpoint comparison only
ATH_FORWARD_POSITIONS=6 ./build/test_forward         # only the first 6 positions of each prompt (no gate)
ATH_FORWARD_LAYERS=1 ./build/test_forward golden     # residual stream per layer, every run
ATH_FORWARD_DETAIL=1 ./build/test_forward golden     # KL, margin and greedy choice per position
```

A `DEBUG=1` build runs the first 4 positions of each prompt (three in one call, one alone) and checks the f32 tolerances; the gate needs every position.

Step 10: the GPU forward records with the prompt kernels (attention on the tensor cores, the chunked Gated DeltaNet for the prose prompt's 83 tokens): pooled KL 4.399e-4 against the bf16 reference's 5.005e-4 (ratio 0.88; step 9 4.295e-4), 0 greedy differences above the noise; f32 KL 3e-12, logits within 8.8e-6, residual stream within 1.3e-5. The hooks of these tests watch the residual stream, so the forward pass adds each output projection in its own kernel here (the fused add waits for no hook).

### Forward records of the MoE models (step 12)

Two more sets of forward records, in `forward_golden`'s layout, replayed by pointing `ATH_MODEL_DIR` and `ATH_GOLDEN_DIR` at them:

- **Qwen3.6-35B-A3B** (`tests/golden/moe.py`, `tests/golden/data/qwen36-35b-a3b/`, 445 MiB, not committed): transformers 5.24.0's modules run one layer at a time over the GGUF, dequantized exactly, with the converter's transforms undone. There are 4 prompts (prose, chat, code, Italian), 234 positions, prompt positions only. `ATH_FORWARD_GPU_ONLY=1`: the CPU's reference ops would take hours on this model.
  - GPU bf16: pooled KL 1.811e-3 against the bf16 reference's 2.970e-3 (ratio 0.61); 0 greedy differences at the 180 positions above the noise (0.718).
  - GPU f32: KL 1e-12, logits within 9.6e-6 (tolerance 1e-4), residual stream within 3.5e-5 (tolerance 2e-4).
- **A tiny Qwen3.5-MoE** (`tests/golden/tiny_moe.py`, `tests/golden/data/tiny-moe/`): random weights, in the published safetensors layout. It has 3 prompts with 4 greedy steps each, 114 positions, run on the CPU and the GPU.
  - f32: KL 3e-13, logits within 5.7e-6, residual stream within 5.1e-6.
  - bf16: the greedy token is right above the noise; the KL is reported, not gated (`ATH_FORWARD_NO_KL_GATE=1`). On random weights it is chance whether athanor's rounding or the reference's lands closer to fp32: the ratio measured 0.14 to 1.65 over five seeds.

### Forward records of Qwen3.5-9B (step 14)

`tests/golden/gguf.py` is `moe.py`'s method for a dense GGUF: Qwen3.5-9B in Q8_0 (`docs/contract/qwen35-9b.md`), one layer at a time over the dequantized weights, the same 4 prompts and 234 positions; `tests/golden/data/qwen35-9b/` (514 MiB, not committed). Both devices run it (the CPU in about 9 minutes, the GPU in seconds). Measured 2026-11-26:

| Run | Check | CPU | GPU |
| --- | --- | --- | --- |
| bf16 | pooled KL against the fp32 reference; the bf16 reference: 3.829e-4 | 1.283e-4 (ratio 0.33) | 1.864e-4 (ratio 0.49) |
| bf16 | greedy token where the margin exceeds the noise (0.238) | 0 differences of 206 | 0 differences of 206 |
| f32 | residual stream, every layer (tolerance 4e-4) | 2.0e-5 | 8.5e-5 |
| f32 | logits (tolerance 5e-4) | 1.6e-5 | 1.1e-4 |

**Why this model has its own f32 tolerances** (`F32_TOL` in `test_forward.c`). The GPU's logits on the chat prompt measured 1.11e-4, over step 4's 1e-4, at position 20; its other positions stay within 4.7e-5, and the other prompts within 5e-6. The error per position and per layer (residual stream against the fp32 reference) shows where it comes from. Two positions are ill-conditioned: 20, the newline after `<|im_end|>`, and 21, the `<|im_start|>` after it. From layer 3 on, their error grows through Gated DeltaNet and attention layers alike, while every other position stays near 3e-6:

| Position | Device | Entering layer 6 | Layer 10 | Layer 20 | Layer 30 |
| --- | --- | --- | --- | --- | --- |
| 20 | GPU | 4.5e-6 | 1.3e-5 | 7.2e-5 | 8.5e-5 |
| 20 | CPU | 1.2e-6 | 2.7e-6 | 1.1e-5 | 1.4e-5 |
| 21 | GPU | 3.6e-6 | 3.4e-5 | 3.4e-5 | 3.7e-5 |
| 21 | CPU | 1.5e-6 | 6.5e-6 | 5.6e-6 | 4.8e-6 |

The two devices amplify at the same positions, by a factor of 4 to 20 over these layers. The GPU enters the amplifying layers with a larger rounding difference (its sums go in another order), and the amplification does the rest. That is a property of the model on this prompt, not a kernel's error: no residual value is large there (at most 21, as at the other positions), the prompt in one call and one token at a time give the same bits, and the KL stays at 1e-9. The tolerances follow the usual rule, about four times the worst measured value: 5e-4 for the logits and 4e-4 for the residual stream (8.5e-5). An error of any real kernel, such as a value rounded to bf16, is 1e-2 or more.

```
ATH_MODEL_DIR=models/Qwen3.5-9B ATH_GOLDEN_DIR=tests/golden/data/qwen35-9b ./build/cuda/test_forward forward_golden
```

### KV cache in int8 (step 14)

The int8 cache (`src/ops/README.md`, The KV cache in int8) is lossy by design, so it is held to a bound, not to the bits.

**The format.** The first version had one scale per row of 256 values, as decided at the start of the step. On the forward records it doubled the KL against the fp32 reference (0.8B on the GPU: 1.18e-3, against 4.4e-4 with the cache in bf16) and changed a greedy token above the bf16 noise. `tests/golden/kv_int8.py` measures what a cache does to the reference itself. It runs the bf16 reference with its keys and values replaced by what the cache reads back, and gives the mean KL against fp32 over `moe.py`'s 4 prompts (0.8B, 234 positions):

| Cache | Mean KL | Against the bf16 reference |
| --- | --- | --- |
| bf16 (the bf16 reference) | 5.26e-4 | 1.00 |
| int8, a scale per row of 256, keys and values | 1.28e-3 | 2.43 |
| the same, keys only | 5.29e-4 | 1.00 |
| the same, values only | 1.24e-3 | 2.36 |
| int8, groups of 64 | 7.82e-4 | 1.49 |
| int8, groups of 32 | 6.39e-4 | 1.21 |

Keys survive one scale per row; values do not: a few large values in a row coarsen the step for all the others. Groups of 32 (the block of Q8_0) take most of the loss away for 2 KiB more per token on the 9B (18 KiB instead of 16.25).

**The gate.** `forward_golden` runs each prompt a third time on each device: bf16 activations with the cache in int8 (`kv8` rows of the table). The run is held to the bf16 gate, with the bound the bf16 reference itself needs with the same cache. Its pooled KL must be at most 1.25 times the bf16 reference's (1.21 measured above, with margin), and the greedy token must be the fp32 reference's wherever the margin exceeds the bf16 noise. The int8 run is also compared with the bf16-cache run directly: mean KL between the two and greedy agreement, reported. Measured 2026-11-26:

| Model, device | Pooled KL, int8 cache (bf16 cache) | Against the bf16 reference | Greedy above the noise | int8 against the bf16 cache: KL, the same greedy token |
| --- | --- | --- | --- | --- |
| 0.8B, CPU | 3.39e-4 (3.02e-4) | 0.68 | 0 differences of 104 | 4.9e-4, 118 of 119 |
| 0.8B, GPU | 4.95e-4 (4.40e-4) | 0.99 | 0 differences of 104 | 4.0e-4, 117 of 119 |
| 9B, GPU | 1.78e-4 (1.86e-4) | 0.47 | 0 differences of 206 | 1.7e-4, 232 of 234 |
| 9B, CPU | 1.78e-4 (1.28e-4) | 0.46 | 0 differences of 206 | 1.6e-4, 231 of 234 |
| Qwen3.6-35B-A3B, GPU | 1.86e-3 (1.81e-3) | 0.62 | 0 differences of 180 | 1.9e-3, 232 of 234 |

`test_engine` (`engine_decoder`) runs the recorded decode step over an int8 cache too: the tokens, and on the host the logits, of the same steps run directly, bit for bit.

### Long context (step 14)

`tests/test_long.c`, opt-in (`ATH_LONG=1`, minutes): a real model at a context that a prompt fills (`ATH_LONG_CTX`, default 131,072; `ATH_MODEL_DIR`). The prompt is some 5,800 short notes, with a passphrase a tenth of the way in and the question at the end. For the KV cache in bf16 and in int8:
- the greedy answer names the passphrase;
- the prompt in pieces of `ATH_LONG_CHUNK` and of 5/8 of it (whole chunks of 64) gives the same state and the same logits, bit for bit;
- the int8 cache against the bf16 one at that depth: the bf16 answer's tokens fed to both, mean KL at most the bf16 noise (3.8e-4, the 9B's bf16 reference against fp32 on the forward records);
- reported: prefill and decode times, device memory, the snapshot's size at 8k, 32k, 64k and the full length, the time to copy it out, write it and read it back.

Qwen3.5-9B, 2026-11-26 (`docs/results/2026-11-26-step14-long-context/`): a prompt of 130,524 tokens, found with both caches, the same bits for pieces of 1024 and 640, int8 against bf16 KL 1.1e-5 with 24 of 24 greedy tokens.

```
ATH_LONG=1 ATH_LONG_CHUNK=1024 ATH_MODEL_DIR=models/Qwen3.5-9B ./build/cuda/test_long
```

## Engine

Step 8 runs the forward pass on the GPU through the engine (`src/infer/engine.h`). Three checks hold it to the reference.

**The forward records, on the GPU too.** `test_forward`'s `forward_golden` runs its four runs again on the GPU after the CPU ones, and holds them to the same gate and tolerances (Forward pass, above). Measured 2026-11-25:

| Run | Check | CPU | GPU |
| --- | --- | --- | --- |
| bf16 | mean KL against the fp32 reference, every position of both prompts; the bf16 reference: 5.0e-4 | 3.0e-4 (ratio 0.60) | 2.4e-4 (ratio 0.49) |
| bf16 | greedy token against the fp32 reference where the margin exceeds the noise (0.21) | 0 differences of 104 | 0 differences of 104 |
| f32 | residual stream, every layer (tolerance 2e-4) | 4.7e-5 | 1.5e-5 |
| f32 | logits (tolerance 1e-4) | 2.7e-5 | 9.4e-6 |

The GPU's f32 run lands closer to the fp32 reference than the CPU's. A likely reason, not measured separately: its sums round less (fused multiply-adds, partial sums added in a tree, as torch's blocked kernels also do), where the CPU references add one term at a time, in order, as their contract asks.

**The gate: greedy generation.** `test_engine`'s `generate_golden` reads `generate.json` (`tests/golden/generate.py`): 50 chat prompts, 256 greedy tokens each from the fp32 reference, the top-2 margin of every step, and each prompt's bf16 noise (the largest change of that margin in the bf16 reference). The engine runs each prompt in bf16 and, at every step, picks its own token and is then fed the reference's (teacher forcing), so every one of the 12,800 steps is checked, also after a near-tie. Where the margin exceeds the prompt's noise the two tokens must be equal: rounding alone cannot flip a choice there, so a difference would be an error. (The largest noise over all prompts, as the forward records use for their two, would check fewer steps: one prompt's noise is 1.237, the others' 0.17 to 0.62. It is reported too.) Below it they may differ, and the report counts how often. As long as the choices agree, teacher forcing is generation, so the report also gives how many tokens the engine generates exactly as the reference before its first difference.

Measured 2026-11-25 (RTX 5080, bf16; reference: transformers 5.24.0, fp32, eager attention):

| Check | Result |
| --- | --- |
| steps whose margin exceeds the prompt's noise (the gate) | 11,445 of 12,800; the engine's token differs at **0** |
| steps below the noise | 1,355; the engine's token differs at 78 |
| the same with the largest noise of all prompts (1.237; the others 0.17 to 0.62) | 8,352 steps above it; 0 differ |
| free-running generation | 15 of 50 prompts equal the reference for all 256 tokens; 6,572 of 12,800 tokens equal before each prompt's first difference |
| the bf16 noise per prompt | 0.171 to 0.619, and 1.237 for one prompt |

The first difference of a prompt is always at a near-tie, and the 0.8B model meets near-ties often: several prompts part at their first or second token (the Italian letter at token 0, where the reference's top-2 margin is 0.023; the C linked list at token 1, margin 0.017), others agree for all 256. Every prompt's run takes about a second (282 tokens/s decoding, for the record), the 50 about 55 s.

**Pieces and devices.** `test_engine` on tiny random checkpoints: the weights on the device byte for byte, states on the device (reset, copy, a pass refused on the other side), and 12 tokens at once, as 5 + 1 + 6 and one at a time on the GPU in f32, against each other (at most 1e-5; measured: the same bits) and against the CPU (at most 1e-4, the tiny model being ill-conditioned; measured 9.8e-6).

```
./build/cuda/test_engine generate                            # the gate only
ATH_GENERATE_DETAIL=1 ./build/cuda/test_engine generate      # equal tokens per prompt
ATH_GENERATE_PROMPTS=5 ATH_GENERATE_STEPS=32 ./build/cuda/test_engine generate
```

A `DEBUG=1` build runs 2 prompts of 8 steps.


Step 9 (`engine_decoder`): the decoder's recorded and replayed steps give the logits of the same steps run directly, bit for bit, and with the same uniform numbers the tokens `ath_sample_ref` chooses, for host sampling (top-k 0), GPU sampling (top-k 3) and greedy; a full state is refused; a reset state decodes again with the same recording. `engine_pieces` now allows 5e-5 between a prompt fed in pieces and at once on the tiny random model (1.22e-5 measured): pieces of up to 8 tokens take the GEMV kernel and the decode fusions, the whole prompt cuBLASLt, and the random network amplifies the difference of their roundings (step 4 found it ill-conditioned). Against the CPU it is 2.8e-6. The real checkpoint's gates are the measure: 0 differences in 11,445 steps above the noise, and the GPU forward records at a pooled KL of 4.295e-4 (the bf16 reference's 5.005e-4; 2.4e-4 at step 8 and 2.477e-4 with the first step 9 kernels: the change is rounding luck on the ill-conditioned positions, as the f32 run shows: KL 3e-12, logits within 9.5e-6).

## Prefix snapshots (step 14)

`src/infer/prefix.h` keeps snapshots of a sequence at prefixes of its tokens, in host memory and on disk. Checked by:

- `test_prefix` (CPU, tiny random models):
  - `prefix_snapshot_roundtrip`: a state saved after 20 tokens has the size the specs give. Loaded into a larger state, it gives the logits of 6 more tokens bit for bit, with the KV cache in bf16 and in int8. A wrong size and a state too small for its rows are refused.
  - `prefix_store_memory`: the longest prefix shorter than the prompt; the minimum length; a prefix saved twice kept once; the budget in host memory, where the least recently used leaves; a loaded snapshot goes on as the sequence it was taken from.
  - `prefix_store_shared`: a shared prefix is written even though a longer snapshot continues it, and stays not superseded after a reopen.
  - `prefix_store_disk`: nothing is written before it must be. At close, only a conversation's latest turn is written, not the turns it supersedes. After a reopen, that turn read from disk goes on with the same logits. A branch point (a prefix continued two ways) is written with the rest. The folder's budget keeps the most recently used file. Files of another model or other dtypes are left alone and not used. A file whose payload was damaged passes the scan, is refused at load with the state untouched, and is deleted; a truncated file is not used.
- `test_engine`, `engine_snapshots`: the same round trip for states on the device, bf16 and int8 KV; the snapshot of a device state loads into a host state too.
- `test_net`, `serve_snapshots_resume` (Qwen3.5-0.8B): a server opened on a snapshot folder answers a conversation with a system prompt of 1112 tokens and stops. A new one on the same folder starts the same conversation from the snapshot on disk (1136 prompt tokens cached) and gives the same greedy answer. A third starts a new conversation with the same system prompt from the shared part (1112 tokens cached).

## Text

`tests/golden/text.py` records, with the pinned reference (transformers 5.24.0, tokenizers 0.23.3), what `test_text` compares exactly:

- **Tokenizer.** For 188 texts, 28 million tokens in all: the ids the reference gives and the text it decodes them to. Hand-written cases (languages, scripts, emoji sequences, code, every kind of white space, contractions in every case, added tokens inside text, normalization), files of this repository and the model card, every Unicode code point but the surrogates in five contexts (`a·`, ` ·`, `·1`, `·!`, `·'`), and every canonical decomposition, decomposed, with up to three extra marks in shuffled order. athanor must give the same ids and decode them to the same bytes: no tolerance.
- **Chat template.** 20 conversations rendered by `apply_chat_template`: with and without system, thinking on and off, history with thinking in the content or apart, tools with and without system, tool calls with typed arguments and their results, tool responses written as user messages, trimming, Unicode; 3 of them the template rejects. athanor must render the same bytes (and give the same ids), or fail where the template raises.
- **Generation.** A chat prompt, 32 greedy steps of the fp32 reference with the top-2 margin of each step, and the bf16 noise of that margin (its largest change from fp32 to the bf16 reference on the same tokens). athanor, with bf16 activations and its own template and tokenizer, must choose the reference's token at every step until the first one whose margin is within the noise (step 27 of 32 for this prompt); after it the two may part legitimately, and the test reports how many tokens are equal in all (32). A `DEBUG=1` build runs 4 steps.

## Runner

A minimal C runner: `ATH_TEST(name) { ... }`, `ATH_EXPECT_*` assertions, a summary line, non-zero exit on failure. No external test framework.

`tests/test.h` holds the runner (header-only); `tests/util.h` the helpers the suites share (scratch directory, location of the real checkpoint); `tests/tiny.h` the tiny synthetic Qwen3.5 checkpoints (`config.json` and safetensors, zero-filled or pseudo-random, with mutations for the failure tests) that `test_model` and `test_forward` share. Each suite is one `tests/test_<area>.c` built into `build/test_<area>` with a table of `ATH_TEST_ENTRY(name)` at the bottom. Assertions: `ATH_EXPECT` (record and continue), `ATH_REQUIRE` (record and end the test), `ATH_EXPECT_OK`, `ATH_EXPECT_STATUS`, `ATH_EXPECT_EQ_INT`, `ATH_EXPECT_EQ_U64`, `ATH_EXPECT_STR_EQ`, `ATH_EXPECT_NEAR`, and `ATH_SKIP(why)`.

```
make test                          # all suites, smoke runs of athanor inspect (with and without --roles), and make check-alloc; about 17 s
make DEBUG=1 test                  # same suites with ASan and UBSan, in build/debug; about 4 minutes
make cuda test                     # every suite of the CUDA build (build/cuda); test_backend and test_gpu_ops run
                                   # on the GPU, test_golden replays every record on it too
make cuda DEBUG=1 test             # the same with ASan and UBSan (build/cuda-debug)
                                   # (test_golden 2.6 minutes there, 5 s in release; test_forward 34 s on 4 positions, 11 s on all;
                                   # test_text 3 s in release, its generation 4 steps instead of 32 there)
./build/test_core json             # only tests whose name contains "json"
ATH_LOG=error ./build/test_core    # show the library's error messages (silenced by default)
```

Synthetic files (safetensors shards, indexes, broken headers) are written by the tests into a fresh directory under `$TMPDIR` and removed afterwards. Tests that need real weights read `$ATH_MODEL_DIR` (default `models/Qwen3.5-0.8B`, set in the Makefile) and are skipped when it is absent; except `test_golden`, they check format-level invariants, so any checkpoint directory works. `test_golden` reads `$ATH_GOLDEN_DIR` (default `tests/golden/data/qwen35-0.8b`) and needs the checkpoint the data was made from.

| Suite | Covers |
| --- | --- |
| `test_core.c` | status and logging, arenas, files, JSON (synthetic and real `config.json`, `tokenizer_config.json`, `tokenizer.json`), dtypes and tensor views, safetensors (single file, sharded, 20 malformed headers, inconsistent indexes, the real checkpoint), the thread pool (every task once, with fewer, as many and more tasks than threads; runs from four threads at once on one pool) |
| `test_model.c` | role names and the registry; qwen35 config parsing (dense, MoE, text-only, schedule fallback, 20 rejected configs); binding of tiny synthetic checkpoints (dense, MoE with fused experts, text-only) and every failure mode (missing, unexpected, misshaped, integer dtype, tie mismatches); the real checkpoint (coverage, schedule against `layer_types`, residual writers) |
| `test_ops.c` | every `_ref` function of `src/ops/` on hand-computed cases (bf16 rounding, norms, RoPE pairing and relative positions, attention averaging and head grouping, conv taps, the delta rule step by step, gates around the softplus threshold) and on exact properties: streaming equals all at once for conv and Gated DeltaNet, decode equals the last prefill row for attention, causality, argument checks; `ath_matmul_fast` bit for bit against `ath_matmul_ref` (7 shapes × 8 dtype combinations, without and with threads) |
| `test_golden.c` | every op the reference computes on Qwen3.5-0.8B (24 layers, prefill and decode, 906 records), replayed through the `_ref` functions on the real weights, native and exact (above); every matmul record also through `ath_matmul_fast` on 8 threads, required bit-identical to `ath_matmul_ref`; with a GPU, every record also through the GPU kernel, against `_ref` (GPU kernels) |
| `test_forward.c` | the forward pass through the contract: state specs and sizes, argument checks, hooks, all at once = in pieces = token by token on tiny random checkpoints; the real checkpoint against the forward records (above), on the CPU and, with a GPU, on the GPU through the engine (Engine) |
| `test_gpu_ops.c` | every GPU kernel against its `_ref` on random inputs (GPU kernels, above): three dtype configurations, odd shapes, a misaligned matmul operand, the matmul's two paths, pieces equal all at once bit for bit, argument checks and the kernels' limits. In the CPU build only the argument checks run |
| `test_engine.c` | the model on a GPU (Engine, above): in the CPU build, that the engine needs a GPU; on the GPU, the weights the engine copies, states on the device, sequences in pieces against each other and the CPU, and the gate, greedy generation on 50 prompts of 256 tokens against the fp32 reference |
| `test_backend.c` | the GPU runtime (`src/backend/backend.h`). In the CPU build: the stub refuses everything cleanly, the GPU tests are skipped. In the CUDA build, on device 0: properties, copies in every direction at odd sizes and offsets from pinned and ordinary memory, memset, three streams ordered only by events (8 rounds overwriting each other's buffers), event timing with a sanity range, an allocation larger than the device, the device arena, a strided 2D copy (step 8); every byte checked |
| `test_text.c` | text, through the contract only: UTF-8, character classes, NFC, the Python-style JSON writer, pre-tokenization pieces, the output parser (17 fixtures, each also split at every byte and fed byte by byte), template errors, sampling (hand-computed cases and a distribution with a fixed seed); against the text golden data (below): the corpus, the templates, the greedy chat generation |
