# The Gated DeltaNet recurrence on the GPU, and why step 9 rewrote it

An explainer for article 10 of the course (`PROGRAM.md`, "Chunkwise Gated DeltaNet on the GPU"). It covers the recurrent kernel in `src/ops/cuda/gdn.cu`, what step 9 changed in it, and what that cost.

## What the kernel computes

Qwen3.5 has two kinds of token mixer. Six layers of the 0.8B model use full attention, with a KV cache that grows by one row per token. The other eighteen use a **Gated DeltaNet**: a linear-attention layer whose memory is a **fixed-size state**, the same size whatever the length of the sequence. For each value head h the state is a matrix S of dk × dv numbers in fp32: 128 × 128 for the 0.8B model, 16 value heads per layer, so 1 MiB per layer.

For each token, with its query q and key k (both L2-normalized, `docs/contract/qwen35-0.8b.md`), its value v, its decay g and its write strength β, the recurrence (`gdn.c`, the reference) is:

```
S ← α · S              α = exp(g): the state fades a little at every token
u = Sᵀ k               what the state already returns for the key k
δ = β · (v − u)        the correction: the delta rule
S ← S + k ⊗ δ          write the correction along k
o = Sᵀ (q · scale)     the output, read with the query
```

Two properties decide the shape of a GPU kernel:

- **Columns are independent.** Column j of S (fixed j, all dk rows) only ever meets element j of v, δ and o. Different columns never mix, so they can go to different threads with no communication.
- **The sums run along the rows.** `u_j = Σ_i S[i][j] k[i]` and `o_j = Σ_i S[i][j] q[i]` add over the dk rows of a column. Whoever holds a column's rows must add them up.

The state is stored row by row, `S[h][i][j]` at `(h · dk + i) · dv + j` (the layout of the reference). Elements of one row are next to each other in memory; elements of one column are dv · 4 = 512 bytes apart.

## The step 7 kernel: a warp per column

The first GPU kernel gave each column to one warp and split its rows over the 32 lanes: lane l held rows l, l + 32, l + 64, l + 96 in registers. Each lane added its four products, and a warp reduction (`k_warp_sum`, five shuffles) gave u, then the same again for o. The state went back to memory once, at the end of the call.

The arithmetic was right and the reductions cheap. Memory was the problem. When a warp loads its column, its 32 lanes ask for 32 addresses 512 bytes apart: 32 different cache sectors, of which each lane uses 4 bytes. A block held 8 neighbouring columns, so the same sectors served 8 warps through L1, but the number of memory transactions stayed high. The writes at the end were scattered the same way. At decode, one token per call, the kernel took about 7.3 µs per layer for 2 MiB of state traffic.

## The step 9 kernel: a lane per column

Step 9 turned the mapping around:

```
step 7:  warp = one column,     lanes = its rows      → reads 512 bytes apart
step 9:  lane = one column,     warps = the rows      → one 128-byte read per row
```

A block takes 32 neighbouring columns of one head, one per lane. When its 32 lanes read row i, they read `S[h][i][j0 … j0 + 31]`: 32 consecutive floats, one 128-byte access, every byte used. The block's 4 warps split the rows: warp w keeps rows w, w + 4, w + 8, … of its lane's column, 32 rows for dk = 128, in registers for the whole call.

Per token (`gdn_kernel` in `src/ops/cuda/gdn.cu`):

1. The block loads the token's k and q · scale into shared memory (`kq`), once for all its warps.
2. Every warp decays its rows (`s *= α`) and adds its part of `u = Sᵀk` over them.
3. The four parts go to shared memory (`part`), and after a barrier each lane adds them **in warp order**: part 0, then 1, 2, 3.
4. Each lane computes `δ = β (v_j − u)` for its column, writes `k ⊗ δ` into its rows, and adds its part of `o = Sᵀq`.
5. The parts of o are added in warp order the same way, and warp 0 writes the output.

At the end of the call the rows go back to memory, again one 128-byte write per row.

### Why the order of the sums is fixed

Floating-point addition is not associative: adding the same numbers in another order can change the last bits. The kernel fixes every order by dk alone. Within a warp, the rows of each lane are added in increasing order; across warps, parts 0, 1, 2, 3. Nothing depends on how many tokens the call has or how the grid is launched.

That gives the property step 7 asked of these kernels (criterion 3): **feeding a prompt in pieces gives exactly the bits of feeding it at once**, because every token sees the same state, updated by the same operations in the same order. `tests/test_gpu_ops.c` checks it (5 + 1 + 6 tokens against 12 at once), and step 9's decoder relies on it: a decoded token gives the bits of the last row of a prompt that ends with it.

## What it cost: prefill

The same kernel runs for one token (decode) and for a whole prompt (prefill). Using a different kernel for prompts would have changed the order of the sums between the two, and the bits of pieces against the whole with it. So step 9 kept one kernel, and paid for it in prefill:

- **Few blocks.** For the 0.8B model there are 16 heads × 128 / 32 = 64 blocks of 4 warps, against 16 × 128 / 8 = 256 blocks of 8 warps before. Fine for one token; little parallelism for thousands.
- **A barrier chain per token.** A prompt is walked one token after the other, with four block barriers per token.

The measurements (`docs/results/2026-11-25-step9-decode/`): prompts became slower than at step 8, pp512 from 19.8k to 17.6k tokens/s and pp4096 from 10.6k to 8.6k. The Gated DeltaNet change is about half of that: pp512 measured 18.6k just before it and 17.6k just after. The other half came with the earlier step 9 changes to the forward pass and attention (keys and values written by position, attention over the whole cache in pieces of 64). For decode the whole step went from 404.5 to 408.3 tokens/s with this change, about 1%; the kernel's own time after the change was not measured separately.

## What comes next: the chunked form (step 10)

A prompt does not need the state after every token, only the outputs. The **chunkwise** form of the Gated DeltaNet (the one flash-linear-attention and llama.cpp implement) cuts the prompt into chunks of, say, 64 tokens. Inside a chunk, the effect of all its tokens on each other and on the state can be written as products of small matrices, which run on tensor cores and in parallel. Only the hand-over of the state from one chunk to the next stays sequential: 64 times fewer sequential steps than token by token.

Step 10 brings that form for prompts. The column-per-lane kernel stays for decode, where one token per call is exactly the case it was written for. The recurrent CPU reference in `gdn.c` stays the truth that both are checked against.

## Where to look

| File | What is there |
| --- | --- |
| `src/ops/gdn.c` | the reference: the recurrence above, one token at a time, in the reference's order |
| `src/ops/cuda/gdn.cu` | the gates kernel and the recurrence kernel described here; the header comment says the same in brief |
| `src/ops/README.md` | GPU kernels (step 7) and Decode kernels (step 9) |
| `tests/test_gpu_ops.c` | `gpu_gdn`: against the reference on random inputs, and pieces against the whole, bit for bit |
| `tests/test_golden.c` | every Gated DeltaNet record of the real model, prefill and decode, against the reference |
| `docs/contract/qwen35-0.8b.md` | the layer as the reference computes it: gates, L2 norm, the state's layout |
