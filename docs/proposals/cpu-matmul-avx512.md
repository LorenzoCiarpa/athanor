# Proposal: an AVX-512 CPU matmul with the reference's bits

Status: **written down, not scheduled.** On 2026-11-24 the maintainer chose the portable fast path (`ath_matmul_fast`, task 4.7 in `docs/STATUS.md`) and asked to keep this variant on file. It may never land. If it does, it needs its own task, and the decisions listed at the end.

## The idea

A dot product `out[m] = Σ_k x[k] · w[m, k]` can use SIMD in two ways:

| Direction | What one register holds | Order of the sum | Same bits as `ath_matmul_ref` |
| --- | --- | --- | --- |
| along k (BLAS libraries, llama.cpp, `vdpbf16ps`) | 16 consecutive k of one output | 16 partial sums, then a tree | no |
| across outputs (this proposal) | the same k of 16 different outputs | each lane is one chain, in order of k | yes |

`ath_matmul_fast` (variant B) gets its speed from independent chains too, but in scalar code: 8 outputs per thread. Variant C does the same with AVX-512, where every lane of a register is one output: 16 outputs per register and 8 registers, so 128 outputs per step. Each lane still computes a rounded product, then a rounded sum, in order of k. The kernel uses `_mm512_mul_ps` followed by `_mm512_add_ps`, never a fused multiply-add. For bf16 × bf16 a fused multiply-add would give the same bits, because the product is exact in fp32; for f32 activations it would not. The results are therefore identical to `ath_matmul_ref` and to `ath_matmul_fast`, with any number of threads.

## Layout

A register must load the same k of 16 rows with one instruction, but in the checkpoint layout `[M, K]` those values are `K` elements apart. The weights are therefore repacked once, at load time, into tiles of 16 rows interleaved by k:

```
packed[(t · K + k) · 16 + i] = w[16 t + i, k]        tile t, lane i
```

The last tile is padded with zero rows, whose outputs are never stored. The alternative without repacking is to load 16 rows × 16 k and transpose them in registers. It saves the copy but costs shuffles and code, and it was not measured.

## Code

The code below was checked against `ath_matmul_ref` on 2026-11-24, bit for bit: 24 cases with N up to 4, M up to 3584 including partial tiles and groups, K up to 1024, x in bf16 and f32, and output in bf16 and f32. It was compiled as part of `src/ops/matmul.c` (it uses `ld` and `st` from `src/ops/check.h`) with `gcc -std=c11 -O2`.

```c
#include <immintrin.h>

enum { LANES = 16, GROUP = 8 };   /* outputs per register; registers per group (128 outputs) */

/* Tile t holds outputs 16t .. 16t + 15, interleaved by k: tile[k * 16 + i] = w[16t + i, k].
 * Rows past M (the last tile) are zero; their outputs are never stored. */
static void pack_bf16(const uint16_t *w, int64_t M, int64_t K, uint16_t *packed)
{
    const int64_t tiles = (M + LANES - 1) / LANES;
    memset(packed, 0, (size_t)(tiles * K * LANES) * sizeof *packed);
    for (int64_t m = 0; m < M; m++)
        for (int64_t k = 0; k < K; k++)
            packed[((m / LANES) * K + k) * LANES + m % LANES] = w[m * K + k];
}

/* 16 bf16 values widened exactly to f32: shift into the high half of each lane. */
__attribute__((target("avx512f,avx512bw")))
static inline __m512 widen16(const uint16_t *p)
{
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)p)), 16));
}

/* Outputs of tiles t0 .. t1 − 1 for every row of x. Lane i of register g is
 * output 16(t + g) + i: one chain per lane, in the order of k, a rounded
 * product and a rounded sum per step (mul, then add: no FMA). */
__attribute__((target("avx512f,avx512bw")))
static void tiles_avx512(const ath_tensor *x, const uint16_t *packed, ath_tensor *out, int64_t N, int64_t M,
                         int64_t K, int64_t t0, int64_t t1)
{
    for (int64_t t = t0; t < t1; t += GROUP) {
        const int g_n = t1 - t < GROUP ? (int)(t1 - t) : GROUP;
        for (int64_t r = 0; r < N; r++) {
            __m512 acc[GROUP];
            for (int g = 0; g < GROUP; g++)
                acc[g] = _mm512_setzero_ps();
            for (int64_t k = 0; k < K; k++) {
                const __m512 xv = _mm512_set1_ps(ld(x, r * K + k));
                for (int g = 0; g < g_n; g++)
                    acc[g] = _mm512_add_ps(acc[g], _mm512_mul_ps(xv, widen16(packed + ((t + g) * K + k) * LANES)));
            }
            float o[GROUP * LANES];
            for (int g = 0; g < GROUP; g++)
                _mm512_storeu_ps(o + g * LANES, acc[g]);
            for (int64_t i = 0; i < (int64_t)g_n * LANES && t * LANES + i < M; i++)
                st(out, r * M + t * LANES + i, o[i]);
        }
    }
}
```

The threads would split the tiles in groups of 8 through the same `ath_pool` as `ath_matmul_fast`. For prefill, the next step is register blocking over rows: for example 4 rows × 4 registers, so that every weight vector loaded serves 4 tokens.

## Measurements

These were measured once on 2026-11-24 with a microbenchmark outside the repository: Ryzen 7 9800X3D (8 cores, AVX-512, 96 MB L3), DDR5, gcc 15.2 `-O2 -std=c11`. Inputs were bf16, output f32, N = 1 (one token, as in decode), best of several runs. This is not an `athanor bench` report (`docs/BENCHMARKS.md`), so it is a record, not a performance claim.

| Variant | 3584 × 1024 (fits in L3) | 248320 × 1024 (lm_head, 508 MB) | Same bits |
| --- | --- | --- | --- |
| `ath_matmul_ref` | 2.03 ms | 142 ms | — |
| B, 1 thread (microbenchmark / in `src/ops/matmul.c`) | 3.4× / 3.5× | 2.9× / 3.4× | yes |
| B, 8 threads (microbenchmark / in `src/ops/matmul.c`) | 14.9× / 12.3× | 15.1× / 16.0× (8.8 ms) | yes |
| C, 1 thread | 15.9× | 7.4× | yes |
| C, 8 threads | 30.7× | 16.3× (8.7 ms) | yes |
| along k with `vdpbf16ps`, 8 threads | 35.6× | 18.4× (7.7 ms) | no: 83% of the outputs differ, up to 5e-2 relative where the sum nearly cancels |

With 8 threads the lm_head runs at about 58 GB/s with B or C, close to what the memory delivers. Decode streams every weight from RAM once per token, so C gains almost nothing there over B. C wins where the work is compute-bound: weights that fit in cache, a single thread, and prefill. There, for the same rows the two variants measured 2× apart, and C would gain more with row blocking.

## Costs

- **Memory.** The packed copy is a second set of weights, about 1.5 GB for Qwen3.5-0.8B. `src/core/README.md` allows no hidden copies of weights, so placing them is an engine decision. Where packed weights live and who owns them needs a contract decision, for example a per-param packed view kept by the engine.
- **Load time.** Repacking touches every weight once. Not measured.
- **Portability.** Needs AVX-512 (`avx512f`, `avx512bw`): AMD Zen 4 and Zen 5, and Intel server parts. Many recent Intel desktop CPUs do not expose AVX-512. It needs runtime detection (`__builtin_cpu_supports`, the planned role of `src/backend/cpu.c`) and a fallback, which is B. An AVX2 version (8 lanes) would cover most other x86 CPUs.
- **Code.** Intrinsics, packing, partial tiles and groups, dispatch: about three times the code of B.
- **Tests.** The same as B: `memcmp` against `ath_matmul_ref` in `tests/test_ops.c` and on the golden records, with f32 inputs too. On a machine without AVX-512 the test can only run the fallback, and must say so.

## When it would be worth it

- CPU prefill matters. It does not in the plan: the GPU does prefill from step 7 on.
- A CPU path must be fast on one thread, or while the other threads are busy.
- Step 12 (MoE experts computed on the CPU). There the weights are quantized (phase 2 formats), so the kernel would be a different one. The idea carries over: lanes across outputs, packed tiles. With integer formats the order of the integer sums does not matter, which changes what "same bits" requires.

## Decisions it would need

1. Ownership and placement of the packed copy (contract, engine).
2. Whether the forward pass picks C automatically when the CPU supports it, or behind an option.
3. Whether to also write the AVX2 version, and the row blocking for prefill.
