/* moe_cpu.c: routed experts computed on the CPU (step 12). See ops.h.
 *
 * At batch 1 a token reads 8 experts per layer, 3.3 MB each in Q8_0, and
 * most of them are not in device memory: the CPU computes those where they
 * are, from RAM, while the GPU computes the others (src/ops/moe.c). Both
 * follow the canonical order of src/ops/q8_0.h, so the bits are the same
 * whoever computes an expert (AGENTS.md invariant 7, option (b)).
 *
 * The canonical order gives every block one partial sum P[b mod 16] and runs
 * a chain of 32 fused multiply-adds inside it. On the CPU the 16 partial
 * sums are the 16 lanes of an AVX-512 register: lane j runs the chain of
 * block 16 g + j. For that the weights are regrouped once, when they are
 * read into memory (ATH_DTYPE_Q8_0_R16, ath_q8_0_repack): per group of 16
 * blocks the 16 scales, then value i of the 16 blocks in 16 consecutive
 * bytes, one load and one sign extension per step. The activations of a
 * row are transposed the same way once per token, so that value i of the 16
 * blocks is one 64-byte load. The tree over the lanes swaps halves, then
 * quarters, then pairs, then neighbours: lane j adds lane j ^ h, the pairs
 * of the canonical tree.
 *
 * Without AVX-512 (or for Q8_0 host copies) the same operations run one lane
 * at a time; the tests compare both with the reference bit for bit.
 */
#include "ops/check.h"
#include "ops/ops.h"
#include "ops/q8_0.h"

#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__)
#include <immintrin.h>
#define HAVE_X86 1
#endif

/* ---- the layout ------------------------------------------------------------------ */

ath_status ath_q8_0_repack(const ath_tensor *src, ath_tensor *dst)
{
    static const char op[] = "q8_0_repack";
    ATH_CHECK(ops_tensor(src, ATH_DTYPE_Q8_0, op, "src"));
    ATH_CHECK(ops_tensor(dst, ATH_DTYPE_Q8_0_R16, op, "dst"));
    if (src->ndim != 2 || !ath_tensor_same_shape(src, dst) || src->shape[1] % ATH_Q8_0_R16_BLOCK)
        return ops_fail(ATH_ERR_INVALID_ARG, op, "src and dst [rows, K], K a multiple of %d", ATH_Q8_0_R16_BLOCK);
    ATH_CHECK(OPS_PLACE(op, OPS_HOST, src, dst));
    const int64_t groups = src->shape[0] * src->shape[1] / ATH_Q8_0_R16_BLOCK;
    const uint8_t *s = src->data;
    uint8_t *d = dst->data;
    for (int64_t g = 0; g < groups; g++, s += 16 * ATH_Q8_0_BYTES, d += ATH_Q8_0_R16_BYTES)
        for (int j = 0; j < 16; j++) {
            const uint8_t *blk = s + j * ATH_Q8_0_BYTES;
            d[2 * j] = blk[0];
            d[2 * j + 1] = blk[1];
            for (int i = 0; i < ATH_Q8_0_BLOCK; i++)
                d[32 + 16 * i + j] = blk[2 + i];
        }
    return ATH_OK;
}

/* ---- one dot product ----------------------------------------------------------------
 *
 * row: K / 512 groups in Q8_0_R16; xt: the activations transposed, group g
 * value i of block j at xt[512 g + 16 i + j]. */

static float dot_r16_scalar(const uint8_t *row, const float *xt, int64_t K)
{
    float P[16] = {0};
    for (int64_t g = 0; g < K / ATH_Q8_0_R16_BLOCK; g++) {
        const uint8_t *grp = row + g * ATH_Q8_0_R16_BYTES;
        const float *xg = xt + g * ATH_Q8_0_R16_BLOCK;
        for (int j = 0; j < 16; j++) {
            float s = (float)(int8_t)grp[32 + j] * xg[j];
            for (int i = 1; i < ATH_Q8_0_BLOCK; i++)
                s = fmaf((float)(int8_t)grp[32 + 16 * i + j], xg[16 * i + j], s);
            P[j] = fmaf(ath_q8_0_scale(grp + 2 * j), s, P[j]);
        }
    }
    return ath_q8_0_tree(P);
}

#ifdef HAVE_X86
__attribute__((target("avx512f,fma,f16c"))) static float dot_r16_avx512(const uint8_t *row, const float *xt, int64_t K)
{
    __m512 P = _mm512_setzero_ps();
    for (int64_t g = 0; g < K / ATH_Q8_0_R16_BLOCK; g++) {
        const uint8_t *grp = row + g * ATH_Q8_0_R16_BYTES;
        const float *xg = xt + g * ATH_Q8_0_R16_BLOCK;
        __m512 s = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(grp + 32)))),
                                 _mm512_loadu_ps(xg));
        for (int i = 1; i < ATH_Q8_0_BLOCK; i++) {
            const __m512 q =
                _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(grp + 32 + 16 * i))));
            s = _mm512_fmadd_ps(q, _mm512_loadu_ps(xg + 16 * i), s);
        }
        const __m512 d = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)grp));
        P = _mm512_fmadd_ps(d, s, P);
    }
    /* the canonical tree: lane j += lane j ^ h for h = 8, 4, 2, 1 */
    P = _mm512_add_ps(P, _mm512_shuffle_f32x4(P, P, 0x4e));   /* 256-bit halves swapped */
    P = _mm512_add_ps(P, _mm512_shuffle_f32x4(P, P, 0xb1));   /* 128-bit lanes in pairs */
    P = _mm512_add_ps(P, _mm512_permute_ps(P, 0x4e));         /* pairs of floats        */
    P = _mm512_add_ps(P, _mm512_permute_ps(P, 0xb1));         /* neighbours             */
    return _mm512_cvtss_f32(P);
}
#endif

/* x [K] (as floats) transposed for dot_r16: value i of block j of group g
 * at 512 g + 16 i + j. */
static void transpose_x(const float *x, int64_t K, float *xt)
{
    for (int64_t g = 0; g < K / ATH_Q8_0_R16_BLOCK; g++)
        for (int j = 0; j < 16; j++)
            for (int i = 0; i < ATH_Q8_0_BLOCK; i++)
                xt[g * ATH_Q8_0_R16_BLOCK + 16 * i + j] = x[g * ATH_Q8_0_R16_BLOCK + 32 * j + i];
}

/* ---- the experts of a layer -------------------------------------------------------- */

typedef struct {
    int64_t t, c, e;     /* token, choice, expert */
} item;

/* The items of one expert: it[first … first + n − 1]. */
typedef struct {
    int64_t e, first, n;
} group;

typedef struct {
    const ath_moe_experts *ex;
    const item            *it;
    const group           *gr;
    int64_t                n_items, n_groups, I, H;
    ath_dtype              act;           /* the dtype of x and y: h and y are rounded to it */
    const float           *xt;            /* [T, H] transposed per token                      */
    float                 *h, *ht;        /* [n_items, I]: h, and h transposed                */
    ath_tensor            *y;
    bool                   r16, avx512;
} job;

static float dot(const job *j, const uint8_t *row, const float *xt, int64_t K)
{
#ifdef HAVE_X86
    if (j->avx512)
        return dot_r16_avx512(row, xt, K);
#endif
    return dot_r16_scalar(row, xt, K);
}

/* The items are sorted by expert, so that a row of an expert's weights is
 * read once for all the tokens that chose it (it stays in the core's cache
 * across them): at batch 1 that changes nothing, in a prompt it reads each
 * expert once per chunk instead of once per token. */

/* Task t of n, phase 1: row i of a group's gate and up for each of its
 * items, h = silu(g) · u. */
static void up_task(void *ctx, int32_t t, int32_t n)
{
    const job *j = ctx;
    const int64_t rows = j->n_groups * j->I, rb = ath_dtype_bytes(ATH_DTYPE_Q8_0_R16, j->H), E = j->ex->n_experts;
    for (int64_t r = rows * t / n; r < rows * (t + 1) / n; r++) {
        const group *g = &j->gr[r / j->I];
        const int64_t i = r % j->I;
        const uint8_t *gw = (const uint8_t *)j->ex->host_tab[g->e] + i * rb;
        const uint8_t *uw = (const uint8_t *)j->ex->host_tab[E + g->e] + i * rb;
        for (int64_t m = g->first; m < g->first + g->n; m++) {
            const float *xt = j->xt + j->it[m].t * j->H;
            const float gv = ath_round_to(j->act, dot(j, gw, xt, j->H)), uv = ath_round_to(j->act, dot(j, uw, xt, j->H));
            j->h[m * j->I + i] = ath_round_to(j->act, ath_q8_silu(gv) * uv);
        }
    }
}

/* Task t of n, phase 2: row o of a group's down projection for each of its items. */
static void down_task(void *ctx, int32_t t, int32_t n)
{
    const job *j = ctx;
    const int64_t rows = j->n_groups * j->H, rb = ath_dtype_bytes(ATH_DTYPE_Q8_0_R16, j->I);
    const int64_t k = j->y->shape[1];
    for (int64_t r = rows * t / n; r < rows * (t + 1) / n; r++) {
        const group *g = &j->gr[r / j->H];
        const int64_t o = r % j->H;
        const uint8_t *d = (const uint8_t *)j->ex->host_tab[2 * j->ex->n_experts + g->e] + o * rb;
        for (int64_t m = g->first; m < g->first + g->n; m++) {
            const item *it = &j->it[m];
            st(j->y, (it->t * k + it->c) * j->H + o, dot(j, d, j->ht + m * j->I, j->I));
        }
    }
}

static int cmp_item(const void *a, const void *b, void *ctx)
{
    (void)ctx;
    const item *x = a, *y = b;
    if (x->e != y->e)
        return x->e < y->e ? -1 : 1;
    if (x->t != y->t)
        return x->t < y->t ? -1 : 1;
    return x->c < y->c ? -1 : (x->c > y->c);
}

ath_status ath_moe_experts_cpu(const ath_tensor *x, const ath_tensor *ids, const ath_moe_experts *ex, ath_tensor *y,
                               ath_pool *pool, ath_arena *a)
{
    static const char op[] = "moe_experts_cpu";
    ATH_CHECK(ops_float(x, op, "x"));
    ATH_CHECK(ops_tensor(ids, ATH_DTYPE_I32, op, "ids"));
    ATH_CHECK(ops_tensor(y, x->dtype, op, "y"));
    if (!ex || !ex->host_tab || !ex->on_dev)
        return ops_fail(ATH_ERR_INVALID_ARG, op, "an expert table with host copies");
    const int64_t T = x->shape[0], H = x->shape[1], k = ids->shape[1], I = ex->inter, E = ex->n_experts;
    if (x->ndim != 2 || ids->ndim != 2 || ids->shape[0] != T || y->ndim != 3 || y->shape[0] != T ||
        y->shape[1] != k || y->shape[2] != H || H != ex->hidden)
        return ops_fail(ATH_ERR_INVALID_ARG, op, "shapes must be x [T, H], ids [T, k], y [T, k, H]");
    ATH_CHECK(OPS_PLACE(op, OPS_HOST, x, ids, y));
    const bool r16 = ex->host_dtype == ATH_DTYPE_Q8_0_R16;
    if (r16 && (H % ATH_Q8_0_R16_BLOCK || I % ATH_Q8_0_R16_BLOCK))
        return ops_fail(ATH_ERR_INVALID_ARG, op, "Q8_0_R16 experts need H and I multiples of 512");

    const int32_t *id = ids->data;
    ath_arena_pos mark = ath_arena_mark(a);
    ath_status status = ATH_OK;
    item *it = ath_arena_alloc(a, (size_t)(T * k) * sizeof *it + 1, _Alignof(item));
    if (!it)
        return ATH_ERR_NOMEM;
    int64_t n = 0;
    for (int64_t t = 0; t < T; t++)
        for (int64_t c = 0; c < k; c++) {
            const int32_t e = id[t * k + c];
            if (e < 0 || e >= E) {
                ath_arena_reset(a, mark);
                return ops_fail(ATH_ERR_RANGE, op, "expert %d of %lld", e, (long long)E);
            }
            if (ex->uses)
                ex->uses[e]++;
            if (!ex->on_dev[e])
                it[n++] = (item){t, c, e};
        }
    if (n == 0)
        goto done;

    if (!r16) {
        /* host copies in another dtype: the reference, item by item */
        for (int64_t m = 0; m < n && status == ATH_OK; m++) {
            ath_tensor xr, yr, gw, uw, dw, ycell;
            const int64_t e = it[m].e;
            const int64_t sh_in[2] = {I, H}, sh_out[2] = {H, I};
            status = ath_tensor_slice(x, 0, it[m].t, 1, &xr);
            if (status == ATH_OK)
                status = ath_tensor_init(&gw, ex->host_dtype, 2, sh_in, (void *)ex->host_tab[e], ATH_TIER_RAM);
            if (status == ATH_OK)
                status = ath_tensor_init(&uw, ex->host_dtype, 2, sh_in, (void *)ex->host_tab[E + e], ATH_TIER_RAM);
            if (status == ATH_OK)
                status = ath_tensor_init(&dw, ex->host_dtype, 2, sh_out, (void *)ex->host_tab[2 * E + e], ATH_TIER_RAM);
            if (status == ATH_OK)
                status = ath_tensor_slice(y, 0, it[m].t, 1, &yr);
            if (status == ATH_OK)
                status = ath_tensor_slice(&yr, 1, it[m].c, 1, &ycell);
            if (status == ATH_OK)
                status = ath_tensor_reshape(&ycell, 2, (const int64_t[]){1, H}, &ycell);
            if (status == ATH_OK)
                status = ath_moe_expert_ref(&xr, &gw, &uw, &dw, &ycell);
        }
        goto done;
    }

    item tmp_item;
    ath_sort(it, (size_t)n, sizeof *it, cmp_item, NULL, &tmp_item);
    group *gr = ath_arena_alloc(a, (size_t)n * sizeof *gr, _Alignof(group));
    if (!gr) {
        status = ATH_ERR_NOMEM;
        goto done;
    }
    int64_t ng = 0;
    for (int64_t m = 0; m < n; m++) {
        if (ng == 0 || gr[ng - 1].e != it[m].e)
            gr[ng++] = (group){.e = it[m].e, .first = m, .n = 0};
        gr[ng - 1].n++;
    }
    job j = {.ex = ex, .it = it, .gr = gr, .n_items = n, .n_groups = ng, .I = I, .H = H, .act = x->dtype, .y = y,
             .r16 = true};
#ifdef HAVE_X86
    /* ATH_CPU_NO_AVX512 set: the one-lane path, so that the tests compare both */
    j.avx512 = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c") &&
               !getenv("ATH_CPU_NO_AVX512");
#endif
    float *xt = ath_arena_alloc(a, (size_t)(T * H) * sizeof(float), 64);
    float *xr = ath_arena_alloc(a, (size_t)H * sizeof(float), 64);
    j.h = ath_arena_alloc(a, (size_t)(n * I) * sizeof(float), 64);
    j.ht = ath_arena_alloc(a, (size_t)(n * I) * sizeof(float), 64);
    if (!xt || !xr || !j.h || !j.ht) {
        status = ATH_ERR_NOMEM;
        goto done;
    }
    for (int64_t t = 0; t < T; t++) {
        for (int64_t i = 0; i < H; i++)
            xr[i] = ld(x, t * H + i);
        transpose_x(xr, H, xt + t * H);
    }
    j.xt = xt;
    const int32_t threads = ath_pool_threads(pool);
    ath_pool_run(pool, threads, up_task, &j);
    for (int64_t m = 0; m < n; m++)
        transpose_x(j.h + m * I, I, j.ht + m * I);
    ath_pool_run(pool, threads, down_task, &j);
done:
    ath_arena_reset(a, mark);
    return status;
}
