/* test_ops.c: the CPU reference ops on hand-computed cases and on properties
 * that must hold exactly (streaming equals all-at-once, causality, head
 * grouping). The comparison against the reference implementation on real
 * weights is tests/test_golden.c.
 */
#include "test.h"

#include "core/arena.h"
#include "ops/ops.h"
#include "ops/q8_0.h"

#include <float.h>

/* ---- helpers --------------------------------------------------------------- */

static ath_arena arena;
static ath_pool  pool;     /* three threads, for the fast paths */

/* A host tensor of the given dtype and shape, filled from vals (or zeros). */
static ath_tensor mk(ath_dtype dt, int32_t ndim, const int64_t *shape, const float *vals)
{
    ath_tensor t;
    ath_status s = ath_tensor_init(&t, dt, ndim, shape, NULL, ATH_TIER_RAM);
    if (s != ATH_OK)
        abort();
    t.data = ath_arena_zalloc(&arena, ath_tensor_nbytes(&t) + 1, 64);
    if (!t.data)
        abort();
    int64_t n = ath_tensor_numel(&t);
    for (int64_t i = 0; vals && i < n; i++)
        ath_store_f32(t.data, dt, i, vals[i]);
    return t;
}

#define SHAPE(...) (const int64_t[]){__VA_ARGS__}, (int32_t)(sizeof((int64_t[]){__VA_ARGS__}) / sizeof(int64_t))
#define VALS(...) (const float[]){__VA_ARGS__}

static ath_tensor mk2(ath_dtype dt, const int64_t *shape, int32_t ndim, const float *vals)
{
    return mk(dt, ndim, shape, vals);
}

/* f32 tensor: F(SHAPE(2, 3), VALS(...)) */
#define F(shape, vals) mk2(ATH_DTYPE_F32, shape, vals)
#define B(shape, vals) mk2(ATH_DTYPE_BF16, shape, vals)
#define Z(dt, shape) mk2(dt, shape, NULL)

static ath_tensor ids_tensor(int64_t n, const int32_t *v)
{
    ath_tensor t = mk(ATH_DTYPE_I32, 1, (int64_t[]){n}, NULL);
    memcpy(t.data, v, (size_t)n * sizeof *v);
    return t;
}

static float at(const ath_tensor *t, int64_t i)
{
    return ath_load_f32(t->data, t->dtype, i);
}

/* Deterministic values in [-1, 1). */
static uint64_t rng_state = 0x9e3779b97f4a7c15ull;
static float rnd(void)
{
    rng_state = rng_state * 6364136223846793005ull + 1442695040888963407ull;
    return (float)((rng_state >> 40) & 0xffffff) / (float)(1 << 23) - 1.0f;
}

static ath_tensor random_tensor(ath_dtype dt, int32_t ndim, const int64_t *shape)
{
    ath_tensor t = mk(dt, ndim, shape, NULL);
    for (int64_t i = 0; i < ath_tensor_numel(&t); i++)
        ath_store_f32(t.data, dt, i, rnd());
    return t;
}

static bool same_bits(const ath_tensor *a, const ath_tensor *b)
{
    return ath_tensor_same_shape(a, b) && a->dtype == b->dtype &&
           memcmp(a->data, b->data, ath_tensor_nbytes(a)) == 0;
}

/* Rows [start, start + len) of a contiguous tensor, as a view. */
static ath_tensor rows(const ath_tensor *t, int64_t start, int64_t len)
{
    ath_tensor v;
    if (ath_tensor_slice(t, 0, start, len, &v) != ATH_OK)
        abort();
    return v;
}

/* ---- bf16 ------------------------------------------------------------------- */

static uint16_t bf16_of_bits(uint32_t u)
{
    float f;
    memcpy(&f, &u, sizeof f);
    return ath_f32_to_bf16(f);
}

ATH_TEST(bf16_rounding)
{
    ATH_EXPECT_EQ_INT(ath_f32_to_bf16(1.0f), 0x3f80);
    ATH_EXPECT(ath_bf16_to_f32(0x3f80) == 1.0f);
    ATH_EXPECT(ath_bf16_to_f32(0xc040) == -3.0f);
    /* Halfway cases round to the even mantissa. */
    ATH_EXPECT_EQ_INT(bf16_of_bits(0x3f808000u), 0x3f80);    /* 1 + 2^-8 -> 1           */
    ATH_EXPECT_EQ_INT(bf16_of_bits(0x3f818000u), 0x3f82);    /* 1 + 3 2^-8 -> 1 + 2^-6  */
    ATH_EXPECT_EQ_INT(bf16_of_bits(0x3f808001u), 0x3f81);    /* just above half: up     */
    ATH_EXPECT_EQ_INT(bf16_of_bits(0x3f807fffu), 0x3f80);    /* just below half: down   */
    ATH_EXPECT_EQ_INT(bf16_of_bits(0xbf808000u), 0xbf80);    /* sign does not matter    */
    /* Special values, as torch rounds them. */
    ATH_EXPECT_EQ_INT(ath_f32_to_bf16(NAN), 0x7fc0);
    ATH_EXPECT_EQ_INT(ath_f32_to_bf16(-NAN), 0x7fc0);
    ATH_EXPECT_EQ_INT(ath_f32_to_bf16(INFINITY), 0x7f80);
    ATH_EXPECT_EQ_INT(ath_f32_to_bf16(-INFINITY), 0xff80);
    ATH_EXPECT_EQ_INT(ath_f32_to_bf16(FLT_MAX), 0x7f80);     /* rounds up to infinity   */
    ATH_EXPECT_EQ_INT(bf16_of_bits(0x00010000u), 0x0001);    /* subnormal, exact        */
    ATH_EXPECT(ath_round_to(ATH_DTYPE_F32, 0.1f) == 0.1f);
    ATH_EXPECT(ath_round_to(ATH_DTYPE_BF16, 0.1f) == ath_bf16_to_f32(ath_f32_to_bf16(0.1f)));
    ATH_EXPECT(ath_round_to(ATH_DTYPE_BF16, 0.1f) != 0.1f);
}

/* ---- argument checks (shared by every op) ---------------------------------- */

ATH_TEST(op_argument_checks)
{
    ath_tensor x = F(SHAPE(2, 3), NULL), w = F(SHAPE(4, 3), NULL), out = F(SHAPE(2, 4), NULL);
    ATH_EXPECT_OK(ath_matmul_ref(&x, &w, &out));
    ATH_EXPECT_STATUS(ath_matmul_ref(NULL, &w, &out), ATH_ERR_INVALID_ARG);
    ath_tensor bad = F(SHAPE(4, 2), NULL);
    ATH_EXPECT_STATUS(ath_matmul_ref(&x, &bad, &out), ATH_ERR_INVALID_ARG);
    ath_tensor xi = Z(ATH_DTYPE_I32, SHAPE(2, 3));
    ATH_EXPECT_STATUS(ath_matmul_ref(&xi, &w, &out), ATH_ERR_TYPE);
    ath_tensor xv = x;
    xv.tier = ATH_TIER_VRAM;
    ATH_EXPECT_STATUS(ath_matmul_ref(&xv, &w, &out), ATH_ERR_UNSUPPORTED);
    ath_tensor col, ok = F(SHAPE(4, 2), NULL);        /* a column slice is not contiguous */
    ATH_REQUIRE_OK(ath_tensor_slice(&w, 1, 0, 2, &col));
    ATH_EXPECT_OK(ath_l2norm_ref(&ok, 0.0f, &ok));
    ATH_EXPECT_STATUS(ath_l2norm_ref(&col, 0.0f, &ok), ATH_ERR_INVALID_ARG);
}

/* ---- embed ------------------------------------------------------------------- */

ATH_TEST(embed_rows)
{
    ath_tensor table = B(SHAPE(3, 2), VALS(1, 2, 3, 4, 5, 6.5f));
    ath_tensor ids = ids_tensor(3, (int32_t[]){2, 0, 2});
    ath_tensor out = F(SHAPE(3, 2), NULL);
    ATH_REQUIRE_OK(ath_embed_ref(&table, &ids, &out));
    const float want[] = {5, 6.5f, 1, 2, 5, 6.5f};
    for (int i = 0; i < 6; i++)
        ATH_EXPECT(at(&out, i) == want[i]);

    ath_tensor bad = ids_tensor(2, (int32_t[]){1, 3});
    ath_tensor out2 = F(SHAPE(2, 2), NULL);
    ATH_EXPECT_STATUS(ath_embed_ref(&table, &bad, &out2), ATH_ERR_RANGE);
    ath_tensor neg = ids_tensor(2, (int32_t[]){-1, 0});
    ATH_EXPECT_STATUS(ath_embed_ref(&table, &neg, &out2), ATH_ERR_RANGE);
    ATH_EXPECT_STATUS(ath_embed_ref(&table, &ids, &out2), ATH_ERR_INVALID_ARG);
}

/* ---- matmul ------------------------------------------------------------------ */

ATH_TEST(matmul_small)
{
    /* x [2, 3] · wᵀ, w [2, 3]: out[r][m] = Σ_k x[r][k] w[m][k]. */
    ath_tensor x = B(SHAPE(2, 3), VALS(1, 2, 3, 4, 5, 6));
    ath_tensor w = B(SHAPE(2, 3), VALS(1, 0, -1, 2, 1, 0));
    ath_tensor out = F(SHAPE(2, 2), NULL);
    ATH_REQUIRE_OK(ath_matmul_ref(&x, &w, &out));
    const float want[] = {-2, 4, -2, 13};
    for (int i = 0; i < 4; i++)
        ATH_EXPECT(at(&out, i) == want[i]);

    /* Leading dimensions are kept; bf16 output is one rounding of the sum. */
    ath_tensor x3 = F(SHAPE(2, 1, 2), VALS(1, 1.0f / 3, 2, 0));
    ath_tensor w2 = F(SHAPE(1, 2), VALS(1, 1));
    ath_tensor o3 = B(SHAPE(2, 1, 1), NULL);
    ATH_REQUIRE_OK(ath_matmul_ref(&x3, &w2, &o3));
    ATH_EXPECT(at(&o3, 0) == ath_round_to(ATH_DTYPE_BF16, 1.0f + 1.0f / 3));
    ATH_EXPECT(at(&o3, 1) == 2.0f);
    ath_tensor o2 = B(SHAPE(2, 1), NULL);
    ATH_EXPECT_STATUS(ath_matmul_ref(&x3, &w2, &o2), ATH_ERR_INVALID_ARG);
}

/* The fast path sums in the reference's order: the same bits for every dtype
 * combination, on one thread or several, with and without a last partial
 * block of outputs. f32 inputs would expose a fused multiply-add, whose
 * single rounding differs from the reference's two. */
ATH_TEST(matmul_fast_same_bits)
{
    static const int64_t shapes[][3] = {   /* N, M, K */
        {1, 1, 1}, {1, 7, 5}, {1, 8, 33}, {2, 9, 16}, {5, 64, 64}, {1, 1029, 512}, {3, 300, 1000},
    };
    static const ath_dtype dt[] = {ATH_DTYPE_BF16, ATH_DTYPE_F32};
    for (size_t i = 0; i < ATH_ARRAY_LEN(shapes); i++)
        for (int c = 0; c < 8; c++) {   /* dtypes of x, w and out */
            const int64_t N = shapes[i][0], M = shapes[i][1], K = shapes[i][2];
            ath_tensor x = random_tensor(dt[c & 1], 2, (const int64_t[]){N, K});
            ath_tensor w = random_tensor(dt[(c >> 1) & 1], 2, (const int64_t[]){M, K});
            ath_tensor ref = mk(dt[c >> 2], 2, (const int64_t[]){N, M}, NULL);
            ath_tensor one = mk(dt[c >> 2], 2, (const int64_t[]){N, M}, NULL);
            ath_tensor many = mk(dt[c >> 2], 2, (const int64_t[]){N, M}, NULL);
            ATH_EXPECT_OK(ath_matmul_ref(&x, &w, &ref));
            ATH_EXPECT_OK(ath_matmul_fast(&x, &w, &one, NULL));
            ATH_EXPECT_OK(ath_matmul_fast(&x, &w, &many, &pool));
            ATH_EXPECT(same_bits(&ref, &one));
            ATH_EXPECT(same_bits(&ref, &many));
        }

    /* the arguments are checked as in the reference */
    ath_tensor x = F(SHAPE(2, 3), NULL), w = F(SHAPE(4, 3), NULL), out = F(SHAPE(2, 4), NULL);
    ATH_EXPECT_OK(ath_matmul_fast(&x, &w, &out, &pool));
    ATH_EXPECT_STATUS(ath_matmul_fast(NULL, &w, &out, &pool), ATH_ERR_INVALID_ARG);
    ath_tensor bad = F(SHAPE(4, 2), NULL);
    ATH_EXPECT_STATUS(ath_matmul_fast(&x, &bad, &out, &pool), ATH_ERR_INVALID_ARG);
    ath_tensor xi = Z(ATH_DTYPE_I32, SHAPE(2, 3));
    ATH_EXPECT_STATUS(ath_matmul_fast(&xi, &w, &out, &pool), ATH_ERR_TYPE);
    ath_tensor xv = x;
    xv.tier = ATH_TIER_VRAM;
    ATH_EXPECT_STATUS(ath_matmul_fast(&xv, &w, &out, &pool), ATH_ERR_UNSUPPORTED);
}

/* ---- Q8_0 (step 12) ---------------------------------------------------------- */

/* A random Q8_0 [M, K] weight: per block a scale of about 1/127 times a
 * random magnitude in f16, and int8 values in [-127, 127]. */
static ath_tensor random_q8_0(int64_t M, int64_t K)
{
    ath_tensor t = mk(ATH_DTYPE_Q8_0, 2, (const int64_t[]){M, K}, NULL);
    uint8_t *p = t.data;
    for (int64_t b = 0; b < M * K / 32; b++, p += 34) {
        /* f16 scales between 2^-10 and 2^-6: exponents 5 .. 9, a random mantissa */
        const uint16_t h = (uint16_t)(((5 + (b % 5)) << 10) | ((uint16_t)((rnd() + 1.0f) * 511.0f) & 0x3ff));
        p[0] = (uint8_t)h;
        p[1] = (uint8_t)(h >> 8);
        for (int i = 0; i < 32; i++)
            p[2 + i] = (uint8_t)(int8_t)(rnd() * 127.0f);
    }
    return t;
}

/* Element m, k of a Q8_0 weight, exactly. */
static double q8_0_at(const ath_tensor *w, int64_t m, int64_t k)
{
    const uint8_t *blk = (const uint8_t *)w->data + (m * w->shape[1] + k) / 32 * 34;
    const uint16_t h = (uint16_t)(blk[0] | blk[1] << 8);
    return (double)ath_f16_to_f32(h) * (double)(int8_t)blk[2 + k % 32];
}

ATH_TEST(q8_0_f16_and_exp)
{
    ATH_EXPECT(ath_f16_to_f32(0x3c00) == 1.0f && ath_f16_to_f32(0x3800) == 0.5f);
    ATH_EXPECT(ath_f16_to_f32(0xc000) == -2.0f && ath_f16_to_f32(0x7bff) == 65504.0f);
    ATH_EXPECT(ath_f16_to_f32(0x0001) == ldexpf(1.0f, -24) && ath_f16_to_f32(0x03ff) == ldexpf(1023.0f, -24));
    ATH_EXPECT(isinf(ath_f16_to_f32(0x7c00)) && isnan(ath_f16_to_f32(0x7e00)));
    ATH_EXPECT(signbit(ath_f16_to_f32(0x8000)) && ath_f16_to_f32(0x8000) == 0.0f);

    /* the canonical exponential: within 2 ulp of exp (in double) on [-87, 88] */
    ATH_EXPECT(ath_q8_expf(0.0f) == 1.0f);
    double worst = 0;
    for (int i = 0; i < 175000; i++) {
        const float x = -87.0f + (float)i * 1e-3f;   /* below 88: the clamp starts there */
        const double e = exp((double)x), got = ath_q8_expf(x);
        const double ulp = ldexp(1.0, ilogb(e) - 23);
        const double err = fabs(got - e) / ulp;
        worst = err > worst ? err : worst;
    }
    ATH_EXPECT(worst <= 2.0);
    ATH_EXPECT(ath_q8_expf(-1000.0f) == ath_q8_expf(-87.0f) && ath_q8_expf(1000.0f) == ath_q8_expf(88.0f));
    ATH_EXPECT(isnan(ath_q8_expf(NAN)));
    ATH_EXPECT(ath_q8_silu(0.0f) == 0.0f && fabsf(ath_q8_silu(1.0f) - 0.7310586f) < 1e-6f);
    printf("  canonical expf: worst %.3f ulp of exp on [-87, 88]\n", worst);
}

ATH_TEST(matmul_q8_0)
{
    /* hand case: K = 64, two blocks; the canonical order is exact here */
    ath_tensor w = mk(ATH_DTYPE_Q8_0, 2, (const int64_t[]){1, 64}, NULL);
    uint8_t *p = w.data;
    p[0] = 0x00; p[1] = 0x38;                      /* 0.5  */
    p[34] = 0x00; p[35] = 0x40;                    /* 2.0  */
    for (int i = 0; i < 32; i++) {
        p[2 + i] = (uint8_t)(int8_t)(i - 16);
        p[36 + i] = (uint8_t)(int8_t)(i % 3 - 1);
    }
    ath_tensor x = mk(ATH_DTYPE_F32, 2, (const int64_t[]){1, 64}, NULL);
    double want = 0;
    for (int k = 0; k < 64; k++) {
        ath_store_f32(x.data, ATH_DTYPE_F32, k, (float)(k % 5) - 2.0f);
        want += q8_0_at(&w, 0, k) * ((k % 5) - 2.0);
    }
    ath_tensor out = F(SHAPE(1, 1), NULL);
    ATH_REQUIRE_OK(ath_matmul_ref(&x, &w, &out));
    ATH_EXPECT(at(&out, 0) == (float)want);

    /* random shapes: the reference within its error bound of the exact sum,
     * the fast path the reference's bits with any number of threads */
    static const int64_t shapes[][3] = {{1, 1, 32}, {1, 7, 64}, {2, 9, 512}, {3, 33, 544}, {1, 300, 2048}, {4, 17, 4096}};
    static const ath_dtype dt[] = {ATH_DTYPE_BF16, ATH_DTYPE_F32};
    double worst = 0;
    for (size_t i = 0; i < ATH_ARRAY_LEN(shapes); i++)
        for (int c = 0; c < 4; c++) {
            const int64_t N = shapes[i][0], M = shapes[i][1], K = shapes[i][2];
            ath_tensor xq = random_tensor(dt[c & 1], 2, (const int64_t[]){N, K});
            ath_tensor wq = random_q8_0(M, K);
            ath_tensor ref = mk(dt[c >> 1], 2, (const int64_t[]){N, M}, NULL);
            ath_tensor ref32 = mk(ATH_DTYPE_F32, 2, (const int64_t[]){N, M}, NULL);
            ath_tensor one = mk(dt[c >> 1], 2, (const int64_t[]){N, M}, NULL);
            ath_tensor many = mk(dt[c >> 1], 2, (const int64_t[]){N, M}, NULL);
            ATH_EXPECT_OK(ath_matmul_ref(&xq, &wq, &ref));
            ATH_EXPECT_OK(ath_matmul_ref(&xq, &wq, &ref32));
            ATH_EXPECT_OK(ath_matmul_fast(&xq, &wq, &one, NULL));
            ATH_EXPECT_OK(ath_matmul_fast(&xq, &wq, &many, &pool));
            ATH_EXPECT(same_bits(&ref, &one));
            ATH_EXPECT(same_bits(&ref, &many));
            /* error bound of a sum of K terms in fp32: K · 2^-24 · Σ|w x|
             * (a loose form of the classic bound, enough for a check) */
            for (int64_t r = 0; r < N; r++)
                for (int64_t m = 0; m < M; m++) {
                    double exact = 0, mag = 0;
                    for (int64_t k = 0; k < K; k++) {
                        const double t = q8_0_at(&wq, m, k) * (double)at(&xq, r * K + k);
                        exact += t;
                        mag += fabs(t);
                    }
                    const double err = fabs((double)at(&ref32, r * M + m) - exact);
                    const double bound = (double)K * ldexp(1.0, -24) * mag;
                    ATH_EXPECT(err <= bound);
                    worst = mag > 0 && err / bound > worst ? err / bound : worst;
                }
        }
    printf("  Q8_0 matmul: worst error %.3f of the bound K 2^-24 sum|w x|\n", worst);

    /* rows must be whole blocks: the dtype refuses others; a Q8_0 x is refused */
    ath_tensor xbad = mk(ATH_DTYPE_Q8_0, 2, (const int64_t[]){1, 64}, NULL);
    ATH_EXPECT_STATUS(ath_matmul_ref(&xbad, &w, &out), ATH_ERR_TYPE);
}

/* ---- norms ------------------------------------------------------------------- */

ATH_TEST(rmsnorm_hand)
{
    /* x = (3, 4): mean(x²) = 12.5, x / sqrt(12.5) = (0.848528, 1.131371). */
    ath_tensor x = F(SHAPE(2, 2), VALS(3, 4, 0, 0));
    ath_tensor w = F(SHAPE(2), VALS(0, 1));
    ath_tensor out = F(SHAPE(2, 2), NULL);
    const double a = 3 / sqrt(12.5), b = 4 / sqrt(12.5);

    ATH_REQUIRE_OK(ath_rmsnorm_ref(&x, &w, 1.0f, 0.0f, &out));      /* zero-centered: scale 1 + w */
    ATH_EXPECT_NEAR(at(&out, 0), a, 1e-6);
    ATH_EXPECT_NEAR(at(&out, 1), 2 * b, 1e-6);
    ATH_REQUIRE_OK(ath_rmsnorm_ref(&x, &w, 0.0f, 0.0f, &out));      /* plain: scale w */
    ATH_EXPECT(at(&out, 0) == 0.0f);
    ATH_EXPECT_NEAR(at(&out, 1), b, 1e-6);
    /* A zero row stays zero thanks to eps. */
    ATH_REQUIRE_OK(ath_rmsnorm_ref(&x, &w, 1.0f, 1e-6f, &out));
    ATH_EXPECT(at(&out, 2) == 0.0f && at(&out, 3) == 0.0f);
    /* eps enters under the square root: (3, 4) with eps 12.5 gives x / 5. */
    ATH_REQUIRE_OK(ath_rmsnorm_ref(&x, &w, 1.0f, 12.5f, &out));
    ATH_EXPECT_NEAR(at(&out, 0), 0.6, 1e-6);

    ath_tensor w3 = F(SHAPE(3), NULL);
    ATH_EXPECT_STATUS(ath_rmsnorm_ref(&x, &w3, 1.0f, 0.0f, &out), ATH_ERR_INVALID_ARG);
    ATH_EXPECT_STATUS(ath_rmsnorm_ref(&x, NULL, 1.0f, 0.0f, &out), ATH_ERR_INVALID_ARG);
}

ATH_TEST(rmsnorm_gated_hand)
{
    const double a = 3 / sqrt(12.5), b = 4 / sqrt(12.5);
    ath_tensor x = F(SHAPE(1, 2), VALS(3, 4));
    ath_tensor z = F(SHAPE(1, 2), VALS(0, 2));
    ath_tensor w = F(SHAPE(2), VALS(2, 0.5f));
    ath_tensor out = F(SHAPE(1, 2), NULL);
    ATH_REQUIRE_OK(ath_rmsnorm_gated_ref(&x, &z, &w, 0.0f, &out));
    const double silu2 = 2 / (1 + exp(-2.0));
    ATH_EXPECT(at(&out, 0) == 0.0f);                              /* silu(0) = 0 */
    ATH_EXPECT_NEAR(at(&out, 1), 0.5 * b * silu2, 1e-6);

    /* With bf16 x the normalized value is rounded to bf16 before the weight. */
    ath_tensor xb = B(SHAPE(1, 2), VALS(3, 4));
    ath_tensor z1 = F(SHAPE(1, 2), VALS(30, 30));                   /* silu(30) = 30 in fp32 */
    ath_tensor w1 = F(SHAPE(2), VALS(1, 1));
    ATH_REQUIRE_OK(ath_rmsnorm_gated_ref(&xb, &z1, &w1, 0.0f, &out));
    ATH_EXPECT(at(&out, 0) == ath_round_to(ATH_DTYPE_BF16, (float)a) * 30.0f);
    ATH_EXPECT(at(&out, 1) == ath_round_to(ATH_DTYPE_BF16, (float)b) * 30.0f);
    ATH_EXPECT(at(&out, 0) != (float)a * 30.0f);
}

ATH_TEST(l2norm_hand)
{
    ath_tensor x = F(SHAPE(2, 2), VALS(3, 4, 0, -2));
    ath_tensor out = F(SHAPE(2, 2), NULL);
    ATH_REQUIRE_OK(ath_l2norm_ref(&x, 0.0f, &out));
    ATH_EXPECT_NEAR(at(&out, 0), 0.6, 1e-7);
    ATH_EXPECT_NEAR(at(&out, 1), 0.8, 1e-7);
    ATH_EXPECT(at(&out, 2) == 0.0f);
    ATH_EXPECT_NEAR(at(&out, 3), -1.0, 1e-7);
    ATH_REQUIRE_OK(ath_l2norm_ref(&x, 75.0f, &out));               /* sqrt(25 + 75) = 10 */
    ATH_EXPECT_NEAR(at(&out, 0), 0.3, 1e-7);
}

/* ---- rope -------------------------------------------------------------------- */

ATH_TEST(rope_hand)
{
    /* Position 0 is the identity. */
    ath_tensor x = random_tensor(ATH_DTYPE_BF16, 3, (int64_t[]){2, 3, 8});
    ath_tensor zero = ids_tensor(2, (int32_t[]){0, 0});
    ath_tensor out = B(SHAPE(2, 3, 8), NULL);
    ATH_REQUIRE_OK(ath_rope_ref(&x, &zero, 4, 1e4f, &out));
    ATH_EXPECT(same_bits(&x, &out));

    /* rope_dim 4, theta 100: f_0 = 1, f_1 = 100^(-1/2) = 0.1. Dimension 0
     * pairs with 2 and 1 with 3 (rotate-half); dimensions 4, 5 pass through. */
    ath_tensor e = F(SHAPE(2, 1, 6), VALS(1, 0, 0, 0, 7, 8, 0, 1, 0, 0, 7, 8));
    ath_tensor one = ids_tensor(2, (int32_t[]){1, 1});
    ath_tensor o = F(SHAPE(2, 1, 6), NULL);
    ATH_REQUIRE_OK(ath_rope_ref(&e, &one, 4, 100.0f, &o));
    ATH_EXPECT(at(&o, 0) == cosf(1.0f) && at(&o, 2) == sinf(1.0f));
    ATH_EXPECT(at(&o, 1) == 0.0f && at(&o, 3) == 0.0f);
    ATH_EXPECT_NEAR(at(&o, 7), cos(0.1), 1e-7);
    ATH_EXPECT_NEAR(at(&o, 9), sin(0.1), 1e-7);
    ATH_EXPECT(at(&o, 4) == 7.0f && at(&o, 5) == 8.0f && at(&o, 10) == 7.0f && at(&o, 11) == 8.0f);

    /* Attention scores depend only on the distance between positions. */
    ath_tensor q = random_tensor(ATH_DTYPE_F32, 3, (int64_t[]){1, 1, 16});
    ath_tensor k = random_tensor(ATH_DTYPE_F32, 3, (int64_t[]){1, 1, 16});
    double dots[2];
    for (int shift = 0; shift < 2; shift++) {
        ath_tensor pq = ids_tensor(1, (int32_t[]){7 + 100 * shift});
        ath_tensor pk = ids_tensor(1, (int32_t[]){3 + 100 * shift});
        ath_tensor rq = F(SHAPE(1, 1, 16), NULL), rk = F(SHAPE(1, 1, 16), NULL);
        ATH_REQUIRE_OK(ath_rope_ref(&q, &pq, 16, 1e4f, &rq));
        ATH_REQUIRE_OK(ath_rope_ref(&k, &pk, 16, 1e4f, &rk));
        dots[shift] = 0;
        for (int i = 0; i < 16; i++)
            dots[shift] += (double)at(&rq, i) * at(&rk, i);
    }
    ATH_EXPECT_NEAR(dots[0], dots[1], 1e-4);

    ATH_EXPECT_STATUS(ath_rope_ref(&e, &one, 3, 100.0f, &o), ATH_ERR_INVALID_ARG);
    ATH_EXPECT_STATUS(ath_rope_ref(&e, &one, 8, 100.0f, &o), ATH_ERR_INVALID_ARG);
}

/* ---- elementwise -------------------------------------------------------------- */

ATH_TEST(act_hand)
{
    const double s1 = 1 / (1 + exp(-1.0));                         /* sigmoid(1) */
    ath_tensor g = F(SHAPE(3), VALS(0, 1, -1)), u = F(SHAPE(3), VALS(5, 2, 2));
    ath_tensor out = F(SHAPE(3), NULL);
    ATH_REQUIRE_OK(ath_swiglu_ref(&g, &u, &out));
    ATH_EXPECT(at(&out, 0) == 0.0f);
    ATH_EXPECT_NEAR(at(&out, 1), 2 * s1, 1e-7);
    ATH_EXPECT_NEAR(at(&out, 2), -2 * (1 - s1), 1e-7);

    ath_tensor x = F(SHAPE(3), VALS(2, 2, 3)), gate = F(SHAPE(3), VALS(0, 100, 1));
    ATH_REQUIRE_OK(ath_sigmoid_mul_ref(&x, &gate, &out));
    ATH_EXPECT(at(&out, 0) == 1.0f && at(&out, 1) == 2.0f);
    ATH_EXPECT_NEAR(at(&out, 2), 3 * s1, 1e-6);

    /* bf16 add is one rounding of the fp32 sum; out may alias an input. */
    ath_tensor a = B(SHAPE(2), VALS(1, 256)), b = B(SHAPE(2), VALS(0.00390625f, 1.5f));
    ATH_REQUIRE_OK(ath_add_ref(&a, &b, &a));
    ATH_EXPECT(at(&a, 0) == 1.0f);                                  /* 1 + 2^-8 ties to 1 */
    ATH_EXPECT(at(&a, 1) == 258.0f);                                /* 257.5 rounds to 258 */
    ath_tensor c = F(SHAPE(3), NULL);
    ATH_EXPECT_STATUS(ath_add_ref(&a, &b, &c), ATH_ERR_INVALID_ARG);
}

/* ---- attention ------------------------------------------------------------------ */

ATH_TEST(attn_hand)
{
    /* q = 0: every visible key scores the same, so each query averages the
     * values it can see: the causal prefix means 1, 1.5, 3. */
    ath_tensor q = F(SHAPE(3, 1, 2), NULL);
    ath_tensor k = random_tensor(ATH_DTYPE_F32, 3, (int64_t[]){3, 1, 2});
    ath_tensor v = F(SHAPE(3, 1, 1), VALS(1, 2, 6));
    ath_tensor out = F(SHAPE(3, 1, 1), NULL);
    ATH_REQUIRE_OK(ath_attn_ref(&q, &k, &v, NULL, 1.0f, &out));
    ATH_EXPECT_NEAR(at(&out, 0), 1.0, 1e-6);
    ATH_EXPECT_NEAR(at(&out, 1), 1.5, 1e-6);
    ATH_EXPECT_NEAR(at(&out, 2), 3.0, 1e-6);

    /* Grouped heads: query heads 0, 1 read KV head 0, heads 2, 3 read KV head 1. */
    ath_tensor q4 = random_tensor(ATH_DTYPE_F32, 3, (int64_t[]){1, 4, 2});
    ath_tensor k2 = random_tensor(ATH_DTYPE_F32, 3, (int64_t[]){1, 2, 2});
    ath_tensor v2 = F(SHAPE(1, 2, 1), VALS(10, 20));
    ath_tensor o4 = F(SHAPE(1, 4, 1), NULL);
    ATH_REQUIRE_OK(ath_attn_ref(&q4, &k2, &v2, NULL, 1.0f, &o4));
    ATH_EXPECT(at(&o4, 0) == 10.0f && at(&o4, 1) == 10.0f && at(&o4, 2) == 20.0f && at(&o4, 3) == 20.0f);

    /* A large score selects its value. */
    ath_tensor qs = F(SHAPE(1, 1, 2), VALS(1, 0));
    ath_tensor ks = F(SHAPE(2, 1, 2), VALS(0, 1, 50, 0));
    ath_tensor vs = F(SHAPE(2, 1, 2), VALS(1, 2, 3, 4));
    ath_tensor os = F(SHAPE(1, 1, 2), NULL);
    ATH_REQUIRE_OK(ath_attn_ref(&qs, &ks, &vs, NULL, 1.0f, &os));
    ATH_EXPECT_NEAR(at(&os, 0), 3.0, 1e-6);
    ATH_EXPECT_NEAR(at(&os, 1), 4.0, 1e-6);

    ath_tensor q3 = F(SHAPE(1, 3, 2), NULL), o3 = F(SHAPE(1, 3, 1), NULL);
    ATH_EXPECT_STATUS(ath_attn_ref(&q3, &k2, &v2, NULL, 1.0f, &o3), ATH_ERR_INVALID_ARG);   /* 3 % 2 */
    ath_tensor qlong = F(SHAPE(2, 4, 2), NULL), olong = F(SHAPE(2, 4, 1), NULL);
    ATH_EXPECT_STATUS(ath_attn_ref(&qlong, &k2, &v2, NULL, 1.0f, &olong), ATH_ERR_INVALID_ARG);  /* S < T */
}

ATH_TEST(attn_decode_matches_prefill)
{
    /* The last query of a prefill and the same query alone against the same
     * keys (decode with a KV cache) do the same arithmetic. */
    const int64_t S = 6, nh = 4, nkv = 2, d = 8;
    ath_tensor q = random_tensor(ATH_DTYPE_BF16, 3, (int64_t[]){S, nh, d});
    ath_tensor k = random_tensor(ATH_DTYPE_BF16, 3, (int64_t[]){S, nkv, d});
    ath_tensor v = random_tensor(ATH_DTYPE_BF16, 3, (int64_t[]){S, nkv, d});
    ath_tensor full = B(SHAPE(S, nh, d), NULL), one = B(SHAPE(1, nh, d), NULL);
    ATH_REQUIRE_OK(ath_attn_ref(&q, &k, &v, NULL, 0.35f, &full));
    ath_tensor qlast = rows(&q, S - 1, 1);
    ATH_REQUIRE_OK(ath_attn_ref(&qlast, &k, &v, NULL, 0.35f, &one));
    ath_tensor flast = rows(&full, S - 1, 1);
    ATH_EXPECT(same_bits(&flast, &one));
    /* And a query never sees later keys: changing the last key leaves the
     * first S - 1 outputs unchanged. */
    ath_tensor full2 = B(SHAPE(S, nh, d), NULL);
    ath_store_f32(k.data, k.dtype, (S - 1) * nkv * d, 9.0f);
    ATH_REQUIRE_OK(ath_attn_ref(&q, &k, &v, NULL, 0.35f, &full2));
    {
        /* positions (step 9): the same queries over a longer cache, whose
         * other rows are never read, give the same bits */
        const int64_t T = q.shape[0], S2 = S + 5;
        ath_tensor kbig = mk(k.dtype, 3, (int64_t[]){S2, nkv, d}, NULL);
        ath_tensor vbig = mk(v.dtype, 3, (int64_t[]){S2, nkv, d}, NULL);
        memcpy(kbig.data, k.data, ath_tensor_nbytes(&k));
        memcpy(vbig.data, v.data, ath_tensor_nbytes(&v));
        for (int64_t i = ath_tensor_numel(&k); i < ath_tensor_numel(&kbig); i++) {
            ath_store_f32(kbig.data, kbig.dtype, i, 1e30f);
            ath_store_f32(vbig.data, vbig.dtype, i, 1e30f);
        }
        int32_t pv[16];
        for (int64_t t = 0; t < T; t++)
            pv[t] = (int32_t)(S - T + t);
        ath_tensor pos = ids_tensor(T, pv), withpos = B(SHAPE(S, nh, d), NULL);
        ATH_REQUIRE_OK(ath_attn_ref(&q, &kbig, &vbig, &pos, 0.35f, &withpos));
        ATH_EXPECT(same_bits(&withpos, &full2));
        ((int32_t *)pos.data)[0] = (int32_t)S2;
        ATH_EXPECT_STATUS(ath_attn_ref(&q, &kbig, &vbig, &pos, 0.35f, &withpos), ATH_ERR_RANGE);
        ath_tensor pos2 = ids_tensor(2, pv);
        ATH_EXPECT_STATUS(ath_attn_ref(&q, &kbig, &vbig, &pos2, 0.35f, &withpos), ATH_ERR_INVALID_ARG);
    }
    ath_tensor a = rows(&full, 0, S - 1), b = rows(&full2, 0, S - 1);
    ATH_EXPECT(same_bits(&a, &b));
}

/* ---- conv -------------------------------------------------------------------------- */

ATH_TEST(conv_hand)
{
    /* K = 4. The last tap multiplies the current input, the first the input
     * three steps back. */
    ath_tensor x = F(SHAPE(5, 1), VALS(1, 2, 3, 4, 5));
    ath_tensor ident = F(SHAPE(1, 4), VALS(0, 0, 0, 1)), delay = F(SHAPE(1, 1, 4), VALS(1, 0, 0, 0));
    ath_tensor state = F(SHAPE(3, 1), NULL), out = F(SHAPE(5, 1), NULL);
    ATH_REQUIRE_OK(ath_conv_ref(&x, &ident, &state, ATH_ACT_NONE, &out));
    for (int t = 0; t < 5; t++)
        ATH_EXPECT(at(&out, t) == at(&x, t));
    ATH_EXPECT(at(&state, 0) == 3.0f && at(&state, 1) == 4.0f && at(&state, 2) == 5.0f);

    memset(state.data, 0, ath_tensor_nbytes(&state));
    ATH_REQUIRE_OK(ath_conv_ref(&x, &delay, &state, ATH_ACT_NONE, &out));
    const float want[] = {0, 0, 0, 1, 2};
    for (int t = 0; t < 5; t++)
        ATH_EXPECT(at(&out, t) == want[t]);

    memset(state.data, 0, ath_tensor_nbytes(&state));
    ATH_REQUIRE_OK(ath_conv_ref(&x, &ident, &state, ATH_ACT_SILU, &out));
    ATH_EXPECT_NEAR(at(&out, 0), 1 / (1 + exp(-1.0)), 1e-7);

    ath_tensor bad_state = F(SHAPE(4, 1), NULL);
    ATH_EXPECT_STATUS(ath_conv_ref(&x, &ident, &bad_state, ATH_ACT_NONE, &out), ATH_ERR_INVALID_ARG);
    ath_tensor bad_w = F(SHAPE(2, 4), NULL);
    ATH_EXPECT_STATUS(ath_conv_ref(&x, &bad_w, &state, ATH_ACT_NONE, &out), ATH_ERR_INVALID_ARG);
}

ATH_TEST(conv_streaming)
{
    /* A sequence fed in pieces of 2, 1 and 3 tokens gives the outputs and the
     * final state of the whole sequence at once, bit for bit. */
    const int64_t T = 6, C = 3, K = 4;
    ath_tensor x = random_tensor(ATH_DTYPE_BF16, 2, (int64_t[]){T, C});
    ath_tensor w = random_tensor(ATH_DTYPE_BF16, 2, (int64_t[]){C, K});
    ath_tensor s_all = B(SHAPE(K - 1, C), NULL), s_step = B(SHAPE(K - 1, C), NULL);
    ath_tensor o_all = B(SHAPE(T, C), NULL), o_step = B(SHAPE(T, C), NULL);
    ATH_REQUIRE_OK(ath_conv_ref(&x, &w, &s_all, ATH_ACT_SILU, &o_all));
    const int64_t piece[] = {2, 1, 3};
    for (int64_t i = 0, t = 0; i < 3; t += piece[i++]) {
        ath_tensor xi = rows(&x, t, piece[i]), oi = rows(&o_step, t, piece[i]);
        ATH_REQUIRE_OK(ath_conv_ref(&xi, &w, &s_step, ATH_ACT_SILU, &oi));
    }
    ATH_EXPECT(same_bits(&o_all, &o_step));
    ATH_EXPECT(same_bits(&s_all, &s_step));
}

/* ---- Gated DeltaNet ------------------------------------------------------------------ */

ATH_TEST(gdn_gates_hand)
{
    ath_tensor a = F(SHAPE(2, 2), VALS(0, 30, 0, -30)), b = F(SHAPE(2, 2), VALS(0, 2, -2, 0));
    ath_tensor a_log = F(SHAPE(2), VALS(0, logf(2.0f))), dt = F(SHAPE(2), VALS(0, 0));
    ath_tensor g = F(SHAPE(2, 2), NULL), beta = F(SHAPE(2, 2), NULL);
    ATH_REQUIRE_OK(ath_gdn_gates_ref(&a, &b, &a_log, false, &dt, &g, &beta));
    ATH_EXPECT_NEAR(at(&g, 0), -log(2.0), 1e-7);                      /* −1 · softplus(0)      */
    ATH_EXPECT_NEAR(at(&g, 1), -60.0, 1e-5);                          /* above the threshold   */
    ATH_EXPECT_NEAR(at(&g, 3), -2 * log1p(exp(-30.0)), 1e-18);        /* tiny, not zero        */
    ATH_EXPECT(at(&g, 3) < 0.0f);
    ATH_EXPECT(at(&beta, 0) == 0.5f);
    ATH_EXPECT_NEAR(at(&beta, 1), 1 / (1 + exp(-2.0)), 1e-7);
    ATH_EXPECT_NEAR(at(&beta, 2), 1 / (1 + exp(2.0)), 1e-7);
}

ATH_TEST(gdn_hand)
{
    /* One head, dk = dv = 2. From the zero state with beta 1 and no decay,
     * one step writes S = k vᵀ and reads out = v (k · q). */
    ath_tensor q = F(SHAPE(1, 1, 2), VALS(0.6f, 0.8f)), k = F(SHAPE(1, 1, 2), VALS(1, 0));
    ath_tensor v = F(SHAPE(1, 1, 2), VALS(3, 5));
    ath_tensor g = F(SHAPE(1, 1), VALS(0)), one = F(SHAPE(1, 1), VALS(1));
    ath_tensor S = F(SHAPE(1, 2, 2), NULL), out = F(SHAPE(1, 1, 2), NULL);
    ATH_REQUIRE_OK(ath_gdn_ref(&q, &k, &v, &g, &one, 1.0f, 0.0f, false, &S, &out));
    ATH_EXPECT_NEAR(at(&out, 0), 1.8, 1e-6);
    ATH_EXPECT_NEAR(at(&out, 1), 3.0, 1e-6);
    ATH_EXPECT(at(&S, 0) == 3.0f && at(&S, 1) == 5.0f && at(&S, 2) == 0.0f && at(&S, 3) == 0.0f);

    /* The delta rule replaces what the key stored: a second write along the
     * same key leaves S = k v2ᵀ; the query scale multiplies the read. */
    ath_tensor v2 = F(SHAPE(1, 1, 2), VALS(-1, 2));
    ATH_REQUIRE_OK(ath_gdn_ref(&q, &k, &v2, &g, &one, 0.5f, 0.0f, false, &S, &out));
    ATH_EXPECT(at(&S, 0) == -1.0f && at(&S, 1) == 2.0f);
    ATH_EXPECT_NEAR(at(&out, 0), -0.3, 1e-6);
    ATH_EXPECT_NEAR(at(&out, 1), 0.6, 1e-6);

    /* beta 0 writes nothing; g = log(1/2) halves the state. */
    ath_tensor half = F(SHAPE(1, 1), VALS(logf(0.5f))), zero = F(SHAPE(1, 1), VALS(0));
    ATH_REQUIRE_OK(ath_gdn_ref(&q, &k, &v, &half, &zero, 1.0f, 0.0f, false, &S, &out));
    ATH_EXPECT_NEAR(at(&S, 0), -0.5, 1e-7);
    ATH_EXPECT_NEAR(at(&S, 1), 1.0, 1e-7);

    ath_tensor Sb = B(SHAPE(1, 2, 2), NULL);
    ATH_EXPECT_STATUS(ath_gdn_ref(&q, &k, &v, &g, &one, 1.0f, 0.0f, false, &Sb, &out), ATH_ERR_TYPE);
}

ATH_TEST(gdn_heads_and_streaming)
{
    /* nk = 2 query/key heads serve nv = 4 value heads: value heads 0, 1 use
     * q/k head 0 and value heads 2, 3 use head 1. Check it against runs with
     * the q/k heads repeated explicitly, and check that token-by-token equals
     * all at once, bit for bit. */
    const int64_t T = 5, nk = 2, nv = 4, dk = 4, dv = 3;
    ath_tensor q = random_tensor(ATH_DTYPE_F32, 3, (int64_t[]){T, nk, dk});
    ath_tensor k = random_tensor(ATH_DTYPE_F32, 3, (int64_t[]){T, nk, dk});
    ath_tensor v = random_tensor(ATH_DTYPE_BF16, 3, (int64_t[]){T, nv, dv});
    ath_tensor g = random_tensor(ATH_DTYPE_F32, 2, (int64_t[]){T, nv});
    ath_tensor beta = random_tensor(ATH_DTYPE_BF16, 2, (int64_t[]){T, nv});
    for (int64_t i = 0; i < T * nv; i++) {               /* g <= 0, beta in (0, 1) */
        ath_store_f32(g.data, g.dtype, i, -fabsf(at(&g, i)));
        ath_store_f32(beta.data, beta.dtype, i, 0.5f + 0.5f * at(&beta, i));
    }
    ath_tensor S = F(SHAPE(nv, dk, dv), NULL), out = B(SHAPE(T, nv, dv), NULL);
    ATH_REQUIRE_OK(ath_gdn_ref(&q, &k, &v, &g, &beta, 0.5f, 1e-6f, false, &S, &out));

    ath_tensor qr = F(SHAPE(T, nv, dk), NULL), kr = F(SHAPE(T, nv, dk), NULL);
    for (int64_t t = 0; t < T; t++)
        for (int64_t h = 0; h < nv; h++)
            for (int64_t i = 0; i < dk; i++) {
                ath_store_f32(qr.data, qr.dtype, (t * nv + h) * dk + i, at(&q, (t * nk + h / 2) * dk + i));
                ath_store_f32(kr.data, kr.dtype, (t * nv + h) * dk + i, at(&k, (t * nk + h / 2) * dk + i));
            }
    ath_tensor S2 = F(SHAPE(nv, dk, dv), NULL), out2 = B(SHAPE(T, nv, dv), NULL);
    ATH_REQUIRE_OK(ath_gdn_ref(&qr, &kr, &v, &g, &beta, 0.5f, 1e-6f, false, &S2, &out2));
    ATH_EXPECT(same_bits(&out, &out2));
    ATH_EXPECT(same_bits(&S, &S2));

    ath_tensor S3 = F(SHAPE(nv, dk, dv), NULL), out3 = B(SHAPE(T, nv, dv), NULL);
    for (int64_t t = 0; t < T; t++) {
        ath_tensor qt = rows(&q, t, 1), kt = rows(&k, t, 1), vt = rows(&v, t, 1);
        ath_tensor gt = rows(&g, t, 1), bt = rows(&beta, t, 1), ot = rows(&out3, t, 1);
        ATH_REQUIRE_OK(ath_gdn_ref(&qt, &kt, &vt, &gt, &bt, 0.5f, 1e-6f, false, &S3, &ot));
    }
    ATH_EXPECT(same_bits(&out, &out3));
    ATH_EXPECT(same_bits(&S, &S3));
}

/* ---- scatter (step 9) ---------------------------------------------------- */

ATH_TEST(scatter_rows)
{
    ath_tensor src = F(SHAPE(2, 3), ((float[]){1, 2, 3, 4, 5, 6}));
    ath_tensor dst = F(SHAPE(4, 3), ((float[]){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
    ath_tensor idx = ids_tensor(2, (int32_t[]){3, 1});
    ATH_REQUIRE_OK(ath_scatter_rows_ref(&src, &idx, &dst));
    static const float want[] = {0, 0, 0, 4, 5, 6, 0, 0, 0, 1, 2, 3};
    for (int i = 0; i < 12; i++)
        ATH_EXPECT(at(&dst, i) == want[i]);
    ath_tensor bad_idx = ids_tensor(2, (int32_t[]){4, 0});
    ATH_EXPECT_STATUS(ath_scatter_rows_ref(&src, &bad_idx, &dst), ATH_ERR_RANGE);
    ath_tensor narrow = F(SHAPE(4, 2), NULL), other = B(SHAPE(4, 3), NULL);
    ATH_EXPECT_STATUS(ath_scatter_rows_ref(&src, &idx, &narrow), ATH_ERR_INVALID_ARG);
    ATH_EXPECT_STATUS(ath_scatter_rows_ref(&src, &idx, &other), ATH_ERR_TYPE);
}

static const ath_test_case tests[] = {
    ATH_TEST_ENTRY(bf16_rounding),
    ATH_TEST_ENTRY(op_argument_checks),
    ATH_TEST_ENTRY(embed_rows),
    ATH_TEST_ENTRY(matmul_small),
    ATH_TEST_ENTRY(matmul_fast_same_bits),
    ATH_TEST_ENTRY(q8_0_f16_and_exp),
    ATH_TEST_ENTRY(matmul_q8_0),
    ATH_TEST_ENTRY(rmsnorm_hand),
    ATH_TEST_ENTRY(rmsnorm_gated_hand),
    ATH_TEST_ENTRY(l2norm_hand),
    ATH_TEST_ENTRY(rope_hand),
    ATH_TEST_ENTRY(act_hand),
    ATH_TEST_ENTRY(attn_hand),
    ATH_TEST_ENTRY(attn_decode_matches_prefill),
    ATH_TEST_ENTRY(conv_hand),
    ATH_TEST_ENTRY(conv_streaming),
    ATH_TEST_ENTRY(gdn_gates_hand),
    ATH_TEST_ENTRY(gdn_hand),
    ATH_TEST_ENTRY(gdn_heads_and_streaming),
    ATH_TEST_ENTRY(scatter_rows),
};

int main(int argc, char **argv)
{
    ath_arena_init(&arena, 0);
    if (ath_pool_init(&pool, 3, &arena) != ATH_OK)
        return 1;
    int rc = ath_test_main(tests, ATH_ARRAY_LEN(tests), argc, argv);
    ath_pool_free(&pool);
    ath_arena_free(&arena);
    return rc;
}
