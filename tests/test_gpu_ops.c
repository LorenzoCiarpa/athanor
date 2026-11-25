/* test_gpu_ops.c: every GPU kernel of src/ops against its CPU reference, on
 * random inputs (the real inputs are tests/test_golden.c's).
 *
 * Each case builds random host tensors, runs ath_<op>_ref on them and
 * ath_<op>_gpu on device copies, and compares the outputs:
 *
 *   score = max |got − want| / (|want| + rms(want)), the measure of
 *   test_golden: a relative error for elements of typical size, an error
 *   relative to the tensor's RMS near zero.
 *
 *   f32 outputs   against a tolerance per op, about 4 times or more the
 *                 worst error measured (tests/README.md, GPU kernels);
 *   bf16 outputs  against n bf16 ulps (n · 2^-7), as test_golden does: both
 *                 sides round an fp32 value once, so they differ by one ulp
 *                 where the two fp32 values straddle a rounding boundary.
 *                 The largest distance in ulps and the number of elements
 *                 that differ are printed too; near zero a distance of
 *                 several ulps is a tiny error, hence the score.
 *
 * Dtypes: f32 throughout (the math, tightly), bf16 throughout (as the
 * forward pass runs), and the mixes the forward pass uses. Shapes are odd
 * on purpose: rows shorter and longer than a block, sizes that are not
 * multiples of a warp, a misaligned matmul operand. Properties that the
 * engine relies on are checked bit for bit: the convolution and the
 * recurrence fed in pieces equal the same tokens at once, and attention for
 * the last token alone equals the last row of the prompt.
 *
 * ATH_GPU_OPS_DETAIL=1 prints every comparison with its measured error. In
 * a build without a GPU backend only the argument checks run.
 */
#include "test.h"

#include "backend/backend.h"
#include "core/arena.h"
#include "ops/ops.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ---- the device and memory ------------------------------------------------- */

typedef struct {
    ath_device    d;
    ath_arena     host;
    ath_dev_arena dev;
} gpu;

static ath_status gpu_open(gpu *g)
{
    ath_arena_init(&g->host, 0);
    ath_status s = ath_device_open(&g->d, 0);
    if (s != ATH_OK)
        return s;
    ath_dev_arena_init(&g->dev, &g->d, &g->host, 0);
    return ATH_OK;
}

static void gpu_close(gpu *g)
{
    ath_dev_arena_free(&g->dev);
    ath_device_close(&g->d);
    ath_arena_free(&g->host);
}

#define GPU_OR_SKIP(g)                                                            \
    do {                                                                          \
        if (ath_gpu_count() == 0)                                                 \
            ATH_SKIP(strcmp(ath_gpu_backend(), "none") == 0                       \
                         ? "built without a GPU backend (make cuda test)"         \
                         : "no usable GPU device (ATH_LOG=info shows why)");      \
        ATH_REQUIRE_OK(gpu_open(&(g)));                                           \
    } while (0)

#define SHAPE(...) (int32_t)(sizeof((int64_t[]){__VA_ARGS__}) / sizeof(int64_t)), (const int64_t[]){__VA_ARGS__}

/* A zeroed host tensor. */
static ath_tensor host(gpu *g, ath_dtype dt, int32_t ndim, const int64_t *shape)
{
    ath_tensor t;
    if (ath_tensor_init(&t, dt, ndim, shape, NULL, ATH_TIER_RAM) != ATH_OK)
        abort();
    t.data = ath_arena_zalloc(&g->host, ath_tensor_nbytes(&t) + 16, 64);
    if (!t.data)
        abort();
    return t;
}

static uint64_t rng = 0x2545f4914f6cdd1dull;

/* Uniform in [lo, hi). */
static float rnd(float lo, float hi)
{
    rng = rng * 6364136223846793005ull + 1442695040888963407ull;
    return lo + (hi - lo) * (float)((rng >> 40) & 0xffffff) / (float)(1 << 24);
}

static ath_tensor rand_t(gpu *g, ath_dtype dt, float lo, float hi, int32_t ndim, const int64_t *shape)
{
    ath_tensor t = host(g, dt, ndim, shape);
    for (int64_t i = 0; i < ath_tensor_numel(&t); i++)
        ath_store_f32(t.data, dt, i, rnd(lo, hi));
    return t;
}

/* A device copy of a host tensor, queued on the compute stream like the ops. */
static ath_tensor up(gpu *g, const ath_tensor *h)
{
    ath_tensor t = *h;
    t.tier = ATH_TIER_VRAM;
    t.data = ath_dev_arena_alloc(&g->dev, ath_tensor_nbytes(h) + 16, 0);
    if (!t.data || ath_copy(&g->d, t.data, h->data, ath_tensor_nbytes(h), ATH_STREAM_COMPUTE) != ATH_OK)
        abort();
    return t;
}

/* The device tensor's values on the host, after everything queued before. */
static ath_tensor down(gpu *g, const ath_tensor *dv)
{
    ath_tensor t = host(g, dv->dtype, dv->ndim, dv->shape);
    if (ath_copy(&g->d, t.data, dv->data, ath_tensor_nbytes(dv), ATH_STREAM_COMPUTE) != ATH_OK ||
        ath_stream_sync(&g->d, ATH_STREAM_COMPUTE) != ATH_OK)
        abort();
    return t;
}

/* ---- comparisons ------------------------------------------------------------- */

static int ulp_distance(float a, float b)
{
    int32_t x = ath_f32_to_bf16(a), y = ath_f32_to_bf16(b);   /* bf16 patterns, ordered like their values */
    x = x & 0x8000 ? 0x8000 - x : x;
    y = y & 0x8000 ? 0x8000 - y : y;
    return x > y ? x - y : y - x;
}

/* GPU output `got` (on the device) against the reference's `want` (host):
 * score at most tol for f32 outputs, at most max_ulp bf16 ulps for bf16. */
static void expect_close(ath_test *t_, gpu *g, const char *what, const ath_tensor *got_dev, const ath_tensor *want,
                         double tol, int max_ulp)
{
    ath_tensor got = down(g, got_dev);
    const int64_t n = ath_tensor_numel(want);
    double ss = 0;
    for (int64_t i = 0; i < n; i++) {
        double w = ath_load_f32(want->data, want->dtype, i);
        ss += w * w;
    }
    const double rms = n ? sqrt(ss / (double)n) : 0;
    double score = 0;
    int ulp = 0;
    int64_t worst = 0, differ = 0;
    for (int64_t i = 0; i < n; i++) {
        const float w = ath_load_f32(want->data, want->dtype, i), x = ath_load_f32(got.data, got.dtype, i);
        const double e = fabs((double)x - (double)w);
        const double s = e == 0 ? 0 : e / (fabs((double)w) + rms);
        if (!(s <= score)) {                                    /* NaN too */
            score = isnan(s) ? INFINITY : s;
            worst = i;
        }
        if (want->dtype == ATH_DTYPE_BF16) {
            const int u = ulp_distance(x, w);
            differ += u > 0;
            if (u > ulp)
                ulp = u;
        }
    }
    const bool bf16 = want->dtype == ATH_DTYPE_BF16;
    const bool ok = score <= (bf16 ? max_ulp / 128.0 : tol);
    if (getenv("ATH_GPU_OPS_DETAIL") || !ok) {
        if (bf16)
            printf("    %-44s bf16: score %.2e, tol %d ulp; %d ulp max, %lld of %lld differ\n", what, score, max_ulp,
                   ulp, (long long)differ, (long long)n);
        else
            printf("    %-44s f32:  score %.2e at [%lld], tol %.0e\n", what, score, (long long)worst, tol);
    }
    if (!ok) {
        char msg[160];
        snprintf(msg, sizeof msg, "%s: outside tolerance", what);
        ath_test_fail_(t_, __FILE__, __LINE__, msg);
    }
}

#define EXPECT_CLOSE(g, what, got, want, tol, ulps) expect_close(t_, &(g), (what), &(got), &(want), (tol), (ulps))

static bool same_bits(gpu *g, const ath_tensor *a_dev, const ath_tensor *b_dev)
{
    ath_tensor a = down(g, a_dev), b = down(g, b_dev);
    return ath_tensor_nbytes(&a) == ath_tensor_nbytes(&b) && memcmp(a.data, b.data, ath_tensor_nbytes(&a)) == 0;
}

/* The two dtype configurations every op runs in besides f32 throughout. */
static const char *dt_name(ath_dtype dt) { return dt == ATH_DTYPE_BF16 ? "bf16" : "f32"; }

/* ---- argument checks (both builds) ---------------------------------------------- */

/* A descriptor that claims device memory; never dereferenced, because each
 * call below fails before anything is launched. */
static ath_tensor fake(ath_dtype dt, ath_tier tier, int32_t ndim, const int64_t *shape)
{
    static char byte;
    ath_tensor t;
    if (ath_tensor_init(&t, dt, ndim, shape, &byte, tier) != ATH_OK)
        abort();
    return t;
}

ATH_TEST(gpu_argument_checks)
{
    const ath_device *d = NULL;   /* the checks come before the device is used */
    /* host tensors to a GPU kernel, device tensors to a reference */
    ath_tensor xh = fake(ATH_DTYPE_F32, ATH_TIER_RAM, SHAPE(2, 8)), oh = fake(ATH_DTYPE_F32, ATH_TIER_RAM, SHAPE(2, 8));
    ath_tensor xv = fake(ATH_DTYPE_F32, ATH_TIER_VRAM, SHAPE(2, 8)), ov = fake(ATH_DTYPE_F32, ATH_TIER_VRAM, SHAPE(2, 8));
    ATH_EXPECT_STATUS(ath_l2norm_gpu(d, &xh, 1e-6f, &ov), ATH_ERR_UNSUPPORTED);
    ATH_EXPECT_STATUS(ath_l2norm_gpu(d, &xv, 1e-6f, &oh), ATH_ERR_UNSUPPORTED);
    ATH_EXPECT_STATUS(ath_l2norm_ref(&xv, 1e-6f, &ov), ATH_ERR_UNSUPPORTED);
    ATH_EXPECT_STATUS(ath_add_gpu(d, &xv, &xh, &ov), ATH_ERR_UNSUPPORTED);
    /* shapes are checked as by the reference */
    ath_tensor bad = fake(ATH_DTYPE_F32, ATH_TIER_VRAM, SHAPE(2, 7));
    ATH_EXPECT_STATUS(ath_l2norm_gpu(d, &xv, 1e-6f, &bad), ATH_ERR_INVALID_ARG);
    ath_tensor wv = fake(ATH_DTYPE_BF16, ATH_TIER_VRAM, SHAPE(5, 7));
    ATH_EXPECT_STATUS(ath_matmul_gpu(d, &xv, &wv, &ov), ATH_ERR_INVALID_ARG);
    ath_tensor iv = fake(ATH_DTYPE_I32, ATH_TIER_VRAM, SHAPE(2, 8));
    ATH_EXPECT_STATUS(ath_add_gpu(d, &iv, &xv, &ov), ATH_ERR_TYPE);

    /* limits of the kernels beyond the reference's */
    ath_tensor q = fake(ATH_DTYPE_BF16, ATH_TIER_VRAM, SHAPE(1, 1, ATH_ATTN_MAX_DV + 1));
    ath_tensor v = fake(ATH_DTYPE_BF16, ATH_TIER_VRAM, SHAPE(1, 1, 64)), o = fake(ATH_DTYPE_BF16, ATH_TIER_VRAM, SHAPE(1, 1, 64));
    ATH_EXPECT_STATUS(ath_attn_gpu(d, &q, &q, &v, NULL, 1.0f, NULL, &o), ATH_ERR_UNSUPPORTED);
    ath_tensor gq = fake(ATH_DTYPE_F32, ATH_TIER_VRAM, SHAPE(1, 1, ATH_GDN_MAX_DK + 1));
    ath_tensor gv = fake(ATH_DTYPE_F32, ATH_TIER_VRAM, SHAPE(1, 1, 8)), gg = fake(ATH_DTYPE_F32, ATH_TIER_VRAM, SHAPE(1, 1));
    ath_tensor gs = fake(ATH_DTYPE_F32, ATH_TIER_VRAM, SHAPE(1, ATH_GDN_MAX_DK + 1, 8));
    ATH_EXPECT_STATUS(ath_gdn_gpu(d, &gq, &gq, &gv, &gg, &gg, 1.0f, 1e-6f, false, &gs, NULL, &gv), ATH_ERR_UNSUPPORTED);
    ath_tensor rx = fake(ATH_DTYPE_F32, ATH_TIER_VRAM, SHAPE(1, 1, ATH_ROPE_MAX_DIM + 2));
    ath_tensor rp = fake(ATH_DTYPE_I32, ATH_TIER_VRAM, SHAPE(1));
    ATH_EXPECT_STATUS(ath_rope_gpu(d, &rx, &rp, ATH_ROPE_MAX_DIM + 2, 1e4f, &rx), ATH_ERR_UNSUPPORTED);

    /* without a GPU backend, valid arguments end in the stub */
    if (strcmp(ath_gpu_backend(), "none") == 0)
        ATH_EXPECT_STATUS(ath_l2norm_gpu(d, &xv, 1e-6f, &ov), ATH_ERR_UNSUPPORTED);
}

/* ---- embed ---------------------------------------------------------------------- */

ATH_TEST(gpu_embed)
{
    gpu g;
    GPU_OR_SKIP(g);
    const int32_t id_vals[] = {0, 36, 5, 5, 17, 1, 30, 2, 36};
    ath_tensor ids = host(&g, ATH_DTYPE_I32, SHAPE(9));
    memcpy(ids.data, id_vals, sizeof id_vals);
    const ath_dtype dts[] = {ATH_DTYPE_F32, ATH_DTYPE_BF16};
    for (int a = 0; a < 2; a++)
        for (int b = 0; b < 2; b++) {
            ath_tensor table = rand_t(&g, dts[a], -1, 1, SHAPE(37, 130));
            ath_tensor want = host(&g, dts[b], SHAPE(9, 130));
            ATH_REQUIRE_OK(ath_embed_ref(&table, &ids, &want));
            ath_tensor td = up(&g, &table), idd = up(&g, &ids), out = up(&g, &want);
            ATH_EXPECT_OK(ath_memset(&g.d, out.data, 0xff, ath_tensor_nbytes(&out), ATH_STREAM_COMPUTE));
            ATH_EXPECT_OK(ath_embed_gpu(&g.d, &td, &idd, &out));
            ath_tensor got = down(&g, &out);
            ATH_EXPECT(memcmp(got.data, want.data, ath_tensor_nbytes(&want)) == 0);   /* a copy: exact */
        }
    /* an id outside the table: a row of NaN, the other rows intact */
    ath_tensor table = rand_t(&g, ATH_DTYPE_BF16, -1, 1, SHAPE(37, 130));
    const int32_t bad_vals[] = {3, 37, -1};
    ath_tensor bad = host(&g, ATH_DTYPE_I32, SHAPE(3));
    memcpy(bad.data, bad_vals, sizeof bad_vals);
    ath_tensor rows = host(&g, ATH_DTYPE_F32, SHAPE(3, 130));
    ath_tensor td = up(&g, &table), bd = up(&g, &bad), out = up(&g, &rows);
    ATH_EXPECT_OK(ath_embed_gpu(&g.d, &td, &bd, &out));
    ath_tensor got = down(&g, &out);
    for (int64_t j = 0; j < 130; j++) {
        ATH_EXPECT(ath_load_f32(got.data, got.dtype, j) == ath_load_f32(table.data, table.dtype, 3 * 130 + j));
        ATH_EXPECT(isnan(ath_load_f32(got.data, got.dtype, 130 + j)));
        ATH_EXPECT(isnan(ath_load_f32(got.data, got.dtype, 260 + j)));
    }
    gpu_close(&g);
}

/* ---- matmul ---------------------------------------------------------------------- */

ATH_TEST(gpu_matmul)
{
    gpu g;
    GPU_OR_SKIP(g);
    static const struct { int64_t N, K, M; } shapes[] = {
        {1, 1024, 3584},   /* one decoded token through an MLP projection */
        {7, 1024, 1000},
        {75, 333, 129},    /* nothing a multiple of anything */
        {3, 64, 5},
    };
    static const struct { ath_dtype x, w, out; const char *path; } kinds[] = {
        {ATH_DTYPE_BF16, ATH_DTYPE_BF16, ATH_DTYPE_BF16, "cuBLASLt"},
        {ATH_DTYPE_BF16, ATH_DTYPE_BF16, ATH_DTYPE_F32,  "cuBLASLt"},
        {ATH_DTYPE_F32,  ATH_DTYPE_F32,  ATH_DTYPE_F32,  "cuBLASLt"},
        {ATH_DTYPE_F32,  ATH_DTYPE_BF16, ATH_DTYPE_F32,  "own kernel"},
        {ATH_DTYPE_F32,  ATH_DTYPE_F32,  ATH_DTYPE_BF16, "own kernel"},
    };
    const ath_dev_arena_pos mark = ath_dev_arena_mark(&g.dev);
    for (size_t s = 0; s < ATH_ARRAY_LEN(shapes); s++)
        for (size_t k = 0; k < ATH_ARRAY_LEN(kinds); k++) {
            const int64_t N = shapes[s].N, K = shapes[s].K, M = shapes[s].M;
            /* x starts one row into a bigger buffer: an operand cuBLASLt cannot assume aligned */
            ath_tensor xbig = rand_t(&g, kinds[k].x, -1, 1, SHAPE(N + 1, K)), x;
            ATH_REQUIRE_OK(ath_tensor_slice(&xbig, 0, 1, N, &x));
            ath_tensor w = rand_t(&g, kinds[k].w, -1, 1, SHAPE(M, K));
            ath_tensor want = host(&g, kinds[k].out, SHAPE(N, M));
            ATH_REQUIRE_OK(ath_matmul_ref(&x, &w, &want));
            ath_tensor xbd = up(&g, &xbig), xd, wd = up(&g, &w), out = up(&g, &want);
            ATH_REQUIRE_OK(ath_tensor_slice(&xbd, 0, 1, N, &xd));
            ATH_EXPECT_OK(ath_matmul_gpu(&g.d, &xd, &wd, &out));
            char what[96];
            /* the GEMV kernel takes a few rows of 16-byte pieces against bf16 weights (step 9) */
            const bool gemv = N <= 8 && kinds[k].w == ATH_DTYPE_BF16 && K % 8 == 0;
            snprintf(what, sizeof what, "matmul %lldx%lldx%lld %s·%s %s", (long long)N, (long long)K, (long long)M,
                     dt_name(kinds[k].x), dt_name(kinds[k].w), gemv ? "GEMV" : kinds[k].path);
            EXPECT_CLOSE(g, what, out, want, 2e-5, 1);
            ath_dev_arena_reset(&g.dev, mark);   /* the comparison has waited for the stream */
        }
    gpu_close(&g);
}

/* ---- Q8_0 (step 12) ------------------------------------------------------------ */

/* A random Q8_0 [M, K] weight on the host: f16 scales between 2^-10 and
 * 2^-6, int8 values in [-127, 127]. */
static ath_tensor rand_q8_0(gpu *g, int64_t M, int64_t K)
{
    ath_tensor t = host(g, ATH_DTYPE_Q8_0, SHAPE(M, K));
    uint8_t *p = (uint8_t *)t.data;
    for (int64_t b = 0; b < M * K / 32; b++, p += 34) {
        const uint16_t h = (uint16_t)(((5 + (b % 5)) << 10) | ((uint16_t)rnd(0, 1023) & 0x3ff));
        p[0] = (uint8_t)h;
        p[1] = (uint8_t)(h >> 8);
        for (int i = 0; i < 32; i++)
            p[2 + i] = (uint8_t)(int8_t)rnd(-127, 128);
    }
    return t;
}

/* The canonical order (src/ops/q8_0.h): every Q8_0 kernel gives the
 * reference's bits, for any number of rows: one row (decode), a few, a
 * prompt in passes of 8; K a multiple of 512 (the staged kernel) or not
 * (the one that reads its own blocks); x in f32 and bf16, out in both; x off
 * a 16-byte boundary. And the embedding lookup of a Q8_0 table. */
ATH_TEST(gpu_matmul_q8_0)
{
    gpu g;
    GPU_OR_SKIP(g);
    static const struct { int64_t N, K, M; } shapes[] = {
        {1, 2048, 512}, {1, 512, 2048}, {3, 4096, 77}, {8, 1024, 33}, {21, 2048, 64}, {1, 96, 5}, {5, 160, 9},
    };
    static const ath_dtype dts[] = {ATH_DTYPE_BF16, ATH_DTYPE_F32};
    const ath_dev_arena_pos mark = ath_dev_arena_mark(&g.dev);
    for (size_t s = 0; s < ATH_ARRAY_LEN(shapes); s++)
        for (int c = 0; c < 4; c++) {
            const int64_t N = shapes[s].N, K = shapes[s].K, M = shapes[s].M;
            ath_tensor xbig = rand_t(&g, dts[c & 1], -1, 1, SHAPE(N + 1, K)), x;
            ATH_REQUIRE_OK(ath_tensor_slice(&xbig, 0, c == 3 ? 1 : 0, N, &x));
            ath_tensor w = rand_q8_0(&g, M, K);
            ath_tensor want = host(&g, dts[c >> 1], SHAPE(N, M));
            ATH_REQUIRE_OK(ath_matmul_ref(&x, &w, &want));
            ath_tensor xbd = up(&g, &xbig), xd, wd = up(&g, &w), out = up(&g, &want);
            ATH_REQUIRE_OK(ath_tensor_slice(&xbd, 0, c == 3 ? 1 : 0, N, &xd));
            ATH_EXPECT_OK(ath_memset(&g.d, out.data, 0xff, ath_tensor_nbytes(&out), ATH_STREAM_COMPUTE));
            ATH_EXPECT_OK(ath_matmul_gpu(&g.d, &xd, &wd, &out));
            ath_tensor got = down(&g, &out);
            if (memcmp(got.data, want.data, ath_tensor_nbytes(&want)) != 0) {
                char msg[128];
                snprintf(msg, sizeof msg, "q8_0 matmul %lldx%lldx%lld x %s out %s: not the reference's bits",
                         (long long)N, (long long)K, (long long)M, dt_name(dts[c & 1]), dt_name(dts[c >> 1]));
                ath_test_fail_(t_, __FILE__, __LINE__, msg);
            }
            ath_dev_arena_reset(&g.dev, mark);
        }

    /* embedding rows of a Q8_0 table */
    const int32_t id_vals[] = {0, 36, 5, 5, 17};
    ath_tensor ids = host(&g, ATH_DTYPE_I32, SHAPE(5));
    memcpy(ids.data, id_vals, sizeof id_vals);
    ath_tensor table = rand_q8_0(&g, 37, 96);
    for (int b = 0; b < 2; b++) {
        ath_tensor want = host(&g, dts[b], SHAPE(5, 96));
        ATH_REQUIRE_OK(ath_embed_ref(&table, &ids, &want));
        ath_tensor td = up(&g, &table), idd = up(&g, &ids), out = up(&g, &want);
        ATH_EXPECT_OK(ath_embed_gpu(&g.d, &td, &idd, &out));
        ath_tensor got = down(&g, &out);
        ATH_EXPECT(memcmp(got.data, want.data, ath_tensor_nbytes(&want)) == 0);
    }
    gpu_close(&g);
}

/* The GEMV kernel (step 9): every row count it takes, K from one 16-byte
 * piece to more than a lane's four, odd M, both activation dtypes into
 * both output dtypes; an x that is not on a 16-byte boundary goes to the
 * other kernels and gives the same answer within the tolerance; and a row
 * gives the same bits alone as among others, since the order of its sum
 * depends on K only. */
ATH_TEST(gpu_gemv)
{
    gpu g;
    GPU_OR_SKIP(g);
    static const int64_t Ks[] = {8, 1024, 1032, 3584};
    static const int64_t Ms[] = {1, 37, 1000};
    static const ath_dtype xs[] = {ATH_DTYPE_BF16, ATH_DTYPE_F32};
    static const ath_dtype outs[] = {ATH_DTYPE_BF16, ATH_DTYPE_F32};
    const ath_dev_arena_pos mark = ath_dev_arena_mark(&g.dev);
    for (size_t ki = 0; ki < ATH_ARRAY_LEN(Ks); ki++)
        for (size_t mi = 0; mi < ATH_ARRAY_LEN(Ms); mi++)
            for (size_t xi = 0; xi < ATH_ARRAY_LEN(xs); xi++)
                for (size_t oi = 0; oi < ATH_ARRAY_LEN(outs); oi++)
                    for (int64_t N = 1; N <= 9; N++) {
                        const int64_t K = Ks[ki], M = Ms[mi];
                        ath_tensor x = rand_t(&g, xs[xi], -1, 1, SHAPE(N, K));
                        ath_tensor w = rand_t(&g, ATH_DTYPE_BF16, -1, 1, SHAPE(M, K));
                        ath_tensor want = host(&g, outs[oi], SHAPE(N, M));
                        ATH_REQUIRE_OK(ath_matmul_ref(&x, &w, &want));
                        ath_tensor xd = up(&g, &x), wd = up(&g, &w), out = up(&g, &want);
                        ATH_EXPECT_OK(ath_matmul_gpu(&g.d, &xd, &wd, &out));
                        char what[96];
                        snprintf(what, sizeof what, "gemv %lldx%lldx%lld %s→%s%s", (long long)N, (long long)K,
                                 (long long)M, dt_name(xs[xi]), dt_name(outs[oi]), N > 8 ? " (9 rows: not GEMV)" : "");
                        EXPECT_CLOSE(g, what, out, want, 2e-5, 1);
                        if (N > 1 && N <= 8) {   /* the last row alone: the same bits */
                            ath_tensor xl, ol, one = host(&g, outs[oi], SHAPE(1, M));
                            ATH_REQUIRE_OK(ath_tensor_slice(&xd, 0, N - 1, 1, &xl));
                            ath_tensor od = up(&g, &one);
                            ATH_EXPECT_OK(ath_matmul_gpu(&g.d, &xl, &wd, &od));
                            ATH_REQUIRE_OK(ath_tensor_slice(&out, 0, N - 1, 1, &ol));
                            ATH_EXPECT(same_bits(&g, &od, &ol));
                        }
                        ath_dev_arena_reset(&g.dev, mark);
                    }
    /* x four bytes past a 16-byte boundary: not the GEMV kernel, the same answer */
    {
        const int64_t N = 2, K = 1024, M = 333;
        ath_tensor xbig = rand_t(&g, ATH_DTYPE_F32, -1, 1, SHAPE(N * K + 1)), x;
        ath_tensor flat;
        ATH_REQUIRE_OK(ath_tensor_slice(&xbig, 0, 1, N * K, &flat));
        ATH_REQUIRE_OK(ath_tensor_reshape(&flat, 2, (const int64_t[]){N, K}, &x));
        ath_tensor w = rand_t(&g, ATH_DTYPE_BF16, -1, 1, SHAPE(M, K));
        ath_tensor want = host(&g, ATH_DTYPE_F32, SHAPE(N, M));
        ATH_REQUIRE_OK(ath_matmul_ref(&x, &w, &want));
        ath_tensor xbd = up(&g, &xbig), xfd, xd, wd = up(&g, &w), out = up(&g, &want);
        ATH_REQUIRE_OK(ath_tensor_slice(&xbd, 0, 1, N * K, &xfd));
        ATH_REQUIRE_OK(ath_tensor_reshape(&xfd, 2, (const int64_t[]){N, K}, &xd));
        ATH_EXPECT_OK(ath_matmul_gpu(&g.d, &xd, &wd, &out));
        EXPECT_CLOSE(g, "gemv, x not on a 16-byte boundary (own kernel)", out, want, 2e-5, 1);
        ath_dev_arena_reset(&g.dev, mark);
    }
    gpu_close(&g);
}

/* ---- sampling (step 9) ----------------------------------------------------------------- */

/* GPU sampling against ath_sample_ref on the same logits and u: greedy and
 * top-k samplers with temperature, top-p and min-p, vocabularies from 7
 * tokens to the 0.8B model's, logits with many ties (rounded to a coarse
 * grid) and without, u across [0, 1). */
ATH_TEST(gpu_sample)
{
    gpu g;
    GPU_OR_SKIP(g);
    static const int64_t Vs[] = {7, 1000, 248320};
    static const ath_sampler ps[] = {
        {0, 0, 1, 0},          {1, 1, 1, 0},           {0.7f, 20, 0.8f, 0},
        {0.6f, 20, 0.95f, 0},  {1, 64, 1, 0.05f},      {1.3f, 5, 0.5f, 0.2f},
    };
    static const double us[] = {0, 0.05, 0.25, 0.5, 0.75, 0.9, 0.999, 0.3333, 0.6667, 0.123};
    int64_t cases = 0, differ = 0;
    for (size_t vi = 0; vi < ATH_ARRAY_LEN(Vs); vi++)
        for (int ties = 0; ties < 2; ties++) {
            const int64_t V = Vs[vi];
            ath_tensor lg = rand_t(&g, ATH_DTYPE_F32, -8, 8, SHAPE(1, V));
            for (int64_t i = 0; ties && i < V; i++)   /* a coarse grid: many equal logits */
                ((float *)lg.data)[i] = roundf(((float *)lg.data)[i] * 2) / 2;
            const ath_dev_arena_pos mark = ath_dev_arena_mark(&g.dev);
            ath_tensor ld = up(&g, &lg);
            double *ud = ath_dev_arena_alloc(&g.dev, sizeof(double), 0);
            int32_t *od = ath_dev_arena_alloc(&g.dev, sizeof(int32_t), 0);
            ATH_REQUIRE(ud && od);
            for (size_t pi = 0; pi < ATH_ARRAY_LEN(ps); pi++)
                for (size_t ui = 0; ui < ATH_ARRAY_LEN(us); ui++) {
                    int32_t want = -1, got = -2;
                    ATH_REQUIRE_OK(ath_sample_ref(&lg, &ps[pi], us[ui], &g.host, &want));
                    ATH_REQUIRE_OK(ath_copy(&g.d, ud, &us[ui], sizeof(double), ATH_STREAM_COMPUTE));
                    ATH_EXPECT_OK(ath_sample_gpu(&g.d, &ld, &ps[pi], ud, &g.dev, od));
                    ATH_REQUIRE_OK(ath_copy(&g.d, &got, od, sizeof got, ATH_STREAM_COMPUTE));
                    ATH_REQUIRE_OK(ath_stream_sync(&g.d, ATH_STREAM_COMPUTE));
                    cases++;
                    if (got != want) {
                        differ++;
                        printf("    sample V %lld ties %d sampler %zu u %g: GPU %d, reference %d\n", (long long)V,
                               ties, pi, us[ui], got, want);
                    }
                }
            ath_dev_arena_reset(&g.dev, mark);
        }
    printf("    sampling: %lld cases, %lld differ from the reference\n", (long long)cases, (long long)differ);
    ATH_EXPECT(differ == 0);
    /* every token a candidate: not on the GPU */
    ath_tensor lg = rand_t(&g, ATH_DTYPE_F32, -1, 1, SHAPE(1, 100)), ld = up(&g, &lg);
    double *ud = ath_dev_arena_alloc(&g.dev, sizeof(double), 0);
    int32_t *od = ath_dev_arena_alloc(&g.dev, sizeof(int32_t), 0);
    const ath_sampler all = {0.7f, 0, 0.9f, 0}, big = {0.7f, ATH_SAMPLE_GPU_MAX_K + 1, 1, 0};
    ATH_EXPECT(!ath_sample_gpu_fits(&all) && !ath_sample_gpu_fits(&big));
    ATH_EXPECT_STATUS(ath_sample_gpu(&g.d, &ld, &all, ud, &g.dev, od), ATH_ERR_UNSUPPORTED);
    gpu_close(&g);
}

/* Several weights sharing x (step 9): one GEMV launch, the bits of one
 * call per weight; and with too many rows for the GEMV kernel, the same. */
ATH_TEST(gpu_matmul_n)
{
    gpu g;
    GPU_OR_SKIP(g);
    static const int64_t Ms[] = {6144, 2048, 16, 16};
    static const ath_dtype odts[] = {ATH_DTYPE_BF16, ATH_DTYPE_F32, ATH_DTYPE_BF16, ATH_DTYPE_F32};
    for (int64_t N = 1; N <= 12; N += 11)
        for (int n = 1; n <= ATH_MATMUL_N_MAX; n++) {
            const int64_t K = 1024;
            ath_tensor x = rand_t(&g, ATH_DTYPE_BF16, -1, 1, SHAPE(N, K)), xd = up(&g, &x);
            ath_tensor wd[ATH_MATMUL_N_MAX], od[ATH_MATMUL_N_MAX], sd[ATH_MATMUL_N_MAX];
            const ath_tensor *wp[ATH_MATMUL_N_MAX];
            ath_tensor *op[ATH_MATMUL_N_MAX];
            for (int i = 0; i < n; i++) {
                ath_tensor w = rand_t(&g, ATH_DTYPE_BF16, -1, 1, SHAPE(Ms[i], K)), o = host(&g, odts[i], SHAPE(N, Ms[i]));
                wd[i] = up(&g, &w), od[i] = up(&g, &o), sd[i] = up(&g, &o);
                wp[i] = &wd[i], op[i] = &od[i];
                ATH_EXPECT_OK(ath_matmul_gpu(&g.d, &xd, &wd[i], &sd[i]));
            }
            ATH_EXPECT_OK(ath_matmul_n_gpu(&g.d, &xd, n, wp, op));
            for (int i = 0; i < n; i++)
                ATH_EXPECT(same_bits(&g, &od[i], &sd[i]));
        }
    ath_tensor x = rand_t(&g, ATH_DTYPE_BF16, -1, 1, SHAPE(1, 8)), xd = up(&g, &x);
    ATH_EXPECT_STATUS(ath_matmul_n_gpu(&g.d, &xd, ATH_MATMUL_N_MAX + 1, NULL, NULL), ATH_ERR_INVALID_ARG);
    gpu_close(&g);
}

/* Gate, up and SwiGLU in one launch (step 9): the bits of the two ops. */
ATH_TEST(gpu_matmul_swiglu)
{
    gpu g;
    GPU_OR_SKIP(g);
    static const ath_dtype dts[] = {ATH_DTYPE_BF16, ATH_DTYPE_F32};
    for (size_t i = 0; i < ATH_ARRAY_LEN(dts); i++) {
        const int64_t K = 1024, M = 3584;
        ath_tensor x = rand_t(&g, ATH_DTYPE_BF16, -1, 1, SHAPE(1, K)), xd = up(&g, &x);
        ath_tensor wg = rand_t(&g, ATH_DTYPE_BF16, -1, 1, SHAPE(M, K)), wu = rand_t(&g, ATH_DTYPE_BF16, -1, 1, SHAPE(M, K));
        ath_tensor o = host(&g, dts[i], SHAPE(1, M));
        ath_tensor wgd = up(&g, &wg), wud = up(&g, &wu), gd = up(&g, &o), ud = up(&g, &o), hd = up(&g, &o), fd = up(&g, &o);
        const ath_tensor *wp[] = {&wgd, &wud};
        ath_tensor *op[] = {&gd, &ud};
        ATH_REQUIRE(ath_matmul_swiglu_fits(&xd, &wgd, &wud));
        ATH_EXPECT_OK(ath_matmul_n_gpu(&g.d, &xd, 2, wp, op));
        ATH_EXPECT_OK(ath_swiglu_gpu(&g.d, &gd, &ud, &hd));
        ATH_EXPECT_OK(ath_matmul_swiglu_gpu(&g.d, &xd, &wgd, &wud, &fd));
        ATH_EXPECT(same_bits(&g, &fd, &hd));
    }
    ath_tensor x2 = rand_t(&g, ATH_DTYPE_BF16, -1, 1, SHAPE(2, 64)), w2 = rand_t(&g, ATH_DTYPE_BF16, -1, 1, SHAPE(8, 64));
    ath_tensor x2d = up(&g, &x2), w2d = up(&g, &w2);
    ATH_EXPECT(!ath_matmul_swiglu_fits(&x2d, &w2d, &w2d));   /* two rows: not this kernel */
    gpu_close(&g);
}

/* ---- norms ----------------------------------------------------------------------- */

/* The residual add fused into the next norm (step 10): x and out with the
 * bits of ath_add_gpu followed by ath_rmsnorm_gpu, in bf16 and f32, rows of
 * 1024 and of 128 (two rows to a warp), and a row count that leaves a warp
 * half empty. */
ATH_TEST(gpu_add_rmsnorm)
{
    gpu g;
    GPU_OR_SKIP(g);
    static const struct { ath_dtype dt; int64_t rows, D; } cases[] = {
        {ATH_DTYPE_BF16, 37, 1024}, {ATH_DTYPE_F32, 37, 1024}, {ATH_DTYPE_BF16, 33, 128}, {ATH_DTYPE_BF16, 5, 64}};
    for (size_t i = 0; i < ATH_ARRAY_LEN(cases); i++) {
        const ath_dtype dt = cases[i].dt;
        const int64_t R = cases[i].rows, D = cases[i].D;
        ath_tensor x = rand_t(&g, dt, -2, 2, SHAPE(R, D)), y = rand_t(&g, dt, -2, 2, SHAPE(R, D));
        ath_tensor w = rand_t(&g, ATH_DTYPE_BF16, -1, 1, SHAPE(D)), o = host(&g, dt, SHAPE(R, D));
        ath_tensor x1 = up(&g, &x), x2 = up(&g, &x), yd = up(&g, &y), wd = up(&g, &w), o1 = up(&g, &o), o2 = up(&g, &o);
        ATH_EXPECT(ath_add_rmsnorm_fits(&x1, &yd, &wd));
        ATH_EXPECT_OK(ath_add_gpu(&g.d, &x1, &yd, &x1));
        ATH_EXPECT_OK(ath_rmsnorm_gpu(&g.d, &x1, &wd, 1.0f, 1e-6f, &o1));
        ATH_EXPECT_OK(ath_add_rmsnorm_gpu(&g.d, &x2, &yd, &wd, 1.0f, 1e-6f, &o2));
        ATH_EXPECT(same_bits(&g, &x1, &x2));
        ATH_EXPECT(same_bits(&g, &o1, &o2));
    }
    gpu_close(&g);
}

ATH_TEST(gpu_norms)
{
    gpu g;
    GPU_OR_SKIP(g);
    const ath_dtype cfg[][2] = {{ATH_DTYPE_F32, ATH_DTYPE_F32}, {ATH_DTYPE_BF16, ATH_DTYPE_BF16},
                                {ATH_DTYPE_BF16, ATH_DTYPE_F32}};
    for (int c = 0; c < 3; c++) {
        const ath_dtype in = cfg[c][0], ot = cfg[c][1];
        char what[96];
        for (int offset = 0; offset < 2; offset++) {
            ath_tensor x = rand_t(&g, in, -2, 2, SHAPE(5, 1000)), w = rand_t(&g, ATH_DTYPE_BF16, -1, 1, SHAPE(1000));
            ath_tensor want = host(&g, ot, SHAPE(5, 1000));
            ATH_REQUIRE_OK(ath_rmsnorm_ref(&x, &w, (float)offset, 1e-6f, &want));
            ath_tensor xd = up(&g, &x), wd = up(&g, &w), out = up(&g, &want);
            ATH_EXPECT_OK(ath_rmsnorm_gpu(&g.d, &xd, &wd, (float)offset, 1e-6f, &out));
            snprintf(what, sizeof what, "rmsnorm w_offset %d, %s in", offset, dt_name(in));
            EXPECT_CLOSE(g, what, out, want, 5e-6, 1);
        }
        {
            ath_tensor x = rand_t(&g, in, -2, 2, SHAPE(17, 128)), z = rand_t(&g, in, -4, 4, SHAPE(17, 128));
            ath_tensor w = rand_t(&g, ATH_DTYPE_F32, 0.5f, 1.5f, SHAPE(128)), want = host(&g, ot, SHAPE(17, 128));
            ATH_REQUIRE_OK(ath_rmsnorm_gated_ref(&x, &z, &w, 1e-6f, &want));
            ath_tensor xd = up(&g, &x), zd = up(&g, &z), wd = up(&g, &w), out = up(&g, &want);
            ATH_EXPECT_OK(ath_rmsnorm_gated_gpu(&g.d, &xd, &zd, &wd, 1e-6f, &out));
            snprintf(what, sizeof what, "rmsnorm_gated, %s in", dt_name(in));
            /* with bf16 x the normalized value is rounded to bf16 first: one
             * flipped rounding there is up to 2^-8 of an output */
            EXPECT_CLOSE(g, what, out, want, in == ATH_DTYPE_BF16 ? 1.0 / 256 : 5e-6, 2);
        }
        {
            ath_tensor x = rand_t(&g, in, -2, 2, SHAPE(9, 7, 100)), want = host(&g, ot, SHAPE(9, 7, 100));
            ATH_REQUIRE_OK(ath_l2norm_ref(&x, 1e-6f, &want));
            ath_tensor xd = up(&g, &x), out = up(&g, &want);
            ATH_EXPECT_OK(ath_l2norm_gpu(&g.d, &xd, 1e-6f, &out));
            snprintf(what, sizeof what, "l2norm, %s in", dt_name(in));
            EXPECT_CLOSE(g, what, out, want, 5e-6, 1);
        }
    }
    gpu_close(&g);
}

/* ---- rope -------------------------------------------------------------------------- */

ATH_TEST(gpu_rope)
{
    gpu g;
    GPU_OR_SKIP(g);
    /* long positions too: the frequencies must be the reference's, bit for bit */
    const int32_t pv[] = {0, 1, 5, 1000, 100000, 262143};
    ath_tensor pos = host(&g, ATH_DTYPE_I32, SHAPE(6));
    memcpy(pos.data, pv, sizeof pv);
    const ath_dtype cfg[][2] = {{ATH_DTYPE_F32, ATH_DTYPE_F32}, {ATH_DTYPE_BF16, ATH_DTYPE_BF16},
                                {ATH_DTYPE_BF16, ATH_DTYPE_F32}};
    const int32_t dims[] = {64, 256};
    for (int c = 0; c < 3; c++)
        for (int r = 0; r < 2; r++) {
            ath_tensor x = rand_t(&g, cfg[c][0], -2, 2, SHAPE(6, 3, 256)), want = host(&g, cfg[c][1], SHAPE(6, 3, 256));
            ATH_REQUIRE_OK(ath_rope_ref(&x, &pos, dims[r], 1e7f, &want));
            ath_tensor xd = up(&g, &x), pd = up(&g, &pos), out = up(&g, &want);
            ATH_EXPECT_OK(ath_rope_gpu(&g.d, &xd, &pd, dims[r], 1e7f, &out));
            char what[96];
            snprintf(what, sizeof what, "rope rope_dim %d, %s in", dims[r], dt_name(cfg[c][0]));
            /* bf16 tables: cos and sin rounded to bf16, a flip there is 2^-8 */
            EXPECT_CLOSE(g, what, out, want, cfg[c][0] == ATH_DTYPE_BF16 ? 1.0 / 256 : 1e-6, 2);
        }
    gpu_close(&g);
}

/* ---- elementwise -------------------------------------------------------------------- */

ATH_TEST(gpu_elementwise)
{
    gpu g;
    GPU_OR_SKIP(g);
    const ath_dtype cfg[][2] = {{ATH_DTYPE_F32, ATH_DTYPE_F32}, {ATH_DTYPE_BF16, ATH_DTYPE_BF16},
                                {ATH_DTYPE_BF16, ATH_DTYPE_F32}};
    for (int c = 0; c < 3; c++) {
        ath_tensor a = rand_t(&g, cfg[c][0], -8, 8, SHAPE(3, 1001)), b = rand_t(&g, cfg[c][0], -8, 8, SHAPE(3, 1001));
        ath_tensor want = host(&g, cfg[c][1], SHAPE(3, 1001));
        ath_tensor ad = up(&g, &a), bd = up(&g, &b), out = up(&g, &want);
        char what[96];
        ATH_REQUIRE_OK(ath_swiglu_ref(&a, &b, &want));
        ATH_EXPECT_OK(ath_swiglu_gpu(&g.d, &ad, &bd, &out));
        snprintf(what, sizeof what, "swiglu, %s in", dt_name(cfg[c][0]));
        EXPECT_CLOSE(g, what, out, want, 1e-6, 1);
        ATH_REQUIRE_OK(ath_sigmoid_mul_ref(&a, &b, &want));
        ATH_EXPECT_OK(ath_sigmoid_mul_gpu(&g.d, &ad, &bd, &out));
        snprintf(what, sizeof what, "sigmoid_mul, %s in", dt_name(cfg[c][0]));
        EXPECT_CLOSE(g, what, out, want, 1e-6, 1);
        ATH_REQUIRE_OK(ath_add_ref(&a, &b, &want));
        ATH_EXPECT_OK(ath_add_gpu(&g.d, &ad, &bd, &out));
        snprintf(what, sizeof what, "add, %s in", dt_name(cfg[c][0]));
        EXPECT_CLOSE(g, what, out, want, 0, 0);   /* one rounding of the same sum */
    }
    /* the residual add in place: out is a */
    ath_tensor a = rand_t(&g, ATH_DTYPE_BF16, -8, 8, SHAPE(4, 999)), b = rand_t(&g, ATH_DTYPE_BF16, -8, 8, SHAPE(4, 999));
    ath_tensor ad = up(&g, &a), bd = up(&g, &b);
    ATH_REQUIRE_OK(ath_add_ref(&a, &b, &a));
    ATH_EXPECT_OK(ath_add_gpu(&g.d, &ad, &bd, &ad));
    EXPECT_CLOSE(g, "add in place", ad, a, 0, 0);
    gpu_close(&g);
}

/* ---- attention ----------------------------------------------------------------------- */

static void attn_case(ath_test *t_, gpu *g, ath_dtype in, ath_dtype ot, int64_t T, int64_t S, int64_t nh, int64_t nkv,
                      int64_t d, int64_t dv)
{
    ath_tensor q = rand_t(g, in, -1, 1, SHAPE(T, nh, d)), k = rand_t(g, in, -1, 1, SHAPE(S, nkv, d));
    ath_tensor v = rand_t(g, in, -1, 1, SHAPE(S, nkv, dv)), want = host(g, ot, SHAPE(T, nh, dv));
    const float scale = 1.0f / sqrtf((float)d);
    ATH_REQUIRE_OK(ath_attn_ref(&q, &k, &v, NULL, scale, &want));
    ath_tensor qd = up(g, &q), kd = up(g, &k), vd = up(g, &v), out = up(g, &want), split = up(g, &want);
    ATH_EXPECT_OK(ath_attn_gpu(&g->d, &qd, &kd, &vd, NULL, scale, NULL, &out));
    char what[96];
    snprintf(what, sizeof what, "attn T %lld S %lld heads %lld/%lld d %lld/%lld, %s in", (long long)T, (long long)S,
             (long long)nh, (long long)nkv, (long long)d, (long long)dv, dt_name(in));
    expect_close(t_, g, what, &out, &want, 1e-5, 1);
    /* with scratch, a few queries go split-K: the same bits as the rows */
    ATH_EXPECT_OK(ath_attn_gpu(&g->d, &qd, &kd, &vd, NULL, scale, &g->dev, &split));
    ATH_EXPECT(same_bits(g, &split, &out));
}

/* Positions (step 9): queries anywhere in a whole cache whose other rows
 * are noise, against the reference with the same positions, rows and
 * split-K alike and with the same bits; a position outside the cache gives
 * a row of NaN on the GPU. */
static void attn_pos_case(ath_test *t_, gpu *g, int64_t T, int64_t S, const int32_t *p)
{
    const int64_t nh = 8, nkv = 2, d = 256;
    ath_tensor q = rand_t(g, ATH_DTYPE_BF16, -1, 1, SHAPE(T, nh, d)), k = rand_t(g, ATH_DTYPE_BF16, -1, 1, SHAPE(S, nkv, d));
    ath_tensor v = rand_t(g, ATH_DTYPE_BF16, -1, 1, SHAPE(S, nkv, d)), want = host(g, ATH_DTYPE_BF16, SHAPE(T, nh, d));
    ath_tensor pos = host(g, ATH_DTYPE_I32, SHAPE(T));
    memcpy(pos.data, p, (size_t)T * sizeof *p);
    ATH_REQUIRE_OK(ath_attn_ref(&q, &k, &v, &pos, 0.0625f, &want));
    ath_tensor qd = up(g, &q), kd = up(g, &k), vd = up(g, &v), pd = up(g, &pos), rows = up(g, &want),
               split = up(g, &want);
    ATH_EXPECT_OK(ath_attn_gpu(&g->d, &qd, &kd, &vd, &pd, 0.0625f, NULL, &rows));
    ATH_EXPECT_OK(ath_attn_gpu(&g->d, &qd, &kd, &vd, &pd, 0.0625f, &g->dev, &split));
    char what[96];
    snprintf(what, sizeof what, "attn with positions, T %lld in a cache of %lld", (long long)T, (long long)S);
    expect_close(t_, g, what, &rows, &want, 1e-5, 1);
    ATH_EXPECT(same_bits(g, &split, &rows));
}

ATH_TEST(gpu_attn)
{
    gpu g;
    GPU_OR_SKIP(g);
    const ath_dtype cfg[][2] = {{ATH_DTYPE_F32, ATH_DTYPE_F32}, {ATH_DTYPE_BF16, ATH_DTYPE_BF16},
                                {ATH_DTYPE_BF16, ATH_DTYPE_F32}};
    for (int c = 0; c < 3; c++) {
        attn_case(t_, &g, cfg[c][0], cfg[c][1], 5, 9, 8, 2, 256, 256);     /* the 0.8B shapes, a short prompt */
        attn_case(t_, &g, cfg[c][0], cfg[c][1], 75, 75, 4, 1, 64, 96);     /* several tiles of keys */
        attn_case(t_, &g, cfg[c][0], cfg[c][1], 1, 300, 6, 3, 100, 1024);  /* one token, a long cache */
    }

    /* decode equals the last row of a prompt short enough not to take the
     * tensor cores, bit for bit (the long prompts: gpu_attn_mma) */
    ath_tensor q = rand_t(&g, ATH_DTYPE_BF16, -1, 1, SHAPE(70, 8, 256)), k = rand_t(&g, ATH_DTYPE_BF16, -1, 1, SHAPE(70, 2, 256));
    ath_tensor v = rand_t(&g, ATH_DTYPE_BF16, -1, 1, SHAPE(70, 2, 256)), o = host(&g, ATH_DTYPE_BF16, SHAPE(70, 8, 256));
    ath_tensor qd = up(&g, &q), kd = up(&g, &k), vd = up(&g, &v), od = up(&g, &o), q1, o1, last, qs, os;
    ATH_REQUIRE_OK(ath_tensor_slice(&qd, 0, 70 - (ATH_ATTN_MMA_MIN_T - 1), ATH_ATTN_MMA_MIN_T - 1, &qs));
    ATH_REQUIRE_OK(ath_tensor_slice(&od, 0, 70 - (ATH_ATTN_MMA_MIN_T - 1), ATH_ATTN_MMA_MIN_T - 1, &os));
    ATH_EXPECT_OK(ath_attn_gpu(&g.d, &qs, &kd, &vd, NULL, 0.0625f, NULL, &os));
    ATH_REQUIRE_OK(ath_tensor_slice(&qd, 0, 69, 1, &q1));
    ath_tensor o1h = host(&g, ATH_DTYPE_BF16, SHAPE(1, 8, 256));
    o1 = up(&g, &o1h);
    ATH_EXPECT_OK(ath_attn_gpu(&g.d, &q1, &kd, &vd, NULL, 0.0625f, &g.dev, &o1));   /* split-K */
    ATH_REQUIRE_OK(ath_tensor_slice(&od, 0, 69, 1, &last));
    ATH_EXPECT(same_bits(&g, &o1, &last));
    /* the same token given its position, over the cache as a whole */
    ath_tensor ph = host(&g, ATH_DTYPE_I32, SHAPE(1));
    ((int32_t *)ph.data)[0] = 69;
    ath_tensor pd = up(&g, &ph), o2 = up(&g, &o1h);
    ATH_EXPECT_OK(ath_attn_gpu(&g.d, &q1, &kd, &vd, &pd, 0.0625f, &g.dev, &o2));
    ATH_EXPECT(same_bits(&g, &o2, &last));

    static const int32_t p1[] = {0}, p3[] = {7, 300, 999}, p8[] = {0, 1, 255, 256, 257, 511, 512, 1999};
    attn_pos_case(t_, &g, 1, 1, p1);
    attn_pos_case(t_, &g, 3, 1000, p3);
    attn_pos_case(t_, &g, 8, 2000, p8);

    /* a position outside the cache: NaN, not a read outside it */
    ((int32_t *)ph.data)[0] = 70;
    pd = up(&g, &ph);
    ATH_EXPECT_OK(ath_attn_gpu(&g.d, &q1, &kd, &vd, &pd, 0.0625f, &g.dev, &o2));
    ath_tensor nan_row = down(&g, &o2);
    ATH_EXPECT(isnan(ath_load_f32(nan_row.data, nan_row.dtype, 0)));
    gpu_close(&g);
}

/* Prompts on the tensor cores (step 10). The probabilities are rounded to
 * bf16 before they multiply the values (ops.h), which moves every weight by
 * a relative 2^-8 at most: out = Σ w_j v_j with Σ w_j = 1 moves by at most
 * 2^-8 Σ_j w_j |v_j − out| <= 2^-8 (A + |out|), A the same attention over
 * |v|. Against the reference in fp32, plus the rounding of the output to
 * bf16 (2^-8 |got|) and a little for the fp32 sums. The scale is larger than
 * a model's, so that the softmax is peaked and a key taken or missed by
 * mistake shows. */
static void attn_mma_case(ath_test *t_, gpu *g, int64_t T, int64_t S, int64_t nh, int64_t nkv, int64_t d,
                            const int32_t *p)
{
    ath_tensor q = rand_t(g, ATH_DTYPE_BF16, -1, 1, SHAPE(T, nh, d)), k = rand_t(g, ATH_DTYPE_BF16, -1, 1, SHAPE(S, nkv, d));
    ath_tensor v = rand_t(g, ATH_DTYPE_BF16, -1, 1, SHAPE(S, nkv, d)), va = host(g, ATH_DTYPE_BF16, SHAPE(S, nkv, d));
    for (int64_t i = 0; i < ath_tensor_numel(&v); i++)
        ath_store_f32(va.data, ATH_DTYPE_BF16, i, fabsf(ath_load_f32(v.data, ATH_DTYPE_BF16, i)));
    ath_tensor pos = host(g, ATH_DTYPE_I32, SHAPE(T));
    if (p)
        memcpy(pos.data, p, (size_t)T * sizeof *p);
    ath_tensor want = host(g, ATH_DTYPE_F32, SHAPE(T, nh, d)), wabs = host(g, ATH_DTYPE_F32, SHAPE(T, nh, d));
    const float scale = 0.25f;
    ATH_REQUIRE_OK(ath_attn_ref(&q, &k, &v, p ? &pos : NULL, scale, &want));
    ATH_REQUIRE_OK(ath_attn_ref(&q, &k, &va, p ? &pos : NULL, scale, &wabs));
    ath_tensor oh = host(g, ATH_DTYPE_BF16, SHAPE(T, nh, d));
    ath_tensor qd = up(g, &q), kd = up(g, &k), vd = up(g, &v), pd = up(g, &pos), od = up(g, &oh);
    ATH_EXPECT_OK(ath_attn_gpu(&g->d, &qd, &kd, &vd, p ? &pd : NULL, scale, &g->dev, &od));
    ath_tensor got = down(g, &od);
    double worst = 0;
    for (int64_t i = 0; i < ath_tensor_numel(&want); i++) {
        const double w = ((const float *)want.data)[i], a = ((const float *)wabs.data)[i];
        const double x = ath_load_f32(got.data, ATH_DTYPE_BF16, i);
        const double bound = (a + fabs(w)) / 255 + fabs(x) / 256 + 1e-6;   /* 2^-8 / (1 - 2^-8), 2^-8 */
        const double r = fabs(x - w) / bound;
        if (!(r <= worst))
            worst = isnan(r) ? INFINITY : r;
    }
    char what[96];
    snprintf(what, sizeof what, "attn on tensor cores, T %lld S %lld heads %lld/%lld d %lld", (long long)T,
             (long long)S, (long long)nh, (long long)nkv, (long long)d);
    if (getenv("ATH_GPU_OPS_DETAIL") || worst > 1)
        printf("    %-56s error %.2f of its bound\n", what, worst);
    ATH_EXPECT(worst <= 1);
}

ATH_TEST(gpu_attn_mma)
{
    gpu g;
    GPU_OR_SKIP(g);
    attn_mma_case(t_, &g, 200, 200, 8, 2, 256, NULL);    /* the 0.8B heads, a prompt of 3 + tiles */
    attn_mma_case(t_, &g, 130, 300, 8, 2, 256, NULL);    /* the last 130 of 300 positions        */
    attn_mma_case(t_, &g, 64, 64, 4, 4, 128, NULL);      /* one tile, no grouping, d 128         */
    attn_mma_case(t_, &g, 77, 77, 6, 2, 64, NULL);       /* d 64, rows past T in the last block   */
    int32_t p[40];
    for (int i = 0; i < 40; i++)
        p[i] = (i * 97 + 13) % 1000;                      /* positions anywhere in a cache of 1000 */
    attn_mma_case(t_, &g, 40, 1000, 8, 2, 256, p);

    /* a prompt in pieces of at least ATH_ATTN_MMA_MIN_T queries, each over the
     * whole cache with its positions, gives the bits of the prompt at once;
     * decoding the last token gives the last row within the bound above */
    const int64_t T = 150, nh = 8, nkv = 2, d = 256;
    ath_tensor q = rand_t(&g, ATH_DTYPE_BF16, -1, 1, SHAPE(T, nh, d)), k = rand_t(&g, ATH_DTYPE_BF16, -1, 1, SHAPE(T, nkv, d));
    ath_tensor v = rand_t(&g, ATH_DTYPE_BF16, -1, 1, SHAPE(T, nkv, d)), o = host(&g, ATH_DTYPE_BF16, SHAPE(T, nh, d));
    ath_tensor ph = host(&g, ATH_DTYPE_I32, SHAPE(T));
    for (int64_t t = 0; t < T; t++)
        ((int32_t *)ph.data)[t] = (int32_t)t;
    ath_tensor qd = up(&g, &q), kd = up(&g, &k), vd = up(&g, &v), pd = up(&g, &ph), all = up(&g, &o), parts = up(&g, &o);
    ATH_EXPECT_OK(ath_attn_gpu(&g.d, &qd, &kd, &vd, &pd, 0.0625f, &g.dev, &all));
    const int64_t cuts[] = {0, 16, 90, 150};
    for (int i = 0; i < 3; i++) {
        ath_tensor qp, pp, op;
        ATH_REQUIRE_OK(ath_tensor_slice(&qd, 0, cuts[i], cuts[i + 1] - cuts[i], &qp));
        ATH_REQUIRE_OK(ath_tensor_slice(&pd, 0, cuts[i], cuts[i + 1] - cuts[i], &pp));
        ATH_REQUIRE_OK(ath_tensor_slice(&parts, 0, cuts[i], cuts[i + 1] - cuts[i], &op));
        ATH_EXPECT_OK(ath_attn_gpu(&g.d, &qp, &kd, &vd, &pp, 0.0625f, &g.dev, &op));
    }
    ATH_EXPECT(same_bits(&g, &all, &parts));

    /* a position outside the cache: a row of NaN, the other rows as before */
    ((int32_t *)ph.data)[100] = (int32_t)T;
    ath_tensor pbad = up(&g, &ph), bad = up(&g, &o);
    ATH_EXPECT_OK(ath_attn_gpu(&g.d, &qd, &kd, &vd, &pbad, 0.0625f, &g.dev, &bad));
    ath_tensor bh = down(&g, &bad), ah = down(&g, &all);
    bool nan_row = true, others = true;
    for (int64_t i = 0; i < T * nh * d; i++) {
        const bool in_row = i / (nh * d) == 100;
        const float x = ath_load_f32(bh.data, ATH_DTYPE_BF16, i);
        if (in_row)
            nan_row = nan_row && isnan(x);
        else
            others = others && ((uint16_t *)bh.data)[i] == ((uint16_t *)ah.data)[i];
    }
    ATH_EXPECT(nan_row);
    ATH_EXPECT(others);
    gpu_close(&g);
}

/* ---- scatter (step 9) ------------------------------------------------------------------ */

ATH_TEST(gpu_scatter_rows)
{
    gpu g;
    GPU_OR_SKIP(g);
    static const struct { ath_dtype dt; int64_t row; } kinds[] = {{ATH_DTYPE_BF16, 512}, {ATH_DTYPE_F32, 7},
                                                                    {ATH_DTYPE_BF16, 3}};
    for (size_t i = 0; i < ATH_ARRAY_LEN(kinds); i++) {
        ath_tensor src = rand_t(&g, kinds[i].dt, -1, 1, SHAPE(4, kinds[i].row));
        ath_tensor dst = rand_t(&g, kinds[i].dt, -1, 1, SHAPE(10, kinds[i].row));
        ath_tensor idx = host(&g, ATH_DTYPE_I32, SHAPE(4));
        static const int32_t rows[] = {9, 0, 4, 5};
        memcpy(idx.data, rows, sizeof rows);
        ath_tensor sd = up(&g, &src), dd = up(&g, &dst), id = up(&g, &idx);
        ATH_REQUIRE_OK(ath_scatter_rows_ref(&src, &idx, &dst));
        ATH_EXPECT_OK(ath_scatter_rows_gpu(&g.d, &sd, &id, &dd));
        EXPECT_CLOSE(g, "scatter_rows", dd, dst, 0, 0);
    }
    gpu_close(&g);
}

/* ---- convolution ---------------------------------------------------------------------- */

ATH_TEST(gpu_conv)
{
    gpu g;
    GPU_OR_SKIP(g);
    const ath_dtype cfg[][2] = {{ATH_DTYPE_F32, ATH_DTYPE_F32}, {ATH_DTYPE_BF16, ATH_DTYPE_BF16},
                                {ATH_DTYPE_BF16, ATH_DTYPE_F32}};
    for (int c = 0; c < 3; c++)
        for (int act = 0; act < 2; act++) {
            const ath_dtype in = cfg[c][0], ot = cfg[c][1];
            ath_tensor x = rand_t(&g, in, -2, 2, SHAPE(7, 300)), w = rand_t(&g, ATH_DTYPE_BF16, -1, 1, SHAPE(300, 1, 4));
            ath_tensor state = rand_t(&g, in, -2, 2, SHAPE(3, 300)), want = host(&g, ot, SHAPE(7, 300));
            ath_tensor sd = up(&g, &state);
            ath_tensor xd = up(&g, &x), wd = up(&g, &w), out = up(&g, &want);
            ATH_REQUIRE_OK(ath_conv_ref(&x, &w, &state, act ? ATH_ACT_SILU : ATH_ACT_NONE, &want));
            ATH_EXPECT_OK(ath_conv_gpu(&g.d, &xd, &wd, &sd, act ? ATH_ACT_SILU : ATH_ACT_NONE, &out));
            char what[96];
            snprintf(what, sizeof what, "conv %s, %s in", act ? "silu" : "none", dt_name(in));
            EXPECT_CLOSE(g, what, out, want, 1e-6, 1);
            snprintf(what, sizeof what, "conv state, %s", dt_name(in));
            EXPECT_CLOSE(g, what, sd, state, 0, 0);   /* the state holds inputs */
        }

    /* 10 tokens at once = 4 + 1 + 5, outputs and state bit for bit */
    ath_tensor x = rand_t(&g, ATH_DTYPE_BF16, -2, 2, SHAPE(10, 77)), w = rand_t(&g, ATH_DTYPE_BF16, -1, 1, SHAPE(77, 4));
    ath_tensor zero = host(&g, ATH_DTYPE_BF16, SHAPE(3, 77)), o = host(&g, ATH_DTYPE_BF16, SHAPE(10, 77));
    ath_tensor xd = up(&g, &x), wd = up(&g, &w), s_all = up(&g, &zero), s_parts = up(&g, &zero);
    ath_tensor o_all = up(&g, &o), o_parts = up(&g, &o);
    ATH_EXPECT_OK(ath_conv_gpu(&g.d, &xd, &wd, &s_all, ATH_ACT_SILU, &o_all));
    const int64_t cuts[] = {0, 4, 5, 10};
    for (int p = 0; p < 3; p++) {
        ath_tensor xp, op;
        ATH_REQUIRE_OK(ath_tensor_slice(&xd, 0, cuts[p], cuts[p + 1] - cuts[p], &xp));
        ATH_REQUIRE_OK(ath_tensor_slice(&o_parts, 0, cuts[p], cuts[p + 1] - cuts[p], &op));
        ATH_EXPECT_OK(ath_conv_gpu(&g.d, &xp, &wd, &s_parts, ATH_ACT_SILU, &op));
    }
    ATH_EXPECT(same_bits(&g, &o_all, &o_parts));
    ATH_EXPECT(same_bits(&g, &s_all, &s_parts));
    gpu_close(&g);
}

/* ---- Gated DeltaNet ------------------------------------------------------------------- */

ATH_TEST(gpu_gdn_gates)
{
    gpu g;
    GPU_OR_SKIP(g);
    const ath_dtype cfg[][2] = {{ATH_DTYPE_F32, ATH_DTYPE_F32}, {ATH_DTYPE_BF16, ATH_DTYPE_BF16},
                                {ATH_DTYPE_BF16, ATH_DTYPE_F32}};
    for (int c = 0; c < 3; c++) {
        /* a spans the softplus threshold (20) */
        ath_tensor a = rand_t(&g, cfg[c][0], -30, 30, SHAPE(9, 16)), b = rand_t(&g, cfg[c][0], -6, 6, SHAPE(9, 16));
        ath_tensor a_log = rand_t(&g, ATH_DTYPE_F32, -2, 2, SHAPE(16)), dt_bias = rand_t(&g, ATH_DTYPE_BF16, -1, 1, SHAPE(16));
        ath_tensor gw = host(&g, ATH_DTYPE_F32, SHAPE(9, 16)), bw = host(&g, cfg[c][1], SHAPE(9, 16));
        ATH_REQUIRE_OK(ath_gdn_gates_ref(&a, &b, &a_log, false, &dt_bias, &gw, &bw));
        ath_tensor ad = up(&g, &a), bd = up(&g, &b), ld = up(&g, &a_log), dd = up(&g, &dt_bias);
        ath_tensor go = up(&g, &gw), bo = up(&g, &bw);
        ATH_EXPECT_OK(ath_gdn_gates_gpu(&g.d, &ad, &bd, &ld, false, &dd, &go, &bo));
        char what[96];
        snprintf(what, sizeof what, "gdn_gates g, %s in", dt_name(cfg[c][0]));
        EXPECT_CLOSE(g, what, go, gw, 1e-6, 1);
        snprintf(what, sizeof what, "gdn_gates beta, %s in", dt_name(cfg[c][0]));
        EXPECT_CLOSE(g, what, bo, bw, 1e-6, 1);
    }
    gpu_close(&g);
}

/* q and k L2-normalized (in f32, as the forward pass gives them), v, gates. */
typedef struct {
    ath_tensor q, k, v, g, beta, state;
} gdn_in;

/* q and k as the forward pass gives them, not normalized (the op does it,
 * step 10), in the dtype of v. */
static gdn_in gdn_inputs(gpu *g, ath_dtype vdt, int64_t T, int64_t nk, int64_t dk, int64_t nv, int64_t dv)
{
    gdn_in in;
    in.q = rand_t(g, vdt, -1, 1, SHAPE(T, nk, dk));
    in.k = rand_t(g, vdt, -1, 1, SHAPE(T, nk, dk));
    in.v = rand_t(g, vdt, -1, 1, SHAPE(T, nv, dv));
    in.g = rand_t(g, ATH_DTYPE_F32, -2, 0, SHAPE(T, nv));
    in.beta = rand_t(g, vdt, 0, 1, SHAPE(T, nv));
    in.state = rand_t(g, ATH_DTYPE_F32, -0.5f, 0.5f, SHAPE(nv, dk, dv));
    return in;
}

ATH_TEST(gpu_gdn)
{
    gpu g;
    GPU_OR_SKIP(g);
    static const struct { int64_t T, nk, dk, nv, dv; } shapes[] = {
        {6, 2, 128, 4, 128},    /* the 0.8B head dims, two value heads per key head */
        {1, 16, 128, 16, 128},  /* one decoded token, the 0.8B head counts */
        {40, 1, 100, 3, 100},   /* dims that are not multiples of a warp or of a block */
    };
    const ath_dtype dts[] = {ATH_DTYPE_F32, ATH_DTYPE_BF16};
    for (size_t s = 0; s < ATH_ARRAY_LEN(shapes); s++)
        for (int c = 0; c < 2; c++) {
            const int64_t T = shapes[s].T, nk = shapes[s].nk, dk = shapes[s].dk, nv = shapes[s].nv, dv = shapes[s].dv;
            gdn_in in = gdn_inputs(&g, dts[c], T, nk, dk, nv, dv);
            ath_tensor want = host(&g, dts[c], SHAPE(T, nv, dv));
            ath_tensor qd = up(&g, &in.q), kd = up(&g, &in.k), vd = up(&g, &in.v), gd = up(&g, &in.g);
            ath_tensor bd = up(&g, &in.beta), sd = up(&g, &in.state), out = up(&g, &want);
            const float q_scale = 1.0f / sqrtf((float)dk);
            ATH_REQUIRE_OK(ath_gdn_ref(&in.q, &in.k, &in.v, &in.g, &in.beta, q_scale, 1e-6f, false, &in.state, &want));
            ATH_EXPECT_OK(ath_gdn_gpu(&g.d, &qd, &kd, &vd, &gd, &bd, q_scale, 1e-6f, false, &sd, NULL, &out));
            char what[96];
            snprintf(what, sizeof what, "gdn T %lld heads %lld/%lld dims %lld/%lld, %s", (long long)T, (long long)nk,
                     (long long)nv, (long long)dk, (long long)dv, dt_name(dts[c]));
            EXPECT_CLOSE(g, what, out, want, 1e-5, 1);
            snprintf(what, sizeof what, "gdn state T %lld dims %lld/%lld, %s", (long long)T, (long long)dk,
                     (long long)dv, dt_name(dts[c]));
            EXPECT_CLOSE(g, what, sd, in.state, 1e-5, 0);
        }

    /* 12 tokens at once = 5 + 1 + 6, outputs and state bit for bit */
    gdn_in in = gdn_inputs(&g, ATH_DTYPE_BF16, 12, 2, 128, 4, 128);
    ath_tensor o = host(&g, ATH_DTYPE_BF16, SHAPE(12, 4, 128));
    ath_tensor qd = up(&g, &in.q), kd = up(&g, &in.k), vd = up(&g, &in.v), gd = up(&g, &in.g), bd = up(&g, &in.beta);
    ath_tensor s_all = up(&g, &in.state), s_parts = up(&g, &in.state), o_all = up(&g, &o), o_parts = up(&g, &o);
    ATH_EXPECT_OK(ath_gdn_gpu(&g.d, &qd, &kd, &vd, &gd, &bd, 0.088f, 1e-6f, false, &s_all, NULL, &o_all));
    const int64_t cuts[] = {0, 5, 6, 12};
    for (int p = 0; p < 3; p++) {
        const int64_t t0 = cuts[p], n = cuts[p + 1] - cuts[p];
        ath_tensor a[6];
        ATH_REQUIRE_OK(ath_tensor_slice(&qd, 0, t0, n, &a[0]));
        ATH_REQUIRE_OK(ath_tensor_slice(&kd, 0, t0, n, &a[1]));
        ATH_REQUIRE_OK(ath_tensor_slice(&vd, 0, t0, n, &a[2]));
        ATH_REQUIRE_OK(ath_tensor_slice(&gd, 0, t0, n, &a[3]));
        ATH_REQUIRE_OK(ath_tensor_slice(&bd, 0, t0, n, &a[4]));
        ATH_REQUIRE_OK(ath_tensor_slice(&o_parts, 0, t0, n, &a[5]));
        ATH_EXPECT_OK(ath_gdn_gpu(&g.d, &a[0], &a[1], &a[2], &a[3], &a[4], 0.088f, 1e-6f, false, &s_parts, NULL, &a[5]));
    }
    ATH_EXPECT(same_bits(&g, &o_all, &o_parts));
    ATH_EXPECT(same_bits(&g, &s_all, &s_parts));
    gpu_close(&g);
}

/* The chunked form for prompts (step 10): from ATH_GDN_CHUNK_MIN_T tokens
 * with a scratch arena. It computes the recurrence with another order of
 * operations, in fp32 (tests/README.md, Prompt kernels). */
ATH_TEST(gpu_gdn_chunk)
{
    gpu g;
    GPU_OR_SKIP(g);
    static const struct { int64_t T, nk, dk, nv, dv; } shapes[] = {
        {64, 2, 128, 4, 128},     /* one chunk, two value heads per key head */
        {200, 16, 128, 16, 128},  /* the 0.8B head counts, a partial last chunk */
        {2100, 2, 128, 4, 128},   /* more than one round of chunks */
        {100, 1, 100, 3, 100},    /* dims that are not multiples of 32 */
        {70, 1, 64, 2, 96},       /* dk and dv apart */
    };
    const ath_dtype dts[] = {ATH_DTYPE_F32, ATH_DTYPE_BF16};
    for (size_t s = 0; s < ATH_ARRAY_LEN(shapes); s++)
        for (int c = 0; c < 2; c++) {
            const int64_t T = shapes[s].T, nk = shapes[s].nk, dk = shapes[s].dk, nv = shapes[s].nv, dv = shapes[s].dv;
            gdn_in in = gdn_inputs(&g, dts[c], T, nk, dk, nv, dv);
            ath_tensor want = host(&g, dts[c], SHAPE(T, nv, dv));
            ath_tensor qd = up(&g, &in.q), kd = up(&g, &in.k), vd = up(&g, &in.v), gd = up(&g, &in.g);
            ath_tensor bd = up(&g, &in.beta), sd = up(&g, &in.state), out = up(&g, &want);
            const float q_scale = 1.0f / sqrtf((float)dk);
            ATH_REQUIRE_OK(ath_gdn_ref(&in.q, &in.k, &in.v, &in.g, &in.beta, q_scale, 1e-6f, false, &in.state, &want));
            ATH_EXPECT_OK(ath_gdn_gpu(&g.d, &qd, &kd, &vd, &gd, &bd, q_scale, 1e-6f, false, &sd, &g.dev, &out));
            char what[96];
            snprintf(what, sizeof what, "gdn chunks T %lld heads %lld/%lld dims %lld/%lld, %s", (long long)T,
                     (long long)nk, (long long)nv, (long long)dk, (long long)dv, dt_name(dts[c]));
            EXPECT_CLOSE(g, what, out, want, 1e-4, 2);
            snprintf(what, sizeof what, "gdn chunks state T %lld dims %lld/%lld, %s", (long long)T, (long long)dk,
                     (long long)dv, dt_name(dts[c]));
            EXPECT_CLOSE(g, what, sd, in.state, 1e-4, 0);
        }

    /* 256 tokens at once = 128 + 64 + 64: pieces of whole chunks, bit for bit */
    gdn_in in = gdn_inputs(&g, ATH_DTYPE_BF16, 256, 2, 128, 4, 128);
    ath_tensor o = host(&g, ATH_DTYPE_BF16, SHAPE(256, 4, 128));
    ath_tensor qd = up(&g, &in.q), kd = up(&g, &in.k), vd = up(&g, &in.v), gd = up(&g, &in.g), bd = up(&g, &in.beta);
    ath_tensor s_all = up(&g, &in.state), s_parts = up(&g, &in.state), o_all = up(&g, &o), o_parts = up(&g, &o);
    ATH_EXPECT_OK(ath_gdn_gpu(&g.d, &qd, &kd, &vd, &gd, &bd, 0.088f, 1e-6f, false, &s_all, &g.dev, &o_all));
    const int64_t cuts[] = {0, 128, 192, 256};
    for (int p = 0; p < 3; p++) {
        const int64_t t0 = cuts[p], n = cuts[p + 1] - cuts[p];
        ath_tensor a[6];
        ATH_REQUIRE_OK(ath_tensor_slice(&qd, 0, t0, n, &a[0]));
        ATH_REQUIRE_OK(ath_tensor_slice(&kd, 0, t0, n, &a[1]));
        ATH_REQUIRE_OK(ath_tensor_slice(&vd, 0, t0, n, &a[2]));
        ATH_REQUIRE_OK(ath_tensor_slice(&gd, 0, t0, n, &a[3]));
        ATH_REQUIRE_OK(ath_tensor_slice(&bd, 0, t0, n, &a[4]));
        ATH_REQUIRE_OK(ath_tensor_slice(&o_parts, 0, t0, n, &a[5]));
        ATH_EXPECT_OK(ath_gdn_gpu(&g.d, &a[0], &a[1], &a[2], &a[3], &a[4], 0.088f, 1e-6f, false, &s_parts, &g.dev, &a[5]));
    }
    ATH_EXPECT(same_bits(&g, &o_all, &o_parts));
    ATH_EXPECT(same_bits(&g, &s_all, &s_parts));
    gpu_close(&g);
}

static const ath_test_case tests[] = {
    ATH_TEST_ENTRY(gpu_argument_checks), ATH_TEST_ENTRY(gpu_embed),       ATH_TEST_ENTRY(gpu_matmul),      ATH_TEST_ENTRY(gpu_gemv), ATH_TEST_ENTRY(gpu_matmul_q8_0), ATH_TEST_ENTRY(gpu_matmul_n), ATH_TEST_ENTRY(gpu_matmul_swiglu), ATH_TEST_ENTRY(gpu_scatter_rows), ATH_TEST_ENTRY(gpu_sample),
    ATH_TEST_ENTRY(gpu_norms),           ATH_TEST_ENTRY(gpu_add_rmsnorm),           ATH_TEST_ENTRY(gpu_rope),        ATH_TEST_ENTRY(gpu_elementwise),
    ATH_TEST_ENTRY(gpu_attn),            ATH_TEST_ENTRY(gpu_attn_mma),    ATH_TEST_ENTRY(gpu_conv),        ATH_TEST_ENTRY(gpu_gdn_gates),
    ATH_TEST_ENTRY(gpu_gdn),             ATH_TEST_ENTRY(gpu_gdn_chunk),
};

int main(int argc, char **argv)
{
    return ath_test_main(tests, ATH_ARRAY_LEN(tests), argc, argv);
}
