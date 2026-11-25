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
    ATH_EXPECT_STATUS(ath_attn_gpu(d, &q, &q, &v, 1.0f, &o), ATH_ERR_UNSUPPORTED);
    ath_tensor gq = fake(ATH_DTYPE_F32, ATH_TIER_VRAM, SHAPE(1, 1, ATH_GDN_MAX_DK + 1));
    ath_tensor gv = fake(ATH_DTYPE_F32, ATH_TIER_VRAM, SHAPE(1, 1, 8)), gg = fake(ATH_DTYPE_F32, ATH_TIER_VRAM, SHAPE(1, 1));
    ath_tensor gs = fake(ATH_DTYPE_F32, ATH_TIER_VRAM, SHAPE(1, ATH_GDN_MAX_DK + 1, 8));
    ATH_EXPECT_STATUS(ath_gdn_gpu(d, &gq, &gq, &gv, &gg, &gg, 1.0f, &gs, &gv), ATH_ERR_UNSUPPORTED);
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
            snprintf(what, sizeof what, "matmul %lldx%lldx%lld %s·%s %s", (long long)N, (long long)K, (long long)M,
                     dt_name(kinds[k].x), dt_name(kinds[k].w), kinds[k].path);
            EXPECT_CLOSE(g, what, out, want, 2e-5, 1);
            ath_dev_arena_reset(&g.dev, mark);   /* the comparison has waited for the stream */
        }
    gpu_close(&g);
}

/* ---- norms ----------------------------------------------------------------------- */

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
    ATH_REQUIRE_OK(ath_attn_ref(&q, &k, &v, scale, &want));
    ath_tensor qd = up(g, &q), kd = up(g, &k), vd = up(g, &v), out = up(g, &want);
    ATH_EXPECT_OK(ath_attn_gpu(&g->d, &qd, &kd, &vd, scale, &out));
    char what[96];
    snprintf(what, sizeof what, "attn T %lld S %lld heads %lld/%lld d %lld/%lld, %s in", (long long)T, (long long)S,
             (long long)nh, (long long)nkv, (long long)d, (long long)dv, dt_name(in));
    expect_close(t_, g, what, &out, &want, 1e-5, 1);
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

    /* decode equals the last row of the prompt, bit for bit */
    ath_tensor q = rand_t(&g, ATH_DTYPE_BF16, -1, 1, SHAPE(70, 8, 256)), k = rand_t(&g, ATH_DTYPE_BF16, -1, 1, SHAPE(70, 2, 256));
    ath_tensor v = rand_t(&g, ATH_DTYPE_BF16, -1, 1, SHAPE(70, 2, 256)), o = host(&g, ATH_DTYPE_BF16, SHAPE(70, 8, 256));
    ath_tensor qd = up(&g, &q), kd = up(&g, &k), vd = up(&g, &v), od = up(&g, &o), q1, o1, last;
    ATH_EXPECT_OK(ath_attn_gpu(&g.d, &qd, &kd, &vd, 0.0625f, &od));
    ATH_REQUIRE_OK(ath_tensor_slice(&qd, 0, 69, 1, &q1));
    ath_tensor o1h = host(&g, ATH_DTYPE_BF16, SHAPE(1, 8, 256));
    o1 = up(&g, &o1h);
    ATH_EXPECT_OK(ath_attn_gpu(&g.d, &q1, &kd, &vd, 0.0625f, &o1));
    ATH_REQUIRE_OK(ath_tensor_slice(&od, 0, 69, 1, &last));
    ATH_EXPECT(same_bits(&g, &o1, &last));
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
        ATH_REQUIRE_OK(ath_gdn_gates_ref(&a, &b, &a_log, &dt_bias, &gw, &bw));
        ath_tensor ad = up(&g, &a), bd = up(&g, &b), ld = up(&g, &a_log), dd = up(&g, &dt_bias);
        ath_tensor go = up(&g, &gw), bo = up(&g, &bw);
        ATH_EXPECT_OK(ath_gdn_gates_gpu(&g.d, &ad, &bd, &ld, &dd, &go, &bo));
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

static gdn_in gdn_inputs(gpu *g, ath_dtype vdt, int64_t T, int64_t nk, int64_t dk, int64_t nv, int64_t dv)
{
    gdn_in in;
    ath_tensor q = rand_t(g, ATH_DTYPE_F32, -1, 1, SHAPE(T, nk, dk)), k = rand_t(g, ATH_DTYPE_F32, -1, 1, SHAPE(T, nk, dk));
    in.q = host(g, ATH_DTYPE_F32, SHAPE(T, nk, dk));
    in.k = host(g, ATH_DTYPE_F32, SHAPE(T, nk, dk));
    if (ath_l2norm_ref(&q, 1e-6f, &in.q) != ATH_OK || ath_l2norm_ref(&k, 1e-6f, &in.k) != ATH_OK)
        abort();
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
            ATH_REQUIRE_OK(ath_gdn_ref(&in.q, &in.k, &in.v, &in.g, &in.beta, q_scale, &in.state, &want));
            ATH_EXPECT_OK(ath_gdn_gpu(&g.d, &qd, &kd, &vd, &gd, &bd, q_scale, &sd, &out));
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
    ATH_EXPECT_OK(ath_gdn_gpu(&g.d, &qd, &kd, &vd, &gd, &bd, 0.088f, &s_all, &o_all));
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
        ATH_EXPECT_OK(ath_gdn_gpu(&g.d, &a[0], &a[1], &a[2], &a[3], &a[4], 0.088f, &s_parts, &a[5]));
    }
    ATH_EXPECT(same_bits(&g, &o_all, &o_parts));
    ATH_EXPECT(same_bits(&g, &s_all, &s_parts));
    gpu_close(&g);
}

static const ath_test_case tests[] = {
    ATH_TEST_ENTRY(gpu_argument_checks), ATH_TEST_ENTRY(gpu_embed),       ATH_TEST_ENTRY(gpu_matmul),
    ATH_TEST_ENTRY(gpu_norms),           ATH_TEST_ENTRY(gpu_rope),        ATH_TEST_ENTRY(gpu_elementwise),
    ATH_TEST_ENTRY(gpu_attn),            ATH_TEST_ENTRY(gpu_conv),        ATH_TEST_ENTRY(gpu_gdn_gates),
    ATH_TEST_ENTRY(gpu_gdn),
};

int main(int argc, char **argv)
{
    return ath_test_main(tests, ATH_ARRAY_LEN(tests), argc, argv);
}
