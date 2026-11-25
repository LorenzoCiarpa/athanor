/* test_engine.c: the model on a GPU (src/infer/engine.h, step 8).
 *
 * On tiny random checkpoints (tests/tiny.h): the weights the engine copies
 * to the device, the state on the device (reset, copy, which side a pass
 * may run on), and a sequence run all at once, in pieces and one token at a
 * time, against itself and against the CPU.
 *
 * On the real checkpoint, the gate of step 8 (docs/STATUS.md): greedy
 * generation on the GPU in bf16 against the fp32 reference, 50 prompts of
 * 256 tokens (tests/golden/generate.py). At every step the engine is fed
 * the reference's token (teacher forcing) and its own choice is compared
 * with the reference's. Where the reference's top-2 margin exceeds the bf16
 * noise of the prompt, the choices must be equal; below it, rounding alone
 * may flip them. (The largest noise over all prompts, as test_forward uses
 * for its two prompts, would check fewer steps: one prompt's noise is 1.24,
 * the others' 0.17 to 0.62.)
 * As long as the choices agree, teacher forcing is free-running generation,
 * so the report also gives how many tokens the engine generates exactly as
 * the reference before the first difference.
 *
 * The CPU build runs only the check that the engine needs a GPU.
 */
#include "test.h"
#include "tiny.h"
#include "util.h"

#include "core/file.h"
#include "core/json.h"
#include "infer/engine.h"
#include "model/model.h"

#include <math.h>
#include <time.h>

#define GPU_OR_SKIP()                                                             \
    do {                                                                          \
        if (ath_gpu_count() == 0)                                                 \
            ATH_SKIP(strcmp(ath_gpu_backend(), "none") == 0                       \
                         ? "built without a GPU backend (make cuda test)"         \
                         : "no usable GPU device (ATH_LOG=info shows why)");      \
    } while (0)

typedef struct {
    tmpdir    d;
    ath_arena a;
    ath_model m;
} tiny_model;

static ath_status tiny_open(tiny_model *t, uint64_t seed)
{
    ath_arena_init(&t->a, 0);
    return open_tiny(&t->a, &t->m, &t->d, &DENSE, (ckpt){.seed = seed, .real_dtypes = true}, NULL);
}

static void tiny_close(tiny_model *t)
{
    ath_model_close(&t->m);
    tmpdir_remove(&t->d);
    ath_arena_free(&t->a);
}

static size_t spec_bytes(const ath_state_spec *sp, ath_dtype act, int32_t capacity)
{
    const size_t es = ath_dtype_size(sp->kind == ATH_STATE_RECURRENT ? ATH_DTYPE_F32 : act);
    return (size_t)(sp->fixed + sp->per_token * capacity) * es;
}

/* A device buffer's bytes on the host, after the work queued before. */
static bool fetch(ath_engine *e, void *dst, const void *src, size_t n)
{
    return ath_copy(&e->dev, dst, src, n, ATH_STREAM_COMPUTE) == ATH_OK &&
           ath_stream_sync(&e->dev, ATH_STREAM_COMPUTE) == ATH_OK;
}

static ath_tensor f32_tensor(float *data, int64_t rows, int64_t cols)
{
    ath_tensor t;
    const int64_t shape[2] = {rows, cols};
    ath_tensor_init(&t, ATH_DTYPE_F32, 2, shape, data, ATH_TIER_RAM);
    return t;
}

static double score(const float *got, const float *want, int64_t n)
{
    double ss = 0, worst = 0;
    for (int64_t i = 0; i < n; i++)
        ss += (double)want[i] * want[i];
    const double rms = n ? sqrt(ss / (double)n) : 0;
    for (int64_t i = 0; i < n; i++) {
        double e = fabs((double)got[i] - want[i]) / (fabs((double)want[i]) + rms);
        if (!(e <= worst))
            worst = isnan(e) ? INFINITY : e;
    }
    return worst;
}

/* ---- without a GPU ------------------------------------------------------------- */

ATH_TEST(engine_without_gpu)
{
    if (strcmp(ath_gpu_backend(), "none") != 0)
        ATH_SKIP("this build has a GPU backend");
    tiny_model t;
    ATH_REQUIRE_OK(tiny_open(&t, 1));
    ath_engine e;
    ATH_EXPECT_STATUS(ath_engine_open(&e, &t.m, 0, &t.a), ATH_ERR_UNSUPPORTED);
    ath_engine_close(&e);   /* safe after a failed open */
    tiny_close(&t);
}

/* ---- the engine on tiny checkpoints ---------------------------------------------- */

ATH_TEST(engine_weights)
{
    GPU_OR_SKIP();
    tiny_model t;
    ATH_REQUIRE_OK(tiny_open(&t, 3));
    ath_engine e;
    ATH_REQUIRE_OK(ath_engine_open(&e, &t.m, 0, &t.a));
    uint32_t used = 0;
    uint64_t bytes = 0;
    bool same = true;
    for (uint32_t i = 0; i < t.m.n_params; i++) {
        const ath_param *p = &t.m.params[i];
        const ath_tensor *w = &e.weights[i];
        if (p->flags & ATH_P_UNUSED) {
            ATH_EXPECT(w->data == NULL);   /* the vision encoder stays on the host */
            continue;
        }
        used++;
        const size_t n = ath_tensor_nbytes(&p->t);
        bytes += n;
        ATH_EXPECT(w->tier == ATH_TIER_VRAM && w->dtype == p->t.dtype && ath_tensor_same_shape(w, &p->t));
        void *h = ath_arena_alloc(&t.a, n + 1, 64);
        ATH_REQUIRE(h && fetch(&e, h, w->data, n));
        same = same && memcmp(h, p->t.data, n) == 0;
    }
    ATH_EXPECT(same);
    ATH_EXPECT_EQ_INT(e.n_copied, used);
    ATH_EXPECT(e.weight_bytes == bytes);
    ATH_EXPECT(e.exec.dev == &e.dev && e.exec.weights == e.weights && e.exec.scratch == &e.scratch);
    ath_engine_close(&e);
    tiny_close(&t);
}

ATH_TEST(engine_state)
{
    GPU_OR_SKIP();
    tiny_model t;
    ATH_REQUIRE_OK(tiny_open(&t, 5));
    const ath_model *m = &t.m;
    ath_engine e;
    ATH_REQUIRE_OK(ath_engine_open(&e, m, 0, &t.a));
    ath_arena scratch;
    ath_arena_init(&scratch, 0);

    /* reset zeroes the fixed-size state on the device */
    ath_state s, c, h;
    ATH_REQUIRE_OK(ath_engine_state(&e, &s, ATH_DTYPE_BF16, 8, &t.a));
    ATH_EXPECT(s.dev == &e.dev);
    for (uint32_t i = 0; i < m->n_states; i++)
        ATH_EXPECT_OK(ath_memset(&e.dev, s.buf[i], 0x5a, spec_bytes(&m->states[i], s.act, s.capacity),
                                 ATH_STREAM_COMPUTE));
    s.n_past = 3;
    ATH_EXPECT_OK(ath_state_reset(&s, m));
    ATH_EXPECT(s.n_past == 0);
    bool zero = true;
    for (uint32_t i = 0; i < m->n_states; i++) {
        if (m->states[i].kind == ATH_STATE_KV)
            continue;
        const size_t n = spec_bytes(&m->states[i], s.act, s.capacity);
        unsigned char *b = ath_arena_alloc(&t.a, n + 1, 64);
        ATH_REQUIRE(b && fetch(&e, b, s.buf[i], n));
        for (size_t j = 0; j < n; j++)
            zero = zero && b[j] == 0;
    }
    ATH_EXPECT(zero);

    /* a copy on the device resumes where the original was */
    const int32_t toks[6] = {3, 15, 0, 7, 7, 2};
    float la[V], lb[V];
    ath_tensor ta = f32_tensor(la, 1, V), tb = f32_tensor(lb, 1, V);
    ATH_REQUIRE_OK(ath_engine_state(&e, &c, ATH_DTYPE_BF16, 8, &t.a));
    ATH_REQUIRE_OK(ath_model_forward(m, &(ath_batch){.tokens = toks, .n_tokens = 4, .logits = NULL}, &s, NULL, &scratch, &e.exec));
    ATH_REQUIRE_OK(ath_state_copy(&c, &s, m));
    ATH_REQUIRE_OK(ath_model_forward(m, &(ath_batch){.tokens = toks + 4, .n_tokens = 2, .logits = &ta}, &s, NULL, &scratch, &e.exec));
    ATH_REQUIRE_OK(ath_model_forward(m, &(ath_batch){.tokens = toks + 4, .n_tokens = 2, .logits = &tb}, &c, NULL, &scratch, &e.exec));
    ATH_EXPECT(c.n_past == 6 && memcmp(la, lb, sizeof la) == 0);

    /* a pass runs where its state is */
    ATH_REQUIRE_OK(ath_state_init(&h, m, ATH_DTYPE_BF16, 8, &t.a));
    ATH_EXPECT_STATUS(ath_model_forward(m, &(ath_batch){.tokens = toks, .n_tokens = 1, .logits = NULL}, &s, NULL, &scratch, NULL), ATH_ERR_INVALID_ARG);
    ATH_EXPECT_STATUS(ath_model_forward(m, &(ath_batch){.tokens = toks, .n_tokens = 1, .logits = NULL}, &h, NULL, &scratch, &e.exec),
                      ATH_ERR_INVALID_ARG);
    ath_exec half = e.exec;
    half.weights = NULL;
    ATH_EXPECT_STATUS(ath_model_forward(m, &(ath_batch){.tokens = toks, .n_tokens = 1, .logits = NULL}, &s, NULL, &scratch, &half), ATH_ERR_INVALID_ARG);
    ATH_EXPECT_STATUS(ath_state_copy(&h, &s, m), ATH_ERR_INVALID_ARG);

    ath_arena_free(&scratch);
    ath_engine_close(&e);
    tiny_close(&t);
}

/* 12 tokens all at once, as 5 + 1 + 6 and one at a time, on the GPU, with
 * f32 activations: against each other and against the CPU. The kernels sum
 * in an order fixed by the shapes, except cuBLASLt, which chooses its
 * algorithm from the number of rows: a row computed in a prompt and alone
 * may differ in the last bits, so the pieces are held to a tolerance. */
ATH_TEST(engine_pieces)
{
    GPU_OR_SKIP();
    enum { NT = 12 };
    static const int32_t toks[NT] = {3, 15, 0, 7, 7, 2, 11, 9, 4, 14, 1, 8};
    static const int whole[] = {NT}, split[] = {5, 1, 6}, single[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    static const struct { const int *sizes; int n; const char *name; } runs[] = {
        {whole, 1, "at once"}, {split, 3, "5 + 1 + 6"}, {single, NT, "one at a time"}};
    tiny_model t;
    ATH_REQUIRE_OK(tiny_open(&t, 7));
    const ath_model *m = &t.m;
    ath_engine e;
    ATH_REQUIRE_OK(ath_engine_open(&e, m, 0, &t.a));
    ath_arena scratch;
    ath_arena_init(&scratch, 0);

    float cpu[NT * V], gpu[3][NT * V];
    ath_state s;
    ath_tensor lc = f32_tensor(cpu, NT, V);
    ATH_REQUIRE_OK(ath_state_init(&s, m, ATH_DTYPE_F32, NT, &t.a));
    ATH_REQUIRE_OK(ath_model_forward(m, &(ath_batch){.tokens = toks, .n_tokens = NT, .logits = &lc}, &s, NULL, &scratch, NULL));
    for (int r = 0; r < 3; r++) {
        ath_state g;
        ATH_REQUIRE_OK(ath_engine_state(&e, &g, ATH_DTYPE_F32, NT, &t.a));
        int done = 0;
        for (int i = 0; i < runs[r].n; i++) {
            const int n = runs[r].sizes[i];
            ath_tensor l = f32_tensor(gpu[r] + done * V, n, V);
            ATH_REQUIRE_OK(ath_model_forward(m, &(ath_batch){.tokens = toks + done, .n_tokens = n, .logits = &l}, &g, NULL, &scratch, &e.exec));
            done += n;
        }
        const double vs_cpu = score(gpu[r], cpu, NT * V), vs_whole = score(gpu[r], gpu[0], NT * V);
        printf("    GPU %-14s against the CPU %.2e, against the GPU at once %.2e%s\n", runs[r].name, vs_cpu, vs_whole,
               r && memcmp(gpu[r], gpu[0], sizeof gpu[0]) == 0 ? " (the same bits)" : "");
        /* the random tiny model is ill-conditioned (test_forward.c); 9.8e-6 measured */
        ATH_EXPECT(vs_cpu <= 1e-4);
        ATH_EXPECT(vs_whole <= 1e-5);
    }
    ath_arena_free(&scratch);
    ath_engine_close(&e);
    tiny_close(&t);
}

/* ---- the gate: greedy generation on the real checkpoint ------------------------- */

static const char *golden_dir(void)
{
    const char *d = getenv("ATH_GOLDEN_DIR");
    return (d && *d) ? d : "tests/golden/data/qwen35-0.8b";
}

static int64_t env_int(const char *name, int64_t fallback)
{
    const char *v = getenv(name);
    return v && *v ? strtoll(v, NULL, 10) : fallback;
}

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* The largest logit, the lowest id among equals (the sampler's greedy rule). */
static int32_t argmax(const float *x, int64_t n)
{
    int64_t best = 0;
    for (int64_t i = 1; i < n; i++)
        if (x[i] > x[best])
            best = i;
    return (int32_t)best;
}

static int32_t *json_ints(ath_arena *a, const ath_json *arr, int64_t *n)
{
    *n = (int64_t)ath_json_count(arr);
    int32_t *v = ath_arena_alloc(a, (size_t)(*n ? *n : 1) * sizeof *v, _Alignof(int32_t));
    int64_t i = 0;
    for (const ath_json *x = ath_json_first(arr); v && x; x = x->next, i++) {
        int64_t t = -1;
        ath_json_to_int(x, &t);
        v[i] = (int32_t)t;
    }
    return v;
}

ATH_TEST(generate_golden)
{
    GPU_OR_SKIP();
    char path[ATH_PATH_MAX];
    if (ath_path_join(path, sizeof path, golden_dir(), "generate.json") != ATH_OK || !ath_file_exists(path))
        ATH_SKIP("no generation golden data (make golden-generate, about 20 minutes)");
    if (!model_file(path, sizeof path, "config.json"))
        ATH_SKIP("no checkpoint in $ATH_MODEL_DIR");

    ath_arena a, scratch;
    ath_arena_init(&a, 0);
    ath_arena_init(&scratch, 0);
    ath_file f = {.fd = -1};
    const ath_json *root = NULL;
    ath_model m = {.family = NULL};
    ATH_REQUIRE_OK(ath_path_join(path, sizeof path, golden_dir(), "generate.json"));
    ATH_REQUIRE_OK(ath_file_open(&f, path));
    ATH_REQUIRE_OK(ath_json_parse(&a, f.data, f.size, &root, NULL));
    ATH_REQUIRE_OK(ath_model_open(&m, &a, model_dir()));
    const ath_json *prompts = ath_json_get(root, "prompts");
    int64_t steps = 0;
    ATH_REQUIRE_OK(ath_json_to_int(ath_json_get(root, "steps"), &steps));
    const size_t n_all = ath_json_count(prompts);
    ATH_REQUIRE(n_all > 0 && steps > 0);
#ifdef ATH_DEBUG
    const int64_t n_prompts = env_int("ATH_GENERATE_PROMPTS", 2), n_steps = env_int("ATH_GENERATE_STEPS", 8);
#else
    const int64_t n_prompts = env_int("ATH_GENERATE_PROMPTS", (int64_t)n_all), n_steps = env_int("ATH_GENERATE_STEPS", steps);
#endif
    const int64_t P = n_prompts < (int64_t)n_all ? n_prompts : (int64_t)n_all, S = n_steps < steps ? n_steps : steps;
    const int64_t Vm = m.vocab_size;

    /* the bf16 noise of the greedy decision: each prompt's own is the gate;
     * the largest over every prompt is reported for comparison */
    double noise = 0;
    for (const ath_json *p = ath_json_first(prompts); p; p = p->next) {
        double v = 0;
        ATH_REQUIRE_OK(ath_json_to_double(ath_json_get(p, "noise"), &v));
        noise = fmax(noise, v);
    }

    ath_engine *e = ath_arena_zalloc(&a, sizeof *e, _Alignof(ath_engine));
    ATH_REQUIRE(e);
    ATH_REQUIRE_OK(ath_engine_open(e, &m, 0, &a));
    float *lg = ath_arena_alloc(&a, (size_t)Vm * sizeof *lg, 64);
    ATH_REQUIRE(lg);
    ath_tensor logits = f32_tensor(lg, 1, Vm);
    ath_state s;
    ATH_REQUIRE_OK(ath_engine_state(e, &s, ATH_DTYPE_BF16, 1024, &a));

    int64_t checked = 0, bad = 0, identical_prompts = 0, identical_tokens = 0, differ_below = 0, tokens_run = 0;
    int64_t max_checked = 0, max_bad = 0;   /* with the largest noise instead of each prompt's own */
    double t_decode = 0;
    int64_t i = 0;
    for (const ath_json *p = ath_json_first(prompts); p && i < P; p = p->next, i++) {
        const char *name = "?";
        ath_json_to_str(ath_json_get(p, "name"), &name, NULL);
        int64_t np, nt, nm;
        const int32_t *prompt = json_ints(&a, ath_json_get(p, "prompt_tokens"), &np);
        const int32_t *ref = json_ints(&a, ath_json_get(p, "tokens"), &nt);
        const ath_json *margins = ath_json_get(p, "margins");
        nm = (int64_t)ath_json_count(margins);
        ATH_REQUIRE(prompt && ref && np > 0 && nt >= S && nm >= S && np + S <= s.capacity);
        double own = 0;
        ath_json_to_double(ath_json_get(p, "noise"), &own);

        ATH_REQUIRE_OK(ath_state_reset(&s, &m));
        ATH_REQUIRE_OK(ath_model_forward(&m, &(ath_batch){.tokens = prompt, .n_tokens = (int32_t)np, .logits = &logits}, &s, NULL, &scratch, &e->exec));
        int64_t first_diff = -1;
        const ath_json *mj = ath_json_first(margins);
        for (int64_t k = 0; k < S; k++, mj = mj->next) {
            double margin = 0;
            ath_json_to_double(mj, &margin);
            const int32_t got = argmax(lg, Vm);
            if (margin > own)
                checked++;
            if (margin > noise) {
                max_checked++;
                max_bad += got != ref[k];
            }
            if (got != ref[k]) {
                if (first_diff < 0)
                    first_diff = k;
                if (margin > own) {
                    bad++;
                    char msg[200];
                    snprintf(msg, sizeof msg, "%s step %lld: %d, the reference %d (margin %.3f > noise %.3f)", name,
                             (long long)k, got, ref[k], margin, own);
                    ath_test_fail_(t_, __FILE__, __LINE__, msg);
                } else {
                    differ_below++;
                }
            }
            if (k + 1 < S) {   /* the reference's token: teacher forcing */
                const double t0 = now();
                ATH_REQUIRE_OK(ath_model_forward(&m, &(ath_batch){.tokens = &ref[k], .n_tokens = 1, .logits = &logits}, &s, NULL, &scratch, &e->exec));
                t_decode += now() - t0;
                tokens_run++;
            }
        }
        const int64_t same = first_diff < 0 ? S : first_diff;
        identical_tokens += same;
        identical_prompts += first_diff < 0;
        if (getenv("ATH_GENERATE_DETAIL"))
            printf("    %-20s %3lld prompt tokens, noise %.3f; the first %3lld of %lld generated tokens equal\n", name,
                   (long long)np, own, (long long)same, (long long)S);
    }
    printf("  %lld prompts, %lld steps each, bf16 on %s\n", (long long)P, (long long)S, e->dev.name);
    printf("  greedy token differs from the fp32 reference at %lld of %lld steps whose margin exceeds the prompt's noise,\n"
           "  and at %lld steps below it; generating freely, %lld of %lld prompts equal the reference for all %lld\n"
           "  tokens, %lld of %lld tokens equal before the first difference; %.0f tokens/s decoding (for the record)\n",
           (long long)bad, (long long)checked, (long long)differ_below, (long long)identical_prompts, (long long)P,
           (long long)S, (long long)identical_tokens, (long long)(P * S), tokens_run / (t_decode > 0 ? t_decode : 1));
    printf("  with the largest noise of all prompts (%.3f) instead: %lld of %lld steps above it differ\n", noise,
           (long long)max_bad, (long long)max_checked);
    ATH_EXPECT(bad == 0);
    ATH_EXPECT(checked > 0);

    ath_engine_close(e);
    ath_model_close(&m);
    ath_file_close(&f);
    ath_arena_free(&scratch);
    ath_arena_free(&a);
}

static const ath_test_case tests[] = {
    ATH_TEST_ENTRY(engine_without_gpu), ATH_TEST_ENTRY(engine_weights), ATH_TEST_ENTRY(engine_state),
    ATH_TEST_ENTRY(engine_pieces),      ATH_TEST_ENTRY(generate_golden),
};

int main(int argc, char **argv)
{
    return ath_test_main(tests, ATH_ARRAY_LEN(tests), argc, argv);
}
