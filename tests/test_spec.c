/* test_spec.c: speculative decoding with the draft head (step 13).
 *
 * On the real checkpoint (ATH_MODEL_DIR, Qwen3.5-0.8B; skipped without it):
 * the tokens of speculative decoding equal those of plain decoding, one
 * token per pass, with the same sampler and the same uniform numbers:
 * greedy, temperature with top-k and top-p, and with a presence penalty;
 * on the CPU, and on the GPU when there is one (the engine with the draft
 * head). The share of guesses kept is printed. And the draft head's
 * argument checks.
 */
#include "test.h"
#include "util.h"

#include "core/rng.h"
#include "infer/engine.h"
#include "infer/spec.h"
#include "model/model.h"
#include "text/bpe.h"

#include <time.h>

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

#ifdef ATH_DEBUG
enum { NG = 24, CAP = 512 };   /* the CPU under ASan, at -O0, is slow */
#else
enum { NG = 96, CAP = 512 };
#endif

static const char PROMPT[] =
    "<|im_start|>user\nWrite a short story about a lighthouse keeper who finds a message in a bottle. "
    "Use simple words.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

/* Plain decoding: the prompt, then one token per pass, chosen on the host. */
static bool plain(ath_test *t_, const ath_model *m, ath_state *s, const ath_exec *x, const int32_t *ids, int32_t n,
                  const ath_sampler *p, const double *u, int32_t *out, ath_arena *scratch)
{
    float *lg = ath_arena_alloc(scratch, (size_t)m->vocab_size * sizeof(float), 64);
    uint8_t *seen = ath_arena_zalloc(scratch, (size_t)m->vocab_size, 1);
    ath_tensor l;
    ATH_EXPECT_OK(ath_tensor_init(&l, ATH_DTYPE_F32, 2, (int64_t[]){1, m->vocab_size}, lg, ATH_TIER_RAM));
    ath_arena work;
    ath_arena_init(&work, 0);
    bool ok = ath_state_reset(s, m) == ATH_OK &&
              ath_model_forward(m, &(ath_batch){.tokens = ids, .n_tokens = n, .logits = &l}, s, NULL, &work, x) == ATH_OK;
    for (int i = 0; ok && i < NG; i++) {
        if (i > 0) {
            ok = ath_model_forward(m, &(ath_batch){.tokens = &out[i - 1], .n_tokens = 1, .logits = &l}, s, NULL, &work,
                                   x) == ATH_OK;
            if (ok && p->presence != 0)
                ok = ath_presence_ref(&l, seen, p->presence) == ATH_OK;
        }
        ok = ok && ath_sample_ref(&l, p, u[i], &work, &out[i]) == ATH_OK;
        if (ok)
            seen[out[i]] = 1;
    }
    ath_arena_free(&work);
    ATH_EXPECT(ok);
    return ok;
}

/* Speculative decoding of the same prompt: the tokens and the counts. */
static bool speculative(ath_test *t_, const ath_model *m, ath_state *s, const ath_exec *x, const int32_t *ids,
                        int32_t n, const ath_sampler *p, const double *u, int32_t *out, ath_spec *sp, ath_arena *a)
{
    float *lg = ath_arena_alloc(a, (size_t)m->vocab_size * sizeof(float), 64);
    ath_tensor l;
    ATH_EXPECT_OK(ath_tensor_init(&l, ATH_DTYPE_F32, 2, (int64_t[]){1, m->vocab_size}, lg, ATH_TIER_RAM));
    bool ok = ath_state_reset(s, m) == ATH_OK && ath_spec_open(sp, m, s, x, p, 64, a) == ATH_OK;
    ok = ok && ath_spec_feed(sp, ids, n, lg) == ATH_OK;
    ok = ok && ath_sample_ref(&l, p, u[0], &sp->scratch, &out[0]) == ATH_OK;
    ok = ok && ath_spec_begin(sp, out[0]) == ATH_OK;
    for (int i = 1; ok && i < NG;) {
        int32_t got[2], k = 0;
        const double uu[2] = {u[i], i + 1 < NG ? u[i + 1] : 0.5};
        ok = ath_spec_step(sp, uu, got, &k) == ATH_OK;
        for (int j = 0; ok && j < k && i < NG; j++)
            out[i++] = got[j];
    }
    ATH_EXPECT(ok);
    return ok;
}

static void compare(ath_test *t_, const ath_model *m, ath_state *s, const ath_exec *x, const char *where, ath_arena *a)
{
    ath_tokenizer tok;
    ATH_REQUIRE_OK(ath_model_tokenizer(m, model_dir(), &tok, a));
    ath_tokens ids;
    ath_tokens_init(&ids, a);
    ath_arena work;
    ath_arena_init(&work, 0);
    ATH_REQUIRE_OK(ath_tokenizer_encode(&tok, PROMPT, sizeof PROMPT - 1, &ids, &work));
    double u[NG];
    ath_rng rng;
    ath_rng_init(&rng, 2026);
    for (int i = 0; i < NG; i++)
        u[i] = ath_rng_uniform(&rng);
    static const ath_sampler ps[] = {{0, 0, 1, 0, 0}, {0.7f, 20, 0.8f, 0, 0}, {0.7f, 20, 0.8f, 0, 1.5f}};
    static const char *names[] = {"greedy", "T 0.7 top-k 20 top-p 0.8", "the same, presence 1.5"};
    for (size_t k = 0; k < ATH_ARRAY_LEN(ps); k++) {
        int32_t want[NG], got[NG];
        double t0 = now_s();
        if (!plain(t_, m, s, x, ids.ids, (int32_t)ids.n, &ps[k], u, want, &work))
            break;
        double t1 = now_s();
        ath_spec sp;
        if (!speculative(t_, m, s, x, ids.ids, (int32_t)ids.n, &ps[k], u, got, &sp, a)) {
            ath_spec_close(&sp);
            break;
        }
        double t2 = now_s();
        int same = 0;
        while (same < NG && got[same] == want[same])
            same++;
        printf("    %s, %-26s %3d of %d tokens as plain decoding; %llu passes, %llu of %llu guesses kept (%.0f%%), "
               "%llu reruns; %.2f s plain, %.2f s speculative\n",
               where, names[k], same, NG, (unsigned long long)sp.passes, (unsigned long long)sp.accepted,
               (unsigned long long)sp.guesses, sp.guesses ? 100.0 * (double)sp.accepted / (double)sp.passes : 0.0,
               (unsigned long long)sp.reruns, t1 - t0, t2 - t1);
        ATH_EXPECT_EQ_INT(same, NG);
        if (getenv("ATH_SPEC_SHOW")) {
            for (int i = 0; i < 24; i++)
                printf(" %d", want[i]);
            printf("\n");
        }
        ath_spec_close(&sp);
        ath_arena_reset(&work, (ath_arena_pos){0});
    }
    ath_arena_free(&work);
}

ATH_TEST(spec_matches_plain_cpu)
{
    char cfg[1024];
    if (!model_file(cfg, sizeof cfg, "config.json"))
        ATH_SKIP("no checkpoint (ATH_MODEL_DIR)");
    ath_arena a;
    ath_arena_init(&a, 0);
    ath_model m;
    ATH_REQUIRE_OK(ath_model_open(&m, &a, model_dir()));
    if (m.n_draft_layers < 1) {
        ath_model_close(&m);
        ath_arena_free(&a);
        ATH_SKIP("the checkpoint has no draft head");
    }
    ath_pool pool;
    ATH_REQUIRE_OK(ath_pool_init(&pool, ath_pool_default_threads(), &a));
    const ath_exec x = {.pool = &pool};
    ath_state s;
    ATH_REQUIRE_OK(ath_state_init(&s, &m, ATH_DTYPE_BF16, CAP, &a));
    compare(t_, &m, &s, &x, "CPU", &a);
    ath_pool_free(&pool);
    ath_model_close(&m);
    ath_arena_free(&a);
}

ATH_TEST(spec_matches_plain_gpu)
{
    char cfg[1024];
    if (!model_file(cfg, sizeof cfg, "config.json"))
        ATH_SKIP("no checkpoint (ATH_MODEL_DIR)");
    if (ath_gpu_count() == 0)
        ATH_SKIP("no GPU (make cuda test)");
    ath_arena a;
    ath_arena_init(&a, 0);
    ath_model m;
    ATH_REQUIRE_OK(ath_model_open(&m, &a, model_dir()));
    ath_engine e;
    const ath_place_opts po = {.slots = -1, .draft = true};
    ATH_REQUIRE_OK(ath_engine_open_tier(&e, &m, 0, &po, &a));
    ATH_EXPECT(e.draft);
    ath_state s;
    ATH_REQUIRE_OK(ath_engine_state(&e, &s, ATH_DTYPE_BF16, CAP, &a));
    compare(t_, &m, &s, &e.exec, "GPU", &a);
    ath_engine_close(&e);
    ath_model_close(&m);
    ath_arena_free(&a);
}

ATH_TEST(draft_checks)
{
    char cfg[1024];
    if (!model_file(cfg, sizeof cfg, "config.json"))
        ATH_SKIP("no checkpoint (ATH_MODEL_DIR)");
    ath_arena a;
    ath_arena_init(&a, 0);
    ath_model m;
    ATH_REQUIRE_OK(ath_model_open(&m, &a, model_dir()));
    ATH_EXPECT_EQ_INT(m.n_draft_layers, 1);
    uint32_t draft = 0;
    for (uint32_t i = 0; i < m.n_params; i++)
        draft += (m.params[i].flags & ATH_P_DRAFT) != 0;
    ATH_EXPECT(draft >= 13);
    ath_state s;
    ATH_REQUIRE_OK(ath_state_init(&s, &m, ATH_DTYPE_BF16, 16, &a));
    ath_arena scratch;
    ath_arena_init(&scratch, 0);
    void *hb = ath_arena_zalloc(&a, (size_t)m.hidden_size * 2 * 4, 64);
    ath_tensor h;
    ATH_REQUIRE_OK(ath_tensor_init(&h, ATH_DTYPE_BF16, 2, (int64_t[]){2, m.hidden_size}, hb, ATH_TIER_RAM));
    const int32_t toks[2] = {1, 2};
    /* a row at a position the main model has not run */
    ATH_EXPECT_STATUS(ath_model_draft(&m, &(ath_draft){.tokens = toks, .hidden = &h, .n_tokens = 2, .pos0 = 0}, &s,
                                      &scratch, NULL),
                      ATH_ERR_RANGE);
    s.n_past = 2;
    ATH_EXPECT_OK(ath_model_draft(&m, &(ath_draft){.tokens = toks, .hidden = &h, .n_tokens = 2, .pos0 = 0}, &s,
                                  &scratch, NULL));
    ath_tensor hf;
    ATH_REQUIRE_OK(ath_tensor_init(&hf, ATH_DTYPE_F32, 2, (int64_t[]){2, m.hidden_size}, hb, ATH_TIER_RAM));
    ATH_EXPECT_STATUS(ath_model_draft(&m, &(ath_draft){.tokens = toks, .hidden = &hf, .n_tokens = 2, .pos0 = 0}, &s,
                                      &scratch, NULL),
                      ATH_ERR_TYPE);
    const int32_t bad[2] = {1, 1 << 30};
    ATH_EXPECT_STATUS(ath_model_draft(&m, &(ath_draft){.tokens = bad, .hidden = &h, .n_tokens = 2, .pos0 = 0}, &s,
                                      &scratch, NULL),
                      ATH_ERR_RANGE);
    ath_arena_free(&scratch);
    ath_model_close(&m);
    ath_arena_free(&a);
}

static const ath_test_case tests[] = {
    ATH_TEST_ENTRY(draft_checks),
    ATH_TEST_ENTRY(spec_matches_plain_gpu),
    ATH_TEST_ENTRY(spec_matches_plain_cpu),
};

int main(int argc, char **argv)
{
    return ath_test_main(tests, ATH_ARRAY_LEN(tests), argc, argv);
}
