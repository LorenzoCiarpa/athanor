/* test_long.c: a long context on a real model, on the GPU (step 14).
 *
 * Opt-in, minutes per run: ATH_LONG=1. The model is ATH_MODEL_DIR (the
 * gate's is Qwen3.5-9B), the context ATH_LONG_CTX tokens (default 131072),
 * the prompt's pieces ATH_LONG_CHUNK tokens (default 4096; the second
 * chunking is 5/8 of it, both whole chunks of 64), the KV dtypes
 * ATH_LONG_KV (bf16, q8, or both by default).
 *
 *   ATH_LONG=1 ATH_MODEL_DIR=models/Qwen3.5-9B ./build/cuda/test_long
 *
 * One prompt fills the context: some 5,000 short notes, the tenth part of
 * the way in one that gives a passphrase, and at the end the question. For
 * the KV cache in bf16 and in int8:
 *   - retrieval: the greedy answer names the passphrase;
 *   - chunkings: the prompt in pieces of 4096 and of 2560 tokens (whole
 *     chunks of the Gated DeltaNet's 64) gives the same state and logits,
 *     bit for bit;
 *   - int8 against bf16 at depth: the last prompt position and the bf16
 *     answer's tokens fed to the int8 state, KL (at most the bf16 noise)
 *     and the same greedy token;
 *   - reported: prefill time, decode speed at depth, the device memory in
 *     use, the snapshot's size and the time to copy it out at 8k, 32k, 64k
 *     and the full length, and to write and read it on disk.
 */
#include "test.h"
#include "util.h"

#include "core/buf.h"
#include "infer/engine.h"
#include "infer/prefix.h"
#include "model/model.h"
#include "text/bpe.h"

#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define PASSPHRASE "amber-falcon-4127"
#define ANSWER_TOKENS 24

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static const char *const COLORS[] = {"red", "green", "blue", "grey", "yellow", "white", "black", "brown"};
static const char *const THINGS[] = {"crate", "barrel", "letter", "lamp", "rope", "ledger", "kettle", "anchor"};
static const char *const PLACES[] = {"north hall", "east dock", "old mill", "harbor office", "south gate",
                                     "upper loft", "river shed", "chapel yard"};
static const char *const VERBS[] = {"counted", "moved", "painted", "checked", "repaired", "weighed", "sealed",
                                    "cleaned"};

/* Note i of the haystack. */
static void note(ath_buf *b, int i)
{
    const unsigned x = (unsigned)i * 2654435761u;
    ath_buf_printf(b, "Note %d: the %s %s in the %s was %s on day %d. ", i, COLORS[x % 8], THINGS[(x >> 3) % 8],
                   PLACES[(x >> 6) % 8], VERBS[(x >> 9) % 8], 1 + (int)((x >> 12) % 365));
}

/* The prompt of n notes, the passphrase after the tenth part of them. */
static ath_status prompt_of(const ath_model *m, const ath_tokenizer *tok, int n, ath_tokens *ids, ath_arena *a,
                            ath_arena *scratch)
{
    ath_buf text, chat;
    ath_buf_init(&text, a);
    ath_buf_init(&chat, a);
    ath_buf_puts(&text, "Below is a long log of notes. Read it, then answer the question at the end.\n\n");
    for (int i = 0; i < n; i++) {
        if (i == n / 10)
            ath_buf_puts(&text, "Note about the passphrase: the secret passphrase is " PASSPHRASE
                                ". Remember it, you will be asked for it. ");
        note(&text, i);
    }
    ath_buf_puts(&text, "\n\nQuestion: what is the secret passphrase given in the note about the passphrase? "
                        "Answer with the passphrase only.");
    const ath_msg msg = {.role = ATH_MSG_USER, .content = ath_buf_str(&text)};
    ATH_CHECK(ath_model_chat(m, &msg, 1, NULL, 0, ATH_CHAT_GENERATE, &chat));
    ath_tokens_init(ids, a);
    return ath_tokenizer_encode(tok, chat.data, chat.len, ids, scratch);
}

static int64_t argmax(const float *x, int64_t n)
{
    int64_t best = 0;
    for (int64_t i = 1; i < n; i++)
        if (x[i] > x[best])
            best = i;
    return best;
}

/* KL(p || q) of two logit rows, in double. */
static double kl(const float *p, const float *q, int64_t n)
{
    double pm = -INFINITY, qm = -INFINITY, ps = 0, qs = 0, k = 0;
    for (int64_t i = 0; i < n; i++)
        pm = fmax(pm, p[i]), qm = fmax(qm, q[i]);
    for (int64_t i = 0; i < n; i++)
        ps += exp(p[i] - pm), qs += exp(q[i] - qm);
    for (int64_t i = 0; i < n; i++) {
        const double lp = p[i] - pm - log(ps), lq = q[i] - qm - log(qs);
        k += exp(lp) * (lp - lq);
    }
    return k;
}

typedef struct {
    ath_engine      *e;
    const ath_model *m;
    ath_arena       *scratch;
    int64_t          V;
} ctx;

/* ids[0, n) into s in pieces of chunk tokens, the last one's logits into
 * last; at each of the lengths in at[] (crossed by a piece's end) the
 * snapshot's size and the seconds to copy it into buf. */
static ath_status prefill(ctx *c, ath_state *s, const int32_t *ids, int32_t n, int32_t chunk, float *last,
                          const int32_t *at, int n_at, size_t *snap_bytes, double *snap_s, char *buf, size_t cap)
{
    ath_tensor lt;
    ath_tensor_init(&lt, ATH_DTYPE_F32, 2, (int64_t[]){1, c->V}, last, ATH_TIER_RAM);
    int k = 0;
    for (int32_t p = 0; p < n; p += chunk) {
        const int32_t len = n - p < chunk ? n - p : chunk;
        const bool end = p + len == n;
        ATH_CHECK(ath_model_forward(c->m, &(ath_batch){.tokens = ids + p, .n_tokens = len, .logits = end ? &lt : NULL},
                                    s, NULL, c->scratch, &c->e->exec));
        while (k < n_at && p + len >= at[k]) {
            const size_t b = ath_state_snap_bytes(c->m, s);
            ATH_CHECK(ath_stream_sync(&c->e->dev, ATH_STREAM_COMPUTE));   /* the pass done: the copy alone timed */
            const double t0 = now();
            if (b <= cap)
                ATH_CHECK(ath_state_snap_save(s, c->m, buf, b));
            snap_bytes[k] = b;
            snap_s[k] = b <= cap ? now() - t0 : -1;
            k++;
        }
    }
    return ath_stream_sync(&c->e->dev, ATH_STREAM_COMPUTE);
}

/* One token at a time from s, its logits into lg[i * V]: tokens given. */
static ath_status forced(ctx *c, ath_state *s, int32_t first, const int32_t *toks, int n, float *lg)
{
    int32_t t = first;
    for (int i = 0; i < n; i++) {
        ath_tensor l;
        ath_tensor_init(&l, ATH_DTYPE_F32, 2, (int64_t[]){1, c->V}, lg + (size_t)i * (size_t)c->V, ATH_TIER_RAM);
        ATH_CHECK(ath_model_forward(c->m, &(ath_batch){.tokens = &t, .n_tokens = 1, .logits = &l}, s, NULL, c->scratch,
                                    &c->e->exec));
        t = toks[i];
    }
    return ATH_OK;
}

static void remove_folder(const char *dir)
{
    DIR *d = opendir(dir);
    char p[1200];
    for (const struct dirent *de; d && (de = readdir(d));)
        if (strcmp(de->d_name, ".") && strcmp(de->d_name, "..") && snprintf(p, sizeof p, "%s/%s", dir, de->d_name) > 0)
            unlink(p);
    if (d)
        closedir(d);
    rmdir(dir);
}

ATH_TEST(long_context)
{
    if (!getenv("ATH_LONG"))
        ATH_SKIP("opt-in: ATH_LONG=1 (minutes)");
    if (ath_gpu_count() < 1)
        ATH_SKIP("no GPU");
    const char *ce = getenv("ATH_LONG_CTX");
    const int32_t C = ce && atoi(ce) >= 4096 ? atoi(ce) : 131072;
    const char *ke = getenv("ATH_LONG_CHUNK");
    const int32_t chunk1 = ke && atoi(ke) >= 128 ? atoi(ke) / 64 * 64 : 4096, chunk2 = chunk1 * 5 / 8 / 64 * 64;
    ath_arena a, scratch;
    ath_arena_init(&a, 0);
    ath_arena_init(&scratch, 0);
    ath_model m = {.family = NULL};
    ath_tokenizer tok;
    ATH_REQUIRE_OK(ath_model_open(&m, &a, model_dir()));
    ATH_REQUIRE_OK(ath_model_tokenizer(&m, model_dir(), &tok, &a));
    const int64_t V = m.vocab_size;

    /* the prompt: as many notes as leave room for the answer */
    ath_tokens ids;
    ATH_REQUIRE_OK(prompt_of(&m, &tok, 100, &ids, &a, &scratch));
    const double per_note = (double)ids.n / 100.0;
    int notes = (int)((C - ANSWER_TOKENS - 64) / per_note);
    for (int tries = 0; tries < 8; tries++) {
        ATH_REQUIRE_OK(prompt_of(&m, &tok, notes, &ids, &a, &scratch));
        if ((int64_t)ids.n <= C - ANSWER_TOKENS - 8)
            break;
        notes -= (int)(((int64_t)ids.n - (C - ANSWER_TOKENS - 8)) / per_note) + 4;
    }
    const int32_t N = (int32_t)ids.n;
    ATH_REQUIRE(N <= C - ANSWER_TOKENS - 8);
    printf("    %s: a prompt of %d tokens (%d notes), context %d\n", model_dir(), N, notes, C);

    ath_engine *e = ath_arena_zalloc(&a, sizeof *e, _Alignof(ath_engine));
    ATH_REQUIRE(e);
    ATH_REQUIRE_OK(ath_engine_open(e, &m, 0, &a));
    ctx c = {.e = e, .m = &m, .scratch = &scratch, .V = V};
    uint64_t free0 = 0, total = 0;
    ath_device_mem(&e->dev, &free0, &total);
    printf("    weights on %s: %.2f GiB; %.2f of %.2f GiB free\n", e->dev.name, (double)e->weight_bytes / 1073741824.0,
           (double)free0 / 1073741824.0, (double)total / 1073741824.0);

    static const int32_t at[] = {8192, 32768, 65536, 1 << 30};
    float *last16 = malloc((size_t)V * sizeof(float)), *last = malloc((size_t)V * sizeof(float));
    float *lastb = malloc((size_t)V * sizeof(float));
    float *ref = malloc((size_t)ANSWER_TOKENS * (size_t)V * sizeof(float));
    float *lg = malloc((size_t)ANSWER_TOKENS * (size_t)V * sizeof(float));
    int32_t answer16[ANSWER_TOKENS];
    ATH_REQUIRE(last16 && last && lastb && ref && lg);
    tmpdir sd;
    ATH_REQUIRE(tmpdir_make(&sd));
    char dir[600];
    snprintf(dir, sizeof dir, "%s/snapshots", sd.path);

    const char *kvs = getenv("ATH_LONG_KV");
    bool have_ref = false;   /* the bf16 run's answer and logits, for the int8 run */
    for (int k = 0; k < 2; k++) {
        const ath_dtype kv = k ? ATH_DTYPE_I8 : ATH_DTYPE_BF16;
        const char *kn = k ? "int8" : "bf16";
        if (kvs && strcmp(kvs, k ? "q8" : "bf16"))
            continue;
        ath_state s;
        ath_status st = ath_engine_state_kv(e, &s, ATH_DTYPE_BF16, kv, C, &a);
        if (st != ATH_OK) {
            printf("    %s KV: a state of %d tokens does not fit (%s)\n", kn, C, ath_status_str(st));
            ATH_EXPECT_OK(st);
            continue;
        }
        size_t full = 0;
        ATH_REQUIRE_OK(ath_state_bytes_kv(&m, ATH_DTYPE_BF16, kv, C, &full));
        char *snap = malloc(full), *snap2 = malloc(full);
        ATH_REQUIRE(snap && snap2);

        /* the prompt in pieces of chunk1, snapshots along the way */
        size_t sb[4] = {0};
        double ss[4] = {0};
        double t0 = now();
        ATH_REQUIRE_OK(prefill(&c, &s, ids.ids, N, chunk1, last, at, 3, sb, ss, snap, full));
        const double t_prefill = now() - t0;
        const size_t bytes = ath_state_snap_bytes(&m, &s);
        t0 = now();
        ATH_REQUIRE_OK(ath_state_snap_save(&s, &m, snap, bytes));
        sb[3] = bytes, ss[3] = now() - t0;
        uint64_t fr = 0;
        ath_device_mem(&e->dev, &fr, &total);

        /* the answer, greedy, with the decoder */
        static const ath_sampler greedy = {0, 0, 1, 0, 0};
        ath_decoder dc;
        ATH_REQUIRE_OK(ath_decoder_open(&dc, e, &s, &greedy, &a));
        int32_t ans[ANSWER_TOKENS], t = (int32_t)argmax(last, V);
        int n_ans = 0;
        t0 = now();
        for (; n_ans < ANSWER_TOKENS; n_ans++) {
            ans[n_ans] = t;
            if (t == ath_tokenizer_added(&tok, "<|im_end|>"))
                break;
            ATH_REQUIRE_OK(ath_decoder_step(&dc, t, 0.5, &t));
        }
        const double t_decode = now() - t0;
        ath_decoder_close(&dc);
        ath_buf text;
        ath_buf_init(&text, &a);
        ATH_REQUIRE_OK(ath_tokenizer_decode(&tok, ans, (size_t)n_ans, &text));
        const bool found = strstr(ath_buf_str(&text), PASSPHRASE) != NULL;
        printf("    %s KV: prefill %.1f s (%.0f tokens/s); decode at depth %.1f tokens/s; device memory in use "
               "%.2f GiB\n",
               kn, t_prefill, N / t_prefill, n_ans / t_decode, (double)(total - fr) / 1073741824.0);
        printf("    %s KV: answer \"%.60s\": the passphrase %s\n", kn, ath_buf_str(&text), found ? "found" : "MISSING");
        ATH_EXPECT(found);
        for (int i = 0; i < 4; i++)
            if (i == 3 || at[i] < N)
                printf("    %s KV: snapshot at %6d tokens: %8.1f MiB, copied out in %.3f s\n", kn, i < 3 ? at[i] : N,
                   (double)sb[i] / 1048576.0, ss[i]);

        /* the same prompt in pieces of chunk2: the same state and logits */
        ATH_REQUIRE_OK(ath_state_reset(&s, &m));
        ATH_REQUIRE_OK(prefill(&c, &s, ids.ids, N, chunk2, lastb, at, 0, NULL, NULL, NULL, 0));
        ATH_REQUIRE_OK(ath_state_snap_save(&s, &m, snap2, bytes));
        const bool same_state = memcmp(snap, snap2, bytes) == 0, same_logits = memcmp(last, lastb, (size_t)V * 4) == 0;
        printf("    %s KV: pieces of %d against %d: state %s, logits %s\n", kn, chunk2, chunk1,
               same_state ? "the same bits" : "DIFFER", same_logits ? "the same bits" : "DIFFER");
        ATH_EXPECT(same_state && same_logits);

        /* the bf16 answer's tokens from the prompt's end */
        ATH_REQUIRE_OK(ath_state_snap_load(&s, &m, N, 0, snap, bytes));
        if (!k) {
            memcpy(last16, last, (size_t)V * 4);
            memcpy(answer16, ans, sizeof ans);
            for (int i = n_ans; i < ANSWER_TOKENS; i++)
                answer16[i] = ans[n_ans - 1 >= 0 ? n_ans - 1 : 0];
            ATH_REQUIRE_OK(forced(&c, &s, answer16[0], answer16 + 1, ANSWER_TOKENS - 1, ref + V));
            memcpy(ref, last16, (size_t)V * 4);
            have_ref = true;
        } else if (have_ref) {
            ATH_REQUIRE_OK(forced(&c, &s, answer16[0], answer16 + 1, ANSWER_TOKENS - 1, lg + V));
            memcpy(lg, last, (size_t)V * 4);
            double sum = 0, worst = 0;
            int same = 0;
            for (int i = 0; i < ANSWER_TOKENS; i++) {
                const double d = kl(ref + (size_t)i * V, lg + (size_t)i * V, V);
                sum += d, worst = fmax(worst, d);
                same += argmax(ref + (size_t)i * V, V) == argmax(lg + (size_t)i * V, V);
            }
            printf("    int8 against bf16 KV at depth %d: mean KL %.3e (max %.3e) over %d positions, the same greedy "
                   "token at %d\n",
                   N, sum / ANSWER_TOKENS, worst, ANSWER_TOKENS, same);
            /* the bound: the bf16 noise, the bf16 reference's KL against fp32
             * on the forward records (3.8e-4 on Qwen3.5-9B, 5.0e-4 on the
             * 0.8B): the int8 cache moves the output less than bf16 rounding
             * does */
            ATH_EXPECT(sum / ANSWER_TOKENS <= 3.8e-4);
        }

        /* the snapshot on disk: written, read back */
        ath_prefix_store *ps;
        ATH_REQUIRE_OK(ath_prefix_open(&ps, &m, ATH_DTYPE_BF16, kv, &(ath_prefix_opts){.dir = dir}));
        ATH_REQUIRE_OK(ath_state_snap_load(&s, &m, N, 0, snap, bytes));
        ATH_REQUIRE_OK(ath_prefix_save(ps, ids.ids, &s));
        t0 = now();
        ath_prefix_close(ps);
        const double t_write = now() - t0;
        ATH_REQUIRE_OK(ath_prefix_open(&ps, &m, ATH_DTYPE_BF16, kv, &(ath_prefix_opts){.dir = dir}));
        ATH_REQUIRE_OK(ath_state_reset(&s, &m));
        ATH_EXPECT_EQ_INT(ath_prefix_find(ps, ids.ids, N + 1), N);
        ath_tokens more = ids;
        ATH_REQUIRE_OK(ath_prefix_load(ps, more.ids, N, N, &s));
        ath_prefix_stats pst;
        ath_prefix_get_stats(ps, &pst);
        ATH_REQUIRE_OK(ath_state_snap_save(&s, &m, snap2, bytes));
        ATH_EXPECT(memcmp(snap, snap2, bytes) == 0);
        printf("    %s KV: snapshot of %d tokens on disk: %.1f MiB, written in %.2f s, read and loaded in %.2f s\n", kn,
               N, (double)bytes / 1048576.0, t_write, pst.load_s);
        ath_prefix_close(ps);
        remove_folder(dir);

        free(snap);
        free(snap2);
        ath_dev_arena_free(&e->state_mem);   /* the device memory itself, for the next state */
    }
    rmdir(sd.path);
    free(last16), free(last), free(lastb), free(ref), free(lg);
    ath_engine_close(e);
    ath_model_close(&m);
    ath_arena_free(&scratch);
    ath_arena_free(&a);
}

static const ath_test_case tests[] = {
    ATH_TEST_ENTRY(long_context),
};

int main(int argc, char **argv)
{
    return ath_test_main(tests, ATH_ARRAY_LEN(tests), argc, argv);
}
