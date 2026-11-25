/* chat.c: `athanor chat <model_dir> [options]`: a conversation, on the GPU
 * when the build and the machine have one (step 8), on the CPU otherwise.
 *
 * Every turn goes through the contract: the family renders the whole
 * conversation with its chat template (ath_model_chat), the tokenizer turns
 * it into ids, the forward pass runs (ath_model_forward, on the engine's
 * device or on the CPU with a thread pool), a token is sampled from the
 * logits on the host (ath_sample_ref), decoded, and the
 * family's parser (ath_model_parse) splits the output into text, thinking
 * and tool calls. Nothing here knows the model family.
 *
 * The state is reused across turns where the prompt allows it. When the new
 * prompt starts with every token the state has seen, only the new tokens
 * run. When it does not (the template rewrites the assistant's turn once
 * it is history: a past turn may lose its thinking block), a copy of the state
 * taken at the end of the previous prompt's last message is restored
 * instead, and when even that does not match, the conversation runs again
 * from the start. A recurrent state cannot be rolled back, hence the copy.
 *
 * athanor never runs a tool (AGENTS.md, invariant 10). With --tools the
 * model sees the definitions and may call them; the call is printed, and
 * the user can answer with /tool <result>, acting as the harness.
 */
#include "commands.h"

#include "core/buf.h"
#include "core/file.h"
#include "core/json.h"
#include "core/rng.h"
#include "infer/engine.h"
#include "infer/spec.h"
#include "model/model.h"
#include "ops/ops.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

void ath_cmd_chat_usage(FILE *out)
{
    fprintf(out,
            "usage: athanor chat <model_dir> [options]\n\n"
            "Chat with the model, on the GPU when this build has one (make cuda) and the\n"
            "machine too, on the CPU otherwise. Without -p, read one message per line\n"
            "from stdin (end a line with \\ to continue it); /reset starts over,\n"
            "/tool <text> answers a tool call, /exit (or end of input) quits.\n\n"
            "  -p, --prompt <text>  send one message, print the answer, exit\n"
            "  --system <text>      system message\n"
            "  --think              let the model think before it answers (thinking is\n"
            "                       shown dimmed on a terminal, on stderr otherwise)\n"
            "  --tools <file.json>  tool definitions, a JSON array in OpenAI format; calls are\n"
            "                       shown, never run\n"
            "  --temp <t>           temperature, 0 for greedy (default 0.7; 0.6 with --think)\n"
            "  --top-k <k>          default 20\n"
            "  --top-p <p>          default 0.8 (0.95 with --think)\n"
            "  --min-p <p>          default 0\n"
            "  --seed <n>           random seed (default: from the clock)\n"
            "  --max-tokens <n>     tokens per answer (default 2048)\n"
            "  --ctx <n>            context length in tokens (default 8192)\n"
            "  --device <cpu|gpu>   where the model runs (default: gpu if there is one)\n"
            "  --threads <n>        CPU threads (default: all)\n"
            "  --experts <n>        MoE: routed experts per layer kept in device memory\n"
            "                       (default: as many as fit; the rest run on the CPU)\n"
            "  --f32                f32 activations instead of bf16 (slower)\n"
            "  --mtp                decode speculatively with the model's draft head: the same\n"
            "                       tokens, fewer passes (on a GPU slower than the default\n"
            "                       decoder until its passes are recorded too)\n"
            "  --stats              print prompt and decode speed after each answer (stderr)\n");
}

typedef struct {
    const char  *dir, *prompt, *system, *tools_file, *device;
    bool         think, f32, stats, mtp;
    ath_sampler  sampler;
    bool         temp_set, top_p_set;
    uint64_t     seed;
    bool         seed_set;
    int32_t      max_tokens, ctx, threads;
    int32_t      experts;     /* routed experts per layer in device memory, -1: as many as fit */
} options;

/* ---- the conversation ------------------------------------------------------------ */

typedef struct {
    ath_msg   *msgs;
    int32_t    n, cap;
    ath_arena *a;       /* the messages and their strings */
} history;

static ath_status history_add(history *h, ath_msg m)
{
    if (h->n == h->cap) {
        int32_t cap = h->cap ? 2 * h->cap : 16;
        ath_msg *g = ath_arena_alloc(h->a, (size_t)cap * sizeof *g, _Alignof(ath_msg));
        if (!g)
            return ATH_ERR_NOMEM;
        if (h->n)
            memcpy(g, h->msgs, (size_t)h->n * sizeof *g);
        h->msgs = g, h->cap = cap;
    }
    h->msgs[h->n++] = m;
    return ATH_OK;
}

/* The tool definitions of a JSON file: [{"type": "function", "function":
 * {"name", "description", "parameters"}}] (or the inner objects). */
static ath_status load_tools(ath_arena *a, const char *path, ath_tool_def **tools, int32_t *n)
{
    ath_file f;
    ATH_CHECK(ath_file_open(&f, path));
    const ath_json *root;
    ath_status s = ath_json_parse(a, f.data, f.size, &root, NULL);
    ath_file_close(&f);
    if (s != ATH_OK || root->type != ATH_JSON_ARRAY) {
        fprintf(stderr, "athanor chat: %s: not a JSON array of tools\n", path);
        return ATH_ERR_PARSE;
    }
    *n = (int32_t)ath_json_count(root);
    *tools = ath_arena_zalloc(a, (size_t)(*n ? *n : 1) * sizeof **tools, _Alignof(ath_tool_def));
    if (!*tools)
        return ATH_ERR_NOMEM;
    int32_t i = 0;
    for (const ath_json *t = ath_json_first(root); t; t = t->next, i++) {
        const ath_json *fn = ath_json_get(t, "function") ? ath_json_get(t, "function") : t;
        ath_tool_def *d = &(*tools)[i];
        if (ath_json_to_str(ath_json_get(fn, "name"), &d->name, NULL) != ATH_OK) {
            fprintf(stderr, "athanor chat: %s: tool %d has no name\n", path, i);
            return ATH_ERR_PARSE;
        }
        ath_json_to_str(ath_json_get(fn, "description"), &d->description, NULL);
        const ath_json *params = ath_json_get(fn, "parameters");
        if (params) {
            ath_buf b;
            ath_buf_init(&b, a);
            ATH_CHECK(ath_json_write(params, &b));
            d->json_schema = b.data;
        }
    }
    return ATH_OK;
}

/* ---- running the model ----------------------------------------------------------- */

typedef struct {
    const options   *opt;
    ath_model        m;
    ath_tokenizer    tok;
    ath_pool         pool;
    ath_engine      *engine;        /* NULL on the CPU                               */
    ath_decoder     *dec;           /* the engine's decoder of state (step 9)         */
    ath_spec        *spec;          /* with --mtp: the speculative decoder (step 13)  */
    ath_exec         exec;          /* where the forward pass runs                   */
    ath_state        state, snap;
    ath_tokens       seen;          /* the tokens the state holds                    */
    size_t           snap_n;        /* the first snap_n of them are in snap; 0: none  */
    ath_tensor       logits;
    ath_arena       *scratch;
    ath_rng          rng;
    int32_t          eos[2];
    ath_tool_def    *tools;
    int32_t          n_tools;
    bool             tty;
} session;

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* Run ids[from .. to) through the model; logits for the last one if want. */
static ath_status feed(session *s, const int32_t *ids, size_t from, size_t to, bool want)
{
    for (size_t i = from; i < to; i++)
        ATH_CHECK(ath_tokens_push(&s->seen, ids[i]));
    if (from == to)
        return ATH_OK;
    ath_batch b = {.tokens = ids + from, .n_tokens = (int32_t)(to - from), .logits = want ? &s->logits : NULL};
    if (s->spec)   /* the draft head runs over the tokens too */
        return ath_spec_feed(s->spec, b.tokens, b.n_tokens, want ? s->logits.data : NULL);
    return ath_model_forward(&s->m, &b, &s->state, NULL, s->scratch, &s->exec);
}

/* Bring the state to ids[0 .. stable) and then ids[stable .. n), keeping a
 * copy at stable. Returns the prompt tokens actually run in *ran. */
static ath_status prefill(session *s, const ath_tokens *ids, size_t stable, size_t *ran)
{
    size_t common = 0;
    while (common < s->seen.n && common < ids->n && s->seen.ids[common] == ids->ids[common])
        common++;
    if (common < s->seen.n) {
        if (s->snap_n && s->snap_n <= common) {
            ATH_CHECK(ath_state_copy(&s->state, &s->snap, &s->m));
            s->seen.n = s->snap_n;
        } else {
            ATH_CHECK(ath_state_reset(&s->state, &s->m));
            s->seen.n = 0;
        }
    }
    size_t start = s->seen.n;
    if (start >= ids->n) {   /* nothing new: the last token's logits are needed */
        ATH_CHECK(ath_state_reset(&s->state, &s->m));
        s->seen.n = 0, start = 0;
    }
    if (start < stable) {
        ATH_CHECK(feed(s, ids->ids, start, stable, false));
        ATH_CHECK(ath_state_copy(&s->snap, &s->state, &s->m));
        s->snap_n = stable;
    }
    ATH_CHECK(feed(s, ids->ids, start > stable ? start : stable, ids->n, true));
    *ran = ids->n - start;
    return ATH_OK;
}

typedef struct {
    ath_buf        text, thinking, args;
    ath_tool_call *calls;
    int32_t        n_calls, cap_calls;
    bool           in_thinking;
} answer;

static void show(session *s, answer *ans, const ath_out_event *ev, int32_t n, ath_arena *a)
{
    for (int32_t i = 0; i < n; i++) {
        const ath_out_event *e = &ev[i];
        switch (e->kind) {
        case ATH_OUT_TEXT:
            if (ans->in_thinking && s->tty)
                fputs("\033[0m\n", stdout);
            ans->in_thinking = false;
            fwrite(e->data, 1, e->len, stdout);
            ath_buf_append(&ans->text, e->data, e->len);
            break;
        case ATH_OUT_THINKING:
            if (!ans->in_thinking && s->tty)
                fputs("\033[2m", stdout);
            ans->in_thinking = true;
            fwrite(e->data, 1, e->len, s->tty ? stdout : stderr);
            ath_buf_append(&ans->thinking, e->data, e->len);
            break;
        case ATH_OUT_TOOL_CALL_BEGIN:
            if (ans->in_thinking && s->tty)
                fputs("\033[0m", stdout);
            ans->in_thinking = false;
            printf("\n[tool call] %s ", e->tool_name);
            if (ans->n_calls == ans->cap_calls) {
                int32_t cap = ans->cap_calls ? 2 * ans->cap_calls : 4;
                ath_tool_call *c = ath_arena_alloc(a, (size_t)cap * sizeof *c, _Alignof(ath_tool_call));
                if (c && ans->n_calls)
                    memcpy(c, ans->calls, (size_t)ans->n_calls * sizeof *c);
                ans->calls = c, ans->cap_calls = c ? cap : 0;
            }
            if (ans->calls)
                ans->calls[ans->n_calls] = (ath_tool_call){.name = e->tool_name};
            ath_buf_init(&ans->args, a);
            break;
        case ATH_OUT_TOOL_CALL_ARGS:
            fwrite(e->data, 1, e->len, stdout);
            ath_buf_append(&ans->args, e->data, e->len);
            break;
        case ATH_OUT_TOOL_CALL_END:
            if (ans->calls && ans->n_calls < ans->cap_calls)
                ans->calls[ans->n_calls++].arguments = ans->args.data;
            printf("\n");
            break;
        }
    }
    fflush(stdout);
    fflush(stderr);
}

/* One assistant turn: render, prefill, generate, show. The answer is added
 * to the history. */
static ath_status turn(session *s, history *h, ath_arena *tmp)
{
    const options *o = s->opt;
    uint32_t opts = ATH_CHAT_GENERATE | (o->think ? ATH_CHAT_THINKING : 0);
    ath_buf prompt, stable;
    ath_buf_init(&prompt, tmp);
    ath_buf_init(&stable, tmp);
    ATH_CHECK(ath_model_chat(&s->m, h->msgs, h->n, s->tools, s->n_tools, opts, &prompt));
    ath_tokens ids, stable_ids;
    ath_tokens_init(&ids, tmp);
    ath_tokens_init(&stable_ids, tmp);
    ATH_CHECK(ath_tokenizer_encode(&s->tok, prompt.data, prompt.len, &ids, s->scratch));
    /* The stable part, where a copy of the state is kept: the conversation
     * up to the last message, without the assistant's header that follows
     * (the next turn renders this answer differently), when its tokens are
     * a prefix of the prompt's. A single prompt (-p) keeps no copy. */
    size_t stable_n = 0;
    if (!o->prompt && ath_model_chat(&s->m, h->msgs, h->n, s->tools, s->n_tools, 0, &stable) == ATH_OK &&
        ath_tokenizer_encode(&s->tok, stable.data, stable.len, &stable_ids, s->scratch) == ATH_OK &&
        stable_ids.n <= ids.n && !memcmp(stable_ids.ids, ids.ids, stable_ids.n * sizeof *ids.ids))
        stable_n = stable_ids.n;
    if ((int64_t)ids.n + 1 > o->ctx) {
        fprintf(stderr, "athanor chat: the conversation (%zu tokens) does not fit in --ctx %d; /reset\n", ids.n,
                o->ctx);
        return ATH_ERR_RANGE;
    }

    double t0 = now_s();
    size_t ran;
    ATH_CHECK(prefill(s, &ids, stable_n, &ran));
    double t1 = now_s();

    ath_parse_state ps = {.opts = opts, .tools = s->tools, .n_tools = s->n_tools, .a = tmp};
    answer ans = {.in_thinking = false};
    ath_buf_init(&ans.text, h->a);
    ath_buf_init(&ans.thinking, h->a);
    ath_out_event ev[64];
    int32_t ne, n_gen = 0;
    int32_t room = o->ctx - (int32_t)s->seen.n;
    int32_t limit = o->max_tokens < room ? o->max_tokens : room;
    /* the first token from the prompt's logits; then, on a GPU, the decoder
     * runs each token and chooses the next on the device (one u per token
     * chosen, as on the CPU) */
    int32_t next = 0;
    /* with --mtp: a speculative step's tokens, and the uniform number a
     * one-token step left unused */
    int32_t queue[2], nq = 0, qi = 0;
    double carry = -1;
    ath_tokens emitted;
    ath_tokens_init(&emitted, tmp);
    const size_t prompt_n = s->seen.n;
    if (limit > 0)
        ATH_CHECK(ath_sample_ref(&s->logits, &o->sampler, ath_rng_uniform(&s->rng), s->scratch, &next));
    for (; n_gen < limit; n_gen++) {
        if (n_gen > 0 && s->spec) {
            if (qi == nq) {
                if (s->state.n_past + ATH_SPEC_ROWS > o->ctx)
                    break;
                double u[2];
                u[0] = carry >= 0 ? carry : ath_rng_uniform(&s->rng);
                u[1] = ath_rng_uniform(&s->rng);
                int32_t k = 0;
                ATH_CHECK(ath_spec_step(s->spec, u, queue, &k));
                nq = k, qi = 0;
                carry = k == 1 ? u[1] : -1;
            }
            next = queue[qi++];
        } else if (n_gen > 0 && !s->dec)
            ATH_CHECK(ath_sample_ref(&s->logits, &o->sampler, ath_rng_uniform(&s->rng), s->scratch, &next));
        if (next == s->eos[0] || next == s->eos[1])
            break;
        size_t len;
        const char *piece = ath_tokenizer_token(&s->tok, next, &len);
        ATH_CHECK(ath_model_parse(&s->m, &ps, piece, len, ev, 64, &ne));
        show(s, &ans, ev, ne, h->a);
        ATH_CHECK(ath_tokens_push(&emitted, next));
        if (s->spec) {
            if (n_gen == 0 && n_gen + 1 < limit)
                ATH_CHECK(ath_spec_begin(s->spec, next));
        } else if (n_gen + 1 < limit && s->dec) {
            ATH_CHECK(ath_tokens_push(&s->seen, next));
            ATH_CHECK(ath_decoder_step(s->dec, next, ath_rng_uniform(&s->rng), &next));
        } else if (n_gen + 1 < limit) {
            ATH_CHECK(feed(s, &next, 0, 1, true));
        }
    }
    /* the state holds the prompt and the answer up to n_past (a speculative
     * step leaves its last token for the next pass) */
    for (size_t i = s->seen.n - prompt_n; s->spec && s->seen.n < (size_t)s->state.n_past && i < emitted.n; i++)
        ATH_CHECK(ath_tokens_push(&s->seen, emitted.ids[i]));
    ATH_CHECK(ath_model_parse(&s->m, &ps, NULL, 0, ev, 64, &ne));
    show(s, &ans, ev, ne, h->a);
    if (ans.in_thinking && s->tty)
        fputs("\033[0m", stdout);
    printf("\n");
    fflush(stdout);
    double t2 = now_s();
    if (n_gen == limit)
        fprintf(stderr, "athanor chat: stopped after %d tokens (--max-tokens or --ctx)\n", n_gen);
    if (o->stats)
        fprintf(stderr, "[prompt %zu tokens, %zu run, %.2f s; answer %d tokens, %.2f s, %.1f tokens/s]\n", ids.n,
                ran, t1 - t0, n_gen, t2 - t1, n_gen / (t2 - t1 > 0 ? t2 - t1 : 1));

    return history_add(h, (ath_msg){.role = ATH_MSG_ASSISTANT,
                                    .content = ath_buf_str(&ans.text),
                                    .reasoning = ans.thinking.len ? ath_buf_str(&ans.thinking) : NULL,
                                    .tool_calls = ans.calls,
                                    .n_tool_calls = ans.n_calls});
}

/* ---- input --------------------------------------------------------------------- */

/* One message from stdin: lines joined while they end with a backslash.
 * NULL at the end of input. */
static const char *read_message(ath_arena *a, bool tty)
{
    ath_buf b;
    ath_buf_init(&b, a);
    char line[4096];
    if (tty) {
        fputs("> ", stdout);
        fflush(stdout);
    }
    bool any = false;
    while (fgets(line, sizeof line, stdin)) {
        any = true;
        size_t n = strlen(line);
        bool full = n && line[n - 1] == '\n';
        if (full)
            line[--n] = '\0';
        if (full && n && line[n - 1] == '\\') {
            line[n - 1] = '\n';
            ath_buf_append(&b, line, n);
            continue;
        }
        ath_buf_append(&b, line, n);
        if (full)
            break;
    }
    return any ? ath_buf_str(&b) : NULL;
}

/* ---- options and main ---------------------------------------------------------- */

static int usage_error(const char *msg, const char *arg)
{
    fprintf(stderr, "athanor chat: %s%s%s\n\n", msg, arg ? " " : "", arg ? arg : "");
    ath_cmd_chat_usage(stderr);
    return ATH_EXIT_USAGE;
}

static bool num(const char *s, double lo, double hi, double *out)
{
    char *end;
    double v = strtod(s, &end);
    if (end == s || *end || !(v >= lo && v <= hi))
        return false;
    *out = v;
    return true;
}

static int parse_options(options *o, int argc, char **argv)
{
    *o = (options){.max_tokens = 2048, .ctx = 8192, .sampler = {0.7f, 20, 0.8f, 0}, .experts = -1};
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        const char *v = i + 1 < argc ? argv[i + 1] : NULL;
        double x;
        if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            ath_cmd_chat_usage(stdout);
            return -1;
        }
        if (!strcmp(a, "--think")) { o->think = true; continue; }
        if (!strcmp(a, "--f32")) { o->f32 = true; continue; }
        if (!strcmp(a, "--stats")) { o->stats = true; continue; }
        if (!strcmp(a, "--mtp")) { o->mtp = true; continue; }
        if (a[0] == '-' && !v)
            return usage_error("missing value for", a);
        if (!strcmp(a, "-p") || !strcmp(a, "--prompt")) o->prompt = v;
        else if (!strcmp(a, "--system")) o->system = v;
        else if (!strcmp(a, "--tools")) o->tools_file = v;
        else if (!strcmp(a, "--temp") && num(v, 0, 100, &x)) o->sampler.temperature = (float)x, o->temp_set = true;
        else if (!strcmp(a, "--top-k") && num(v, 0, 1e9, &x)) o->sampler.top_k = (int32_t)x;
        else if (!strcmp(a, "--top-p") && num(v, 1e-6, 1, &x)) o->sampler.top_p = (float)x, o->top_p_set = true;
        else if (!strcmp(a, "--min-p") && num(v, 0, 1, &x)) o->sampler.min_p = (float)x;
        else if (!strcmp(a, "--seed") && num(v, 0, 1.8e19, &x)) o->seed = (uint64_t)x, o->seed_set = true;
        else if (!strcmp(a, "--max-tokens") && num(v, 1, 1e9, &x)) o->max_tokens = (int32_t)x;
        else if (!strcmp(a, "--ctx") && num(v, 16, 1 << 24, &x)) o->ctx = (int32_t)x;
        else if (!strcmp(a, "--threads") && num(v, 1, 4096, &x)) o->threads = (int32_t)x;
        else if (!strcmp(a, "--experts") && num(v, 0, 1e6, &x)) o->experts = (int32_t)x;
        else if (!strcmp(a, "--device") && (!strcmp(v, "cpu") || !strcmp(v, "gpu"))) o->device = v;
        else if (a[0] == '-')
            return usage_error("bad option or value:", a);
        else if (!o->dir) {
            o->dir = a;
            continue;
        } else
            return usage_error("unexpected argument", a);
        i++;
    }
    if (!o->dir)
        return usage_error("missing <model_dir>", NULL);
    /* The settings model cards of thinking models recommend: thinking 0.6 / 0.95 / 20,
     * otherwise 0.7 / 0.8 / 20 (docs/contract/ for the model at hand). */
    if (o->think && !o->temp_set)
        o->sampler.temperature = 0.6f;
    if (o->think && !o->top_p_set)
        o->sampler.top_p = 0.95f;
    if (!o->seed_set)
        o->seed = (uint64_t)time(NULL);
    if (!o->threads)
        o->threads = ath_pool_default_threads();
    return 0;
}

static int run(session *s, ath_arena *a)
{
    const options *o = s->opt;
    ath_arena hist_arena, turn_arena;
    ath_arena_init(&hist_arena, 0);
    ath_arena_init(&turn_arena, 0);
    history h = {.a = &hist_arena};
    const ath_arena_pos turn0 = ath_arena_mark(&turn_arena);
    int rc = ATH_EXIT_OK;
    if (o->system && history_add(&h, (ath_msg){.role = ATH_MSG_SYSTEM, .content = o->system}) != ATH_OK)
        rc = ATH_EXIT_FAILURE;

    while (rc == ATH_EXIT_OK) {
        ath_arena_reset(&turn_arena, turn0);
        const char *line = o->prompt ? o->prompt : read_message(&turn_arena, s->tty);
        if (!line || !strcmp(line, "/exit"))
            break;
        if (!strcmp(line, "/reset")) {
            h.n = o->system ? 1 : 0;
            continue;
        }
        bool tool = !strncmp(line, "/tool ", 6);
        const char *content = ath_arena_strndup(&hist_arena, tool ? line + 6 : line, strlen(tool ? line + 6 : line));
        if (!content || history_add(&h, (ath_msg){.role = tool ? ATH_MSG_TOOL : ATH_MSG_USER, .content = content})) {
            rc = ATH_EXIT_FAILURE;
            break;
        }
        ath_status st = turn(s, &h, &turn_arena);
        if (st == ATH_ERR_RANGE || st == ATH_ERR_INVALID_ARG) {
            h.n--;   /* the message is dropped; the conversation goes on */
            if (o->prompt)
                rc = ATH_EXIT_FAILURE;
        } else if (st != ATH_OK) {
            fprintf(stderr, "athanor chat: %s\n", ath_status_str(st));
            rc = ATH_EXIT_FAILURE;
        }
        if (o->prompt)
            break;
    }
    ath_arena_free(&turn_arena);
    ath_arena_free(&hist_arena);
    (void)a;
    return rc;
}

int ath_cmd_chat(ath_arena *a, int argc, char **argv)
{
    options o;
    int r = parse_options(&o, argc, argv);
    if (r)
        return r < 0 ? ATH_EXIT_OK : r;

    session *s = ath_arena_zalloc(a, sizeof *s, _Alignof(session));
    ath_arena scratch;
    ath_arena_init(&scratch, 0);
    if (!s)
        return ATH_EXIT_FAILURE;
    s->opt = &o;
    s->scratch = &scratch;
    s->tty = isatty(STDOUT_FILENO);
    ath_rng_init(&s->rng, o.seed);
    int rc = ATH_EXIT_FAILURE;
    ath_status st = ath_model_open(&s->m, a, o.dir);
    if (st != ATH_OK) {
        fprintf(stderr, "athanor chat: cannot open %s: %s\n", o.dir, ath_status_str(st));
        ath_arena_free(&scratch);
        return ATH_EXIT_FAILURE;
    }
    const ath_dtype act = o.f32 ? ATH_DTYPE_F32 : ATH_DTYPE_BF16;
    float *lg = ath_arena_alloc(a, (size_t)s->m.vocab_size * sizeof *lg, _Alignof(float));
    if (!lg)
        st = ATH_ERR_NOMEM;
    if (st == ATH_OK)
        st = ath_model_tokenizer(&s->m, o.dir, &s->tok, a);
    if (st == ATH_OK && o.tools_file)
        st = load_tools(a, o.tools_file, &s->tools, &s->n_tools);
    if (st == ATH_OK)
        st = ath_tensor_init(&s->logits, ATH_DTYPE_F32, 2, (int64_t[]){1, s->m.vocab_size}, lg, ATH_TIER_RAM);
    /* the device: the GPU unless asked otherwise or there is none */
    bool gpu = o.device ? !strcmp(o.device, "gpu") : ath_gpu_count() > 0;
    if (st == ATH_OK && gpu) {
        s->engine = ath_arena_zalloc(a, sizeof *s->engine, _Alignof(ath_engine));
        const ath_place_opts po = {.slots = s->opt->experts, .draft = o.mtp};
        st = s->engine ? ath_engine_open_tier(s->engine, &s->m, 0, &po, a) : ATH_ERR_NOMEM;
        if (st != ATH_OK) {
            fprintf(stderr, "athanor chat: no GPU (%s)%s\n", ath_status_str(st),
                    o.device ? "" : "; running on the CPU (--device cpu)");
            s->engine = NULL;
            if (!o.device)
                st = ATH_OK;
        }
    }
    if (st == ATH_OK && s->engine) {
        s->exec = s->engine->exec;
        st = ath_engine_state(s->engine, &s->state, act, o.ctx, a);
        if (st == ATH_OK && !o.prompt)
            st = ath_engine_state(s->engine, &s->snap, act, o.ctx, a);
        if (st == ATH_OK && !o.mtp) {
            ath_decoder *dec = ath_arena_zalloc(a, sizeof *dec, _Alignof(ath_decoder));
            st = dec ? ath_decoder_open(dec, s->engine, &s->state, &o.sampler, a) : ATH_ERR_NOMEM;
            if (st == ATH_OK)
                s->dec = dec;
        }
        if (st == ATH_OK && o.stats)
            fprintf(stderr, "athanor chat: on %s, %.2f GiB of weights\n", s->engine->dev.name,
                    (double)s->engine->weight_bytes / 1073741824.0);
    } else if (st == ATH_OK) {
        st = ath_state_init(&s->state, &s->m, act, o.ctx, a);
        if (st == ATH_OK && !o.prompt)
            st = ath_state_init(&s->snap, &s->m, act, o.ctx, a);
    }
    if (st == ATH_OK)
        st = ath_pool_init(&s->pool, o.threads, a);
    if (st == ATH_OK && !s->engine)
        s->exec = (ath_exec){.pool = &s->pool};
    if (st == ATH_OK && o.mtp) {
        ath_spec *sp = ath_arena_zalloc(a, sizeof *sp, _Alignof(ath_spec));
        st = sp ? ath_spec_open(sp, &s->m, &s->state, &s->exec, &o.sampler, 4096, a) : ATH_ERR_NOMEM;
        if (st == ATH_OK)
            s->spec = sp;
        else if (st == ATH_ERR_UNSUPPORTED)
            fprintf(stderr, "athanor chat: --mtp: the model has no draft head\n");
    }
    if (st == ATH_OK) {
        ath_tokens_init(&s->seen, a);
        s->eos[0] = ath_tokenizer_added(&s->tok, "<|im_end|>");
        s->eos[1] = ath_tokenizer_added(&s->tok, "<|endoftext|>");
        rc = run(s, a);
        ath_pool_free(&s->pool);
    } else {
        fprintf(stderr, "athanor chat: %s\n", ath_status_str(st));
    }
    if (s->dec)
        ath_decoder_close(s->dec);
    if (s->spec) {
        if (o.stats)
            fprintf(stderr, "athanor chat: %llu passes, %llu guesses kept\n", (unsigned long long)s->spec->passes,
                    (unsigned long long)s->spec->accepted);
        ath_spec_close(s->spec);
    }
    if (s->engine)
        ath_engine_close(s->engine);
    ath_model_close(&s->m);
    ath_arena_free(&scratch);
    return rc;
}
