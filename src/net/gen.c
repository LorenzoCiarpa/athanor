/* gen.c: generation for the server. See gen.h.
 *
 * The turn is the one of tools/chat.c (prefill with the snapshot, the first
 * token from the prompt's logits, then the decoder on a GPU), with what a
 * server adds: the prompt runs in chunks with a call to the sink between
 * them, the output goes through stop strings and reaches the sink in whole
 * UTF-8 characters, and a sink that returns false ends the generation.
 */
#include "net/gen.h"

#include <string.h>

#define DEFAULT_CHUNK 4096

ath_status ath_gen_open(ath_gen *g, const char *dir, const ath_gen_opts *o, ath_arena *a)
{
    memset(g, 0, sizeof *g);
    g->a = a;
    g->ctx = o->ctx;
    g->chunk = o->chunk > 0 ? o->chunk : DEFAULT_CHUNK;
    ath_arena_init(&g->scratch, 0);
    ath_rng_init(&g->rng, o->seed);
    ath_status st = ath_model_open(&g->m, a, dir);
    if (st != ATH_OK) {
        ath_arena_free(&g->scratch);
        return st;
    }
    st = ath_model_tokenizer(&g->m, dir, &g->tok, a);
    float *lg = ath_arena_alloc(a, (size_t)g->m.vocab_size * sizeof *lg, _Alignof(float));
    if (st == ATH_OK && !lg)
        st = ATH_ERR_NOMEM;
    if (st == ATH_OK)
        st = ath_tensor_init(&g->logits, ATH_DTYPE_F32, 2, (int64_t[]){1, g->m.vocab_size}, lg, ATH_TIER_RAM);
    if (st == ATH_OK && !(g->generated = ath_arena_zalloc(a, (size_t)g->m.vocab_size, 1)))
        st = ATH_ERR_NOMEM;
    if (st == ATH_OK)
        st = ath_pool_init(&g->pool, o->threads > 0 ? o->threads : ath_pool_default_threads(), a);
    if (st == ATH_OK && !o->cpu && ath_gpu_count() > 0) {
        g->engine = ath_arena_zalloc(a, sizeof *g->engine, _Alignof(ath_engine));
        /* the experts leave room for the state and its snapshot, besides
         * the working memory */
        size_t sb = 0;
        ath_state_bytes(&g->m, ATH_DTYPE_BF16, g->ctx, &sb);
        const ath_place_opts po = {.slots = o->experts, .vram_reserve = ATH_PLACE_RESERVE + 2 * (uint64_t)sb,
                                   .draft = o->mtp};
        ath_log(ATH_LOG_INFO, "serve: a state of %d tokens takes %.2f GiB; two are kept", g->ctx,
                (double)sb / 1073741824.0);
        st = g->engine ? ath_engine_open_tier(g->engine, &g->m, 0, &po, a) : ATH_ERR_NOMEM;
        if (st != ATH_OK) {
            ath_log(ATH_LOG_WARN, "serve: no GPU (%s); running on the CPU", ath_status_str(st));
            g->engine = NULL;
            st = ATH_OK;
        }
    }
    if (st == ATH_OK && g->engine) {
        uint64_t fr0 = 0, tot0 = 0;
        ath_device_mem(&g->engine->dev, &fr0, &tot0);
        ath_log(ATH_LOG_INFO, "serve: engine open; %.2f GiB of device memory free", (double)fr0 / 1073741824.0);
        g->exec = g->engine->exec;
        st = ath_engine_state(g->engine, &g->state, ATH_DTYPE_BF16, g->ctx, a);
        if (st == ATH_OK)
            st = ath_engine_state(g->engine, &g->snap, ATH_DTYPE_BF16, g->ctx, a);
        uint64_t fr = 0, tot = 0;
        ath_device_mem(&g->engine->dev, &fr, &tot);
        ath_log(ATH_LOG_INFO, "serve: states %s; %.2f GiB of device memory free", st == ATH_OK ? "allocated" : "failed",
                (double)fr / 1073741824.0);
    } else if (st == ATH_OK) {
        g->exec = (ath_exec){.pool = &g->pool};
        st = ath_state_init(&g->state, &g->m, ATH_DTYPE_BF16, g->ctx, a);
        if (st == ATH_OK)
            st = ath_state_init(&g->snap, &g->m, ATH_DTYPE_BF16, g->ctx, a);
    }
    if (st == ATH_OK && o->mtp) {
        static const ath_sampler greedy = {0, 0, 1, 0, 0};
        st = ath_spec_open(&g->spec, &g->m, &g->state, &g->exec, &greedy, g->chunk, a);
        g->mtp = st == ATH_OK;
        if (st == ATH_ERR_UNSUPPORTED) {
            ath_log(ATH_LOG_WARN, "serve: the model has no draft head; decoding without one");
            st = ATH_OK;
        }
    }
    if (st == ATH_OK) {
        ath_tokens_init(&g->seen, a);
        ath_tokens_init(&g->snap_ids, a);
        g->eos[0] = ath_tokenizer_added(&g->tok, "<|im_end|>");
        g->eos[1] = ath_tokenizer_added(&g->tok, "<|endoftext|>");
    }
    if (st != ATH_OK)
        ath_gen_close(g);
    return st;
}

void ath_gen_close(ath_gen *g)
{
    if (g->mtp)
        ath_spec_close(&g->spec);
    g->mtp = false;
    if (g->dec_open)
        ath_decoder_close(&g->dec);
    g->dec_open = false;
    if (g->engine)
        ath_engine_close(g->engine);
    g->engine = NULL;
    ath_pool_free(&g->pool);
    if (g->m.family)
        ath_model_close(&g->m);
    g->m.family = NULL;
    ath_arena_free(&g->scratch);
}

static uint32_t chat_opts(const ath_gen_request *r, bool generate)
{
    return (generate ? ATH_CHAT_GENERATE : 0) | (r->think ? ATH_CHAT_THINKING : 0);
}

ath_status ath_gen_tokens(ath_gen *g, const ath_gen_request *r, ath_tokens *out, ath_arena *a)
{
    ath_buf prompt;
    ath_buf_init(&prompt, a);
    ATH_CHECK(ath_model_chat(&g->m, r->msgs, r->n_msgs, r->tools, r->n_tools, chat_opts(r, true), &prompt));
    ath_tokens_init(out, a);
    ath_status st = ath_tokenizer_encode(&g->tok, prompt.data, prompt.len, out, &g->scratch);
    ath_arena_reset(&g->scratch, (ath_arena_pos){0});
    return st;
}

/* ---- the prompt ------------------------------------------------------------------ */

/* Run ids[from .. to) through the model in chunks, logits for the last one
 * if want; the sink's wait between chunks (*cancel when it says stop). */
static ath_status feed(ath_gen *g, const int32_t *ids, size_t from, size_t to, bool want, const ath_gen_sink *sink,
                       bool *cancel)
{
    while (from < to) {
        size_t end = to - from > (size_t)g->chunk ? from + (size_t)g->chunk : to;
        for (size_t i = from; i < end; i++)
            ATH_CHECK(ath_tokens_push(&g->seen, ids[i]));
        ath_batch b = {.tokens = ids + from,
                       .n_tokens = (int32_t)(end - from),
                       .logits = want && end == to ? &g->logits : NULL};
        /* with the draft head, it runs over the prompt too (spec.h) */
        ath_status st = g->mtp ? ath_spec_feed(&g->spec, b.tokens, b.n_tokens, b.logits ? g->logits.data : NULL)
                               : ath_model_forward(&g->m, &b, &g->state, NULL, &g->scratch, &g->exec);
        ath_arena_reset(&g->scratch, (ath_arena_pos){0});
        ATH_CHECK(st);
        from = end;
        if (from < to && sink && sink->wait && !sink->wait(sink->u)) {
            *cancel = true;
            return ATH_OK;
        }
    }
    return ATH_OK;
}

static size_t common_prefix(const ath_tokens *a, const ath_tokens *b)
{
    size_t n = 0;
    while (n < a->n && n < b->n && a->ids[n] == b->ids[n])
        n++;
    return n;
}

static ath_status set_tokens(ath_tokens *dst, const int32_t *ids, size_t n)
{
    dst->n = 0;
    for (size_t i = 0; i < n; i++)
        ATH_CHECK(ath_tokens_push(dst, ids[i]));
    return ATH_OK;
}

/* Bring the state to the start of what is new in ids: on from the state
 * when it holds a prefix of ids, else from the snapshot when that is one,
 * else from the start. The tokens kept in *kept. */
static ath_status reuse(ath_gen *g, const ath_tokens *ids, size_t *kept)
{
    const bool snap_fits = g->snap_n && g->snap_n < ids->n && common_prefix(&g->snap_ids, ids) >= g->snap_n;
    /* requests in a row that could not start from the snapshot */
    g->snap_misses = g->snap_n && !snap_fits ? g->snap_misses + 1 : 0;
    if (g->seen.n < ids->n && common_prefix(&g->seen, ids) == g->seen.n) {
        /* the state goes on */
    } else if (snap_fits) {
        ATH_CHECK(ath_state_copy(&g->state, &g->snap, &g->m));
        ATH_CHECK(set_tokens(&g->seen, g->snap_ids.ids, g->snap_n));
    } else {
        ATH_CHECK(ath_state_reset(&g->state, &g->m));
        g->seen.n = 0;
    }
    *kept = g->seen.n;
    return ATH_OK;
}

/* The rest of the prompt, with a snapshot at stable. A much shorter
 * snapshot (under a quarter) does not replace a longer one: an agent's side
 * request (a title, a summary) must not cost its conversation the next
 * prefill. Unless the snapshot has gone unused twice in a row: another
 * conversation now. */
static ath_status prefill(ath_gen *g, const ath_tokens *ids, size_t stable, const ath_gen_sink *sink, bool *cancel)
{
    size_t start = g->seen.n;
    if (start < stable && 4 * stable < g->snap_n && g->snap_misses < 2)
        stable = start;   /* no new snapshot */
    if (start < stable) {
        ATH_CHECK(feed(g, ids->ids, start, stable, false, sink, cancel));
        if (*cancel)
            return ATH_OK;
        ATH_CHECK(ath_state_copy(&g->snap, &g->state, &g->m));
        ATH_CHECK(set_tokens(&g->snap_ids, ids->ids, stable));
        g->snap_n = stable;
        g->snap_misses = 0;
        if (sink->wait && !sink->wait(sink->u)) {
            *cancel = true;
            return ATH_OK;
        }
    }
    return feed(g, ids->ids, start > stable ? start : stable, ids->n, true, sink, cancel);
}

static bool same_sampler(const ath_sampler *a, const ath_sampler *b)
{
    return a->temperature == b->temperature && a->top_k == b->top_k && a->top_p == b->top_p && a->min_p == b->min_p &&
           a->presence == b->presence;
}

/* The decoder for sampler p: a recorded step includes the choice, so
 * another sampler means another decoder. */
static ath_status decoder_for(ath_gen *g, const ath_sampler *p)
{
    if (!g->engine || g->mtp || (g->dec_open && same_sampler(&g->dec_sampler, p)))
        return ATH_OK;
    if (g->dec_open)
        ath_decoder_close(&g->dec);
    g->dec_open = false;
    ATH_CHECK(ath_decoder_open(&g->dec, g->engine, &g->state, p, g->a));
    g->dec_open = true;
    g->dec_sampler = *p;
    return ATH_OK;
}

/* ---- the output ------------------------------------------------------------------ */

typedef struct {
    const ath_gen_request *r;
    const ath_gen_sink    *sink;
    char                   part[2][4];   /* text, thinking: a character not complete yet */
    int                    n_part[2];
    ath_buf                text;         /* the answer so far, for stop strings          */
    size_t                 sent;         /* of it, the bytes passed on                   */
    size_t                 hold;         /* the longest stop string less one             */
    bool                   any_call;
    bool                   cancel;
    int32_t                stop_hit;     /* the stop string found, or -1                 */
} emitter;

/* The length of the longest prefix of s[0 .. n) that does not end inside a
 * UTF-8 character, looking only at the last few bytes. */
static size_t utf8_whole(const char *s, size_t n)
{
    for (size_t back = 1; back <= 4 && back <= n; back++) {
        unsigned char c = (unsigned char)s[n - back];
        if ((c & 0xC0) == 0x80)
            continue;                                 /* a continuation byte  */
        size_t len = c < 0x80 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 1;
        return back < len ? n - back : n;
    }
    return n;   /* only continuation bytes: not UTF-8, pass them on */
}

/* Answer text after stop strings: a tail that could start a stop string
 * waits for what follows; a stop string ends the answer before it. */
static void out_text(emitter *e, const char *s, size_t n, bool flush)
{
    if (e->cancel || e->stop_hit >= 0)
        return;
    if (n && ath_buf_append(&e->text, s, n) != ATH_OK) {
        e->cancel = true;
        return;
    }
    if (!e->text.data)
        return;
    size_t end = e->text.len;
    for (int32_t i = 0; i < e->r->n_stop; i++) {
        const char *t = e->r->stop[i];
        if (!*t)
            continue;
        const char *hit = memmem(e->text.data + e->sent, e->text.len - e->sent, t, strlen(t));
        if (hit && (size_t)(hit - e->text.data) < end) {
            end = (size_t)(hit - e->text.data);
            e->stop_hit = i;
        }
    }
    size_t upto = end;
    if (e->stop_hit < 0 && !flush)
        upto = utf8_whole(e->text.data, end > e->hold ? end - e->hold : 0);
    if (upto > e->sent) {
        if (!e->sink->text(e->sink->u, e->text.data + e->sent, upto - e->sent))
            e->cancel = true;
        e->sent = upto;
    }
}

/* Text (k 0) or thinking (k 1) bytes: whole characters go on, a broken one
 * at the end waits for the next piece (flush: goes on as it is). */
static void out_chars(emitter *e, int k, const char *s, size_t n, bool flush)
{
    char buf[512];
    do {
        if (e->cancel || e->stop_hit >= 0)
            return;
        size_t take = n < sizeof buf - 4 ? n : sizeof buf - 4;
        memcpy(buf, e->part[k], (size_t)e->n_part[k]);
        if (take)
            memcpy(buf + e->n_part[k], s, take);
        size_t len = (size_t)e->n_part[k] + take;
        s += take, n -= take;
        size_t whole = flush && !n ? len : utf8_whole(buf, len);
        e->n_part[k] = (int)(len - whole);
        memcpy(e->part[k], buf + whole, (size_t)e->n_part[k]);
        if (k == 0)
            out_text(e, buf, whole, flush && !n);
        else if (whole && !e->sink->thinking(e->sink->u, buf, whole))
            e->cancel = true;
    } while (n);
}

static void out_events(emitter *e, const ath_out_event *ev, int32_t n)
{
    const ath_gen_sink *s = e->sink;
    for (int32_t i = 0; i < n && !e->cancel && e->stop_hit < 0; i++) {
        const ath_out_event *x = &ev[i];
        bool ok = true;
        switch (x->kind) {
        case ATH_OUT_TEXT:
            out_chars(e, 0, x->data, x->len, false);
            break;
        case ATH_OUT_THINKING:
            out_chars(e, 1, x->data, x->len, false);
            break;
        case ATH_OUT_TOOL_CALL_BEGIN:
            out_chars(e, 1, NULL, 0, true);
            out_chars(e, 0, NULL, 0, true);
            if (e->cancel || e->stop_hit >= 0)
                return;
            e->any_call = true;
            ok = s->tool_begin(s->u, x->tool_name);
            break;
        case ATH_OUT_TOOL_CALL_ARGS:
            ok = s->tool_args(s->u, x->data, x->len);
            break;
        case ATH_OUT_TOOL_CALL_END:
            ok = s->tool_end(s->u);
            break;
        }
        if (!ok)
            e->cancel = true;
    }
}

/* ---- a request ------------------------------------------------------------------- */

ath_status ath_gen_run(ath_gen *g, const ath_gen_request *r, const ath_gen_sink *sink, ath_gen_result *res,
                       ath_arena *a)
{
    memset(res, 0, sizeof *res);
    res->stop_index = -1;
    ath_buf prompt, stable;
    ath_buf_init(&prompt, a);
    ath_buf_init(&stable, a);
    ATH_CHECK(ath_model_chat(&g->m, r->msgs, r->n_msgs, r->tools, r->n_tools, chat_opts(r, true), &prompt));
    ath_tokens ids, stable_ids;
    ath_tokens_init(&ids, a);
    ath_tokens_init(&stable_ids, a);
    ATH_CHECK(ath_tokenizer_encode(&g->tok, prompt.data, prompt.len, &ids, &g->scratch));
    /* the stable part: the conversation up to its last message, without the
     * assistant's header, when its tokens are a prefix of the prompt's */
    size_t stable_n = 0;
    if (ath_model_chat(&g->m, r->msgs, r->n_msgs, r->tools, r->n_tools, chat_opts(r, false), &stable) == ATH_OK &&
        ath_tokenizer_encode(&g->tok, stable.data, stable.len, &stable_ids, &g->scratch) == ATH_OK &&
        stable_ids.n <= ids.n && !memcmp(stable_ids.ids, ids.ids, stable_ids.n * sizeof *ids.ids))
        stable_n = stable_ids.n;
    ath_arena_reset(&g->scratch, (ath_arena_pos){0});
    res->prompt_tokens = (int32_t)ids.n;
    if ((int64_t)ids.n + 1 > g->ctx) {
        ath_log(ATH_LOG_WARN, "serve: a prompt of %zu tokens does not fit in a context of %d", ids.n, g->ctx);
        return ATH_ERR_RANGE;
    }
    size_t kept;
    ATH_CHECK(reuse(g, &ids, &kept));
    res->cached_tokens = (int32_t)kept;
    bool cancel = sink->begin && !sink->begin(sink->u, res);
    if (!cancel)
        ATH_CHECK(prefill(g, &ids, stable_n, sink, &cancel));
    if (cancel) {
        res->finish = ATH_GEN_CANCELLED;
        return ATH_OK;
    }
    ATH_CHECK(decoder_for(g, &r->sampler));
    if (g->dec_open)
        ATH_CHECK(ath_decoder_forget(&g->dec));
    memset(g->generated, 0, (size_t)g->m.vocab_size);

    ath_parse_state ps = {.opts = chat_opts(r, true), .tools = r->tools, .n_tools = r->n_tools, .a = a};
    emitter e = {.r = r, .sink = sink, .stop_hit = -1};
    for (int32_t i = 0; i < r->n_stop; i++)
        if (strlen(r->stop[i]) > e.hold + 1)
            e.hold = strlen(r->stop[i]) - 1;
    ath_buf_init(&e.text, a);
    ath_out_event ev[64];
    int32_t ne = 0;
    const int32_t room = g->ctx - (int32_t)g->seen.n;
    const int32_t limit = r->max_tokens > 0 && r->max_tokens < room ? r->max_tokens : room;
    int32_t next = 0, n_gen = 0;
    bool ended = false;
    /* with the draft head: the tokens of a speculative step, one or two, and
     * the uniform number a one-token step left unused */
    ath_tokens emitted;
    ath_tokens_init(&emitted, a);
    const size_t prompt_n = g->seen.n;
    int32_t queue[2], nq = 0, qi = 0;
    double carry = -1;
    if (g->mtp)
        g->spec.p = r->sampler;
    if (limit > 0)
        ATH_CHECK(ath_sample_ref(&g->logits, &r->sampler, ath_rng_uniform(&g->rng), &g->scratch, &next));
    for (; n_gen < limit && !e.cancel && e.stop_hit < 0; n_gen++) {
        if (n_gen > 0 && g->mtp) {
            if (qi == nq) {
                if (g->state.n_past + ATH_SPEC_ROWS > g->ctx)
                    break;   /* no room for a pass: the context is full */
                double u[2];
                u[0] = carry >= 0 ? carry : ath_rng_uniform(&g->rng);
                u[1] = ath_rng_uniform(&g->rng);
                int32_t k = 0;
                ATH_CHECK(ath_spec_step(&g->spec, u, queue, &k));
                nq = k, qi = 0;
                carry = k == 1 ? u[1] : -1;
            }
            next = queue[qi++];
        } else if (n_gen > 0 && !g->engine) {   /* on the GPU the decoder applies the penalty */
            if (r->sampler.presence != 0)
                ATH_CHECK(ath_presence_ref(&g->logits, g->generated, r->sampler.presence));
            ATH_CHECK(ath_sample_ref(&g->logits, &r->sampler, ath_rng_uniform(&g->rng), &g->scratch, &next));
        }
        g->generated[next] = 1;
        ath_arena_reset(&g->scratch, (ath_arena_pos){0});
        if (next == g->eos[0] || next == g->eos[1]) {
            ended = true;
            break;
        }
        ATH_CHECK(ath_tokens_push(&emitted, next));
        size_t len;
        const char *piece = ath_tokenizer_token(&g->tok, next, &len);
        ATH_CHECK(ath_model_parse(&g->m, &ps, piece, len, ev, 64, &ne));
        out_events(&e, ev, ne);
        if (g->mtp) {
            if (n_gen == 0 && n_gen + 1 < limit)
                ATH_CHECK(ath_spec_begin(&g->spec, next));
        } else if (n_gen + 1 < limit && g->engine) {
            ATH_CHECK(ath_tokens_push(&g->seen, next));
            ATH_CHECK(ath_decoder_step(&g->dec, next, ath_rng_uniform(&g->rng), &next));
        } else if (n_gen + 1 < limit) {
            ATH_CHECK(feed(g, &next, 0, 1, true, NULL, &cancel));
        }
    }
    /* what the state holds: the prompt and the answer's tokens up to n_past
     * (a speculative step leaves its last token for the next pass) */
    for (size_t i = g->seen.n - prompt_n; g->mtp && g->seen.n < (size_t)g->state.n_past && i < emitted.n; i++)
        ATH_CHECK(ath_tokens_push(&g->seen, emitted.ids[i]));
    if (!e.cancel && e.stop_hit < 0) {
        ATH_CHECK(ath_model_parse(&g->m, &ps, NULL, 0, ev, 64, &ne));
        out_events(&e, ev, ne);
        out_chars(&e, 1, NULL, 0, true);
        out_chars(&e, 0, NULL, 0, true);
    }
    res->output_tokens = n_gen + (ended ? 1 : 0);
    res->stop_index = e.stop_hit;
    res->finish = e.cancel           ? ATH_GEN_CANCELLED
                  : e.stop_hit >= 0  ? ATH_GEN_STOP
                  : e.any_call       ? ATH_GEN_TOOL_CALLS
                  : ended            ? ATH_GEN_END
                                     : ATH_GEN_LENGTH;
    return ATH_OK;
}
