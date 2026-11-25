/* engine.c: a model on a GPU. See engine.h.
 *
 * Every param the forward pass reads gets a copy in device memory with the
 * same dtype, shape and bytes; the copies are indexed like m->params, which
 * is how a family's forward pass finds them (model.h, ath_exec.weights).
 * The copies are made from the memory-mapped checkpoint (ordinary host
 * memory: the runtime stages each copy, so the pages are read once, in
 * order) on the host-to-device stream, and the engine waits for them before
 * it returns.
 */
#include "infer/engine.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

ath_status ath_engine_open(ath_engine *e, const ath_model *m, int32_t device, ath_arena *a)
{
    return ath_engine_open_tier(e, m, device, NULL, a);
}

ath_status ath_engine_open_tier(ath_engine *e, const ath_model *m, int32_t device, const ath_place_opts *tier,
                                ath_arena *a)
{
    if (!e || !m || !a)
        return ATH_ERR_INVALID_ARG;
    memset(e, 0, sizeof *e);
    e->m = m;
    ATH_CHECK(ath_device_open(&e->dev, device));
    ath_dev_arena_init(&e->weights_mem, &e->dev, a, 0);
    ath_dev_arena_init(&e->state_mem, &e->dev, a, 0);
    ath_dev_arena_init(&e->scratch, &e->dev, a, 0);
    ath_status st = ATH_OK;
    e->weights = ath_arena_zalloc(a, (m->n_params ? m->n_params : 1) * sizeof *e->weights, _Alignof(ath_tensor));
    if (!e->weights)
        st = ATH_ERR_NOMEM;
    for (uint32_t i = 0; st == ATH_OK && i < m->n_params; i++) {
        const ath_param *p = &m->params[i];
        if (p->flags & ATH_P_UNUSED)
            continue;
        if ((p->flags & ATH_P_DRAFT) && !(tier && tier->draft))
            continue;                  /* the draft head on request (step 13) */
        if (m->n_experts > 0 && (p->role == ATH_ROLE_EXPERT_IN || p->role == ATH_ROLE_EXPERT_OUT))
            continue;                  /* the tier places them, below */
        if (!ath_tensor_is_contiguous(&p->t) || (p->t.tier != ATH_TIER_RAM && p->t.tier != ATH_TIER_PINNED)) {
            ath_log(ATH_LOG_ERROR, "engine: param %s is not a contiguous host tensor", p->name);
            st = ATH_ERR_UNSUPPORTED;
            break;
        }
        const size_t bytes = ath_tensor_nbytes(&p->t);
        ath_tensor *t = &e->weights[i];
        *t = p->t;
        t->tier = ATH_TIER_VRAM;
        t->data = ath_dev_arena_alloc(&e->weights_mem, bytes ? bytes : 1, 0);
        if (!t->data) {
            ath_log(ATH_LOG_ERROR, "engine: %s does not fit on %s (%.2f GiB of weights so far)", p->name,
                    e->dev.name, (double)e->weight_bytes / 1073741824.0);
            st = ATH_ERR_NOMEM;
            break;
        }
        st = ath_copy(&e->dev, t->data, p->t.data, bytes, ATH_STREAM_H2D);
        e->weight_bytes += bytes;
        e->n_copied++;
    }
    if (st == ATH_OK)
        st = ath_stream_sync(&e->dev, ATH_STREAM_H2D);
    if (st == ATH_OK && m->n_experts > 0) {
        e->tier = ath_arena_zalloc(a, sizeof *e->tier, _Alignof(ath_placement));
        st = e->tier ? ath_place_open(e->tier, m, &e->dev, tier, a) : ATH_ERR_NOMEM;
        if (st != ATH_OK)
            e->tier = NULL;
    }
    if (st != ATH_OK) {
        ath_engine_close(e);
        return st;
    }
    e->draft = tier && tier->draft && m->n_draft_layers > 0;
    e->exec = (ath_exec){.dev = &e->dev, .weights = e->weights, .scratch = &e->scratch};
    if (e->tier) {
        e->exec.experts = e->tier->layers;
        e->exec.moe_ws = e->tier->ws;
    }
    ath_log(ATH_LOG_INFO, "engine: %u params, %.2f GiB on %s", e->n_copied, (double)e->weight_bytes / 1073741824.0,
            e->dev.name);
    return ATH_OK;
}

void ath_engine_close(ath_engine *e)
{
    if (!e)
        return;
    if (e->dev.stream[0])
        ath_device_sync(&e->dev);
    ath_place_close(e->tier);
    ath_dev_arena_free(&e->scratch);
    ath_dev_arena_free(&e->state_mem);
    ath_dev_arena_free(&e->weights_mem);
    ath_device_close(&e->dev);
    memset(e, 0, sizeof *e);
}

ath_status ath_engine_state(ath_engine *e, ath_state *s, ath_dtype act, int32_t capacity, ath_arena *a)
{
    if (!e || !e->m)
        return ATH_ERR_INVALID_ARG;
    return ath_state_init_gpu(s, e->m, act, capacity, &e->dev, &e->state_mem, a);
}

/* ---- decoding ------------------------------------------------------------------- */

ath_status ath_decoder_open(ath_decoder *dc, ath_engine *e, ath_state *s, const ath_sampler *p, ath_arena *a)
{
    if (!dc || !e || !e->m || !s || !p || !a || s->dev != &e->dev)
        return ATH_ERR_INVALID_ARG;
    if (!(p->temperature >= 0) || p->top_k < 0 || !(p->top_p > 0 && p->top_p <= 1) || !(p->min_p >= 0 && p->min_p <= 1) ||
        !isfinite(p->presence))
        return ATH_ERR_INVALID_ARG;
    memset(dc, 0, sizeof *dc);
    dc->e = e;
    dc->s = s;
    dc->p = *p;
    dc->gpu_sampling = ath_sample_gpu_fits(p);
    ath_arena_init(&dc->scratch, 0);
    const int64_t V = e->m->vocab_size;
    const size_t lbytes = (size_t)V * sizeof(float);
    void *io = NULL, *hio = NULL, *lg = NULL, *hlg = NULL;
    ath_status st = ath_dev_alloc(&e->dev, sizeof(ath_decode_io), &io);
    if (st == ATH_OK)
        st = ath_pinned_alloc(&e->dev, sizeof(ath_decode_io), &hio);
    if (st == ATH_OK)
        st = ath_dev_alloc(&e->dev, lbytes, &lg);
    if (st == ATH_OK && !dc->gpu_sampling)
        st = ath_pinned_alloc(&e->dev, lbytes, &hlg);
    void *seen = NULL;
    if (st == ATH_OK && (st = ath_dev_alloc(&e->dev, (size_t)V, &seen)) == ATH_OK) {
        dc->seen = seen;
        st = ath_memset(&e->dev, seen, 0, (size_t)V, ATH_STREAM_COMPUTE);
    }
    dc->io = io, dc->hio = hio;
    if (st == ATH_OK)
        st = ath_tensor_init(&dc->logits, ATH_DTYPE_F32, 2, (int64_t[]){1, V}, lg, ATH_TIER_VRAM);
    else if (lg)
        ath_dev_free(&e->dev, lg);
    if (st == ATH_OK && hlg)
        st = ath_tensor_init(&dc->hlogits, ATH_DTYPE_F32, 2, (int64_t[]){1, V}, hlg, ATH_TIER_PINNED);
    else if (hlg)
        ath_pinned_free(&e->dev, hlg);
    if (st != ATH_OK) {
        ath_decoder_close(dc);
        return st;
    }
    memset(dc->hio, 0, sizeof *dc->hio);
    return ATH_OK;
}

/* The step's work on the compute stream: its inputs from pinned memory, the
 * forward pass as a device batch, the choice, the result to pinned memory.
 * Nothing in it waits, so it can be recorded. */
static ath_status queue_step(ath_decoder *dc)
{
    ath_engine *e = dc->e;
    ATH_CHECK(ath_copy(&e->dev, dc->io, dc->hio, offsetof(ath_decode_io, next), ATH_STREAM_COMPUTE));
    ath_batch b = {.tokens = &dc->io->token, .n_tokens = 1, .logits = &dc->logits, .pos = &dc->io->pos,
                   .flags = ATH_BATCH_DEVICE};
    ATH_CHECK(ath_model_forward(e->m, &b, dc->s, NULL, &dc->scratch, &e->exec));
    if (dc->p.presence != 0)
        ATH_CHECK(ath_presence_gpu(&e->dev, &dc->logits, dc->seen, &dc->io->token, dc->p.presence));
    if (!dc->gpu_sampling)
        return ath_copy(&e->dev, dc->hlogits.data, dc->logits.data, ath_tensor_nbytes(&dc->logits),
                        ATH_STREAM_COMPUTE);
    ATH_CHECK(ath_sample_gpu(&e->dev, &dc->logits, &dc->p, &dc->io->u, e->exec.scratch, &dc->io->next));
    return ath_copy(&e->dev, &dc->hio->next, &dc->io->next, sizeof(int32_t), ATH_STREAM_COMPUTE);
}

ath_status ath_decoder_step(ath_decoder *dc, int32_t token, double u, int32_t *next)
{
    if (!dc || !dc->e || !next || !(u >= 0 && u < 1))
        return ATH_ERR_INVALID_ARG;
    ath_engine *e = dc->e;
    ath_state *s = dc->s;
    if (token < 0 || token >= e->m->vocab_size)
        return ATH_ERR_RANGE;
    if (s->n_past < 0 || s->n_past >= s->capacity)
        return ATH_ERR_RANGE;
    /* the previous step has finished (its result was read), so the pinned
     * inputs can be written */
    dc->hio->token = token;
    dc->hio->pos = s->n_past;
    dc->hio->u = u;
    if (dc->steps == 1 && !dc->g.exec) {
        /* the second step: record it, then run the recording. ath_model_forward
         * advances n_past while recording, for the work the launch does. */
        ATH_CHECK(ath_graph_begin(&e->dev, ATH_STREAM_COMPUTE));
        ath_status st = queue_step(dc);
        ath_status end = ath_graph_end(&e->dev, ATH_STREAM_COMPUTE, &dc->g);
        if (st != ATH_OK || end != ATH_OK) {
            ath_graph_destroy(&e->dev, &dc->g);
            s->n_past = dc->hio->pos;
            return st != ATH_OK ? st : end;
        }
        ATH_CHECK(ath_graph_launch(&e->dev, &dc->g, ATH_STREAM_COMPUTE));
    } else if (dc->g.exec) {
        /* a replay: the work of ath_model_forward without the call, so this
         * keeps the state's count as the call would */
        ATH_CHECK(ath_graph_launch(&e->dev, &dc->g, ATH_STREAM_COMPUTE));
        s->n_past++;
    } else {
        ATH_CHECK(queue_step(dc));    /* the first step, directly */
    }
    ATH_CHECK(ath_stream_sync(&e->dev, ATH_STREAM_COMPUTE));
    dc->steps++;
    if (e->tier)
        ATH_CHECK(ath_place_update(e->tier));   /* experts move between steps, never during one */
    if (dc->gpu_sampling) {
        *next = dc->hio->next;
        return ATH_OK;
    }
    return ath_sample_ref(&dc->hlogits, &dc->p, u, &dc->scratch, next);
}

ath_status ath_decoder_forget(ath_decoder *dc)
{
    if (!dc || !dc->e || !dc->seen)
        return ATH_ERR_INVALID_ARG;
    return ath_memset(&dc->e->dev, dc->seen, 0, (size_t)dc->e->m->vocab_size, ATH_STREAM_COMPUTE);
}

void ath_decoder_close(ath_decoder *dc)
{
    if (!dc || !dc->e)
        return;
    ath_engine *e = dc->e;
    ath_stream_sync(&e->dev, ATH_STREAM_COMPUTE);
    ath_graph_destroy(&e->dev, &dc->g);
    if (dc->io)
        ath_dev_free(&e->dev, dc->io);
    if (dc->hio)
        ath_pinned_free(&e->dev, dc->hio);
    if (dc->logits.data)
        ath_dev_free(&e->dev, dc->logits.data);
    if (dc->hlogits.data)
        ath_pinned_free(&e->dev, dc->hlogits.data);
    if (dc->seen)
        ath_dev_free(&e->dev, dc->seen);
    ath_arena_free(&dc->scratch);
    memset(dc, 0, sizeof *dc);
}
