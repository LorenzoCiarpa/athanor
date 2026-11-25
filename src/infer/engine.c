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

#include <string.h>

ath_status ath_engine_open(ath_engine *e, const ath_model *m, int32_t device, ath_arena *a)
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
    if (st != ATH_OK) {
        ath_engine_close(e);
        return st;
    }
    e->exec = (ath_exec){.dev = &e->dev, .weights = e->weights, .scratch = &e->scratch};
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
