/* tier.c: where the routed experts of a MoE model live. See tier.h.
 *
 * Every expert's three matrices are copied once into one block of pinned
 * host memory, regrouped for the CPU (Q8_0_R16) when they are Q8_0, so that
 * the CPU reads them with one load per 16 values and the copy of an expert
 * into a device slot is a plain asynchronous copy of the same bytes (the
 * GPU's kernels read Q8_0_R16 too). The slots are one block of device
 * memory, [layer][slot] experts of expert_bytes each.
 *
 * When every expert fits in the slots there are no host copies: the tables
 * have no host side, and the passes never call the CPU.
 */
#include "infer/tier.h"

#include <string.h>

typedef struct {
    ath_placement          *t;
    const ath_param  **p;        /* [n_layers][3][n]: the experts' params, NULL for dense layers */
    ath_dtype          dt;       /* the experts' dtype in the checkpoint                          */
    size_t             mb;       /* one matrix in the host and slot layout, padded               */
    ath_status         status;
} fill_job;

static ath_status copy_matrix(const ath_param *p, ath_dtype dt, void *dst)
{
    if (dt != ATH_DTYPE_Q8_0) {
        memcpy(dst, p->t.data, ath_tensor_nbytes(&p->t));
        return ATH_OK;
    }
    ath_tensor out;
    ATH_CHECK(ath_tensor_init(&out, ATH_DTYPE_Q8_0_R16, 2, p->t.shape, dst, ATH_TIER_PINNED));
    return ath_q8_0_repack(&p->t, &out);
}

/* Task k of n: the host copies of a share of the layers. */
static void fill_task(void *ctx, int32_t k, int32_t n)
{
    fill_job *j = ctx;
    ath_placement *t = j->t;
    const int64_t E = t->n_experts;
    for (int32_t l = (int32_t)((int64_t)t->n_layers * k / n); l < (int32_t)((int64_t)t->n_layers * (k + 1) / n); l++) {
        if (!j->p[(size_t)l * 3 * E])
            continue;
        for (int64_t i = 0; i < 3 * E; i++) {
            void *dst = (char *)t->host_mem + ((size_t)l * 3 * E + i) * j->mb;
            ath_status s = copy_matrix(j->p[(size_t)l * 3 * E + i], j->dt, dst);
            if (s != ATH_OK)
                j->status = s;
            t->host_tab[(size_t)l * 3 * E + i] = dst;
        }
    }
}

/* Device address of matrix m of the expert in slot s of layer l. */
static void *slot_addr(const ath_placement *t, int32_t l, int32_t s, int m, size_t mb)
{
    return (char *)t->slot_mem + (((size_t)l * t->slots + s) * 3 + m) * mb;
}

ath_status ath_place_open(ath_placement *t, const ath_model *m, const ath_device *d, const ath_place_opts *opts,
                         ath_arena *a)
{
    if (!t || !m || !d || !a)
        return ATH_ERR_INVALID_ARG;
    memset(t, 0, sizeof *t);
    if (m->n_experts < 1)
        return ATH_ERR_UNSUPPORTED;
    t->m = m;
    t->d = d;
    t->o = opts ? *opts : (ath_place_opts){.slots = -1};
    if (!t->o.vram_reserve)
        t->o.vram_reserve = ATH_PLACE_RESERVE;
    if (!t->o.cpu_tokens)
        t->o.cpu_tokens = 512;
    if (!t->o.period)
        t->o.period = 4;
    if (!t->o.moves)
        t->o.moves = 2;
    t->n_layers = m->n_layers;
    t->n_experts = m->n_experts;
    const int64_t E = t->n_experts, L = t->n_layers;

    /* the experts' params by layer, matrix and expert: per expert, in param
     * order, gate, up and down (model.h, ath_exec.experts) */
    const ath_param **p = ath_arena_zalloc(a, (size_t)(L * 3 * E) * sizeof *p, _Alignof(const ath_param *));
    int8_t *seen = ath_arena_zalloc(a, (size_t)(L * E) + 1, 1);
    if (!p || !seen)
        return ATH_ERR_NOMEM;
    ath_dtype dt = ATH_DTYPE_COUNT;
    for (uint32_t i = 0; i < m->n_params; i++) {
        const ath_param *q = &m->params[i];
        if (q->role != ATH_ROLE_EXPERT_IN && q->role != ATH_ROLE_EXPERT_OUT)
            continue;
        if (q->layer < 0 || q->layer >= L || q->expert < 0 || q->expert >= E || q->t.ndim != 2) {
            ath_log(ATH_LOG_ERROR, "tier: expert param %s has no layer or expert", q->name);
            return ATH_ERR_INVALID_ARG;
        }
        int8_t *k = &seen[q->layer * E + q->expert];
        if (*k >= 3 || (*k < 2) != (q->role == ATH_ROLE_EXPERT_IN) || (dt != ATH_DTYPE_COUNT && q->t.dtype != dt)) {
            ath_log(ATH_LOG_ERROR, "tier: expert %d of layer %d is not gate, up, down in one dtype", q->expert, q->layer);
            return ATH_ERR_INVALID_ARG;
        }
        dt = q->t.dtype;
        p[((size_t)q->layer * 3 + (size_t)*k) * E + (size_t)q->expert] = q;
        (*k)++;
    }
    int32_t moe_layers = 0;
    for (int64_t l = 0; l < L; l++) {
        const ath_param *g = p[(size_t)l * 3 * E];
        if (!g)
            continue;
        moe_layers++;
        t->inter = g->t.shape[0];
        t->hidden = g->t.shape[1];
        for (int64_t e = 0; e < E; e++)
            if (seen[l * E + e] != 3) {
                ath_log(ATH_LOG_ERROR, "tier: layer %lld lacks expert %lld", (long long)l, (long long)e);
                return ATH_ERR_INVALID_ARG;
            }
    }
    if (moe_layers == 0 || dt == ATH_DTYPE_COUNT)
        return ATH_ERR_UNSUPPORTED;
    /* the host and slot layout: Q8_0 regrouped, other dtypes as they are */
    const ath_dtype lay = dt == ATH_DTYPE_Q8_0 ? ATH_DTYPE_Q8_0_R16 : dt;
    if (lay == ATH_DTYPE_Q8_0_R16 && (t->inter % ATH_Q8_0_R16_BLOCK || t->hidden % ATH_Q8_0_R16_BLOCK)) {
        ath_log(ATH_LOG_ERROR, "tier: Q8_0 experts of %lld x %lld are not whole groups of 512", (long long)t->inter,
                (long long)t->hidden);
        return ATH_ERR_UNSUPPORTED;
    }
    const size_t mb = (ath_dtype_bytes(lay, t->inter * t->hidden) + ATH_DEV_ALIGN - 1) / ATH_DEV_ALIGN * ATH_DEV_ALIGN;
    t->expert_bytes = 3 * mb;

    /* how many slots per layer */
    uint64_t free_b = 0, total_b = 0;
    ATH_CHECK(ath_device_mem(d, &free_b, &total_b));
    const uint64_t avail = free_b > t->o.vram_reserve ? free_b - t->o.vram_reserve : 0;
    int64_t slots = (int64_t)(avail / ((uint64_t)moe_layers * t->expert_bytes));
    if (t->o.slots >= 0 && t->o.slots < slots)
        slots = t->o.slots;
    slots = slots > E ? E : slots;
    /* Only Q8_0 experts have one arithmetic order on the CPU and the GPU
     * (src/ops/q8_0.h): experts in another dtype go to the device whole,
     * or not at all, so that placement never changes a result (invariant 7) */
    if (slots < E && dt != ATH_DTYPE_Q8_0) {
        ath_log(ATH_LOG_ERROR, "tier: %s experts must all be in device memory (%lld of %lld per layer fit); only "
                "Q8_0 experts can be split between the GPU and the CPU", ath_dtype_name(dt), (long long)slots,
                (long long)E);
        return ATH_ERR_UNSUPPORTED;
    }
    t->slots = (int32_t)slots;
    const bool all = slots == E;

    /* tables */
    t->layers = ath_arena_zalloc(a, (size_t)L * sizeof *t->layers, _Alignof(ath_moe_experts));
    t->dev_host = ath_arena_zalloc(a, (size_t)(L * 3 * E) * sizeof *t->dev_host, _Alignof(void *));
    t->host_tab = ath_arena_zalloc(a, (size_t)(L * 3 * E) * sizeof *t->host_tab, _Alignof(void *));
    t->on_dev = ath_arena_zalloc(a, (size_t)(L * E) + 1, 1);
    t->uses = ath_arena_zalloc(a, (size_t)(L * E) * sizeof *t->uses + 1, _Alignof(uint32_t));
    t->slot_of = ath_arena_alloc(a, (size_t)(L * E) * sizeof *t->slot_of + 1, _Alignof(int32_t));
    t->in_slot = ath_arena_alloc(a, (size_t)(L * (slots ? slots : 1)) * sizeof *t->in_slot, _Alignof(int32_t));
    if (!t->layers || !t->dev_host || !t->host_tab || !t->on_dev || !t->uses || !t->slot_of || !t->in_slot)
        return ATH_ERR_NOMEM;
    for (int64_t i = 0; i < L * E; i++)
        t->slot_of[i] = -1;
    for (int64_t i = 0; i < L * slots; i++)
        t->in_slot[i] = -1;

    ath_status st = ath_pool_init(&t->pool, ath_pool_default_threads(), a);
    /* host copies, unless every expert has a slot */
    if (st == ATH_OK && !all) {
        t->host_bytes = (size_t)(L * 3 * E) * mb;
        st = ath_pinned_alloc(d, t->host_bytes, &t->host_mem);
        if (st == ATH_OK) {
            fill_job j = {.t = t, .p = p, .dt = dt, .mb = mb};
            ath_pool_run(&t->pool, ath_pool_threads(&t->pool), fill_task, &j);
            st = j.status;
        }
    }
    /* slots, filled with the first experts of each layer */
    if (st == ATH_OK && slots > 0)
        st = ath_dev_alloc(d, (size_t)(L * slots) * t->expert_bytes, &t->slot_mem);
    if (st == ATH_OK)
        st = ath_dev_alloc(d, (size_t)(L * 3 * E) * sizeof(void *), &t->dev_tab);
    for (int32_t l = 0; st == ATH_OK && l < L; l++) {
        if (!p[(size_t)l * 3 * E])
            continue;
        for (int32_t s = 0; st == ATH_OK && s < slots; s++) {
            const int32_t e = s;
            for (int mx = 0; st == ATH_OK && mx < 3; mx++) {
                void *dst = slot_addr(t, l, s, mx, mb);
                const size_t i = ((size_t)l * 3 + (size_t)mx) * E + (size_t)e;
                if (all) {
                    /* no host copies: from the checkpoint, regrouped on the way when Q8_0 */
                    if (dt == ATH_DTYPE_Q8_0) {
                        if (!t->staging)
                            st = ath_pinned_alloc(d, mb, &t->staging);
                        if (st == ATH_OK)
                            st = copy_matrix(p[i], dt, t->staging);
                        if (st == ATH_OK)
                            st = ath_copy(d, dst, t->staging, ath_tensor_nbytes(&p[i]->t), ATH_STREAM_H2D);
                        if (st == ATH_OK)
                            st = ath_stream_sync(d, ATH_STREAM_H2D);
                    } else {
                        st = ath_copy(d, dst, p[i]->t.data, ath_tensor_nbytes(&p[i]->t), ATH_STREAM_H2D);
                    }
                } else {
                    st = ath_copy(d, dst, t->host_tab[((size_t)l * 3 * E) + (size_t)mx * E + (size_t)e],
                                  ath_tensor_nbytes(&p[i]->t), ATH_STREAM_H2D);
                }
                t->dev_host[i] = dst;
            }
            t->slot_of[l * E + e] = s;
            t->in_slot[l * slots + s] = e;
            t->on_dev[l * E + e] = 1;
        }
        t->layers[l] = (ath_moe_experts){
            .n_experts = E, .inter = t->inter, .hidden = t->hidden, .dev_dtype = lay,
            .dev_tab = (const void *const *)((void **)t->dev_tab + (size_t)l * 3 * E), .host_dtype = lay,
            .host_tab = all ? NULL : t->host_tab + (size_t)l * 3 * E, .on_dev = t->on_dev + l * E,
            .uses = t->uses + l * E,
        };
    }
    if (st == ATH_OK)
        st = ath_copy(d, t->dev_tab, t->dev_host, (size_t)(L * 3 * E) * sizeof(void *), ATH_STREAM_H2D);
    if (st == ATH_OK)
        st = ath_stream_sync(d, ATH_STREAM_H2D);
    if (st == ATH_OK)
        st = ath_moe_ws_create(&t->ws, d, &t->pool, t->o.cpu_tokens, m->n_experts_active, t->hidden, t->inter, a);
    if (st != ATH_OK) {
        ath_place_close(t);
        return st;
    }
    ath_log(ATH_LOG_INFO, "tier: %d of %lld experts per layer in device memory (%.2f GiB), %.2f GiB in host memory",
            t->slots, (long long)E, (double)((uint64_t)moe_layers * (uint64_t)slots * t->expert_bytes) / 1073741824.0,
            (double)t->host_bytes / 1073741824.0);
    return ATH_OK;
}

ath_status ath_place_update(ath_placement *t)
{
    if (!t || !t->m)
        return ATH_ERR_INVALID_ARG;
    if (t->slots == t->n_experts || t->slots == 0 || ++t->passes % t->o.period)
        return ATH_OK;
    const int64_t E = t->n_experts;
    const size_t mb = t->expert_bytes / 3;
    bool changed = false;
    for (int32_t l = 0; l < t->n_layers; l++) {
        if (!t->layers[l].host_tab)
            continue;
        uint32_t *u = t->uses + (size_t)l * E;
        for (int32_t mv = 0; mv < t->o.moves; mv++) {
            /* the most chosen expert away from the device, the least chosen in a slot */
            int32_t best = -1, worst = -1;
            for (int32_t e = 0; e < E; e++) {
                if (t->slot_of[l * E + e] < 0) {
                    if (best < 0 || u[e] > u[best])
                        best = e;
                } else if (worst < 0 || u[e] < u[worst]) {
                    worst = e;
                }
            }
            if (best < 0 || worst < 0 || u[best] <= u[worst] + 1)
                break;
            const int32_t s = t->slot_of[l * E + worst];
            for (int mx = 0; mx < 3; mx++) {
                const size_t iw = ((size_t)l * 3 + (size_t)mx) * E + (size_t)worst;
                const size_t ib = ((size_t)l * 3 + (size_t)mx) * E + (size_t)best;
                t->dev_host[iw] = NULL;
                ATH_CHECK(ath_copy(t->d, slot_addr(t, l, s, mx, mb), t->host_tab[ib], mb, ATH_STREAM_H2D));
                t->dev_host[ib] = slot_addr(t, l, s, mx, mb);
            }
            t->on_dev[l * E + worst] = 0;
            t->on_dev[l * E + best] = 1;
            t->slot_of[l * E + worst] = -1;
            t->slot_of[l * E + best] = s;
            t->in_slot[l * t->slots + s] = best;
            t->moved++;
            changed = true;
        }
        for (int32_t e = 0; e < E; e++)   /* age the counts: recent choices weigh more */
            u[e] -= u[e] / 4;
    }
    if (!changed)
        return ATH_OK;
    /* the table after the copies, before the next pass */
    ATH_CHECK(ath_copy(t->d, t->dev_tab, t->dev_host, (size_t)(t->n_layers * 3 * E) * sizeof(void *),
                       ATH_STREAM_H2D));
    return ath_stream_sync(t->d, ATH_STREAM_H2D);
}

void ath_place_close(ath_placement *t)
{
    if (!t || !t->d)
        return;
    ath_device_sync(t->d);
    ath_moe_ws_free(t->ws);
    ath_dev_free(t->d, t->slot_mem);
    ath_dev_free(t->d, t->dev_tab);
    ath_pinned_free(t->d, t->host_mem);
    ath_pinned_free(t->d, t->staging);
    ath_pool_free(&t->pool);
    memset(t, 0, sizeof *t);
}
