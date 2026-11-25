/* state.c: per-sequence state and the forward entry point. See model.h.
 *
 * A model lists its state as specs (kind, layer, element counts); this file
 * turns them into buffers for one sequence, in host memory or on a device
 * (step 8), and is the only way into a family's forward pass, so that every
 * family gets the same argument checks and the same bookkeeping of
 * positions and scratch memory on both.
 */
#include "model/model.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* Buffers start on a cache line, which also satisfies every element type. */
#define STATE_ALIGN 64

static const char *const kind_names[] = {"kv", "conv", "recurrent"};
_Static_assert(ATH_ARRAY_LEN(kind_names) == ATH_STATE_KIND_COUNT, "one name per state kind");

const char *ath_state_kind_name(ath_state_kind k)
{
    return (unsigned)k < ATH_STATE_KIND_COUNT ? kind_names[k] : "?";
}

static bool act_ok(ath_dtype act)
{
    return act == ATH_DTYPE_BF16 || act == ATH_DTYPE_F32;
}

/* Bytes of one buffer: the recurrent state is fp32 (it accumulates every
 * token of the sequence), the others hold activations. */
static ath_status spec_bytes(const ath_state_spec *sp, ath_dtype act, int32_t capacity, size_t *out)
{
    const ath_dtype dt = sp->kind == ATH_STATE_RECURRENT ? ATH_DTYPE_F32 : act;
    if (sp->fixed < 0 || sp->per_token < 0 || capacity < 0)
        return ATH_ERR_RANGE;
    const uint64_t es = ath_dtype_size(dt), cap = (uint64_t)capacity;
    const uint64_t pt = (uint64_t)sp->per_token, fx = (uint64_t)sp->fixed;
    if (fx > SIZE_MAX / es || (pt && cap > (SIZE_MAX / es - fx) / pt))
        return ATH_ERR_RANGE;
    *out = (size_t)((fx + pt * cap) * es);
    return ATH_OK;
}

ath_status ath_state_bytes(const ath_model *m, ath_dtype act, int32_t capacity, size_t *bytes)
{
    if (!m || !bytes || !act_ok(act) || capacity < 0)
        return ATH_ERR_INVALID_ARG;
    size_t total = 0;
    for (uint32_t i = 0; i < m->n_states; i++) {
        size_t b;
        ATH_CHECK(spec_bytes(&m->states[i], act, capacity, &b));
        b = (b + STATE_ALIGN - 1) / STATE_ALIGN * STATE_ALIGN;   /* as allocated */
        if (b > SIZE_MAX - total)
            return ATH_ERR_RANGE;
        total += b;
    }
    *bytes = total;
    return ATH_OK;
}

/* The buffers of one sequence in host memory (mem NULL) or in device
 * memory (mem set). */
static ath_status init(ath_state *s, const ath_model *m, ath_dtype act, int32_t capacity, const ath_device *d,
                       ath_dev_arena *mem, ath_arena *a)
{
    if (!s || !m || !a)
        return ATH_ERR_INVALID_ARG;
    if (!act_ok(act)) {
        ath_log(ATH_LOG_ERROR, "state: activation dtype %s; expected bf16 or f32", ath_dtype_name(act));
        return ATH_ERR_TYPE;
    }
    if (capacity < 1) {
        ath_log(ATH_LOG_ERROR, "state: capacity %d; expected at least 1 token", capacity);
        return ATH_ERR_INVALID_ARG;
    }
    *s = (ath_state){.act = act, .capacity = capacity, .dev = d};
    s->buf = ath_arena_alloc(a, (m->n_states ? m->n_states : 1) * sizeof *s->buf, _Alignof(void *));
    if (!s->buf)
        return ATH_ERR_NOMEM;
    /* In device memory the buffers are carved from one allocation: one by
     * one, a KV buffer of a long context would leave most of an arena
     * block unused each time (128 MiB buffers in 256 MiB blocks). */
    size_t total = 0;
    for (int pass = mem ? 0 : 1; pass < 2; pass++) {
        char *base = NULL;
        if (mem && pass == 1 && !(base = ath_dev_arena_alloc(mem, total ? total : 1, 0)))
            return ATH_ERR_NOMEM;
        size_t off = 0;
        for (uint32_t i = 0; i < m->n_states; i++) {
            size_t b;
            ath_status st = spec_bytes(&m->states[i], act, capacity, &b);
            if (st != ATH_OK) {
                ath_log(ATH_LOG_ERROR, "state: %s state of layer %d does not fit in memory for %d tokens",
                        ath_state_kind_name(m->states[i].kind), m->states[i].layer, capacity);
                return st;
            }
            b = b ? b : 1;
            if (!mem) {
                if (!(s->buf[i] = ath_arena_alloc(a, b, STATE_ALIGN)))
                    return ATH_ERR_NOMEM;
                continue;
            }
            const size_t sz = (b + ATH_DEV_ALIGN - 1) / ATH_DEV_ALIGN * ATH_DEV_ALIGN;
            if (pass == 0 && sz > SIZE_MAX - total)
                return ATH_ERR_RANGE;
            if (pass == 0)
                total += sz;
            else
                s->buf[i] = base + off;
            off += sz;
        }
    }
    return ath_state_reset(s, m);
}

ath_status ath_state_init(ath_state *s, const ath_model *m, ath_dtype act, int32_t capacity,
                          ath_arena *a)
{
    return init(s, m, act, capacity, NULL, NULL, a);
}

ath_status ath_state_init_gpu(ath_state *s, const ath_model *m, ath_dtype act, int32_t capacity,
                              const ath_device *d, ath_dev_arena *mem, ath_arena *a)
{
    if (!d || !mem)
        return ATH_ERR_INVALID_ARG;
    return init(s, m, act, capacity, d, mem, a);
}

ath_status ath_state_reset(ath_state *s, const ath_model *m)
{
    if (!s || !m || !s->buf)
        return ATH_ERR_INVALID_ARG;
    s->n_past = 0;
    for (uint32_t i = 0; i < m->n_states; i++) {
        const ath_state_spec *sp = &m->states[i];
        if (sp->kind == ATH_STATE_KV)
            continue;
        size_t b = 0;
        ATH_CHECK(spec_bytes(sp, s->act, s->capacity, &b));
        if (s->dev)
            ATH_CHECK(ath_memset(s->dev, s->buf[i], 0, b, ATH_STREAM_COMPUTE));
        else
            memset(s->buf[i], 0, b);
    }
    return ATH_OK;
}

/* ---- forward ------------------------------------------------------------------ */

ath_status ath_state_copy(ath_state *dst, const ath_state *src, const ath_model *m)
{
    if (!dst || !src || !m || dst->act != src->act || dst->capacity != src->capacity || !dst->buf || !src->buf ||
        dst->dev != src->dev)
        return ATH_ERR_INVALID_ARG;
    for (uint32_t i = 0; i < m->n_states; i++) {
        size_t b;
        ATH_CHECK(spec_bytes(&m->states[i], src->act, src->capacity, &b));
        if (src->dev)
            ATH_CHECK(ath_copy(src->dev, dst->buf[i], src->buf[i], b, ATH_STREAM_COMPUTE));
        else
            memcpy(dst->buf[i], src->buf[i], b);
    }
    dst->n_past = src->n_past;
    return ATH_OK;
}

static ath_status bad(ath_status st, const char *fmt, ...) __attribute__((format(printf, 2, 3)));

static ath_status bad(ath_status st, const char *fmt, ...)
{
    char msg[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    ath_log(ATH_LOG_ERROR, "forward: %s", msg);
    return st;
}

static ath_status check_logits(const ath_model *m, const ath_batch *b)
{
    const ath_tensor *l = b->logits;
    if (!l)
        return ATH_OK;
    if (!l->data)
        return bad(ATH_ERR_INVALID_ARG, "logits: no data");
    if (l->dtype != ATH_DTYPE_F32)
        return bad(ATH_ERR_TYPE, "logits: dtype %s; expected f32", ath_dtype_name(l->dtype));
    const bool on_dev = b->flags & ATH_BATCH_DEVICE;
    if (on_dev ? l->tier != ATH_TIER_VRAM : l->tier != ATH_TIER_RAM && l->tier != ATH_TIER_PINNED)
        return bad(ATH_ERR_UNSUPPORTED, "logits: tensor in %s; expected %s memory", ath_tier_name(l->tier),
                   on_dev ? "device" : "host");
    if (l->ndim != 2 || l->shape[1] != m->vocab_size || (l->shape[0] != b->n_tokens && l->shape[0] != 1) ||
        !ath_tensor_is_contiguous(l))
        return bad(ATH_ERR_INVALID_ARG, "logits: expected a contiguous [%d, %d] or [1, %d] tensor",
                   b->n_tokens, m->vocab_size, m->vocab_size);
    return ATH_OK;
}

ath_status ath_model_forward(const ath_model *m, const ath_batch *b, ath_state *s, const ath_hooks *h,
                             ath_arena *scratch, const ath_exec *x)
{
    if (!m || !m->family || !b || !s || !scratch)
        return bad(ATH_ERR_INVALID_ARG, "missing model, batch, state or scratch arena");
    if (!m->family->forward)
        return bad(ATH_ERR_UNSUPPORTED, "family %s has no forward pass", m->family->name);
    if (!s->buf || !act_ok(s->act))
        return bad(ATH_ERR_INVALID_ARG, "state not initialized (ath_state_init)");
    if (b->n_tokens < 1 || !b->tokens)
        return bad(ATH_ERR_INVALID_ARG, "empty batch");
    if (s->n_past < 0 || b->n_tokens > s->capacity - s->n_past)
        return bad(ATH_ERR_RANGE, "%d tokens after %d do not fit in a state of %d", b->n_tokens, s->n_past,
                   s->capacity);
    if (b->flags & ~ATH_BATCH_DEVICE)
        return bad(ATH_ERR_INVALID_ARG, "unknown batch flags 0x%x", b->flags);
    if ((b->flags & ATH_BATCH_DEVICE) ? !b->pos || !x || !x->dev : b->pos != NULL)
        return bad(ATH_ERR_INVALID_ARG, "positions are given with a device batch (ATH_BATCH_DEVICE), on a GPU, "
                                        "and only then");
    for (int32_t i = 0; !(b->flags & ATH_BATCH_DEVICE) && i < b->n_tokens; i++)
        if (b->tokens[i] < 0 || b->tokens[i] >= m->vocab_size)
            return bad(ATH_ERR_RANGE, "token %d at index %d is outside the vocabulary [0, %d)", b->tokens[i], i,
                       m->vocab_size);
    ATH_CHECK(check_logits(m, b));
    const ath_exec cpu = {.pool = NULL};
    if (!x)
        x = &cpu;
    if (x->dev != s->dev)
        return bad(ATH_ERR_INVALID_ARG, "the state is %s, the pass would run %s", s->dev ? "on a device" : "in host memory",
                   x->dev ? "on a device" : "on the CPU");
    if (x->dev && (!x->weights || !x->scratch))
        return bad(ATH_ERR_INVALID_ARG, "a GPU pass needs the weights on the device and a device scratch arena");

    ath_arena_pos mark = ath_arena_mark(scratch);
    ath_dev_arena_pos dmark = x->dev ? ath_dev_arena_mark(x->scratch) : (ath_dev_arena_pos){0};
    ath_status st = m->family->forward(m, b, s, h, scratch, x);
    if (x->dev)
        ath_dev_arena_reset(x->scratch, dmark);   /* later work on the stream reuses it in order */
    ath_arena_reset(scratch, mark);
    if (st == ATH_OK)
        s->n_past += b->n_tokens;
    return st;
}
