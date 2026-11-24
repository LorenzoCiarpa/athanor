/* weights.c: checkpoint tensors → ath_param, for Qwen3.5 dense and MoE.
 *
 * Binding is driven by the config, not by the checkpoint: for every layer the
 * family knows which tensors must exist and what shape each one has, finds
 * them by name, and fails on the first one that is missing, has a non-float
 * dtype or has a shape that does not follow from config.json. Tensors the
 * family does not ask for are caught afterwards by the registry's coverage
 * check, unless they fall in one of the two ignored groups (vision encoder,
 * MTP head), the same prefixes the reference text model drops.
 *
 * Roles and flags per tensor: docs/contract/qwen35-0.8b.md (Parameter
 * binding) and docs/contract/qwen35-35b-a3b.md for MoE. Params come in
 * forward order: embedding, then per layer input norm, token mixer,
 * post-attention norm, MLP or MoE, then final norm and output head.
 */
#include "qwen35.h"

#include <stdio.h>
#include <string.h>

typedef struct {
    int32_t ndim;
    int64_t d[3];
} shape;

#define S1(a)       ((shape){1, {(a), 0, 0}})
#define S2(a, b)    ((shape){2, {(a), (b), 0}})
#define S3(a, b, c) ((shape){3, {(a), (b), (c)}})

enum {
    RESID  = ATH_P_WRITES_RESIDUAL,
    SENS   = ATH_P_SENSITIVE,
    SHARED = ATH_P_SHARED_EXPERT
};

typedef struct {
    ath_model               *m;
    const ath_qwen35_config *c;
    uint32_t                 cap;   /* capacity of m->params */
} binder;

static void fmt_shape(char *out, size_t cap, int32_t ndim, const int64_t *d)
{
    size_t n = (size_t)snprintf(out, cap, "[");
    for (int32_t i = 0; i < ndim && n < cap; i++)
        n += (size_t)snprintf(out + n, cap - n, i ? ", %lld" : "%lld", (long long)d[i]);
    if (n < cap)
        snprintf(out + n, cap - n, "]");
}

/* Checkpoint name of a tensor: <prefix>layers.<layer>.<suffix> for layer
 * tensors, <prefix><suffix> for global ones. */
static ath_status tensor_name(char *out, size_t cap, const binder *b, int32_t layer, const char *suffix)
{
    int n = layer >= 0 ? snprintf(out, cap, "%slayers.%d.%s", b->c->prefix, layer, suffix)
                       : snprintf(out, cap, "%s%s", b->c->prefix, suffix);
    return n >= 0 && (size_t)n < cap ? ATH_OK : ATH_ERR_RANGE;
}

/* The tensor called name, with a float dtype and exactly the expected shape. */
static ath_status find(const binder *b, const char *name, shape want, const ath_safetensors_entry **out)
{
    const ath_safetensors_entry *e = ath_safetensors_find(&b->m->ckpt, name);
    if (!e) {
        ath_log(ATH_LOG_ERROR, "qwen35: missing tensor %s", name);
        return ATH_ERR_NOT_FOUND;
    }
    ath_dtype dt = e->t.dtype;
    if (dt != ATH_DTYPE_F32 && dt != ATH_DTYPE_F16 && dt != ATH_DTYPE_BF16) {
        ath_log(ATH_LOG_ERROR, "qwen35: tensor %s has dtype %s; expected f32, f16 or bf16", name,
                ath_dtype_name(dt));
        return ATH_ERR_FORMAT;
    }
    bool same = e->t.ndim == want.ndim;
    for (int32_t i = 0; same && i < want.ndim; i++)
        same = e->t.shape[i] == want.d[i];
    if (!same) {
        char got[96], exp[96];
        fmt_shape(got, sizeof got, e->t.ndim, e->t.shape);
        fmt_shape(exp, sizeof exp, want.ndim, want.d);
        ath_log(ATH_LOG_ERROR, "qwen35: tensor %s has shape %s; config.json implies %s", name, got, exp);
        return ATH_ERR_FORMAT;
    }
    *out = e;
    return ATH_OK;
}

static void add(binder *b, const ath_tensor *t, const char *name, ath_role role, int32_t layer,
                int32_t expert, uint32_t flags)
{
    ATH_ASSERT(b->m->n_params < b->cap);
    b->m->params[b->m->n_params++] = (ath_param){
        .t = *t, .role = role, .layer = layer, .expert = expert, .flags = flags, .name = name,
    };
}

/* Bind a whole tensor as one param. */
static ath_status bind(binder *b, int32_t layer, const char *suffix, shape want, ath_role role, uint32_t flags)
{
    char name[256];
    ATH_CHECK(tensor_name(name, sizeof name, b, layer, suffix));
    const ath_safetensors_entry *e;
    ATH_CHECK(find(b, name, want, &e));
    add(b, &e->t, e->name, role, layer, -1, flags);
    return ATH_OK;
}

/* A fused expert tensor [n_experts, rows, cols] becomes one [rows, cols]
 * param per expert: a view of slice e, nothing copied. */
static ath_status bind_experts(binder *b, int32_t layer, const char *suffix, int64_t rows, int64_t cols,
                               ath_role role, uint32_t flags)
{
    char name[256];
    ATH_CHECK(tensor_name(name, sizeof name, b, layer, suffix));
    const ath_safetensors_entry *e;
    ATH_CHECK(find(b, name, S3(b->c->n_experts, rows, cols), &e));
    const int64_t sh[2] = {rows, cols};
    for (int32_t x = 0; x < b->c->n_experts; x++) {
        ath_tensor slice, view;
        ATH_CHECK(ath_tensor_slice(&e->t, 0, x, 1, &slice));
        ATH_CHECK(ath_tensor_reshape(&slice, 2, sh, &view));
        add(b, &view, e->name, role, layer, x, flags);
    }
    return ATH_OK;
}

/* ---- blocks (contract: Gated DeltaNet layer, Full-attention layer, MLP, MoE) -- */

static ath_status bind_gated_deltanet(binder *b, int32_t l)
{
    const ath_qwen35_config *c = b->c;
    const int64_t h = c->hidden, nv = c->lin_v_heads;
    const int64_t key_dim = (int64_t)c->lin_k_heads * c->lin_k_dim;
    const int64_t value_dim = nv * c->lin_v_dim;
    const int64_t conv_dim = 2 * key_dim + value_dim;   /* q, k, v channels */

    ATH_CHECK(bind(b, l, "linear_attn.in_proj_qkv.weight", S2(conv_dim, h), ATH_ROLE_MIX_IN, 0));
    ATH_CHECK(bind(b, l, "linear_attn.in_proj_z.weight", S2(value_dim, h), ATH_ROLE_MIX_IN, 0));
    ATH_CHECK(bind(b, l, "linear_attn.in_proj_b.weight", S2(nv, h), ATH_ROLE_GATE, SENS));
    ATH_CHECK(bind(b, l, "linear_attn.in_proj_a.weight", S2(nv, h), ATH_ROLE_GATE, SENS));
    ATH_CHECK(bind(b, l, "linear_attn.conv1d.weight", S3(conv_dim, 1, c->conv_kernel), ATH_ROLE_CONV, 0));
    ATH_CHECK(bind(b, l, "linear_attn.A_log", S1(nv), ATH_ROLE_OTHER, 0));
    ATH_CHECK(bind(b, l, "linear_attn.dt_bias", S1(nv), ATH_ROLE_OTHER, 0));
    ATH_CHECK(bind(b, l, "linear_attn.norm.weight", S1(c->lin_v_dim), ATH_ROLE_NORM, 0));
    return bind(b, l, "linear_attn.out_proj.weight", S2(h, value_dim), ATH_ROLE_MIX_OUT, RESID);
}

static ath_status bind_attention(binder *b, int32_t l)
{
    const ath_qwen35_config *c = b->c;
    const int64_t h = c->hidden, hd = c->head_dim;
    const int64_t q = (int64_t)c->n_heads * hd, kv = (int64_t)c->n_kv_heads * hd;

    /* q_proj holds query and output gate, interleaved per head: 2 x q rows */
    ATH_CHECK(bind(b, l, "self_attn.q_proj.weight", S2(2 * q, h), ATH_ROLE_MIX_IN, 0));
    ATH_CHECK(bind(b, l, "self_attn.k_proj.weight", S2(kv, h), ATH_ROLE_MIX_IN, 0));
    ATH_CHECK(bind(b, l, "self_attn.v_proj.weight", S2(kv, h), ATH_ROLE_MIX_IN, 0));
    ATH_CHECK(bind(b, l, "self_attn.q_norm.weight", S1(hd), ATH_ROLE_NORM, 0));
    ATH_CHECK(bind(b, l, "self_attn.k_norm.weight", S1(hd), ATH_ROLE_NORM, 0));
    return bind(b, l, "self_attn.o_proj.weight", S2(h, q), ATH_ROLE_MIX_OUT, RESID);
}

static ath_status bind_mlp(binder *b, int32_t l)
{
    const int64_t h = b->c->hidden, in = b->c->intermediate;
    ATH_CHECK(bind(b, l, "mlp.gate_proj.weight", S2(in, h), ATH_ROLE_MLP_IN, 0));
    ATH_CHECK(bind(b, l, "mlp.up_proj.weight", S2(in, h), ATH_ROLE_MLP_IN, 0));
    return bind(b, l, "mlp.down_proj.weight", S2(h, in), ATH_ROLE_MLP_OUT, RESID);
}

static ath_status bind_moe(binder *b, int32_t l)
{
    const ath_qwen35_config *c = b->c;
    const int64_t h = c->hidden, ie = c->expert_inter, is = c->shared_inter;

    ATH_CHECK(bind(b, l, "mlp.gate.weight", S2(c->n_experts, h), ATH_ROLE_ROUTER, SENS));
    /* gate_up_proj[e] rows: gate projection [0, ie), up projection [ie, 2 ie) */
    ATH_CHECK(bind_experts(b, l, "mlp.experts.gate_up_proj", 2 * ie, h, ATH_ROLE_EXPERT_IN, 0));
    ATH_CHECK(bind_experts(b, l, "mlp.experts.down_proj", h, ie, ATH_ROLE_EXPERT_OUT, RESID));
    ATH_CHECK(bind(b, l, "mlp.shared_expert.gate_proj.weight", S2(is, h), ATH_ROLE_MLP_IN, SHARED));
    ATH_CHECK(bind(b, l, "mlp.shared_expert.up_proj.weight", S2(is, h), ATH_ROLE_MLP_IN, SHARED));
    ATH_CHECK(bind(b, l, "mlp.shared_expert.down_proj.weight", S2(h, is), ATH_ROLE_MLP_OUT, SHARED | RESID));
    return bind(b, l, "mlp.shared_expert_gate.weight", S2(1, h), ATH_ROLE_GATE, SHARED | SENS);
}

/* ---- entry point -------------------------------------------------------------- */

static const ath_ignored ignored_groups[] = {
    {.prefix = "model.visual.", .reason = "vision encoder: text path only (docs/MODELS.md)"},
    {.prefix = "mtp.", .reason = "multi-token prediction head: speculative decoding, step 13"},
};

ath_status ath_qwen35_bind(ath_model *m, ath_arena *a, const ath_qwen35_config *c)
{
    /* Upper bound on the param count: 3 global params and, per layer, two
     * norms, at most 9 mixer tensors and the MLP or MoE params. */
    uint64_t per_layer = 2 + 9 + (c->moe ? 5 + 2 * (uint64_t)c->n_experts : 3);
    uint64_t cap = 3 + (uint64_t)c->n_layers * per_layer;
    if (cap > UINT32_MAX)
        return ATH_ERR_RANGE;
    m->params = ath_arena_alloc(a, (size_t)cap * sizeof *m->params, _Alignof(ath_param));
    m->ignored = ath_arena_alloc(a, sizeof ignored_groups, _Alignof(ath_ignored));
    if (!m->params || !m->ignored)
        return ATH_ERR_NOMEM;
    memcpy(m->ignored, ignored_groups, sizeof ignored_groups);
    m->n_ignored = ATH_ARRAY_LEN(ignored_groups);
    m->n_params = 0;

    binder b = {m, c, (uint32_t)cap};
    const int64_t h = c->hidden, v = c->vocab;

    /* With tied embeddings the table is also the output projection, which
     * quantization must treat with care (contract, Parameter binding). */
    ATH_CHECK(bind(&b, -1, "embed_tokens.weight", S2(v, h), ATH_ROLE_EMBED,
                   RESID | (c->tie_embeddings ? SENS : 0)));
    for (int32_t l = 0; l < c->n_layers; l++) {
        ATH_CHECK(bind(&b, l, "input_layernorm.weight", S1(h), ATH_ROLE_NORM, 0));
        if (c->mixer[l] == ATH_QWEN35_ATTENTION)
            ATH_CHECK(bind_attention(&b, l));
        else
            ATH_CHECK(bind_gated_deltanet(&b, l));
        ATH_CHECK(bind(&b, l, "post_attention_layernorm.weight", S1(h), ATH_ROLE_NORM, 0));
        ATH_CHECK(c->moe ? bind_moe(&b, l) : bind_mlp(&b, l));
    }
    ATH_CHECK(bind(&b, -1, "norm.weight", S1(h), ATH_ROLE_NORM, 0));

    /* The output head is a top-level tensor in both checkpoint layouts. */
    const ath_safetensors_entry *head = ath_safetensors_find(&m->ckpt, "lm_head.weight");
    if (c->tie_embeddings) {
        if (head) {
            ath_log(ATH_LOG_ERROR, "qwen35: tie_word_embeddings is true but the checkpoint has lm_head.weight");
            return ATH_ERR_FORMAT;
        }
        return ATH_OK;
    }
    const ath_safetensors_entry *e;
    ATH_CHECK(find(&b, "lm_head.weight", S2(v, h), &e));
    add(&b, &e->t, e->name, ATH_ROLE_LM_HEAD, -1, -1, SENS);
    return ATH_OK;
}
