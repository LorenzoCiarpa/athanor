/* weights.c: checkpoint tensors → ath_param, for Qwen3.5 dense and MoE.
 *
 * Binding is driven by the config, not by the checkpoint: for every layer the
 * family knows which tensors must exist and what shape each one has, finds
 * them by name, and fails on the first one that is missing, has a non-float
 * dtype or has a shape that does not follow from config.json. The tensors
 * of the two ignored groups (vision encoder, MTP head: the prefixes the
 * reference text model drops) are bound too, each as one param flagged
 * ATH_P_UNUSED, after all the others. Any other tensor is caught afterwards
 * by the registry's coverage check.
 *
 * Roles and flags per tensor: docs/contract/qwen35-0.8b.md (Parameter
 * binding) and docs/contract/qwen35-35b-a3b.md for MoE. Params come in
 * forward order: embedding, then per layer input norm, token mixer,
 * post-attention norm, MLP or MoE, then final norm and output head.
 *
 * Two naming schemes (step 12): a safetensors checkpoint has Hugging Face's
 * names; a GGUF file has the names llama.cpp's converter gives them
 * (docs/contract/qwen36-35b-a3b.md, GGUF), with the same shapes except the
 * convolution kernel (squeezed to [C, K]) and the shared expert's gate
 * ([H]), which are bound as views of the reference's shapes. Some
 * converters merge projections that share an input (qkv with z, beta with
 * alpha); their halves are bound as row slices, nothing copied. Matmul and
 * embedding weights may be Q8_0 in a GGUF file.
 */
#include "qwen35.h"

#include <stddef.h>
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
    bool                     gguf;  /* GGUF names (step 12)  */
} binder;

/* The name in the scheme of the checkpoint: Hugging Face's or GGUF's. */
#define NM(b, hf, gg) ((b)->gguf ? (gg) : (hf))

/* Where a bound param is recorded for the forward pass; NULL to skip. */
typedef const ath_param **slot;

static void fmt_shape(char *out, size_t cap, int32_t ndim, const int64_t *d)
{
    size_t n = (size_t)snprintf(out, cap, "[");
    for (int32_t i = 0; i < ndim && n < cap; i++)
        n += (size_t)snprintf(out + n, cap - n, i ? ", %lld" : "%lld", (long long)d[i]);
    if (n < cap)
        snprintf(out + n, cap - n, "]");
}

/* Checkpoint name of a tensor: <prefix>layers.<layer>.<suffix> for layer
 * tensors, <prefix><suffix> for global ones; in a GGUF file
 * blk.<layer>.<suffix> and <suffix>. */
static ath_status tensor_name(char *out, size_t cap, const binder *b, int32_t layer, const char *suffix)
{
    int n;
    if (b->gguf)
        n = layer >= 0 ? snprintf(out, cap, "blk.%d.%s", layer, suffix) : snprintf(out, cap, "%s", suffix);
    else
        n = layer >= 0 ? snprintf(out, cap, "%slayers.%d.%s", b->c->prefix, layer, suffix)
                       : snprintf(out, cap, "%s%s", b->c->prefix, suffix);
    return n >= 0 && (size_t)n < cap ? ATH_OK : ATH_ERR_RANGE;
}

/* Roles whose params feed a matmul or the embedding lookup: Q8_0 is allowed
 * there (step 12), the ops take it. */
static bool takes_q8_0(ath_role r)
{
    return r == ATH_ROLE_EMBED || r == ATH_ROLE_LM_HEAD || r == ATH_ROLE_MIX_IN || r == ATH_ROLE_MIX_OUT ||
           r == ATH_ROLE_MLP_IN || r == ATH_ROLE_MLP_OUT || r == ATH_ROLE_ROUTER || r == ATH_ROLE_EXPERT_IN ||
           r == ATH_ROLE_EXPERT_OUT || r == ATH_ROLE_GATE;
}

/* The tensor called name, with a float dtype (or Q8_0 where the role takes
 * it) and exactly the expected shape. */
static ath_status find(const binder *b, const char *name, shape want, ath_role role, const ath_ckpt_tensor **out)
{
    const ath_ckpt_tensor *e = ath_ckpt_find(&b->m->ckpt, name);
    if (!e) {
        ath_log(ATH_LOG_ERROR, "qwen35: missing tensor %s", name);
        return ATH_ERR_NOT_FOUND;
    }
    ath_dtype dt = e->t.dtype;
    if (dt != ATH_DTYPE_F32 && dt != ATH_DTYPE_F16 && dt != ATH_DTYPE_BF16 &&
        !(dt == ATH_DTYPE_Q8_0 && takes_q8_0(role))) {
        ath_log(ATH_LOG_ERROR, "qwen35: tensor %s has dtype %s; expected f32, f16 or bf16%s", name,
                ath_dtype_name(dt), takes_q8_0(role) ? " or q8_0" : "");
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

/* m->params never moves (allocated once with its capacity), so the returned
 * pointer stays valid for the life of the model. */
static const ath_param *add(binder *b, const ath_tensor *t, const char *name, ath_role role, int32_t layer,
                            int32_t expert, uint32_t flags)
{
    ATH_ASSERT(b->m->n_params < b->cap);
    ath_param *p = &b->m->params[b->m->n_params++];
    *p = (ath_param){
        .t = *t, .role = role, .layer = layer, .expert = expert, .flags = flags, .name = name,
    };
    return p;
}

/* Bind a whole tensor as one param and record it in *out. */
static ath_status bind(binder *b, int32_t layer, const char *suffix, shape want, ath_role role, uint32_t flags,
                       slot out)
{
    char name[256];
    ATH_CHECK(tensor_name(name, sizeof name, b, layer, suffix));
    const ath_ckpt_tensor *e;
    ATH_CHECK(find(b, name, want, role, &e));
    const ath_param *p = add(b, &e->t, e->name, role, layer, -1, flags);
    if (out)
        *out = p;
    return ATH_OK;
}

/* Bind a tensor of shape `stored` as a param of shape `as` (the same
 * elements): GGUF's squeezed convolution kernel and shared-expert gate. */
static ath_status bind_as(binder *b, int32_t layer, const char *suffix, shape stored, shape as, ath_role role,
                          uint32_t flags, slot out)
{
    char name[256];
    ATH_CHECK(tensor_name(name, sizeof name, b, layer, suffix));
    const ath_ckpt_tensor *e;
    ATH_CHECK(find(b, name, stored, role, &e));
    ath_tensor v;
    ATH_CHECK(ath_tensor_reshape(&e->t, as.ndim, as.d, &v));
    const ath_param *p = add(b, &v, e->name, role, layer, -1, flags);
    if (out)
        *out = p;
    return ATH_OK;
}

/* A tensor that holds two projections one after the other, [r0 + r1, cols],
 * bound as two params: rows [0, r0) and [r0, r0 + r1). */
static ath_status bind_split(binder *b, int32_t layer, const char *suffix, int64_t r0, int64_t r1, int64_t cols,
                             ath_role role0, uint32_t flags0, slot out0, ath_role role1, uint32_t flags1, slot out1)
{
    char name[256];
    ATH_CHECK(tensor_name(name, sizeof name, b, layer, suffix));
    const ath_ckpt_tensor *e;
    ATH_CHECK(find(b, name, S2(r0 + r1, cols), role0, &e));
    ath_tensor v0, v1;
    ATH_CHECK(ath_tensor_slice(&e->t, 0, 0, r0, &v0));
    ATH_CHECK(ath_tensor_slice(&e->t, 0, r0, r1, &v1));
    *out0 = add(b, &v0, e->name, role0, layer, -1, flags0);
    *out1 = add(b, &v1, e->name, role1, layer, -1, flags1);
    return ATH_OK;
}

/* A fused expert tensor [n_experts, r0 + r1, cols] becomes, per expert, a
 * [r0, cols] view of slice e (rows [0, r0)) and, with r1 > 0, a [r1, cols]
 * view (rows [r0, r0 + r1)): the gate and up halves of Hugging Face's
 * gate_up_proj, or GGUF's separate tensors with r1 = 0. Nothing copied. */
static ath_status bind_experts(binder *b, int32_t layer, const char *suffix, int64_t r0, int64_t r1, int64_t cols,
                               ath_role role, uint32_t flags, const ath_param **out0, const ath_param **out1)
{
    char name[256];
    ATH_CHECK(tensor_name(name, sizeof name, b, layer, suffix));
    const ath_ckpt_tensor *e;
    ATH_CHECK(find(b, name, S3(b->c->n_experts, r0 + r1, cols), role, &e));
    for (int32_t x = 0; x < b->c->n_experts; x++) {
        ath_tensor slice, view, part;
        ATH_CHECK(ath_tensor_slice(&e->t, 0, x, 1, &slice));
        ATH_CHECK(ath_tensor_reshape(&slice, 2, (const int64_t[]){r0 + r1, cols}, &view));
        ATH_CHECK(ath_tensor_slice(&view, 0, 0, r0, &part));
        out0[x] = add(b, &part, e->name, role, layer, x, flags);
        if (r1 > 0) {
            ATH_CHECK(ath_tensor_slice(&view, 0, r0, r1, &part));
            out1[x] = add(b, &part, e->name, role, layer, x, flags);
        }
    }
    return ATH_OK;
}

/* Whether the checkpoint has the tensor <layer>.<suffix>. */
static bool has(const binder *b, int32_t layer, const char *suffix)
{
    char name[256];
    return tensor_name(name, sizeof name, b, layer, suffix) == ATH_OK && ath_ckpt_find(&b->m->ckpt, name);
}

/* ---- blocks (contract: Gated DeltaNet layer, Full-attention layer, MLP, MoE) -- */

static ath_status bind_gated_deltanet(binder *b, int32_t l, ath_qwen35_layer *L)
{
    const ath_qwen35_config *c = b->c;
    const int64_t h = c->hidden, nv = c->lin_v_heads;
    const int64_t key_dim = (int64_t)c->lin_k_heads * c->lin_k_dim;
    const int64_t value_dim = nv * c->lin_v_dim;
    const int64_t conv_dim = 2 * key_dim + value_dim;   /* q, k, v channels */

    if (b->gguf && has(b, l, "attn_qkv_gate.weight"))   /* qkv and z merged (llama.cpp's dense converter) */
        ATH_CHECK(bind_split(b, l, "attn_qkv_gate.weight", conv_dim, value_dim, h, ATH_ROLE_MIX_IN, 0, &L->qkv_proj,
                             ATH_ROLE_MIX_IN, 0, &L->z_proj));
    else {
        ATH_CHECK(bind(b, l, NM(b, "linear_attn.in_proj_qkv.weight", "attn_qkv.weight"), S2(conv_dim, h),
                       ATH_ROLE_MIX_IN, 0, &L->qkv_proj));
        ATH_CHECK(bind(b, l, NM(b, "linear_attn.in_proj_z.weight", "attn_gate.weight"), S2(value_dim, h),
                       ATH_ROLE_MIX_IN, 0, &L->z_proj));
    }
    if (b->gguf && has(b, l, "ssm_beta_alpha.weight"))  /* beta then alpha, merged */
        ATH_CHECK(bind_split(b, l, "ssm_beta_alpha.weight", nv, nv, h, ATH_ROLE_GATE, SENS, &L->b_proj, ATH_ROLE_GATE,
                             SENS, &L->a_proj));
    else {
        ATH_CHECK(bind(b, l, NM(b, "linear_attn.in_proj_b.weight", "ssm_beta.weight"), S2(nv, h), ATH_ROLE_GATE, SENS,
                       &L->b_proj));
        ATH_CHECK(bind(b, l, NM(b, "linear_attn.in_proj_a.weight", "ssm_alpha.weight"), S2(nv, h), ATH_ROLE_GATE, SENS,
                       &L->a_proj));
    }
    if (b->gguf)
        ATH_CHECK(bind_as(b, l, "ssm_conv1d.weight", S2(conv_dim, c->conv_kernel), S3(conv_dim, 1, c->conv_kernel),
                          ATH_ROLE_CONV, 0, &L->conv));
    else
        ATH_CHECK(bind(b, l, "linear_attn.conv1d.weight", S3(conv_dim, 1, c->conv_kernel), ATH_ROLE_CONV, 0,
                       &L->conv));
    /* GGUF: ssm_a holds A = −exp(A_log) (the gates take it as it is) */
    ATH_CHECK(bind(b, l, NM(b, "linear_attn.A_log", "ssm_a"), S1(nv), ATH_ROLE_OTHER, 0, &L->a_log));
    ATH_CHECK(bind(b, l, NM(b, "linear_attn.dt_bias", "ssm_dt.bias"), S1(nv), ATH_ROLE_OTHER, 0, &L->dt_bias));
    ATH_CHECK(bind(b, l, NM(b, "linear_attn.norm.weight", "ssm_norm.weight"), S1(c->lin_v_dim), ATH_ROLE_NORM, 0,
                   &L->norm));
    return bind(b, l, NM(b, "linear_attn.out_proj.weight", "ssm_out.weight"), S2(h, value_dim), ATH_ROLE_MIX_OUT,
                RESID, &L->out_proj);
}

static ath_status bind_attention(binder *b, int32_t l, ath_qwen35_layer *L)
{
    const ath_qwen35_config *c = b->c;
    const int64_t h = c->hidden, hd = c->head_dim;
    const int64_t q = (int64_t)c->n_heads * hd, kv = (int64_t)c->n_kv_heads * hd;

    /* q_proj holds query and output gate, interleaved per head: 2 x q rows */
    ATH_CHECK(bind(b, l, NM(b, "self_attn.q_proj.weight", "attn_q.weight"), S2(2 * q, h), ATH_ROLE_MIX_IN, 0,
                   &L->q_proj));
    ATH_CHECK(bind(b, l, NM(b, "self_attn.k_proj.weight", "attn_k.weight"), S2(kv, h), ATH_ROLE_MIX_IN, 0, &L->k_proj));
    ATH_CHECK(bind(b, l, NM(b, "self_attn.v_proj.weight", "attn_v.weight"), S2(kv, h), ATH_ROLE_MIX_IN, 0, &L->v_proj));
    ATH_CHECK(bind(b, l, NM(b, "self_attn.q_norm.weight", "attn_q_norm.weight"), S1(hd), ATH_ROLE_NORM, 0, &L->q_norm));
    ATH_CHECK(bind(b, l, NM(b, "self_attn.k_norm.weight", "attn_k_norm.weight"), S1(hd), ATH_ROLE_NORM, 0, &L->k_norm));
    return bind(b, l, NM(b, "self_attn.o_proj.weight", "attn_output.weight"), S2(h, q), ATH_ROLE_MIX_OUT, RESID,
                &L->o_proj);
}

static ath_status bind_mlp(binder *b, int32_t l, ath_qwen35_layer *L)
{
    const int64_t h = b->c->hidden, in = b->c->intermediate;
    ATH_CHECK(bind(b, l, NM(b, "mlp.gate_proj.weight", "ffn_gate.weight"), S2(in, h), ATH_ROLE_MLP_IN, 0,
                   &L->gate_proj));
    ATH_CHECK(bind(b, l, NM(b, "mlp.up_proj.weight", "ffn_up.weight"), S2(in, h), ATH_ROLE_MLP_IN, 0, &L->up_proj));
    return bind(b, l, NM(b, "mlp.down_proj.weight", "ffn_down.weight"), S2(h, in), ATH_ROLE_MLP_OUT, RESID,
                &L->down_proj);
}

static ath_status bind_moe(binder *b, int32_t l, ath_qwen35_layer *L, ath_arena *a)
{
    const ath_qwen35_config *c = b->c;
    const int64_t h = c->hidden, ie = c->expert_inter, is = c->shared_inter, E = c->n_experts;

    L->exp_gate = ath_arena_alloc(a, (size_t)E * sizeof *L->exp_gate, _Alignof(const ath_param *));
    L->exp_up = ath_arena_alloc(a, (size_t)E * sizeof *L->exp_up, _Alignof(const ath_param *));
    L->exp_down = ath_arena_alloc(a, (size_t)E * sizeof *L->exp_down, _Alignof(const ath_param *));
    if (!L->exp_gate || !L->exp_up || !L->exp_down)
        return ATH_ERR_NOMEM;
    ATH_CHECK(bind(b, l, NM(b, "mlp.gate.weight", "ffn_gate_inp.weight"), S2(c->n_experts, h), ATH_ROLE_ROUTER, SENS,
                   &L->router));
    if (b->gguf) {
        ATH_CHECK(bind_experts(b, l, "ffn_gate_exps.weight", ie, 0, h, ATH_ROLE_EXPERT_IN, 0, L->exp_gate, NULL));
        ATH_CHECK(bind_experts(b, l, "ffn_up_exps.weight", ie, 0, h, ATH_ROLE_EXPERT_IN, 0, L->exp_up, NULL));
        ATH_CHECK(bind_experts(b, l, "ffn_down_exps.weight", h, 0, ie, ATH_ROLE_EXPERT_OUT, RESID, L->exp_down, NULL));
    } else {
        /* gate_up_proj[e] rows: gate projection [0, ie), up projection [ie, 2 ie) */
        ATH_CHECK(bind_experts(b, l, "mlp.experts.gate_up_proj", ie, ie, h, ATH_ROLE_EXPERT_IN, 0, L->exp_gate,
                               L->exp_up));
        ATH_CHECK(bind_experts(b, l, "mlp.experts.down_proj", h, 0, ie, ATH_ROLE_EXPERT_OUT, RESID, L->exp_down, NULL));
    }
    ATH_CHECK(bind(b, l, NM(b, "mlp.shared_expert.gate_proj.weight", "ffn_gate_shexp.weight"), S2(is, h),
                   ATH_ROLE_MLP_IN, SHARED, &L->sh_gate));
    ATH_CHECK(bind(b, l, NM(b, "mlp.shared_expert.up_proj.weight", "ffn_up_shexp.weight"), S2(is, h), ATH_ROLE_MLP_IN,
                   SHARED, &L->sh_up));
    ATH_CHECK(bind(b, l, NM(b, "mlp.shared_expert.down_proj.weight", "ffn_down_shexp.weight"), S2(h, is),
                   ATH_ROLE_MLP_OUT, SHARED | RESID, &L->sh_down));
    if (b->gguf)
        return bind_as(b, l, "ffn_gate_inp_shexp.weight", S1(h), S2(1, h), ATH_ROLE_GATE, SHARED | SENS,
                       &L->sh_gate_w);
    return bind(b, l, "mlp.shared_expert_gate.weight", S2(1, h), ATH_ROLE_GATE, SHARED | SENS, &L->sh_gate_w);
}

/* ---- entry point -------------------------------------------------------------- */

static const ath_ignored ignored_groups[] = {
    {.prefix = "model.visual.", .reason = "vision encoder: text path only (docs/MODELS.md)"},
    {.prefix = "mtp.", .reason = "multi-token prediction head: speculative decoding, step 13"},
};

/* In a GGUF file the MTP head is one more block, blk.<n_layers>, and the
 * vision encoder is another file (mmproj). */
#define MTP_REASON "multi-token prediction head (blk.<n_layers>): speculative decoding, step 13"

static uint32_t count_prefix(const ath_ckpt *st, const char *prefix)
{
    uint32_t n = 0;
    for (uint32_t i = 0; i < st->n_tensors; i++)
        n += strncmp(st->tensors[i].name, prefix, strlen(prefix)) == 0;
    return n;
}

/* Every tensor of an ignored group becomes one param flagged ATH_P_UNUSED:
 * the whole tensor, role other, no layer. Shapes and dtypes are not checked:
 * the text path never reads these tensors, and an export copies them as they
 * are. */
static void bind_unused(binder *b, const char *prefix)
{
    const ath_ckpt *st = &b->m->ckpt;
    for (uint32_t i = 0; i < st->n_tensors; i++) {
        const ath_ckpt_tensor *e = &st->tensors[i];
        if (strncmp(e->name, prefix, strlen(prefix)) == 0)
            add(b, &e->t, e->name, ATH_ROLE_OTHER, -1, -1, ATH_P_UNUSED);
    }
}

/* A GGUF file says what it holds in its metadata: llama.cpp's architecture
 * names for this family, and dimensions that must agree with config.json. */
static ath_status check_gguf(const ath_gguf *g, const ath_qwen35_config *c)
{
    const ath_gguf_kv *arch = ath_gguf_get(g, "general.architecture");
    const char *want = c->moe ? "qwen35moe" : "qwen35";
    if (!arch || arch->type != ATH_GGUF_STRING || strcmp(arch->v.str.ptr, want) != 0) {
        ath_log(ATH_LOG_ERROR, "qwen35: GGUF general.architecture is %s; config.json implies %s",
                arch && arch->type == ATH_GGUF_STRING ? arch->v.str.ptr : "(missing)", want);
        return ATH_ERR_FORMAT;
    }
    static const struct { const char *key; size_t off; } dims[] = {
        {"embedding_length", offsetof(ath_qwen35_config, hidden)},
        {"attention.head_count", offsetof(ath_qwen35_config, n_heads)},
        {"attention.head_count_kv", offsetof(ath_qwen35_config, n_kv_heads)},
        {"ssm.group_count", offsetof(ath_qwen35_config, lin_k_heads)},
        {"ssm.time_step_rank", offsetof(ath_qwen35_config, lin_v_heads)},
        {"expert_count", offsetof(ath_qwen35_config, n_experts)},
        {"expert_used_count", offsetof(ath_qwen35_config, n_experts_active)},
    };
    for (size_t i = 0; i < ATH_ARRAY_LEN(dims); i++) {
        char key[96];
        snprintf(key, sizeof key, "%s.%s", want, dims[i].key);
        const ath_gguf_kv *kv = ath_gguf_get(g, key);
        if (!kv)
            continue;
        const int32_t v = *(const int32_t *)((const char *)c + dims[i].off);
        if (kv->type != ATH_GGUF_U32 || kv->v.u != (uint64_t)v) {
            ath_log(ATH_LOG_ERROR, "qwen35: GGUF %s disagrees with config.json (%d)", key, v);
            return ATH_ERR_FORMAT;
        }
    }
    return ATH_OK;
}

ath_status ath_qwen35_bind(ath_model *m, ath_arena *a, ath_qwen35_model *q)
{
    const ath_qwen35_config *c = &q->c;
    q->gguf = m->ckpt.format == ATH_CKPT_GGUF;
    q->norm_offset = q->gguf ? 0.0f : 1.0f;
    if (q->gguf)
        ATH_CHECK(check_gguf(&m->ckpt.gguf, c));

    /* the ignored groups: the static table, or for GGUF the MTP block */
    const ath_ignored *groups = ignored_groups;
    size_t n_groups = ATH_ARRAY_LEN(ignored_groups);
    ath_ignored gguf_group;
    if (q->gguf) {
        char prefix[32];
        snprintf(prefix, sizeof prefix, "blk.%d.", c->n_layers);
        gguf_group = (ath_ignored){.prefix = ath_arena_strndup(a, prefix, strlen(prefix)), .reason = MTP_REASON};
        if (!gguf_group.prefix)
            return ATH_ERR_NOMEM;
        groups = &gguf_group;
        n_groups = 1;
    }

    /* Upper bound on the param count: 3 global params and, per layer, two
     * norms, at most 9 mixer tensors and the MLP or MoE params; then one
     * param per tensor of the ignored groups. */
    uint64_t per_layer = 2 + 9 + (c->moe ? 5 + 3 * (uint64_t)c->n_experts : 3);
    uint64_t cap = 3 + (uint64_t)c->n_layers * per_layer;
    for (size_t g = 0; g < n_groups; g++)
        cap += count_prefix(&m->ckpt, groups[g].prefix);
    if (cap > UINT32_MAX)
        return ATH_ERR_RANGE;
    m->params = ath_arena_alloc(a, (size_t)cap * sizeof *m->params, _Alignof(ath_param));
    m->ignored = ath_arena_alloc(a, n_groups * sizeof *m->ignored, _Alignof(ath_ignored));
    if (!m->params || !m->ignored)
        return ATH_ERR_NOMEM;
    memcpy(m->ignored, groups, n_groups * sizeof *groups);
    m->n_ignored = (uint32_t)n_groups;
    m->n_params = 0;

    binder b = {m, c, (uint32_t)cap, q->gguf};
    const int64_t h = c->hidden, v = c->vocab;

    /* With tied embeddings the table is also the output projection, which
     * quantization must treat with care (contract, Parameter binding). */
    ATH_CHECK(bind(&b, -1, NM(&b, "embed_tokens.weight", "token_embd.weight"), S2(v, h), ATH_ROLE_EMBED,
                   RESID | (c->tie_embeddings ? SENS : 0), &q->embed));
    for (int32_t l = 0; l < c->n_layers; l++) {
        ath_qwen35_layer *L = &q->layer[l];
        ATH_CHECK(bind(&b, l, NM(&b, "input_layernorm.weight", "attn_norm.weight"), S1(h), ATH_ROLE_NORM, 0,
                       &L->input_norm));
        if (c->mixer[l] == ATH_QWEN35_ATTENTION)
            ATH_CHECK(bind_attention(&b, l, L));
        else
            ATH_CHECK(bind_gated_deltanet(&b, l, L));
        ATH_CHECK(bind(&b, l, NM(&b, "post_attention_layernorm.weight", "post_attention_norm.weight"), S1(h),
                       ATH_ROLE_NORM, 0, &L->post_norm));
        ATH_CHECK(c->moe ? bind_moe(&b, l, L, a) : bind_mlp(&b, l, L));
    }
    ATH_CHECK(bind(&b, -1, NM(&b, "norm.weight", "output_norm.weight"), S1(h), ATH_ROLE_NORM, 0, &q->norm));

    /* The output head is a top-level tensor in every checkpoint layout. */
    const char *head_name = q->gguf ? "output.weight" : "lm_head.weight";
    const ath_ckpt_tensor *head = ath_ckpt_find(&m->ckpt, head_name);
    if (c->tie_embeddings) {
        if (head) {
            ath_log(ATH_LOG_ERROR, "qwen35: tie_word_embeddings is true but the checkpoint has %s", head_name);
            return ATH_ERR_FORMAT;
        }
        q->head = q->embed;
    } else {
        const ath_ckpt_tensor *e;
        ATH_CHECK(find(&b, head_name, S2(v, h), ATH_ROLE_LM_HEAD, &e));
        q->head = add(&b, &e->t, e->name, ATH_ROLE_LM_HEAD, -1, -1, SENS);
    }

    for (size_t g = 0; g < n_groups; g++)
        bind_unused(&b, groups[g].prefix);
    return ATH_OK;
}
