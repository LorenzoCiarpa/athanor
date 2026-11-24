/* forward.c: the Qwen3.5 forward pass, composed from the reference ops and
 * ath_matmul_fast, which gives the matmuls the same bits as ath_matmul_ref
 * on the threads of the caller's pool.
 *
 * One call runs T tokens of one sequence at positions P … P + T − 1, where P
 * is the number of tokens the state has already seen: a whole prompt
 * (prefill), the next piece of one, or a single token (decode) all take the
 * same path. Per layer (docs/contract/qwen35-0.8b.md, Decoder layer):
 *
 *   h = x + mixer(rmsnorm(x))        Gated DeltaNet or full attention
 *   x = h + mlp(rmsnorm(h))
 *
 * then logits = rmsnorm(x) · headᵀ, in fp32, for every token or the last.
 *
 * Every activation is stored in the state's activation dtype: bf16 rounds
 * each op's output once, where the reference rounds it too (it also rounds
 * some intermediate values that athanor keeps in fp32: contract and
 * tests/README.md); f32 keeps everything in fp32, an exact forward pass
 * against which the bf16 one and later the GPU are measured. Values the
 * reference keeps in fp32 whatever the activation dtype stay fp32 here too:
 * the L2-normalized q and k, the decay g, the recurrent state, the logits.
 *
 * The state buffers (model.h, state specs declared in family.c):
 *   KV         [capacity, n_kv_heads, head_dim] keys, then the same for values
 *   CONV       [K − 1, conv_dim], the last inputs of the convolution
 *   RECURRENT  [n_v_heads, k_dim, v_dim] fp32
 *
 * Working tensors come from the scratch arena and are released after every
 * layer, so scratch memory is one layer's worth for T tokens.
 */
#include "qwen35.h"

#include "ops/ops.h"

#include <math.h>
#include <string.h>

/* l2norm() in the reference uses a fixed epsilon, not rms_norm_eps
 * (contract, Gated DeltaNet layer). */
#define L2NORM_EPS 1e-6f

typedef struct {
    const ath_model         *m;
    const ath_qwen35_config *c;
    ath_state               *s;
    ath_arena               *a;
    ath_pool                *pool;  /* threads for the matmuls, or NULL */
    ath_dtype                act;
    int64_t                  T;     /* tokens in this call            */
    int64_t                  P;     /* tokens before them (n_past)    */
    float                    eps;   /* rms_norm_eps                   */
    ath_tensor               pos;   /* i32 [T]: P, P + 1, ...         */
} fwd;

/* ---- tensors ------------------------------------------------------------------- */

/* A new contiguous tensor in the scratch arena. */
static ath_status tmp(fwd *f, ath_tensor *t, ath_dtype dt, int32_t ndim, int64_t d0, int64_t d1, int64_t d2)
{
    const int64_t shape[3] = {d0, d1, d2};
    ATH_CHECK(ath_tensor_init(t, dt, ndim, shape, NULL, ATH_TIER_RAM));
    size_t bytes = ath_tensor_nbytes(t);
    t->data = ath_arena_alloc(f->a, bytes ? bytes : 1, 64);
    return t->data ? ATH_OK : ATH_ERR_NOMEM;
}

/* The same elements as t, seen with another shape. */
static ath_status view(const ath_tensor *t, ath_tensor *out, int32_t ndim, int64_t d0, int64_t d1, int64_t d2)
{
    const int64_t shape[3] = {d0, d1, d2};
    return ath_tensor_reshape(t, ndim, shape, out);
}

/* A host tensor over memory the forward pass does not own (state, tokens). */
static ath_status wrap(ath_tensor *t, ath_dtype dt, void *data, int32_t ndim, int64_t d0, int64_t d1, int64_t d2)
{
    const int64_t shape[3] = {d0, d1, d2};
    return ath_tensor_init(t, dt, ndim, shape, data, ATH_TIER_RAM);
}

/* dst[r, :] = src[r, col : col + n] for every row r, n the last dimension of
 * dst: a slice of the last dimension copied out (exactly) so that the ops get
 * contiguous tensors. src and dst have the same dtype and number of rows. */
static void copy_cols(const ath_tensor *src, int64_t col, ath_tensor *dst)
{
    const size_t es = ath_dtype_size(src->dtype);
    const int64_t sc = src->shape[src->ndim - 1], dc = dst->shape[dst->ndim - 1];
    const int64_t rows = ath_tensor_numel(dst) / dc;
    for (int64_t r = 0; r < rows; r++)
        memcpy((char *)dst->data + (size_t)(r * dc) * es, (const char *)src->data + (size_t)(r * sc + col) * es,
               (size_t)dc * es);
}

/* ---- mixers --------------------------------------------------------------------- */

/* Full attention (contract, Full-attention layer): q_proj gives query and
 * output gate interleaved per head; QK-norm, RoPE, the new keys and values
 * appended to the KV cache; causal attention over every position so far;
 * the output gated by sigmoid(gate), then o_proj. */
static ath_status attention(fwd *f, const ath_qwen35_layer *L, const ath_tensor *xn, ath_tensor *out)
{
    const ath_qwen35_config *c = f->c;
    const int64_t T = f->T, P = f->P, S = P + T, cap = f->s->capacity;
    const int64_t nh = c->n_heads, nkv = c->n_kv_heads, hd = c->head_dim;
    const ath_dtype act = f->act;

    /* KV cache of this layer: keys [cap, nkv, hd], then values. */
    char *kv = f->s->buf[L->kv];
    ath_tensor kc, vc, k_new, v_new, k_all, v_all;
    ATH_CHECK(wrap(&kc, act, kv, 3, cap, nkv, hd));
    ATH_CHECK(wrap(&vc, act, kv + ath_tensor_nbytes(&kc), 3, cap, nkv, hd));
    ATH_CHECK(ath_tensor_slice(&kc, 0, P, T, &k_new));
    ATH_CHECK(ath_tensor_slice(&vc, 0, P, T, &v_new));
    ATH_CHECK(ath_tensor_slice(&kc, 0, 0, S, &k_all));
    ATH_CHECK(ath_tensor_slice(&vc, 0, 0, S, &v_all));

    /* projections; the values go straight into the cache */
    ath_tensor qg, qg_rows, q, q_rows, gate, gate_rows, k, k_rows, v_flat;
    ATH_CHECK(tmp(f, &qg, act, 2, T, nh * 2 * hd, 0));
    ATH_CHECK(ath_matmul_fast(xn, &L->q_proj->t, &qg, f->pool));
    ATH_CHECK(tmp(f, &q, act, 3, T, nh, hd));
    ATH_CHECK(tmp(f, &gate, act, 3, T, nh, hd));
    ATH_CHECK(view(&qg, &qg_rows, 2, T * nh, 2 * hd, 0));
    ATH_CHECK(view(&q, &q_rows, 2, T * nh, hd, 0));
    ATH_CHECK(view(&gate, &gate_rows, 2, T * nh, hd, 0));
    copy_cols(&qg_rows, 0, &q_rows);     /* per head: query [0, hd), gate [hd, 2 hd) */
    copy_cols(&qg_rows, hd, &gate_rows);
    ATH_CHECK(tmp(f, &k, act, 2, T, nkv * hd, 0));
    ATH_CHECK(ath_matmul_fast(xn, &L->k_proj->t, &k, f->pool));
    ATH_CHECK(view(&v_new, &v_flat, 2, T, nkv * hd, 0));
    ATH_CHECK(ath_matmul_fast(xn, &L->v_proj->t, &v_flat, f->pool));

    /* QK-norm (zero-centered, per head), then RoPE; rotated keys into the cache */
    ath_tensor qn, kn, qn3, kn3, qr;
    ATH_CHECK(view(&k, &k_rows, 2, T * nkv, hd, 0));
    ATH_CHECK(tmp(f, &qn, act, 2, T * nh, hd, 0));
    ATH_CHECK(tmp(f, &kn, act, 2, T * nkv, hd, 0));
    ATH_CHECK(ath_rmsnorm_ref(&q_rows, &L->q_norm->t, 1.0f, f->eps, &qn));
    ATH_CHECK(ath_rmsnorm_ref(&k_rows, &L->k_norm->t, 1.0f, f->eps, &kn));
    ATH_CHECK(view(&qn, &qn3, 3, T, nh, hd));
    ATH_CHECK(view(&kn, &kn3, 3, T, nkv, hd));
    ATH_CHECK(tmp(f, &qr, act, 3, T, nh, hd));
    ATH_CHECK(ath_rope_ref(&qn3, &f->pos, c->rope_dim, (float)c->rope_theta, &qr));
    ATH_CHECK(ath_rope_ref(&kn3, &f->pos, c->rope_dim, (float)c->rope_theta, &k_new));

    /* attention over positions 0 … S − 1, output gate, output projection */
    ath_tensor o, o_flat, gate_flat, og;
    ATH_CHECK(tmp(f, &o, act, 3, T, nh, hd));
    ATH_CHECK(ath_attn_ref(&qr, &k_all, &v_all, (float)(1.0 / sqrt((double)hd)), &o));
    ATH_CHECK(view(&o, &o_flat, 2, T, nh * hd, 0));
    ATH_CHECK(view(&gate, &gate_flat, 2, T, nh * hd, 0));
    ATH_CHECK(tmp(f, &og, act, 2, T, nh * hd, 0));
    ATH_CHECK(ath_sigmoid_mul_ref(&o_flat, &gate_flat, &og));
    return ath_matmul_fast(&og, &L->o_proj->t, out, f->pool);
}

/* Gated DeltaNet (contract, Gated DeltaNet layer): projections; causal conv
 * with SiLU over the q, k, v channels, continuing from the conv state; L2
 * norm of q and k; gates; the gated delta rule on the recurrent state; gated
 * RMSNorm with z per value head; out_proj. */
static ath_status gated_deltanet(fwd *f, const ath_qwen35_layer *L, const ath_tensor *xn, ath_tensor *out)
{
    const ath_qwen35_config *c = f->c;
    const int64_t T = f->T;
    const int64_t nk = c->lin_k_heads, nv = c->lin_v_heads, dk = c->lin_k_dim, dv = c->lin_v_dim;
    const int64_t key_dim = nk * dk, value_dim = nv * dv, conv_dim = 2 * key_dim + value_dim;
    const ath_dtype act = f->act;

    ath_tensor qkv, z, a, b;
    ATH_CHECK(tmp(f, &qkv, act, 2, T, conv_dim, 0));
    ATH_CHECK(tmp(f, &z, act, 2, T, value_dim, 0));
    ATH_CHECK(tmp(f, &a, act, 2, T, nv, 0));
    ATH_CHECK(tmp(f, &b, act, 2, T, nv, 0));
    ATH_CHECK(ath_matmul_fast(xn, &L->qkv_proj->t, &qkv, f->pool));
    ATH_CHECK(ath_matmul_fast(xn, &L->z_proj->t, &z, f->pool));
    ATH_CHECK(ath_matmul_fast(xn, &L->a_proj->t, &a, f->pool));
    ATH_CHECK(ath_matmul_fast(xn, &L->b_proj->t, &b, f->pool));

    /* convolution over all conv_dim channels, then the split into q, k, v */
    ath_tensor conv_state, mixed, q, k, v, q2, k2, v2;
    ATH_CHECK(wrap(&conv_state, act, f->s->buf[L->conv_state], 2, c->conv_kernel - 1, conv_dim, 0));
    ATH_CHECK(tmp(f, &mixed, act, 2, T, conv_dim, 0));
    ATH_CHECK(ath_conv_ref(&qkv, &L->conv->t, &conv_state, ATH_ACT_SILU, &mixed));
    ATH_CHECK(tmp(f, &q, act, 3, T, nk, dk));
    ATH_CHECK(tmp(f, &k, act, 3, T, nk, dk));
    ATH_CHECK(tmp(f, &v, act, 3, T, nv, dv));
    ATH_CHECK(view(&q, &q2, 2, T, key_dim, 0));
    ATH_CHECK(view(&k, &k2, 2, T, key_dim, 0));
    ATH_CHECK(view(&v, &v2, 2, T, value_dim, 0));
    copy_cols(&mixed, 0, &q2);
    copy_cols(&mixed, key_dim, &k2);
    copy_cols(&mixed, 2 * key_dim, &v2);

    /* fp32 from here on, as in the reference: L2 norm, gates, recurrence */
    ath_tensor qn, kn, g, beta, rec, o;
    ATH_CHECK(tmp(f, &qn, ATH_DTYPE_F32, 3, T, nk, dk));
    ATH_CHECK(tmp(f, &kn, ATH_DTYPE_F32, 3, T, nk, dk));
    ATH_CHECK(ath_l2norm_ref(&q, L2NORM_EPS, &qn));
    ATH_CHECK(ath_l2norm_ref(&k, L2NORM_EPS, &kn));
    /* beta = sigmoid(b) is rounded to the activation dtype, as the reference
     * computes it on the bf16 projection; g is fp32 */
    ATH_CHECK(tmp(f, &g, ATH_DTYPE_F32, 2, T, nv, 0));
    ATH_CHECK(tmp(f, &beta, act, 2, T, nv, 0));
    ATH_CHECK(ath_gdn_gates_ref(&a, &b, &L->a_log->t, &L->dt_bias->t, &g, &beta));
    ATH_CHECK(wrap(&rec, ATH_DTYPE_F32, f->s->buf[L->recurrent], 3, nv, dk, dv));
    ATH_CHECK(tmp(f, &o, act, 3, T, nv, dv));
    ATH_CHECK(ath_gdn_ref(&qn, &kn, &v, &g, &beta, (float)(1.0 / sqrt((double)dk)), &rec, &o));

    /* gated RMSNorm per value head, then the output projection */
    ath_tensor o_rows, z_rows, on, on_flat;
    ATH_CHECK(view(&o, &o_rows, 2, T * nv, dv, 0));
    ATH_CHECK(view(&z, &z_rows, 2, T * nv, dv, 0));
    ATH_CHECK(tmp(f, &on, act, 2, T * nv, dv, 0));
    ATH_CHECK(ath_rmsnorm_gated_ref(&o_rows, &z_rows, &L->norm->t, f->eps, &on));
    ATH_CHECK(view(&on, &on_flat, 2, T, value_dim, 0));
    return ath_matmul_fast(&on_flat, &L->out_proj->t, out, f->pool);
}

/* SwiGLU MLP (contract, MLP): down(silu(gate(x)) · up(x)). */
static ath_status mlp(fwd *f, const ath_qwen35_layer *L, const ath_tensor *xn, ath_tensor *out)
{
    const int64_t T = f->T, I = f->c->intermediate;
    ath_tensor gate, up, h;
    ATH_CHECK(tmp(f, &gate, f->act, 2, T, I, 0));
    ATH_CHECK(tmp(f, &up, f->act, 2, T, I, 0));
    ATH_CHECK(tmp(f, &h, f->act, 2, T, I, 0));
    ATH_CHECK(ath_matmul_fast(xn, &L->gate_proj->t, &gate, f->pool));
    ATH_CHECK(ath_matmul_fast(xn, &L->up_proj->t, &up, f->pool));
    ATH_CHECK(ath_swiglu_ref(&gate, &up, &h));
    return ath_matmul_fast(&h, &L->down_proj->t, out, f->pool);
}

/* One decoder layer, in place on the residual stream x [T, hidden]. Both
 * residual additions round once to the activation dtype, as the reference's. */
static ath_status layer(fwd *f, int32_t l, ath_tensor *x)
{
    const ath_qwen35_layer *L = &((const ath_qwen35_model *)f->m->family_data)->layer[l];
    const int64_t T = f->T, H = f->c->hidden;
    ath_tensor xn, y;
    ATH_CHECK(tmp(f, &xn, f->act, 2, T, H, 0));
    ATH_CHECK(tmp(f, &y, f->act, 2, T, H, 0));

    ATH_CHECK(ath_rmsnorm_ref(x, &L->input_norm->t, 1.0f, f->eps, &xn));
    if (f->c->mixer[l] == ATH_QWEN35_ATTENTION)
        ATH_CHECK(attention(f, L, &xn, &y));
    else
        ATH_CHECK(gated_deltanet(f, L, &xn, &y));
    ATH_CHECK(ath_add_ref(x, &y, x));

    ATH_CHECK(ath_rmsnorm_ref(x, &L->post_norm->t, 1.0f, f->eps, &xn));
    ATH_CHECK(mlp(f, L, &xn, &y));
    return ath_add_ref(x, &y, x);
}

/* ---- entry point ------------------------------------------------------------------ */

ath_status ath_qwen35_forward(const ath_model *m, const ath_batch *b, ath_state *s, const ath_hooks *h,
                              ath_arena *scratch, ath_pool *pool)
{
    const ath_qwen35_model *q = m->family_data;
    const ath_qwen35_config *c = &q->c;
    if (c->moe) {
        ath_log(ATH_LOG_ERROR, "qwen35: the forward pass of MoE layers arrives in step 12");
        return ATH_ERR_UNSUPPORTED;
    }
    fwd f = {
        .m = m, .c = c, .s = s, .a = scratch, .pool = pool, .act = s->act,
        .T = b->n_tokens, .P = s->n_past, .eps = (float)c->rms_eps,
    };
    const int64_t T = f.T, H = c->hidden;

    ath_tensor ids, x;
    ATH_CHECK(wrap(&ids, ATH_DTYPE_I32, (void *)b->tokens, 1, T, 0, 0));
    ATH_CHECK(tmp(&f, &f.pos, ATH_DTYPE_I32, 1, T, 0, 0));
    for (int64_t i = 0; i < T; i++)
        ((int32_t *)f.pos.data)[i] = (int32_t)(f.P + i);
    ATH_CHECK(tmp(&f, &x, f.act, 2, T, H, 0));
    ATH_CHECK(ath_embed_ref(&q->embed->t, &ids, &x));

    for (int32_t l = 0; l < c->n_layers; l++) {
        if (h && h->on_residual)
            h->on_residual(h->u, l, &x);
        ath_arena_pos mark = ath_arena_mark(scratch);
        ath_status st = layer(&f, l, &x);
        ath_arena_reset(scratch, mark);
        ATH_CHECK(st);
    }
    if (h && h->on_residual)
        h->on_residual(h->u, c->n_layers, &x);

    if (!b->logits)
        return ATH_OK;
    /* final norm and output projection on the rows that need logits */
    ath_tensor rows = x, xn;
    if (b->logits->shape[0] == 1)
        ATH_CHECK(ath_tensor_slice(&x, 0, T - 1, 1, &rows));
    ATH_CHECK(tmp(&f, &xn, f.act, 2, rows.shape[0], H, 0));
    ATH_CHECK(ath_rmsnorm_ref(&rows, &q->norm->t, 1.0f, f.eps, &xn));
    return ath_matmul_fast(&xn, &q->head->t, b->logits, f.pool);
}
