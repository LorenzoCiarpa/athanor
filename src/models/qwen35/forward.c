/* forward.c: the Qwen3.5 forward pass, composed from the ops' entry points
 * (ops.h, dispatch), once for both devices: on the CPU they are the
 * reference ops and ath_matmul_fast (the reference's bits on the threads of
 * the caller's pool), on a GPU the kernels of step 7 (step 8).
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
 * Working tensors come from the scratch arena (on a GPU, the device scratch
 * arena of the ath_exec) and are released after every layer, so scratch
 * memory is one layer's worth for T tokens. On a GPU the weights are the
 * params' device copies (x->weights, by param index), the token ids and
 * positions are copied to the device, every op is queued on the compute
 * stream, and the logits come back to the batch's host tensor at the end.
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
    ath_pool                *pool;  /* CPU: threads for the matmuls, or NULL */
    const ath_device        *dev;   /* NULL: the CPU                        */
    ath_dev_arena           *da;    /* GPU: working memory                  */
    const ath_tensor        *w;     /* GPU: the params' device copies       */
    ath_dtype                act;
    int64_t                  T;     /* tokens in this call            */
    int64_t                  P;     /* tokens before them (n_past)    */
    float                    eps;   /* rms_norm_eps                   */
    ath_tensor               pos;   /* i32 [T]: P, P + 1, ...         */
} fwd;

/* ---- tensors ------------------------------------------------------------------- */

static ath_tier tier(const fwd *f) { return f->dev ? ATH_TIER_VRAM : ATH_TIER_RAM; }

/* A new contiguous tensor in the scratch arena of the device the pass runs on. */
static ath_status tmp(fwd *f, ath_tensor *t, ath_dtype dt, int32_t ndim, int64_t d0, int64_t d1, int64_t d2)
{
    const int64_t shape[3] = {d0, d1, d2};
    ATH_CHECK(ath_tensor_init(t, dt, ndim, shape, NULL, tier(f)));
    size_t bytes = ath_tensor_nbytes(t);
    t->data = f->dev ? ath_dev_arena_alloc(f->da, bytes ? bytes : 1, 0) : ath_arena_alloc(f->a, bytes ? bytes : 1, 64);
    return t->data ? ATH_OK : ATH_ERR_NOMEM;
}

/* The tensor of param p where the pass reads it: the param itself on the
 * CPU, its device copy on a GPU (the layer table points into m->params, so
 * the index is the pointer's offset). */
static const ath_tensor *W(const fwd *f, const ath_param *p)
{
    return f->w ? &f->w[p - f->m->params] : &p->t;
}

/* The same elements as t, seen with another shape. */
static ath_status view(const ath_tensor *t, ath_tensor *out, int32_t ndim, int64_t d0, int64_t d1, int64_t d2)
{
    const int64_t shape[3] = {d0, d1, d2};
    return ath_tensor_reshape(t, ndim, shape, out);
}

/* A tensor over state memory, on the side the pass runs on. */
static ath_status wrap(const fwd *f, ath_tensor *t, ath_dtype dt, void *data, int32_t ndim, int64_t d0, int64_t d1,
                       int64_t d2)
{
    const int64_t shape[3] = {d0, d1, d2};
    return ath_tensor_init(t, dt, ndim, shape, data, tier(f));
}

/* dst[r, :] = src[r, col : col + n] for every row r, n the last dimension of
 * dst: a slice of the last dimension copied out (exactly) so that the ops get
 * contiguous tensors. src and dst have the same dtype and number of rows. On
 * a GPU, one strided copy on the compute stream. */
static ath_status copy_cols(const fwd *f, const ath_tensor *src, int64_t col, ath_tensor *dst)
{
    const size_t es = ath_dtype_size(src->dtype);
    const int64_t sc = src->shape[src->ndim - 1], dc = dst->shape[dst->ndim - 1];
    const int64_t rows = ath_tensor_numel(dst) / dc;
    if (f->dev)
        return ath_copy_2d(f->dev, dst->data, (size_t)dc * es, (const char *)src->data + (size_t)col * es,
                           (size_t)sc * es, (size_t)dc * es, (size_t)rows, ATH_STREAM_COMPUTE);
    for (int64_t r = 0; r < rows; r++)
        memcpy((char *)dst->data + (size_t)(r * dc) * es, (const char *)src->data + (size_t)(r * sc + col) * es,
               (size_t)dc * es);
    return ATH_OK;
}

/* The matmuls of rmsnorm(x) (norm: the param of its weights, zero-centered):
 * the norm into a temporary, then the projections that share it. */
static ath_status norm_matmuls(fwd *f, const ath_tensor *x, const ath_param *norm, int n, const ath_tensor *const *w,
                               ath_tensor *const *out)
{
    ath_tensor xn;
    ATH_CHECK(tmp(f, &xn, f->act, 2, x->shape[0], x->shape[1], 0));
    ATH_CHECK(ath_rmsnorm(f->dev, x, W(f, norm), 1.0f, f->eps, &xn));
    return ath_matmul_n(f->dev, f->pool, &xn, n, w, out);
}

/* x += h · wᵀ, the output projection added to the residual stream (both
 * roundings of the reference's two steps). On a GPU, where the GEMV kernel
 * takes it, one launch (ath_matmul_add_gpu, step 9); otherwise the two ops. */
static ath_status matmul_add(fwd *f, const ath_tensor *h, const ath_tensor *w, ath_tensor *x)
{
    if (f->dev && ath_matmul_fits_gemv(h, w))
        return ath_matmul_add_gpu(f->dev, h, w, x);
    ath_tensor y;
    ATH_CHECK(tmp(f, &y, f->act, 2, x->shape[0], x->shape[1], 0));
    ATH_CHECK(ath_matmul(f->dev, f->pool, h, w, &y));
    return ath_add(f->dev, x, &y, x);
}

/* ---- mixers --------------------------------------------------------------------- */

/* Full attention (contract, Full-attention layer): q_proj gives query and
 * output gate interleaved per head; QK-norm, RoPE, the new keys and values
 * written into the KV cache at their positions; causal attention over every
 * position so far; the output gated by sigmoid(gate), then o_proj.
 *
 * The cache is passed whole and the positions say which rows each query
 * reads and where the new rows go (step 9): no argument of any op depends
 * on the number of tokens already seen, so a decode step can be captured
 * once and replayed. */
static ath_status attention(fwd *f, const ath_qwen35_layer *L, ath_tensor *x)
{
    const ath_qwen35_config *c = f->c;
    const int64_t T = f->T, cap = f->s->capacity;
    const int64_t nh = c->n_heads, nkv = c->n_kv_heads, hd = c->head_dim;
    const ath_dtype act = f->act;

    /* KV cache of this layer: keys [cap, nkv, hd], then values. */
    char *kv = f->s->buf[L->kv];
    ath_tensor kc, vc;
    ATH_CHECK(wrap(f, &kc, act, kv, 3, cap, nkv, hd));
    ATH_CHECK(wrap(f, &vc, act, kv + ath_tensor_nbytes(&kc), 3, cap, nkv, hd));

    /* projections */
    ath_tensor qg, qg_rows, q, q_rows, gate, gate_rows, k, k_rows, v_new, v_flat;
    ATH_CHECK(tmp(f, &qg, act, 2, T, nh * 2 * hd, 0));
    ATH_CHECK(tmp(f, &k, act, 2, T, nkv * hd, 0));
    ATH_CHECK(tmp(f, &v_new, act, 3, T, nkv, hd));
    ATH_CHECK(view(&v_new, &v_flat, 2, T, nkv * hd, 0));
    /* the three projections of the normalized x in one pass over their
     * weights (step 9) */
    const ath_tensor *pw[] = {W(f, L->q_proj), W(f, L->k_proj), W(f, L->v_proj)};
    ath_tensor *po[] = {&qg, &k, &v_flat};
    ATH_CHECK(norm_matmuls(f, x, L->input_norm, 3, pw, po));
    ATH_CHECK(tmp(f, &q, act, 3, T, nh, hd));
    ATH_CHECK(tmp(f, &gate, act, 3, T, nh, hd));
    ATH_CHECK(view(&qg, &qg_rows, 2, T * nh, 2 * hd, 0));
    ATH_CHECK(view(&q, &q_rows, 2, T * nh, hd, 0));
    ATH_CHECK(view(&gate, &gate_rows, 2, T * nh, hd, 0));
    ATH_CHECK(copy_cols(f, &qg_rows, 0, &q_rows));     /* per head: query [0, hd), gate [hd, 2 hd) */
    ATH_CHECK(copy_cols(f, &qg_rows, hd, &gate_rows));

    /* QK-norm (zero-centered, per head), then RoPE; the new keys and values
     * into the cache at their positions */
    ath_tensor qn, kn, qn3, kn3, qr, k_new;
    ATH_CHECK(view(&k, &k_rows, 2, T * nkv, hd, 0));
    if (T == 1) {
        /* one token: the normalized q and k side by side, so that one RoPE
         * call rotates both (the same values, step 9) */
        const size_t es = ath_dtype_size(act);
        ath_tensor qkn, qkn3, qkr;
        ATH_CHECK(tmp(f, &qkn, act, 2, nh + nkv, hd, 0));
        ATH_CHECK(wrap(f, &qn, act, qkn.data, 2, nh, hd, 0));
        ATH_CHECK(wrap(f, &kn, act, (char *)qkn.data + (size_t)(nh * hd) * es, 2, nkv, hd, 0));
        ATH_CHECK(ath_rmsnorm(f->dev, &q_rows, W(f, L->q_norm), 1.0f, f->eps, &qn));
        ATH_CHECK(ath_rmsnorm(f->dev, &k_rows, W(f, L->k_norm), 1.0f, f->eps, &kn));
        ATH_CHECK(view(&qkn, &qkn3, 3, 1, nh + nkv, hd));
        ATH_CHECK(tmp(f, &qkr, act, 3, 1, nh + nkv, hd));
        ATH_CHECK(ath_rope(f->dev, &qkn3, &f->pos, c->rope_dim, (float)c->rope_theta, &qkr));
        ATH_CHECK(wrap(f, &qr, act, qkr.data, 3, 1, nh, hd));
        ATH_CHECK(wrap(f, &k_new, act, (char *)qkr.data + (size_t)(nh * hd) * es, 3, 1, nkv, hd));
    } else {
        ATH_CHECK(tmp(f, &qn, act, 2, T * nh, hd, 0));
        ATH_CHECK(tmp(f, &kn, act, 2, T * nkv, hd, 0));
        ATH_CHECK(ath_rmsnorm(f->dev, &q_rows, W(f, L->q_norm), 1.0f, f->eps, &qn));
        ATH_CHECK(ath_rmsnorm(f->dev, &k_rows, W(f, L->k_norm), 1.0f, f->eps, &kn));
        ATH_CHECK(view(&qn, &qn3, 3, T, nh, hd));
        ATH_CHECK(view(&kn, &kn3, 3, T, nkv, hd));
        ATH_CHECK(tmp(f, &qr, act, 3, T, nh, hd));
        ATH_CHECK(tmp(f, &k_new, act, 3, T, nkv, hd));
        ATH_CHECK(ath_rope(f->dev, &qn3, &f->pos, c->rope_dim, (float)c->rope_theta, &qr));
        ATH_CHECK(ath_rope(f->dev, &kn3, &f->pos, c->rope_dim, (float)c->rope_theta, &k_new));
    }
    ATH_CHECK(ath_scatter_rows(f->dev, &k_new, &f->pos, &kc));
    ATH_CHECK(ath_scatter_rows(f->dev, &v_new, &f->pos, &vc));

    /* attention up to each query's position, output gate, output projection */
    ath_tensor o, o_flat, gate_flat, og;
    ATH_CHECK(tmp(f, &o, act, 3, T, nh, hd));
    ATH_CHECK(ath_attn(f->dev, &qr, &kc, &vc, &f->pos, (float)(1.0 / sqrt((double)hd)), f->da, &o));
    ATH_CHECK(view(&o, &o_flat, 2, T, nh * hd, 0));
    ATH_CHECK(view(&gate, &gate_flat, 2, T, nh * hd, 0));
    ATH_CHECK(tmp(f, &og, act, 2, T, nh * hd, 0));
    ATH_CHECK(ath_sigmoid_mul(f->dev, &o_flat, &gate_flat, &og));
    return matmul_add(f, &og, W(f, L->o_proj), x);
}

/* Gated DeltaNet (contract, Gated DeltaNet layer): projections; causal conv
 * with SiLU over the q, k, v channels, continuing from the conv state; L2
 * norm of q and k; gates; the gated delta rule on the recurrent state; gated
 * RMSNorm with z per value head; out_proj. */
static ath_status gated_deltanet(fwd *f, const ath_qwen35_layer *L, ath_tensor *x)
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
    const ath_tensor *pw[] = {W(f, L->qkv_proj), W(f, L->z_proj), W(f, L->a_proj), W(f, L->b_proj)};
    ath_tensor *po[] = {&qkv, &z, &a, &b};
    ATH_CHECK(norm_matmuls(f, x, L->input_norm, 4, pw, po));

    /* convolution over all conv_dim channels, then the split into q, k, v */
    ath_tensor conv_state, mixed, q, k, v, q2, k2, v2;
    ATH_CHECK(wrap(f, &conv_state, act, f->s->buf[L->conv_state], 2, c->conv_kernel - 1, conv_dim, 0));
    ATH_CHECK(tmp(f, &mixed, act, 2, T, conv_dim, 0));
    ATH_CHECK(ath_conv(f->dev, &qkv, W(f, L->conv), &conv_state, ATH_ACT_SILU, &mixed));
    /* fp32 from here on, as in the reference: L2 norm, gates, recurrence */
    ath_tensor qn, kn, g, beta, rec, o;
    if (T == 1) {
        /* one token (decode): q, k and v are contiguous pieces of the one row
         * of mixed, q and k side by side, so they are views, and one L2 norm
         * over the 2 nk heads gives qn and kn (step 9); the same values */
        const size_t es = ath_dtype_size(act);
        ath_tensor qk, qkn;
        ATH_CHECK(wrap(f, &qk, act, mixed.data, 2, 2 * nk, dk, 0));
        ATH_CHECK(wrap(f, &v, act, (char *)mixed.data + (size_t)(2 * key_dim) * es, 3, 1, nv, dv));
        ATH_CHECK(tmp(f, &qkn, ATH_DTYPE_F32, 2, 2 * nk, dk, 0));
        ATH_CHECK(ath_l2norm(f->dev, &qk, L2NORM_EPS, &qkn));
        ATH_CHECK(wrap(f, &qn, ATH_DTYPE_F32, qkn.data, 3, 1, nk, dk));
        ATH_CHECK(wrap(f, &kn, ATH_DTYPE_F32, (char *)qkn.data + (size_t)key_dim * sizeof(float), 3, 1, nk, dk));
    } else {
        ATH_CHECK(tmp(f, &q, act, 3, T, nk, dk));
        ATH_CHECK(tmp(f, &k, act, 3, T, nk, dk));
        ATH_CHECK(tmp(f, &v, act, 3, T, nv, dv));
        ATH_CHECK(view(&q, &q2, 2, T, key_dim, 0));
        ATH_CHECK(view(&k, &k2, 2, T, key_dim, 0));
        ATH_CHECK(view(&v, &v2, 2, T, value_dim, 0));
        ATH_CHECK(copy_cols(f, &mixed, 0, &q2));
        ATH_CHECK(copy_cols(f, &mixed, key_dim, &k2));
        ATH_CHECK(copy_cols(f, &mixed, 2 * key_dim, &v2));
        ATH_CHECK(tmp(f, &qn, ATH_DTYPE_F32, 3, T, nk, dk));
        ATH_CHECK(tmp(f, &kn, ATH_DTYPE_F32, 3, T, nk, dk));
        ATH_CHECK(ath_l2norm(f->dev, &q, L2NORM_EPS, &qn));
        ATH_CHECK(ath_l2norm(f->dev, &k, L2NORM_EPS, &kn));
    }
    /* beta = sigmoid(b) is rounded to the activation dtype, as the reference
     * computes it on the bf16 projection; g is fp32 */
    ATH_CHECK(tmp(f, &g, ATH_DTYPE_F32, 2, T, nv, 0));
    ATH_CHECK(tmp(f, &beta, act, 2, T, nv, 0));
    ATH_CHECK(ath_gdn_gates(f->dev, &a, &b, W(f, L->a_log), W(f, L->dt_bias), &g, &beta));
    ATH_CHECK(wrap(f, &rec, ATH_DTYPE_F32, f->s->buf[L->recurrent], 3, nv, dk, dv));
    ATH_CHECK(tmp(f, &o, act, 3, T, nv, dv));
    ATH_CHECK(ath_gdn(f->dev, &qn, &kn, &v, &g, &beta, (float)(1.0 / sqrt((double)dk)), &rec, f->da, &o));

    /* gated RMSNorm per value head, then the output projection */
    ath_tensor o_rows, z_rows, on, on_flat;
    ATH_CHECK(view(&o, &o_rows, 2, T * nv, dv, 0));
    ATH_CHECK(view(&z, &z_rows, 2, T * nv, dv, 0));
    ATH_CHECK(tmp(f, &on, act, 2, T * nv, dv, 0));
    ATH_CHECK(ath_rmsnorm_gated(f->dev, &o_rows, &z_rows, W(f, L->norm), f->eps, &on));
    ATH_CHECK(view(&on, &on_flat, 2, T, value_dim, 0));
    return matmul_add(f, &on_flat, W(f, L->out_proj), x);
}

/* SwiGLU MLP (contract, MLP): down(silu(gate(x)) · up(x)). */
static ath_status mlp(fwd *f, const ath_qwen35_layer *L, ath_tensor *x)
{
    const int64_t T = f->T, I = f->c->intermediate;
    ath_tensor gate, up, h;
    ATH_CHECK(tmp(f, &gate, f->act, 2, T, I, 0));
    ATH_CHECK(tmp(f, &up, f->act, 2, T, I, 0));
    ATH_CHECK(tmp(f, &h, f->act, 2, T, I, 0));
    const ath_tensor *pw[] = {W(f, L->gate_proj), W(f, L->up_proj)};
    if (f->dev && ath_matmul_swiglu_fits(x, pw[0], pw[1])) {
        /* one token on a GPU: gate, up and SwiGLU in one launch (step 9) */
        ath_tensor xn;
        ATH_CHECK(tmp(f, &xn, f->act, 2, T, f->c->hidden, 0));
        ATH_CHECK(ath_rmsnorm(f->dev, x, W(f, L->post_norm), 1.0f, f->eps, &xn));
        ATH_CHECK(ath_matmul_swiglu_gpu(f->dev, &xn, pw[0], pw[1], &h));
        return matmul_add(f, &h, W(f, L->down_proj), x);
    }
    ath_tensor *po[] = {&gate, &up};
    ATH_CHECK(norm_matmuls(f, x, L->post_norm, 2, pw, po));
    ATH_CHECK(ath_swiglu(f->dev, &gate, &up, &h));
    return matmul_add(f, &h, W(f, L->down_proj), x);
}

/* One decoder layer, in place on the residual stream x [T, hidden]: each
 * half normalizes x into its first projections and adds its output
 * projection back to x (norm_matmuls, matmul_add). Both residual additions
 * round once to the activation dtype, as the reference's. */
static ath_status layer(fwd *f, int32_t l, ath_tensor *x)
{
    const ath_qwen35_layer *L = &((const ath_qwen35_model *)f->m->family_data)->layer[l];
    if (f->c->mixer[l] == ATH_QWEN35_ATTENTION)
        ATH_CHECK(attention(f, L, x));
    else
        ATH_CHECK(gated_deltanet(f, L, x));
    return mlp(f, L, x);
}

/* ---- entry point ------------------------------------------------------------------ */

ath_status ath_qwen35_forward(const ath_model *m, const ath_batch *b, ath_state *s, const ath_hooks *h,
                              ath_arena *scratch, const ath_exec *x_)
{
    const ath_qwen35_model *q = m->family_data;
    const ath_qwen35_config *c = &q->c;
    if (c->moe) {
        ath_log(ATH_LOG_ERROR, "qwen35: the forward pass of MoE layers arrives in step 12");
        return ATH_ERR_UNSUPPORTED;
    }
    fwd f = {
        .m = m, .c = c, .s = s, .a = scratch, .pool = x_->pool, .dev = x_->dev, .da = x_->scratch,
        .w = x_->dev ? x_->weights : NULL, .act = s->act, .T = b->n_tokens, .P = s->n_past,
        .eps = (float)c->rms_eps,
    };
    const int64_t T = f.T, H = c->hidden;
    const bool on_dev = b->flags & ATH_BATCH_DEVICE;

    /* token ids and positions: given in device memory with a device batch;
     * otherwise computed on the host and, on a GPU, copied to the device (a
     * copy from ordinary memory consumes the source before it returns, so
     * the host buffers may go) */
    ath_tensor ids, x;
    const int64_t one[1] = {T};
    if (on_dev) {
        ATH_CHECK(ath_tensor_init(&ids, ATH_DTYPE_I32, 1, one, (void *)b->tokens, ATH_TIER_VRAM));
        ATH_CHECK(ath_tensor_init(&f.pos, ATH_DTYPE_I32, 1, one, (void *)b->pos, ATH_TIER_VRAM));
    } else {
        ath_tensor pos_host;
        ATH_CHECK(ath_tensor_init(&pos_host, ATH_DTYPE_I32, 1, one, NULL, ATH_TIER_RAM));
        pos_host.data = ath_arena_alloc(scratch, (size_t)T * sizeof(int32_t), 64);
        if (!pos_host.data)
            return ATH_ERR_NOMEM;
        for (int64_t i = 0; i < T; i++)
            ((int32_t *)pos_host.data)[i] = (int32_t)(f.P + i);
        if (f.dev) {
            ATH_CHECK(tmp(&f, &ids, ATH_DTYPE_I32, 1, T, 0, 0));
            ATH_CHECK(tmp(&f, &f.pos, ATH_DTYPE_I32, 1, T, 0, 0));
            ATH_CHECK(ath_copy(f.dev, ids.data, b->tokens, (size_t)T * sizeof(int32_t), ATH_STREAM_COMPUTE));
            ATH_CHECK(ath_copy(f.dev, f.pos.data, pos_host.data, (size_t)T * sizeof(int32_t), ATH_STREAM_COMPUTE));
        } else {
            ATH_CHECK(ath_tensor_init(&ids, ATH_DTYPE_I32, 1, one, (void *)b->tokens, ATH_TIER_RAM));
            f.pos = pos_host;
        }
    }
    ATH_CHECK(tmp(&f, &x, f.act, 2, T, H, 0));
    ATH_CHECK(ath_embed(f.dev, W(&f, q->embed), &ids, &x));

    for (int32_t l = 0; l < c->n_layers; l++) {
        if (h && h->on_residual)
            h->on_residual(h->u, l, &x);
        ath_arena_pos mark = ath_arena_mark(scratch);
        ath_dev_arena_pos dmark = f.dev ? ath_dev_arena_mark(f.da) : (ath_dev_arena_pos){0};
        ath_status st = layer(&f, l, &x);
        if (f.dev)
            ath_dev_arena_reset(f.da, dmark);   /* the next layer's work comes after on the stream */
        ath_arena_reset(scratch, mark);
        ATH_CHECK(st);
    }
    if (h && h->on_residual)
        h->on_residual(h->u, c->n_layers, &x);

    if (!b->logits)
        return ATH_OK;
    /* final norm and output projection on the rows that need logits */
    ath_tensor rows = x, logits = *b->logits;
    if (b->logits->shape[0] == 1)
        ATH_CHECK(ath_tensor_slice(&x, 0, T - 1, 1, &rows));
    if (f.dev && !on_dev)
        ATH_CHECK(tmp(&f, &logits, ATH_DTYPE_F32, 2, rows.shape[0], m->vocab_size, 0));
    const ath_tensor *hw[] = {W(&f, q->head)};
    ath_tensor *ho[] = {&logits};
    ATH_CHECK(norm_matmuls(&f, &rows, q->norm, 1, hw, ho));
    if (!f.dev || on_dev)
        return ATH_OK;          /* a device batch: no copy and no wait */
    ATH_CHECK(ath_copy(f.dev, b->logits->data, logits.data, ath_tensor_nbytes(&logits), ATH_STREAM_COMPUTE));
    return ath_stream_sync(f.dev, ATH_STREAM_COMPUTE);
}
