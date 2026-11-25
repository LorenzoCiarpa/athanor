/* engine.h: a model on a GPU (step 8).
 *
 * The engine opens a device, copies there every param the forward pass
 * reads (the weights, once, as stored in the checkpoint), and holds what a
 * forward pass on that device needs: the ath_exec of model.h and its
 * working memory. Sequence states are allocated on the same device. A
 * pipeline then runs the model through the contract as on the CPU:
 *
 *   ath_engine e;
 *   ATH_CHECK(ath_engine_open(&e, &model, 0, &arena));
 *   ath_state s;
 *   ATH_CHECK(ath_engine_state(&e, &s, ATH_DTYPE_BF16, 8192));
 *   ATH_CHECK(ath_model_forward(&model, &batch, &s, NULL, &scratch, &e.exec));
 *   ...
 *   ath_engine_close(&e);
 *
 * Batches are one sequence at a time; several sequences per batch and paged
 * KV memory come with the server (step 13). Decoding one token at a time
 * goes through ath_decoder (below, step 9).
 */
#ifndef ATH_ENGINE_H
#define ATH_ENGINE_H

#include "backend/backend.h"
#include "model/model.h"
#include "ops/ops.h"

/* An open engine must stay where it is: its arenas and exec point at its
 * device. */
typedef struct {
    const ath_model *m;
    ath_device       dev;
    ath_dev_arena    weights_mem;    /* the params' copies, for the engine's life     */
    ath_dev_arena    state_mem;      /* sequence states (ath_engine_state)            */
    ath_dev_arena    scratch;        /* working memory of one forward pass            */
    ath_tensor      *weights;        /* [n_params]: device copies; empty for ignored  */
    uint64_t         weight_bytes;   /* bytes copied to the device                    */
    uint32_t         n_copied;       /* params copied                                 */
    ath_exec         exec;           /* pass &exec to ath_model_forward               */
} ath_engine;

/* Open device `device`, copy the model's params to it and prepare the
 * exec. Params flagged ATH_P_UNUSED (the vision encoder, the MTP head) stay
 * on the host. Host bookkeeping goes into a, which must outlive the
 * engine. ATH_ERR_UNSUPPORTED without a GPU backend, ATH_ERR_NOMEM if the
 * weights do not fit. On failure everything is released. */
ath_status ath_engine_open(ath_engine *e, const ath_model *m, int32_t device, ath_arena *a);

/* Wait for the device, then free everything (states included). */
void ath_engine_close(ath_engine *e);

/* The state of one sequence on the engine's device (ath_state_init_gpu),
 * reset. The pointer array goes into a. */
ath_status ath_engine_state(ath_engine *e, ath_state *s, ath_dtype act, int32_t capacity, ath_arena *a);

/* ---- decoding (step 9) ----------------------------------------------------------
 *
 * One token at a time on the engine's device: the forward pass of a token
 * and the choice of the next one, recorded once in a graph (backend.h) and
 * replayed for every token after. The host writes the token, its position
 * and the uniform number into pinned memory, launches the graph and reads
 * the next token back: one launch and 4 bytes per token, instead of some
 * 500 launches and the logits. The pass runs as a device batch
 * (ATH_BATCH_DEVICE), so nothing in it depends on the position; sampling
 * runs on the GPU when the sampler fits it (ath_sample_gpu_fits), and
 * otherwise the graph ends by copying the logits to pinned memory and the
 * host chooses (ath_sample_ref). The first step runs the work directly,
 * which also sizes the scratch memory; the second records it.
 *
 *   ath_decoder dc;
 *   ATH_CHECK(ath_decoder_open(&dc, &e, &s, &sampler, &arena));
 *   int32_t tok = first;                        // after the prompt
 *   for (...) ATH_CHECK(ath_decoder_step(&dc, tok, ath_rng_uniform(&rng), &tok));
 *   ath_decoder_close(&dc);
 *
 * A decoder serves one state and one sampler; the state may be reset or
 * copied into (its buffers stay where they are), and other forward calls
 * on it may come between steps. */
typedef struct {
    int32_t token;     /* the token run by the step               */
    int32_t pos;       /* its position: the state's n_past        */
    double  u;         /* the uniform number of the choice        */
    int32_t next;      /* the token chosen                        */
    int32_t pad;
} ath_decode_io;

typedef struct {
    ath_engine     *e;
    ath_state      *s;
    ath_sampler     p;
    bool            gpu_sampling;    /* the choice on the device                      */
    ath_decode_io  *io;              /* device                                        */
    ath_decode_io  *hio;             /* pinned host mirror                            */
    ath_tensor      logits;          /* device f32 [1, vocab]                         */
    ath_tensor      hlogits;         /* pinned f32 [1, vocab], host sampling only     */
    ath_arena       scratch;         /* host working memory of the forward calls      */
    ath_graph       g;
    int64_t         steps;           /* steps run so far                              */
} ath_decoder;

/* A decoder for state s (on e's device) and sampler p. ATH_ERR_INVALID_ARG
 * for a state on another device or a sampler that is not valid. */
ath_status ath_decoder_open(ath_decoder *dc, ath_engine *e, ath_state *s, const ath_sampler *p, ath_arena *a);
/* Run `token` at position s->n_past and choose the next token with the
 * uniform number u in [0, 1): *next. Advances s->n_past by one;
 * ATH_ERR_RANGE when the state is full. */
ath_status ath_decoder_step(ath_decoder *dc, int32_t token, double u, int32_t *next);
void       ath_decoder_close(ath_decoder *dc);

#endif /* ATH_ENGINE_H */
