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
 * Batches are one sequence at a time; several sequences per batch, paged KV
 * memory and graph capture come with the server and fast decode (steps 9
 * and 13).
 */
#ifndef ATH_ENGINE_H
#define ATH_ENGINE_H

#include "backend/backend.h"
#include "model/model.h"

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

#endif /* ATH_ENGINE_H */
