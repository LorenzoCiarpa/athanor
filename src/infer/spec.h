/* spec.h: speculative decoding with the model's draft head (step 13).
 *
 * The draft head (multi-token prediction: model.h, ath_model_draft) guesses
 * the token after the next from the main model's output and the next token.
 * A step runs the next token and the guess through the main model in one
 * pass (a pass of two tokens costs about one of one at batch 1: the weights
 * are read once), then chooses the token after the next token from the
 * first row's logits exactly as plain decoding would, with the same uniform
 * number. If it is the guess, the second row's logits are those plain
 * decoding would compute next, and the token after it is chosen from them
 * with the next uniform number: two tokens for one pass. If not, one token,
 * and the pass's work on the guess is undone.
 *
 * So the tokens are those of plain decoding with the same sampler and the
 * same uniform numbers, whatever the guesses, as long as a pass of two or
 * three tokens gives each row the bits of a pass of one (the GEMV kernels
 * and the recurrent kernels of short batches do: tests/test_spec in
 * test_engine.c). The guess only changes how many tokens a pass yields.
 *
 * Undoing: the attention caches need nothing (rows past n_past are never
 * read); the convolution and recurrent states, and the draft head's input
 * (ATH_STATE_HIDDEN), are copied before the pass and copied back. The token
 * chosen after a rejected guess is not run at once: the next pass runs it
 * with the token before it and the next guess (three tokens).
 *
 * Bookkeeping in the state: the main model has run [0, n_past); the draft
 * head has run positions [0, n_draft), each with the token after it; the
 * state's ATH_STATE_HIDDEN holds the main model's output at n_past − 1.
 * ath_spec_feed keeps this for prompts; between steps the tokens chosen and
 * not run yet are the spec's own (ath_spec.pend).
 *
 * This version drives each pass from the host (no recorded graphs) and
 * chooses on the host (ath_sample_ref, ath_presence_ref); the GPU decoder's
 * recording is later work. On the CPU and the GPU alike.
 */
#ifndef ATH_INFER_SPEC_H
#define ATH_INFER_SPEC_H

#include "model/model.h"
#include "ops/ops.h"

#define ATH_SPEC_ROWS 3    /* tokens of a verifying pass at most */

typedef struct {
    const ath_model *m;
    ath_state       *s;
    ath_exec         exec;
    ath_sampler      p;
    int32_t          chunk;          /* prompt tokens per pass                           */
    ath_arena        scratch;        /* host working memory                              */
    int32_t          hid_state;      /* index of the ATH_STATE_HIDDEN buffer             */
    /* where the passes run (device memory on a GPU) */
    ath_tensor       hid;            /* [ATH_SPEC_ROWS, hidden] act: a pass's outputs    */
    ath_tensor       dhid;           /* [ATH_SPEC_ROWS, hidden] act: the draft's inputs  */
    void           **saved;          /* [n_states]: copies of the fixed-size buffers     */
    /* host */
    float           *logits;         /* [ATH_SPEC_ROWS, vocab] a pass's logits           */
    float           *dlogits;        /* [vocab] the guess's logits                       */
    uint8_t         *seen;           /* [vocab] tokens of this answer (presence penalty) */
    int32_t          pend[2];        /* chosen, not run yet: 1, or 2 after a rejection   */
    int32_t          n_pend;
    int32_t          guess;          /* the draft's guess at the token after pend        */
    /* counts since ath_spec_open */
    uint64_t         passes, guesses, accepted, reruns;
    bool             own_scratch_dev;
} ath_spec;

/* A speculative decoder for state s, run where x says (NULL: the CPU; a GPU
 * exec from an engine opened with the draft head: ath_place_opts.draft),
 * choosing with p. ATH_ERR_UNSUPPORTED if the model has no draft head.
 * Device buffers come from the device of x (freed by ath_spec_close). */
ath_status ath_spec_open(ath_spec *sp, const ath_model *m, ath_state *s, const ath_exec *x, const ath_sampler *p,
                         int32_t chunk, ath_arena *a);
void       ath_spec_close(ath_spec *sp);

/* Run ids[0 .. n) through the main model and the draft head, from the
 * state's n_past, in passes of at most chunk tokens; the last token's
 * logits into logits (host, [1, vocab]) if not NULL. Forgets the tokens a
 * previous answer chose and did not run. */
ath_status ath_spec_feed(ath_spec *sp, const int32_t *ids, int32_t n, float *logits);

/* A new answer whose first token, chosen from the prompt's logits, is
 * first: its guess is made. Clears the presence penalty's memory. */
ath_status ath_spec_begin(ath_spec *sp, int32_t first);

/* One pass: the next tokens, one or two, into out (*n_out), chosen with
 * u[0] and then u[1] (uniform in [0, 1)). ATH_ERR_RANGE when the state has
 * no room for the pass. */
ath_status ath_spec_step(ath_spec *sp, const double u[2], int32_t out[2], int32_t *n_out);

#endif /* ATH_INFER_SPEC_H */
