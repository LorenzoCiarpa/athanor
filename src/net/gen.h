/* gen.h: generation for the server (step 13), whatever the API.
 *
 * A request comes in as messages and tool definitions (the API parsers of
 * api.c fill an ath_gen_request); the family renders them into the prompt
 * (model.h, chat), the tokenizer turns it into ids, the model runs them and
 * then decodes, and the family's parser turns the output into events
 * (text, thinking, tool calls), which go to the API's writer through an
 * ath_gen_sink as they come.
 *
 * One sequence at a time. A recurrent state cannot be rolled back, so the
 * state is kept as a snapshot after each prompt's stable part (the
 * conversation up to its last message, as `athanor chat` does); a request
 * whose prompt extends that part starts from the snapshot and runs only
 * the new tokens. Agents send the whole conversation every turn, so most of
 * a long session's prompt is run once.
 */
#ifndef ATH_NET_GEN_H
#define ATH_NET_GEN_H

#include "core/rng.h"
#include "infer/engine.h"
#include "model/model.h"
#include "text/bpe.h"

#define ATH_GEN_MAX_STOP 8

typedef struct {
    const ath_msg      *msgs;
    int32_t             n_msgs;
    const ath_tool_def *tools;
    int32_t             n_tools;
    ath_sampler         sampler;
    float               presence_penalty;
    int32_t             max_tokens;         /* 0: as many as fit                     */
    bool                think;              /* the model thinks before it answers    */
    const char         *stop[ATH_GEN_MAX_STOP];
    int32_t             n_stop;
} ath_gen_request;

typedef enum {
    ATH_GEN_END,          /* the model ended its turn                         */
    ATH_GEN_TOOL_CALLS,   /* it ended its turn with tool calls                */
    ATH_GEN_LENGTH,       /* the token limit (or the context) was reached     */
    ATH_GEN_STOP,         /* a stop string                                    */
    ATH_GEN_CANCELLED     /* the sink asked to stop (the client went away)    */
} ath_gen_finish;

typedef struct {
    int32_t        prompt_tokens, cached_tokens, output_tokens;
    ath_gen_finish finish;
    int32_t        stop_index;       /* ATH_GEN_STOP: which stop string        */
} ath_gen_result;

/* The writer of one API. begin comes once, after the prompt is tokenized
 * and before it runs (res has the prompt and cached token counts), so a
 * stream can start; wait comes between chunks of a long prompt (a
 * keep-alive, a check that the client is still there). Then events, in
 * the order the model produced them: text and thinking in whole UTF-8
 * characters; a tool call is begin, argument bytes (one JSON object in
 * all), end. Every member returns false to cancel the generation. */
typedef struct {
    bool (*begin)(void *u, const ath_gen_result *res);
    bool (*wait)(void *u);
    bool (*text)(void *u, const char *s, size_t n);
    bool (*thinking)(void *u, const char *s, size_t n);
    bool (*tool_begin)(void *u, const char *name);
    bool (*tool_args)(void *u, const char *s, size_t n);
    bool (*tool_end)(void *u);
    void  *u;
} ath_gen_sink;

typedef struct {
    int32_t ctx;             /* tokens a sequence holds                            */
    int32_t experts;         /* MoE: experts per layer in VRAM, -1 as many as fit  */
    int32_t threads;         /* CPU threads, 0: all                                */
    int32_t chunk;           /* prompt tokens per forward call, 0: 4096            */
    uint64_t seed;           /* of the sampling                                    */
    bool    cpu;             /* run on the CPU even with a GPU                     */
} ath_gen_opts;

typedef struct {
    ath_model       m;
    ath_tokenizer   tok;
    ath_pool        pool;
    ath_engine     *engine;          /* NULL on the CPU                                 */
    ath_decoder     dec;
    bool            dec_open;
    ath_sampler     dec_sampler;     /* the sampler the decoder was recorded with       */
    ath_exec        exec;
    ath_state       state, snap;
    ath_tokens      seen;            /* the tokens the state holds                      */
    size_t          snap_n;          /* the first snap_n of them are in snap; 0: none   */
    ath_tensor      logits;
    ath_arena       scratch;
    ath_rng         rng;
    int32_t         eos[2];
    int32_t         ctx;
    int32_t         chunk;
    ath_arena      *a;
} ath_gen;

/* Open the model in dir with its tokenizer, on the GPU when there is one. */
ath_status ath_gen_open(ath_gen *g, const char *dir, const ath_gen_opts *o, ath_arena *a);
void       ath_gen_close(ath_gen *g);

/* The prompt of r in tokens (for token counts), in a. */
ath_status ath_gen_tokens(ath_gen *g, const ath_gen_request *r, ath_tokens *out, ath_arena *a);

/* Run r: prefill (from the snapshot when it fits), decode, events to sink.
 * Working memory in a. */
ath_status ath_gen_run(ath_gen *g, const ath_gen_request *r, const ath_gen_sink *sink, ath_gen_result *res,
                       ath_arena *a);

#endif /* ATH_NET_GEN_H */
