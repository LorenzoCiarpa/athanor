/* api.h: the HTTP APIs of `athanor serve` (step 13).
 *
 * Three wire formats over one generation (gen.h), one per family of coding
 * agents:
 *
 *   POST /v1/messages                Anthropic Messages (Claude Code)
 *   POST /v1/messages/count_tokens   its token count
 *   POST /v1/chat/completions        OpenAI Chat Completions (OpenCode and most clients)
 *   POST /v1/responses               OpenAI Responses (Codex CLI)
 *   GET  /v1/models, /v1/models/<id> the model, in a shape both families read
 *   GET  /health
 *
 * A request is parsed into an ath_api_request: messages (system and
 * developer messages at the start merged into one, later ones become user
 * messages, since the chat template takes one system message and only at
 * the start), tool definitions, the calls and results in the history,
 * sampling, limits, stop strings, streaming and thinking. Unknown fields
 * are ignored; images become a placeholder (the models run text only).
 * The generation's events are written back in the request's format, as
 * server-sent events when it asks for a stream, whole otherwise. Errors
 * have each API's error shape; a prompt longer than the context gets the
 * message each family's agents recognize (Claude Code: "prompt is too
 * long"; OpenAI: code context_length_exceeded).
 *
 * The parsing is apart from the transport (ath_api_parse) so the tests can
 * run it on fixtures.
 */
#ifndef ATH_NET_API_H
#define ATH_NET_API_H

#include "core/json.h"
#include "net/gen.h"
#include "net/http.h"

#include <stdio.h>

typedef enum { ATH_API_ANTHROPIC, ATH_API_CHAT, ATH_API_RESPONSES } ath_api_kind;

typedef struct {
    ath_gen_request g;
    ath_api_kind    kind;
    bool            stream;
    bool            include_usage;   /* chat, streaming: a last chunk with the usage     */
    const char     *model;           /* as the request names it (echoed back)            */
    const char    **custom;          /* responses: the names of freeform ("custom") tools,
                                      * declared to the model with one string parameter
                                      * "input" and returned as custom_tool_call items  */
    int32_t         n_custom;
} ath_api_request;

/* The server's defaults, for what a request leaves out. */
typedef struct {
    const char *model;               /* the name /v1/models lists                         */
    bool        think;               /* thinking when a request does not say              */
    ath_sampler sampler_think;       /* sampling with and without thinking (the model     */
    ath_sampler sampler_plain;       /* card's settings)                                  */
    float       presence_think, presence_plain;
    int32_t     max_tokens;          /* a request without a limit: 0, as many as fit      */
    bool        client_sampling;     /* take the request's temperature, top-p, top-k...   */
} ath_api_defaults;

typedef struct {
    ath_gen         *gen;
    ath_api_defaults d;
    uint64_t         n_ids;          /* ids handed out, for unique ids                   */
    int64_t          started;        /* the server's start time, seconds                 */
    FILE            *log;            /* a line per request; NULL: none                   */
} ath_api;

void ath_api_init(ath_api *api, ath_gen *gen, const ath_api_defaults *d);

/* Parse body (a request of kind) into r, in a. ATH_ERR_INVALID_ARG with a
 * message in *err for a request that cannot be served. */
ath_status ath_api_parse(const ath_api *api, ath_api_kind kind, const ath_json *body, ath_api_request *r,
                         ath_arena *a, const char **err);

/* The handler of ath_http_run; ctx is an ath_api. */
void ath_api_handle(void *ctx, const ath_http_req *req, ath_http_out *out);

#endif /* ATH_NET_API_H */
