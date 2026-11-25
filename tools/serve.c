/* serve.c: `athanor serve <model_dir> [options]`: the model behind the HTTP
 * APIs of coding agents (step 13): Anthropic Messages for Claude Code,
 * OpenAI Responses for Codex CLI, OpenAI Chat Completions for OpenCode and
 * most other clients (src/net/api.h).
 *
 * One model, one sequence at a time, on the GPU when there is one. Requests
 * that arrive while one runs wait their turn. The server never runs a tool
 * (AGENTS.md, invariant 10): the agent does, and sends the results back.
 * No TLS and no keys: it listens on 127.0.0.1 unless told otherwise.
 */
#include "commands.h"

#include "net/api.h"
#include "net/gen.h"
#include "net/http.h"

#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void ath_cmd_serve_usage(FILE *out)
{
    fprintf(out,
            "usage: athanor serve <model_dir> [options]\n\n"
            "Serve the model over HTTP for coding agents, on the GPU when this build has\n"
            "one (make cuda) and the machine too:\n"
            "  POST /v1/messages            Anthropic Messages (Claude Code)\n"
            "  POST /v1/responses           OpenAI Responses (Codex CLI)\n"
            "  POST /v1/chat/completions    OpenAI Chat Completions (OpenCode, others)\n"
            "  GET  /v1/models\n"
            "One request runs at a time; tools are never run by the server.\n\n"
            "  --host <addr>        IPv4 address to listen on (default 127.0.0.1)\n"
            "  --port <n>           default 8080\n"
            "  --name <id>          the model's name in /v1/models (default: the folder's name)\n"
            "  --ctx <n>            context length in tokens (default 65536)\n"
            "  --experts <n>        MoE: routed experts per layer kept in device memory\n"
            "                       (default: as many as fit; the rest run on the CPU)\n"
            "  --threads <n>        CPU threads (default: all)\n"
            "  --device <cpu|gpu>   where the model runs (default: gpu if there is one)\n"
            "  --no-think           requests that do not say get no thinking (default: thinking)\n"
            "  --max-tokens <n>     tokens per answer when a request gives no limit\n"
            "                       (default: as many as fit)\n"
            "  --server-sampling    ignore the sampling settings of requests and use the\n"
            "                       model card's (thinking: temperature 0.6, top-p 0.95,\n"
            "                       top-k 20; otherwise 0.7, 0.8, 20)\n"
            "  --chunk <n>          prompt tokens per forward call (default 4096)\n"
            "  --seed <n>           random seed (default: from the clock)\n"
            "  --dump <dir>         save every request body in dir (for debugging a client)\n"
            "  --quiet              no line per request on stderr\n");
}

static volatile int stop_flag;

static void on_signal(int sig)
{
    (void)sig;
    stop_flag = 1;
}

static bool num(const char *s, double lo, double hi, double *out)
{
    char *end;
    double v = strtod(s, &end);
    if (end == s || *end || !(v >= lo && v <= hi))
        return false;
    *out = v;
    return true;
}

static int usage_error(const char *msg, const char *arg)
{
    fprintf(stderr, "athanor serve: %s%s%s\n\n", msg, arg ? " " : "", arg ? arg : "");
    ath_cmd_serve_usage(stderr);
    return ATH_EXIT_USAGE;
}

int ath_cmd_serve(ath_arena *a, int argc, char **argv)
{
    const char *dir = NULL, *host = "127.0.0.1", *name = NULL, *device = NULL, *dump = NULL;
    int port = 8080;
    bool think = true, quiet = false, server_sampling = false, seed_set = false;
    ath_gen_opts go = {.ctx = 65536, .experts = -1};
    int32_t max_tokens = 0;
    for (int i = 1; i < argc; i++) {
        const char *o = argv[i];
        const char *v = i + 1 < argc ? argv[i + 1] : NULL;
        double x;
        if (!strcmp(o, "-h") || !strcmp(o, "--help")) {
            ath_cmd_serve_usage(stdout);
            return ATH_EXIT_OK;
        }
        if (!strcmp(o, "--no-think")) { think = false; continue; }
        if (!strcmp(o, "--quiet")) { quiet = true; continue; }
        if (!strcmp(o, "--server-sampling")) { server_sampling = true; continue; }
        if (o[0] == '-' && !v)
            return usage_error("missing value for", o);
        if (!strcmp(o, "--host")) host = v;
        else if (!strcmp(o, "--port") && num(v, 0, 65535, &x)) port = (int)x;
        else if (!strcmp(o, "--name")) name = v;
        else if (!strcmp(o, "--dump")) dump = v;
        else if (!strcmp(o, "--ctx") && num(v, 64, 1 << 24, &x)) go.ctx = (int32_t)x;
        else if (!strcmp(o, "--experts") && num(v, 0, 1e6, &x)) go.experts = (int32_t)x;
        else if (!strcmp(o, "--threads") && num(v, 1, 4096, &x)) go.threads = (int32_t)x;
        else if (!strcmp(o, "--max-tokens") && num(v, 1, 1e9, &x)) max_tokens = (int32_t)x;
        else if (!strcmp(o, "--chunk") && num(v, 1, 1 << 20, &x)) go.chunk = (int32_t)x;
        else if (!strcmp(o, "--seed") && num(v, 0, 1.8e19, &x)) go.seed = (uint64_t)x, seed_set = true;
        else if (!strcmp(o, "--device") && (!strcmp(v, "cpu") || !strcmp(v, "gpu"))) device = v;
        else if (o[0] == '-')
            return usage_error("bad option or value:", o);
        else if (!dir) {
            dir = o;
            continue;
        } else
            return usage_error("unexpected argument", o);
        i++;
    }
    if (!dir)
        return usage_error("missing <model_dir>", NULL);
    if (!seed_set)
        go.seed = (uint64_t)time(NULL);
    go.cpu = device && !strcmp(device, "cpu");
    if (!name) {
        size_t n = strlen(dir);
        while (n > 1 && dir[n - 1] == '/')
            n--;
        const char *s = dir + n;
        while (s > dir && s[-1] != '/')
            s--;
        name = ath_arena_strndup(a, s, (size_t)(dir + n - s));
    }

    ath_http_server srv;
    if (ath_http_listen(&srv, host, port) != ATH_OK)
        return ATH_EXIT_FAILURE;
    ath_gen *g = ath_arena_zalloc(a, sizeof *g, _Alignof(ath_gen));
    if (!g) {
        ath_http_close(&srv);
        return ATH_EXIT_FAILURE;
    }
    fprintf(stderr, "athanor serve: loading %s\n", dir);
    ath_status st = ath_gen_open(g, dir, &go, a);
    if (st != ATH_OK) {
        fprintf(stderr, "athanor serve: cannot open %s: %s\n", dir, ath_status_str(st));
        ath_http_close(&srv);
        return ATH_EXIT_FAILURE;
    }
    if (device && !strcmp(device, "gpu") && !g->engine) {
        fprintf(stderr, "athanor serve: no GPU\n");
        ath_gen_close(g);
        ath_http_close(&srv);
        return ATH_EXIT_FAILURE;
    }
    const ath_api_defaults d = {
        .model = name,
        .think = think,
        .sampler_think = {0.6f, 20, 0.95f, 0},
        .sampler_plain = {0.7f, 20, 0.8f, 0},
        .presence_think = 0,
        .presence_plain = 1.5f,
        .max_tokens = max_tokens,
        .client_sampling = !server_sampling,
    };
    ath_api api;
    ath_api_init(&api, g, &d);
    api.log = quiet ? NULL : stderr;
    api.dump = dump;

    struct sigaction sa = {.sa_handler = on_signal};
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    signal(SIGPIPE, SIG_IGN);
    fprintf(stderr, "athanor serve: %s on %s, context %d; listening on http://%s:%d (Ctrl-C stops)\n", name,
            g->engine ? g->engine->dev.name : "the CPU", go.ctx, host, srv.port);
    st = ath_http_run(&srv, ath_api_handle, &api, &stop_flag);
    fprintf(stderr, "athanor serve: stopped\n");
    ath_gen_close(g);
    ath_http_close(&srv);
    return st == ATH_OK ? ATH_EXIT_OK : ATH_EXIT_FAILURE;
}
