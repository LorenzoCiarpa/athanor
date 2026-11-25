/* api.c: the HTTP APIs of `athanor serve`. See api.h. */
#include "net/api.h"

#include "core/buf.h"

#include <stdio.h>
#include <string.h>
#include <time.h>

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

void ath_api_init(ath_api *api, ath_gen *gen, const ath_api_defaults *d)
{
    memset(api, 0, sizeof *api);
    api->gen = gen;
    api->d = *d;
    api->started = (int64_t)time(NULL);
}

/* A unique id: prefix and 24 hex digits. */
static void new_id(ath_api *api, const char *prefix, char *out, size_t cap)
{
    uint64_t x = ++api->n_ids * 0x9E3779B97F4A7C15ull + (uint64_t)api->started;
    uint64_t y = x ^ 0xD1B54A32D192ED03ull;
    for (int i = 0; i < 2; i++) {
        uint64_t *z = i ? &y : &x;
        *z ^= *z >> 30, *z *= 0xBF58476D1CE4E5B9ull;
        *z ^= *z >> 27, *z *= 0x94D049BB133111EBull;
        *z ^= *z >> 31;
    }
    snprintf(out, cap, "%s%016llx%08llx", prefix, (unsigned long long)x, (unsigned long long)(y & 0xffffffffu));
}

/* ---- JSON in ---------------------------------------------------------------------- */

static const char *jstr(const ath_json *o, const char *key)
{
    const char *s;
    return ath_json_to_str(ath_json_get(o, key), &s, NULL) == ATH_OK ? s : NULL;
}

static bool is(const char *s, const char *t)
{
    return s && !strcmp(s, t);
}

static const char *IMAGE_NOTE = "[an image was here; this server runs the model on text only]";

/* The text of a content value: a string, or an array of parts whose texts
 * are joined as the chat template joins them (directly); images become a
 * note. */
static const char *content_text(const ath_json *c, ath_arena *a)
{
    if (!c || c->type == ATH_JSON_NULL)
        return "";
    if (c->type == ATH_JSON_STRING)
        return c->u.str.ptr;
    if (c->type != ATH_JSON_ARRAY)
        return "";
    ath_buf b;
    ath_buf_init(&b, a);
    for (const ath_json *p = ath_json_first(c); p; p = p->next) {
        if (p->type == ATH_JSON_STRING) {
            ath_buf_append(&b, p->u.str.ptr, p->u.str.len);
            continue;
        }
        const char *ty = jstr(p, "type"), *t = jstr(p, "text");
        if (t)
            ath_buf_puts(&b, t);
        else if (is(ty, "refusal") && jstr(p, "refusal"))
            ath_buf_puts(&b, jstr(p, "refusal"));
        else if (ty && (strstr(ty, "image") || strstr(ty, "document") || strstr(ty, "file")))
            ath_buf_puts(&b, IMAGE_NOTE);
    }
    return ath_buf_str(&b);
}

/* ---- the request being built ------------------------------------------------------ */

typedef struct {
    ath_msg      *v;
    int32_t       n, cap;
    ath_buf       system;    /* system and developer messages before any other    */
    ath_tool_def *tools;
    int32_t       n_tools, cap_tools;
    ath_arena    *a;
} builder;

/* A new message at the end; its index (the array may move). */
static int32_t push(builder *b, ath_msg_role role, const char *content)
{
    if (b->n == b->cap) {
        int32_t cap = b->cap ? 2 * b->cap : 16;
        ath_msg *v = ath_arena_alloc(b->a, (size_t)cap * sizeof *v, _Alignof(ath_msg));
        if (!v)
            return -1;
        if (b->n)
            memcpy(v, b->v, (size_t)b->n * sizeof *v);
        b->v = v, b->cap = cap;
    }
    b->v[b->n] = (ath_msg){.role = role, .content = content ? content : ""};
    return b->n++;
}

/* A system or developer message: part of the system message while nothing
 * else came before it, a user message after. */
static int32_t add_system(builder *b, const char *text)
{
    if (b->n == 0) {
        if (text && *text) {
            if (b->system.len)
                ath_buf_puts(&b->system, "\n\n");
            ath_buf_puts(&b->system, text);
        }
        return 0;
    }
    return push(b, ATH_MSG_USER, text);
}

/* The arguments of a call in the history: a JSON object as text. */
static const char *object_text(const char *args, ath_arena *a)
{
    const ath_json *j;
    if (!args || !*args)
        return "{}";
    if (ath_json_parse(a, args, strlen(args), &j, NULL) == ATH_OK && j->type == ATH_JSON_OBJECT)
        return args;
    ath_log(ATH_LOG_WARN, "serve: the arguments of a call in the history are not a JSON object; sent as {}");
    return "{}";
}

static const char *json_text(const ath_json *v, ath_arena *a)
{
    ath_buf b;
    ath_buf_init(&b, a);
    if (!v || ath_json_write(v, &b) != ATH_OK)
        return NULL;
    return ath_buf_str(&b);
}

static ath_status add_call(builder *b, int32_t m, const char *id, const char *name, const char *args)
{
    ath_msg *msg = &b->v[m];
    ath_tool_call *c = ath_arena_alloc(b->a, (size_t)(msg->n_tool_calls + 1) * sizeof *c, _Alignof(ath_tool_call));
    if (!c)
        return ATH_ERR_NOMEM;
    if (msg->n_tool_calls)
        memcpy(c, msg->tool_calls, (size_t)msg->n_tool_calls * sizeof *c);
    c[msg->n_tool_calls] = (ath_tool_call){.id = id, .name = name ? name : "", .arguments = args};
    msg->tool_calls = c;
    msg->n_tool_calls++;
    return ATH_OK;
}

static ath_status add_tool(builder *b, const char *name, const char *desc, const char *schema)
{
    if (!name || !*name)
        return ATH_OK;
    if (b->n_tools == b->cap_tools) {
        int32_t cap = b->cap_tools ? 2 * b->cap_tools : 16;
        ath_tool_def *v = ath_arena_alloc(b->a, (size_t)cap * sizeof *v, _Alignof(ath_tool_def));
        if (!v)
            return ATH_ERR_NOMEM;
        if (b->n_tools)
            memcpy(v, b->tools, (size_t)b->n_tools * sizeof *v);
        b->tools = v, b->cap_tools = cap;
    }
    b->tools[b->n_tools++] = (ath_tool_def){.name = name, .description = desc, .json_schema = schema};
    return ATH_OK;
}

/* The messages, the system message first. */
static ath_status finish_messages(builder *b, ath_gen_request *g)
{
    int32_t n = b->n + (b->system.len ? 1 : 0);
    ath_msg *v = ath_arena_alloc(b->a, (size_t)(n ? n : 1) * sizeof *v, _Alignof(ath_msg));
    if (!v)
        return ATH_ERR_NOMEM;
    int32_t k = 0;
    if (b->system.len)
        v[k++] = (ath_msg){.role = ATH_MSG_SYSTEM, .content = ath_buf_str(&b->system)};
    if (b->n)
        memcpy(v + k, b->v, (size_t)b->n * sizeof *v);
    g->msgs = v;
    g->n_msgs = n;
    g->tools = b->tools;
    g->n_tools = b->n_tools;
    return ATH_OK;
}

/* ---- sampling, limits, stop strings (every API) ------------------------------------ */

static ath_status sampling(const ath_api *api, const ath_json *body, ath_gen_request *g, const char **err)
{
    const ath_api_defaults *d = &api->d;
    g->sampler = g->think ? d->sampler_think : d->sampler_plain;
    g->presence_penalty = g->think ? d->presence_think : d->presence_plain;
    if (!d->client_sampling)
        return ATH_OK;
    double x;
    if (ath_json_to_double(ath_json_get(body, "temperature"), &x) == ATH_OK) {
        if (!(x >= 0 && x <= 100))
            return *err = "temperature: must be at least 0", ATH_ERR_INVALID_ARG;
        g->sampler.temperature = (float)x;
    }
    if (ath_json_to_double(ath_json_get(body, "top_p"), &x) == ATH_OK) {
        if (!(x > 0 && x <= 1))
            return *err = "top_p: must be in (0, 1]", ATH_ERR_INVALID_ARG;
        g->sampler.top_p = (float)x;
    }
    if (ath_json_to_double(ath_json_get(body, "top_k"), &x) == ATH_OK) {
        if (!(x >= 0 && x <= 1e9))
            return *err = "top_k: must be at least 0", ATH_ERR_INVALID_ARG;
        g->sampler.top_k = (int32_t)x;
    }
    if (ath_json_to_double(ath_json_get(body, "min_p"), &x) == ATH_OK) {
        if (!(x >= 0 && x <= 1))
            return *err = "min_p: must be in [0, 1]", ATH_ERR_INVALID_ARG;
        g->sampler.min_p = (float)x;
    }
    if (ath_json_to_double(ath_json_get(body, "presence_penalty"), &x) == ATH_OK) {
        if (!(x >= -2 && x <= 2))
            return *err = "presence_penalty: must be in [-2, 2]", ATH_ERR_INVALID_ARG;
        g->presence_penalty = (float)x;
    }
    return ATH_OK;
}

static void max_tokens(const ath_api *api, const ath_json *body, const char *key, const char *key2,
                       ath_gen_request *g)
{
    int64_t n;
    g->max_tokens = api->d.max_tokens;
    if (ath_json_to_int(ath_json_get(body, key), &n) == ATH_OK ||
        (key2 && ath_json_to_int(ath_json_get(body, key2), &n) == ATH_OK))
        g->max_tokens = n > 0 && n < (1 << 30) ? (int32_t)n : 0;
}

static ath_status stops(const ath_json *v, ath_gen_request *g, const char **err)
{
    if (!v || v->type == ATH_JSON_NULL)
        return ATH_OK;
    if (v->type == ATH_JSON_STRING) {
        g->stop[g->n_stop++] = v->u.str.ptr;
        return ATH_OK;
    }
    if (v->type != ATH_JSON_ARRAY)
        return *err = "stop: a string or an array of strings", ATH_ERR_INVALID_ARG;
    for (const ath_json *s = ath_json_first(v); s; s = s->next) {
        if (s->type != ATH_JSON_STRING)
            return *err = "stop: a string or an array of strings", ATH_ERR_INVALID_ARG;
        if (g->n_stop == ATH_GEN_MAX_STOP)
            return *err = "stop: at most 8 strings", ATH_ERR_INVALID_ARG;
        if (s->u.str.len)
            g->stop[g->n_stop++] = s->u.str.ptr;
    }
    return ATH_OK;
}

static bool flag(const ath_json *o, const char *key)
{
    bool b = false;
    return ath_json_to_bool(ath_json_get(o, key), &b) == ATH_OK && b;
}

/* ---- Anthropic Messages ------------------------------------------------------------ */

static ath_status parse_anthropic(const ath_api *api, const ath_json *body, ath_api_request *r, builder *b,
                                  const char **err)
{
    ath_arena *a = b->a;
    const ath_json *sys = ath_json_get(body, "system");
    if (sys && sys->type == ATH_JSON_STRING)
        add_system(b, sys->u.str.ptr);
    else if (sys && sys->type == ATH_JSON_ARRAY)
        for (const ath_json *s = ath_json_first(sys); s; s = s->next)
            add_system(b, content_text(s->type == ATH_JSON_STRING ? s : ath_json_get(s, "text"), a));

    const ath_json *msgs = ath_json_get(body, "messages");
    if (!msgs || msgs->type != ATH_JSON_ARRAY)
        return *err = "messages: an array is required", ATH_ERR_INVALID_ARG;
    for (const ath_json *m = ath_json_first(msgs); m; m = m->next) {
        const char *role = jstr(m, "role");
        const ath_json *c = ath_json_get(m, "content");
        if (is(role, "user")) {
            if (!c || c->type != ATH_JSON_ARRAY) {
                push(b, ATH_MSG_USER, content_text(c, a));
                continue;
            }
            /* tool results first, as their own messages; the text after them */
            ath_buf text;
            ath_buf_init(&text, a);
            bool any_text = false;
            for (const ath_json *k = ath_json_first(c); k; k = k->next) {
                const char *ty = jstr(k, "type");
                if (is(ty, "tool_result")) {
                    int32_t i = push(b, ATH_MSG_TOOL, content_text(ath_json_get(k, "content"), a));
                    if (i >= 0)
                        b->v[i].tool_call_id = jstr(k, "tool_use_id");
                } else if (is(ty, "text") || (ty && (strstr(ty, "image") || strstr(ty, "document")))) {
                    ath_buf_puts(&text, is(ty, "text") ? (jstr(k, "text") ? jstr(k, "text") : "") : IMAGE_NOTE);
                    any_text = true;
                }
            }
            if (any_text)
                push(b, ATH_MSG_USER, ath_buf_str(&text));
        } else if (is(role, "assistant")) {
            if (!c || c->type != ATH_JSON_ARRAY) {
                push(b, ATH_MSG_ASSISTANT, content_text(c, a));
                continue;
            }
            ath_buf text, think;
            ath_buf_init(&text, a);
            ath_buf_init(&think, a);
            int32_t i = push(b, ATH_MSG_ASSISTANT, "");
            if (i < 0)
                return ATH_ERR_NOMEM;
            for (const ath_json *k = ath_json_first(c); k; k = k->next) {
                const char *ty = jstr(k, "type");
                if (is(ty, "text") && jstr(k, "text"))
                    ath_buf_puts(&text, jstr(k, "text"));
                else if (is(ty, "thinking") && jstr(k, "thinking"))
                    ath_buf_puts(&think, jstr(k, "thinking"));
                else if (is(ty, "tool_use")) {
                    const ath_json *in = ath_json_get(k, "input");
                    const char *args = in && in->type == ATH_JSON_OBJECT ? json_text(in, a) : "{}";
                    ATH_CHECK(add_call(b, i, jstr(k, "id"), jstr(k, "name"), args));
                }
            }
            b->v[i].content = ath_buf_str(&text);
            b->v[i].reasoning = think.len ? ath_buf_str(&think) : NULL;
        } else {
            return *err = "messages: each role must be user or assistant", ATH_ERR_INVALID_ARG;
        }
    }

    const ath_json *tc = ath_json_get(body, "tool_choice");
    if (!is(jstr(tc, "type"), "none"))
        for (const ath_json *t = ath_json_first(ath_json_get(body, "tools")); t; t = t->next) {
            const char *ty = jstr(t, "type");
            if (ty && !is(ty, "custom"))
                continue;   /* a server tool (web search...): not a function the model calls */
            ATH_CHECK(add_tool(b, jstr(t, "name"), jstr(t, "description"),
                               json_text(ath_json_get(t, "input_schema"), a)));
        }
    const char *thinking = jstr(ath_json_get(body, "thinking"), "type");
    r->g.think = is(thinking, "enabled") || is(thinking, "adaptive") ? true
                 : is(thinking, "disabled")                         ? false
                                                                    : api->d.think;
    max_tokens(api, body, "max_tokens", NULL, &r->g);
    ATH_CHECK(stops(ath_json_get(body, "stop_sequences"), &r->g, err));
    r->stream = flag(body, "stream");
    return sampling(api, body, &r->g, err);
}

/* ---- OpenAI Chat Completions ------------------------------------------------------- */

static ath_status parse_chat(const ath_api *api, const ath_json *body, ath_api_request *r, builder *b,
                             const char **err)
{
    ath_arena *a = b->a;
    const ath_json *msgs = ath_json_get(body, "messages");
    if (!msgs || msgs->type != ATH_JSON_ARRAY)
        return *err = "messages: an array is required", ATH_ERR_INVALID_ARG;
    for (const ath_json *m = ath_json_first(msgs); m; m = m->next) {
        const char *role = jstr(m, "role");
        const char *content = content_text(ath_json_get(m, "content"), a);
        if (is(role, "system") || is(role, "developer")) {
            add_system(b, content);
        } else if (is(role, "user")) {
            push(b, ATH_MSG_USER, content);
        } else if (is(role, "assistant")) {
            int32_t i = push(b, ATH_MSG_ASSISTANT, content);
            if (i < 0)
                return ATH_ERR_NOMEM;
            const char *rs = jstr(m, "reasoning_content") ? jstr(m, "reasoning_content") : jstr(m, "reasoning");
            b->v[i].reasoning = rs && *rs ? rs : NULL;
            for (const ath_json *t = ath_json_first(ath_json_get(m, "tool_calls")); t; t = t->next) {
                const ath_json *fn = ath_json_get(t, "function");
                const ath_json *args = ath_json_get(fn, "arguments");
                const char *at = args && args->type == ATH_JSON_STRING ? object_text(args->u.str.ptr, a)
                                 : args && args->type == ATH_JSON_OBJECT ? json_text(args, a)
                                                                         : "{}";
                ATH_CHECK(add_call(b, i, jstr(t, "id"), jstr(fn, "name"), at));
            }
        } else if (is(role, "tool") || is(role, "function")) {
            int32_t i = push(b, ATH_MSG_TOOL, content);
            if (i >= 0)
                b->v[i].tool_call_id = jstr(m, "tool_call_id");
        } else {
            return *err = "messages: unknown role", ATH_ERR_INVALID_ARG;
        }
    }
    if (!is(jstr(body, "tool_choice"), "none"))
        for (const ath_json *t = ath_json_first(ath_json_get(body, "tools")); t; t = t->next) {
            if (jstr(t, "type") && !is(jstr(t, "type"), "function"))
                continue;
            const ath_json *fn = ath_json_get(t, "function");
            ATH_CHECK(add_tool(b, jstr(fn, "name"), jstr(fn, "description"),
                               json_text(ath_json_get(fn, "parameters"), a)));
        }
    /* thinking: chat_template_kwargs.enable_thinking (vLLM, SGLang), then
     * reasoning_effort ("none" turns it off), then the server's default */
    bool on;
    const char *effort = jstr(body, "reasoning_effort");
    if (ath_json_to_bool(ath_json_get(ath_json_get(body, "chat_template_kwargs"), "enable_thinking"), &on) == ATH_OK)
        r->g.think = on;
    else if (effort)
        r->g.think = !is(effort, "none");
    else
        r->g.think = api->d.think;
    max_tokens(api, body, "max_completion_tokens", "max_tokens", &r->g);
    ATH_CHECK(stops(ath_json_get(body, "stop"), &r->g, err));
    r->stream = flag(body, "stream");
    r->include_usage = flag(ath_json_get(body, "stream_options"), "include_usage");
    return sampling(api, body, &r->g, err);
}

/* ---- OpenAI Responses -------------------------------------------------------------- */

/* A reasoning item's text: its content (reasoning_text), else its summary. */
static const char *reasoning_text(const ath_json *item, ath_arena *a)
{
    ath_buf b;
    ath_buf_init(&b, a);
    for (const ath_json *p = ath_json_first(ath_json_get(item, "content")); p; p = p->next)
        if (jstr(p, "text"))
            ath_buf_puts(&b, jstr(p, "text"));
    if (!b.len)
        for (const ath_json *p = ath_json_first(ath_json_get(item, "summary")); p; p = p->next)
            if (jstr(p, "text")) {
                if (b.len)
                    ath_buf_puts(&b, "\n\n");
                ath_buf_puts(&b, jstr(p, "text"));
            }
    return ath_buf_str(&b);
}

static bool empty(const char *s)
{
    return !s || !*s;
}

static ath_status parse_responses(const ath_api *api, const ath_json *body, ath_api_request *r, builder *b,
                                  const char **err)
{
    ath_arena *a = b->a;
    if (jstr(body, "previous_response_id"))
        return *err = "previous_response_id: this server keeps no responses; send the whole input (store: false)",
               ATH_ERR_INVALID_ARG;
    if (jstr(body, "instructions"))
        add_system(b, jstr(body, "instructions"));
    const ath_json *input = ath_json_get(body, "input");
    if (input && input->type == ATH_JSON_STRING) {
        push(b, ATH_MSG_USER, input->u.str.ptr);
    } else if (input && input->type == ATH_JSON_ARRAY) {
        int32_t open = -1;   /* the assistant message the next items of its turn join */
        for (const ath_json *it = ath_json_first(input); it; it = it->next) {
            const char *ty = jstr(it, "type");
            const char *role = jstr(it, "role");
            if (is(ty, "message") || (!ty && role)) {
                const char *text = content_text(ath_json_get(it, "content"), a);
                if (is(role, "assistant")) {
                    if (open >= 0 && !b->v[open].n_tool_calls && empty(b->v[open].content)) {
                        b->v[open].content = text;
                    } else if (open >= 0 && !b->v[open].n_tool_calls) {
                        ath_buf j;
                        ath_buf_init(&j, a);
                        ath_buf_puts(&j, b->v[open].content);
                        ath_buf_puts(&j, text);
                        b->v[open].content = ath_buf_str(&j);
                    } else {
                        open = push(b, ATH_MSG_ASSISTANT, text);
                    }
                } else {
                    open = -1;
                    if (is(role, "system") || is(role, "developer"))
                        add_system(b, text);
                    else
                        push(b, ATH_MSG_USER, text);
                }
            } else if (is(ty, "reasoning")) {
                const char *text = reasoning_text(it, a);
                if (open >= 0 && !b->v[open].n_tool_calls && empty(b->v[open].content) &&
                    empty(b->v[open].reasoning))
                    b->v[open].reasoning = text;
                else {
                    open = push(b, ATH_MSG_ASSISTANT, "");
                    if (open >= 0)
                        b->v[open].reasoning = text;
                }
            } else if (is(ty, "function_call") || is(ty, "custom_tool_call")) {
                if (open < 0)
                    open = push(b, ATH_MSG_ASSISTANT, "");
                if (open < 0)
                    return ATH_ERR_NOMEM;
                const char *args;
                if (is(ty, "function_call")) {
                    args = object_text(jstr(it, "arguments"), a);
                } else {
                    ath_buf j;
                    ath_buf_init(&j, a);
                    ath_buf_puts(&j, "{\"input\": ");
                    const char *in = jstr(it, "input") ? jstr(it, "input") : "";
                    ath_json_write_str(&j, in, strlen(in));
                    ath_buf_putc(&j, '}');
                    args = ath_buf_str(&j);
                }
                ATH_CHECK(add_call(b, open, jstr(it, "call_id"), jstr(it, "name"), args));
            } else if (is(ty, "function_call_output") || is(ty, "custom_tool_call_output")) {
                open = -1;
                int32_t i = push(b, ATH_MSG_TOOL, content_text(ath_json_get(it, "output"), a));
                if (i >= 0)
                    b->v[i].tool_call_id = jstr(it, "call_id");
            }
            /* other items (references, server tool calls) carry nothing the model reads */
        }
    } else {
        return *err = "input: a string or an array of items is required", ATH_ERR_INVALID_ARG;
    }

    if (!is(jstr(body, "tool_choice"), "none"))
        for (const ath_json *t = ath_json_first(ath_json_get(body, "tools")); t; t = t->next) {
            const char *ty = jstr(t, "type"), *name = jstr(t, "name");
            if (is(ty, "function")) {
                ATH_CHECK(add_tool(b, name, jstr(t, "description"), json_text(ath_json_get(t, "parameters"), a)));
            } else if (is(ty, "custom") && name) {
                /* freeform input: one string parameter */
                ath_buf s;
                ath_buf_init(&s, a);
                ath_buf_puts(&s, "{\"type\": \"object\", \"properties\": {\"input\": {\"type\": \"string\", "
                                 "\"description\": ");
                const char *fd = "The tool's input, as free text";
                const ath_json *def = ath_json_get(ath_json_get(t, "format"), "definition");
                ath_buf d;
                ath_buf_init(&d, a);
                ath_buf_puts(&d, fd);
                if (def && def->type == ATH_JSON_STRING) {
                    ath_buf_puts(&d, ", following this grammar:\n");
                    ath_buf_puts(&d, def->u.str.ptr);
                }
                ath_json_write_str(&s, d.data, d.len);
                ath_buf_puts(&s, "}}, \"required\": [\"input\"]}");
                ATH_CHECK(add_tool(b, name, jstr(t, "description"), ath_buf_str(&s)));
                const char **c = ath_arena_alloc(a, (size_t)(r->n_custom + 1) * sizeof *c, _Alignof(const char *));
                if (!c)
                    return ATH_ERR_NOMEM;
                if (r->n_custom)
                    memcpy(c, r->custom, (size_t)r->n_custom * sizeof *c);
                c[r->n_custom++] = name;
                r->custom = c;
            }
        }
    const char *effort = jstr(ath_json_get(body, "reasoning"), "effort");
    r->g.think = effort ? !(is(effort, "none") || is(effort, "minimal")) : api->d.think;
    max_tokens(api, body, "max_output_tokens", NULL, &r->g);
    r->stream = flag(body, "stream");
    return sampling(api, body, &r->g, err);
}

ath_status ath_api_parse(const ath_api *api, ath_api_kind kind, const ath_json *body, ath_api_request *r,
                         ath_arena *a, const char **err)
{
    memset(r, 0, sizeof *r);
    r->kind = kind;
    *err = NULL;
    if (!body || body->type != ATH_JSON_OBJECT)
        return *err = "the body must be a JSON object", ATH_ERR_INVALID_ARG;
    r->model = jstr(body, "model") ? jstr(body, "model") : api->d.model;
    builder b = {.a = a};
    ath_buf_init(&b.system, a);
    ath_status st = kind == ATH_API_ANTHROPIC ? parse_anthropic(api, body, r, &b, err)
                    : kind == ATH_API_CHAT    ? parse_chat(api, body, r, &b, err)
                                              : parse_responses(api, body, r, &b, err);
    if (st == ATH_OK && b.n == 0)
        return *err = "no messages", ATH_ERR_INVALID_ARG;
    if (st == ATH_OK)
        st = finish_messages(&b, &r->g);
    if (st != ATH_OK && !*err)
        *err = ath_status_str(st);
    return st;
}

/* ---- writing back ------------------------------------------------------------------ */

typedef enum { B_NONE, B_TEXT, B_THINK, B_CALL } block_kind;

typedef struct {
    ath_api               *api;
    const ath_api_request *r;
    ath_http_out          *out;
    ath_arena             *a;
    char                   id[64];
    int64_t                created;
    ath_gen_result         res;       /* the prompt's counts, from begin                   */
    bool                   begun;     /* the stream's first events are out                 */
    block_kind             block;     /* the open block                                    */
    int32_t                index;     /* its index: blocks (output items) before it        */
    int32_t                n_calls;
    char                   item_id[64], call_id[64];
    const char            *call_name;
    bool                   custom;    /* responses: the open call is to a custom tool      */
    ath_buf                acc;       /* the open block's bytes                            */
    ath_buf                items;     /* closed blocks as JSON, comma-separated            */
    ath_buf                text, think, calls;   /* chat: the whole message                */
    ath_buf                e;         /* the event being written                           */
    int64_t                seq;       /* responses: sequence numbers                       */
    uint32_t               ticks;
    double                 t_begin, t_first;
} writer;

static void put(ath_buf *b, const char *s)
{
    ath_json_write_str(b, s ? s : "", s ? strlen(s) : 0);
}

/* Start an event of type (its JSON object stays open). */
static ath_buf *ev(writer *w, const char *type)
{
    ath_buf_clear(&w->e);
    if (w->r->kind == ATH_API_CHAT) {
        ath_buf_printf(&w->e, "{\"id\": \"%s\", \"object\": \"chat.completion.chunk\", \"created\": %lld, \"model\": ",
                       w->id, (long long)w->created);
        put(&w->e, w->r->model);
        return &w->e;
    }
    ath_buf_puts(&w->e, "{\"type\": ");
    put(&w->e, type);
    if (w->r->kind == ATH_API_RESPONSES)
        ath_buf_printf(&w->e, ", \"sequence_number\": %lld", (long long)w->seq++);
    return &w->e;
}

/* Close the event and send it. */
static bool send_ev(writer *w, const char *type)
{
    ath_buf_putc(&w->e, '}');
    ath_http_sse(w->out, w->r->kind == ATH_API_CHAT ? NULL : type, w->e.data, w->e.len);
    return !w->out->broken;
}

/* A chat chunk with one choice whose delta is the JSON object text d. */
static bool chat_delta(writer *w, const char *d, const char *finish)
{
    ath_buf *b = ev(w, NULL);
    ath_buf_printf(b, ", \"choices\": [{\"index\": 0, \"delta\": %s, \"finish_reason\": ", d);
    if (finish)
        put(b, finish);
    else
        ath_buf_puts(b, "null");
    ath_buf_puts(b, "}]");
    return send_ev(w, NULL);
}

static void usage_json(const writer *w, ath_buf *b, int32_t out_tokens)
{
    const int32_t p = w->res.prompt_tokens, c = w->res.cached_tokens;
    switch (w->r->kind) {
    case ATH_API_ANTHROPIC:
        ath_buf_printf(b,
                       "{\"input_tokens\": %d, \"cache_creation_input_tokens\": 0, \"cache_read_input_tokens\": %d, "
                       "\"output_tokens\": %d}",
                       p - c, c, out_tokens);
        break;
    case ATH_API_CHAT:
        ath_buf_printf(b,
                       "{\"prompt_tokens\": %d, \"completion_tokens\": %d, \"total_tokens\": %d, "
                       "\"prompt_tokens_details\": {\"cached_tokens\": %d}}",
                       p, out_tokens, p + out_tokens, c);
        break;
    case ATH_API_RESPONSES:
        ath_buf_printf(b,
                       "{\"input_tokens\": %d, \"input_tokens_details\": {\"cached_tokens\": %d}, \"output_tokens\": "
                       "%d, \"output_tokens_details\": {\"reasoning_tokens\": 0}, \"total_tokens\": %d}",
                       p, c, out_tokens, p + out_tokens);
        break;
    }
}

/* The Responses object: status, its output items so far, usage or null. */
static void response_json(writer *w, ath_buf *b, const char *status, const char *incomplete, int32_t out_tokens,
                          bool with_usage, const char *error)
{
    ath_buf_printf(b, "{\"id\": \"%s\", \"object\": \"response\", \"created_at\": %lld, \"status\": \"%s\", \"model\": ",
                   w->id, (long long)w->created, status);
    put(b, w->r->model);
    ath_buf_puts(b, ", \"output\": [");
    if (w->items.len)
        ath_buf_append(b, w->items.data, w->items.len);
    ath_buf_puts(b, "], \"parallel_tool_calls\": true, \"tool_choice\": \"auto\", \"store\": false, \"error\": ");
    if (error) {
        ath_buf_puts(b, "{\"code\": \"server_error\", \"message\": ");
        put(b, error);
        ath_buf_puts(b, "}");
    } else {
        ath_buf_puts(b, "null");
    }
    ath_buf_puts(b, ", \"incomplete_details\": ");
    if (incomplete)
        ath_buf_printf(b, "{\"reason\": \"%s\"}", incomplete);
    else
        ath_buf_puts(b, "null");
    ath_buf_puts(b, ", \"usage\": ");
    if (with_usage)
        usage_json(w, b, out_tokens);
    else
        ath_buf_puts(b, "null");
    ath_buf_putc(b, '}');
}

/* Is the client still there? Checked now and then on a whole response. */
static bool alive(writer *w)
{
    if (w->out->broken)
        return false;
    if (!w->r->stream && (++w->ticks & 31) == 0)
        return ath_http_alive(w->out);
    return true;
}

static bool w_begin(void *u, const ath_gen_result *res)
{
    writer *w = u;
    w->res = *res;
    w->t_begin = now_s();
    if (!w->r->stream)
        return ath_http_alive(w->out);
    ath_http_sse_start(w->out);
    w->begun = true;
    ath_buf *b;
    switch (w->r->kind) {
    case ATH_API_ANTHROPIC:
        b = ev(w, "message_start");
        ath_buf_printf(b, ", \"message\": {\"id\": \"%s\", \"type\": \"message\", \"role\": \"assistant\", \"model\": ",
                       w->id);
        put(b, w->r->model);
        ath_buf_puts(b, ", \"content\": [], \"stop_reason\": null, \"stop_sequence\": null, \"usage\": ");
        usage_json(w, b, 0);
        ath_buf_putc(b, '}');
        send_ev(w, "message_start");
        ev(w, "ping");
        return send_ev(w, "ping");
    case ATH_API_CHAT:
        return chat_delta(w, "{\"role\": \"assistant\", \"content\": \"\"}", NULL);
    case ATH_API_RESPONSES:
        for (int k = 0; k < 2; k++) {
            const char *t = k ? "response.in_progress" : "response.created";
            b = ev(w, t);
            ath_buf_puts(b, ", \"response\": ");
            response_json(w, b, "in_progress", NULL, 0, false, NULL);
            send_ev(w, t);
        }
        return !w->out->broken;
    }
    return true;
}

static bool w_wait(void *u)
{
    writer *w = u;
    if (!w->r->stream)
        return ath_http_alive(w->out);
    if (w->r->kind == ATH_API_ANTHROPIC) {
        ev(w, "ping");
        return send_ev(w, "ping");
    }
    ath_http_sse_ping(w->out);
    return !w->out->broken;
}

/* The JSON of the open block, as the whole response (and Responses' done
 * events) give it. */
static void block_json(writer *w, ath_buf *b)
{
    const char *s = w->acc.data ? w->acc.data : "";
    const size_t n = w->acc.len;
    if (w->r->kind == ATH_API_ANTHROPIC) {
        if (w->block == B_TEXT) {
            ath_buf_puts(b, "{\"type\": \"text\", \"text\": ");
            ath_json_write_str(b, s, n);
            ath_buf_putc(b, '}');
        } else if (w->block == B_THINK) {
            ath_buf_puts(b, "{\"type\": \"thinking\", \"thinking\": ");
            ath_json_write_str(b, s, n);
            ath_buf_puts(b, ", \"signature\": \"athanor\"}");
        } else {
            ath_buf_printf(b, "{\"type\": \"tool_use\", \"id\": \"%s\", \"name\": ", w->call_id);
            put(b, w->call_name);
            ath_buf_puts(b, ", \"input\": ");
            if (n)
                ath_buf_append(b, s, n);
            else
                ath_buf_puts(b, "{}");
            ath_buf_putc(b, '}');
        }
        return;
    }
    /* responses */
    if (w->block == B_TEXT) {
        ath_buf_printf(b, "{\"type\": \"message\", \"id\": \"%s\", \"status\": \"completed\", \"role\": \"assistant\", "
                          "\"content\": [{\"type\": \"output_text\", \"text\": ",
                       w->item_id);
        ath_json_write_str(b, s, n);
        ath_buf_puts(b, ", \"annotations\": []}]}");
    } else if (w->block == B_THINK) {
        ath_buf_printf(b, "{\"type\": \"reasoning\", \"id\": \"%s\", \"summary\": [{\"type\": \"summary_text\", \"text\": ",
                       w->item_id);
        ath_json_write_str(b, s, n);
        ath_buf_puts(b, "}], \"content\": [{\"type\": \"reasoning_text\", \"text\": ");
        ath_json_write_str(b, s, n);
        ath_buf_puts(b, "}]}");
    } else {
        ath_buf_printf(b, "{\"type\": \"%s\", \"id\": \"%s\", \"call_id\": \"%s\", \"name\": ",
                       w->custom ? "custom_tool_call" : "function_call", w->item_id, w->call_id);
        put(b, w->call_name);
        ath_buf_puts(b, w->custom ? ", \"input\": " : ", \"arguments\": ");
        ath_json_write_str(b, s, n);
        ath_buf_puts(b, ", \"status\": \"completed\"}");
    }
}

/* A custom tool's input, from the arguments {"input": "..."} the model
 * wrote (the arguments themselves if they are not that). */
static void custom_input(writer *w)
{
    const ath_json *j, *in;
    if (w->acc.len && ath_json_parse(w->a, w->acc.data, w->acc.len, &j, NULL) == ATH_OK &&
        (in = ath_json_get(j, "input")) && in->type == ATH_JSON_STRING) {
        ath_buf_clear(&w->acc);
        ath_buf_append(&w->acc, in->u.str.ptr, in->u.str.len);
    }
}

static bool close_block(writer *w)
{
    if (w->block == B_NONE)
        return true;
    const bool stream = w->r->stream;
    ath_buf *b;
    switch (w->r->kind) {
    case ATH_API_ANTHROPIC:
        if (stream) {
            if (w->block == B_THINK) {
                b = ev(w, "content_block_delta");
                ath_buf_printf(b, ", \"index\": %d, \"delta\": {\"type\": \"signature_delta\", \"signature\": \"athanor\"}",
                               w->index);
                send_ev(w, "content_block_delta");
            }
            b = ev(w, "content_block_stop");
            ath_buf_printf(b, ", \"index\": %d", w->index);
            send_ev(w, "content_block_stop");
        } else {
            if (w->items.len)
                ath_buf_puts(&w->items, ", ");
            block_json(w, &w->items);
        }
        break;
    case ATH_API_CHAT:
        if (w->block == B_TEXT)
            ath_buf_append(&w->text, w->acc.data ? w->acc.data : "", w->acc.len);
        else if (w->block == B_THINK)
            ath_buf_append(&w->think, w->acc.data ? w->acc.data : "", w->acc.len);
        else {
            if (w->calls.len)
                ath_buf_puts(&w->calls, ", ");
            ath_buf_printf(&w->calls, "{\"id\": \"%s\", \"type\": \"function\", \"function\": {\"name\": ", w->call_id);
            put(&w->calls, w->call_name);
            ath_buf_puts(&w->calls, ", \"arguments\": ");
            ath_json_write_str(&w->calls, w->acc.len ? w->acc.data : "{}", w->acc.len ? w->acc.len : 2);
            ath_buf_puts(&w->calls, "}}");
            w->n_calls++;
        }
        break;
    case ATH_API_RESPONSES: {
        if (w->block == B_CALL && w->custom)
            custom_input(w);
        const char *s = w->acc.data ? w->acc.data : "";
        if (stream && w->block == B_TEXT) {
            b = ev(w, "response.output_text.done");
            ath_buf_printf(b, ", \"item_id\": \"%s\", \"output_index\": %d, \"content_index\": 0, \"text\": ", w->item_id,
                           w->index);
            ath_json_write_str(b, s, w->acc.len);
            ath_buf_puts(b, ", \"logprobs\": []");
            send_ev(w, "response.output_text.done");
            b = ev(w, "response.content_part.done");
            ath_buf_printf(b, ", \"item_id\": \"%s\", \"output_index\": %d, \"content_index\": 0, \"part\": "
                              "{\"type\": \"output_text\", \"text\": ",
                           w->item_id, w->index);
            ath_json_write_str(b, s, w->acc.len);
            ath_buf_puts(b, ", \"annotations\": []}");
            send_ev(w, "response.content_part.done");
        } else if (stream && w->block == B_THINK) {
            b = ev(w, "response.reasoning_summary_text.done");
            ath_buf_printf(b, ", \"item_id\": \"%s\", \"output_index\": %d, \"summary_index\": 0, \"text\": ",
                           w->item_id, w->index);
            ath_json_write_str(b, s, w->acc.len);
            send_ev(w, "response.reasoning_summary_text.done");
            b = ev(w, "response.reasoning_summary_part.done");
            ath_buf_printf(b, ", \"item_id\": \"%s\", \"output_index\": %d, \"summary_index\": 0, \"part\": "
                              "{\"type\": \"summary_text\", \"text\": ",
                           w->item_id, w->index);
            ath_json_write_str(b, s, w->acc.len);
            ath_buf_putc(b, '}');
            send_ev(w, "response.reasoning_summary_part.done");
        } else if (stream) {
            const char *t = w->custom ? "response.custom_tool_call_input.delta" : NULL;
            if (t) {   /* a custom call's input goes out whole, once known */
                b = ev(w, t);
                ath_buf_printf(b, ", \"item_id\": \"%s\", \"output_index\": %d, \"delta\": ", w->item_id, w->index);
                ath_json_write_str(b, s, w->acc.len);
                send_ev(w, t);
            }
            t = w->custom ? "response.custom_tool_call_input.done" : "response.function_call_arguments.done";
            b = ev(w, t);
            ath_buf_printf(b, ", \"item_id\": \"%s\", \"output_index\": %d, \"%s\": ", w->item_id, w->index,
                           w->custom ? "input" : "arguments");
            ath_json_write_str(b, s, w->acc.len);
            send_ev(w, t);
        }
        if (w->items.len)
            ath_buf_puts(&w->items, ", ");
        size_t at = w->items.len;
        block_json(w, &w->items);
        if (stream) {
            b = ev(w, "response.output_item.done");
            ath_buf_printf(b, ", \"output_index\": %d, \"item\": ", w->index);
            ath_buf_append(b, w->items.data + at, w->items.len - at);
            send_ev(w, "response.output_item.done");
        }
        break;
    }
    }
    w->block = B_NONE;
    w->index++;
    ath_buf_clear(&w->acc);
    return !w->out->broken;
}

static bool open_block(writer *w, block_kind k, const char *name)
{
    if (w->block == k && k != B_CALL)
        return true;
    if (!close_block(w))
        return false;
    w->block = k;
    if (k == B_CALL) {
        w->call_name = ath_arena_strndup(w->a, name, strlen(name));
        w->custom = false;
        for (int32_t i = 0; i < w->r->n_custom; i++)
            w->custom |= !strcmp(w->r->custom[i], name);
        new_id(w->api, w->r->kind == ATH_API_ANTHROPIC ? "toolu_" : "call_", w->call_id, sizeof w->call_id);
    }
    if (w->r->kind == ATH_API_RESPONSES)
        new_id(w->api, k == B_TEXT ? "msg_" : k == B_THINK ? "rs_" : w->custom ? "ctc_" : "fc_", w->item_id,
               sizeof w->item_id);
    if (!w->r->stream)
        return true;
    ath_buf *b;
    switch (w->r->kind) {
    case ATH_API_ANTHROPIC:
        b = ev(w, "content_block_start");
        ath_buf_printf(b, ", \"index\": %d, \"content_block\": ", w->index);
        if (k == B_TEXT)
            ath_buf_puts(b, "{\"type\": \"text\", \"text\": \"\"}");
        else if (k == B_THINK)
            ath_buf_puts(b, "{\"type\": \"thinking\", \"thinking\": \"\", \"signature\": \"\"}");
        else {
            ath_buf_printf(b, "{\"type\": \"tool_use\", \"id\": \"%s\", \"name\": ", w->call_id);
            put(b, name);
            ath_buf_puts(b, ", \"input\": {}}");
        }
        return send_ev(w, "content_block_start");
    case ATH_API_CHAT:
        if (k != B_CALL)
            return true;
        b = ev(w, NULL);
        ath_buf_printf(b,
                       ", \"choices\": [{\"index\": 0, \"delta\": {\"tool_calls\": [{\"index\": %d, \"id\": \"%s\", "
                       "\"type\": \"function\", \"function\": {\"name\": ",
                       w->n_calls, w->call_id);
        put(b, name);
        ath_buf_puts(b, ", \"arguments\": \"\"}}]}, \"finish_reason\": null}]");
        return send_ev(w, NULL);
    case ATH_API_RESPONSES:
        b = ev(w, "response.output_item.added");
        ath_buf_printf(b, ", \"output_index\": %d, \"item\": ", w->index);
        if (k == B_TEXT)
            ath_buf_printf(b, "{\"type\": \"message\", \"id\": \"%s\", \"status\": \"in_progress\", \"role\": "
                              "\"assistant\", \"content\": []}",
                           w->item_id);
        else if (k == B_THINK)
            ath_buf_printf(b, "{\"type\": \"reasoning\", \"id\": \"%s\", \"summary\": []}", w->item_id);
        else {
            ath_buf_printf(b, "{\"type\": \"%s\", \"id\": \"%s\", \"call_id\": \"%s\", \"name\": ",
                           w->custom ? "custom_tool_call" : "function_call", w->item_id, w->call_id);
            put(b, name);
            ath_buf_puts(b, w->custom ? ", \"input\": \"\", \"status\": \"in_progress\"}"
                                      : ", \"arguments\": \"\", \"status\": \"in_progress\"}");
        }
        send_ev(w, "response.output_item.added");
        if (k == B_TEXT) {
            b = ev(w, "response.content_part.added");
            ath_buf_printf(b, ", \"item_id\": \"%s\", \"output_index\": %d, \"content_index\": 0, \"part\": "
                              "{\"type\": \"output_text\", \"text\": \"\", \"annotations\": []}",
                           w->item_id, w->index);
            send_ev(w, "response.content_part.added");
        } else if (k == B_THINK) {
            b = ev(w, "response.reasoning_summary_part.added");
            ath_buf_printf(b, ", \"item_id\": \"%s\", \"output_index\": %d, \"summary_index\": 0, \"part\": "
                              "{\"type\": \"summary_text\", \"text\": \"\"}",
                           w->item_id, w->index);
            send_ev(w, "response.reasoning_summary_part.added");
        }
        return !w->out->broken;
    }
    return true;
}

/* Bytes of the open block: kept, and sent as a delta on a stream. */
static bool delta(writer *w, const char *s, size_t n)
{
    if (!w->t_first)
        w->t_first = now_s();
    ath_buf_append(&w->acc, s, n);
    if (!w->r->stream)
        return alive(w);
    ath_buf *b;
    switch (w->r->kind) {
    case ATH_API_ANTHROPIC:
        b = ev(w, "content_block_delta");
        ath_buf_printf(b, ", \"index\": %d, \"delta\": {\"type\": \"%s\", \"%s\": ", w->index,
                       w->block == B_TEXT ? "text_delta" : w->block == B_THINK ? "thinking_delta" : "input_json_delta",
                       w->block == B_TEXT ? "text" : w->block == B_THINK ? "thinking" : "partial_json");
        ath_json_write_str(b, s, n);
        ath_buf_putc(b, '}');
        return send_ev(w, "content_block_delta");
    case ATH_API_CHAT:
        b = ev(w, NULL);
        ath_buf_puts(b, ", \"choices\": [{\"index\": 0, \"delta\": {");
        if (w->block == B_CALL)
            ath_buf_printf(b, "\"tool_calls\": [{\"index\": %d, \"function\": {\"arguments\": ", w->n_calls);
        else
            ath_buf_puts(b, w->block == B_TEXT ? "\"content\": " : "\"reasoning_content\": ");
        ath_json_write_str(b, s, n);
        ath_buf_puts(b, w->block == B_CALL ? "}}]}, \"finish_reason\": null}]" : "}, \"finish_reason\": null}]");
        return send_ev(w, NULL);
    case ATH_API_RESPONSES: {
        if (w->block == B_CALL && w->custom)
            return true;   /* sent whole at the end */
        const char *t = w->block == B_TEXT    ? "response.output_text.delta"
                        : w->block == B_THINK ? "response.reasoning_summary_text.delta"
                                              : "response.function_call_arguments.delta";
        b = ev(w, t);
        ath_buf_printf(b, ", \"item_id\": \"%s\", \"output_index\": %d, ", w->item_id, w->index);
        if (w->block == B_TEXT)
            ath_buf_puts(b, "\"content_index\": 0, ");
        else if (w->block == B_THINK)
            ath_buf_puts(b, "\"summary_index\": 0, ");
        ath_buf_puts(b, "\"delta\": ");
        ath_json_write_str(b, s, n);
        if (w->block == B_TEXT)
            ath_buf_puts(b, ", \"logprobs\": []");
        return send_ev(w, t);
    }
    }
    return true;
}

static bool w_text(void *u, const char *s, size_t n)
{
    writer *w = u;
    return open_block(w, B_TEXT, NULL) && delta(w, s, n);
}

static bool w_thinking(void *u, const char *s, size_t n)
{
    writer *w = u;
    return open_block(w, B_THINK, NULL) && delta(w, s, n);
}

static bool w_tool_begin(void *u, const char *name)
{
    writer *w = u;
    return open_block(w, B_CALL, name);
}

static bool w_tool_args(void *u, const char *s, size_t n)
{
    writer *w = u;
    return w->block == B_CALL ? delta(w, s, n) : true;
}

static bool w_tool_end(void *u)
{
    writer *w = u;
    return close_block(w);
}

/* ---- errors ------------------------------------------------------------------------ */

/* An error as a whole response, in kind's shape. */
static void reply_error(ath_http_out *out, ath_api_kind kind, int status, const char *type, const char *code,
                        const char *msg)
{
    ath_buf b;
    ath_buf_init(&b, out->a);
    if (kind == ATH_API_ANTHROPIC) {
        ath_buf_printf(&b, "{\"type\": \"error\", \"error\": {\"type\": \"%s\", \"message\": ", type);
        put(&b, msg);
        ath_buf_puts(&b, "}}");
    } else {
        ath_buf_puts(&b, "{\"error\": {\"message\": ");
        put(&b, msg);
        ath_buf_printf(&b, ", \"type\": \"%s\", \"param\": null, \"code\": ", type);
        if (code)
            put(&b, code);
        else
            ath_buf_puts(&b, "null");
        ath_buf_puts(&b, "}}");
    }
    ath_http_reply(out, status, "application/json", b.data, b.len);
}

/* An error after the stream began. */
static void stream_error(writer *w, const char *msg)
{
    ath_buf *b;
    close_block(w);
    switch (w->r->kind) {
    case ATH_API_ANTHROPIC:
        b = ev(w, "error");
        ath_buf_puts(b, ", \"error\": {\"type\": \"api_error\", \"message\": ");
        put(b, msg);
        ath_buf_putc(b, '}');
        send_ev(w, "error");
        break;
    case ATH_API_CHAT:
        ath_buf_clear(&w->e);
        ath_buf_puts(&w->e, "{\"error\": {\"message\": ");
        put(&w->e, msg);
        ath_buf_puts(&w->e, ", \"type\": \"server_error\", \"param\": null, \"code\": null}");
        send_ev(w, NULL);
        break;
    case ATH_API_RESPONSES:
        b = ev(w, "response.failed");
        ath_buf_puts(b, ", \"response\": ");
        response_json(w, b, "failed", NULL, 0, false, msg);
        send_ev(w, "response.failed");
        break;
    }
}

/* ---- the end of a response ---------------------------------------------------------- */

static const char *anthropic_stop(ath_gen_finish f)
{
    return f == ATH_GEN_TOOL_CALLS ? "tool_use" : f == ATH_GEN_LENGTH ? "max_tokens" : f == ATH_GEN_STOP ? "stop_sequence" : "end_turn";
}

static const char *chat_finish(ath_gen_finish f)
{
    return f == ATH_GEN_TOOL_CALLS ? "tool_calls" : f == ATH_GEN_LENGTH ? "length" : "stop";
}

static void finish(writer *w, const ath_gen_result *res)
{
    close_block(w);
    const ath_api_request *r = w->r;
    const char *stop_seq = res->finish == ATH_GEN_STOP && res->stop_index >= 0 ? r->g.stop[res->stop_index] : NULL;
    ath_buf body;
    ath_buf_init(&body, w->a);
    ath_buf *b;
    switch (r->kind) {
    case ATH_API_ANTHROPIC:
        if (r->stream) {
            b = ev(w, "message_delta");
            ath_buf_printf(b, ", \"delta\": {\"stop_reason\": \"%s\", \"stop_sequence\": ", anthropic_stop(res->finish));
            if (stop_seq)
                put(b, stop_seq);
            else
                ath_buf_puts(b, "null");
            ath_buf_puts(b, "}, \"usage\": ");
            usage_json(w, b, res->output_tokens);
            send_ev(w, "message_delta");
            ev(w, "message_stop");
            send_ev(w, "message_stop");
            return;
        }
        ath_buf_printf(&body, "{\"id\": \"%s\", \"type\": \"message\", \"role\": \"assistant\", \"model\": ", w->id);
        put(&body, r->model);
        ath_buf_puts(&body, ", \"content\": [");
        if (w->items.len)
            ath_buf_append(&body, w->items.data, w->items.len);
        ath_buf_printf(&body, "], \"stop_reason\": \"%s\", \"stop_sequence\": ", anthropic_stop(res->finish));
        if (stop_seq)
            put(&body, stop_seq);
        else
            ath_buf_puts(&body, "null");
        ath_buf_puts(&body, ", \"usage\": ");
        usage_json(w, &body, res->output_tokens);
        ath_buf_putc(&body, '}');
        break;
    case ATH_API_CHAT:
        if (r->stream) {
            chat_delta(w, "{}", chat_finish(res->finish));
            if (r->include_usage) {
                b = ev(w, NULL);
                ath_buf_puts(b, ", \"choices\": [], \"usage\": ");
                usage_json(w, b, res->output_tokens);
                send_ev(w, NULL);
            }
            ath_http_sse(w->out, NULL, "[DONE]", 6);
            return;
        }
        ath_buf_printf(&body, "{\"id\": \"%s\", \"object\": \"chat.completion\", \"created\": %lld, \"model\": ", w->id,
                       (long long)w->created);
        put(&body, r->model);
        ath_buf_puts(&body, ", \"choices\": [{\"index\": 0, \"message\": {\"role\": \"assistant\", \"content\": ");
        if (w->text.len || !w->calls.len)
            ath_json_write_str(&body, w->text.data ? w->text.data : "", w->text.len);
        else
            ath_buf_puts(&body, "null");
        if (w->think.len) {
            ath_buf_puts(&body, ", \"reasoning_content\": ");
            ath_json_write_str(&body, w->think.data, w->think.len);
        }
        if (w->calls.len) {
            ath_buf_puts(&body, ", \"tool_calls\": [");
            ath_buf_append(&body, w->calls.data, w->calls.len);
            ath_buf_putc(&body, ']');
        }
        ath_buf_printf(&body, "}, \"finish_reason\": \"%s\"}], \"usage\": ", chat_finish(res->finish));
        usage_json(w, &body, res->output_tokens);
        ath_buf_putc(&body, '}');
        break;
    case ATH_API_RESPONSES: {
        const bool cut = res->finish == ATH_GEN_LENGTH;
        if (r->stream) {
            const char *t = cut ? "response.incomplete" : "response.completed";
            b = ev(w, t);
            ath_buf_puts(b, ", \"response\": ");
            response_json(w, b, cut ? "incomplete" : "completed", cut ? "max_output_tokens" : NULL,
                          res->output_tokens, true, NULL);
            send_ev(w, t);
            return;
        }
        response_json(w, &body, cut ? "incomplete" : "completed", cut ? "max_output_tokens" : NULL, res->output_tokens,
                      true, NULL);
        break;
    }
    }
    ath_http_reply(w->out, 200, "application/json", body.data, body.len);
}

/* ---- the routes --------------------------------------------------------------------- */

static const char *const kind_name[] = {"messages", "chat", "responses"};

static bool parse_body(const ath_http_req *req, ath_http_out *out, ath_api_kind kind, const ath_json **body)
{
    ath_json_error je;
    if (ath_json_parse(out->a, req->body ? req->body : "", req->body_len, body, &je) == ATH_OK)
        return true;
    char msg[160];
    snprintf(msg, sizeof msg, "the body is not valid JSON (line %zu, column %zu: %s)", je.line, je.col,
             je.msg ? je.msg : "error");
    reply_error(out, kind, 400, "invalid_request_error", NULL, msg);
    return false;
}

static void serve_generate(ath_api *api, ath_api_kind kind, const ath_http_req *req, ath_http_out *out)
{
    const ath_json *body;
    if (!parse_body(req, out, kind, &body))
        return;
    ath_api_request r;
    const char *err;
    if (ath_api_parse(api, kind, body, &r, out->a, &err) != ATH_OK) {
        reply_error(out, kind, 400, "invalid_request_error", NULL, err);
        return;
    }
    writer w = {.api = api, .r = &r, .out = out, .a = out->a, .created = (int64_t)time(NULL)};
    new_id(api, kind == ATH_API_ANTHROPIC ? "msg_" : kind == ATH_API_CHAT ? "chatcmpl-" : "resp_", w.id, sizeof w.id);
    ath_buf_init(&w.acc, out->a);
    ath_buf_init(&w.items, out->a);
    ath_buf_init(&w.text, out->a);
    ath_buf_init(&w.think, out->a);
    ath_buf_init(&w.calls, out->a);
    ath_buf_init(&w.e, out->a);
    const ath_gen_sink sink = {.begin = w_begin,
                               .wait = w_wait,
                               .text = w_text,
                               .thinking = w_thinking,
                               .tool_begin = w_tool_begin,
                               .tool_args = w_tool_args,
                               .tool_end = w_tool_end,
                               .u = &w};
    const double t0 = now_s();
    ath_gen_result res;
    ath_status st = ath_gen_run(api->gen, &r.g, &sink, &res, out->a);
    const double t1 = now_s();
    if (st == ATH_ERR_RANGE && !w.begun) {
        char msg[200];
        if (kind == ATH_API_ANTHROPIC)
            snprintf(msg, sizeof msg, "prompt is too long: %d tokens > %d maximum", res.prompt_tokens,
                     api->gen->ctx - 1);
        else
            snprintf(msg, sizeof msg,
                     "This model's maximum context length is %d tokens. However, your messages resulted in %d tokens.",
                     api->gen->ctx, res.prompt_tokens);
        reply_error(out, kind, 400, "invalid_request_error", "context_length_exceeded", msg);
    } else if (st == ATH_ERR_INVALID_ARG && !w.begun) {
        reply_error(out, kind, 400, "invalid_request_error", NULL,
                    "the conversation cannot be rendered with the model's chat template (the server's log says why)");
    } else if (st != ATH_OK) {
        char msg[160];
        snprintf(msg, sizeof msg, "generation failed: %s", ath_status_str(st));
        if (w.begun)
            stream_error(&w, msg);
        else if (!out->broken)
            reply_error(out, kind, 500, kind == ATH_API_ANTHROPIC ? "api_error" : "server_error", NULL, msg);
    } else if (res.finish != ATH_GEN_CANCELLED) {
        finish(&w, &res);
    }
    if (api->log) {
        static const char *const fin[] = {"end", "tool_calls", "length", "stop", "cancelled"};
        const double pre = (w.t_first ? w.t_first : t1) - (w.t_begin ? w.t_begin : t0);
        const double dec = w.t_first ? t1 - w.t_first : 0;
        fprintf(api->log,
                "serve: %s%s %s: prompt %d (%d cached) in %.2f s, %d out in %.2f s (%.1f/s), %s%s%s\n",
                kind_name[kind], r.stream ? " stream" : "", r.g.think ? "think" : "no-think", res.prompt_tokens,
                res.cached_tokens, pre, res.output_tokens, dec, dec > 0 ? res.output_tokens / dec : 0.0,
                st == ATH_OK ? fin[res.finish] : "error: ", st == ATH_OK ? "" : ath_status_str(st),
                r.g.n_tools ? "" : " (no tools)");
        fflush(api->log);
    }
}

static void serve_count(ath_api *api, const ath_http_req *req, ath_http_out *out)
{
    const ath_json *body;
    if (!parse_body(req, out, ATH_API_ANTHROPIC, &body))
        return;
    ath_api_request r;
    const char *err;
    ath_tokens t;
    if (ath_api_parse(api, ATH_API_ANTHROPIC, body, &r, out->a, &err) != ATH_OK) {
        reply_error(out, ATH_API_ANTHROPIC, 400, "invalid_request_error", NULL, err);
        return;
    }
    if (ath_gen_tokens(api->gen, &r.g, &t, out->a) != ATH_OK) {
        reply_error(out, ATH_API_ANTHROPIC, 400, "invalid_request_error", NULL,
                    "the conversation cannot be rendered with the model's chat template");
        return;
    }
    char b[64];
    int n = snprintf(b, sizeof b, "{\"input_tokens\": %zu}", t.n);
    ath_http_reply(out, 200, "application/json", b, (size_t)n);
}

static void model_json(const ath_api *api, ath_buf *b)
{
    char date[32];
    time_t t = (time_t)api->started;
    struct tm tm;
    gmtime_r(&t, &tm);
    strftime(date, sizeof date, "%Y-%m-%dT%H:%M:%SZ", &tm);
    ath_buf_puts(b, "{\"id\": ");
    put(b, api->d.model);
    ath_buf_printf(b, ", \"object\": \"model\", \"created\": %lld, \"owned_by\": \"athanor\", \"type\": \"model\", "
                      "\"display_name\": ",
                   (long long)api->started);
    put(b, api->d.model);
    ath_buf_printf(b, ", \"created_at\": \"%s\", \"max_model_len\": %d}", date, api->gen->ctx);
}

void ath_api_handle(void *ctx, const ath_http_req *req, ath_http_out *out)
{
    ath_api *api = ctx;
    const char *p = req->path;
    if (!strncmp(p, "/v1/", 4))
        p += 3;
    const bool get = !strcmp(req->method, "GET") || !strcmp(req->method, "HEAD");
    const bool post = !strcmp(req->method, "POST");
    ath_buf b;
    ath_buf_init(&b, out->a);

    if (get && (!strcmp(p, "/") || !strcmp(p, "/health"))) {
        static const char ok[] = "{\"status\": \"ok\"}";
        ath_http_reply(out, 200, "application/json", ok, sizeof ok - 1);
    } else if (get && !strcmp(p, "/models")) {
        ath_buf_puts(&b, "{\"object\": \"list\", \"data\": [");
        model_json(api, &b);
        ath_buf_puts(&b, "], \"has_more\": false, \"first_id\": ");
        put(&b, api->d.model);
        ath_buf_puts(&b, ", \"last_id\": ");
        put(&b, api->d.model);
        ath_buf_putc(&b, '}');
        ath_http_reply(out, 200, "application/json", b.data, b.len);
    } else if (get && !strncmp(p, "/models/", 8)) {
        model_json(api, &b);   /* whatever the name: the one model answers to all */
        ath_http_reply(out, 200, "application/json", b.data, b.len);
    } else if (post && !strcmp(p, "/messages")) {
        serve_generate(api, ATH_API_ANTHROPIC, req, out);
    } else if (post && !strcmp(p, "/messages/count_tokens")) {
        serve_count(api, req, out);
    } else if (post && !strcmp(p, "/chat/completions")) {
        serve_generate(api, ATH_API_CHAT, req, out);
    } else if (post && !strcmp(p, "/responses")) {
        serve_generate(api, ATH_API_RESPONSES, req, out);
    } else {
        const bool known = !strcmp(p, "/messages") || !strcmp(p, "/chat/completions") || !strcmp(p, "/responses") ||
                           !strcmp(p, "/models") || !strcmp(p, "/messages/count_tokens");
        char msg[600];
        snprintf(msg, sizeof msg, "%s %s: %s", req->method, req->path, known ? "method not allowed" : "not found");
        reply_error(out, ATH_API_CHAT, known ? 405 : 404, "invalid_request_error", NULL, msg);
    }
}
