/* text.c: Qwen3.5 text handling: pre-tokenization, chat template, output
 * parser.
 *
 * Pre-tokenization is the Split regex of tokenizer.json, written by hand:
 *
 *   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+ | \p{N}
 *   | ?[^\s\p{L}\p{M}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
 *
 * At each position the first alternative that matches wins (the regex is
 * leftmost-first, as Oniguruma runs it), with each alternative's own
 * backtracking spelled out below.
 *
 * The chat template is the checkpoint's chat_template.jinja, rendered by
 * hand for text messages: the same strings in the same order, with the
 * Python semantics the template relies on (str.strip for Jinja's trim,
 * json.dumps for tojson, str() for argument values). docs/contract/
 * qwen35-0.8b.md (Chat template) lists the markup; tests/test_text.c
 * compares the result byte for byte with the reference's rendering.
 *
 * The parser turns the model's output into events. Its markup, from the
 * template:
 *
 *   thinking (when the prompt opened it)  ...\n</think>\n\n
 *   tool call   <tool_call>\n<function=NAME>\n<parameter=P>\nVALUE\n</parameter>\n...</function>\n</tool_call>
 *
 * Newlines that are only template formatting (around </think>, before a
 * tool call, around a parameter value) are dropped. Parameter values become
 * JSON by the type of the parameter in the tool's schema: strings stay
 * strings, numbers, booleans, objects and arrays are written as such when
 * the value parses as one, and anything that does not is kept as a string.
 */
#include "qwen35.h"

#include "core/json.h"
#include "text/unicode.h"

#include <string.h>
#include <strings.h>

/* ---- pre-tokenization ----------------------------------------------------------- */

const char ath_qwen35_pretokenize_regex[] =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|"
    "\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

typedef struct {
    const char *s;
    size_t      n;
} text;

/* The code point at i and its class; *next is the position after it. */
static ath_uc_class at(text t, size_t i, uint32_t *cp, size_t *next)
{
    uint32_t c;
    size_t len = ath_utf8_decode(t.s + i, t.n - i, &c);
    *next = i + len;
    if (cp)
        *cp = c;
    return c == ATH_UTF8_INVALID ? ATH_UC_OTHER : ath_unicode_class(c);
}

static bool is_crlf(uint32_t c) { return c == '\r' || c == '\n'; }

/* Lower-case ASCII letter of the code point at i, as case-insensitive
 * matching sees it ('S' and U+017F LATIN SMALL LETTER LONG S fold to 's');
 * 0 otherwise. */
static char folded(text t, size_t i, size_t *next)
{
    uint32_t c;
    at(t, i, &c, next);
    if (c == 0x17F)
        return 's';
    if (c >= 'A' && c <= 'Z')
        return (char)(c - 'A' + 'a');
    return c >= 'a' && c <= 'z' ? (char)c : 0;
}

/* (?i:'s|'t|'re|'ve|'m|'ll|'d) at i: the end, or 0. */
static size_t contraction(text t, size_t i)
{
    if (t.s[i] != '\'' || i + 1 >= t.n)
        return 0;
    size_t j, k;
    char c = folded(t, i + 1, &j);
    if (c == 's' || c == 't' || c == 'm' || c == 'd')
        return j;
    if (j >= t.n)
        return 0;
    char d = folded(t, j, &k);
    if ((c == 'r' || c == 'v') && d == 'e')
        return k;
    if (c == 'l' && d == 'l')
        return k;
    return 0;
}

/* The end of the run of letters and marks starting at i (i if none). */
static size_t letters(text t, size_t i)
{
    while (i < t.n) {
        size_t next;
        ath_uc_class k = at(t, i, NULL, &next);
        if (k != ATH_UC_LETTER && k != ATH_UC_MARK)
            break;
        i = next;
    }
    return i;
}

/* The end of the next piece starting at i < n. */
static size_t piece_end(text t, size_t i)
{
    size_t e = contraction(t, i);
    if (e)
        return e;

    uint32_t c0;
    size_t   j;
    ath_uc_class k0 = at(t, i, &c0, &j);

    /* [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+ : first with the optional character,
     * then without it. */
    if (!is_crlf(c0) && k0 != ATH_UC_LETTER && k0 != ATH_UC_NUMBER) {
        e = letters(t, j);
        if (e > j)
            return e;
    }
    if (k0 == ATH_UC_LETTER || k0 == ATH_UC_MARK)
        return letters(t, i);

    /* \p{N} */
    if (k0 == ATH_UC_NUMBER)
        return j;

    /* ' ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*' */
    size_t p = c0 == ' ' ? j : i;
    size_t q = p;
    while (q < t.n) {
        size_t next;
        if (at(t, q, NULL, &next) != ATH_UC_OTHER)
            break;
        q = next;
    }
    if (q > p) {
        while (q < t.n && is_crlf((unsigned char)t.s[q]))
            q++;
        return q;
    }

    /* The three white-space alternatives share the run of white space. */
    if (k0 != ATH_UC_SPACE)
        return j;   /* not reached: every class is handled above */
    size_t run = i, last_crlf = 0, before_last = i;
    while (run < t.n) {
        uint32_t c;
        size_t next;
        if (at(t, run, &c, &next) != ATH_UC_SPACE)
            break;
        if (is_crlf(c))
            last_crlf = next;
        before_last = run;
        run = next;
    }
    /* \s*[\r\n]+ : up to the last CR or LF of the run */
    if (last_crlf)
        return last_crlf;
    /* \s+(?!\S) : the whole run at the end of the text, else all but its last
     * character, if that leaves one */
    if (run == t.n)
        return run;
    if (before_last > i)
        return before_last;
    /* \s+ */
    return run;
}

ath_status ath_qwen35_pretokenize(const char *s, size_t n, ath_spans *out)
{
    text t = {s, n};
    for (size_t i = 0; i < n;) {
        i = piece_end(t, i);
        ATH_CHECK(ath_spans_push(out, i));
    }
    return ATH_OK;
}

/* ---- Python string semantics the template relies on ----------------------------- */

/* str.isspace(): the template's trim filter is Python's str.strip(). */
static bool py_space(uint32_t c)
{
    return (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x20) || c == 0x85 || c == 0xA0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F ||
           c == 0x3000;
}

/* s.strip() */
static text trim(text t)
{
    size_t b = 0, e = t.n;
    while (b < e) {
        uint32_t c;
        size_t len = ath_utf8_decode(t.s + b, e - b, &c);
        if (!py_space(c))
            break;
        b += len;
    }
    while (e > b) {
        /* step back to the start of the last code point */
        size_t s = e - 1;
        while (s > b && ((unsigned char)t.s[s] & 0xC0) == 0x80 && e - s < 4)
            s--;
        uint32_t c;
        if (ath_utf8_decode(t.s + s, e - s, &c) != e - s)
            s = e - 1, c = ATH_UTF8_INVALID;
        if (!py_space(c))
            break;
        e = s;
    }
    return (text){t.s + b, e - b};
}

static text cstr(const char *s) { return (text){s ? s : "", s ? strlen(s) : 0}; }

static bool starts(text t, const char *p) { size_t n = strlen(p); return t.n >= n && memcmp(t.s, p, n) == 0; }
static bool ends(text t, const char *p) { size_t n = strlen(p); return t.n >= n && memcmp(t.s + t.n - n, p, n) == 0; }

/* Offset of the first (or last) occurrence of p in t, or -1. */
static ptrdiff_t find(text t, const char *p, bool last)
{
    size_t n = strlen(p);
    ptrdiff_t r = -1;
    for (size_t i = 0; i + n <= t.n; i++)
        if (memcmp(t.s + i, p, n) == 0) {
            r = (ptrdiff_t)i;
            if (!last)
                break;
        }
    return r;
}

static text strip_nl(text t, bool left, bool right)
{
    while (left && t.n && t.s[0] == '\n')
        t.s++, t.n--;
    while (right && t.n && t.s[t.n - 1] == '\n')
        t.n--;
    return t;
}

static ath_status put(ath_buf *b, text t) { return ath_buf_append(b, t.s, t.n); }

/* {{ value | string }} for an argument value: Python's str() of what
 * json.loads gave, or tojson for a mapping or a list. */
static ath_status put_py_value(ath_buf *b, const ath_json *v)
{
    switch (v->type) {
    case ATH_JSON_STRING: return ath_buf_append(b, v->u.str.ptr, v->u.str.len);
    case ATH_JSON_NULL:   return ath_buf_puts(b, "None");
    case ATH_JSON_BOOL:   return ath_buf_puts(b, v->u.b ? "True" : "False");
    default:              return ath_json_write(v, b);
    }
}

/* {{ tool | tojson }} for a tool given as OpenAI does it:
 * {"type": "function", "function": {"name", "description", "parameters"}}. */
static ath_status put_tool(ath_buf *b, const ath_tool_def *t, ath_arena *a)
{
    ATH_CHECK(ath_buf_puts(b, "{\"type\": \"function\", \"function\": {\"name\": "));
    ATH_CHECK(ath_json_write_str(b, t->name, strlen(t->name)));
    if (t->description) {
        ATH_CHECK(ath_buf_puts(b, ", \"description\": "));
        ATH_CHECK(ath_json_write_str(b, t->description, strlen(t->description)));
    }
    if (t->json_schema) {
        const ath_json *schema;
        if (ath_json_parse(a, t->json_schema, strlen(t->json_schema), &schema, NULL) != ATH_OK) {
            ath_log(ATH_LOG_ERROR, "qwen35: the schema of tool '%s' is not valid JSON", t->name);
            return ATH_ERR_INVALID_ARG;
        }
        ATH_CHECK(ath_buf_puts(b, ", \"parameters\": "));
        ATH_CHECK(ath_json_write(schema, b));
    }
    return ath_buf_puts(b, "}}");
}

/* ---- chat template ----------------------------------------------------------- */

static const char tools_header[] = "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>";

static const char tools_footer[] =
    "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested "
    "within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language BEFORE the function call, "
    "but NOT after\n"
    "- If there is no function call available, answer the question like normal with your current knowledge and "
    "do not tell the user about function calls\n</IMPORTANT>";

static ath_status put_tool_calls(ath_buf *b, const ath_msg *m, text content, ath_arena *a)
{
    for (int32_t k = 0; k < m->n_tool_calls; k++) {
        const ath_tool_call *tc = &m->tool_calls[k];
        if (k == 0)
            ATH_CHECK(ath_buf_puts(b, trim(content).n ? "\n\n<tool_call>\n<function=" : "<tool_call>\n<function="));
        else
            ATH_CHECK(ath_buf_puts(b, "\n<tool_call>\n<function="));
        ATH_CHECK(ath_buf_puts(b, tc->name));
        ATH_CHECK(ath_buf_puts(b, ">\n"));
        if (tc->arguments) {
            const ath_json *args;
            if (ath_json_parse(a, tc->arguments, strlen(tc->arguments), &args, NULL) != ATH_OK ||
                args->type != ATH_JSON_OBJECT) {
                ath_log(ATH_LOG_ERROR, "qwen35: the arguments of a call to '%s' are not a JSON object", tc->name);
                return ATH_ERR_INVALID_ARG;
            }
            for (const ath_json *v = ath_json_first(args); v; v = v->next) {
                ATH_CHECK(ath_buf_puts(b, "<parameter="));
                ATH_CHECK(ath_buf_append(b, v->key, v->key_len));
                ATH_CHECK(ath_buf_puts(b, ">\n"));
                ATH_CHECK(put_py_value(b, v));
                ATH_CHECK(ath_buf_puts(b, "\n</parameter>\n"));
            }
        }
        ATH_CHECK(ath_buf_puts(b, "</function>\n</tool_call>"));
    }
    return ATH_OK;
}

static ath_status put_assistant(ath_buf *b, const ath_msg *m, text content, bool after_query, ath_arena *a)
{
    text reasoning = {"", 0};
    if (m->reasoning) {
        reasoning = cstr(m->reasoning);
    } else if (find(content, "</think>", false) >= 0) {
        /* content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') */
        text before = {content.s, (size_t)find(content, "</think>", false)};
        before = strip_nl(before, false, true);
        ptrdiff_t open = find(before, "<think>", true);
        if (open >= 0)
            before = (text){before.s + open + 7, before.n - (size_t)open - 7};
        reasoning = strip_nl(before, true, false);
        /* content.split('</think>')[-1].lstrip('\n') */
        size_t after = (size_t)find(content, "</think>", true) + 8;
        content = strip_nl((text){content.s + after, content.n - after}, true, false);
    }
    reasoning = trim(reasoning);
    ATH_CHECK(ath_buf_puts(b, "<|im_start|>assistant\n"));
    if (after_query) {
        ATH_CHECK(ath_buf_puts(b, "<think>\n"));
        ATH_CHECK(put(b, reasoning));
        ATH_CHECK(ath_buf_puts(b, "\n</think>\n\n"));
    }
    ATH_CHECK(put(b, content));
    ATH_CHECK(put_tool_calls(b, m, content, a));
    return ath_buf_puts(b, "<|im_end|>\n");
}

ath_status ath_qwen35_chat(const ath_msg *msgs, int32_t n, const ath_tool_def *tools, int32_t n_tools, uint32_t opts,
                           ath_buf *b)
{
    if (n < 1) {
        ath_log(ATH_LOG_ERROR, "qwen35: no messages");
        return ATH_ERR_INVALID_ARG;
    }
    /* Parsed tool schemas and arguments live in the buffer's arena until it
     * is reset with it. */
    ath_arena *a = b->a;

    if (n_tools > 0) {
        ATH_CHECK(ath_buf_puts(b, tools_header));
        for (int32_t i = 0; i < n_tools; i++) {
            ATH_CHECK(ath_buf_putc(b, '\n'));
            ATH_CHECK(put_tool(b, &tools[i], a));
        }
        ATH_CHECK(ath_buf_puts(b, tools_footer));
        if (msgs[0].role == ATH_MSG_SYSTEM) {
            text c = trim(cstr(msgs[0].content));
            if (c.n) {
                ATH_CHECK(ath_buf_puts(b, "\n\n"));
                ATH_CHECK(put(b, c));
            }
        }
        ATH_CHECK(ath_buf_puts(b, "<|im_end|>\n"));
    } else if (msgs[0].role == ATH_MSG_SYSTEM) {
        ATH_CHECK(ath_buf_puts(b, "<|im_start|>system\n"));
        ATH_CHECK(put(b, trim(cstr(msgs[0].content))));
        ATH_CHECK(ath_buf_puts(b, "<|im_end|>\n"));
    }

    /* The last user message that is not only a tool response: assistant
     * turns after it show their thinking. */
    int32_t last_query = -1;
    for (int32_t i = n - 1; i >= 0 && last_query < 0; i--) {
        if (msgs[i].role != ATH_MSG_USER)
            continue;
        text c = trim(cstr(msgs[i].content));
        if (!(starts(c, "<tool_response>") && ends(c, "</tool_response>")))
            last_query = i;
    }
    if (last_query < 0) {
        ath_log(ATH_LOG_ERROR, "qwen35: no user query in the messages");
        return ATH_ERR_INVALID_ARG;
    }

    for (int32_t i = 0; i < n; i++) {
        const ath_msg *m = &msgs[i];
        text content = trim(cstr(m->content));
        switch (m->role) {
        case ATH_MSG_SYSTEM:
            if (i != 0) {
                ath_log(ATH_LOG_ERROR, "qwen35: a system message must come first");
                return ATH_ERR_INVALID_ARG;
            }
            break;
        case ATH_MSG_USER:
            ATH_CHECK(ath_buf_puts(b, "<|im_start|>user\n"));
            ATH_CHECK(put(b, content));
            ATH_CHECK(ath_buf_puts(b, "<|im_end|>\n"));
            break;
        case ATH_MSG_ASSISTANT:
            ATH_CHECK(put_assistant(b, m, content, i > last_query, a));
            break;
        case ATH_MSG_TOOL:
            /* consecutive tool results share one user turn */
            if (i > 0 && msgs[i - 1].role != ATH_MSG_TOOL)
                ATH_CHECK(ath_buf_puts(b, "<|im_start|>user"));
            ATH_CHECK(ath_buf_puts(b, "\n<tool_response>\n"));
            ATH_CHECK(put(b, content));
            ATH_CHECK(ath_buf_puts(b, "\n</tool_response>"));
            if (i == n - 1 || msgs[i + 1].role != ATH_MSG_TOOL)
                ATH_CHECK(ath_buf_puts(b, "<|im_end|>\n"));
            break;
        }
    }
    if (opts & ATH_CHAT_GENERATE)
        ATH_CHECK(ath_buf_puts(b, opts & ATH_CHAT_THINKING ? "<|im_start|>assistant\n<think>\n"
                                                           : "<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    return ATH_OK;
}

/* ---- output parser ------------------------------------------------------------- */

typedef enum {
    P_TEXT,     /* answer text                                                    */
    P_THINK,    /* thinking, until </think>                                       */
    P_CALL,     /* after <tool_call>, until <function=NAME> is complete           */
    P_FUNC,     /* inside <function=...>, between parameters                      */
    P_VALUE,    /* inside <parameter=...>                                         */
    P_CLOSE,    /* after </function>, until </tool_call>                          */
} pmode;

typedef struct {
    pmode               mode;
    bool                skip_nl;    /* drop newlines at the start of text or thinking */
    ath_buf             in;         /* bytes received and not consumed yet            */
    ath_buf             out;        /* event bytes of the current call                */
    ath_buf             name;       /* the current parameter's name                    */
    int32_t             n_params;   /* parameters written for the current call        */
    size_t              scanned;    /* bytes of a value already searched for its end   */
    const ath_json     *props;      /* the current tool's schema "properties", or NULL */
    ath_out_event      *ev;
    int32_t             n_ev, cap;
    size_t             *ev_off;     /* event data offsets into out, fixed up at the end */
    int32_t             off_cap;
} parser;

/* Append an event; consecutive events of the same kind with data merge. */
static ath_status emit(parser *p, ath_out_kind kind, const char *data, size_t len, const char *tool_name)
{
    bool has_data = kind == ATH_OUT_TEXT || kind == ATH_OUT_THINKING || kind == ATH_OUT_TOOL_CALL_ARGS;
    if (has_data && !len)
        return ATH_OK;
    if (has_data && p->n_ev > 0 && p->ev[p->n_ev - 1].kind == kind) {
        p->ev[p->n_ev - 1].len += len;
        return ath_buf_append(&p->out, data, len);
    }
    if (p->n_ev == p->cap)
        return ATH_ERR_RANGE;
    p->ev_off[p->n_ev] = p->out.len;
    p->ev[p->n_ev++] = (ath_out_event){.kind = kind, .len = len, .tool_name = tool_name};
    return ath_buf_append(&p->out, data, len);
}

/* Drop the first n bytes of the input. */
static void consume(parser *p, size_t n)
{
    memmove(p->in.data, p->in.data + n, p->in.len - n);
    p->in.len -= n;
    p->in.data[p->in.len] = '\0';
}

/* 1 if s[0 .. n) starts with tag, 0 if it cannot, -1 if it is a proper
 * prefix of tag (more bytes could complete it). */
static int match(const char *s, size_t n, const char *tag)
{
    size_t t = strlen(tag);
    size_t k = n < t ? n : t;
    if (memcmp(s, tag, k) != 0)
        return 0;
    return n >= t ? 1 : -1;
}

/* The type a tool's schema gives to a parameter, or NULL. */
static const char *param_type(const parser *p, const char *name, size_t len)
{
    for (const ath_json *v = ath_json_first(p->props); v; v = v->next) {
        if (v->key_len != len || memcmp(v->key, name, len) != 0)
            continue;
        const ath_json *ty = ath_json_get(v, "type");
        if (ty && ty->type == ATH_JSON_ARRAY)   /* ["integer", "null"]: the first non-null */
            for (const ath_json *e = ath_json_first(ty); e; e = e->next)
                if (e->type == ATH_JSON_STRING && strcmp(e->u.str.ptr, "null") != 0)
                    return e->u.str.ptr;
        const char *s;
        return ath_json_to_str(ty, &s, NULL) == ATH_OK ? s : NULL;
    }
    return NULL;
}

/* Is v[0 .. n) a JSON number (the whole of it)? */
static bool json_number(const char *v, size_t n, bool integer)
{
    size_t i = 0;
    if (i < n && v[i] == '-')
        i++;
    if (i == n || v[i] < '0' || v[i] > '9')
        return false;
    if (v[i] == '0')
        i++;
    else
        while (i < n && v[i] >= '0' && v[i] <= '9')
            i++;
    if (!integer && i < n && v[i] == '.') {
        if (++i == n || v[i] < '0' || v[i] > '9')
            return false;
        while (i < n && v[i] >= '0' && v[i] <= '9')
            i++;
    }
    if (!integer && i < n && (v[i] == 'e' || v[i] == 'E')) {
        i++;
        if (i < n && (v[i] == '+' || v[i] == '-'))
            i++;
        if (i == n || v[i] < '0' || v[i] > '9')
            return false;
        while (i < n && v[i] >= '0' && v[i] <= '9')
            i++;
    }
    return i == n;
}

/* The finished parameter: , "name": value (the first one opens the object). */
static ath_status put_param(parser *p, ath_arena *a, const char *v, size_t n)
{
    ath_buf b;
    ath_buf_init(&b, a);
    ATH_CHECK(ath_buf_puts(&b, p->n_params ? ", " : "{"));
    ATH_CHECK(ath_json_write_str(&b, p->name.data, p->name.len));
    ATH_CHECK(ath_buf_puts(&b, ": "));
    const char *ty = param_type(p, p->name.data, p->name.len);
    bool done = false;
    if (ty && strcmp(ty, "string") != 0) {
        if (n == 4 && memcmp(v, "null", 4) == 0) {
            ATH_CHECK(ath_buf_puts(&b, "null"));
            done = true;
        } else if (!strcmp(ty, "integer") || !strcmp(ty, "number")) {
            if (json_number(v, n, !strcmp(ty, "integer"))) {
                ATH_CHECK(ath_buf_append(&b, v, n));
                done = true;
            }
        } else if (!strcmp(ty, "boolean")) {
            if ((n == 4 && !strncasecmp(v, "true", 4)) || (n == 5 && !strncasecmp(v, "false", 5))) {
                ATH_CHECK(ath_buf_puts(&b, n == 4 ? "true" : "false"));
                done = true;
            }
        } else {   /* object, array, or a type this parser does not know: JSON if it parses */
            const ath_json *j;
            if (ath_json_parse(a, v, n, &j, NULL) == ATH_OK) {
                ATH_CHECK(ath_json_write(j, &b));
                done = true;
            }
        }
    }
    if (!done)
        ATH_CHECK(ath_json_write_str(&b, v, n));
    p->n_params++;
    return emit(p, ATH_OUT_TOOL_CALL_ARGS, b.data, b.len, NULL);
}

static ath_status end_call(parser *p)
{
    ATH_CHECK(emit(p, ATH_OUT_TOOL_CALL_ARGS, p->n_params ? "}" : "{}", p->n_params ? 1 : 2, NULL));
    return emit(p, ATH_OUT_TOOL_CALL_END, NULL, 0, NULL);
}

/* The tags that end prose: in thinking only </think>; in text <think> and
 * <tool_call>. Newlines just before </think> or <tool_call> are template
 * formatting and are dropped. */
static const char *const think_tags[] = {"</think>"};
static const char *const text_tags[] = {"<think>", "<tool_call>"};

/* Text or thinking, up to the next tag. A run of newlines, or a '<' that
 * may start a tag, at the end of the input is held until the next piece
 * shows what follows. */
static ath_status step_prose(parser *p, bool final, bool *progress)
{
    const bool think = p->mode == P_THINK;
    const ath_out_kind kind = think ? ATH_OUT_THINKING : ATH_OUT_TEXT;
    const char *const *tags = think ? think_tags : text_tags;
    const int n_tags = think ? 1 : 2;
    const char *s = p->in.data;
    size_t n = p->in.len, i = 0;

    if (p->skip_nl) {
        while (i < n && s[i] == '\n')
            i++;
        if (i < n)
            p->skip_nl = false;
    }
    size_t start = i, hold = n;
    while (i < n) {
        if (s[i] != '<' && s[i] != '\n') {
            i++;
            continue;
        }
        size_t t = i;   /* the '<' after a run of newlines, or i itself */
        while (t < n && s[t] == '\n')
            t++;
        if (t == n) {
            if (!final)
                hold = i;
            break;
        }
        if (s[t] != '<') {
            i = t;
            continue;
        }
        int which = -1;
        bool partial = false;
        for (int k = 0; k < n_tags; k++) {
            int r = match(s + t, n - t, tags[k]);
            which = r == 1 ? k : which;
            partial |= r == -1;
        }
        if (which < 0 && partial && !final) {
            hold = i;
            break;
        }
        if (which < 0) {
            i = t + 1;   /* a '<' that starts no tag is text */
            continue;
        }
        const char *tag = tags[which];
        bool formatting = !strcmp(tag, "</think>") || !strcmp(tag, "<tool_call>");
        ATH_CHECK(emit(p, kind, s + start, (formatting ? i : t) - start, NULL));
        if (!strcmp(tag, "<tool_call>")) {
            consume(p, t);   /* the call keeps its tag until it is known to be well formed */
            p->mode = P_CALL;
        } else {
            consume(p, t + strlen(tag));
            p->mode = think ? P_TEXT : P_THINK;
            p->skip_nl = true;
        }
        *progress = true;
        return ATH_OK;
    }
    ATH_CHECK(emit(p, kind, s + start, hold - start, NULL));
    *progress = hold > 0;
    consume(p, hold);
    return ATH_OK;
}

/* <tool_call>\n<function=NAME>\n : the call begins once NAME is complete. */
static ath_status step_call(parser *p, ath_parse_state *st, bool final, bool *progress)
{
    const char *s = p->in.data;
    size_t n = p->in.len, i = 11;   /* after <tool_call> */
    while (i < n && (s[i] == '\n' || s[i] == ' '))
        i++;
    int r = i < n ? match(s + i, n - i, "<function=") : -1;
    size_t name = i + 10, end = name;
    if (r == 1) {
        while (end < n && s[end] != '>' && s[end] != '<' && s[end] != '\n' && end - name < 256)
            end++;
        if (end < n && s[end] == '>' && end > name) {
            char *fn = ath_arena_strndup(st->a, s + name, end - name);
            if (!fn)
                return ATH_ERR_NOMEM;
            p->props = NULL;
            for (int32_t k = 0; k < st->n_tools; k++) {
                const ath_tool_def *t = &st->tools[k];
                const ath_json *schema;
                if (!strcmp(t->name, fn) && t->json_schema &&
                    ath_json_parse(st->a, t->json_schema, strlen(t->json_schema), &schema, NULL) == ATH_OK)
                    p->props = ath_json_get(schema, "properties");
            }
            ATH_CHECK(emit(p, ATH_OUT_TOOL_CALL_BEGIN, NULL, 0, fn));
            p->n_params = 0;
            consume(p, end + 1);
            p->mode = P_FUNC;
            *progress = true;
            return ATH_OK;
        }
        if (end == n && !final)
            return ATH_OK;   /* the name may continue */
    } else if (r == -1 && !final) {
        return ATH_OK;
    }
    /* Malformed: <tool_call> and what follows are text. */
    ATH_CHECK(emit(p, ATH_OUT_TEXT, s, 11, NULL));
    consume(p, 11);
    p->mode = P_TEXT;
    *progress = true;
    return ATH_OK;
}

/* Between parameters: <parameter=NAME>, </function>, or the end of the call. */
static ath_status step_func(parser *p, bool final, bool *progress)
{
    const char *s = p->in.data;
    size_t n = p->in.len, i = 0;
    while (i < n && (s[i] == '\n' || s[i] == ' '))
        i++;
    if (i == n) {
        if (final) {
            ATH_CHECK(end_call(p));
            p->mode = P_TEXT;
        }
        consume(p, i);
        *progress = i > 0 || final;
        return ATH_OK;
    }
    int rp = match(s + i, n - i, "<parameter="), rf = match(s + i, n - i, "</function>"),
        rt = match(s + i, n - i, "</tool_call>");
    if (rp == 1) {
        size_t name = i + 11, end = name;
        while (end < n && s[end] != '>' && s[end] != '\n' && end - name < 256)
            end++;
        if (end < n && s[end] == '>' && end > name) {
            ath_buf_clear(&p->name);
            ATH_CHECK(ath_buf_append(&p->name, s + name, end - name));
            end++;
            if (end < n && s[end] == '\n')
                end++;
            else if (end == n && !final)
                return ATH_OK;   /* wait for the newline that follows the tag */
            consume(p, end);
            p->mode = P_VALUE;
            *progress = true;
            return ATH_OK;
        }
        if (end == n && !final)
            return ATH_OK;
    } else if (rf == 1 || rt == 1) {
        ATH_CHECK(end_call(p));
        consume(p, i + (rf == 1 ? 11 : 12));
        p->mode = rf == 1 ? P_CLOSE : P_TEXT;
        p->skip_nl = true;
        *progress = true;
        return ATH_OK;
    } else if ((rp == -1 || rf == -1 || rt == -1) && !final) {
        return ATH_OK;
    }
    /* Malformed: close the call; the rest is text. */
    ATH_CHECK(end_call(p));
    consume(p, i);
    p->mode = P_TEXT;
    *progress = true;
    return ATH_OK;
}

/* A parameter value, until \n</parameter> (or </function>, if the model
 * forgot to close the parameter). */
static ath_status step_value(parser *p, ath_parse_state *st, bool final, bool *progress)
{
    /* Search only the bytes not searched before (a tag may straddle). */
    size_t from = p->scanned > 12 ? p->scanned - 12 : 0;
    text t = {p->in.data, p->in.len}, rest = {t.s + from, t.n - from};
    ptrdiff_t e = find(rest, "</parameter>", false), f = find(rest, "</function>", false);
    e = e < 0 ? -1 : e + (ptrdiff_t)from;
    f = f < 0 ? -1 : f + (ptrdiff_t)from;
    p->scanned = t.n;
    size_t end, skip;
    if (e >= 0 && (f < 0 || e < f)) {
        end = (size_t)e, skip = 12;
    } else if (f >= 0) {
        end = (size_t)f, skip = 0;   /* </function> stays for step_func */
    } else if (final) {
        end = t.n, skip = 0;
    } else {
        return ATH_OK;   /* the value continues */
    }
    p->scanned = 0;
    size_t len = end;
    if (len && t.s[len - 1] == '\n')
        len--;
    ATH_CHECK(put_param(p, st->a, t.s, len));
    consume(p, end + skip);
    p->mode = P_FUNC;
    *progress = true;
    return ATH_OK;
}

/* After </function>: \n</tool_call>. Anything else ends the call there. */
static ath_status step_close(parser *p, bool final, bool *progress)
{
    const char *s = p->in.data;
    size_t n = p->in.len, i = 0;
    while (i < n && (s[i] == '\n' || s[i] == ' '))
        i++;
    int r = i < n ? match(s + i, n - i, "</tool_call>") : -1;
    if (r == -1 && !final)
        return ATH_OK;
    consume(p, r == 1 ? i + 12 : i);
    p->mode = P_TEXT;
    p->skip_nl = true;   /* newlines after a call are formatting */
    *progress = true;
    return ATH_OK;
}

ath_status ath_qwen35_parse(ath_parse_state *st, const char *piece, size_t n, ath_out_event *events, int32_t cap,
                            int32_t *n_events)
{
    parser *p = st->priv;
    if (!p) {
        p = ath_arena_zalloc(st->a, sizeof *p, _Alignof(parser));
        if (!p)
            return ATH_ERR_NOMEM;
        ath_buf_init(&p->in, st->a);
        ath_buf_init(&p->out, st->a);
        ath_buf_init(&p->name, st->a);
        p->mode = st->opts & ATH_CHAT_THINKING ? P_THINK : P_TEXT;
        p->skip_nl = true;
        st->priv = p;
    }
    const bool final = piece == NULL;
    *n_events = 0;
    p->ev = events;
    p->cap = cap;
    p->n_ev = 0;
    if (cap > p->off_cap) {
        p->ev_off = ath_arena_alloc(st->a, (size_t)cap * sizeof *p->ev_off, _Alignof(size_t));
        if (!p->ev_off)
            return ATH_ERR_NOMEM;
        p->off_cap = cap;
    }
    ath_buf_clear(&p->out);
    if (n)
        ATH_CHECK(ath_buf_append(&p->in, piece, n));
    if (!p->in.data)
        ATH_CHECK(ath_buf_append(&p->in, "", 0));

    for (bool progress = true; progress;) {
        progress = false;
        switch (p->mode) {
        case P_TEXT:
        case P_THINK: ATH_CHECK(step_prose(p, final, &progress)); break;
        case P_CALL:  ATH_CHECK(step_call(p, st, final, &progress)); break;
        case P_FUNC:  ATH_CHECK(step_func(p, final, &progress)); break;
        case P_VALUE: ATH_CHECK(step_value(p, st, final, &progress)); break;
        case P_CLOSE: ATH_CHECK(step_close(p, final, &progress)); break;
        }
        if (final && !p->in.len && (p->mode == P_TEXT || p->mode == P_THINK))
            break;
    }
    for (int32_t i = 0; i < p->n_ev; i++)
        events[i].data = events[i].len ? p->out.data + p->ev_off[i] : NULL;
    *n_events = p->n_ev;
    return ATH_OK;
}

