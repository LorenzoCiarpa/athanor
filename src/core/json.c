/* json.c: recursive-descent JSON parser. See json.h.
 *
 * The grammar is small enough that each production is one function:
 *
 *   value  = object | array | string | number | "true" | "false" | "null"
 *   object = "{" [ string ":" value { "," string ":" value } ] "}"
 *   array  = "[" [ value { "," value } ] "]"
 *
 * Recursion depth is bounded by ATH_JSON_MAX_DEPTH so hostile input cannot
 * exhaust the stack.
 */
#include "core/json.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *s;
    size_t      n;
    size_t      pos;
    int         depth;
    ath_arena  *a;
    const char *err;      /* first error message, static */
    size_t      err_pos;
    ath_status  status;
} parser;

static ath_status fail(parser *p, const char *msg)
{
    if (!p->err) {
        p->err = msg;
        p->err_pos = p->pos;
        p->status = ATH_ERR_PARSE;
    }
    return p->status;
}

static ath_status oom(parser *p)
{
    if (!p->err) {
        p->err = "out of memory";
        p->err_pos = p->pos;
        p->status = ATH_ERR_NOMEM;
    }
    return p->status;
}

static void skip_ws(parser *p)
{
    while (p->pos < p->n) {
        char c = p->s[p->pos];
        if (c != ' ' && c != '\t' && c != '\n' && c != '\r')
            break;
        p->pos++;
    }
}

static ath_json *new_node(parser *p, ath_json_type type)
{
    ath_json *j = ath_arena_alloc(p->a, sizeof *j, _Alignof(ath_json));
    if (!j)
        return NULL;
    memset(j, 0, sizeof *j);
    j->type = type;
    return j;
}

/* ---- strings ---------------------------------------------------------------- */

/* Length of the valid UTF-8 sequence starting at s (at most n bytes), or 0.
 * Rejects overlong forms, surrogates and code points above U+10FFFF. */
static size_t utf8_len(const unsigned char *s, size_t n)
{
    unsigned char c = s[0];
    if (c < 0x80)
        return 1;
    size_t len;
    uint32_t cp, min;
    if (c >= 0xc2 && c <= 0xdf) {
        len = 2; cp = c & 0x1f; min = 0x80;
    } else if (c >= 0xe0 && c <= 0xef) {
        len = 3; cp = c & 0x0f; min = 0x800;
    } else if (c >= 0xf0 && c <= 0xf4) {
        len = 4; cp = c & 0x07; min = 0x10000;
    } else {
        return 0;
    }
    if (len > n)
        return 0;
    for (size_t i = 1; i < len; i++) {
        if ((s[i] & 0xc0) != 0x80)
            return 0;
        cp = (cp << 6) | (s[i] & 0x3f);
    }
    if (cp < min || cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff))
        return 0;
    return len;
}

static size_t utf8_put(char *out, uint32_t cp)
{
    unsigned char *o = (unsigned char *)out;
    if (cp < 0x80) {
        o[0] = (unsigned char)cp;
        return 1;
    }
    if (cp < 0x800) {
        o[0] = (unsigned char)(0xc0 | (cp >> 6));
        o[1] = (unsigned char)(0x80 | (cp & 0x3f));
        return 2;
    }
    if (cp < 0x10000) {
        o[0] = (unsigned char)(0xe0 | (cp >> 12));
        o[1] = (unsigned char)(0x80 | ((cp >> 6) & 0x3f));
        o[2] = (unsigned char)(0x80 | (cp & 0x3f));
        return 3;
    }
    o[0] = (unsigned char)(0xf0 | (cp >> 18));
    o[1] = (unsigned char)(0x80 | ((cp >> 12) & 0x3f));
    o[2] = (unsigned char)(0x80 | ((cp >> 6) & 0x3f));
    o[3] = (unsigned char)(0x80 | (cp & 0x3f));
    return 4;
}

/* Four hex digits at p->pos; advances past them. */
static bool read_hex4(parser *p, uint32_t *out)
{
    if (p->n - p->pos < 4)
        return false;
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) {
        char c = p->s[p->pos + (size_t)i];
        v <<= 4;
        if (c >= '0' && c <= '9')
            v |= (uint32_t)(c - '0');
        else if (c >= 'a' && c <= 'f')
            v |= (uint32_t)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F')
            v |= (uint32_t)(c - 'A' + 10);
        else
            return false;
    }
    p->pos += 4;
    *out = v;
    return true;
}

/* Parse a string starting at the opening quote. The decoded text is never
 * longer than the raw text (every escape shrinks or keeps its length), so a
 * first pass finds the closing quote and sizes the buffer exactly once. */
static ath_status parse_string(parser *p, const char **out, size_t *out_len)
{
    p->pos++; /* opening quote */
    size_t start = p->pos, end = start;
    while (end < p->n && p->s[end] != '"')
        end += (p->s[end] == '\\') ? 2 : 1;
    if (end >= p->n) {
        p->pos = p->n;
        return fail(p, "unterminated string");
    }

    char *buf = ath_arena_alloc(p->a, end - start + 1, 1);
    if (!buf)
        return oom(p);
    size_t w = 0;
    while (p->pos < end) {
        unsigned char c = (unsigned char)p->s[p->pos];
        if (c < 0x20)
            return fail(p, "control character in string");
        if (c >= 0x80) {
            size_t l = utf8_len((const unsigned char *)p->s + p->pos, end - p->pos);
            if (!l)
                return fail(p, "invalid UTF-8 in string");
            memcpy(buf + w, p->s + p->pos, l);
            w += l;
            p->pos += l;
            continue;
        }
        if (c != '\\') {
            buf[w++] = (char)c;
            p->pos++;
            continue;
        }
        p->pos++; /* backslash */
        char e = p->s[p->pos++];
        switch (e) {
        case '"':  buf[w++] = '"';  break;
        case '\\': buf[w++] = '\\'; break;
        case '/':  buf[w++] = '/';  break;
        case 'b':  buf[w++] = '\b'; break;
        case 'f':  buf[w++] = '\f'; break;
        case 'n':  buf[w++] = '\n'; break;
        case 'r':  buf[w++] = '\r'; break;
        case 't':  buf[w++] = '\t'; break;
        case 'u': {
            uint32_t cp;
            if (!read_hex4(p, &cp))
                return fail(p, "invalid \\u escape");
            if (cp >= 0xdc00 && cp <= 0xdfff)
                return fail(p, "lone low surrogate in \\u escape");
            if (cp >= 0xd800 && cp <= 0xdbff) {
                /* UTF-16 surrogate pair: 😀 is U+1F600. */
                uint32_t lo;
                if (end - p->pos < 6 || p->s[p->pos] != '\\' || p->s[p->pos + 1] != 'u')
                    return fail(p, "high surrogate not followed by \\u escape");
                p->pos += 2;
                if (!read_hex4(p, &lo))
                    return fail(p, "invalid \\u escape");
                if (lo < 0xdc00 || lo > 0xdfff)
                    return fail(p, "high surrogate not followed by low surrogate");
                cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00);
            }
            w += utf8_put(buf + w, cp);
            break;
        }
        default:
            p->pos--;
            return fail(p, "invalid escape in string");
        }
    }
    buf[w] = '\0';
    p->pos = end + 1; /* closing quote */
    *out = buf;
    *out_len = w;
    return ATH_OK;
}

/* ---- numbers ---------------------------------------------------------------- */

static bool is_digit(char c) { return c >= '0' && c <= '9'; }

static ath_status parse_number(parser *p, ath_json **out)
{
    size_t start = p->pos, i = p->pos;
    bool neg = false, integral = true;
    if (i < p->n && p->s[i] == '-') {
        neg = true;
        i++;
    }
    if (i >= p->n || !is_digit(p->s[i])) {
        p->pos = i;
        return fail(p, "invalid number");
    }
    if (p->s[i] == '0') {
        i++;
        if (i < p->n && is_digit(p->s[i])) {
            p->pos = i;
            return fail(p, "leading zero in number");
        }
    } else {
        while (i < p->n && is_digit(p->s[i]))
            i++;
    }
    if (i < p->n && p->s[i] == '.') {
        integral = false;
        i++;
        if (i >= p->n || !is_digit(p->s[i])) {
            p->pos = i;
            return fail(p, "digit expected after decimal point");
        }
        while (i < p->n && is_digit(p->s[i]))
            i++;
    }
    if (i < p->n && (p->s[i] == 'e' || p->s[i] == 'E')) {
        integral = false;
        i++;
        if (i < p->n && (p->s[i] == '+' || p->s[i] == '-'))
            i++;
        if (i >= p->n || !is_digit(p->s[i])) {
            p->pos = i;
            return fail(p, "digit expected in exponent");
        }
        while (i < p->n && is_digit(p->s[i]))
            i++;
    }
    p->pos = i;

    if (integral) {
        /* Accumulate toward the sign so that INT64_MIN is representable. */
        int64_t v = 0;
        bool overflow = false;
        for (size_t k = start + (neg ? 1 : 0); k < i; k++) {
            int d = p->s[k] - '0';
            if (neg) {
                if (v < (INT64_MIN + d) / 10) { overflow = true; break; }
                v = v * 10 - d;
            } else {
                if (v > (INT64_MAX - d) / 10) { overflow = true; break; }
                v = v * 10 + d;
            }
        }
        if (!overflow) {
            ath_json *j = new_node(p, ATH_JSON_INT);
            if (!j)
                return oom(p);
            j->u.i = v;
            *out = j;
            return ATH_OK;
        }
        /* fall through: integers beyond int64 become doubles */
    }

    /* strtod needs a NUL-terminated copy. It honors the C locale, which is
     * the locale of a program that never calls setlocale(). Values beyond the
     * double range become +-inf, as in most JSON readers. */
    size_t len = i - start;
    char small[64];
    char *tmp = small;
    if (len >= sizeof small) {
        tmp = ath_arena_alloc(p->a, len + 1, 1);
        if (!tmp)
            return oom(p);
    }
    memcpy(tmp, p->s + start, len);
    tmp[len] = '\0';
    ath_json *j = new_node(p, ATH_JSON_DOUBLE);
    if (!j)
        return oom(p);
    j->u.d = strtod(tmp, NULL);
    *out = j;
    return ATH_OK;
}

/* ---- values ----------------------------------------------------------------- */

static ath_status parse_value(parser *p, ath_json **out);

static bool match_word(parser *p, const char *w)
{
    size_t l = strlen(w);
    if (p->n - p->pos < l || memcmp(p->s + p->pos, w, l) != 0)
        return false;
    p->pos += l;
    return true;
}

static ath_status parse_container(parser *p, ath_json **out, bool is_object)
{
    char close = is_object ? '}' : ']';
    if (++p->depth > ATH_JSON_MAX_DEPTH)
        return fail(p, "nesting too deep");
    ath_json *node = new_node(p, is_object ? ATH_JSON_OBJECT : ATH_JSON_ARRAY);
    if (!node)
        return oom(p);
    p->pos++; /* opening bracket */
    skip_ws(p);
    if (p->pos < p->n && p->s[p->pos] == close) {
        p->pos++;
        p->depth--;
        *out = node;
        return ATH_OK;
    }

    ath_json *tail = NULL;
    for (;;) {
        const char *key = NULL;
        size_t key_len = 0;
        skip_ws(p);
        if (is_object) {
            if (p->pos >= p->n || p->s[p->pos] != '"')
                return fail(p, "object key expected");
            ATH_CHECK(parse_string(p, &key, &key_len));
            skip_ws(p);
            if (p->pos >= p->n || p->s[p->pos] != ':')
                return fail(p, "':' expected after object key");
            p->pos++;
        }
        ath_json *child;
        ATH_CHECK(parse_value(p, &child));
        child->key = key;
        child->key_len = key_len;
        if (tail)
            tail->next = child;
        else
            node->u.kids.first = child;
        tail = child;
        node->u.kids.count++;

        skip_ws(p);
        if (p->pos >= p->n)
            return fail(p, is_object ? "unterminated object" : "unterminated array");
        char c = p->s[p->pos];
        if (c == ',') {
            p->pos++;
            continue;
        }
        if (c == close) {
            p->pos++;
            break;
        }
        return fail(p, is_object ? "',' or '}' expected" : "',' or ']' expected");
    }
    p->depth--;
    *out = node;
    return ATH_OK;
}

static ath_status parse_value(parser *p, ath_json **out)
{
    skip_ws(p);
    if (p->pos >= p->n)
        return fail(p, "value expected");
    char c = p->s[p->pos];
    switch (c) {
    case '{':
        return parse_container(p, out, true);
    case '[':
        return parse_container(p, out, false);
    case '"': {
        ath_json *j = new_node(p, ATH_JSON_STRING);
        if (!j)
            return oom(p);
        ATH_CHECK(parse_string(p, &j->u.str.ptr, &j->u.str.len));
        *out = j;
        return ATH_OK;
    }
    case 't':
    case 'f':
    case 'n': {
        ath_json_type type;
        bool b = false;
        if (match_word(p, "true")) {
            type = ATH_JSON_BOOL;
            b = true;
        } else if (match_word(p, "false")) {
            type = ATH_JSON_BOOL;
        } else if (match_word(p, "null")) {
            type = ATH_JSON_NULL;
        } else {
            return fail(p, "invalid literal");
        }
        ath_json *j = new_node(p, type);
        if (!j)
            return oom(p);
        j->u.b = b;
        *out = j;
        return ATH_OK;
    }
    default:
        if (c == '-' || is_digit(c))
            return parse_number(p, out);
        return fail(p, "unexpected character");
    }
}

ath_status ath_json_parse(ath_arena *a, const char *text, size_t len,
                          const ath_json **out, ath_json_error *err)
{
    parser p = {.s = text, .n = len, .a = a, .status = ATH_OK};
    ath_json *root = NULL;
    ath_status s = parse_value(&p, &root);
    if (s == ATH_OK) {
        skip_ws(&p);
        if (p.pos != p.n)
            s = fail(&p, "unexpected data after the root value");
    }
    if (s != ATH_OK) {
        if (err) {
            size_t line = 1, col = 1;
            for (size_t i = 0; i < p.err_pos && i < len; i++) {
                if (text[i] == '\n') {
                    line++;
                    col = 1;
                } else {
                    col++;
                }
            }
            err->offset = p.err_pos;
            err->line = line;
            err->col = col;
            err->msg = p.err ? p.err : "parse error";
        }
        return s;
    }
    *out = root;
    return ATH_OK;
}

/* ---- navigation ------------------------------------------------------------- */

const ath_json *ath_json_first(const ath_json *n)
{
    if (!n || (n->type != ATH_JSON_ARRAY && n->type != ATH_JSON_OBJECT))
        return NULL;
    return n->u.kids.first;
}

size_t ath_json_count(const ath_json *n)
{
    if (!n || (n->type != ATH_JSON_ARRAY && n->type != ATH_JSON_OBJECT))
        return 0;
    return n->u.kids.count;
}

static const ath_json *get_n(const ath_json *obj, const char *key, size_t len)
{
    if (!obj || obj->type != ATH_JSON_OBJECT)
        return NULL;
    for (const ath_json *c = obj->u.kids.first; c; c = c->next)
        if (c->key_len == len && memcmp(c->key, key, len) == 0)
            return c;
    return NULL;
}

const ath_json *ath_json_get(const ath_json *obj, const char *key)
{
    return get_n(obj, key, strlen(key));
}

const ath_json *ath_json_at(const ath_json *arr, size_t i)
{
    if (!arr || arr->type != ATH_JSON_ARRAY)
        return NULL;
    const ath_json *c = arr->u.kids.first;
    while (c && i--)
        c = c->next;
    return c;
}

const ath_json *ath_json_path(const ath_json *root, const char *path)
{
    const ath_json *n = root;
    const char *seg = path;
    while (n) {
        const char *dot = strchr(seg, '.');
        size_t len = dot ? (size_t)(dot - seg) : strlen(seg);
        if (n->type == ATH_JSON_ARRAY) {
            if (len == 0 || len > 18)
                return NULL;
            size_t idx = 0;
            for (size_t k = 0; k < len; k++) {
                if (!is_digit(seg[k]))
                    return NULL;
                idx = idx * 10 + (size_t)(seg[k] - '0');
            }
            n = ath_json_at(n, idx);
        } else {
            n = get_n(n, seg, len);
        }
        if (!dot)
            return n;
        seg = dot + 1;
    }
    return NULL;
}

/* ---- typed access ----------------------------------------------------------- */

ath_status ath_json_to_int(const ath_json *n, int64_t *out)
{
    if (!n)
        return ATH_ERR_NOT_FOUND;
    if (n->type != ATH_JSON_INT)
        return ATH_ERR_TYPE;
    *out = n->u.i;
    return ATH_OK;
}

ath_status ath_json_to_double(const ath_json *n, double *out)
{
    if (!n)
        return ATH_ERR_NOT_FOUND;
    if (n->type == ATH_JSON_INT)
        *out = (double)n->u.i;
    else if (n->type == ATH_JSON_DOUBLE)
        *out = n->u.d;
    else
        return ATH_ERR_TYPE;
    return ATH_OK;
}

ath_status ath_json_to_bool(const ath_json *n, bool *out)
{
    if (!n)
        return ATH_ERR_NOT_FOUND;
    if (n->type != ATH_JSON_BOOL)
        return ATH_ERR_TYPE;
    *out = n->u.b;
    return ATH_OK;
}

ath_status ath_json_to_str(const ath_json *n, const char **s, size_t *len)
{
    if (!n)
        return ATH_ERR_NOT_FOUND;
    if (n->type != ATH_JSON_STRING)
        return ATH_ERR_TYPE;
    *s = n->u.str.ptr;
    if (len)
        *len = n->u.str.len;
    return ATH_OK;
}

const char *ath_json_type_name(ath_json_type t)
{
    switch (t) {
    case ATH_JSON_NULL:   return "null";
    case ATH_JSON_BOOL:   return "bool";
    case ATH_JSON_INT:    return "int";
    case ATH_JSON_DOUBLE: return "double";
    case ATH_JSON_STRING: return "string";
    case ATH_JSON_ARRAY:  return "array";
    case ATH_JSON_OBJECT: return "object";
    }
    return "unknown";
}
