/* json.h: read-only JSON parser (RFC 8259) into an arena.
 *
 * Used for config.json, tokenizer files and safetensors headers. The input
 * does not need to be NUL-terminated, so it can be parsed straight out of an
 * mmap. Every node and every decoded string lives in the arena; the tree is
 * immutable and dies with the arena.
 *
 * Children of arrays and objects are a singly linked list in document order:
 *
 *   for (const ath_json *c = ath_json_first(obj); c; c = c->next)
 *       printf("%s\n", c->key);
 *
 * Strictness: only valid JSON is accepted. Strings must be valid UTF-8 with no
 * raw control characters; \u escapes must form valid code points (a lone
 * surrogate is rejected); numbers follow the JSON grammar exactly; nothing but
 * whitespace may follow the root value. Duplicate object keys are kept, and
 * lookups return the first one.
 */
#ifndef ATH_JSON_H
#define ATH_JSON_H

#include "core/arena.h"
#include "core/base.h"
#include "core/buf.h"

#define ATH_JSON_MAX_DEPTH 256

typedef enum {
    ATH_JSON_NULL,
    ATH_JSON_BOOL,
    ATH_JSON_INT,      /* integer literal that fits in int64_t        */
    ATH_JSON_DOUBLE,   /* fraction, exponent, or integer out of range */
    ATH_JSON_STRING,
    ATH_JSON_ARRAY,
    ATH_JSON_OBJECT
} ath_json_type;

typedef struct ath_json ath_json;
struct ath_json {
    ath_json_type type;
    const char   *key;       /* member name if the parent is an object, else NULL */
    size_t        key_len;   /* keys and strings may contain \u0000             */
    ath_json     *next;      /* next sibling, NULL for the last child           */
    union {
        bool    b;
        int64_t i;
        double  d;
        struct { const char *ptr; size_t len; } str;   /* NUL-terminated as well */
        struct { ath_json *first; size_t count; } kids;
    } u;
};

typedef struct {
    size_t      offset;   /* byte offset of the error in the input */
    size_t      line;     /* 1-based                               */
    size_t      col;      /* 1-based, in bytes                     */
    const char *msg;      /* static string                         */
} ath_json_error;

/* Parse text[0..len) into a tree allocated in a. err may be NULL.
 * ATH_ERR_PARSE on invalid JSON, ATH_ERR_NOMEM if the arena fails. */
ath_status ath_json_parse(ath_arena *a, const char *text, size_t len,
                          const ath_json **out, ath_json_error *err);

/* ---- navigation: all return NULL when the node is NULL, has the wrong type,
 *      or the child does not exist ------------------------------------------ */

/* Member of an object (first match). */
const ath_json *ath_json_get(const ath_json *obj, const char *key);
/* Element of an array, O(i). */
const ath_json *ath_json_at(const ath_json *arr, size_t i);
/* Dotted path through objects and arrays: "text_config.layer_types.3".
 * A segment made only of digits indexes an array. Keys containing '.'
 * cannot be reached this way; use ath_json_get. */
const ath_json *ath_json_path(const ath_json *root, const char *path);
/* First child of an array or object. */
const ath_json *ath_json_first(const ath_json *n);
/* Number of children of an array or object, 0 for anything else. */
size_t ath_json_count(const ath_json *n);

/* ---- typed access: ATH_ERR_NOT_FOUND if n is NULL, ATH_ERR_TYPE if the type
 *      does not match ------------------------------------------------------- */

ath_status ath_json_to_int(const ath_json *n, int64_t *out);
ath_status ath_json_to_double(const ath_json *n, double *out);   /* INT or DOUBLE */
ath_status ath_json_to_bool(const ath_json *n, bool *out);
ath_status ath_json_to_str(const ath_json *n, const char **s, size_t *len); /* len may be NULL */

const char *ath_json_type_name(ath_json_type t);

/* ---- writing ------------------------------------------------------------------
 *
 * The text Python's json.dumps(x, ensure_ascii=False) gives for the same
 * value, because chat templates (Jinja's tojson in Hugging Face
 * transformers) serialize with it: ", " between items, ": " after keys, no
 * indentation; strings escape '"', '\\', \b \f \n \r \t and other control
 * characters as \u00xx (lower-case hex) and keep every other character as
 * it is; doubles in Python's shortest round-trip form (1.0, 1e-05, 1e+16).
 * Keys keep their order; with duplicate keys Python would keep the last
 * value at the first key's place, this writer writes them all. */
ath_status ath_json_write(const ath_json *n, ath_buf *out);
/* A string value, quotes included. */
ath_status ath_json_write_str(ath_buf *out, const char *s, size_t n);
/* A double in Python's repr form (as json.dumps writes it). */
ath_status ath_json_write_double(ath_buf *out, double d);

#endif /* ATH_JSON_H */
