/* unicode.h: UTF-8, character classes and NFC, for tokenization.
 *
 * Pre-tokenization splits text by character class (letter, mark, number,
 * white space), and Hugging Face tokenizers normalize the text to NFC
 * first. Both need Unicode data; it is generated once from Python's
 * unicodedata by tools/dev/gen_unicode.py into unicode_tables.h (committed),
 * so no library is needed at build or run time.
 *
 * Text is UTF-8 bytes. Bytes that are not valid UTF-8 (a stray continuation
 * byte, an overlong form, a surrogate, a truncated sequence) are kept as
 * they are: each one decodes as ATH_UTF8_INVALID with length 1, belongs to
 * no class, and NFC copies it unchanged.
 */
#ifndef ATH_UNICODE_H
#define ATH_UNICODE_H

#include "core/arena.h"
#include "core/base.h"

#define ATH_UTF8_INVALID 0xFFFFFFFFu

typedef enum {
    ATH_UC_OTHER,
    ATH_UC_LETTER,   /* general category L*                                      */
    ATH_UC_MARK,     /* M*                                                        */
    ATH_UC_NUMBER,   /* N*                                                        */
    ATH_UC_SPACE,    /* white space as Oniguruma's \s: U+0009..U+000D, U+0085,
                      * and the categories Zs, Zl, Zp                            */
} ath_uc_class;

/* The Unicode version of the tables ("16.0.0"). */
const char *ath_unicode_version(void);

/* Decode the code point at s[0 .. n): returns its length in bytes (1 to 4)
 * and stores it in *cp, or returns 1 and ATH_UTF8_INVALID for a byte that
 * does not start a valid sequence. n >= 1. */
size_t ath_utf8_decode(const char *s, size_t n, uint32_t *cp);

/* Encode cp (at most 0x10FFFF, not a surrogate) into out; returns the length. */
size_t ath_utf8_encode(uint32_t cp, char out[4]);

ath_uc_class ath_unicode_class(uint32_t cp);

/* Canonical combining class (0 for starters and for ATH_UTF8_INVALID). */
uint8_t ath_unicode_ccc(uint32_t cp);

/* NFC (canonical decomposition, canonical ordering, canonical composition)
 * of s[0 .. n). *out is s itself when the text is already NFC by a quick
 * check (every code point below U+0300), otherwise a new buffer in a. The
 * result is not NUL-terminated. */
ath_status ath_nfc(const char *s, size_t n, ath_arena *a, const char **out, size_t *out_len);

#endif /* ATH_UNICODE_H */
