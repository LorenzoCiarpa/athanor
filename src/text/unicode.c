/* unicode.c: UTF-8, character classes and NFC. See unicode.h.
 *
 * NFC follows UAX #15: decompose every code point fully (the tables hold
 * full canonical decompositions; Hangul syllables are computed), sort each
 * run of non-starters by combining class (stable), then compose: a code
 * point combines with the last starter if it is not blocked, i.e. no code
 * point between them has a combining class of zero or one at least as high.
 * Every code point below U+0300 is NFC on its own and a starter, and nothing
 * composes with it as a second element, so text made of those is returned
 * as it is.
 */
#include "text/unicode.h"

#include <string.h>

#include "text/unicode_tables.h"

/* Hangul syllables (Unicode, section 3.12). */
enum {
    S_BASE = 0xAC00, L_BASE = 0x1100, V_BASE = 0x1161, T_BASE = 0x11A7,
    L_COUNT = 19, V_COUNT = 21, T_COUNT = 28, N_COUNT = V_COUNT * T_COUNT, S_COUNT = L_COUNT * N_COUNT
};

/* An invalid byte b is carried through NFC as INVALID_BASE | b. */
#define INVALID_BASE 0x80000000u

const char *ath_unicode_version(void) { return ATH_UNICODE_VERSION; }

size_t ath_utf8_decode(const char *str, size_t n, uint32_t *cp)
{
    const unsigned char *s = (const unsigned char *)str;
    unsigned char b = s[0];
    if (b < 0x80) {
        *cp = b;
        return 1;
    }
    size_t len;
    uint32_t c, min;
    if (b >= 0xC2 && b <= 0xDF) {
        len = 2, c = b & 0x1F, min = 0x80;
    } else if (b >= 0xE0 && b <= 0xEF) {
        len = 3, c = b & 0x0F, min = 0x800;
    } else if (b >= 0xF0 && b <= 0xF4) {
        len = 4, c = b & 0x07, min = 0x10000;
    } else {
        *cp = ATH_UTF8_INVALID;
        return 1;
    }
    if (n < len) {
        *cp = ATH_UTF8_INVALID;
        return 1;
    }
    for (size_t i = 1; i < len; i++) {
        if ((s[i] & 0xC0) != 0x80) {
            *cp = ATH_UTF8_INVALID;
            return 1;
        }
        c = c << 6 | (s[i] & 0x3F);
    }
    if (c < min || c > 0x10FFFF || (c >= 0xD800 && c <= 0xDFFF)) {
        *cp = ATH_UTF8_INVALID;
        return 1;
    }
    *cp = c;
    return len;
}

size_t ath_utf8_encode(uint32_t cp, char out[4])
{
    if (cp < 0x80) {
        out[0] = (char)cp;
        return 1;
    }
    if (cp < 0x800) {
        out[0] = (char)(0xC0 | cp >> 6);
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        out[0] = (char)(0xE0 | cp >> 12);
        out[1] = (char)(0x80 | (cp >> 6 & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    out[0] = (char)(0xF0 | cp >> 18);
    out[1] = (char)(0x80 | (cp >> 12 & 0x3F));
    out[2] = (char)(0x80 | (cp >> 6 & 0x3F));
    out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

/* The value of the range holding cp, 0 if none. */
static uint8_t range_value(const uc_range *r, size_t n, uint32_t cp)
{
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (cp < r[mid].lo)
            hi = mid;
        else if (cp > r[mid].hi)
            lo = mid + 1;
        else
            return r[mid].v;
    }
    return 0;
}

ath_uc_class ath_unicode_class(uint32_t cp)
{
    if (cp < 0x80) {
        if ((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z'))
            return ATH_UC_LETTER;
        if (cp >= '0' && cp <= '9')
            return ATH_UC_NUMBER;
        return cp == ' ' || (cp >= 0x09 && cp <= 0x0D) ? ATH_UC_SPACE : ATH_UC_OTHER;
    }
    return (ath_uc_class)range_value(uc_classes, ATH_ARRAY_LEN(uc_classes), cp);
}

uint8_t ath_unicode_ccc(uint32_t cp)
{
    return cp < 0x300 || cp > 0x10FFFF ? 0 : range_value(uc_ccc, ATH_ARRAY_LEN(uc_ccc), cp);
}

static const uc_decomp *find_decomp(uint32_t cp)
{
    size_t lo = 0, hi = ATH_ARRAY_LEN(uc_decomp_index);
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (cp < uc_decomp_index[mid].cp)
            hi = mid;
        else if (cp > uc_decomp_index[mid].cp)
            lo = mid + 1;
        else
            return &uc_decomp_index[mid];
    }
    return NULL;
}

/* The primary composite of a and b, or 0. */
static uint32_t compose(uint32_t a, uint32_t b)
{
    if (a >= L_BASE && a < L_BASE + L_COUNT && b >= V_BASE && b < V_BASE + V_COUNT)
        return S_BASE + ((a - L_BASE) * V_COUNT + (b - V_BASE)) * T_COUNT;
    if (a >= S_BASE && a < S_BASE + S_COUNT && (a - S_BASE) % T_COUNT == 0 && b > T_BASE &&
        b < T_BASE + T_COUNT)
        return a + (b - T_BASE);
    size_t lo = 0, hi = ATH_ARRAY_LEN(uc_compose);
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        const uc_pair *p = &uc_compose[mid];
        if (a < p->a || (a == p->a && b < p->b))
            hi = mid;
        else if (a > p->a || b > p->b)
            lo = mid + 1;
        else
            return p->c;
    }
    return 0;
}

static uint8_t ccc_of(uint32_t c) { return c & INVALID_BASE ? 0 : ath_unicode_ccc(c); }

ath_status ath_nfc(const char *s, size_t n, ath_arena *a, const char **out, size_t *out_len)
{
    /* Quick check: below U+0300 every code point is NFC and a starter. */
    bool quick = true;
    for (size_t i = 0; i < n && quick; i++)
        quick = (unsigned char)s[i] < 0xCC;   /* lead bytes up to 0xCB encode U+0000 .. U+02FF */
    if (quick) {
        *out = s;
        *out_len = n;
        return ATH_OK;
    }

    /* Decompose into code points. A byte of input gives at most one code
     * point before decomposition; decompositions can make it longer, so the
     * buffer grows when needed. */
    size_t cap = n + 16;
    uint32_t *cp = ath_arena_alloc(a, cap * sizeof *cp, _Alignof(uint32_t));
    if (!cp)
        return ATH_ERR_NOMEM;
    size_t m = 0;
    for (size_t i = 0; i < n;) {
        uint32_t c;
        size_t len = ath_utf8_decode(s + i, n - i, &c);
        if (c == ATH_UTF8_INVALID)
            c = INVALID_BASE | (unsigned char)s[i];
        i += len;
        const uc_decomp *d = c >= 0xC0 && !(c & INVALID_BASE) ? find_decomp(c) : NULL;
        size_t need = d ? d->len : 3;
        if (m + need > cap) {
            size_t ncap = 2 * cap + need;
            uint32_t *g = ath_arena_alloc(a, ncap * sizeof *g, _Alignof(uint32_t));
            if (!g)
                return ATH_ERR_NOMEM;
            memcpy(g, cp, m * sizeof *g);
            cp = g, cap = ncap;
        }
        if (c >= S_BASE && c < S_BASE + S_COUNT) {
            uint32_t si = c - S_BASE;
            cp[m++] = L_BASE + si / N_COUNT;
            cp[m++] = V_BASE + si % N_COUNT / T_COUNT;
            if (si % T_COUNT)
                cp[m++] = T_BASE + si % T_COUNT;
        } else if (d) {
            for (uint8_t k = 0; k < d->len; k++)
                cp[m++] = uc_decomp_pool[d->off + k];
        } else {
            cp[m++] = c;
        }
    }

    /* Canonical ordering: a stable insertion sort of each run of non-starters. */
    for (size_t i = 1; i < m; i++) {
        uint8_t ci = ccc_of(cp[i]);
        if (!ci)
            continue;
        for (size_t j = i; j > 0; j--) {
            uint8_t cj = ccc_of(cp[j - 1]);
            if (!cj || cj <= ci)
                break;
            uint32_t t = cp[j];
            cp[j] = cp[j - 1];
            cp[j - 1] = t;
        }
    }

    /* Canonical composition, in place. */
    size_t w = 0;
    if (m) {
        size_t starter = 0;
        unsigned last = ccc_of(cp[0]) ? 256 : 0;   /* text that starts with a non-starter */
        w = 1;
        for (size_t i = 1; i < m; i++) {
            uint32_t c = cp[i];
            unsigned cc = ccc_of(c);
            uint32_t comp = (c | cp[starter]) & INVALID_BASE ? 0 : compose(cp[starter], c);
            if (comp && (last < cc || last == 0) && last != 256) {
                cp[starter] = comp;
                continue;
            }
            if (cc == 0)
                starter = w;
            last = cc;
            cp[w++] = c;
        }
    }

    char *o = ath_arena_alloc(a, w * 4 + 1, 1);
    if (!o)
        return ATH_ERR_NOMEM;
    size_t len = 0;
    for (size_t i = 0; i < w; i++) {
        if (cp[i] & INVALID_BASE)
            o[len++] = (char)(cp[i] & 0xFF);
        else
            len += ath_utf8_encode(cp[i], o + len);
    }
    *out = o;
    *out_len = len;
    return ATH_OK;
}
