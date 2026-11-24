/* test_text.c: text handling. Unicode and NFC, the JSON writer, the
 * tokenizer, the chat template, the output parser; and, on the real
 * checkpoint, against the reference's golden data (tests/golden/text.py):
 * token ids of a corpus, rendered prompts, and a greedy chat generation.
 *
 * The family is reached only through the contract (ath_family_at), as a
 * pipeline would.
 */
#include "test.h"
#include "util.h"

#include "core/buf.h"
#include "core/file.h"
#include "core/json.h"
#include "core/rng.h"
#include "core/safetensors.h"
#include "model/model.h"
#include "ops/ops.h"
#include "text/bpe.h"
#include "text/unicode.h"

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

static const char *golden_dir(void)
{
    const char *d = getenv("ATH_GOLDEN_DIR");
    return (d && *d) ? d : "tests/golden/data/qwen35-0.8b";
}

/* A model that has only its family: enough for the text functions. */
static ath_model family_only(void)
{
    ath_model m;
    memset(&m, 0, sizeof m);
    m.family = ath_family_at(0);
    return m;
}

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* ---- Unicode ---------------------------------------------------------------------- */

ATH_TEST(utf8_round_trip)
{
    static const uint32_t cps[] = {0, 0x41, 0x7F, 0x80, 0x7FF, 0x800, 0xD7FF, 0xE000, 0xFFFD, 0xFFFF, 0x10000, 0x10FFFF};
    for (size_t i = 0; i < ATH_ARRAY_LEN(cps); i++) {
        char b[4];
        size_t n = ath_utf8_encode(cps[i], b);
        uint32_t c;
        ATH_EXPECT_EQ_INT(ath_utf8_decode(b, n, &c), n);
        ATH_EXPECT_EQ_U64(c, cps[i]);
    }
    /* invalid: stray continuation, overlong, surrogate, beyond U+10FFFF, truncated */
    static const char *const bad[] = {"\x80", "\xC0\xAF", "\xC1\xBF", "\xE0\x80\xAF", "\xED\xA0\x80",
                                      "\xF4\x90\x80\x80", "\xF5\x80\x80\x80", "\xE2\x82", "\xFF"};
    for (size_t i = 0; i < ATH_ARRAY_LEN(bad); i++) {
        uint32_t c;
        ATH_EXPECT_EQ_INT(ath_utf8_decode(bad[i], strlen(bad[i]), &c), 1);
        ATH_EXPECT_EQ_U64(c, ATH_UTF8_INVALID);
    }
}

ATH_TEST(unicode_classes)
{
    ATH_EXPECT_STR_EQ(ath_unicode_version(), "16.0.0");
    ATH_EXPECT_EQ_INT(ath_unicode_class('a'), ATH_UC_LETTER);
    ATH_EXPECT_EQ_INT(ath_unicode_class(0x4E2D), ATH_UC_LETTER);   /* 中 */
    ATH_EXPECT_EQ_INT(ath_unicode_class(0x1C89), ATH_UC_LETTER);   /* new in Unicode 16 */
    ATH_EXPECT_EQ_INT(ath_unicode_class(0x301), ATH_UC_MARK);
    ATH_EXPECT_EQ_INT(ath_unicode_class(0x0903), ATH_UC_MARK);     /* Mc */
    ATH_EXPECT_EQ_INT(ath_unicode_class('7'), ATH_UC_NUMBER);
    ATH_EXPECT_EQ_INT(ath_unicode_class(0x2167), ATH_UC_NUMBER);   /* Ⅷ, Nl */
    ATH_EXPECT_EQ_INT(ath_unicode_class(0xBD), ATH_UC_NUMBER);     /* ½, No */
    ATH_EXPECT_EQ_INT(ath_unicode_class(' '), ATH_UC_SPACE);
    ATH_EXPECT_EQ_INT(ath_unicode_class(0x85), ATH_UC_SPACE);
    ATH_EXPECT_EQ_INT(ath_unicode_class(0x3000), ATH_UC_SPACE);
    ATH_EXPECT_EQ_INT(ath_unicode_class(0x2029), ATH_UC_SPACE);
    ATH_EXPECT_EQ_INT(ath_unicode_class(0x180E), ATH_UC_OTHER);    /* no longer white space */
    ATH_EXPECT_EQ_INT(ath_unicode_class(0x200B), ATH_UC_OTHER);
    ATH_EXPECT_EQ_INT(ath_unicode_class('!'), ATH_UC_OTHER);
    ATH_EXPECT_EQ_INT(ath_unicode_class(0x1F600), ATH_UC_OTHER);   /* 😀 */
    ATH_EXPECT_EQ_INT(ath_unicode_ccc(0x301), 230);
    ATH_EXPECT_EQ_INT(ath_unicode_ccc(0x323), 220);
    ATH_EXPECT_EQ_INT(ath_unicode_ccc('a'), 0);
}

ATH_TEST(nfc)
{
    static const struct { const char *in, *want; } cases[] = {
        {"plain ascii", "plain ascii"},
        {"e\xCC\x81", "\xC3\xA9"},                                   /* e + acute → é */
        {"\xC3\xA9", "\xC3\xA9"},
        {"\xE2\x84\xAB", "\xC3\x85"},                                 /* Å ANGSTROM SIGN → Å */
        {"q\xCC\x87\xCC\xA3", "q\xCC\xA3\xCC\x87"},                   /* dot above, dot below: reordered */
        {"\xE1\x84\x80\xE1\x85\xA1\xE1\x86\xA8", "\xEA\xB0\x81"},    /* Hangul L V T → 각 */
        {"\xE0\xA4\x95\xE0\xA4\xBC", "\xE0\xA4\x95\xE0\xA4\xBC"},    /* क + nukta: क़ is excluded */
        {"\xE0\xA5\x98", "\xE0\xA4\x95\xE0\xA4\xBC"},                /* क़ decomposes and stays so */
        {"\xCC\x81x", "\xCC\x81x"},                                   /* leading mark */
        {"a\xCC\x80\xCC\x81", "\xC3\xA0\xCC\x81"},                   /* only the first mark composes */
        {"a\xFF\xCC\x81", "a\xFF\xCC\x81"},                           /* invalid byte kept, blocks nothing */
    };
    ath_arena a;
    ath_arena_init(&a, 0);
    for (size_t i = 0; i < ATH_ARRAY_LEN(cases); i++) {
        const char *out;
        size_t n;
        ATH_EXPECT_OK(ath_nfc(cases[i].in, strlen(cases[i].in), &a, &out, &n));
        if (n != strlen(cases[i].want) || memcmp(out, cases[i].want, n) != 0) {
            char msg[64];
            snprintf(msg, sizeof msg, "nfc case %zu", i);
            ath_test_fail_(t_, __FILE__, __LINE__, msg);
        }
    }
    ath_arena_free(&a);
}

/* ---- JSON writer -------------------------------------------------------------------- */

ATH_TEST(json_write_python_style)
{
    ath_arena a;
    ath_arena_init(&a, 0);
    static const struct { const char *in, *want; } cases[] = {
        {"{\"a\":1,\"b\":[true,false,null],\"c\":{}}", "{\"a\": 1, \"b\": [true, false, null], \"c\": {}}"},
        {"[1.0, 0.5, 1e-5, 1e16, 1e15, 123456789.0, 0.0001, 0.00001, -0.0, 1E+22, 2.5e-7, 3.14159]",
         "[1.0, 0.5, 1e-05, 1e+16, 1000000000000000.0, 123456789.0, 0.0001, 1e-05, -0.0, 1e+22, 2.5e-07, "
         "3.14159]"},
        {"[0.1, 0.2, 100, -7, 1.7976931348623157e308, 5e-324]", "[0.1, 0.2, 100, -7, 1.7976931348623157e+308, 5e-324]"},
        {"\"q\\\"b\\\\s\\/n\\n\\r\\t\\b\\f\\u0001\\u001f\\u007f é 😀\"",
         "\"q\\\"b\\\\s/n\\n\\r\\t\\b\\f\\u0001\\u001f\x7f é 😀\""},
        {"[]", "[]"},
    };
    for (size_t i = 0; i < ATH_ARRAY_LEN(cases); i++) {
        const ath_json *j;
        ATH_REQUIRE_OK(ath_json_parse(&a, cases[i].in, strlen(cases[i].in), &j, NULL));
        ath_buf b;
        ath_buf_init(&b, &a);
        ATH_EXPECT_OK(ath_json_write(j, &b));
        ATH_EXPECT_STR_EQ(ath_buf_str(&b), cases[i].want);
    }
    ath_arena_free(&a);
}

/* ---- pre-tokenization --------------------------------------------------------------- */

/* The pieces of text joined by '|'. */
static const char *pieces(ath_arena *a, const char *text)
{
    ath_model m = family_only();
    ath_spans sp;
    ath_spans_init(&sp, a);
    if (m.family->pretokenize(text, strlen(text), &sp) != ATH_OK)
        return "(error)";
    ath_buf b;
    ath_buf_init(&b, a);
    size_t start = 0;
    for (size_t i = 0; i < sp.n; i++) {
        if (i)
            ath_buf_putc(&b, '|');
        ath_buf_append(&b, text + start, sp.end[i] - start);
        start = sp.end[i];
    }
    return ath_buf_str(&b);
}

ATH_TEST(pretokenize_pieces)
{
    static const struct { const char *in, *want; } cases[] = {
        {"Hello world", "Hello| world"},
        {"I'm here, they'RE", "I|'m| here|,| they|'RE"},
        {"'sure 'Sure", "'s|ure| '|Sure"},
        {"x 123", "x| |1|2|3"},
        {"a  b", "a| | b"},
        {"a\n\nb", "a|\n\n|b"},
        {"a \n b", "a| \n| b"},
        {"end  ", "end|  "},
        {"x \t\n\t y", "x| \t\n|\t| y"},
        {"!!! ...x", "!!!| ...|x"},
        {"(foo)", "(foo|)"},
        {"a.\n\nb", "a|.\n\n|b"},
        {"\xCC\x81x \xCC\x81", "\xCC\x81x| \xCC\x81"},          /* a mark takes the optional slot */
        {"\xE4\xB8\xAD\xE6\x96\x87", "\xE4\xB8\xAD\xE6\x96\x87"},  /* 中文 */
        {"a\xFF" "b", "a|\xFF" "b"},                              /* an invalid byte is "other" */
        {"", ""},
    };
    ath_arena a;
    ath_arena_init(&a, 0);
    for (size_t i = 0; i < ATH_ARRAY_LEN(cases); i++)
        ATH_EXPECT_STR_EQ(pieces(&a, cases[i].in), cases[i].want);
    ath_arena_free(&a);
}

/* ---- output parser ------------------------------------------------------------------ */

/* The events of an output as one string: T"text" H"thinking" B(name)
 * A"args" E, consecutive events of one kind merged. */
static const char *events_str(ath_arena *a, const char *const *pieces, size_t n_pieces, uint32_t opts,
                              const ath_tool_def *tools, int32_t n_tools)
{
    ath_model m = family_only();
    ath_parse_state st = {.opts = opts, .tools = tools, .n_tools = n_tools, .a = a};
    ath_buf b;
    ath_buf_init(&b, a);
    int last = -1;
    for (size_t k = 0; k <= n_pieces; k++) {
        const char *piece = k < n_pieces ? pieces[k] : NULL;
        size_t n = piece ? strlen(piece) : 0;
        ath_out_event ev[64];
        int32_t ne;
        if (ath_model_parse(&m, &st, piece, n, ev, (int32_t)ATH_ARRAY_LEN(ev), &ne) != ATH_OK)
            return "(error)";
        for (int32_t i = 0; i < ne; i++) {
            static const char kinds[] = "THBAE";
            int kind = (int)ev[i].kind;
            bool data = ev[i].kind == ATH_OUT_TEXT || ev[i].kind == ATH_OUT_THINKING ||
                        ev[i].kind == ATH_OUT_TOOL_CALL_ARGS;
            if (data && kind == last) {
                b.len--;   /* reopen the quoted string */
            } else {
                ath_buf_putc(&b, kinds[kind]);
                if (ev[i].kind == ATH_OUT_TOOL_CALL_BEGIN)
                    ath_buf_printf(&b, "(%s)", ev[i].tool_name);
                if (data)
                    ath_buf_putc(&b, '"');
            }
            if (data) {
                ath_buf_append(&b, ev[i].data, ev[i].len);
                ath_buf_putc(&b, '"');
            }
            last = data ? kind : -1;
        }
    }
    return ath_buf_str(&b);
}

typedef struct {
    const char *name;
    uint32_t    opts;
    const char *out;
    const char *want;
} parse_case;

static const parse_case parse_cases[] = {
    {"text", 0, "Hello, world.", "T\"Hello, world.\""},
    {"leading newlines dropped", 0, "\n\nHi\n\nthere\n", "T\"Hi\n\nthere\n\""},
    {"thinking", ATH_CHAT_THINKING, "Let me think.\nOk.\n</think>\n\nThe answer is 4.",
     "H\"Let me think.\nOk.\"T\"The answer is 4.\""},
    {"thinking unterminated", ATH_CHAT_THINKING, "still thinking <b>", "H\"still thinking <b>\""},
    {"think tag in text", 0, "<think>\nhmm\n</think>\n\nyes", "H\"hmm\"T\"yes\""},
    {"less-than in text", 0, "a < b and <b>bold</b> <tool", "T\"a < b and <b>bold</b> <tool\""},
    {"tool call", 0,
     "Checking.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
     "<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>",
     "T\"Checking.\"B(get_weather)A\"{\"city\": \"Paris\", \"days\": 3}\"E"},
    {"two calls", 0,
     "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>\n"
     "<tool_call>\n<function=now>\n</function>\n</tool_call>",
     "B(get_weather)A\"{\"city\": \"Paris\"}\"EB(now)A\"{}\"E"},
    {"typed values", 0,
     "<tool_call>\n<function=search_files>\n<parameter=pattern>\n\"a\" \\ b\nline 2\n</parameter>\n"
     "<parameter=limit>\n10\n</parameter>\n<parameter=score>\n0.5e1\n</parameter>\n"
     "<parameter=recursive>\nTrue\n</parameter>\n<parameter=paths>\n[\"src\",  \"docs\"]\n</parameter>\n"
     "<parameter=options>\nnot json\n</parameter>\n<parameter=unknown>\n42\n</parameter>\n"
     "</function>\n</tool_call>",
     "B(search_files)A\"{\"pattern\": \"\\\"a\\\" \\\\ b\\nline 2\", \"limit\": 10, \"score\": 0.5e1, "
     "\"recursive\": true, \"paths\": [\"src\", \"docs\"], \"options\": \"not json\", \"unknown\": \"42\"}\"E"},
    {"null and bad number", 0,
     "<tool_call>\n<function=search_files>\n<parameter=limit>\nnull\n</parameter>\n<parameter=score>\n1.\n"
     "</parameter>\n</function>\n</tool_call>",
     "B(search_files)A\"{\"limit\": null, \"score\": \"1.\"}\"E"},
    {"malformed call is text", 0, "x <tool_call>\nnot a function\n</tool_call> y",
     "T\"x <tool_call>\nnot a function\n</tool_call> y\""},
    {"truncated call", 0, "<tool_call>\n<function=get_weather>\n<parameter=city>\nPar",
     "B(get_weather)A\"{\"city\": \"Par\"}\"E"},
    {"truncated before name", 0, "ok <tool_call>\n<funct", "T\"ok <tool_call>\n<funct\""},
    {"missing close parameter", 0,
     "<tool_call>\n<function=get_weather>\n<parameter=city>\nRome\n</function>\n</tool_call>after",
     "B(get_weather)A\"{\"city\": \"Rome\"}\"ET\"after\""},
    {"text after call", 0, "<tool_call>\n<function=now>\n</function>\n</tool_call>\n\nDone.",
     "B(now)A\"{}\"ET\"Done.\""},
    {"thinking then call", ATH_CHAT_THINKING,
     "I need the time.\n</think>\n\n<tool_call>\n<function=now>\n</function>\n</tool_call>",
     "H\"I need the time.\"B(now)A\"{}\"E"},
    {"tool tag inside thinking is thinking", ATH_CHAT_THINKING,
     "maybe <tool_call> later\n</think>\n\nno", "H\"maybe <tool_call> later\"T\"no\""},
};

static const ath_tool_def parse_tools[] = {
    {"get_weather", "w", "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"},\"days\":{\"type\":\"integer\"}}}"},
    {"search_files", NULL,
     "{\"type\":\"object\",\"properties\":{\"pattern\":{\"type\":\"string\"},\"limit\":{\"type\":[\"integer\",\"null\"]},"
     "\"score\":{\"type\":\"number\"},\"recursive\":{\"type\":\"boolean\"},\"paths\":{\"type\":\"array\"},"
     "\"options\":{\"type\":\"object\"}}}"},
    {"now", NULL, NULL},
};

ATH_TEST(parse_fixtures)
{
    ath_arena a;
    ath_arena_init(&a, 0);
    for (size_t c = 0; c < ATH_ARRAY_LEN(parse_cases); c++) {
        const parse_case *pc = &parse_cases[c];
        const char *whole[] = {pc->out};
        const char *got = events_str(&a, whole, 1, pc->opts, parse_tools, 3);
        if (strcmp(got, pc->want) != 0) {
            printf("  parse '%s'\n    got  %s\n    want %s\n", pc->name, got, pc->want);
            ath_test_fail_(t_, __FILE__, __LINE__, pc->name);
            continue;
        }
        /* The same output in two pieces split at every byte, and one byte at
         * a time, gives the same events. */
        size_t n = strlen(pc->out);
        for (size_t cut = 0; cut <= n; cut++) {
            char *head = ath_arena_strndup(&a, pc->out, cut);
            const char *two[] = {head, pc->out + cut};
            const char *g = events_str(&a, two, 2, pc->opts, parse_tools, 3);
            if (strcmp(g, pc->want) != 0) {
                printf("  parse '%s' split at %zu\n    got  %s\n    want %s\n", pc->name, cut, g, pc->want);
                ath_test_fail_(t_, __FILE__, __LINE__, pc->name);
                break;
            }
        }
        const char **bytes = ath_arena_alloc(&a, (n ? n : 1) * sizeof *bytes, _Alignof(const char *));
        for (size_t i = 0; i < n; i++)
            bytes[i] = ath_arena_strndup(&a, pc->out + i, 1);
        const char *g = events_str(&a, bytes, n, pc->opts, parse_tools, 3);
        if (strcmp(g, pc->want) != 0) {
            printf("  parse '%s' byte by byte\n    got  %s\n    want %s\n", pc->name, g, pc->want);
            ath_test_fail_(t_, __FILE__, __LINE__, pc->name);
        }
    }
    ath_arena_free(&a);
}

/* ---- chat template: errors and arguments (no golden data needed) ------------------ */

ATH_TEST(chat_errors)
{
    ath_model m = family_only();
    ath_arena a;
    ath_arena_init(&a, 0);
    ath_buf b;
    ath_buf_init(&b, &a);
    const ath_msg sys = {.role = ATH_MSG_SYSTEM, .content = "s"}, user = {.role = ATH_MSG_USER, .content = "u"};
    ATH_EXPECT_STATUS(ath_model_chat(&m, NULL, 0, NULL, 0, 0, &b), ATH_ERR_INVALID_ARG);
    ATH_EXPECT_STATUS(ath_model_chat(&m, &sys, 1, NULL, 0, 0, &b), ATH_ERR_INVALID_ARG);
    const ath_msg late[] = {user, sys};
    ATH_EXPECT_STATUS(ath_model_chat(&m, late, 2, NULL, 0, 0, &b), ATH_ERR_INVALID_ARG);
    const ath_tool_call bad_args = {.name = "f", .arguments = "[1, 2]"};
    const ath_msg with_call[] = {user, {.role = ATH_MSG_ASSISTANT, .tool_calls = &bad_args, .n_tool_calls = 1}};
    ATH_EXPECT_STATUS(ath_model_chat(&m, with_call, 2, NULL, 0, 0, &b), ATH_ERR_INVALID_ARG);
    ATH_EXPECT_STATUS(ath_model_chat(&m, &user, 1, NULL, 0, 4u, &b), ATH_ERR_INVALID_ARG);
    ath_buf_clear(&b);
    ATH_EXPECT_OK(ath_model_chat(&m, &user, 1, NULL, 0, ATH_CHAT_GENERATE, &b));
    ATH_EXPECT_STR_EQ(ath_buf_str(&b), "<|im_start|>user\nu<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    ath_arena_free(&a);
}

/* ---- sampling ------------------------------------------------------------------------ */

ATH_TEST(sample_rules)
{
    ath_arena a;
    ath_arena_init(&a, 0);
    float l[6] = {1.0f, 3.0f, 2.0f, 3.0f, -1.0f, 0.0f};
    ath_tensor t;
    ATH_REQUIRE_OK(ath_tensor_init(&t, ATH_DTYPE_F32, 1, (int64_t[]){6}, l, ATH_TIER_RAM));
    int32_t id;
    /* greedy: the largest, the lowest id of equals; u ignored */
    ATH_EXPECT_OK(ath_sample_ref(&t, &(ath_sampler){0, 0, 1, 0}, 0.99, &a, &id));
    ATH_EXPECT_EQ_INT(id, 1);
    ATH_EXPECT_OK(ath_sample_ref(&t, &(ath_sampler){1, 1, 1, 0}, 0.99, &a, &id));
    ATH_EXPECT_EQ_INT(id, 1);
    /* T = 1, top-k 3: candidates 1, 3, 2 with weights 1, 1, e^-1 */
    const double w2 = exp(-1.0), tot = 2 + w2;
    ath_sampler s = {1, 3, 1, 0};
    ATH_EXPECT_OK(ath_sample_ref(&t, &s, 0.0, &a, &id));
    ATH_EXPECT_EQ_INT(id, 1);
    ATH_EXPECT_OK(ath_sample_ref(&t, &s, 1.0 / tot + 1e-9, &a, &id));
    ATH_EXPECT_EQ_INT(id, 3);
    ATH_EXPECT_OK(ath_sample_ref(&t, &s, 2.0 / tot + 1e-9, &a, &id));
    ATH_EXPECT_EQ_INT(id, 2);
    /* top-p 0.5 keeps the first only (1 / tot = 0.42 < 0.5 keeps two) */
    s.top_p = 0.4f;
    ATH_EXPECT_OK(ath_sample_ref(&t, &s, 0.99, &a, &id));
    ATH_EXPECT_EQ_INT(id, 1);
    s.top_p = 0.5f;
    ATH_EXPECT_OK(ath_sample_ref(&t, &s, 0.99, &a, &id));
    ATH_EXPECT_EQ_INT(id, 3);
    /* min-p 0.5 drops e^-1 = 0.37 of the best */
    s = (ath_sampler){1, 0, 1, 0.5f};
    ATH_EXPECT_OK(ath_sample_ref(&t, &s, 0.99, &a, &id));
    ATH_EXPECT_EQ_INT(id, 3);
    /* no top-k: the whole vocabulary, sorted */
    s = (ath_sampler){1, 0, 1, 0};
    ATH_EXPECT_OK(ath_sample_ref(&t, &s, 0.999999, &a, &id));
    ATH_EXPECT_EQ_INT(id, 4);
    /* arguments */
    ATH_EXPECT_STATUS(ath_sample_ref(&t, &(ath_sampler){-1, 0, 1, 0}, 0.5, &a, &id), ATH_ERR_INVALID_ARG);
    ATH_EXPECT_STATUS(ath_sample_ref(&t, &(ath_sampler){1, 0, 0, 0}, 0.5, &a, &id), ATH_ERR_INVALID_ARG);
    ATH_EXPECT_STATUS(ath_sample_ref(&t, &(ath_sampler){1, 0, 1, 0}, 1.0, &a, &id), ATH_ERR_INVALID_ARG);
    ATH_EXPECT_EQ_U64(ath_arena_used(&a), 0);   /* scratch released */
    ath_arena_free(&a);
}

ATH_TEST(sample_distribution)
{
    ath_arena a;
    ath_arena_init(&a, 0);
    float l[4] = {0.0f, 1.0f, 2.0f, -50.0f};
    ath_tensor t;
    ATH_REQUIRE_OK(ath_tensor_init(&t, ATH_DTYPE_F32, 2, (int64_t[]){1, 4}, l, ATH_TIER_RAM));
    ath_rng r;
    ath_rng_init(&r, 42);
    int counts[4] = {0};
    const int N = 40000;
    ath_sampler s = {0.5f, 0, 1, 0};   /* T = 0.5: weights e^0, e^2, e^4 */
    for (int i = 0; i < N; i++) {
        int32_t id;
        ATH_REQUIRE_OK(ath_sample_ref(&t, &s, ath_rng_uniform(&r), &a, &id));
        counts[id]++;
    }
    const double z = 1 + exp(2.0) + exp(4.0);
    const double want[3] = {1 / z, exp(2.0) / z, exp(4.0) / z};
    for (int i = 0; i < 3; i++) {
        double sd = sqrt(want[i] * (1 - want[i]) / N);
        ATH_EXPECT_NEAR((double)counts[i] / N, want[i], 5 * sd);
    }
    ATH_EXPECT_EQ_INT(counts[3], 0);
    ath_arena_free(&a);
}

/* ---- golden data ---------------------------------------------------------------------- */

typedef struct {
    ath_arena       a;
    ath_file        mf;
    const ath_json *manifest;
    ath_safetensors data;
    bool            ok;
} golden;

static bool golden_open(golden *g)
{
    memset(g, 0, sizeof *g);
    g->mf.fd = -1;
    ath_arena_init(&g->a, 0);
    char path[ATH_PATH_MAX];
    if (ath_path_join(path, sizeof path, golden_dir(), "text.json") != ATH_OK || !ath_file_exists(path))
        return false;
    if (ath_file_open(&g->mf, path) != ATH_OK ||
        ath_json_parse(&g->a, g->mf.data, g->mf.size, &g->manifest, NULL) != ATH_OK)
        return false;
    if (ath_path_join(path, sizeof path, golden_dir(), "text.safetensors") != ATH_OK ||
        ath_safetensors_open_file(&g->data, &g->a, path) != ATH_OK)
        return false;
    g->ok = true;
    return true;
}

static void golden_close(golden *g)
{
    if (g->data.shards)
        ath_safetensors_close(&g->data);
    if (g->mf.fd >= 0)
        ath_file_close(&g->mf);
    ath_arena_free(&g->a);
}

/* A tensor of the golden file: its data and element count. */
static const void *tensor(const golden *g, const char *fmt, size_t i, size_t *n)
{
    char name[128];
    snprintf(name, sizeof name, fmt, i);
    const ath_safetensors_entry *e = ath_safetensors_find(&g->data, name);
    if (!e)
        return NULL;
    *n = (size_t)ath_tensor_numel(&e->t);
    return e->t.data;
}

static bool load_tokenizer(ath_arena *a, ath_tokenizer *tok)
{
    ath_model m = family_only();
    return ath_model_tokenizer(&m, model_dir(), tok, a) == ATH_OK;
}

ATH_TEST(golden_tokenizer_corpus)
{
    char path[ATH_PATH_MAX];
    if (!model_file(path, sizeof path, "tokenizer.json"))
        ATH_SKIP("no checkpoint in $ATH_MODEL_DIR");
    golden g;
    if (!golden_open(&g)) {
        golden_close(&g);
        ATH_SKIP("no text golden data (tests/golden/README.md: run tests/golden/text.py)");
    }
    ath_arena a, scratch;
    ath_arena_init(&a, 0);
    ath_arena_init(&scratch, 0);
    ath_tokenizer tok;
    double t0 = now_s();
    ATH_REQUIRE(load_tokenizer(&a, &tok));
    double t_load = now_s() - t0;

    const ath_json *corpus = ath_json_get(g.manifest, "corpus");
    size_t n_cases = ath_json_count(corpus), total = 0, bytes = 0;
    int bad = 0;
    t0 = now_s();
    for (size_t i = 0; i < n_cases; i++) {
        const char *name = "?";
        ath_json_to_str(ath_json_get(ath_json_at(corpus, i), "name"), &name, NULL);
        size_t n_text = 0, n_ids = 0, n_dec = 0;
        const char *text = tensor(&g, "corpus.%zu.text", i, &n_text);
        const int32_t *ids = tensor(&g, "corpus.%zu.ids", i, &n_ids);
        const char *dec = tensor(&g, "corpus.%zu.decoded", i, &n_dec);
        if (!text || !ids || !dec) {
            ath_test_fail_(t_, __FILE__, __LINE__, "corpus tensor missing");
            break;
        }
        ath_arena_pos pos = ath_arena_mark(&a);
        ath_tokens out;
        ath_tokens_init(&out, &a);
        ATH_EXPECT_OK(ath_tokenizer_encode(&tok, text, n_text, &out, &scratch));
        size_t k = 0;
        while (k < out.n && k < n_ids && out.ids[k] == ids[k])
            k++;
        if (k < out.n || k < n_ids) {
            if (bad++ < 5) {
                printf("  corpus '%s': %zu ids, want %zu; first difference at %zu: got %d, want %d\n", name, out.n,
                       n_ids, k, k < out.n ? out.ids[k] : -1, k < n_ids ? ids[k] : -1);
                ath_buf ctx;
                ath_buf_init(&ctx, &a);
                size_t from = k > 3 ? k - 3 : 0;
                ath_tokenizer_decode(&tok, ids + from, (k + 3 < n_ids ? k + 3 : n_ids) - from, &ctx);
                printf("    want around it: [%s]\n", ath_buf_str(&ctx));
                ath_buf_clear(&ctx);
                ath_tokenizer_decode(&tok, out.ids + from, (k + 3 < out.n ? k + 3 : out.n) - from, &ctx);
                printf("    got around it:  [%s]\n", ath_buf_str(&ctx));
            }
        }
        ath_buf back;
        ath_buf_init(&back, &a);
        ATH_EXPECT_OK(ath_tokenizer_decode(&tok, ids, n_ids, &back));
        if (back.len != n_dec || (n_dec && memcmp(back.data, dec, n_dec) != 0)) {
            if (bad++ < 5)
                printf("  corpus '%s': decoded text differs\n", name);
        }
        total += n_ids;
        bytes += n_text;
        ath_arena_reset(&a, pos);
    }
    double t_enc = now_s() - t0;
    if (bad)
        ath_test_fail_(t_, __FILE__, __LINE__, "token ids or decoded text differ from the reference");
    printf("  %zu texts, %.1f MB, %zu tokens identical to the reference; load %.2f s, encode %.2f s\n", n_cases,
           (double)bytes / 1e6, total, t_load, t_enc);
    ath_arena_free(&scratch);
    ath_arena_free(&a);
    golden_close(&g);
}

/* The messages and tools of a template record, as ath_msg and ath_tool_def. */
static bool template_inputs(ath_arena *a, const ath_json *rec, ath_msg **msgs, int32_t *n_msgs,
                            ath_tool_def **tools, int32_t *n_tools)
{
    const ath_json *jm = ath_json_get(rec, "messages"), *jt = ath_json_get(rec, "tools");
    *n_msgs = (int32_t)ath_json_count(jm);
    *n_tools = (int32_t)ath_json_count(jt);
    *msgs = ath_arena_zalloc(a, (size_t)(*n_msgs + 1) * sizeof **msgs, _Alignof(ath_msg));
    *tools = ath_arena_zalloc(a, (size_t)(*n_tools + 1) * sizeof **tools, _Alignof(ath_tool_def));
    int32_t i = 0;
    for (const ath_json *m = ath_json_first(jm); m; m = m->next, i++) {
        ath_msg *g = &(*msgs)[i];
        const char *role = "";
        ath_json_to_str(ath_json_get(m, "role"), &role, NULL);
        g->role = !strcmp(role, "system") ? ATH_MSG_SYSTEM : !strcmp(role, "user") ? ATH_MSG_USER
                : !strcmp(role, "assistant") ? ATH_MSG_ASSISTANT : ATH_MSG_TOOL;
        ath_json_to_str(ath_json_get(m, "content"), &g->content, NULL);
        ath_json_to_str(ath_json_get(m, "reasoning"), &g->reasoning, NULL);
        ath_json_to_str(ath_json_get(m, "tool_call_id"), &g->tool_call_id, NULL);
        const ath_json *calls = ath_json_get(m, "tool_calls");
        g->n_tool_calls = (int32_t)ath_json_count(calls);
        ath_tool_call *tc = ath_arena_zalloc(a, (size_t)(g->n_tool_calls + 1) * sizeof *tc, _Alignof(ath_tool_call));
        int32_t k = 0;
        for (const ath_json *c = ath_json_first(calls); c; c = c->next, k++) {
            ath_json_to_str(ath_json_get(c, "name"), &tc[k].name, NULL);
            ath_json_to_str(ath_json_get(c, "arguments"), &tc[k].arguments, NULL);
        }
        g->tool_calls = tc;
    }
    i = 0;
    for (const ath_json *t = ath_json_first(jt); t; t = t->next, i++) {
        ath_json_to_str(ath_json_get(t, "name"), &(*tools)[i].name, NULL);
        ath_json_to_str(ath_json_get(t, "description"), &(*tools)[i].description, NULL);
        ath_json_to_str(ath_json_get(t, "json_schema"), &(*tools)[i].json_schema, NULL);
    }
    return true;
}

ATH_TEST(golden_chat_template)
{
    char path[ATH_PATH_MAX];
    if (!model_file(path, sizeof path, "tokenizer.json"))
        ATH_SKIP("no checkpoint in $ATH_MODEL_DIR");
    golden g;
    if (!golden_open(&g)) {
        golden_close(&g);
        ATH_SKIP("no text golden data (tests/golden/README.md: run tests/golden/text.py)");
    }
    ath_model m = family_only();
    ath_arena a, scratch;
    ath_arena_init(&a, 0);
    ath_arena_init(&scratch, 0);
    ath_tokenizer tok;
    ATH_REQUIRE(load_tokenizer(&a, &tok));

    const ath_json *templates = ath_json_get(g.manifest, "templates");
    size_t n = ath_json_count(templates);
    int rendered = 0, errors = 0;
    for (size_t i = 0; i < n; i++) {
        const ath_json *rec = ath_json_at(templates, i);
        const char *name = "?";
        ath_json_to_str(ath_json_get(rec, "name"), &name, NULL);
        bool gen = false, think = false;
        ath_json_to_bool(ath_json_get(rec, "generate"), &gen);
        ath_json_to_bool(ath_json_get(rec, "thinking"), &think);
        ath_msg *msgs;
        ath_tool_def *tools;
        int32_t n_msgs, n_tools;
        template_inputs(&a, rec, &msgs, &n_msgs, &tools, &n_tools);
        uint32_t opts = (gen ? ATH_CHAT_GENERATE : 0) | (think ? ATH_CHAT_THINKING : 0);
        ath_buf b;
        ath_buf_init(&b, &a);
        ath_status s = ath_model_chat(&m, msgs, n_msgs, tools, n_tools, opts, &b);
        if (ath_json_get(rec, "error")) {
            if (s != ATH_ERR_INVALID_ARG) {
                printf("  template '%s': the reference raises an error, athanor returns %s\n", name,
                       ath_status_str(s));
                ath_test_fail_(t_, __FILE__, __LINE__, name);
            }
            errors++;
            continue;
        }
        size_t n_text = 0, n_ids = 0;
        const char *want = tensor(&g, "template.%zu.text", i, &n_text);
        const int32_t *ids = tensor(&g, "template.%zu.ids", i, &n_ids);
        ATH_REQUIRE(want && ids);
        if (s != ATH_OK || b.len != n_text || memcmp(b.data, want, n_text) != 0) {
            size_t k = 0;
            while (k < b.len && k < n_text && b.data[k] == want[k])
                k++;
            printf("  template '%s' (%s): differs at byte %zu of %zu\n    got  ...%.60s\n    want ...%.60s\n", name,
                   ath_status_str(s), k, n_text, ath_buf_str(&b) + (k > 20 ? k - 20 : 0),
                   want + (k > 20 ? k - 20 : 0));
            ath_test_fail_(t_, __FILE__, __LINE__, name);
            continue;
        }
        ath_tokens out;
        ath_tokens_init(&out, &a);
        ATH_EXPECT_OK(ath_tokenizer_encode(&tok, b.data, b.len, &out, &scratch));
        if (out.n != n_ids || memcmp(out.ids, ids, n_ids * sizeof *ids) != 0) {
            printf("  template '%s': token ids differ\n", name);
            ath_test_fail_(t_, __FILE__, __LINE__, name);
        }
        rendered++;
    }
    printf("  %d prompts byte-identical to the reference's chat template, %d errors where it raises\n", rendered,
           errors);
    ath_arena_free(&scratch);
    ath_arena_free(&a);
    golden_close(&g);
}

/* Greedy chat on the real checkpoint: the prompt, rendered and tokenized
 * by athanor, then decoding with bf16 activations; athanor's tokens must be
 * the fp32 reference's at every step until the first whose fp32 top-2
 * margin is within the bf16 noise (after such a step the two may part
 * legitimately). The decoded output goes through the parser. */
ATH_TEST(golden_chat_generate)
{
    char path[ATH_PATH_MAX];
    if (!model_file(path, sizeof path, "config.json"))
        ATH_SKIP("no checkpoint in $ATH_MODEL_DIR");
    golden g;
    if (!golden_open(&g)) {
        golden_close(&g);
        ATH_SKIP("no text golden data (tests/golden/README.md: run tests/golden/text.py)");
    }
    const ath_json *gen = ath_json_get(g.manifest, "generate");
    const ath_json *jprompt = ath_json_get(gen, "prompt_tokens"), *jtokens = ath_json_get(gen, "tokens"),
                   *jmargins = ath_json_get(gen, "margins");
    double noise = 0;
    ATH_REQUIRE(ath_json_to_double(ath_json_get(gen, "noise"), &noise) == ATH_OK);

    ath_arena a, scratch;
    ath_arena_init(&a, 0);
    ath_arena_init(&scratch, 0);
    ath_model m;
    ath_tokenizer tok;
    ATH_REQUIRE_OK(ath_model_open(&m, &a, model_dir()));
    ATH_REQUIRE_OK(ath_model_tokenizer(&m, model_dir(), &tok, &a));

    const ath_msg user = {.role = ATH_MSG_USER, .content = "Explain in two sentences why the sky is blue."};
    ath_buf prompt;
    ath_buf_init(&prompt, &a);
    ATH_REQUIRE_OK(ath_model_chat(&m, &user, 1, NULL, 0, ATH_CHAT_GENERATE, &prompt));
    ath_tokens ids;
    ath_tokens_init(&ids, &a);
    ATH_REQUIRE_OK(ath_tokenizer_encode(&tok, prompt.data, prompt.len, &ids, &scratch));
    ATH_REQUIRE(ids.n == ath_json_count(jprompt));
    for (size_t i = 0; i < ids.n; i++) {
        int64_t v = -1;
        ath_json_to_int(ath_json_at(jprompt, i), &v);
        ATH_EXPECT_EQ_INT(ids.ids[i], v);
    }

    const int steps = (int)ath_json_count(jtokens);
#ifdef ATH_DEBUG
    const int max_steps = 4;   /* sanitizers make the forward pass slow */
#else
    const int max_steps = steps;
#endif
    ath_pool pool = {.n_threads = 0};
    ATH_REQUIRE_OK(ath_pool_init(&pool, ath_pool_default_threads(), &a));
    ath_state st;
    ATH_REQUIRE_OK(ath_state_init(&st, &m, ATH_DTYPE_BF16, (int32_t)ids.n + steps + 1, &a));
    float *lg = ath_arena_alloc(&a, (size_t)m.vocab_size * sizeof *lg, _Alignof(float));
    ath_tensor logits;
    ATH_REQUIRE_OK(ath_tensor_init(&logits, ATH_DTYPE_F32, 2, (int64_t[]){1, m.vocab_size}, lg, ATH_TIER_RAM));
    ath_status s = ath_model_forward(&m, &(ath_batch){ids.ids, (int32_t)ids.n, &logits}, &st, NULL, &scratch, &pool);
    ATH_EXPECT_OK(s);

    int compared = 0, safe = 1, equal = 0;
    ath_buf text;
    ath_buf_init(&text, &a);
    for (int k = 0; k < max_steps && s == ATH_OK; k++) {
        int32_t next;
        ATH_REQUIRE_OK(ath_sample_ref(&logits, &(ath_sampler){0, 0, 1, 0}, 0, &scratch, &next));
        int64_t want = -1;
        double margin = 0;
        ath_json_to_int(ath_json_at(jtokens, (size_t)k), &want);
        ath_json_to_double(ath_json_at(jmargins, (size_t)k), &margin);
        equal += next == want;
        if (safe) {
            if (next != want) {
                printf("  step %d: athanor %d, reference %lld (margin %.3f, noise %.3f)\n", k, next, (long long)want,
                       margin, noise);
                ath_test_fail_(t_, __FILE__, __LINE__, "greedy token differs above the bf16 noise");
            }
            compared++;
            safe = margin > noise && next == want;
        }
        ATH_REQUIRE_OK(ath_tokenizer_decode(&tok, &next, 1, &text));
        s = ath_model_forward(&m, &(ath_batch){&next, 1, &logits}, &st, NULL, &scratch, &pool);
        ATH_EXPECT_OK(s);
    }
    /* The parser gives the decoded text back as one text event. */
    ath_parse_state ps = {.opts = 0, .a = &a};
    ath_out_event ev[8];
    int32_t ne = 0, ne2 = 0;
    ATH_EXPECT_OK(ath_model_parse(&m, &ps, text.data, text.len, ev, 8, &ne));
    ATH_EXPECT_OK(ath_model_parse(&m, &ps, NULL, 0, ev + ne, 8 - ne, &ne2));
    ATH_EXPECT_EQ_INT(ne + ne2, 1);
    ATH_EXPECT(ev[0].kind == ATH_OUT_TEXT && ev[0].len == text.len);
    printf("  %d of %d greedy tokens above the bf16 noise, all equal to the fp32 reference's; %d of %d equal in "
           "all: \"%s\"\n", compared, steps, equal, max_steps, ath_buf_str(&text));
    ath_pool_free(&pool);
    ath_model_close(&m);
    ath_arena_free(&scratch);
    ath_arena_free(&a);
    golden_close(&g);
}

/* ---- runner -------------------------------------------------------------------------- */

static const ath_test_case tests[] = {
    ATH_TEST_ENTRY(utf8_round_trip),
    ATH_TEST_ENTRY(unicode_classes),
    ATH_TEST_ENTRY(nfc),
    ATH_TEST_ENTRY(json_write_python_style),
    ATH_TEST_ENTRY(pretokenize_pieces),
    ATH_TEST_ENTRY(parse_fixtures),
    ATH_TEST_ENTRY(chat_errors),
    ATH_TEST_ENTRY(sample_rules),
    ATH_TEST_ENTRY(sample_distribution),
    ATH_TEST_ENTRY(golden_tokenizer_corpus),
    ATH_TEST_ENTRY(golden_chat_template),
    ATH_TEST_ENTRY(golden_chat_generate),
};

int main(int argc, char **argv)
{
    return ath_test_main(tests, ATH_ARRAY_LEN(tests), argc, argv);
}
