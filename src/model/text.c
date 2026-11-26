/* text.c: the text side of the contract: tokenizer, chat template, output
 * parser. See model.h.
 *
 * The family supplies the model-specific parts (pre-tokenization, the
 * template, the markup of its output); these functions check the arguments
 * once, for every family, like ath_model_forward does for the forward pass.
 */
#include "model/model.h"

#include "core/file.h"

ath_status ath_model_tokenizer(const ath_model *m, const char *dir, ath_tokenizer *t, ath_arena *a)
{
    if (!m || !m->family || !dir || !t || !a)
        return ATH_ERR_INVALID_ARG;
    if (!m->family->pretokenize || !m->family->pretokenize_regex) {
        ath_log(ATH_LOG_ERROR, "%s: the family has no tokenizer support", m->family->name);
        return ATH_ERR_UNSUPPORTED;
    }
    char path[ATH_PATH_MAX];
    ATH_CHECK(ath_path_join(path, sizeof path, dir, "tokenizer.json"));
    if (!ath_file_exists(path)) {
        ath_log(ATH_LOG_ERROR, "%s: not found", path);
        return ATH_ERR_NOT_FOUND;
    }
    return ath_tokenizer_load(t, a, path, m->family->pretokenize, m->family->pretokenize_regex);
}

ath_status ath_model_chat(const ath_model *m, const ath_msg *msgs, int32_t n_msgs, const ath_tool_def *tools,
                          int32_t n_tools, uint32_t opts, ath_buf *out)
{
    if (!m || !m->family || !out || n_msgs < 0 || n_tools < 0 || (n_msgs && !msgs) || (n_tools && !tools) ||
        (opts & ~(ATH_CHAT_GENERATE | ATH_CHAT_THINKING)))
        return ATH_ERR_INVALID_ARG;
    if (!m->family->chat)
        return ATH_ERR_UNSUPPORTED;
    for (int32_t i = 0; i < n_msgs; i++) {
        const ath_msg *g = &msgs[i];
        if ((unsigned)g->role > ATH_MSG_TOOL || g->n_tool_calls < 0 || (g->n_tool_calls && !g->tool_calls))
            return ATH_ERR_INVALID_ARG;
        for (int32_t k = 0; k < g->n_tool_calls; k++)
            if (!g->tool_calls[k].name)
                return ATH_ERR_INVALID_ARG;
    }
    for (int32_t i = 0; i < n_tools; i++)
        if (!tools[i].name)
            return ATH_ERR_INVALID_ARG;
    return m->family->chat(msgs, n_msgs, tools, n_tools, opts, out);
}

size_t ath_model_chat_shared(const ath_model *m, const ath_tokenizer *tok, const ath_msg *msgs, int32_t n_msgs,
                             const ath_tool_def *tools, int32_t n_tools, const int32_t *ids, size_t n,
                             ath_arena *scratch)
{
    int32_t n_sys = 0;
    while (n_sys < n_msgs && msgs[n_sys].role == ATH_MSG_SYSTEM)
        n_sys++;
    if ((!n_sys && !n_tools) || n_sys == n_msgs || !scratch)
        return 0;
    const ath_arena_pos mark = ath_arena_mark(scratch);
    ath_arena tk;   /* the tokenizer's working memory, apart from the tokens it writes */
    ath_arena_init(&tk, 0);
    ath_msg *two = ath_arena_alloc(scratch, (size_t)(n_sys + 1) * sizeof *two, _Alignof(ath_msg));
    ath_buf b[2];
    ath_tokens t[2];
    bool ok = two != NULL;
    for (int k = 0; ok && k < 2; k++) {
        memcpy(two, msgs, (size_t)n_sys * sizeof *two);
        two[n_sys] = (ath_msg){.role = ATH_MSG_USER, .content = k ? "b" : "a"};
        ath_buf_init(&b[k], scratch);
        ath_tokens_init(&t[k], scratch);
        ok = ath_model_chat(m, two, n_sys + 1, tools, n_tools, 0, &b[k]) == ATH_OK &&
             ath_tokenizer_encode(tok, b[k].data, b[k].len, &t[k], &tk) == ATH_OK;
    }
    size_t c = 0;
    while (ok && c < t[0].n && c < t[1].n && t[0].ids[c] == t[1].ids[c])
        c++;
    if (!ok || c > n || memcmp(t[0].ids, ids, c * sizeof *ids))
        c = 0;
    ath_arena_free(&tk);
    ath_arena_reset(scratch, mark);
    return c;
}

ath_status ath_model_parse(const ath_model *m, ath_parse_state *st, const char *piece, size_t n,
                           ath_out_event *events, int32_t cap, int32_t *n_events)
{
    if (!m || !m->family || !st || !st->a || (n && !piece) || !events || cap < 1 || !n_events ||
        st->n_tools < 0 || (st->n_tools && !st->tools))
        return ATH_ERR_INVALID_ARG;
    if (!m->family->parse)
        return ATH_ERR_UNSUPPORTED;
    return m->family->parse(st, piece, n, events, cap, n_events);
}
