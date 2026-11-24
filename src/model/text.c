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
