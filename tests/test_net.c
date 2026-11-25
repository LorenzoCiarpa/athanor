/* test_net.c: the server of step 13 (src/net/).
 *
 *   - the HTTP parser on whole, split, chunked, pipelined and malformed
 *     requests;
 *   - each API's request parsed into messages, tools and settings, on
 *     fixtures shaped as Claude Code, Codex CLI and OpenCode send them;
 *   - the HTTP loop over a real socket with a handler of the test's own:
 *     keep-alive, 100-continue, server-sent events in chunks;
 *   - on the real checkpoint (ATH_MODEL_DIR; skipped without it), the
 *     whole server over loopback: whole and streamed answers in the three
 *     APIs, a tool call, stop strings, the snapshot of the conversation
 *     reused across turns and kept across a side request, errors.
 */
#include "test.h"
#include "util.h"

#include "core/buf.h"
#include "core/json.h"
#include "net/api.h"
#include "net/gen.h"
#include "net/http.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <pthread.h>
#include <sys/socket.h>
#include <unistd.h>

/* ---- the HTTP parser ---------------------------------------------------------------- */

ATH_TEST(http_parse_whole_and_pipelined)
{
    ath_arena a;
    ath_arena_init(&a, 0);
    static const char r[] = "POST /v1/messages?beta=true HTTP/1.1\r\nHost: x\r\ncontent-length: 7\r\n\r\n{\"a\":1}"
                            "GET /health HTTP/1.0\r\n\r\n";
    const size_t n = sizeof r - 1;
    ath_http_req q;
    size_t used;
    ATH_REQUIRE_OK(ath_http_parse(r, n, &a, &q, &used));
    ATH_EXPECT_STR_EQ(q.method, "POST");
    ATH_EXPECT(!strcmp(q.path, "/v1/messages"));
    ATH_EXPECT_EQ_INT(q.body_len, 7);
    ATH_EXPECT_STR_EQ(q.body, "{\"a\":1}");
    ATH_EXPECT(q.keep_alive);
    ATH_EXPECT(!q.expect_continue);
    ATH_EXPECT_EQ_INT(used, strstr(r, "GET") - r);
    size_t used2;
    ATH_REQUIRE_OK(ath_http_parse(r + used, n - used, &a, &q, &used2));
    ATH_EXPECT_STR_EQ(q.method, "GET");
    ATH_EXPECT(!strcmp(q.path, "/health"));
    ATH_EXPECT_EQ_INT(q.body_len, 0);
    ATH_EXPECT(!q.keep_alive);   /* HTTP/1.0 */
    ATH_EXPECT_EQ_INT(used + used2, n);
    ath_arena_free(&a);
}

ATH_TEST(http_parse_split)
{
    ath_arena a;
    ath_arena_init(&a, 0);
    static const char r[] = "POST /x HTTP/1.1\r\nContent-Length: 5\r\nExpect: 100-continue\r\n\r\nhello";
    const size_t n = sizeof r - 1, he = (size_t)(strstr(r, "hello") - r);
    for (size_t k = 0; k < n; k++) {
        ath_http_req q;
        size_t used = 99;
        ATH_EXPECT_STATUS(ath_http_parse(r, k, &a, &q, &used), ATH_ERR_RANGE);
        ATH_EXPECT_EQ_INT(used, k < he ? 0 : he);
        if (k >= he)
            ATH_EXPECT(q.expect_continue);
    }
    ath_http_req q;
    size_t used;
    ATH_EXPECT_OK(ath_http_parse(r, n, &a, &q, &used));
    ATH_EXPECT_STR_EQ(q.body, "hello");
    ath_arena_free(&a);
}

ATH_TEST(http_parse_chunked)
{
    ath_arena a;
    ath_arena_init(&a, 0);
    static const char r[] = "POST /c HTTP/1.1\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
                            "4\r\nWiki\r\n5;ext=1\r\npedia\r\n0\r\nTrailer: x\r\n\r\nNEXT";
    ath_http_req q;
    size_t used;
    ATH_REQUIRE_OK(ath_http_parse(r, sizeof r - 1, &a, &q, &used));
    ATH_EXPECT_STR_EQ(q.body, "Wikipedia");
    ATH_EXPECT_EQ_INT(q.body_len, 9);
    ATH_EXPECT(!q.keep_alive);
    ATH_EXPECT_EQ_INT(used, sizeof r - 1 - 4);
    /* cut anywhere in the body: wait for more */
    const size_t he = (size_t)(strstr(r, "4\r\n") - r);
    for (size_t k = he; k < sizeof r - 1 - 4; k++)
        ATH_EXPECT_STATUS(ath_http_parse(r, k, &a, &q, &used), ATH_ERR_RANGE);
    ath_arena_free(&a);
}

ATH_TEST(http_parse_malformed)
{
    ath_arena a;
    ath_arena_init(&a, 0);
    static const char *const bad[] = {
        "GARBAGE\r\n\r\n",
        "GET /\r\n\r\n",
        "GET / HTTP/2.0\r\n\r\n",
        "GET  HTTP/1.1\r\n\r\n",
        "POST / HTTP/1.1\r\nContent-Length: x\r\n\r\n",
        "POST / HTTP/1.1\r\nContent-Length: -1\r\n\r\n",
        "POST / HTTP/1.1\r\nContent-Length: 5x\r\n\r\nhello",
        "POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\nzz\r\nab\r\n0\r\n\r\n",
        "POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n2\r\nabc\r\n0\r\n\r\n",
    };
    for (size_t i = 0; i < ATH_ARRAY_LEN(bad); i++) {
        ath_http_req q;
        size_t used;
        ath_status st = ath_http_parse(bad[i], strlen(bad[i]), &a, &q, &used);
        if (st != ATH_ERR_PARSE)
            printf("    malformed request %zu: %s\n", i, ath_status_str(st));
        ATH_EXPECT(st == ATH_ERR_PARSE);
    }
    static const char big[] = "POST / HTTP/1.1\r\nContent-Length: 99999999999\r\n\r\n";
    ath_http_req q;
    size_t used;
    ATH_EXPECT_STATUS(ath_http_parse(big, sizeof big - 1, &a, &q, &used), ATH_ERR_UNSUPPORTED);
    ath_arena_free(&a);
}

/* ---- the presence penalty's reference ---------------------------------------------- */

ATH_TEST(presence_reference)
{
    float l[5] = {1, 2, 3, 4, 5};
    const uint8_t seen[5] = {0, 1, 0, 9, 0};
    ath_tensor t;
    ATH_REQUIRE_OK(ath_tensor_init(&t, ATH_DTYPE_F32, 2, (int64_t[]){1, 5}, l, ATH_TIER_RAM));
    ATH_EXPECT_OK(ath_presence_ref(&t, seen, 1.5f));
    ATH_EXPECT(l[0] == 1 && l[1] == 0.5f && l[2] == 3 && l[3] == 2.5f && l[4] == 5);
    ATH_EXPECT_OK(ath_presence_ref(&t, seen, -1.0f));   /* a negative penalty favors them */
    ATH_EXPECT(l[1] == 1.5f && l[3] == 3.5f);
    ATH_EXPECT_STATUS(ath_presence_ref(&t, NULL, 1.0f), ATH_ERR_INVALID_ARG);
    uint16_t b[5] = {0};
    ath_tensor tb;
    ATH_REQUIRE_OK(ath_tensor_init(&tb, ATH_DTYPE_BF16, 1, (int64_t[]){5}, b, ATH_TIER_RAM));
    ATH_EXPECT_STATUS(ath_presence_ref(&tb, seen, 1.0f), ATH_ERR_TYPE);
}

/* ---- the APIs' requests ------------------------------------------------------------- */

static const ath_api_defaults DEFAULTS = {
    .model = "test-model",
    .think = true,
    .sampler_think = {0.6f, 20, 0.95f, 0, 0},
    .sampler_plain = {0.7f, 20, 0.8f, 0, 1.5f},
    .client_sampling = true,
};

static bool parse(ath_test *t_, ath_api_kind kind, const char *json, ath_api_request *r, ath_arena *a,
                  const char **err)
{
    ath_api api;
    ath_api_init(&api, NULL, &DEFAULTS);
    const ath_json *body;
    if (ath_json_parse(a, json, strlen(json), &body, NULL) != ATH_OK) {
        ATH_EXPECT(!"fixture is valid JSON");
        return false;
    }
    return ath_api_parse(&api, kind, body, r, a, err) == ATH_OK;
}

ATH_TEST(api_anthropic_request)
{
    ath_arena a;
    ath_arena_init(&a, 0);
    /* as Claude Code 2.1.332 sends it: a billing line in the system blocks,
     * the environment as a system message after the first user message */
    static const char req[] =
        "{\"model\": \"claude-x\", \"max_tokens\": 32000, \"stream\": true, \"temperature\": 1,"
        " \"thinking\": {\"type\": \"adaptive\"}, \"stop_sequences\": [\"END\"],"
        " \"system\": [{\"type\": \"text\", \"text\": \"x-anthropic-billing-header: cc_version=1;\"},"
        "   {\"type\": \"text\", \"text\": \"You are an agent.\"}, {\"type\": \"text\", \"text\": \"Be brief.\"}],"
        " \"tools\": [{\"name\": \"Read\", \"description\": \"Read a file\", \"input_schema\": {\"type\": \"object\","
        "   \"properties\": {\"file_path\": {\"type\": \"string\"}}}},"
        "   {\"type\": \"web_search_20250305\", \"name\": \"web_search\"}],"
        " \"messages\": ["
        "  {\"role\": \"user\", \"content\": [{\"type\": \"text\", \"text\": \"<r>a</r>\"}, {\"type\": \"text\", \"text\": \"Read x.\"}]},"
        "  {\"role\": \"system\", \"content\": [{\"type\": \"text\", \"text\": \"# Environment\"}]},"
        "  {\"role\": \"assistant\", \"content\": [{\"type\": \"thinking\", \"thinking\": \"I read it.\", \"signature\": \"s\"},"
        "    {\"type\": \"tool_use\", \"id\": \"toolu_1\", \"name\": \"Read\", \"input\": {\"file_path\": \"x\"}}]},"
        "  {\"role\": \"user\", \"content\": [{\"type\": \"tool_result\", \"tool_use_id\": \"toolu_1\","
        "    \"content\": [{\"type\": \"text\", \"text\": \"hello\"}]}, {\"type\": \"text\", \"text\": \"<r>b</r>\"},"
        "    {\"type\": \"image\", \"source\": {}}]}]}";
    ath_api_request r;
    const char *err = NULL;
    ATH_REQUIRE(parse(t_, ATH_API_ANTHROPIC, req, &r, &a, &err));
    const ath_gen_request *g = &r.g;
    ATH_EXPECT(r.stream);
    ATH_EXPECT(g->think);
    ATH_EXPECT_STR_EQ(r.model, "claude-x");
    ATH_EXPECT_EQ_INT(g->max_tokens, 32000);
    ATH_EXPECT(g->sampler.temperature == 1.0f && g->sampler.top_p == 0.95f && g->sampler.top_k == 20);
    ATH_REQUIRE(g->n_stop == 1);
    ATH_EXPECT_STR_EQ(g->stop[0], "END");
    ATH_REQUIRE(g->n_tools == 1);   /* the server tool is not a function */
    ATH_EXPECT_STR_EQ(g->tools[0].name, "Read");
    ATH_EXPECT(strstr(g->tools[0].json_schema, "file_path") != NULL);
    ATH_REQUIRE(g->n_msgs == 6);
    ATH_EXPECT(g->msgs[0].role == ATH_MSG_SYSTEM);
    ATH_EXPECT_STR_EQ(g->msgs[0].content, "You are an agent.\n\nBe brief.");
    ATH_EXPECT(g->msgs[1].role == ATH_MSG_USER);
    ATH_EXPECT_STR_EQ(g->msgs[1].content, "<r>a</r>Read x.");
    ATH_EXPECT(g->msgs[2].role == ATH_MSG_USER);
    ATH_EXPECT_STR_EQ(g->msgs[2].content, "# Environment");
    ATH_EXPECT(g->msgs[3].role == ATH_MSG_ASSISTANT);
    ATH_EXPECT_STR_EQ(g->msgs[3].reasoning, "I read it.");
    ATH_REQUIRE(g->msgs[3].n_tool_calls == 1);
    ATH_EXPECT_STR_EQ(g->msgs[3].tool_calls[0].id, "toolu_1");
    ATH_EXPECT_STR_EQ(g->msgs[3].tool_calls[0].name, "Read");
    ATH_EXPECT_STR_EQ(g->msgs[3].tool_calls[0].arguments, "{\"file_path\": \"x\"}");
    ATH_EXPECT(g->msgs[4].role == ATH_MSG_TOOL);
    ATH_EXPECT_STR_EQ(g->msgs[4].content, "hello");
    ATH_EXPECT_STR_EQ(g->msgs[4].tool_call_id, "toolu_1");
    ATH_EXPECT(g->msgs[5].role == ATH_MSG_USER);
    ATH_EXPECT(strncmp(g->msgs[5].content, "<r>b</r>[an image", 17) == 0);

    /* thinking off, tools off, a bad role */
    ATH_REQUIRE(parse(t_, ATH_API_ANTHROPIC,
                      "{\"thinking\": {\"type\": \"disabled\"}, \"tool_choice\": {\"type\": \"none\"},"
                      " \"tools\": [{\"name\": \"A\", \"input_schema\": {}}],"
                      " \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}]}",
                      &r, &a, &err));
    ATH_EXPECT(!r.g.think);
    ATH_EXPECT_EQ_INT(r.g.n_tools, 0);
    ATH_EXPECT(r.g.sampler.presence == 1.5f);   /* the plain sampler's default */
    ATH_EXPECT_STR_EQ(r.model, "test-model");
    ATH_EXPECT(!parse(t_, ATH_API_ANTHROPIC, "{\"messages\": [{\"role\": \"robot\", \"content\": \"x\"}]}", &r, &a, &err));
    ATH_EXPECT(err && strstr(err, "role"));
    ATH_EXPECT(!parse(t_, ATH_API_ANTHROPIC, "{\"messages\": 3}", &r, &a, &err));
    ATH_EXPECT(!parse(t_, ATH_API_ANTHROPIC, "{\"top_p\": 0, \"messages\": [{\"role\": \"user\", \"content\": \"x\"}]}",
                      &r, &a, &err));
    ath_arena_free(&a);
}

ATH_TEST(api_chat_request)
{
    ath_arena a;
    ath_arena_init(&a, 0);
    static const char req[] =
        "{\"model\": \"m\", \"stream\": true, \"stream_options\": {\"include_usage\": true}, \"max_tokens\": 99,"
        " \"stop\": \"\\n\\n\", \"presence_penalty\": 0.5, \"top_k\": 5, \"reasoning_effort\": \"none\","
        " \"tools\": [{\"type\": \"function\", \"function\": {\"name\": \"get\", \"parameters\": {\"type\": \"object\"}}}],"
        " \"messages\": [{\"role\": \"system\", \"content\": \"S1\"}, {\"role\": \"developer\", \"content\": \"S2\"},"
        "  {\"role\": \"user\", \"content\": [{\"type\": \"text\", \"text\": \"Q\"}]},"
        "  {\"role\": \"assistant\", \"content\": null, \"reasoning_content\": \"R\", \"tool_calls\": [{\"id\": \"c1\","
        "    \"type\": \"function\", \"function\": {\"name\": \"get\", \"arguments\": \"{\\\"k\\\": 1}\"}},"
        "    {\"id\": \"c2\", \"type\": \"function\", \"function\": {\"name\": \"get\", \"arguments\": \"not json\"}}]},"
        "  {\"role\": \"tool\", \"tool_call_id\": \"c1\", \"content\": \"one\"},"
        "  {\"role\": \"tool\", \"tool_call_id\": \"c2\", \"content\": \"two\"},"
        "  {\"role\": \"system\", \"content\": \"late\"}]}";
    ath_api_request r;
    const char *err = NULL;
    ATH_REQUIRE(parse(t_, ATH_API_CHAT, req, &r, &a, &err));
    const ath_gen_request *g = &r.g;
    ATH_EXPECT(r.stream && r.include_usage);
    ATH_EXPECT(!g->think);
    ATH_EXPECT_EQ_INT(g->max_tokens, 99);
    ATH_EXPECT(g->sampler.presence == 0.5f && g->sampler.top_k == 5 && g->sampler.temperature == 0.7f);
    ATH_REQUIRE(g->n_stop == 1);
    ATH_EXPECT_STR_EQ(g->stop[0], "\n\n");
    ATH_REQUIRE(g->n_tools == 1);
    ATH_REQUIRE(g->n_msgs == 6);
    ATH_EXPECT(g->msgs[0].role == ATH_MSG_SYSTEM);
    ATH_EXPECT_STR_EQ(g->msgs[0].content, "S1\n\nS2");
    ATH_EXPECT_STR_EQ(g->msgs[1].content, "Q");
    ATH_EXPECT(g->msgs[2].role == ATH_MSG_ASSISTANT);
    ATH_EXPECT_STR_EQ(g->msgs[2].content, "");
    ATH_EXPECT_STR_EQ(g->msgs[2].reasoning, "R");
    ATH_REQUIRE(g->msgs[2].n_tool_calls == 2);
    ATH_EXPECT_STR_EQ(g->msgs[2].tool_calls[0].arguments, "{\"k\": 1}");
    ATH_EXPECT_STR_EQ(g->msgs[2].tool_calls[1].arguments, "{}");   /* not a JSON object */
    ATH_EXPECT(g->msgs[3].role == ATH_MSG_TOOL && g->msgs[4].role == ATH_MSG_TOOL);
    ATH_EXPECT_STR_EQ(g->msgs[4].tool_call_id, "c2");
    ATH_EXPECT(g->msgs[5].role == ATH_MSG_USER);   /* a late system message */
    ATH_EXPECT_STR_EQ(g->msgs[5].content, "late");

    ATH_REQUIRE(parse(t_, ATH_API_CHAT,
                      "{\"chat_template_kwargs\": {\"enable_thinking\": true}, \"reasoning_effort\": \"none\","
                      " \"max_completion_tokens\": 7, \"max_tokens\": 9, \"stop\": [\"a\", \"\", \"b\"],"
                      " \"messages\": [{\"role\": \"user\", \"content\": \"x\"}]}",
                      &r, &a, &err));
    ATH_EXPECT(r.g.think);
    ATH_EXPECT_EQ_INT(r.g.max_tokens, 7);
    ATH_EXPECT_EQ_INT(r.g.n_stop, 2);
    ATH_EXPECT(!parse(t_, ATH_API_CHAT, "{\"stop\": [1], \"messages\": [{\"role\": \"user\", \"content\": \"x\"}]}",
                      &r, &a, &err));
    ATH_EXPECT(!parse(t_, ATH_API_CHAT, "{\"messages\": [{\"role\": \"system\", \"content\": \"x\"}]}", &r, &a, &err));
    ath_arena_free(&a);
}

ATH_TEST(api_responses_request)
{
    ath_arena a;
    ath_arena_init(&a, 0);
    /* as Codex CLI 0.167.0 sends a second turn */
    static const char req[] =
        "{\"model\": \"m\", \"instructions\": \"You are Codex.\", \"stream\": true, \"store\": false,"
        " \"reasoning\": {\"effort\": \"medium\", \"summary\": \"auto\"}, \"max_output_tokens\": 500,"
        " \"tools\": [{\"type\": \"function\", \"name\": \"exec_command\", \"parameters\": {\"type\": \"object\"}},"
        "   {\"type\": \"custom\", \"name\": \"apply_patch\", \"description\": \"Patch\","
        "    \"format\": {\"type\": \"grammar\", \"definition\": \"start: x\"}},"
        "   {\"type\": \"web_search\"}, {\"type\": \"namespace\", \"name\": \"ns\", \"tools\": []}],"
        " \"input\": [{\"type\": \"message\", \"role\": \"developer\", \"content\": [{\"type\": \"input_text\", \"text\": \"D\"}]},"
        "  {\"type\": \"message\", \"role\": \"user\", \"content\": [{\"type\": \"input_text\", \"text\": \"U\"}]},"
        "  {\"type\": \"reasoning\", \"id\": \"rs_1\", \"summary\": [{\"type\": \"summary_text\", \"text\": \"sum\"}],"
        "   \"content\": [{\"type\": \"reasoning_text\", \"text\": \"raw\"}]},"
        "  {\"type\": \"message\", \"role\": \"assistant\", \"content\": [{\"type\": \"output_text\", \"text\": \"A\"}]},"
        "  {\"type\": \"function_call\", \"call_id\": \"call_1\", \"name\": \"exec_command\", \"arguments\": \"{\\\"cmd\\\": \\\"ls\\\"}\"},"
        "  {\"type\": \"custom_tool_call\", \"call_id\": \"call_2\", \"name\": \"apply_patch\", \"input\": \"*** Begin\"},"
        "  {\"type\": \"function_call_output\", \"call_id\": \"call_1\", \"output\": \"a.txt\"},"
        "  {\"type\": \"custom_tool_call_output\", \"call_id\": \"call_2\", \"output\": \"done\"},"
        "  {\"type\": \"reasoning\", \"summary\": [{\"type\": \"summary_text\", \"text\": \"s1\"},"
        "   {\"type\": \"summary_text\", \"text\": \"s2\"}]},"
        "  {\"type\": \"function_call\", \"call_id\": \"call_3\", \"name\": \"exec_command\", \"arguments\": \"{}\"},"
        "  {\"type\": \"function_call_output\", \"call_id\": \"call_3\", \"output\": [{\"type\": \"input_text\", \"text\": \"ok\"}]}]}";
    ath_api_request r;
    const char *err = NULL;
    ATH_REQUIRE(parse(t_, ATH_API_RESPONSES, req, &r, &a, &err));
    const ath_gen_request *g = &r.g;
    ATH_EXPECT(r.stream && g->think);
    ATH_EXPECT_EQ_INT(g->max_tokens, 500);
    ATH_REQUIRE(g->n_tools == 2);
    ATH_EXPECT_STR_EQ(g->tools[1].name, "apply_patch");
    ATH_EXPECT(strstr(g->tools[1].json_schema, "\"input\"") && strstr(g->tools[1].json_schema, "start: x"));
    ATH_REQUIRE(r.n_custom == 1);
    ATH_EXPECT_STR_EQ(r.custom[0], "apply_patch");
    ATH_REQUIRE(g->n_msgs == 7);
    ATH_EXPECT(g->msgs[0].role == ATH_MSG_SYSTEM);
    ATH_EXPECT_STR_EQ(g->msgs[0].content, "You are Codex.\n\nD");
    ATH_EXPECT_STR_EQ(g->msgs[1].content, "U");
    const ath_msg *m = &g->msgs[2];
    ATH_EXPECT(m->role == ATH_MSG_ASSISTANT);
    ATH_EXPECT_STR_EQ(m->reasoning, "raw");
    ATH_EXPECT_STR_EQ(m->content, "A");
    ATH_REQUIRE(m->n_tool_calls == 2);
    ATH_EXPECT_STR_EQ(m->tool_calls[0].id, "call_1");
    ATH_EXPECT_STR_EQ(m->tool_calls[0].arguments, "{\"cmd\": \"ls\"}");
    ATH_EXPECT_STR_EQ(m->tool_calls[1].arguments, "{\"input\": \"*** Begin\"}");
    ATH_EXPECT(g->msgs[3].role == ATH_MSG_TOOL && g->msgs[4].role == ATH_MSG_TOOL);
    ATH_EXPECT_STR_EQ(g->msgs[3].content, "a.txt");
    m = &g->msgs[5];
    ATH_EXPECT(m->role == ATH_MSG_ASSISTANT);
    ATH_EXPECT_STR_EQ(m->reasoning, "s1\n\ns2");
    ATH_EXPECT_EQ_INT(m->n_tool_calls, 1);
    ATH_EXPECT(g->msgs[6].role == ATH_MSG_TOOL);
    ATH_EXPECT_STR_EQ(g->msgs[6].content, "ok");
    ATH_EXPECT_STR_EQ(g->msgs[6].tool_call_id, "call_3");

    ATH_REQUIRE(parse(t_, ATH_API_RESPONSES, "{\"input\": \"hi\", \"reasoning\": {\"effort\": \"none\"}}", &r, &a, &err));
    ATH_EXPECT(!r.g.think);
    ATH_EXPECT_EQ_INT(r.g.n_msgs, 1);
    ATH_EXPECT(!parse(t_, ATH_API_RESPONSES, "{\"input\": \"hi\", \"previous_response_id\": \"resp_1\"}", &r, &a, &err));
    ATH_EXPECT(err && strstr(err, "previous_response_id"));
    ath_arena_free(&a);
}

/* ---- the HTTP loop over a socket ---------------------------------------------------- */

typedef struct {
    ath_http_server srv;
    ath_http_handler h;
    void            *ctx;
    volatile int     stop;
    pthread_t        thread;
} server;

static void *serve_thread(void *p)
{
    server *s = p;
    ath_http_run(&s->srv, s->h, s->ctx, &s->stop);
    return NULL;
}

static bool server_start(server *s, ath_http_handler h, void *ctx)
{
    memset(s, 0, sizeof *s);
    s->h = h, s->ctx = ctx;
    if (ath_http_listen(&s->srv, "127.0.0.1", 0) != ATH_OK)
        return false;
    return pthread_create(&s->thread, NULL, serve_thread, s) == 0;
}

static void server_stop(server *s)
{
    s->stop = 1;
    pthread_join(s->thread, NULL);
    ath_http_close(&s->srv);
}

static int dial(int port)
{
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in a = {.sin_family = AF_INET, .sin_port = htons((uint16_t)port)};
    inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
    if (fd >= 0 && connect(fd, (struct sockaddr *)&a, sizeof a) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

static bool send_str(int fd, const char *s)
{
    size_t n = strlen(s);
    while (n) {
        ssize_t w = send(fd, s, n, MSG_NOSIGNAL);
        if (w <= 0)
            return false;
        s += w, n -= (size_t)w;
    }
    return true;
}

/* Read until `until` appears in what was read (or the peer closes, or 60 s
 * pass without data); the bytes in b. */
static bool read_until(int fd, ath_buf *b, const char *until)
{
    for (;;) {
        if (b->data && strstr(b->data, until))
            return true;
        struct pollfd p = {.fd = fd, .events = POLLIN};
        if (poll(&p, 1, 60000) <= 0)
            return false;
        char tmp[4096];
        ssize_t r = recv(fd, tmp, sizeof tmp, 0);
        if (r <= 0)
            return b->data && strstr(b->data, until);
        ath_buf_append(b, tmp, (size_t)r);
    }
}

/* The test's handler: /echo answers with the body, /sse streams three events. */
static void echo_handler(void *ctx, const ath_http_req *req, ath_http_out *out)
{
    (void)ctx;
    if (!strcmp(req->path, "/echo")) {
        ath_http_reply(out, 200, "text/plain", req->body, req->body_len);
    } else if (!strcmp(req->path, "/sse")) {
        ath_http_sse_start(out);
        ath_http_sse(out, "first", "{\"n\": 1}", 8);
        ath_http_sse_ping(out);
        ath_http_sse(out, NULL, "{\"n\": 2}", 8);
        ath_http_sse(out, "last", "[DONE]", 6);
    }   /* anything else: no response, the loop answers 500 */
}

ATH_TEST(http_loopback)
{
    server s;
    ATH_REQUIRE(server_start(&s, echo_handler, NULL));
    int fd = dial(s.srv.port);
    ATH_REQUIRE(fd >= 0);
    ath_arena a;
    ath_arena_init(&a, 0);
    ath_buf b;
    ath_buf_init(&b, &a);
    /* two requests on one connection, the second sent in two pieces */
    ATH_EXPECT(send_str(fd, "POST /echo HTTP/1.1\r\nContent-Length: 5\r\n\r\nhello"));
    ATH_EXPECT(read_until(fd, &b, "hello"));
    ATH_EXPECT(strncmp(b.data, "HTTP/1.1 200 OK\r\n", 17) == 0);
    ATH_EXPECT(strstr(b.data, "Connection: keep-alive") != NULL);
    ath_buf_clear(&b);
    ATH_EXPECT(send_str(fd, "POST /echo HTTP/1.1\r\nContent-Length: 6\r\nExpect: 100-continue\r\n\r\n"));
    ATH_EXPECT(read_until(fd, &b, "100 Continue\r\n\r\n"));
    ATH_EXPECT(send_str(fd, "world!"));
    ATH_EXPECT(read_until(fd, &b, "world!"));
    ath_buf_clear(&b);
    /* server-sent events in chunks, then the connection still serves */
    ATH_EXPECT(send_str(fd, "GET /sse HTTP/1.1\r\n\r\n"));
    ATH_EXPECT(read_until(fd, &b, "0\r\n\r\n"));
    ATH_EXPECT(strstr(b.data, "Transfer-Encoding: chunked") != NULL);
    ATH_EXPECT(strstr(b.data, "event: first\ndata: {\"n\": 1}\n\n") != NULL);
    ATH_EXPECT(strstr(b.data, "\r\n: ping\n\n\r\n") != NULL);
    ATH_EXPECT(strstr(b.data, "data: {\"n\": 2}\n\n") != NULL);
    ATH_EXPECT(strstr(b.data, "event: last\ndata: [DONE]\n\n") != NULL);
    /* the chunk sizes add up: each line "<hex>\r\n" is followed by that many bytes and "\r\n" */
    const char *p = strstr(b.data, "\r\n\r\n") + 4;
    bool sizes_ok = true;
    for (;;) {
        char *e;
        unsigned long n = strtoul(p, &e, 16);
        if (e == p || strncmp(e, "\r\n", 2)) {
            sizes_ok = false;
            break;
        }
        p = e + 2 + n;
        if (strncmp(p, "\r\n", 2)) {
            sizes_ok = false;
            break;
        }
        p += 2;
        if (n == 0)
            break;
    }
    ATH_EXPECT(sizes_ok);
    ath_buf_clear(&b);
    ATH_EXPECT(send_str(fd, "GET /nothing HTTP/1.1\r\nConnection: close\r\n\r\n"));
    ATH_EXPECT(read_until(fd, &b, "no response"));
    ATH_EXPECT(strncmp(b.data, "HTTP/1.1 500", 12) == 0);
    close(fd);
    /* a malformed request: 400, and the connection closes */
    fd = dial(s.srv.port);
    ATH_REQUIRE(fd >= 0);
    ath_buf_clear(&b);
    ATH_EXPECT(send_str(fd, "NONSENSE\r\n\r\n"));
    ATH_EXPECT(read_until(fd, &b, "malformed"));
    ATH_EXPECT(strncmp(b.data, "HTTP/1.1 400", 12) == 0);
    close(fd);
    server_stop(&s);
    ath_arena_free(&a);
}

/* ---- the whole server on the real checkpoint ------------------------------------------ */

typedef struct {
    int    status;
    char  *body;      /* the body (chunks joined for a stream)        */
} response;

/* One request on a new connection; the response's status and body. */
static bool request(int port, const char *path, const char *json, response *r, ath_arena *a)
{
    int fd = dial(port);
    if (fd < 0)
        return false;
    ath_buf q, b;
    ath_buf_init(&q, a);
    ath_buf_init(&b, a);
    ath_buf_printf(&q, "POST %s HTTP/1.1\r\nContent-Type: application/json\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n%s",
                   path, strlen(json), json);
    bool ok = send_str(fd, q.data) && read_until(fd, &b, "\x01never");
    close(fd);
    (void)ok;
    if (!b.data || strncmp(b.data, "HTTP/1.1 ", 9))
        return false;
    r->status = atoi(b.data + 9);
    char *body = strstr(b.data, "\r\n\r\n");
    if (!body)
        return false;
    body += 4;
    if (!strstr(b.data, "Transfer-Encoding: chunked")) {
        r->body = body;
        return true;
    }
    ath_buf j;
    ath_buf_init(&j, a);
    for (char *p = body;;) {
        char *e;
        unsigned long n = strtoul(p, &e, 16);
        if (e == p || n == 0)
            break;
        ath_buf_append(&j, e + 2, n);
        p = e + 2 + n + 2;
    }
    r->body = (char *)ath_buf_str(&j);
    return true;
}

/* The JSON of every "data: " line of an event stream, in order. */
static int events(const char *body, const ath_json **out, int cap, ath_arena *a)
{
    int n = 0;
    for (const char *p = strstr(body, "data: "); p && n < cap; p = strstr(p, "data: ")) {
        p += 6;
        const char *e = strchr(p, '\n');
        if (!e)
            break;
        const ath_json *j;
        if (ath_json_parse(a, p, (size_t)(e - p), &j, NULL) == ATH_OK)
            out[n++] = j;
        p = e;
    }
    return n;
}

static const char *jtype(const ath_json *j)
{
    const char *s = NULL;
    ath_json_to_str(ath_json_get(j, "type"), &s, NULL);
    return s ? s : "";
}

static int64_t jint(const ath_json *j, const char *path)
{
    int64_t v = -1;
    ath_json_to_int(ath_json_path(j, path), &v);
    return v;
}

ATH_TEST(serve_real_model)
{
    char cfg[1024];
    if (!model_file(cfg, sizeof cfg, "config.json"))
        ATH_SKIP("no checkpoint (ATH_MODEL_DIR)");
    ath_arena a;
    ath_arena_init(&a, 0);
    static ath_gen g;
    const ath_gen_opts go = {.ctx = 2048, .experts = -1, .seed = 1};
    ATH_REQUIRE_OK(ath_gen_open(&g, model_dir(), &go, &a));
    ath_api api;
    const ath_api_defaults d = {.model = "qwen-test",
                                .think = false,
                                .sampler_think = {0, 20, 0.95f, 0, 0},
                                .sampler_plain = {0, 20, 0.8f, 0, 0},
                                .client_sampling = true};
    ath_api_init(&api, &g, &d);
    server s;
    ATH_REQUIRE(server_start(&s, ath_api_handle, &api));
    printf("    on %s\n", g.engine ? "the GPU" : "the CPU");
    response r;
    const ath_json *j, *ev[2048];

    /* whole answer, chat completions, greedy */
    ATH_REQUIRE(request(s.srv.port, "/v1/chat/completions",
                        "{\"model\": \"x\", \"max_tokens\": 24, \"messages\": [{\"role\": \"user\", \"content\":"
                        " \"Count from 1 to 20, separated by commas.\"}]}",
                        &r, &a));
    ATH_EXPECT_EQ_INT(r.status, 200);
    ATH_REQUIRE_OK(ath_json_parse(&a, r.body, strlen(r.body), &j, NULL));
    const char *text = NULL;
    ath_json_to_str(ath_json_path(j, "choices.0.message.content"), &text, NULL);
    ATH_REQUIRE(text);
    printf("    answer: %.60s\n", text);
    ATH_EXPECT(strstr(text, "1, 2, 3") != NULL);
    ATH_EXPECT(jint(j, "usage.prompt_tokens") > 10);
    ATH_EXPECT(jint(j, "usage.completion_tokens") == 24);
    const char *fin = NULL;
    ath_json_to_str(ath_json_path(j, "choices.0.finish_reason"), &fin, NULL);
    ATH_EXPECT_STR_EQ(fin, "length");

    /* the same with a stop string: the answer ends before it */
    ATH_REQUIRE(request(s.srv.port, "/v1/chat/completions",
                        "{\"max_tokens\": 40, \"stop\": [\", 5\"], \"messages\": [{\"role\": \"user\", \"content\":"
                        " \"Count from 1 to 20, separated by commas.\"}]}",
                        &r, &a));
    ATH_REQUIRE_OK(ath_json_parse(&a, r.body, strlen(r.body), &j, NULL));
    text = NULL;
    ath_json_to_str(ath_json_path(j, "choices.0.message.content"), &text, NULL);
    ATH_REQUIRE(text);
    ATH_EXPECT(strstr(text, ", 4") && !strstr(text, ", 5"));
    fin = NULL;
    ath_json_to_str(ath_json_path(j, "choices.0.finish_reason"), &fin, NULL);
    ATH_EXPECT_STR_EQ(fin, "stop");
    /* the second request repeats the first's prompt: its stable part comes from the snapshot */
    ATH_EXPECT(jint(j, "usage.prompt_tokens_details.cached_tokens") > 0);

    /* Anthropic, streamed: the events in their order */
    ATH_REQUIRE(request(s.srv.port, "/v1/messages",
                        "{\"model\": \"claude-x\", \"max_tokens\": 16, \"stream\": true, \"messages\": [{\"role\": \"user\","
                        " \"content\": \"Say hello.\"}, {\"role\": \"assistant\", \"content\": \"Hello!\"},"
                        " {\"role\": \"user\", \"content\": \"Again, in Italian.\"}]}",
                        &r, &a));
    ATH_EXPECT_EQ_INT(r.status, 200);
    int n = events(r.body, ev, 2048, &a);
    ATH_REQUIRE(n >= 5);
    ATH_EXPECT_STR_EQ(jtype(ev[0]), "message_start");
    ATH_EXPECT_STR_EQ(jtype(ev[n - 1]), "message_stop");
    ATH_EXPECT_STR_EQ(jtype(ev[n - 2]), "message_delta");
    int open = 0, starts = 0, deltas = 0;
    bool order_ok = true;
    for (int i = 1; i < n - 2; i++) {
        const char *t = jtype(ev[i]);
        if (!strcmp(t, "content_block_start"))
            order_ok &= open == 0, open = 1, starts++;
        else if (!strcmp(t, "content_block_stop"))
            order_ok &= open == 1, open = 0;
        else if (!strcmp(t, "content_block_delta"))
            order_ok &= open == 1, deltas++;
        else if (strcmp(t, "ping"))
            order_ok = false;
    }
    ATH_EXPECT(order_ok && open == 0 && starts >= 1 && deltas >= 1);
    ATH_EXPECT(jint(ev[n - 2], "usage.output_tokens") > 0);

    /* a conversation that grows: each turn starts from the snapshot of the one before */
    static const char conv1[] =
        "{\"max_tokens\": 8, \"messages\": [{\"role\": \"system\", \"content\": \"You are terse. You answer in one or two words. You never explain. You never use punctuation other than a final period. You like colors, shapes and animals, and you answer questions about them with care.\"},"
        " {\"role\": \"user\", \"content\": \"Name a color.\"}]}";
    static const char conv2[] =
        "{\"max_tokens\": 8, \"messages\": [{\"role\": \"system\", \"content\": \"You are terse. You answer in one or two words. You never explain. You never use punctuation other than a final period. You like colors, shapes and animals, and you answer questions about them with care.\"},"
        " {\"role\": \"user\", \"content\": \"Name a color.\"}, {\"role\": \"assistant\", \"content\": \"Blue.\"},"
        " {\"role\": \"user\", \"content\": \"Another one.\"}]}";
    static const char side[] =
        "{\"max_tokens\": 4, \"messages\": [{\"role\": \"user\", \"content\": \"Title: colors\"}]}";
    static const char conv3[] =
        "{\"max_tokens\": 8, \"messages\": [{\"role\": \"system\", \"content\": \"You are terse. You answer in one or two words. You never explain. You never use punctuation other than a final period. You like colors, shapes and animals, and you answer questions about them with care.\"},"
        " {\"role\": \"user\", \"content\": \"Name a color.\"}, {\"role\": \"assistant\", \"content\": \"Blue.\"},"
        " {\"role\": \"user\", \"content\": \"Another one.\"}, {\"role\": \"assistant\", \"content\": \"Red.\"},"
        " {\"role\": \"user\", \"content\": \"And one more.\"}]}";
    int64_t cached[4], prompt[4];
    const char *conv[4] = {conv1, conv2, side, conv3};
    for (int i = 0; i < 4; i++) {
        ATH_REQUIRE(request(s.srv.port, "/v1/chat/completions", conv[i], &r, &a));
        ATH_REQUIRE_OK(ath_json_parse(&a, r.body, strlen(r.body), &j, NULL));
        cached[i] = jint(j, "usage.prompt_tokens_details.cached_tokens");
        prompt[i] = jint(j, "usage.prompt_tokens");
    }
    printf("    turns: prompt/cached %lld/%lld, %lld/%lld, side %lld/%lld, %lld/%lld\n", (long long)prompt[0],
           (long long)cached[0], (long long)prompt[1], (long long)cached[1], (long long)prompt[2], (long long)cached[2],
           (long long)prompt[3], (long long)cached[3]);
    ATH_EXPECT(cached[1] > prompt[0] / 2);          /* from the first turn's snapshot   */
    ATH_EXPECT(cached[3] > cached[1]);              /* the side request kept it         */

    /* a tool call, OpenAI Responses, streamed */
    ATH_REQUIRE(request(s.srv.port, "/v1/responses",
                        "{\"stream\": true, \"max_output_tokens\": 200, \"reasoning\": {\"effort\": \"none\"},"
                        " \"tools\": [{\"type\": \"function\", \"name\": \"get_weather\", \"description\": \"The weather in a city\","
                        "  \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}},"
                        "  \"required\": [\"city\"]}}],"
                        " \"input\": \"What is the weather in Rome? Use the tool.\"}",
                        &r, &a));
    n = events(r.body, ev, 2048, &a);
    ATH_REQUIRE(n >= 4);
    ATH_EXPECT_STR_EQ(jtype(ev[0]), "response.created");
    ATH_EXPECT_STR_EQ(jtype(ev[n - 1]), "response.completed");
    bool called = false, seq_ok = true;
    for (int i = 0; i < n; i++) {
        seq_ok &= jint(ev[i], "sequence_number") == i;
        if (!strcmp(jtype(ev[i]), "response.output_item.done")) {
            const char *it = NULL, *name = NULL, *args = NULL;
            ath_json_to_str(ath_json_path(ev[i], "item.type"), &it, NULL);
            ath_json_to_str(ath_json_path(ev[i], "item.name"), &name, NULL);
            ath_json_to_str(ath_json_path(ev[i], "item.arguments"), &args, NULL);
            if (it && !strcmp(it, "function_call")) {
                called = name && !strcmp(name, "get_weather") && args && strstr(args, "Rome");
                printf("    call: %s %s\n", name ? name : "?", args ? args : "?");
            }
        }
    }
    ATH_EXPECT(seq_ok);
    ATH_EXPECT(called);

    /* errors: a prompt longer than the context, a malformed body, an unknown path */
    ath_buf big;
    ath_buf_init(&big, &a);
    ath_buf_puts(&big, "{\"messages\": [{\"role\": \"user\", \"content\": \"");
    for (int i = 0; i < 3000; i++)
        ath_buf_puts(&big, "word ");
    ath_buf_puts(&big, "\"}]}");
    ATH_REQUIRE(request(s.srv.port, "/v1/messages", big.data, &r, &a));
    ATH_EXPECT_EQ_INT(r.status, 400);
    ATH_EXPECT(strstr(r.body, "prompt is too long") != NULL);
    ATH_REQUIRE(request(s.srv.port, "/v1/chat/completions", big.data, &r, &a));
    ATH_EXPECT(strstr(r.body, "context_length_exceeded") != NULL);
    ATH_REQUIRE(request(s.srv.port, "/v1/responses", "{\"input\": ", &r, &a));
    ATH_EXPECT_EQ_INT(r.status, 400);
    ATH_REQUIRE(request(s.srv.port, "/v1/nothing", "{}", &r, &a));
    ATH_EXPECT_EQ_INT(r.status, 404);
    ATH_REQUIRE(request(s.srv.port, "/v1/messages/count_tokens",
                        "{\"messages\": [{\"role\": \"user\", \"content\": \"Say hello.\"}]}", &r, &a));
    ATH_EXPECT(strstr(r.body, "\"input_tokens\": ") != NULL);

    server_stop(&s);
    ath_gen_close(&g);
    ath_arena_free(&a);
}

/* ---- the draft head in the server: the same answers ---------------------------------- */

typedef struct {
    ath_buf text;
} collect;

static bool c_text(void *u, const char *s, size_t n) { return ath_buf_append(&((collect *)u)->text, s, n) == ATH_OK; }
static bool c_think(void *u, const char *s, size_t n) { (void)u, (void)s, (void)n; return true; }
static bool c_begin(void *u, const char *name) { return ath_buf_printf(&((collect *)u)->text, "<call %s>", name) == ATH_OK; }
static bool c_end(void *u) { return ath_buf_puts(&((collect *)u)->text, "</call>") == ATH_OK; }

/* Greedy answers to a few conversations, the second extending the first. */
static bool answers(ath_test *t_, bool mtp, const char **out, int n, ath_arena *a)
{
    static ath_gen g;
    const ath_gen_opts go = {.ctx = 1024, .experts = -1, .seed = 1, .mtp = mtp};
    if (ath_gen_open(&g, model_dir(), &go, a) != ATH_OK) {
        ATH_EXPECT(!"gen opens");
        return false;
    }
    ATH_EXPECT(g.mtp == mtp);
    const ath_msg m1[] = {{.role = ATH_MSG_USER, .content = "Write three sentences about the sea."}};
    const ath_msg m2[] = {{.role = ATH_MSG_USER, .content = "Write three sentences about the sea."},
                          {.role = ATH_MSG_ASSISTANT, .content = "The sea is wide."},
                          {.role = ATH_MSG_USER, .content = "Now three about mountains, with a list."}};
    const ath_msg *convs[] = {m1, m2, m1};
    const int32_t lens[] = {1, 3, 1};
    bool ok = true;
    for (int i = 0; i < n && ok; i++) {
        collect c;
        ath_buf_init(&c.text, a);
        const ath_gen_sink sink = {.text = c_text, .thinking = c_think, .tool_begin = c_begin, .tool_args = c_text,
                                   .tool_end = c_end, .u = &c};
        ath_gen_request r = {.msgs = convs[i], .n_msgs = lens[i], .sampler = {0, 0, 1, 0, i == 2 ? 1.5f : 0},
                             .max_tokens = 60};
        ath_gen_result res;
        ok = ath_gen_run(&g, &r, &sink, &res, a) == ATH_OK;
        out[i] = ath_buf_str(&c.text);
    }
    if (mtp)
        printf("    with the draft head: %llu passes, %llu guesses kept\n", (unsigned long long)g.spec.passes,
               (unsigned long long)g.spec.accepted);
    ath_gen_close(&g);
    ATH_EXPECT(ok);
    return ok;
}

ATH_TEST(serve_mtp_same_answers)
{
    char cfg[1024];
    if (!model_file(cfg, sizeof cfg, "config.json"))
        ATH_SKIP("no checkpoint (ATH_MODEL_DIR)");
    ath_arena a;
    ath_arena_init(&a, 0);
    const char *plain[3], *spec[3];
    if (answers(t_, false, plain, 3, &a) && answers(t_, true, spec, 3, &a))
        for (int i = 0; i < 3; i++) {
            ATH_EXPECT_STR_EQ(spec[i], plain[i]);
            if (i == 0)
                printf("    answer: %.70s\n", plain[i]);
        }
    ath_arena_free(&a);
}

static const ath_test_case tests[] = {
    ATH_TEST_ENTRY(http_parse_whole_and_pipelined),
    ATH_TEST_ENTRY(http_parse_split),
    ATH_TEST_ENTRY(http_parse_chunked),
    ATH_TEST_ENTRY(http_parse_malformed),
    ATH_TEST_ENTRY(presence_reference),
    ATH_TEST_ENTRY(api_anthropic_request),
    ATH_TEST_ENTRY(api_chat_request),
    ATH_TEST_ENTRY(api_responses_request),
    ATH_TEST_ENTRY(http_loopback),
    ATH_TEST_ENTRY(serve_real_model),
    ATH_TEST_ENTRY(serve_mtp_same_answers),
};

int main(int argc, char **argv)
{
    return ath_test_main(tests, ATH_ARRAY_LEN(tests), argc, argv);
}
