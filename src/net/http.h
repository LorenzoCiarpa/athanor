/* http.h: a minimal HTTP/1.1 server on POSIX sockets (step 13).
 *
 * What a model server on localhost needs and nothing more: requests with a
 * body of known length or in chunks, keep-alive, `Expect: 100-continue`
 * (curl sends it before a long body), and responses either whole or as
 * server-sent events, the streaming format of every API the server speaks.
 * Streamed responses go in chunks (Transfer-Encoding: chunked), so the
 * connection stays usable after them.
 *
 * One thread, one loop: poll() over the listening socket and the open
 * connections; when a connection holds a whole request, the handler runs to
 * completion (a generation streams its events from inside it) and the loop
 * goes on. The engine runs one sequence at a time, so requests that arrive
 * meanwhile wait in their sockets. No TLS: localhost and trusted networks
 * only (src/net/README.md).
 */
#ifndef ATH_HTTP_H
#define ATH_HTTP_H

#include "core/arena.h"
#include "core/base.h"

#define ATH_HTTP_MAX_HEADER (64 * 1024)          /* request line and headers  */
#define ATH_HTTP_MAX_BODY   ((size_t)64 << 20)   /* a request body: 64 MiB    */
#define ATH_HTTP_MAX_CONNS  32

typedef struct {
    char        method[16];
    char        path[512];         /* without the query string                       */
    const char *body;              /* the body, NUL-terminated (chunks joined)        */
    size_t      body_len;
    bool        keep_alive;        /* HTTP/1.1 without "Connection: close"            */
    bool        expect_continue;   /* "Expect: 100-continue"                         */
} ath_http_req;

/* Parse one request at the start of buf[0 .. n): ATH_OK with *used the bytes
 * it took (a chunked body is joined into a copy in a, which then holds the
 * body); ATH_ERR_RANGE if the bytes end before the request does (read more;
 * *used is 0, or the length of the headers when they are complete, so that
 * a caller can answer 100-continue); ATH_ERR_PARSE for a malformed request;
 * ATH_ERR_UNSUPPORTED for one larger than the limits above. */
ath_status ath_http_parse(const char *buf, size_t n, ath_arena *a, ath_http_req *req, size_t *used);

/* Where a handler writes its response. */
typedef struct {
    int        fd;
    ath_arena *a;      /* the request's memory (the body is there): for the
                        * handler's working memory, reset after the response */
    bool keep_alive;   /* the connection stays open after this response    */
    bool streaming;    /* between ath_http_sse_start and ath_http_sse_end   */
    bool done;         /* a response was written                           */
    bool broken;       /* the client went away: a handler may stop early    */
} ath_http_out;

/* A whole response: status, content type, body. */
ath_status ath_http_reply(ath_http_out *o, int status, const char *content_type, const void *body, size_t n);
/* A stream of server-sent events: the headers, then events, then the end. */
ath_status ath_http_sse_start(ath_http_out *o);
/* One event: "event: <name>" (when name is not NULL) and "data: <data>"; the
 * data must not contain a newline (JSON does not). */
ath_status ath_http_sse(ath_http_out *o, const char *name, const char *data, size_t n);
ath_status ath_http_sse_end(ath_http_out *o);
/* A comment line, which clients ignore: a keep-alive while nothing else
 * goes out (a long prompt runs). */
ath_status ath_http_sse_ping(ath_http_out *o);

/* Is the client still there? False (and o->broken set) once it has closed
 * the connection; does not block. */
bool ath_http_alive(ath_http_out *o);

typedef void (*ath_http_handler)(void *ctx, const ath_http_req *req, ath_http_out *out);

typedef struct {
    int listen_fd;
    int port;          /* the port bound (with port 0: the one the system chose) */
} ath_http_server;

/* Listen on host:port (port 0: any free port, then in s->port). */
ath_status ath_http_listen(ath_http_server *s, const char *host, int port);

/* Serve requests with h until *stop becomes nonzero (checked every few
 * hundred milliseconds; NULL: forever). Each connection's memory comes
 * from an arena of its own, reset after every request. */
ath_status ath_http_run(ath_http_server *s, ath_http_handler h, void *ctx, const volatile int *stop);

void ath_http_close(ath_http_server *s);

#endif /* ATH_HTTP_H */
