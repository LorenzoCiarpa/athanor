/* http.c: the HTTP/1.1 server. See http.h. */
#include "net/http.h"

#include "core/buf.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <unistd.h>

/* ---- parsing -------------------------------------------------------------------- */

/* The end of the headers: the offset just after "\r\n\r\n", or 0. */
static size_t header_end(const char *b, size_t n)
{
    for (size_t i = 3; i < n; i++)
        if (b[i] == '\n' && b[i - 1] == '\r' && b[i - 2] == '\n' && b[i - 3] == '\r')
            return i + 1;
    return 0;
}

/* The value of header `name` in the header block [h, end), trimmed, into
 * out; false if absent. */
static bool header(const char *h, const char *end, const char *name, char *out, size_t cap)
{
    const size_t nl = strlen(name);
    for (const char *line = h; line < end;) {
        const char *eol = memchr(line, '\n', (size_t)(end - line));
        if (!eol)
            eol = end;
        if ((size_t)(eol - line) > nl && line[nl] == ':' && !strncasecmp(line, name, nl)) {
            const char *v = line + nl + 1, *ve = eol;
            while (v < ve && (*v == ' ' || *v == '\t'))
                v++;
            while (ve > v && (ve[-1] == '\r' || ve[-1] == ' ' || ve[-1] == '\t'))
                ve--;
            size_t len = (size_t)(ve - v) < cap - 1 ? (size_t)(ve - v) : cap - 1;
            memcpy(out, v, len);
            out[len] = '\0';
            return true;
        }
        line = eol + 1;
    }
    return false;
}

static bool has_token(const char *v, const char *tok)
{
    for (const char *p = v; *p;) {
        while (*p == ' ' || *p == ',')
            p++;
        size_t l = strcspn(p, ", ");
        if (l == strlen(tok) && !strncasecmp(p, tok, l))
            return true;
        p += l;
    }
    return false;
}

/* A chunked body at b[0 .. n): ATH_OK with the bytes it took and the data
 * joined in out; ATH_ERR_RANGE if incomplete. */
static ath_status chunked(const char *b, size_t n, ath_buf *out, size_t *used)
{
    size_t at = 0;
    for (;;) {
        const char *eol = memchr(b + at, '\n', n - at);
        if (!eol)
            return ATH_ERR_RANGE;
        char *endp;
        unsigned long long len = strtoull(b + at, &endp, 16);
        if (endp == b + at)
            return ATH_ERR_PARSE;
        at = (size_t)(eol - b) + 1;
        if (len > ATH_HTTP_MAX_BODY || out->len + len > ATH_HTTP_MAX_BODY)
            return ATH_ERR_UNSUPPORTED;
        if (len == 0) {
            /* trailers, then an empty line */
            for (;;) {
                const char *t = memchr(b + at, '\n', n - at);
                if (!t)
                    return ATH_ERR_RANGE;
                bool empty = t == b + at || (t == b + at + 1 && b[at] == '\r');
                at = (size_t)(t - b) + 1;
                if (empty) {
                    *used = at;
                    return ATH_OK;
                }
            }
        }
        if (n - at < len + 2)
            return ATH_ERR_RANGE;
        ATH_CHECK(ath_buf_append(out, b + at, (size_t)len));
        at += (size_t)len;
        if (b[at] == '\r')
            at++;
        if (at >= n || b[at] != '\n')
            return ATH_ERR_PARSE;
        at++;
    }
}

ath_status ath_http_parse(const char *buf, size_t n, ath_arena *a, ath_http_req *req, size_t *used)
{
    memset(req, 0, sizeof *req);
    *used = 0;
    const size_t he = header_end(buf, n);
    if (!he)
        return n > ATH_HTTP_MAX_HEADER ? ATH_ERR_UNSUPPORTED : ATH_ERR_RANGE;
    /* request line: METHOD SP target SP HTTP/1.x */
    const char *eol = memchr(buf, '\n', he);
    const char *sp1 = memchr(buf, ' ', (size_t)(eol - buf));
    const char *sp2 = sp1 ? memchr(sp1 + 1, ' ', (size_t)(eol - sp1 - 1)) : NULL;
    if (!sp1 || !sp2 || sp1 == buf || (size_t)(sp1 - buf) >= sizeof req->method || eol - sp2 < 9 ||
        strncmp(sp2 + 1, "HTTP/1.", 7))
        return ATH_ERR_PARSE;
    memcpy(req->method, buf, (size_t)(sp1 - buf));
    const char *target = sp1 + 1, *q = memchr(target, '?', (size_t)(sp2 - target));
    const size_t plen = (size_t)((q ? q : sp2) - target);
    if (plen == 0 || plen >= sizeof req->path)
        return ATH_ERR_PARSE;
    memcpy(req->path, target, plen);
    const bool http10 = sp2[8] == '0';
    const char *h = eol + 1, *hend = buf + he;
    char v[256];
    req->keep_alive = !http10;
    if (header(h, hend, "Connection", v, sizeof v)) {
        if (has_token(v, "close"))
            req->keep_alive = false;
        else if (has_token(v, "keep-alive"))
            req->keep_alive = true;
    }
    req->expect_continue = header(h, hend, "Expect", v, sizeof v) && !strcasecmp(v, "100-continue");

    if (header(h, hend, "Transfer-Encoding", v, sizeof v) && has_token(v, "chunked")) {
        ath_buf body;
        ath_buf_init(&body, a);
        size_t took = 0;
        ath_status s = chunked(buf + he, n - he, &body, &took);
        if (s == ATH_ERR_RANGE) {
            *used = he;             /* the headers are complete: 100-continue may go */
            return ATH_ERR_RANGE;
        }
        ATH_CHECK(s);
        req->body = ath_buf_str(&body);
        req->body_len = body.len;
        *used = he + took;
        return ATH_OK;
    }
    unsigned long long len = 0;
    if (header(h, hend, "Content-Length", v, sizeof v)) {
        char *endp;
        len = strtoull(v, &endp, 10);
        if (v[0] < '0' || v[0] > '9' || *endp)
            return ATH_ERR_PARSE;
    }
    if (len > ATH_HTTP_MAX_BODY)
        return ATH_ERR_UNSUPPORTED;
    if (n - he < len) {
        *used = he;                 /* the headers are complete: 100-continue may go */
        return ATH_ERR_RANGE;
    }
    /* a NUL-terminated copy, so that the JSON parser and strtod stop there */
    char *body = ath_arena_alloc(a, (size_t)len + 1, 1);
    if (!body)
        return ATH_ERR_NOMEM;
    memcpy(body, buf + he, (size_t)len);
    body[len] = '\0';
    req->body = body;
    req->body_len = (size_t)len;
    *used = he + (size_t)len;
    return ATH_OK;
}

/* ---- writing -------------------------------------------------------------------- */

static ath_status send_all(ath_http_out *o, const void *p, size_t n)
{
    const char *c = p;
    while (n && !o->broken) {
        ssize_t w = send(o->fd, c, n, MSG_NOSIGNAL);
        if (w < 0 && errno == EINTR)
            continue;
        if (w <= 0) {
            o->broken = true;
            return ATH_ERR_IO;
        }
        c += w;
        n -= (size_t)w;
    }
    return o->broken ? ATH_ERR_IO : ATH_OK;
}

static const char *reason(int status)
{
    switch (status) {
    case 200: return "OK";
    case 400: return "Bad Request";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 413: return "Payload Too Large";
    case 500: return "Internal Server Error";
    case 503: return "Service Unavailable";
    }
    return "Status";
}

ath_status ath_http_reply(ath_http_out *o, int status, const char *content_type, const void *body, size_t n)
{
    char head[512];
    int hl = snprintf(head, sizeof head,
                      "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\nConnection: %s\r\n\r\n", status,
                      reason(status), content_type, n, o->keep_alive ? "keep-alive" : "close");
    o->done = true;
    ATH_CHECK(send_all(o, head, (size_t)hl));
    return n ? send_all(o, body, n) : ATH_OK;
}

ath_status ath_http_sse_start(ath_http_out *o)
{
    static const char head[] = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
                               "Transfer-Encoding: chunked\r\nConnection: keep-alive\r\n\r\n";
    o->done = true;
    o->streaming = true;
    return send_all(o, head, sizeof head - 1);
}

ath_status ath_http_sse(ath_http_out *o, const char *name, const char *data, size_t n)
{
    char pre[160];
    int pl = name ? snprintf(pre, sizeof pre, "event: %s\ndata: ", name) : snprintf(pre, sizeof pre, "data: ");
    char size[32];
    int sl = snprintf(size, sizeof size, "%zx\r\n", (size_t)pl + n + 2);
    ATH_CHECK(send_all(o, size, (size_t)sl));
    ATH_CHECK(send_all(o, pre, (size_t)pl));
    ATH_CHECK(send_all(o, data, n));
    return send_all(o, "\n\n\r\n", 4);
}

ath_status ath_http_sse_ping(ath_http_out *o)
{
    static const char ping[] = "8\r\n: ping\n\n\r\n";
    return send_all(o, ping, sizeof ping - 1);
}

bool ath_http_alive(ath_http_out *o)
{
    if (o->broken)
        return false;
    struct pollfd p = {.fd = o->fd, .events = POLLRDHUP};
    if (poll(&p, 1, 0) > 0 && (p.revents & (POLLRDHUP | POLLHUP | POLLERR)))
        o->broken = true;
    return !o->broken;
}

ath_status ath_http_sse_end(ath_http_out *o)
{
    o->streaming = false;
    return send_all(o, "0\r\n\r\n", 5);
}

/* ---- the loop ------------------------------------------------------------------- */

typedef struct {
    int       fd;
    ath_arena a;          /* the request's memory, reset after each one         */
    ath_buf   in;         /* bytes read and not yet taken by a request          */
    bool      continued;  /* 100-continue sent for the request being read        */
} conn;

ath_status ath_http_listen(ath_http_server *s, const char *host, int port)
{
    memset(s, 0, sizeof *s);
    s->listen_fd = -1;
    struct sockaddr_in addr = {.sin_family = AF_INET, .sin_port = htons((uint16_t)port)};
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
        ath_log(ATH_LOG_ERROR, "http: '%s' is not an IPv4 address", host);
        return ATH_ERR_INVALID_ARG;
    }
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0)
        return ATH_ERR_IO;
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    if (bind(fd, (struct sockaddr *)&addr, sizeof addr) || listen(fd, 64)) {
        ath_log(ATH_LOG_ERROR, "http: cannot listen on %s:%d: %s", host, port, strerror(errno));
        close(fd);
        return ATH_ERR_IO;
    }
    socklen_t al = sizeof addr;
    getsockname(fd, (struct sockaddr *)&addr, &al);
    s->listen_fd = fd;
    s->port = ntohs(addr.sin_port);
    return ATH_OK;
}

void ath_http_close(ath_http_server *s)
{
    if (s && s->listen_fd >= 0)
        close(s->listen_fd);
    if (s)
        s->listen_fd = -1;
}

static void conn_close(conn *c)
{
    close(c->fd);
    ath_arena_free(&c->a);
    c->fd = -1;
}

/* Serve every whole request in c's buffer; false when the connection is to
 * be closed. */
static bool serve_conn(conn *c, ath_http_handler h, void *ctx)
{
    for (;;) {
        ath_http_req req;
        size_t used = 0;
        ath_arena_pos mark = ath_arena_mark(&c->a);
        ath_status st = ath_http_parse(c->in.data ? c->in.data : "", c->in.len, &c->a, &req, &used);
        ath_http_out out = {.fd = c->fd, .a = &c->a, .keep_alive = req.keep_alive};
        if (st == ATH_ERR_RANGE) {
            ath_arena_reset(&c->a, mark);
            if (used && req.expect_continue && !c->continued) {
                c->continued = true;
                static const char cont[] = "HTTP/1.1 100 Continue\r\n\r\n";
                if (send_all(&out, cont, sizeof cont - 1) != ATH_OK)
                    return false;
            }
            return true;
        }
        c->continued = false;
        if (st != ATH_OK) {
            out.keep_alive = false;
            static const char msg[] = "{\"error\": {\"type\": \"invalid_request_error\", \"message\": \"malformed HTTP request\"}}";
            ath_http_reply(&out, st == ATH_ERR_UNSUPPORTED ? 413 : 400, "application/json", msg, sizeof msg - 1);
            return false;
        }
        h(ctx, &req, &out);
        if (out.streaming)
            ath_http_sse_end(&out);
        if (!out.done) {
            static const char msg[] = "{\"error\": {\"type\": \"api_error\", \"message\": \"no response\"}}";
            ath_http_reply(&out, 500, "application/json", msg, sizeof msg - 1);
        }
        /* drop the request's bytes, keep what follows */
        const size_t rest = c->in.len - used;
        char keep[4096];
        if (rest <= sizeof keep) {
            if (rest)
                memcpy(keep, c->in.data + used, rest);
            ath_arena_reset(&c->a, (ath_arena_pos){0});
            ath_buf_init(&c->in, &c->a);
            if (rest && ath_buf_append(&c->in, keep, rest) != ATH_OK)
                return false;
        } else {
            /* a long pipelined tail: move it down within the buffer */
            memmove(c->in.data, c->in.data + used, rest);
            c->in.len = rest;
        }
        if (!out.keep_alive || out.broken)
            return false;
        if (!c->in.len)
            return true;
    }
}

ath_status ath_http_run(ath_http_server *s, ath_http_handler h, void *ctx, const volatile int *stop)
{
    conn conns[ATH_HTTP_MAX_CONNS];
    for (int i = 0; i < ATH_HTTP_MAX_CONNS; i++)
        conns[i].fd = -1;
    struct pollfd pf[ATH_HTTP_MAX_CONNS + 1];
    int map[ATH_HTTP_MAX_CONNS + 1];
    while (!stop || !*stop) {
        int n = 0;
        pf[n] = (struct pollfd){.fd = s->listen_fd, .events = POLLIN};
        map[n++] = -1;
        for (int i = 0; i < ATH_HTTP_MAX_CONNS; i++)
            if (conns[i].fd >= 0) {
                pf[n] = (struct pollfd){.fd = conns[i].fd, .events = POLLIN};
                map[n++] = i;
            }
        int r = poll(pf, (nfds_t)n, 300);
        if (r < 0 && errno == EINTR)
            continue;
        if (r < 0)
            return ATH_ERR_IO;
        for (int k = 0; k < n; k++) {
            if (!pf[k].revents)
                continue;
            if (map[k] < 0) {
                int fd = accept(s->listen_fd, NULL, NULL);
                if (fd < 0)
                    continue;
                int slot = -1;
                for (int i = 0; i < ATH_HTTP_MAX_CONNS && slot < 0; i++)
                    if (conns[i].fd < 0)
                        slot = i;
                if (slot < 0) {
                    close(fd);
                    continue;
                }
                int one = 1;
                setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);   /* events go out at once */
                conns[slot].fd = fd;
                conns[slot].continued = false;
                ath_arena_init(&conns[slot].a, 0);
                ath_buf_init(&conns[slot].in, &conns[slot].a);
                continue;
            }
            conn *c = &conns[map[k]];
            char buf[65536];
            ssize_t got = recv(c->fd, buf, sizeof buf, 0);
            if (got < 0 && errno == EINTR)
                continue;
            if (got <= 0 || ath_buf_append(&c->in, buf, (size_t)got) != ATH_OK || !serve_conn(c, h, ctx))
                conn_close(c);
        }
    }
    for (int i = 0; i < ATH_HTTP_MAX_CONNS; i++)
        if (conns[i].fd >= 0)
            conn_close(&conns[i]);
    return ATH_OK;
}
