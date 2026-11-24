/* threads.c: fixed pthread pool, fork and join. See threads.h.
 *
 * A run publishes (fn, ctx, n_tasks) under the mutex, bumps the generation
 * and wakes the workers; each worker runs its share, decrements `pending`
 * and goes back to sleep, and the last one wakes the caller. A worker
 * remembers the generation it last ran, so a wake-up without a new run (a
 * spurious one) puts it back to sleep. The caller waits until every worker
 * has finished before returning, so a worker can never miss a run.
 */
#include "core/threads.h"

#include <string.h>
#include <unistd.h>

struct ath_pool_worker {
    ath_pool *pool;
    int32_t   index;              /* 1 .. n_threads − 1; the caller is 0 */
    pthread_t thread;
};

static void run_share(ath_pool_fn fn, void *ctx, int32_t first, int32_t step, int32_t n_tasks)
{
    for (int32_t t = first; t < n_tasks; t += step)
        fn(ctx, t, n_tasks);
}

static void *worker_main(void *arg)
{
    const ath_pool_worker *w = arg;
    ath_pool *p = w->pool;
    uint64_t seen = 0;
    pthread_mutex_lock(&p->mu);
    for (;;) {
        while (p->gen == seen && !p->stop)
            pthread_cond_wait(&p->wake, &p->mu);
        if (p->stop)
            break;
        seen = p->gen;
        ath_pool_fn fn = p->fn;
        void *ctx = p->ctx;
        int32_t n_tasks = p->n_tasks;
        pthread_mutex_unlock(&p->mu);
        run_share(fn, ctx, w->index, p->n_threads, n_tasks);
        pthread_mutex_lock(&p->mu);
        if (--p->pending == 0)
            pthread_cond_signal(&p->done);
    }
    pthread_mutex_unlock(&p->mu);
    return NULL;
}

int32_t ath_pool_default_threads(void)
{
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n < 1 ? 1 : n > 4096 ? 4096 : (int32_t)n;
}

static void destroy_sync(ath_pool *p)
{
    pthread_cond_destroy(&p->done);
    pthread_cond_destroy(&p->wake);
    pthread_mutex_destroy(&p->mu);
    pthread_mutex_destroy(&p->turn);
}

ath_status ath_pool_init(ath_pool *p, int32_t n_threads, ath_arena *a)
{
    if (!p || n_threads < 1 || (n_threads > 1 && !a)) {
        ath_log(ATH_LOG_ERROR, "pool: %s", !p ? "missing pool" : n_threads < 1 ? "needs at least one thread"
                                                                                : "missing arena");
        return ATH_ERR_INVALID_ARG;
    }
    memset(p, 0, sizeof *p);
    if (pthread_mutex_init(&p->turn, NULL) || pthread_mutex_init(&p->mu, NULL) ||
        pthread_cond_init(&p->wake, NULL) || pthread_cond_init(&p->done, NULL)) {
        ath_log(ATH_LOG_ERROR, "pool: cannot create its mutexes and condition variables");
        return ATH_ERR_NOMEM;   /* on Linux these never fail; nothing to undo */
    }
    p->n_threads = 1;
    if (n_threads == 1)
        return ATH_OK;
    p->workers = ath_arena_zalloc(a, (size_t)(n_threads - 1) * sizeof *p->workers, _Alignof(ath_pool_worker));
    if (!p->workers) {
        destroy_sync(p);
        p->n_threads = 0;
        return ATH_ERR_NOMEM;
    }
    /* Workers read n_threads only during a run, and no run starts before
     * init returns, so it can be set once at the end. */
    for (int32_t i = 1; i < n_threads; i++) {
        ath_pool_worker *w = &p->workers[i - 1];
        w->pool = p;
        w->index = i;
        int e = pthread_create(&w->thread, NULL, worker_main, w);
        if (e) {
            ath_log(ATH_LOG_ERROR, "pool: cannot start thread %d of %d: %s", i + 1, n_threads, strerror(e));
            p->n_threads = i;   /* the workers started so far */
            ath_pool_free(p);
            return ATH_ERR_NOMEM;
        }
    }
    p->n_threads = n_threads;
    return ATH_OK;
}

void ath_pool_free(ath_pool *p)
{
    if (!p || p->n_threads == 0)
        return;
    pthread_mutex_lock(&p->mu);
    p->stop = true;
    pthread_cond_broadcast(&p->wake);
    pthread_mutex_unlock(&p->mu);
    for (int32_t i = 0; i + 1 < p->n_threads; i++)
        pthread_join(p->workers[i].thread, NULL);
    destroy_sync(p);
    memset(p, 0, sizeof *p);
}

int32_t ath_pool_threads(const ath_pool *p)
{
    return p && p->n_threads > 1 ? p->n_threads : 1;
}

void ath_pool_run(ath_pool *p, int32_t n_tasks, ath_pool_fn fn, void *ctx)
{
    if (!fn || n_tasks < 1)
        return;
    if (ath_pool_threads(p) == 1 || n_tasks == 1) {
        run_share(fn, ctx, 0, 1, n_tasks);
        return;
    }
    pthread_mutex_lock(&p->turn);
    pthread_mutex_lock(&p->mu);
    p->fn = fn;
    p->ctx = ctx;
    p->n_tasks = n_tasks;
    p->pending = p->n_threads - 1;
    p->gen++;
    pthread_cond_broadcast(&p->wake);
    pthread_mutex_unlock(&p->mu);
    run_share(fn, ctx, 0, p->n_threads, n_tasks);
    pthread_mutex_lock(&p->mu);
    while (p->pending > 0)
        pthread_cond_wait(&p->done, &p->mu);
    pthread_mutex_unlock(&p->mu);
    pthread_mutex_unlock(&p->turn);
}
