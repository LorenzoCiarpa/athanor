/* threads.h: a fixed pool of threads for CPU work.
 *
 * A pool of n threads is the calling thread plus n − 1 workers, started once.
 * ath_pool_run hands one function to all of them and returns when every task
 * has run (fork and join). Tasks are assigned statically: thread i runs tasks
 * i, i + n, i + 2n, ... and the caller is thread 0. The tasks of one run must
 * be independent of each other, so that which thread runs a task, and when,
 * never changes a result.
 *
 *   ath_pool p;
 *   ATH_CHECK(ath_pool_init(&p, ath_pool_default_threads(), &arena));
 *   ath_pool_run(&p, n_tasks, fn, ctx);     // fn(ctx, task, n_tasks) for every task
 *   ath_pool_free(&p);
 *
 * Several threads may call ath_pool_run on the same pool; their runs take
 * turns. A NULL pool, or a pool of one thread, runs the tasks in order on the
 * caller. Workers sleep on a condition variable between runs.
 */
#ifndef ATH_THREADS_H
#define ATH_THREADS_H

#include "core/arena.h"

#include <pthread.h>

/* One task of a run: task in [0, n_tasks). */
typedef void (*ath_pool_fn)(void *ctx, int32_t task, int32_t n_tasks);

typedef struct ath_pool_worker ath_pool_worker;

/* The fields are private. A pool must not be moved or copied after init:
 * its workers keep its address. */
typedef struct {
    int32_t          n_threads;   /* 0 before init and after free            */
    ath_pool_worker *workers;     /* n_threads − 1, from the arena of init    */
    pthread_mutex_t  turn;        /* held by the caller of the current run    */
    pthread_mutex_t  mu;          /* guards the fields below                  */
    pthread_cond_t   wake, done;
    uint64_t         gen;         /* incremented by every run                 */
    int32_t          pending;     /* workers still busy with the current run  */
    bool             stop;
    ath_pool_fn      fn;
    void            *ctx;
    int32_t          n_tasks;
} ath_pool;

/* Number of online CPUs, at least 1: the default size of a pool. */
int32_t ath_pool_default_threads(void);

/* Start n_threads − 1 workers (n_threads >= 1; 1 starts none). Their
 * descriptors come from a, which must outlive the pool. On failure nothing
 * is left running. */
ath_status ath_pool_init(ath_pool *p, int32_t n_threads, ath_arena *a);

/* Stop and join the workers. Safe on a pool that was never initialized
 * (zeroed) or is already freed. */
void ath_pool_free(ath_pool *p);

/* Threads of the pool, counting the caller; 1 for NULL. */
int32_t ath_pool_threads(const ath_pool *p);

/* fn(ctx, task, n_tasks) for every task in [0, n_tasks), spread over the
 * threads of the pool; returns when all have returned. */
void ath_pool_run(ath_pool *p, int32_t n_tasks, ath_pool_fn fn, void *ctx);

#endif /* ATH_THREADS_H */
