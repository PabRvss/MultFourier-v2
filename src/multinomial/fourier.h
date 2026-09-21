#pragma once

typedef struct {  // data dependent on n
    vec_t cos;  // cosine n
    vec_t sin;  // sine n
} local_t;

typedef struct {
    real_t log_mod;
    vec_t unit_re;
    vec_t unit_im;
} result_t;

typedef struct {
    const multinomial* mult;

    local_t* restrict local;   // scratch: exp(i*m*theta) for one category, all STRIDE lanes
    result_t* restrict vec_arr;
    // complex_t* restrict res_arr;
    qcomplex_t* restrict res_arr;
    const void* restrict plan;

    // Phase recurrence, all indexed by cell k*(N+1)+i. See phase_load below.
    complex_t* restrict phase;         // per thread: exp(i*n*STRIDE*theta) for its current vector
    const local_t* restrict lane;      // shared: exp(i*s*theta) for s = 0 .. STRIDE-1
    const complex_t* restrict step_z;  // shared: exp(i*stride_n*STRIDE*theta)

    int_t thread;
    int_t start;
    int_t end;
    int_t stride_n;  // this thread evaluates n = start, start+stride_n, ... < end
    int_t warm;      // 0: seed phase from scratch at n = start; 1: continue the recurrence
} arguments_t;

// The evaluators all need exp(i*m*theta_{k,i}), with theta = T*reward the transformed reward of
// cell (k,i) and m = n*STRIDE + s the STRIDE frequencies carried by vector index n. Done directly
// that is K*(N+1)*STRIDE cos/sin calls per vector -- measured at roughly three times the cost of
// the convolution those values feed into.
//
// They are not independent, and the only thing that moves is n:
//
//     exp(i*m*theta) = exp(i*n*STRIDE*theta) * exp(i*s*theta).
//
// The right factor has no n in it at all (lane, built once and shared by every thread), and the
// left one is multiplied by a constant exp(i*stride_n*STRIDE*theta) whenever this thread steps to
// its next vector (step_z, also shared). So a thread carries one complex number per cell (phase)
// and the whole frequency loop runs on multiplications: no transcendental call survives inside it.
//
// This is why eval_fourier_parallel hands thread t the frequencies n = t (mod threads) instead of a
// contiguous block. A fixed stride in n is what makes the advance factor constant, and it is also
// what keeps the recurrence alive across the chunk boundary in series(), which re-creates the
// threads for every chunk; thread t's frequencies form one arithmetic progression over the whole
// run, so it only ever seeds its phase once.
//
// Accuracy improves as a side effect: cos(m*theta) first rounds its argument, costing
// |m*theta|*eps, which at m ~ 1e5 is already ~3e-11, whereas the recurrence accumulates about one
// eps per chunk and drifts in modulus only as sqrt(chunks)*eps.

static void phase_seed(const arguments_t* restrict args, const int_t n0, const int_t cells) {
    const cell_t* restrict global = args->mult->matrix;
    complex_t* restrict phase = args->phase;
    const real_t m0 = (real_t)(n0*STRIDE);
    for (int_t c=0; c<cells; c++) {
        const real_t arg = m0*global[c].reward;
        phase[c] = cos(arg) + I*sin(arg);
    }
}

static inline void phase_advance(const arguments_t* restrict args, const int_t cells) {
    complex_t* restrict phase = args->phase;
    const complex_t* restrict step_z = args->step_z;
    for (int_t c=0; c<cells; c++) phase[c] *= step_z[c];
}

// Materialise exp(i*m*theta) for every count i of category k into args->local[0..N].
static inline void phase_load(const arguments_t* restrict args, const int_t index_k, const int_t N) {
    const complex_t* restrict phase = args->phase;
    const local_t* restrict lane = args->lane;
    local_t* restrict out = args->local;
    for (int_t i=0; i<=N; i++) {
        const complex_t p = phase[index_k + i];
        const real_t pr = creal(p);
        const real_t pi = cimag(p);
        const local_t l = lane[index_k + i];
        out[i].cos = pr*l.cos - pi*l.sin;
        out[i].sin = pr*l.sin + pi*l.cos;
    }
}

// Build the two shared tables. Called once per series(), after fill_cells has scaled the rewards.
static void phase_tables(
        const multinomial* restrict mult, const int_t stride_n,
        local_t* restrict lane, complex_t* restrict step_z
) {
    const int_t cells = mult->K*(mult->N + 1);
    const cell_t* restrict global = mult->matrix;
    for (int_t c=0; c<cells; c++) {
        const real_t theta = global[c].reward;
#ifdef USE_SIMD
        for (int s=0; s<STRIDE; s++) {
            lane[c].cos[s] = cos((real_t)s*theta);
            lane[c].sin[s] = sin((real_t)s*theta);
        }
#else
        lane[c].cos = 1;
        lane[c].sin = 0;
#endif
        const real_t a = (real_t)(stride_n*STRIDE)*theta;
        step_z[c] = cos(a) + I*sin(a);
    }
}

#include "error_bound.h"
#include "extrapolate.h"
#include "fourier_fft.h"
#include "fourier_fft_precompute.h"
#include "fourier_poisson.h"
// #include "../multinomial_quad/fourier_fft_quad.h"
// #include "../multinomial_quad/fourier_fft_quad_precompute.h"
// #include "../multinomial_dd/fourier_fft_dd.h"
// #include "../multinomial_dd/fourier_fft_dd_precompute.h"

static void* eval_fourier_thread(void* args_void) {
    arguments_t* args = args_void;

    const multinomial* restrict mult = args->mult;
    const cell_t* restrict global = mult->matrix;
    local_t* restrict local = args->local;
    result_t* restrict vec_arr = args->vec_arr;
    // complex_t* restrict res_arr = args->res_arr;
    qcomplex_t* restrict res_arr = args->res_arr;

    const int_t N = mult->N;
    const int_t K = mult->K;
    const int_t cells = K*(N+1);

    if (!args->warm) phase_seed(args, args->start, cells);

    for (int_t n=args->start; n<args->end; n+=args->stride_n) {
        for (int_t i=0; i<=N; i++) {
            vec_arr[i].log_mod = -INFINITY;
            vec_arr[i].unit_re = VEC_ONE;
            vec_arr[i].unit_im = VEC_ZERO;
        }
        vec_arr[0].log_mod = 0;

        for (int_t k=K-1; k>=0; k--) {  // reversed
            const int_t index_k = k*(N+1);

            phase_load(args, index_k, N);  // exp(i*m*theta) for this k, no transcendentals

            for (int_t i=N; i>=0; i--) {  // reversed
                const real_t shift_ki = -global[index_k + i].shift;
                vec_t res_re = VEC_ZERO;
                vec_t res_im = VEC_ZERO;

                for (int_t j=0; j<=i; j++) {
                    const int_t ij = i-j;
                    const vec_t term_unit_re = local[ij].cos;
                    const vec_t term_unit_im = local[ij].sin;

                    const result_t vecj = vec_arr[j];
                    const vec_t z_re = term_unit_re*vecj.unit_re + term_unit_im*vecj.unit_im;
                    const vec_t z_im = term_unit_re*vecj.unit_im - term_unit_im*vecj.unit_re;

                    const real_t exponent_re = global[index_k + ij].log_prob + shift_ki + vecj.log_mod;
                    const real_t term_mod = EXP(exponent_re);

                    res_re += term_mod*z_re;
                    res_im += term_mod*z_im;
                }

                vec_arr[i].log_mod = -shift_ki;
                vec_arr[i].unit_re = res_re;
                vec_arr[i].unit_im = res_im;
            }
        }

        const real_t s2_re = M_PI*mult->gamma/mult->T;
        // const double resN_exp = exp((double)vec_arr[N].log_mod + lgac[N]);
        const quad_t resN_exp = expq(vec_arr[N].log_mod + lgac[N]);
#ifdef USE_SIMD
        vec_t s2_im; for (int_t i=0; i<STRIDE; i++) s2_im[i] = (n*STRIDE + i)*M_PI;

        for (int_t i=0; i<STRIDE; i++) {
            const complex_t unit = vec_arr[N].unit_re[i] + I*vec_arr[N].unit_im[i];
            const complex_t s2 = s2_re + I*s2_im[i];
            complex_t sinc = (1-EXPC(-s2))/s2;
            if ((n*STRIDE+i == 0) && (ABS(s2_re)<=EPS)) sinc = 1 - s2*(0.5 - s2/6);
            res_arr[n*STRIDE+i] = resN_exp*(unit*sinc);
        }
#else
        const complex_t s2 = s2_re + I*n*M_PI;
        complex_t sinc = (1-EXPC(-s2))/s2;
        if ((n == 0) && (ABS(s2_re)<=EPS)) sinc = 1 - s2*(0.5 - s2/6);

        const complex_t unit = vec_arr[N].unit_re + I*vec_arr[N].unit_im;
        res_arr[n] = resN_exp*(unit*sinc);
#endif
        phase_advance(args, cells);
    }
    return 0;
}

static void* (*select_fourier_eval_fn(const options_t* restrict options))(void*) {
    if (options->poisson) return eval_fourier_poisson_thread;
    switch (options->fft_precision) {
        case FFT_NONE: return eval_fourier_thread;
        case FFT_REAL: return options->use_fft_precompute ? eval_fourier_fft_precompute_thread : eval_fourier_fft_sp_thread;
        default: break;
    }
    return eval_fourier_thread;
}

// Thread i takes n = n_min+i, n_min+i+threads, ... : a fixed stride, so that its phase recurrence
// has a constant advance factor and survives the chunk boundary (see phase_load).
static void eval_fourier_parallel(
        const multinomial* restrict mult, const int_t n_min, const int_t n_max,
        local_t* restrict local_arrs, result_t* restrict vec_arrs,
        // complex_t* res_arr,
        qcomplex_t* res_arr,
        options_t* options, const void* restrict plan,
        complex_t* restrict phase_arrs, const local_t* restrict lane,
        const complex_t* restrict step_z, const int_t warm
) {
    const int_t threads = options->threads;
    const int_t N = mult->N;
    const int_t cells = mult->K*(N+1);
    pthread_t* tid = (pthread_t*)malloc(threads*sizeof(pthread_t));
    arguments_t* args_array = malloc(threads*sizeof(arguments_t));

    void* (*eval_fn)(void*) = select_fourier_eval_fn(options);

    for (int_t i=0; i<threads; i++) {
        arguments_t* args_i = &args_array[i];
        args_i->mult = mult;
        args_i->local = &local_arrs[i*(N+1)];
        args_i->vec_arr = &vec_arrs[i*(N+1)];
        args_i->res_arr = res_arr;
        args_i->plan = plan;

        args_i->phase = &phase_arrs[i*cells];
        args_i->lane = lane;
        args_i->step_z = step_z;

        args_i->thread = i;
        args_i->start = n_min + i;
        args_i->end = n_max;
        args_i->stride_n = threads;
        args_i->warm = warm;

        pthread_create(&tid[i], NULL, eval_fn, args_i);
    }
    for (size_t i=0; i<threads; i++) pthread_join(tid[i], NULL);

    free(tid);
    free(args_array);
}

static void eval_fourier_sequential(
        const multinomial* restrict mult, const int_t n_min, const int_t n_max,
        local_t* restrict local_arrs, result_t* restrict vec_arrs,
        // complex_t* res_arr,
        qcomplex_t* res_arr,
        options_t* options, const void* restrict plan,
        complex_t* restrict phase_arrs, const local_t* restrict lane,
        const complex_t* restrict step_z, const int_t warm
) {
    arguments_t args;
    arguments_t* args_i = &args;
    args_i->mult = mult;
    args_i->local = local_arrs;
    args_i->vec_arr = vec_arrs;
    args_i->res_arr = res_arr;
    args_i->plan = plan;

    args_i->phase = phase_arrs;
    args_i->lane = lane;
    args_i->step_z = step_z;

    args_i->thread = 0;
    args_i->start = n_min;
    args_i->end = n_max;
    args_i->stride_n = 1;
    args_i->warm = warm;

    select_fourier_eval_fn(options)(args_i);
}

// A persistent thread pool for series(), which calls the evaluator once per chunk (typically
// hundreds to tens of thousands of times per run). eval_fourier_parallel above creates and joins
// `threads` pthreads on every call, which is wasted work repeated once per chunk; on this machine
// creating and joining 8 threads costs on the order of 10 microseconds, comparable to or larger
// than a whole chunk's worth of series evaluation once the Poissonised evaluator or a small
// instance is in use (see latex/poisson.tex, Table on per-frequency cost). Here the threads are
// created once, park on a condition variable between chunks, and are joined once at the end.
//
// Everything that does not change from one chunk to the next -- mult, res_arr, plan, the phase
// tables, the evaluator function pointer -- is written into each worker's arguments_t once, at
// pool creation. Only start/end/warm change per chunk, and fourier_pool_run writes those under the
// pool's mutex before waking the workers, exactly the state a reusable barrier needs to protect.
typedef struct {
    int_t threads;              // 1 means: no real threads, run on the caller
    arguments_t* restrict args; // one persistent slot per thread
    void* (*eval_fn)(void*);

    pthread_t* restrict tid;
    pthread_mutex_t mutex;
    pthread_cond_t work_cond;   // workers wait on this for the next chunk
    pthread_cond_t done_cond;   // the caller waits on this for the chunk to finish
    int_t generation;           // bumped by the caller to release the workers for one round
    int_t done_count;
    int stop;
} fourier_pool_t;

typedef struct { fourier_pool_t* pool; int_t idx; } fourier_pool_worker_ctx_t;

static void* fourier_pool_worker_run(void* arg) {
    fourier_pool_worker_ctx_t* ctx = arg;
    fourier_pool_t* restrict pool = ctx->pool;
    arguments_t* restrict my_args = &pool->args[ctx->idx];
    int_t my_gen = 0;

    for (;;) {
        pthread_mutex_lock(&pool->mutex);
        while (pool->generation == my_gen && !pool->stop) pthread_cond_wait(&pool->work_cond, &pool->mutex);
        const int stopping = pool->stop;
        my_gen = pool->generation;
        pthread_mutex_unlock(&pool->mutex);
        if (stopping) break;

        pool->eval_fn(my_args);

        pthread_mutex_lock(&pool->mutex);
        pool->done_count++;
        if (pool->done_count == pool->threads) pthread_cond_signal(&pool->done_cond);
        pthread_mutex_unlock(&pool->mutex);
    }
    free(ctx);
    return NULL;
}

static fourier_pool_t* fourier_pool_create(
        const multinomial* restrict mult, options_t* restrict options, const void* restrict plan,
        qcomplex_t* restrict res_arr, local_t* restrict local_arrs, result_t* restrict vec_arrs,
        complex_t* restrict phase_arrs, const local_t* restrict lane, const complex_t* restrict step_z
) {
    const int_t N = mult->N;
    const int_t cells = mult->K*(N+1);
    const int_t threads = options->threads > 1 ? options->threads : 1;

    fourier_pool_t* pool = malloc(sizeof(fourier_pool_t));
    pool->threads = threads;
    pool->args = malloc(threads*sizeof(arguments_t));
    pool->eval_fn = select_fourier_eval_fn(options);
    pool->tid = NULL;
    pool->generation = 0;
    pool->done_count = 0;
    pool->stop = 0;

    for (int_t i=0; i<threads; i++) {
        arguments_t* restrict a = &pool->args[i];
        a->mult = mult;
        a->local = &local_arrs[i*(N+1)];
        a->vec_arr = &vec_arrs[i*(N+1)];
        a->res_arr = res_arr;
        a->plan = plan;
        a->phase = &phase_arrs[i*cells];
        a->lane = lane;
        a->step_z = step_z;
        a->thread = i;
        a->stride_n = threads;
    }

    if (threads > 1) {
        pthread_mutex_init(&pool->mutex, NULL);
        pthread_cond_init(&pool->work_cond, NULL);
        pthread_cond_init(&pool->done_cond, NULL);
        pool->tid = malloc(threads*sizeof(pthread_t));
        for (int_t i=0; i<threads; i++) {
            fourier_pool_worker_ctx_t* ctx = malloc(sizeof(fourier_pool_worker_ctx_t));
            ctx->pool = pool;
            ctx->idx = i;
            pthread_create(&pool->tid[i], NULL, fourier_pool_worker_run, ctx);
        }
    }
    return pool;
}

// Run one chunk: n = n_min+i, n_min+i+threads, ... < n_max for thread i (thread 0 alone when there
// is no pool), matching the striding eval_fourier_parallel uses so the phase recurrence's constant
// advance factor (see phase_load) is unaffected by this refactor.
static void fourier_pool_run(fourier_pool_t* restrict pool, const int_t n_min, const int_t n_max, const int_t warm) {
    const int_t threads = pool->threads;

    if (threads <= 1) {
        arguments_t* restrict a = &pool->args[0];
        a->start = n_min;
        a->end = n_max;
        a->stride_n = 1;
        a->warm = warm;
        pool->eval_fn(a);
        return;
    }

    pthread_mutex_lock(&pool->mutex);
    for (int_t i=0; i<threads; i++) {
        arguments_t* restrict a = &pool->args[i];
        a->start = n_min + i;
        a->end = n_max;
        a->warm = warm;
    }
    pool->done_count = 0;
    pool->generation++;
    pthread_cond_broadcast(&pool->work_cond);
    while (pool->done_count < threads) pthread_cond_wait(&pool->done_cond, &pool->mutex);
    pthread_mutex_unlock(&pool->mutex);
}

static void fourier_pool_free(fourier_pool_t* pool) {
    if (pool->threads > 1) {
        pthread_mutex_lock(&pool->mutex);
        pool->stop = 1;
        pthread_cond_broadcast(&pool->work_cond);
        pthread_mutex_unlock(&pool->mutex);
        for (int_t i=0; i<pool->threads; i++) pthread_join(pool->tid[i], NULL);
        free(pool->tid);
        pthread_mutex_destroy(&pool->mutex);
        pthread_cond_destroy(&pool->work_cond);
        pthread_cond_destroy(&pool->done_cond);
    }
    free(pool->args);
    free(pool);
}

// The shallow table depth and window fraction for the extrapolate >= 2 stopping rule; see the
// comment where they are used, in series() below.
#define STOP_EPSALG_DEPTH 8
#define STOP_WINDOW_FRAC 0.5

static void series(multinomial* mult, multinomial_result* mult_res, options_t* options) {
    const int_t verbose = options->verbose;
    const int_t threads = options->threads;
    const int_t vecs_per_thread = options->vecs_per_thread;
    const int_t N = mult->N;
    const int_t K = mult->K;
    const int_t cells = K*(N+1);

    const int_t step = threads*vecs_per_thread*STRIDE;
    const int_t max_chunks = (options->max_iter+step-1)/step;
    if (options->verbose) {
        char p0_buf[64];
        quadmath_snprintf(p0_buf, sizeof(p0_buf), "%.10Qe", mult->p0);
        printf("p0 =\t%s\n", p0_buf);
    }

    local_t* local_arrs = malloc(threads*(N+1)*sizeof(local_t));
    result_t* vec_arrs = malloc(threads*(N+1)*sizeof(result_t));
    // complex_t* res_arr = malloc(max_chunks*step*sizeof(complex_t));
    qcomplex_t* res_arr = malloc(max_chunks*step*sizeof(qcomplex_t));
    quad_t* sums_arr = malloc(max_chunks*step*sizeof(quad_t));

    // Shared phase tables; each thread additionally carries one complex per cell.
    const int_t stride_n = threads > 1 ? threads : 1;
    local_t* lane = malloc(cells*sizeof(local_t));
    complex_t* step_z = malloc(cells*sizeof(complex_t));
    complex_t* phase_arrs = malloc(threads*cells*sizeof(complex_t));
    phase_tables(mult, stride_n, lane, step_z);

    void* plan = NULL;
    if (options->poisson) {
        plan = build_poisson_plan(mult, verbose);  // never builds the merge-factor plan it replaces
    }
    else if (options->use_fft_precompute) {
        switch (options->fft_precision) {
            case FFT_REAL: plan = build_fourier_fft_plan(mult); break;
            default: break;
        }
    }

    // One pool of threads for the whole run: see fourier_pool_t above for why per-chunk
    // pthread_create/pthread_join (the old eval_fourier_parallel) is wasted work here.
    fourier_pool_t* pool = fourier_pool_create(mult, options, plan, res_arr, local_arrs, vec_arrs,
                                               phase_arrs, lane, step_z);

    const int_t B = options->B > 0 ? options->B : CONV_COUNT;
    const quad_t threshold = options->eps_rel / (quad_t)B;
    quad_t delta_acc = 0;
    int_t terms_acc = 0;

    quad_t f = 0;
    quad_t p = 0;
    quad_t c = 0;

    int converged_full = 0;
    int_t converged_count = 0;
    quad_t p_prev = INFINITY;

    // Convergence acceleration (see extrapolate.h). The partial sums are fed in scaled by a power
    // of two so the extrapolation runs in double regardless of how extreme p is.
    epsalg_t eps;
    epsalg_init(&eps);
    int_t eps_shift = 0;
    int_t eps_have = 0;

    // extrapolate >= 2's stopping decision: a second, shallow-table extrapolation (depth
    // STOP_EPSALG_DEPTH, chosen for stability rather than for ultimate ringing cancellation), its
    // output median-smoothed over 5 consecutive values, and a check that the smoothed value has
    // been flat over a trailing window rather than that DQELG's own noisy abserr happened to read
    // small CONV_COUNT times in a row. See extrapolate.h's header comment and
    // latex/poisson.tex for why the old rule could move by 50% or fail to converge at all under a
    // single-ulp perturbation of the terms, and why this one does not.
    epsalg_t eps_stop;
    epsalg_init(&eps_stop);
    double raw_hist[5] = {0, 0, 0, 0, 0};
    int_t raw_count = 0;
    double* smooth_arr = NULL;
    int_t* dqmax = NULL;
    int_t* dqmin = NULL;
    int_t hmax=0, tmax=0, hmin=0, tmin=0;
    int_t smooth_start = -1;
    double last_smooth = NAN;       // most recent smoothed statistic, whether or not it was stable
    int_t last_smooth_iter = -1;
    double last_rel_spread = NAN;   // its trailing-window relative spread, at the same iter
    if (options->extrapolate >= 2) {
        smooth_arr = malloc(max_chunks*step*sizeof(double));
        dqmax = malloc(max_chunks*step*sizeof(int_t));
        dqmin = malloc(max_chunks*step*sizeof(int_t));
    }

    err_diag_t diag;
    err_diag_init(&diag, max_chunks*step);

    if (verbose) printf("terms\tcumulative sum\t\tmean change\n");
    int_t iter = 0;
    for (int_t n=0; n < max_chunks; n++) {
        if (options->max_time > 0.0 && isfinite(options->max_time)) {
            struct timespec t_now;
            timespec_get(&t_now, TIME_UTC);
            if (get_dt(options->t_start, t_now) >= options->max_time) {
                mult_res->status = 1; // Maximum time reached
                if (n > 0) iter--;
                goto finish;
            }
        }
        if (!finiteq(p) || isnanq(p) || fabsq(p) > 1e100) {
            p = NAN;
            mult_res->status = 4; // Magnitude explosion / numerical instability
            converged_full = 0;
            goto finish;
        }

        const int_t n_min = n*threads*vecs_per_thread;
        const int_t n_max = (n+1)*threads*vecs_per_thread;
        fourier_pool_run(pool, n_min, n_max, n > 0);

        // must run on the raw terms, before res_arr[0] is halved below
        if (options->error_bound) err_diag_accum(&diag, res_arr, n_min*STRIDE, n_max*STRIDE, mult->gamma, mult->T);

        if (n==0) {
            if (res_arr[0] == 0.0) {  // sum of underflows
                if (verbose) {
                    char p_buf[64];
                    quadmath_snprintf(p_buf, sizeof(p_buf), "%.10Qe", p);
                    printf("%04lld\t%s\n", iter, p_buf);
                }
                p = 0;
                converged_full = 1;
                goto finish;
            }
            res_arr[0] /= 2;
        }

        for (iter=n_min*STRIDE; (iter<n_max*STRIDE) && (iter<options->max_iter); iter++) {
            const quad_t delta_re = crealq(res_arr[iter]);
            const quad_t y = delta_re - c;
            const quad_t t = f + y;
            c = (t - f) - y;
            f = t;
            p = f;
            sums_arr[iter] = p;

            // Set by the window-stability check below, and read after the term-ratio update --
            // must NOT be folded into converged_count itself, because that counter is
            // unconditionally overwritten by the term-ratio rule a few lines down every iteration
            // (converged_count = converged ? converged_count+1 : 0), which would silently discard
            // a same-iteration forced stop before it was ever read.
            int window_stop = 0;

            if (options->extrapolate) {
                if (!eps_have && p != 0) {
                    int e2;
                    frexpq(p, &e2);
                    eps_shift = e2;
                    eps_have = 1;
                }
                if (eps_have) {
                    const double ps = (double)scalbnq(p, -(int)eps_shift);
                    if (isfinite(ps)) {
                        epsalg_add(&eps, ps, EPSALG_LIMEXP);  // never poison the table

                        if (options->extrapolate >= 2) {
                            epsalg_add(&eps_stop, ps, STOP_EPSALG_DEPTH);
                            if (epsalg_ready(&eps_stop)) {
                                for (int_t q=0; q<4; q++) raw_hist[q] = raw_hist[q+1];
                                raw_hist[4] = eps_stop.result;
                                raw_count++;

                                if (raw_count >= 5) {
                                    const double med = epsalg_median5(raw_hist);
                                    smooth_arr[iter] = med;
                                    if (smooth_start < 0) smooth_start = iter;
                                    last_smooth = med;
                                    last_smooth_iter = iter;

                                    // trailing-window spread of the smoothed sequence, via two
                                    // monotonic deques of indices into smooth_arr
                                    while (tmax > hmax && smooth_arr[dqmax[tmax-1]] <= med) tmax--;
                                    dqmax[tmax++] = iter;
                                    while (tmin > hmin && smooth_arr[dqmin[tmin-1]] >= med) tmin--;
                                    dqmin[tmin++] = iter;

                                    int_t w = (int_t)(STOP_WINDOW_FRAC*(double)iter);
                                    if (w < CONV_COUNT) w = CONV_COUNT;
                                    const int_t lo = iter - w;
                                    while (hmax < tmax && dqmax[hmax] < lo) hmax++;
                                    while (hmin < tmin && dqmin[hmin] < lo) hmin++;

                                    if (lo >= smooth_start) {
                                        const double spread = fmax(smooth_arr[dqmax[hmax]] - med, med - smooth_arr[dqmin[hmin]]);
                                        last_rel_spread = spread/fabs(med);
                                        if (spread <= (double)options->eps_rel*fabs(med)) window_stop = 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (verbose) {
                delta_acc += delta_re;
                terms_acc++;
                if (iter % options->print_freq == 0) {
                    char p_buf[64];
                    quadmath_snprintf(p_buf, sizeof(p_buf), "%.10Qe", p);
                    if (iter == 0) {
                        printf("%04lld\t%s\n", iter, p_buf);
                    }
                    else {
                        char delta_buf[64];
                        quadmath_snprintf(delta_buf, sizeof(delta_buf), "%.3Qe", delta_acc/terms_acc);
                        printf("%04lld\t%s\t%s%s\n", iter, p_buf, delta_acc>=0 ? " " : "", delta_buf);
                    }
                    delta_acc = 0;
                    terms_acc = 0;
                }
            }

            const quad_t frac = fabsq(delta_re/p_prev);
            const int converged = (p > 0.0) & (frac < threshold);
            converged_count = converged ? converged_count + 1 : 0;
            p_prev = p;

            // OR with the window-stability check (set above, in the epsalg_add block): whichever
            // of the two rules is satisfied first ends the series. window_stop is a one-shot flag
            // rather than a counter -- the window it is computed over already spans many terms of
            // stability, so no further consecutive-pass requirement is layered on top of it.
            if (converged_count >= B || window_stop) {
                if (verbose) {
                    char p_buf[64], delta_buf[64];
                    quadmath_snprintf(p_buf, sizeof(p_buf), "%.10Qe", p);
                    quadmath_snprintf(delta_buf, sizeof(delta_buf), "%.3Qe", delta_acc/terms_acc);
                    printf("*%04lld\t%s\t%s%s\n", iter, p_buf, delta_acc>=0 ? " " : "", delta_buf);
                }
                converged_full = 1;
                goto finish;
            }
        }
    }
    iter--;

    finish:

    // Both guarded by eps_have: the early exits above (underflow to zero, non-finite partial sum)
    // jump here without ever writing sums_arr or feeding the extrapolator, and must keep their p.
    if (eps_have && options->average_window > 0) {
        quad_t p_average = 0;
        double weight = 0.0;
        const int_t window = (int_t)(options->average_window*iter);
        for (int_t i=0; i<=window; i++) {
            const int_t j = iter-window+i;
            const double w = options->average_flat? 1.0:(double)i+1;
            p_average += sums_arr[j]*w;
            weight += w;
        }
        p_average /= weight;
        p = p_average;
    }
    else if (eps_have && options->extrapolate >= 2 && last_smooth_iter >= 0) {
        // Prefer the same robust (shallow-table, median-smoothed) statistic that justified
        // stopping over the deep depth-50 table below: that table can itself land on a bad
        // extrapolate at an unlucky iteration (see extrapolate.h and latex/poisson.tex), and
        // reporting from it would silently reintroduce the fragility just removed from the
        // stopping decision -- a stable stopping point paired with an unstable reported value
        // is not actually fixed.
        p = scalbnq((quad_t)last_smooth, (int)eps_shift);
    }
    else if (eps_have && options->extrapolate && epsalg_ready(&eps)) {
        p = scalbnq((quad_t)eps.result, (int)eps_shift);
    }

    p += mult->p0/2;
    if (verbose) {
        char p_buf[64]; quadmath_snprintf(p_buf, sizeof(p_buf), "%.10Qe", p);
        printf("p = %s\n", p_buf);
        printf("converged: %s\n", converged_full ? "true" : "false");
    }

    mult_res->pval = p;
    mult_res->converged = converged_full;
    mult_res->terms = iter+1;
    if (eps_have && options->extrapolate >= 2 && last_smooth_iter >= 0 && isfinite(last_rel_spread)) {
        mult_res->err_rel_ext = last_rel_spread;
    } else {
        mult_res->err_rel_ext = (eps_have && epsalg_ready(&eps) && eps.result != 0)
                              ? eps.abserr/fabs(eps.result) : NAN;
    }

    mult_res->err_rel_est = NAN;
    mult_res->n_eff = NAN;
    mult_res->nu_max_ratio = NAN;
    mult_res->nu_at_max = -1;
    if (options->error_bound) {
        int_t n_at = -1;
        mult_res->err_rel_est = err_rel_estimate(&diag, p, mult->gamma, mult->T);
        mult_res->n_eff = err_n_eff(&diag);
        mult_res->nu_max_ratio = err_max_revival(&diag, &n_at);
        mult_res->nu_at_max = n_at;
        if (verbose) {
            printf("N_eff = %.4e, late max |nuhat(n)|/nuhat(0) = %.4e at n=%lld\n",
                   mult_res->n_eff, mult_res->nu_max_ratio, mult_res->nu_at_max);
            printf("estimated relative truncation error = %.4e\n", mult_res->err_rel_est);
            if (options->extrapolate >= 1) printf("estimated relative extrapolation error = %.4e\n", mult_res->err_rel_ext);
        }
    }
    err_diag_free(&diag);
    fourier_pool_free(pool);

    if (plan && options->poisson) {
        free_poisson_plan(plan);
    }
    else if (plan) {
        switch (options->fft_precision) {
            case FFT_REAL: free_fourier_fft_plan(plan); break;
            default: break;
        }
    }

    free(local_arrs);
    free(vec_arrs);
    free(res_arr);
    free(sums_arr);
    free(lane);
    free(step_z);
    free(phase_arrs);
    free(smooth_arr);
    free(dqmax);
    free(dqmin);
}
