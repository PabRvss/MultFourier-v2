#pragma once
#include <stdatomic.h>

#ifndef BISECT_MAX_DEPTH
#define BISECT_MAX_DEPTH 64
#endif

#ifndef BISECT_CHUNK
#define BISECT_CHUNK 4
#endif

#ifndef BISECT_MIN_RANGE
#define BISECT_MIN_RANGE 64
#endif

#ifdef USE_SIMD
#ifdef USE_DOUBLE_PRECISION
#define VEC_LOADU(ptr) _mm512_loadu_pd(ptr)
#else
#define VEC_LOADU(ptr) _mm512_loadu_ps(ptr)
#endif
#endif

__attribute__((optimize("no-fast-math"), noinline))
static void bisect_eval(
        const real_t* restrict reward, const real_t* restrict log_prob,
        const int_t index_k, const int_t index_last, const int_t N_remain_d,
        const real_t r_acc_d, const real_t logp_acc_d, const int_t n,
        real_t* restrict r_out, real_t* restrict logp_out
) {
    const int_t N_next = N_remain_d - n;
    *r_out = r_acc_d + reward[index_k + n] + reward[index_last + N_next];
    *logp_out = logp_acc_d + log_prob[index_k + n] + log_prob[index_last + N_next];
}

static int_t bisect_find_peak(
        const real_t* restrict reward, const real_t* restrict log_prob,
        const int_t index_k, const int_t index_last, const int_t N_remain_d,
        const real_t r_acc_d, const real_t logp_acc_d, int_t lo, int_t hi
) {
    while (lo < hi) {
        const int_t mid = lo + (hi - lo)/2;
        real_t r_mid, r_mid1, dummy;
        bisect_eval(reward, log_prob, index_k, index_last, N_remain_d, r_acc_d, logp_acc_d, mid, &r_mid, &dummy);
        bisect_eval(reward, log_prob, index_k, index_last, N_remain_d, r_acc_d, logp_acc_d, mid+1, &r_mid1, &dummy);
        if (r_mid < r_mid1) lo = mid+1;  // peak strictly to the right
        else hi = mid;                    // peak at or left of mid
    }
    return lo;
}

static int_t bisect_find_left_boundary(
        const real_t* restrict reward, const real_t* restrict log_prob,
        const int_t index_k, const int_t index_last, const int_t N_remain_d,
        const real_t r_acc_d, const real_t logp_acc_d, const int_t lo, const int_t hi
) {
    real_t r_lo, dummy;
    bisect_eval(reward, log_prob, index_k, index_last, N_remain_d, r_acc_d, logp_acc_d, lo, &r_lo, &dummy);
    if (r_lo > 0) return lo - 1;

    int_t left = lo, right = hi;
    while (left < right) {
        const int_t mid = left + (right - left + 1)/2;
        real_t r_mid;
        bisect_eval(reward, log_prob, index_k, index_last, N_remain_d, r_acc_d, logp_acc_d, mid, &r_mid, &dummy);
        if (r_mid <= 0) left = mid; else right = mid - 1;
    }
    return left;
}

static int_t bisect_find_right_boundary(
        const real_t* restrict reward, const real_t* restrict log_prob,
        const int_t index_k, const int_t index_last, const int_t N_remain_d,
        const real_t r_acc_d, const real_t logp_acc_d, const int_t lo, const int_t hi
) {
    real_t r_hi, dummy;
    bisect_eval(reward, log_prob, index_k, index_last, N_remain_d, r_acc_d, logp_acc_d, hi, &r_hi, &dummy);
    if (r_hi > 0) return hi + 1;

    int_t left = lo, right = hi;
    while (left < right) {
        const int_t mid = left + (right - left)/2;
        real_t r_mid;
        bisect_eval(reward, log_prob, index_k, index_last, N_remain_d, r_acc_d, logp_acc_d, mid, &r_mid, &dummy);
        if (r_mid <= 0) right = mid; else left = mid + 1;
    }
    return left;
}

#ifndef BISECT_BOUNDARY_MARGIN
#define BISECT_BOUNDARY_MARGIN 1
#endif

static int_t bisect_refine_left(
        const real_t* restrict reward, const real_t* restrict log_prob,
        const int_t index_k, const int_t index_last, const int_t N_remain_d,
        const real_t r_acc_d, const real_t logp_acc_d, int_t n_left, const int_t peak
) {
    const int_t extend_limit = (peak < n_left + BISECT_BOUNDARY_MARGIN) ? peak : n_left + BISECT_BOUNDARY_MARGIN;
    while (n_left < extend_limit) {
        real_t r, dummy;
        bisect_eval(reward, log_prob, index_k, index_last, N_remain_d, r_acc_d, logp_acc_d, n_left + 1, &r, &dummy);
        if (r > 0) break;
        n_left++;
    }
    const int_t retract_limit = n_left - BISECT_BOUNDARY_MARGIN > -1 ? n_left - BISECT_BOUNDARY_MARGIN : -1;
    while (n_left > retract_limit) {
        real_t r, dummy;
        bisect_eval(reward, log_prob, index_k, index_last, N_remain_d, r_acc_d, logp_acc_d, n_left, &r, &dummy);
        if (r <= 0) break;
        n_left--;
    }
    return n_left;
}

static int_t bisect_refine_right(
        const real_t* restrict reward, const real_t* restrict log_prob,
        const int_t index_k, const int_t index_last, const int_t N_remain_d,
        const real_t r_acc_d, const real_t logp_acc_d, int_t n_right, const int_t peak
) {
    const int_t extend_limit = (peak > n_right - BISECT_BOUNDARY_MARGIN) ? peak : n_right - BISECT_BOUNDARY_MARGIN;
    while (n_right > extend_limit) {
        real_t r, dummy;
        bisect_eval(reward, log_prob, index_k, index_last, N_remain_d, r_acc_d, logp_acc_d, n_right - 1, &r, &dummy);
        if (r > 0) break;
        n_right--;
    }
    const int_t retract_limit = n_right + BISECT_BOUNDARY_MARGIN < N_remain_d + 1 ? n_right + BISECT_BOUNDARY_MARGIN : N_remain_d + 1;
    while (n_right < retract_limit) {
        real_t r, dummy;
        bisect_eval(reward, log_prob, index_k, index_last, N_remain_d, r_acc_d, logp_acc_d, n_right, &r, &dummy);
        if (r <= 0) break;
        n_right++;
    }
    return n_right;
}

static real_t bisect_sum_range(
        const real_t* restrict log_prob,
        const int_t index_k, const int_t index_last, const int_t N_remain_d,
        const real_t logp_acc_d, const int_t lo, const int_t hi
) {
    if (lo > hi) return 0;
    real_t sum = 0;
    int_t n = lo;

#ifdef USE_SIMD
    const vec_t logp_acc_vec = logp_acc_d*VEC_ONE;

    for (; n + STRIDE - 1 <= hi; n += STRIDE) {
        const vec_t logp_vec = VEC_LOADU(&log_prob[index_k + n]);

        vec_t logp_last_vec;
        for (int s = 0; s < STRIDE; s++) {
            const int_t N_next = N_remain_d - (n + s);
            logp_last_vec[s] = log_prob[index_last + N_next];
        }

        const vec_t logp_final_vec = logp_acc_vec + logp_vec + logp_last_vec;
        for (int s = 0; s < STRIDE; s++) sum += EXP(MAX_EXP + logp_final_vec[s]);
    }
#endif
    for (; n <= hi; n++) {
        const int_t N_next = N_remain_d - n;
        const real_t logp_final = logp_acc_d + log_prob[index_k + n] + log_prob[index_last + N_next];
        sum += EXP(MAX_EXP + logp_final);
    }
    return sum;
}

static void iterative_bisection_core(
        const int_t N,
        const int_t K,
        const real_t* restrict reward,
        const real_t* restrict log_prob,
        real_t* restrict p_acc,
        const int_t k_start,
        const int_t N_remain_start,
        const real_t r_acc_start,
        const real_t logp_acc_start,
        int_t* restrict N_remain,
        real_t* restrict r_acc,
        real_t* restrict logp_acc,
        int_t* restrict n_cur
) {
    if (k_start == K-1) {
        const int_t idx = (K-1)*(N+1) + N_remain_start;
        const real_t r_final = r_acc_start + reward[idx];
        const real_t logp_final = logp_acc_start + log_prob[idx];
        if (r_final <= 0) *p_acc = *p_acc + EXP(MAX_EXP + logp_final);
        return;
    }

    int_t d = 0;
    N_remain[0] = N_remain_start;
    r_acc[0] = r_acc_start;
    logp_acc[0] = logp_acc_start;
    n_cur[0] = 0;

    while (d >= 0) {
        if (n_cur[d] > N_remain[d]) {
            d--;
            if (d >= 0) n_cur[d]++;
            continue;
        }

        const int_t k = k_start + d;

        if (k == K-2) {  // last category is always the remainder
            const int_t index_k = k*(N+1);
            const int_t index_last = (K-1)*(N+1);
            const real_t r_acc_d = r_acc[d];
            const real_t logp_acc_d = logp_acc[d];
            const int_t N_remain_d = N_remain[d];
            real_t p_leaf;

            if (N_remain_d < BISECT_MIN_RANGE) {
                p_leaf = 0;
                for (int_t n = 0; n <= N_remain_d; n++) {
                    real_t r_final, logp_final;
                    bisect_eval(reward, log_prob, index_k, index_last, N_remain_d, r_acc_d, logp_acc_d, n, &r_final, &logp_final);
                    if (r_final <= 0) p_leaf += EXP(MAX_EXP + logp_final);
                }
            } else {
                const int_t peak = bisect_find_peak(reward, log_prob, index_k, index_last, N_remain_d, r_acc_d, logp_acc_d, 0, N_remain_d);
                real_t r_peak, logp_peak_dummy;
                bisect_eval(reward, log_prob, index_k, index_last, N_remain_d, r_acc_d, logp_acc_d, peak, &r_peak, &logp_peak_dummy);

                if (r_peak <= 0) {
                    p_leaf = bisect_sum_range(log_prob, index_k, index_last, N_remain_d, logp_acc_d, 0, N_remain_d);
                } else {
                    int_t n_left = bisect_find_left_boundary(reward, log_prob, index_k, index_last, N_remain_d, r_acc_d, logp_acc_d, 0, peak);
                    int_t n_right = bisect_find_right_boundary(reward, log_prob, index_k, index_last, N_remain_d, r_acc_d, logp_acc_d, peak, N_remain_d);
                    n_left = bisect_refine_left(reward, log_prob, index_k, index_last, N_remain_d, r_acc_d, logp_acc_d, n_left, peak);
                    n_right = bisect_refine_right(reward, log_prob, index_k, index_last, N_remain_d, r_acc_d, logp_acc_d, n_right, peak);
                    p_leaf = bisect_sum_range(log_prob, index_k, index_last, N_remain_d, logp_acc_d, 0, n_left)
                           + bisect_sum_range(log_prob, index_k, index_last, N_remain_d, logp_acc_d, n_right, N_remain_d);
                }
            }

            *p_acc = *p_acc + p_leaf;

            d--;
            if (d >= 0) n_cur[d]++;
            continue;
        }

        const int_t n = n_cur[d];
        const int_t index_k = k*(N+1);
        const real_t r_next = r_acc[d] + reward[index_k + n];
        const real_t logp_next = logp_acc[d] + log_prob[index_k + n];
        const int_t N_next = N_remain[d] - n;

        d++;
        N_remain[d] = N_next;
        r_acc[d] = r_next;
        logp_acc[d] = logp_next;
        n_cur[d] = 0;
    }
}

typedef struct {
    int_t N, K;
    const real_t* restrict reward;
    const real_t* restrict log_prob;
    real_t logp_acc;
    atomic_int_fast64_t* restrict next_n0;
    real_t p_acc;
} iterative_bisection_args_t;

static void* iterative_bisection_thread(void* args_void) {
    iterative_bisection_args_t* args = args_void;
    const int_t N = args->N;
    const int_t K = args->K;
    real_t p_acc = 0;

    int_t N_remain_stack[BISECT_MAX_DEPTH];
    real_t r_acc_stack[BISECT_MAX_DEPTH];
    real_t logp_acc_stack[BISECT_MAX_DEPTH];
    int_t n_cur_stack[BISECT_MAX_DEPTH];

    int_t* N_remain = N_remain_stack;
    real_t* r_acc = r_acc_stack;
    real_t* logp_acc = logp_acc_stack;
    int_t* n_cur = n_cur_stack;
    int_t* N_remain_heap = NULL;
    real_t* r_acc_heap = NULL;
    real_t* logp_acc_heap = NULL;
    int_t* n_cur_heap = NULL;

    if (K > BISECT_MAX_DEPTH) {
        N_remain_heap = malloc(K*sizeof(int_t));
        r_acc_heap = malloc(K*sizeof(real_t));
        logp_acc_heap = malloc(K*sizeof(real_t));
        n_cur_heap = malloc(K*sizeof(int_t));
        N_remain = N_remain_heap;
        r_acc = r_acc_heap;
        logp_acc = logp_acc_heap;
        n_cur = n_cur_heap;
    }

    int_t n0_start;
    while ((n0_start = atomic_fetch_add_explicit(args->next_n0, BISECT_CHUNK, memory_order_relaxed)) <= N) {
        const int_t n0_end = n0_start + BISECT_CHUNK - 1 > N ? N : n0_start + BISECT_CHUNK - 1;
        for (int_t n0 = n0_start; n0 <= n0_end; n0++) {
            const real_t r0 = args->reward[n0];
            const real_t logp0 = args->logp_acc + args->log_prob[n0];
            iterative_bisection_core(args->N, K, args->reward, args->log_prob, &p_acc, 1, N-n0, r0, logp0,
                                      N_remain, r_acc, logp_acc, n_cur);
        }
    }

    free(N_remain_heap);
    free(r_acc_heap);
    free(logp_acc_heap);
    free(n_cur_heap);

    args->p_acc = p_acc;
    return NULL;
}

static void iterative_bisection_parallel(
        const int_t N,
        const int_t K,
        const real_t* restrict reward,
        const real_t* restrict log_prob,
        real_t logp_acc,
        real_t* restrict p_acc,
        const int_t threads
) {
    if (K == 1 || threads <= 1 || N == 0) {
        int_t N_remain_stack[BISECT_MAX_DEPTH], n_cur_stack[BISECT_MAX_DEPTH];
        real_t r_acc_stack[BISECT_MAX_DEPTH], logp_acc_stack[BISECT_MAX_DEPTH];
        int_t* N_remain = N_remain_stack;
        real_t* r_acc = r_acc_stack;
        real_t* logp_acc_buf = logp_acc_stack;
        int_t* n_cur = n_cur_stack;
        if (K > BISECT_MAX_DEPTH) {
            N_remain = malloc(K*sizeof(int_t));
            r_acc = malloc(K*sizeof(real_t));
            logp_acc_buf = malloc(K*sizeof(real_t));
            n_cur = malloc(K*sizeof(int_t));
        }
        iterative_bisection_core(N, K, reward, log_prob, p_acc, 0, N, 0, logp_acc,
                                  N_remain, r_acc, logp_acc_buf, n_cur);
        if (K > BISECT_MAX_DEPTH) { free(N_remain); free(r_acc); free(logp_acc_buf); free(n_cur); }
        return;
    }

    const int_t n_threads = threads > N+1 ? N+1 : threads;

    atomic_int_fast64_t next_n0;
    atomic_init(&next_n0, 0);

    pthread_t* tid = malloc(n_threads*sizeof(pthread_t));
    iterative_bisection_args_t* args = malloc(n_threads*sizeof(iterative_bisection_args_t));

    for (int_t t=0; t<n_threads; t++) {
        args[t].N = N;
        args[t].K = K;
        args[t].reward = reward;
        args[t].log_prob = log_prob;
        args[t].logp_acc = logp_acc;
        args[t].next_n0 = &next_n0;
        args[t].p_acc = 0;
        pthread_create(&tid[t], NULL, iterative_bisection_thread, &args[t]);
    }

    real_t total = 0;
    for (int_t t=0; t<n_threads; t++) {
        pthread_join(tid[t], NULL);
        total += args[t].p_acc;
    }
    *p_acc = total;

    free(tid);
    free(args);
}

static void exhaustive_bisection(multinomial* mult, multinomial_result* mult_res, options_t* options) {
    const int_t N = mult->N;
    const int_t K = mult->K;
    const int_t matrix_size = K*(N+1);

    real_t* restrict reward = malloc(matrix_size*sizeof(real_t));
    real_t* restrict log_prob = malloc(matrix_size*sizeof(real_t));
    for (int_t idx=0; idx<matrix_size; idx++) {
        reward[idx] = mult->matrix[idx].reward;
        log_prob[idx] = mult->matrix[idx].log_prob;
    }

    real_t p_acc = 0;
    const real_t logp_acc = lgac[N];
    if (options->threads > 1) iterative_bisection_parallel(N, K, reward, log_prob, logp_acc, &p_acc, options->threads);
    else iterative_bisection_parallel(N, K, reward, log_prob, logp_acc, &p_acc, 1);
    p_acc = p_acc*EXP(-MAX_EXP);
    if (options->verbose) printf("exhaustive = %.8e\n", p_acc);
    mult_res->pval = p_acc;
    mult_res->converged = 1;
    mult_res->terms = -1;

    free(reward);
    free(log_prob);
}
