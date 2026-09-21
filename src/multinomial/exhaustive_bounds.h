#pragma once
#include <stdatomic.h>

// Pruning variant of the exhaustive enumeration method (compare with exhaustive_bisection.h).
//
// The decision tree for computing the p-value has one node per (category k, count of trials
// N_remain still to be distributed among categories k..K-1); a path from the root to a leaf is a
// full composition x_0,...,x_{K-1} with sum N, and the p-value is the total probability of the
// leaves whose cumulative reward S(x) - S(x0) <= 0.
//
// Instead of bisecting the reward's sign change in the second-to-last category only (the current
// method, exhaustive_bisection.h), this variant precomputes, for every node, a lower and an upper
// bound on the total reward still attainable from categories k..K-1 given N_remain trials left --
// bound_min[k][N_remain] and bound_max[k][N_remain] -- via the same min-plus/max-plus dynamic
// program the old (pre-interval.h) fill_interval used to find the statistic's global range, except
// every intermediate (k, i) pair is kept here instead of only the final row. Both bounds are exact
// (achieved by some composition), so at any node, given the cumulative reward r_acc so far:
//
//   * if r_acc + bound_min[k][N_remain] > 0, no completion of this path can bring the reward back
//     to <= 0: the whole subtree contributes 0, and is never visited.
//   * if r_acc + bound_max[k][N_remain] <= 0, every completion already satisfies the criterion: the
//     whole subtree's probability mass is added in closed form (as if categories k..K-1 were a
//     single virtual category with probability equal to their probability sum -- the multinomial
//     theorem gives the exact total mass of that cylinder set without visiting any of it) and the
//     subtree is never visited either.
//   * otherwise the node is genuinely ambiguous and is explored one level deeper.
//
// The last category is never bisected (or even branched on) because there both bounds are exact
// and identical (bound_min[K-1] == bound_max[K-1] == the category's own reward row, no combination
// needed), so the first two checks above are already exhaustive and always resolve it -- the
// recursion is only ever entered with k <= K-2, and this file has no bisection/peak-finding code at
// all.
//
// Building the bound table costs O(K*N^2), the same as the pre-interval.h fill_interval; it is a
// one-off up front, and is trivial next to the sizes for which exhaustive enumeration is even
// considered (see options->enum_cutoff -- the regime this method is used in is exactly where plain
// enumeration is tractable). What it buys is that the *enumeration* itself only explores the part
// of the tree whose bound genuinely straddles S0, which is a thin sliver of the tree exactly when
// the true p-value is very close to 0 or to 1 (the rest of the mass is unambiguously all-reject or
// all-accept) -- the extreme-p-value regime this method targets -- and it never adds together the
// many tiny leaf probabilities that a very small p-value is otherwise built from one at a time,
// which is also friendlier to floating-point accuracy than summing a long tail of near-zero terms.

#ifndef BOUNDS_CHUNK
#define BOUNDS_CHUNK 4
#endif

#ifndef BOUNDS_ROW_CHUNK
#define BOUNDS_ROW_CHUNK 8
#endif

// ---- Bound table construction ----
//
// This inner loop (a min-plus/max-plus reduction, no data-dependent branching) would be safe to
// vectorize with no risk of changing the result -- min/max are exactly associative and commutative
// in IEEE floating point, so the reduction is bit-for-bit independent of how it's grouped, unlike
// the traversal below. It was tried (two variants: a per-lane gather of bound_min/bound_max, since
// that row is indexed by i-n which runs backwards as n increases so it can't be loaded directly
// like the reward row can; and the same gather routed through a small stack buffer instead of
// direct lane inserts) and measured 2-4x *slower* than the plain scalar loop below on this machine
// (Apple Silicon / arm64, where USE_SIMD compiles through simde's software emulation of AVX-512 --
// there is no native 512-bit vector unit here, so each simde min/max/load decomposes into several
// narrower emulated ops plus glue, which loses to a loop simple enough for GCC to auto-vectorize to
// native NEON on its own). Left scalar; revisit only with a measurement on real AVX-512 hardware.

typedef struct {
    int_t N;
    const real_t* restrict rj;         // reward row j, size N+1
    const real_t* restrict bmin_next;  // bound_min row j+1, size N+1
    const real_t* restrict bmax_next;  // bound_max row j+1, size N+1
    real_t* restrict bmin_j;           // bound_min row j (output), size N+1
    real_t* restrict bmax_j;           // bound_max row j (output), size N+1
    atomic_int_fast64_t* restrict next_i;
} bounds_row_args_t;

static void* bounds_row_thread(void* args_void) {
    bounds_row_args_t* a = args_void;
    const int_t N = a->N;

    int_t i_start;
    while ((i_start = atomic_fetch_add_explicit(a->next_i, BOUNDS_ROW_CHUNK, memory_order_relaxed)) <= N) {
        const int_t i_end = i_start + BOUNDS_ROW_CHUNK - 1 > N ? N : i_start + BOUNDS_ROW_CHUNK - 1;
        for (int_t i = i_start; i <= i_end; i++) {
            real_t new_min = INFINITY;
            real_t new_max = -INFINITY;
            for (int_t n = 0; n <= i; n++) {
                const real_t rew = a->rj[n];
                const real_t tmin = rew + a->bmin_next[i-n];
                const real_t tmax = rew + a->bmax_next[i-n];
                if (tmin < new_min) new_min = tmin;
                if (tmax > new_max) new_max = tmax;
            }
            a->bmin_j[i] = new_min;
            a->bmax_j[i] = new_max;
        }
    }
    return NULL;
}

// bound_min/bound_max must have room for K*(N+1) entries; only rows 1..K-1 are written (row 0 is
// never looked up by the traversal below, since the root splits on category 0 directly).
static void build_bounds(
        const int_t K, const int_t N,
        const real_t* restrict reward,
        real_t* restrict bound_min, real_t* restrict bound_max,
        const int_t threads
) {
    const int_t stride = N+1;

    for (int_t i = 0; i <= N; i++) {
        bound_min[(K-1)*stride+i] = reward[(K-1)*stride+i];
        bound_max[(K-1)*stride+i] = reward[(K-1)*stride+i];
    }

    const int_t n_threads = threads < 1 ? 1 : threads;

    for (int_t j = K-2; j >= 1; j--) {
        atomic_int_fast64_t next_i;
        atomic_init(&next_i, 0);

        bounds_row_args_t args_base = {
            .N = N,
            .rj = &reward[j*stride],
            .bmin_next = &bound_min[(j+1)*stride],
            .bmax_next = &bound_max[(j+1)*stride],
            .bmin_j = &bound_min[j*stride],
            .bmax_j = &bound_max[j*stride],
            .next_i = &next_i,
        };

        if (n_threads <= 1) {
            bounds_row_thread(&args_base);
        } else {
            pthread_t* tid = malloc(n_threads*sizeof(pthread_t));
            bounds_row_args_t* args = malloc(n_threads*sizeof(bounds_row_args_t));
            for (int_t t = 0; t < n_threads; t++) args[t] = args_base;
            for (int_t t = 0; t < n_threads; t++) pthread_create(&tid[t], NULL, bounds_row_thread, &args[t]);
            for (int_t t = 0; t < n_threads; t++) pthread_join(tid[t], NULL);
            free(tid);
            free(args);
        }
    }
}

// ---- Category ordering ----
//
// Pruning is order-sensitive: sending the highest-probability categories through first tends to
// pin down the cumulative reward's sign early, which is what lets the bound checks resolve whole
// subtrees without descending. Measured across several instances, descending-by-probability order
// was consistently the best (or tied-best) of the orderings tried. The caller is not guaranteed to
// have sorted categories that way already (main_benchmark.h's sort_instance_desc, for instance, is
// only applied by that one call site), so this function sorts internally -- the relabelling is
// just a permutation of which category is "0", "1", etc., and the p-value (a sum over all
// compositions) is invariant to it, so this is free to do regardless of the caller.

// Indices 0..K-1 with probs[perm[0]] >= probs[perm[1]] >= ... >= probs[perm[K-1]]. Insertion sort
// (K is small in the regime this method is used -- see options->enum_cutoff), matching the style
// of sort_instance_desc in main_benchmark.h.
static void bounds_sort_perm_desc(const int_t K, const real_t* restrict probs, int_t* restrict perm) {
    for (int_t k = 0; k < K; k++) perm[k] = k;
    for (int_t i = 1; i < K; i++) {
        const int_t p = perm[i];
        const real_t v = probs[p];
        int_t j = i-1;
        while (j >= 0 && probs[perm[j]] < v) { perm[j+1] = perm[j]; j--; }
        perm[j+1] = p;
    }
}

// ---- Traversal ----

typedef struct {
    int_t N, K;
    const real_t* restrict reward;      // K*(N+1)
    const real_t* restrict log_prob;    // K*(N+1)
    const real_t* restrict bound_min;   // K*(N+1), rows 1..K-1 valid
    const real_t* restrict bound_max;   // K*(N+1), rows 1..K-1 valid
    const real_t* restrict log_psuffix; // size K; log_psuffix[j] = log(sum_{m=j}^{K-1} probs[m]), j=1..K-1
} bounds_ctx_t;

// Closed-form log-probability of the whole cylinder set {categories j..K-1 sum to i}, treating
// them as one virtual category with probability equal to their sum (multinomial theorem).
static inline real_t bounds_suffix_logp(const bounds_ctx_t* restrict ctx, const int_t j, const int_t i) {
    return -lgac[i] + (real_t)i*ctx->log_psuffix[j];
}

static real_t exhaustive_bounds_recurse(
        const bounds_ctx_t* restrict ctx,
        const int_t k, const int_t N_remain,
        const real_t r_acc, const real_t logp_acc
) {
    const int_t stride = ctx->N+1;
    const int_t k1 = k+1;
    const real_t* restrict reward_k = &ctx->reward[k*stride];
    const real_t* restrict logp_k = &ctx->log_prob[k*stride];
    const real_t* restrict bmin_k1 = &ctx->bound_min[k1*stride];
    const real_t* restrict bmax_k1 = &ctx->bound_max[k1*stride];

    real_t p = 0;
    for (int_t n = 0; n <= N_remain; n++) {
        const int_t N_next = N_remain - n;
        const real_t r = r_acc + reward_k[n];

        if (r + bmin_k1[N_next] > 0) continue;  // whole subtree rejected

        const real_t logp = logp_acc + logp_k[n];

        if (r + bmax_k1[N_next] <= 0) {  // whole subtree accepted (closed form)
            p += EXP(MAX_EXP + logp + bounds_suffix_logp(ctx, k1, N_next));
            continue;
        }

        // Ambiguous: recurse one level deeper. k1 == K-1 can never reach here, since
        // bound_min[K-1] == bound_max[K-1] (both equal that category's own reward row), so one of
        // the two checks above always fires when k1 == K-1.
        p += exhaustive_bounds_recurse(ctx, k1, N_next, r, logp);
    }
    return p;
}

typedef struct {
    bounds_ctx_t ctx;
    real_t logp_acc0;  // lgac[N]
    atomic_int_fast64_t* restrict next_n0;
    real_t p_acc;
    const options_t* options;
    atomic_int* timed_out;
} bounds_thread_args_t;

static void* exhaustive_bounds_thread(void* args_void) {
    bounds_thread_args_t* args = args_void;
    const bounds_ctx_t* restrict ctx = &args->ctx;
    const int_t N = ctx->N;
    const int_t K = ctx->K;
    const int_t stride = N+1;
    real_t p_acc = 0;

    int_t n0_start;
    while ((n0_start = atomic_fetch_add_explicit(args->next_n0, BOUNDS_CHUNK, memory_order_relaxed)) <= N) {
        if (args->timed_out && atomic_load(args->timed_out)) break;
        if (args->options && args->options->max_time > 0.0 && isfinite(args->options->max_time)) {
            struct timespec t_now;
            timespec_get(&t_now, TIME_UTC);
            if (get_dt(args->options->t_start, t_now) >= args->options->max_time) {
                if (args->timed_out) atomic_store(args->timed_out, 1);
                break;
            }
        }
        const int_t n0_end = n0_start + BOUNDS_CHUNK - 1 > N ? N : n0_start + BOUNDS_CHUNK - 1;
        for (int_t n0 = n0_start; n0 <= n0_end; n0++) {
            const int_t N_next = N - n0;
            const real_t r = ctx->reward[n0];

            if (K == 1) {
                if (r <= 0) p_acc += EXP(MAX_EXP + args->logp_acc0 + ctx->log_prob[n0]);
                continue;
            }

            if (r + ctx->bound_min[stride+N_next] > 0) continue;  // row 1

            const real_t logp = args->logp_acc0 + ctx->log_prob[n0];

            if (r + ctx->bound_max[stride+N_next] <= 0) {
                p_acc += EXP(MAX_EXP + logp + bounds_suffix_logp(ctx, 1, N_next));
                continue;
            }

            p_acc += exhaustive_bounds_recurse(ctx, 1, N_next, r, logp);
        }
    }

    args->p_acc = p_acc;
    return NULL;
}

static void exhaustive_bounds_parallel(
        const bounds_ctx_t* restrict ctx, const real_t logp_acc0, real_t* restrict p_acc, const int_t threads,
        const options_t* options, atomic_int* timed_out
) {
    const int_t N = ctx->N;

    if (threads <= 1 || N == 0) {
        if (options && options->max_time > 0.0 && isfinite(options->max_time)) {
            struct timespec t_now;
            timespec_get(&t_now, TIME_UTC);
            if (get_dt(options->t_start, t_now) >= options->max_time) {
                if (timed_out) atomic_store(timed_out, 1);
                return;
            }
        }
        atomic_int_fast64_t next_n0;
        atomic_init(&next_n0, 0);
        bounds_thread_args_t args = { .ctx = *ctx, .logp_acc0 = logp_acc0, .next_n0 = &next_n0, .p_acc = 0, .options = options, .timed_out = timed_out };
        exhaustive_bounds_thread(&args);
        *p_acc = args.p_acc;
        return;
    }

    const int_t n_threads = threads > N+1 ? N+1 : threads;

    atomic_int_fast64_t next_n0;
    atomic_init(&next_n0, 0);

    pthread_t* tid = malloc(n_threads*sizeof(pthread_t));
    bounds_thread_args_t* args = malloc(n_threads*sizeof(bounds_thread_args_t));

    for (int_t t = 0; t < n_threads; t++) {
        args[t].ctx = *ctx;
        args[t].logp_acc0 = logp_acc0;
        args[t].next_n0 = &next_n0;
        args[t].p_acc = 0;
        args[t].options = options;
        args[t].timed_out = timed_out;
        pthread_create(&tid[t], NULL, exhaustive_bounds_thread, &args[t]);
    }

    real_t total = 0;
    for (int_t t = 0; t < n_threads; t++) {
        pthread_join(tid[t], NULL);
        total += args[t].p_acc;
    }
    *p_acc = total;

    free(tid);
    free(args);
}

static void exhaustive_bounds(multinomial* mult, multinomial_result* mult_res, options_t* options) {
    if (options->max_time > 0.0 && isfinite(options->max_time)) {
        struct timespec t_now;
        timespec_get(&t_now, TIME_UTC);
        if (get_dt(options->t_start, t_now) >= options->max_time) {
            mult_res->status = 1;
            mult_res->converged = 0;
            return;
        }
    }
    const int_t N = mult->N;
    const int_t K = mult->K;
    const int_t stride = N+1;
    const int_t matrix_size = K*stride;

    int_t* restrict perm = malloc(K*sizeof(int_t));
    bounds_sort_perm_desc(K, mult->probs, perm);

    // Categories are relabelled pk = 0..K-1 -> original category perm[pk], descending by
    // probability; every array below is built and indexed in this internal order.
    real_t* restrict reward = malloc(matrix_size*sizeof(real_t));
    real_t* restrict log_prob = malloc(matrix_size*sizeof(real_t));
    for (int_t pk = 0; pk < K; pk++) {
        const int_t src_index = perm[pk]*stride;
        const int_t dst_index = pk*stride;
        for (int_t i = 0; i <= N; i++) {
            reward[dst_index+i] = mult->matrix[src_index+i].reward;
            log_prob[dst_index+i] = mult->matrix[src_index+i].log_prob;
        }
    }

    real_t* restrict bound_min = malloc(matrix_size*sizeof(real_t));
    real_t* restrict bound_max = malloc(matrix_size*sizeof(real_t));
    if (K >= 2) build_bounds(K, N, reward, bound_min, bound_max, options->threads);

    real_t* restrict log_psuffix = malloc(K*sizeof(real_t));
    if (K >= 2) {
        real_t suffix = mult->probs[perm[K-1]];
        log_psuffix[K-1] = LOG(suffix);
        for (int_t j = K-2; j >= 1; j--) {
            suffix += mult->probs[perm[j]];
            log_psuffix[j] = LOG(suffix);
        }
    }

    bounds_ctx_t ctx = {
        .K = K, .N = N,
        .reward = reward, .log_prob = log_prob,
        .bound_min = bound_min, .bound_max = bound_max,
        .log_psuffix = log_psuffix,
    };

    real_t p_acc = 0;
    const real_t logp_acc0 = lgac[N];
    atomic_int timed_out;
    atomic_init(&timed_out, 0);
    exhaustive_bounds_parallel(&ctx, logp_acc0, &p_acc, options->threads > 1 ? options->threads : 1, options, &timed_out);

    if (atomic_load(&timed_out)) {
        mult_res->status = 1;
        mult_res->converged = 0;
    } else {
        p_acc = p_acc*EXP(-MAX_EXP);
        if (options->verbose) printf("exhaustive (bounds) = %.8e\n", p_acc);
        mult_res->pval = p_acc;
        mult_res->converged = 1;
        mult_res->terms = -1;
        mult_res->status = 0;
    }

    free(reward);
    free(log_prob);
    free(bound_min);
    free(bound_max);
    free(log_psuffix);
    free(perm);
}
