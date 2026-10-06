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
    const int_t* restrict peak_max;     // K*(N+1) or NULL; see "Binary search" below
    const int_t* restrict peak_min;     // K*(N+1) or NULL
    const real_t* restrict tail_lo;     // tail tables, or NULL; see "Tail tables" below
    const real_t* restrict tail_hi;
    const real_t* restrict tail_scale;  // K*(N+1)
    int_t* hint;                        // per thread, 4*K; see "Warm start" below
} bounds_ctx_t;

// Closed-form log-probability of the whole cylinder set {categories j..K-1 sum to i}, treating
// them as one virtual category with probability equal to their sum (multinomial theorem).
static inline real_t bounds_suffix_logp(const bounds_ctx_t* restrict ctx, const int_t j, const int_t i) {
    return -lgac[i] + (real_t)i*ctx->log_psuffix[j];
}

// ---- Binary search (the classification step of Bejerano et al.'s 2004 "Convex" algorithm) ----
//
// At a node (k, N_remain) with accumulated reward r_acc, the two bound functions of the next
// category's count n,
//   g(n) = r_acc + reward_k[n] + bound_max[k+1][N_remain-n]   (subtree accepted iff g(n) <= 0)
//   h(n) = r_acc + reward_k[n] + bound_min[k+1][N_remain-n]   (subtree rejected iff h(n) >  0)
// are concave in n whenever the reward rows are concave in the count: bound_max is then a max-plus
// convolution of concave sequences, hence concave, and bound_min is attained at the vertex that
// sends all trials to a single category, hence a minimum of concave sequences, also concave. That
// holds for the power divergence with lambda > -1 (llr and chi2 included) and for the probability
// mass function (x log p - log x!). Then {g > 0} = [ga, gb] and {h > 0} = [ha, hb] are intervals,
// with [ha, hb] inside [ga, gb] because h <= g, and the values of n split, in increasing order, into
//   accepted [0, ga) | descended [ga, ha) | rejected [ha, hb] | descended (hb, gb] | accepted (gb, N_remain],
// so each node needs four binary searches, O(log N_remain) bound evaluations, instead of
// N_remain+1. The traversal and the summation order are those of the linear scan, so the p-value
// is the same. The maximizers of g and h do not depend on r_acc, only on (k, N_remain), and are
// tabulated once (peak_max, peak_min). exhaustive_bounds checks concavity of the rows first and
// leaves the tables NULL, falling back to the linear scan, when it fails.
//
// Warm start. Consecutive calls at the same level k are, almost always, consecutive siblings: the
// children n and n+1 of the same parent, with N_remain differing by one and r_acc by
// reward_{k-1}[n+1] - reward_{k-1}[n]. Their cut points are close, so each thread keeps the last
// four cut points found at every level (ctx->hint, 4 per level) and the next search at that level
// starts from them: it gallops away from the guess (steps 1, 2, 4, ...) until it brackets the
// boundary, then bisects the bracket. That costs O(log |shift|) evaluations instead of
// O(log N_remain). The guess only decides where the search starts: each side of the peak is
// monotone, so the cut points found are the same as with a cold search, and so is the p-value.
//
// Tail tables. The accepted values of a node (k, N_remain), [0, ga) and (gb, N_remain], are summed
// in closed form. With P_k = sum_{j>=k} pi_j, the mass of value n is
//   exp(logp_acc) * pi_k^n/n! * P_{k+1}^(m-n)/(m-n)!  =  exp(logp_acc) * P_k^m/m! * w_{k,m}(n),   m = N_remain,
// where w_{k,m} is the Binomial(m, pi_k/P_k) probability function. So the accepted mass is the
// node's scale exp(logp_acc + tail_scale[k][m]) times a lower tail plus an upper tail of w_{k,m},
// both read from tables of cumulative sums built once, one from each end (tail_lo[k][m][j] =
// sum_{n<=j} w, tail_hi[k][m][j] = sum_{n>=j} w), so no tail is obtained by subtraction. That is
// two lookups and one exp per node instead of one exp per accepted value, which is where most of
// the time went after the binary search. The summation order changes, so the p-value agrees with
// the linear scan only up to rounding (relative 4e-14 in the tests), not bit for bit. Memory:
// 2 (K-2) (N+1)(N+2)/2 doubles (96 MB for K = 5, N = 2000); above BOUNDS_TAIL_MAX_BYTES the tables
// are not built and the accepted values are summed one at a time, as in the linear scan.

// Integer maximizer of reward_k[n] + brow[nbar-n] over n = 0..nbar (concave): the first n whose
// forward difference is <= 0.
static int_t bounds_peak(const real_t* restrict reward_k, const real_t* restrict brow, const int_t nbar) {
    int_t a = 0, b = nbar;
    while (a < b) {
        const int_t mid = a + (b-a)/2;
        if (reward_k[mid+1] + brow[nbar-mid-1] > reward_k[mid] + brow[nbar-mid]) a = mid+1; else b = mid;
    }
    return a;
}

// {n in 0..nbar : r_acc + reward_k[n] + brow[nbar-n] > 0} = [*lo, *hi] (empty if *lo > *hi), for
// the concave function with maximizer pk. On entry *lo and *hi hold the guesses of the warm start
// (any values; they are clamped). Evaluated exactly as in the linear scan.
static inline void bounds_superlevel(
        const real_t* restrict reward_k, const real_t* restrict brow, const real_t r_acc,
        const int_t nbar, const int_t pk, int_t* lo, int_t* hi
) {
#define BOUNDS_F(n) (r_acc + reward_k[(n)] + brow[nbar-(n)])
    if (!(BOUNDS_F(pk) > 0)) { *lo = 1; *hi = 0; return; }
    int_t a, b;
    // first n in [0, pk] with f > 0 (f nondecreasing there, f(pk) > 0)
    int_t g = *lo < 0 ? 0 : *lo > pk ? pk : *lo;
    if (BOUNDS_F(g) > 0) {
        for (int_t step = 1;; step <<= 1) {
            const int_t t = g - step;
            if (t < 0) { a = 0; b = g; break; }
            if (BOUNDS_F(t) > 0) g = t; else { a = t+1; b = g; break; }
        }
    } else {
        for (int_t step = 1;; step <<= 1) {
            const int_t t = g + step;
            if (t >= pk) { a = g+1; b = pk; break; }
            if (BOUNDS_F(t) > 0) { a = g+1; b = t; break; }
            g = t;
        }
    }
    while (a < b) {
        const int_t mid = a + (b-a)/2;
        if (BOUNDS_F(mid) > 0) b = mid; else a = mid+1;
    }
    *lo = a;
    // last n in [pk, nbar] with f > 0 (f nonincreasing there, f(pk) > 0)
    g = *hi < pk ? pk : *hi > nbar ? nbar : *hi;
    if (BOUNDS_F(g) > 0) {
        for (int_t step = 1;; step <<= 1) {
            const int_t t = g + step;
            if (t > nbar) { a = g; b = nbar; break; }
            if (BOUNDS_F(t) > 0) g = t; else { a = g; b = t-1; break; }
        }
    } else {
        for (int_t step = 1;; step <<= 1) {
            const int_t t = g - step;
            if (t <= pk) { a = pk; b = g-1; break; }
            if (BOUNDS_F(t) > 0) { a = t; b = g-1; break; }
            g = t;
        }
    }
    while (a < b) {
        const int_t mid = a + (b-a+1)/2;
        if (BOUNDS_F(mid) > 0) a = mid; else b = mid-1;
    }
    *hi = a;
#undef BOUNDS_F
}

#ifndef BOUNDS_TAIL_MAX_BYTES
#define BOUNDS_TAIL_MAX_BYTES ((int64_t)1 << 30)
#endif

// Start of row (k, m) in tail_lo / tail_hi: rows of length m+1, stored by k, then m.
static inline int64_t bounds_tail_offset(const int_t N, const int_t k, const int_t m) {
    return (int64_t)k*((int64_t)(N+1)*(N+2)/2) + (int64_t)m*(m+1)/2;
}

// 1 if every row has nonpositive second differences up to rounding.
static int bounds_rows_concave(const real_t* restrict rows, const int_t n_rows, const int_t N) {
    const int_t stride = N+1;
    for (int_t j = 0; j < n_rows; j++) {
        const real_t* restrict f = &rows[j*stride];
        for (int_t i = 1; i < N; i++) {
            const real_t d2 = f[i+1] - 2*f[i] + f[i-1];
            if (d2 > 1e-9*(1 + ABS(f[i+1]) + ABS(f[i]) + ABS(f[i-1]))) return 0;
        }
    }
    return 1;
}

static real_t exhaustive_bounds_recurse(
        const bounds_ctx_t* restrict ctx,
        const int_t k, const int_t N_remain,
        const real_t r_acc, const real_t logp_acc
);

// Same as the linear scan in exhaustive_bounds_recurse, with the classification by binary search.
static real_t exhaustive_bounds_recurse_bs(
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

    int_t* restrict hk = &ctx->hint[4*k];  // warm start: cut points of the previous call at level k
    int_t ga = hk[0], gb = hk[1], ha = 1, hb = 0;
    bounds_superlevel(reward_k, bmax_k1, r_acc, N_remain, ctx->peak_max[k*stride+N_remain], &ga, &gb);
    if (ga <= gb) {
        hk[0] = ga; hk[1] = gb;
        ha = hk[2]; hb = hk[3];
        bounds_superlevel(reward_k, bmin_k1, r_acc, N_remain, ctx->peak_min[k*stride+N_remain], &ha, &hb);
        if (ha <= hb) {
            hk[2] = ha; hk[3] = hb;
            if (ha < ga) ha = ga;  // h <= g, so [ha, hb] lies inside [ga, gb]
            if (hb > gb) hb = gb;
        }
    }
    if (!(ha <= hb)) ha = hb = gb+1;  // nothing rejected: descend all of [ga, gb]

    real_t p = 0;
    if (ctx->tail_lo) {  // accepted values in closed form (see "Tail tables")
        if (ga > gb) { ga = N_remain+1; gb = N_remain; }  // everything accepted
        const int64_t off = bounds_tail_offset(ctx->N, k, N_remain);
        real_t w = 0;
        if (ga > 0) w += ctx->tail_lo[off+ga-1];
        if (gb < N_remain) w += ctx->tail_hi[off+gb+1];
        if (w > 0) p += EXP(MAX_EXP + logp_acc + ctx->tail_scale[k*stride+N_remain])*w;
        for (int_t n = ga; n < ha; n++) p += exhaustive_bounds_recurse(ctx, k1, N_remain-n, r_acc + reward_k[n], logp_acc + logp_k[n]);
        for (int_t n = hb < gb ? hb+1 : gb+1; n <= gb; n++) p += exhaustive_bounds_recurse(ctx, k1, N_remain-n, r_acc + reward_k[n], logp_acc + logp_k[n]);
        return p;
    }
    for (int_t n = 0; n <= N_remain; n++) {
        if (n == ga) {  // descended, then rejected, then descended
            for (; n < ha; n++) p += exhaustive_bounds_recurse(ctx, k1, N_remain-n, r_acc + reward_k[n], logp_acc + logp_k[n]);
            n = hb < gb ? hb+1 : gb+1;
            for (; n <= gb; n++) p += exhaustive_bounds_recurse(ctx, k1, N_remain-n, r_acc + reward_k[n], logp_acc + logp_k[n]);
            if (n > N_remain) break;
        }
        const real_t logp = logp_acc + logp_k[n];  // accepted (closed form)
        p += EXP(MAX_EXP + logp + bounds_suffix_logp(ctx, k1, N_remain-n));
    }
    return p;
}

static real_t exhaustive_bounds_recurse(
        const bounds_ctx_t* restrict ctx,
        const int_t k, const int_t N_remain,
        const real_t r_acc, const real_t logp_acc
) {
    if (ctx->peak_max) return exhaustive_bounds_recurse_bs(ctx, k, N_remain, r_acc, logp_acc);

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
    real_t r_start;    // -tau: accumulated reward at the root (see exhaustive_tie_tolerance)
    atomic_int_fast64_t* restrict next_n0;
    real_t p_acc;
    const options_t* options;
    atomic_int* timed_out;
} bounds_thread_args_t;

static void* exhaustive_bounds_thread(void* args_void) {
    bounds_thread_args_t* args = args_void;
    {  // warm-start cut points of this thread: lower ones start at 0, upper ones at N
        const int_t K = args->ctx.K;
        int_t* hint = malloc(4*K*sizeof(int_t));
        for (int_t k = 0; k < K; k++) { hint[4*k] = 0; hint[4*k+1] = args->ctx.N; hint[4*k+2] = 0; hint[4*k+3] = args->ctx.N; }
        args->ctx.hint = hint;
    }
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
            const real_t r = args->r_start + ctx->reward[n0];

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

    free(args->ctx.hint);
    args->ctx.hint = NULL;
    args->p_acc = p_acc;
    return NULL;
}

static void exhaustive_bounds_parallel(
        const bounds_ctx_t* restrict ctx, const real_t logp_acc0, const real_t r_start, real_t* restrict p_acc, const int_t threads,
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
        bounds_thread_args_t args = { .ctx = *ctx, .logp_acc0 = logp_acc0, .r_start = r_start, .next_n0 = &next_n0, .p_acc = 0, .options = options, .timed_out = timed_out };
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
        args[t].r_start = r_start;
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

    // Binary search tables, when the reward rows 0..K-2 and the bound rows 1..K-1 are concave
    // (always for the statistics the library supports; see "Binary search" above). O(K N log N).
    int_t* peak_max = NULL;
    int_t* peak_min = NULL;
    if (K >= 3 && bounds_rows_concave(reward, K-1, N) && bounds_rows_concave(&bound_min[stride], K-1, N)
            && bounds_rows_concave(&bound_max[stride], K-1, N)) {
        peak_max = malloc(matrix_size*sizeof(int_t));
        peak_min = malloc(matrix_size*sizeof(int_t));
        for (int_t k = 1; k <= K-2; k++) {  // row 0 is the root, scanned linearly
            for (int_t nbar = 0; nbar <= N; nbar++) {
                peak_max[k*stride+nbar] = bounds_peak(&reward[k*stride], &bound_max[(k+1)*stride], nbar);
                peak_min[k*stride+nbar] = bounds_peak(&reward[k*stride], &bound_min[(k+1)*stride], nbar);
            }
        }
    }

    // Tail tables (see "Tail tables" above), only with the binary search and within the memory cap.
    real_t* tail_lo = NULL;
    real_t* tail_hi = NULL;
    real_t* tail_scale = NULL;
    const int64_t tail_size = (int64_t)(K-1)*((int64_t)(N+1)*(N+2)/2);  // rows k = 1..K-2 used
    if (peak_max && 2*tail_size*(int64_t)sizeof(real_t) <= BOUNDS_TAIL_MAX_BYTES) {
        tail_lo = malloc((size_t)tail_size*sizeof(real_t));
        tail_hi = malloc((size_t)tail_size*sizeof(real_t));
        tail_scale = malloc(matrix_size*sizeof(real_t));
        if (!tail_lo || !tail_hi || !tail_scale) {
            free(tail_lo); free(tail_hi); free(tail_scale);
            tail_lo = tail_hi = tail_scale = NULL;
        } else {
            for (int_t k = 1; k <= K-2; k++) {
                const real_t* logp_k = &log_prob[k*stride];
                const real_t lq = log_psuffix[k+1];
                for (int_t m = 0; m <= N; m++) {
                    const real_t sc = (real_t)m*log_psuffix[k] - lgac[m];  // log(P_k^m/m!)
                    tail_scale[k*stride+m] = sc;
                    real_t* lo = &tail_lo[bounds_tail_offset(N, k, m)];
                    real_t* hi = &tail_hi[bounds_tail_offset(N, k, m)];
                    for (int_t n = 0; n <= m; n++) lo[n] = EXP(logp_k[n] - lgac[m-n] + (real_t)(m-n)*lq - sc);
                    hi[m] = lo[m];
                    for (int_t n = m-1; n >= 0; n--) hi[n] = lo[n] + hi[n+1];
                    for (int_t n = 1; n <= m; n++) lo[n] += lo[n-1];
                }
            }
        }
    }

    bounds_ctx_t ctx = {
        .K = K, .N = N,
        .reward = reward, .log_prob = log_prob,
        .bound_min = bound_min, .bound_max = bound_max,
        .log_psuffix = log_psuffix,
        .peak_max = peak_max, .peak_min = peak_min,
        .tail_lo = tail_lo, .tail_hi = tail_hi, .tail_scale = tail_scale,
        .hint = NULL,
    };

    real_t p_acc = 0;
    const real_t logp_acc0 = lgac[N];
    const real_t r_start = -exhaustive_tie_tolerance(mult);  // accept S - S0 <= tau: exact ties count
    atomic_int timed_out;
    atomic_init(&timed_out, 0);
    exhaustive_bounds_parallel(&ctx, logp_acc0, r_start, &p_acc, options->threads > 1 ? options->threads : 1, options, &timed_out);

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
    free(peak_max);
    free(peak_min);
    free(tail_lo);
    free(tail_hi);
    free(tail_scale);
    free(perm);
}
