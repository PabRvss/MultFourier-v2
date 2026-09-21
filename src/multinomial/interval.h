#pragma once

// Fast replacement for fill_interval's O(K*N^2) dynamic program (multinomial.h). Computes the
// exact range [s_min, s_max] of S(x) - S(x0) = sum_k [g_k(x_k) - g_k(x0_k)] over integer
// observations x with x_k >= 0, sum_k x_k = N, where g_k(x_k) is the per-category contribution to
// the statistic (mult->matrix[...].reward, already shifted by the observed-data baseline).
//
// Two families are handled, both in O(K log K) or better:
//
//  * The power-divergence family (options->lambda a real number, the LLR limit lambda=0
//    included): g_k is smooth and strictly concave for every real lambda, which gives both an
//    exact O(K) minimum and a closed-form O(1) *upper bound* on the maximum (no search needed at
//    all). See "POWER-DIVERGENCE FAMILY" below.
//
//  * The probability-mass statistic (options->lambda = NaN): g_k is also concave, so the same
//    O(K) minimum formula applies unchanged; the maximum, however, is exactly the mode of the
//    Multinomial(N, probs) distribution (maximizing S(x) and maximizing the multinomial PMF are
//    the same integer program, since P(x) = N! exp(S(x))), found exactly via GreedyModeFind
//    (White & Hendy, 2010, latex/white2010.pdf), an O(K log K) heap-based algorithm. See
//    "PROBABILITY-MASS STATISTIC" below.
//
// ---- POWER-DIVERGENCE FAMILY ----
//
// Both endpoints reduce to O(K) here because g_k is, for every real lambda, a smooth strictly
// concave function of x_k on (0, N]:
//
//   g_k(x) = -1/(lambda*(lambda+1)) * x * ((x/(N p_k))^lambda - 1)   (lambda != 0)
//   g_k(x) = -x * log(x/(N p_k))                                    (lambda  = 0, the LLR limit)
//
//   g_k''(x) = -(N p_k)^{-lambda} * x^{lambda-1} < 0  for all x > 0 and all real lambda,
//
// (the sign of lambda*(lambda+1) cancels against the same factor picked up by differentiating
// x^{lambda+1} twice, so the family is concave uniformly, with no special case at lambda in
// (-1,0)). S(x) = sum_k g_k(x_k) is therefore concave and additively separable in x, over the
// convex polytope {x >= 0, sum x = N}.
//
// MINIMUM (exact, O(K)): a concave function attains its minimum over a convex polytope at one of
// its extreme points -- for any x in the polytope, writing x = sum_k (x_k/N)*(N e_k) as a convex
// combination of the vertices N*e_k and applying concavity gives S(x) >= min_k S(N e_k), while the
// vertices themselves are feasible, so the minimum over the whole polytope equals the minimum
// over just the K vertices "put everything in category k". The (shifted) value at vertex k is
//   V_k = [g_k(N) - r_k] + sum_{j != k} [g_j(0) - r_j] = [g_k(N) - r_k] - (r0 - r_k) = g_k(N) - r0,
// using g_j(0) = 0 and r0 = sum_j r_j. Since g_k(N) - r_k is already stored at matrix[k][N], and
// r_k = -matrix[k][0] (because matrix[k][0] = g_k(0) - r_k = -r_k), this needs only two already-
// computed matrix lookups per category plus mult->r0 -- no new evaluation of reward_power at all.
// This argument only used concavity of g_k, not any property specific to the power-divergence
// family, so the identical formula is exact for the pmf statistic too (reward_logp is concave via
// g_k''(x) = -trigamma(x+1) < 0) -- see interval_min_fast below, shared by both families.
//
// MAXIMUM (closed-form upper bound, O(1)): dropping integrality (x_k allowed to be any real >= 0
// summing to N) turns the maximization of the same concave, separable S into a smooth equality-
// constrained problem. Substituting x_k = N*p_k directly into g_k' shows every category reaches
// the *same* stationary value there:
//   g_k'(N p_k) = -1/(lambda+1)   for every k, independent of p_k,
// i.e. x = N*p satisfies the KKT condition for a single shared Lagrange multiplier, and by
// concavity that stationary point is the global maximum of the continuous relaxation. Since
// g_k(N p_k) = 0 for every k (the base of the power ratio is 1 there), the continuous maximum of
// S is exactly 0, so the continuous maximum of S(x) - S(x0) is exactly -r0. Because the true
// (integer-constrained) maximum can only be smaller -- the integer lattice is a subset of the
// relaxed domain -- s_max <= -r0 is a valid upper bound, and a tight one whenever N*p_k is close
// to an integer for every k. This is just the standard fact that a power-divergence statistic is
// nonnegative and vanishes only when the observation matches its expectation; no Newton search
// is needed at all to locate it (the "cheap Newton on a diagonal Hessian" this file was expected
// to need turns out to have a closed-form root for this particular family).
//
// A looser (or in degenerate cases negative-appearing) s_max only ever widens W = max(s_max,
// -s_min); it can raise the FFT/series cost slightly but never the correctness of the result,
// and in every case checked so far s_min dominates W (matching the intuition that the left tail
// -- concentrating observations into the least-probable categories -- is the binding constraint).

static real_t interval_min_fast(const multinomial* mult) {
    const int_t K = mult->K;
    const int_t N = mult->N;

    real_t s_min = INFINITY;
    for (int_t k=0; k<K; k++) {
        const int_t index_k = k*(N+1);
        // matrix[index_k+N] = g_k(N)-r_k, matrix[index_k+0] = -r_k; r0 folded in by the caller
        const real_t v = mult->matrix[index_k+N].reward - mult->matrix[index_k].reward;
        if (v < s_min) s_min = v;
    }
    return s_min - (real_t)mult->r0;
}

// ---- PROBABILITY-MASS STATISTIC: GreedyModeFind (White & Hendy, 2010) ----
//
// S(x) = sum_k reward_logp(log_probs[k], x_k) = log P(x) - lgac[N] is maximized over integer
// x >= 0, sum x = N, exactly at a mode of Multinomial(N, probs). GreedyModeFind finds one such
// mode in O(K log K): it starts from the unique mode of a nearby, trivially computed distribution
// (Theorem 1: w_k = floor(a*p_k) is the unique mode of p in K_{m,K} for m = sum_k w_k, any real
// a > 0), then walks frequency-by-frequency from m to N, at each step incrementing (decrementing,
// if m > N) whichever category most improves the likelihood ratio (Theorems 2 and 3: the best
// category to bump maximizes p_k/(x_k+1), resp. x_k/p_k). A binary max-heap keyed on that
// per-category score turns "pick the best category, then update its score" into an O(log K)
// operation, and at most K/2 such steps are needed (see the paper's section 3), for O(K log K)
// total. We only need one mode (not the full set of joint modes), so the tie-handling machinery
// of the paper's GreedyModeFind proper is unnecessary -- this is its simpler GreedyModeFind1.

typedef struct {
    real_t score;
    int_t elem;
} heap_node_t;

static inline void heap_sift_down(heap_node_t* restrict h, const int_t size, int_t i) {
    for (;;) {
        int_t largest = i;
        const int_t l = 2*i+1, r = 2*i+2;
        if (l < size && h[l].score > h[largest].score) largest = l;
        if (r < size && h[r].score > h[largest].score) largest = r;
        if (largest == i) return;
        const heap_node_t tmp = h[i]; h[i] = h[largest]; h[largest] = tmp;
        i = largest;
    }
}

// x must have room for K entries; filled with a mode of Multinomial(N, probs).
static void greedy_mode_find(const int_t N, const int_t K, const real_t* restrict probs, int_t* restrict x) {
    const real_t a = (real_t)N + (real_t)K/2;
    int_t m = 0;
    for (int_t k=0; k<K; k++) {
        x[k] = (int_t)floor(a*probs[k]);
        m += x[k];
    }
    if (m == N) return;  // Theorem 1: x is already the unique mode, no heap needed

    const int increasing = m < N;
    heap_node_t* restrict h = malloc(K*sizeof(heap_node_t));
    for (int_t k=0; k<K; k++) {
        h[k].elem = k;
        h[k].score = increasing ? probs[k]/(real_t)(x[k]+1) : (real_t)x[k]/probs[k];
    }
    for (int_t i=K/2-1; i>=0; i--) heap_sift_down(h, K, i);

    // m moves monotonically towards N by 1 per iteration without crossing it, so the direction
    // decided above ("increasing") never needs to be rechecked.
    while (m != N) {
        const int_t best = h[0].elem;
        if (increasing) {
            x[best] += 1;
            h[0].score = probs[best]/(real_t)(x[best]+1);
            m += 1;
        } else {
            x[best] -= 1;
            h[0].score = (real_t)x[best]/probs[best];
            m -= 1;
        }
        heap_sift_down(h, K, 0);  // root's key only ever decreases, so a sift-down suffices
    }
    free(h);
}

static real_t interval_max_pmf(const multinomial* mult) {
    const int_t K = mult->K;
    int_t* restrict x = malloc(K*sizeof(int_t));
    greedy_mode_find(mult->N, K, mult->probs, x);

    real_t s = 0;
    for (int_t k=0; k<K; k++) s += reward_logp(mult->log_probs[k], x[k]);
    free(x);

    return s - (real_t)mult->r0;
}

static void fill_interval_fast(multinomial* mult, const options_t* options) {
    const real_t s_min = interval_min_fast(mult);
    const real_t s_max = real_isnan(options->lambda) ? interval_max_pmf(mult) : -(real_t)mult->r0;

    const real_t W_max = MAX(s_max, -s_min);
    const real_t W_eff = W_max/options->undersampling;

    mult->W = W_eff;
    mult->T = W_eff == 0? 1.0 : M_PI/W_eff;

    if (options->verbose) {
        printf("\tS in %.3f + [%.3f, %.3f]  (fast interval)\n", mult->r0, s_min, s_max);
        printf("\tW = %.3f, T = %f\n", W_eff, mult->T);
    }
}
