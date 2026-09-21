#pragma once

#ifndef GAMMA_NU_BOUND
#define GAMMA_NU_BOUND 100.0
#endif
#ifndef GAMMA_NEWTON_ITERS
#define GAMMA_NEWTON_ITERS 30
#endif

// Saddlepoint search for the tilt parameter gamma.
//
// The objective is psi(gamma) = log M(-gamma) + log sinc(pi*gamma/T), where M(-gamma) is the
// moment generating function of the reward sum S and the sinc factor is the transform of the
// uniform smoothing variable U (see error_bound.h). psi is therefore the cumulant generating
// function of S+U, which makes it CONVEX, and psi' is increasing: the minimiser is the root of a
// monotone function and the sign of psi' brackets it. The grid refinement in fill_gamma_grid
// ignores all of that -- it evaluates 64 gammas per round and keeps the smallest.
//
// Two things make a derivative search much cheaper than that grid.
//
// 1. THE SURROGATE. The exact M(-gamma) requires the K-fold constrained convolution over the
//    compositions of N. Dropping the sum_k x_k = N constraint Poissonises the problem: the
//    categories become independent and
//
//        log M_pois(gamma, nu) = sum_k log Z_k - nu*N + lgac[N],
//        Z_k = sum_i exp(-gamma*r_ki + logp_ki + nu*i),
//
//    where nu is the tilt that puts the expected total back at N, i.e. sum_k E_k[i] = N. That is
//    the saddlepoint approximation to the exact quantity, it agrees with it to O(1/N), and it
//    costs one O(K*(N+1)) pass instead of an O(K*M log M) convolution. It is also the same nu the
//    FFT evaluator already solves for in find_nu_gamma_vec -- the surrogate is not a new
//    approximation to the pipeline, it is the one the pipeline is already built on.
//
// 2. THE DERIVATIVES ARE FREE. Within one pass we can accumulate, per category, the normalised
//    moments of the count i and the reward r and their covariance. Then, because nu(gamma) is
//    defined by a stationarity condition, the envelope theorem gives the first derivative with no
//    reference to dnu/dgamma at all:
//
//        psi'(gamma)  = -sum_k E_k[r] + (pi/T)*g'(x),
//        psi''(gamma) =  sum_k Var_k[r] - (sum_k Cov_k[r,i])^2 / (sum_k Var_k[i]) + (pi/T)^2*g''(x)
//
//    with x = pi*gamma/T and g = log sinc. The second derivative is the variance of the reward
//    sum left after conditioning on the total count -- non-negative, as convexity requires.
//
// What this buys: the search runs in about six gamma evaluations instead of 128, and each
// evaluation is a K*(N+1) pass instead of a convolution. On the default path nothing exact is
// needed at all, because the only output fill_gamma has to hand downstream is mult->l_rate, which
// IS nu. Only the FFT_NONE evaluator needs more (the per-cell exponent shifts of the exact
// recursion), and then exactly once, at the chosen gamma, via fill_shifts below.
//
// Note on the other obvious idea: exp(-(gamma0 + j*delta)*r) is a geometric family in j, so the
// grid's exponentials could in principle be produced by multiplication rather than by calling exp.
// That does not survive contact with this code, because every sum here is a log-sum-exp whose
// VEC_MAX pivot is chosen per gamma; the shifts differ from lane to lane and break the geometric
// structure. Reducing the number of evaluations, as below, is the way out instead.

// g(x) = log((1 - exp(-x))/x) and its first two derivatives, over the whole real line.
// pi*gamma/T is in the thousands for typical instances, so the large-|x| branches are the common
// case, not the exceptional one: 1/(exp(x) - 1) must never be formed there.
static inline void sinc_log_derivs(const real_t x, real_t* restrict g0, real_t* restrict g1, real_t* restrict g2) {
    if (ABS(x) < 1e-4) {  // series; the closed forms cancel catastrophically here
        *g0 = -x*(0.5 - x/24);
        *g1 = -0.5 + x/12;
        *g2 = 1.0/12;
        return;
    }
    if (x > MAX_EXP) {    // exp(-x) underflows, 1/(exp(x) - 1) is zero to working precision
        *g0 = -LOG(x);
        *g1 = -1.0/x;
        *g2 = 1.0/(x*x);
        return;
    }
    if (x < -MAX_EXP) {   // (1 - exp(-x))/x ~ -exp(-x)/x
        *g0 = -x - LOG(-x);
        *g1 = -1.0 - 1.0/x;
        *g2 = 1.0/(x*x);
        return;
    }
    const real_t ex = EXP(x);
    const real_t em1 = ex - 1.0;
    *g0 = LOG((1.0 - EXP(-x))/x);
    *g1 = 1.0/em1 - 1.0/x;
    *g2 = 1.0/(x*x) - ex/(em1*em1);
}

// Per-category normalised moments at (gamma, nu), summed over categories. One pass over every
// cell; everything the search needs is here.
typedef struct {
    real_t log_z;  // sum_k log Z_k
    real_t m_i;    // sum_k E_k[i]
    real_t v_i;    // sum_k Var_k[i]
    real_t m_r;    // sum_k E_k[r]
    real_t v_r;    // sum_k Var_k[r]
    real_t c_ri;   // sum_k Cov_k[r,i]
} gamma_moments_t;

static gamma_moments_t gamma_moments(
        const cell_t* restrict global, const int_t N, const int_t K,
        const real_t gamma, const real_t nu
) {
    gamma_moments_t g = {0, 0, 0, 0, 0, 0};

    for (int_t k=0; k<K; k++) {
        const int_t index_k = k*(N+1);

        real_t max_exp = -INFINITY;
        for (int_t i=0; i<=N; i++) {
            const cell_t c = global[index_k+i];
            const real_t e = -c.reward*gamma + c.log_prob + nu*(real_t)i;
            if (e > max_exp) max_exp = e;
        }
        if (!(max_exp > -INFINITY)) continue;

        real_t z=0, si=0, sii=0, sr=0, srr=0, sri=0;
        for (int_t i=0; i<=N; i++) {
            const cell_t c = global[index_k+i];
            const real_t w = EXP(-c.reward*gamma + c.log_prob + nu*(real_t)i - max_exp);
            const real_t xi = (real_t)i;
            z += w;
            si += w*xi;        sii += w*xi*xi;
            sr += w*c.reward;  srr += w*c.reward*c.reward;
            sri += w*c.reward*xi;
        }
        if (!(z > 0)) continue;

        const real_t mi = si/z;
        const real_t mr = sr/z;
        g.log_z += max_exp + LOG(z);
        g.m_i  += mi;
        g.v_i  += sii/z - mi*mi;
        g.m_r  += mr;
        g.v_r  += srr/z - mr*mr;
        g.c_ri += sri/z - mr*mi;
    }
    return g;
}

typedef struct {
    real_t psi;  // surrogate objective at gamma
    real_t d1;   // dpsi/dgamma
    real_t d2;   // d2psi/dgamma2, >= 0
    real_t nu;   // the Poissonisation tilt (this is mult->l_rate)
    int_t passes;
} gamma_eval_t;

// Solve sum_k E_k[i] = N for nu, then read off the objective and its derivatives there.
// The nu equation is monotone in nu (its derivative is sum_k Var_k[i]), so the same safeguarded
// Newton as find_nu_gamma_vec applies; nu_start warm-starts it from the previous gamma, which is
// what keeps this to two or three passes after the first evaluation.
static gamma_eval_t gamma_eval(const multinomial* restrict mult, const real_t gamma, const real_t nu_start) {
    const cell_t* restrict global = mult->matrix;
    const int_t N = mult->N;
    const int_t K = mult->K;

    real_t nu = nu_start;
    real_t lo = -GAMMA_NU_BOUND;
    real_t hi = GAMMA_NU_BOUND;
    gamma_moments_t g = {0, 0, 0, 0, 0, 0};
    int_t passes = 0;
    int settled = 0;

    for (int_t it=0; it<GAMMA_NEWTON_ITERS; it++) {
        g = gamma_moments(global, N, K, gamma, nu);
        passes++;

        const real_t f = g.m_i - (real_t)N;
        if (ABS(f) <= 1e-11*((real_t)N + 1.0)) { settled = 1; break; }

        if (f > 0) hi = nu; else lo = nu;
        const real_t fp = g.v_i > 1e-300 ? g.v_i : 1e-300;
        real_t next = nu - f/fp;
        if (!(next > lo && next < hi)) next = 0.5*(lo + hi);
        if (next == nu) { settled = 1; break; }
        nu = next;
    }
    if (!settled) { g = gamma_moments(global, N, K, gamma, nu); passes++; }  // keep g in step with nu

    real_t g0, g1, g2;
    const real_t c = M_PI/mult->T;
    sinc_log_derivs(c*gamma, &g0, &g1, &g2);

    const real_t vi = g.v_i > 1e-300 ? g.v_i : 1e-300;

    gamma_eval_t e;
    e.nu = nu;
    e.psi = g.log_z - nu*(real_t)N + lgac[N] + g0;
    e.d1 = -g.m_r + c*g1;
    e.d2 = g.v_r - g.c_ri*g.c_ri/vi + c*c*g2;
    e.passes = passes;
    return e;
}

// Minimise psi over [lo, hi] by Newton on psi' with a bisection safeguard, exactly as
// find_nu_gamma_vec does for nu. The bracket is the caller's and is never widened: a minimum on
// the boundary is returned as the boundary.
static real_t gamma_newton_search(
        const multinomial* restrict mult, real_t lo, real_t hi, const real_t eps_gamma,
        real_t* restrict nu_out, real_t* restrict psi_out, int_t* restrict evals
) {
    int_t n_eval = 0;
    gamma_eval_t e;

    e = gamma_eval(mult, lo, 0); n_eval++;
    if (!(e.d1 < 0)) { *nu_out = e.nu; *psi_out = e.psi; *evals = n_eval; return lo; }
    real_t nu = e.nu;

    e = gamma_eval(mult, hi, nu); n_eval++;
    if (!(e.d1 > 0)) { *nu_out = e.nu; *psi_out = e.psi; *evals = n_eval; return hi; }
    nu = e.nu;

    // Newton where it behaves, bisection where it does not, and the bracket alone decides when to
    // stop. The step size cannot be used as the stopping rule here: psi'' spans six orders of
    // magnitude across the bracket -- the sinc term contributes (pi/T)^2/12 at gamma = 0 but only
    // 1/gamma^2 once pi*gamma/T exceeds one -- so near zero Newton takes steps far smaller than
    // the distance to the minimiser (it doubles gamma each iteration rather than converging).
    // Forcing a bisection whenever Newton fails to halve the previous step keeps the bracket
    // shrinking geometrically regardless.
    real_t gamma = 0.5*(lo + hi);
    real_t dx = hi - lo;

    for (int_t it=0; it<GAMMA_NEWTON_ITERS; it++) {
        e = gamma_eval(mult, gamma, nu); n_eval++;
        nu = e.nu;

        if (e.d1 > 0) hi = gamma; else lo = gamma;
        if (hi - lo <= eps_gamma) break;

        const real_t d2 = e.d2 > 1e-300 ? e.d2 : 1e-300;
        const real_t next = gamma - e.d1/d2;
        const real_t dx_old = dx;

        if (!(next > lo && next < hi) || ABS(2.0*e.d1) > ABS(dx_old*d2)) {
            dx = 0.5*(hi - lo);
            gamma = lo + dx;
        } else {
            dx = gamma - next;
            gamma = next;
        }
    }

    // eps_gamma bounds the bracket, but a derivative method can do much better than its bracket
    // for free: the Newton refinement of the last iterate is accurate to the square of its error.
    // That is worth taking here. The series length is acutely sensitive to gamma near the optimum
    // -- on an N=1000, K=7 instance, moving gamma by 0.005 inside a 0.01-wide bracket changed the
    // terms to convergence from 2171 to 6711 -- so returning an arbitrary point of the bracket,
    // which is all the grid can do, leaves a large factor on the table. The refinement stays
    // clipped to the bracket that was actually established, so it never weakens the guarantee.
    for (int_t it=0; it<8; it++) {
        const real_t d2 = e.d2 > 1e-300 ? e.d2 : 1e-300;
        real_t next = gamma - e.d1/d2;
        if (next < lo) next = lo;
        if (next > hi) next = hi;
        if (ABS(next - gamma) <= 1e-12*(1.0 + ABS(gamma))) break;

        gamma = next;
        e = gamma_eval(mult, gamma, nu); n_eval++;
        nu = e.nu;
    }

    *nu_out = e.nu;
    *psi_out = e.psi;
    *evals = n_eval;
    return gamma;
}

// The FFT_NONE evaluator reads mult->matrix[].shift, the per-cell exponent pivots of the exact
// recursion. Nothing in the search above needs them, so this runs once, at the chosen gamma --
// where the old grid paid for them on all 128 evaluations and discarded 127 sets.
static void fill_shifts(multinomial* restrict mult, const real_t gamma) {
    const int_t N = mult->N;
    const int_t K = mult->K;
    cell_t* restrict global = mult->matrix;

    real_t* restrict acc = malloc((size_t)(N+1)*sizeof(real_t));
    acc[0] = 0;
    for (int_t i=1; i<=N; i++) acc[i] = -INFINITY;

    for (int_t k=K-1; k>=0; k--) {  // reversed, in place: acc[i] is written after its last read
        const int_t index_k = k*(N+1);
        for (int_t i=N; i>=0; i--) {
            real_t max_exp = -INFINITY;
            for (int_t j=0; j<=i; j++) {
                const cell_t c = global[index_k + i-j];
                const real_t e = -c.reward*gamma + c.log_prob + acc[j];
                if (e > max_exp) max_exp = e;
            }

            real_t res = 0;
            for (int_t j=0; j<=i; j++) {
                const cell_t c = global[index_k + i-j];
                res += EXP(-c.reward*gamma + c.log_prob + acc[j] - max_exp);
            }

            acc[i] = max_exp + LOG(res);
            global[index_k + i].shift = max_exp;
        }
    }

    free(acc);
}
