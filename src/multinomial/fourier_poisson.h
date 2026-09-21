#pragma once

#ifndef GAMMA_NU_BOUND
#define GAMMA_NU_BOUND 100.0
#endif
#ifndef GAMMA_NEWTON_ITERS
#define GAMMA_NEWTON_ITERS 30
#endif

// Poissonised evaluator for the series terms. See latex/poisson.tex for the full derivation; the
// short version:
//
// Every other evaluator computes the constrained transform
//
//     M_S(z_m) = N! [t^N] prod_k phi_k(t; m),     phi_k(t; m) = sum_i t^i p_k(i) exp(-z_m a_k(i)),
//
// by multiplying the K polynomials together and cutting the running product back to degree N after
// each category. That truncation is what forces the FFT tree to work at size M ~ 2N and to pay a
// full descent, leaf convolution and ascent per category.
//
// Here the constraint is dropped instead of enforced. The categories become independent, each
// phi_k is evaluated on its own, and [t^N] is recovered exactly afterwards by Cauchy's formula on a
// circle of radius rho = exp(nu), discretised with Q equispaced points:
//
//     [t^N] F = rho^{-N} (1/Q) sum_{q<Q} omega^{-qN} F(rho omega^q)  +  aliasing,  omega = e^{2 pi i/Q}.
//
// So per frequency the work is, for each category, one fold of its N+1 coefficients mod Q and one
// length-Q FFT (which evaluates phi_k at all Q points at once), then a pointwise product over
// categories and a single dot product. No convolution, no truncation, no plan of merge factors.
//
// The only error is aliasing: the trapezoid rule returns sum_j [t^{N+jQ}] F rho^{jQ}, so the
// coefficients at total counts N +- Q, N +- 2Q, ... leak in. Choosing nu as the saddlepoint -- the
// same Poissonisation tilt the gamma search solves for, with sum_k E_k[i] = N -- centres the
// tilted total-count distribution on N, where those far coefficients are suppressed like its tails.
// In practice Q of 7 to 10 standard deviations of that distribution is enough. Two facts make the
// choice certified rather than heuristic:
//
//   1. At frequency zero every coefficient is positive, and at any other frequency each coefficient
//      is bounded in modulus by its frequency-zero counterpart. So the aliasing at m = 0 bounds it,
//      on the absolute scale of the m = 0 term, for EVERY term of the series. (Measured: the bound
//      is attained to two or three significant digits -- the worst term is the first one.)
//   2. That m = 0 aliasing can be measured. S(0) at Q minus S(0) at 2Q is exactly alpha(Q) -
//      alpha(2Q), where alpha is the relative aliasing; and alpha(2Q) <= alpha(Q)^2 whenever the
//      tilted total-count law is log-concave (proved for the PMF statistic with gamma <= 1 in
//      latex/poisson.tex; for LLR the certificate is confirmed by measurement instead), so the
//      difference IS alpha(Q) to first order. build_poisson_plan starts low, at POISSON_SIGMAS standard deviations, and doubles Q
//      until the difference is below POISSON_ALIAS_TOL, or until Q exceeds max(N, (K-1)*N), past
//      which no alias exists at all and the formula is exact. Starting low matters because Q moves
//      in factors of two and the certificate can only ever raise it.
//
// The tolerance is on the terms, which is conservative for p: aliasing is concentrated in the
// lowest frequencies and largely cancels in the sum, and its measured effect on the partial sums
// is two to six orders of magnitude below the certificate.
//
// Precision layout mirrors the precompute evaluator: everything that does not depend on the
// frequency -- the per-category scale, the normalised weights (each category sums to one), the
// overall magnitude in quad -- lives in the plan, and the per-frequency arithmetic runs on
// quantities of modulus at most one. The FFT and the product vectorise across the STRIDE lanes
// directly, because every lane shares the fold positions, the roots and the product structure.

#define POISSON_Q_MIN     16
#define POISSON_Q_MAX     (1LL << 20)
#define POISSON_SIGMAS    4.0     // starting Q, in standard deviations of the total count
#define POISSON_ALIAS_TOL 1e-10   // accepted m = 0 aliasing, relative; term rounding floor is ~1e-12

typedef struct {
    int_t N;
    int_t K;
    int_t Q;                    // points on the circle, a power of two
    real_t nu;                  // radius rho = exp(nu)
    real_t var;                 // variance of the tilted total count
    double alias_rel;           // measured m = 0 aliasing at the chosen Q (0: no alias possible, exact)
    double* restrict weight;    // K*(N+1): exp(log_prob + nu*i - scale_k), summing to one per k
    int_t* restrict dest;       // N+1: bit-reversed bin of count i, so the FFT needs no permutation
    double* restrict root_re;   // Q/2: exp(-2 pi i j/Q)
    double* restrict root_im;
    double* restrict out_re;    // Q: exp(+2 pi i q N/Q)/Q, the coefficient extractor
    double* restrict out_im;
    quad_t resN_exp;            // exp(lgac[N] + sum_k scale_k - nu*N)
} poisson_plan_t;

// The saddlepoint radius. After fill_cells log_prob already carries the gamma tilt, so gamma_moments
// with gamma = 0 sees exactly the Poissonised weights exp(log_prob + nu*i). Warm-started from
// mult->l_rate, which is this same quantity whenever the gamma search supplied it; re-solved here
// so the evaluator does not depend on which search or gamma_precision produced the cells.
static real_t poisson_tilt(const multinomial* restrict mult, gamma_moments_t* restrict g_out) {
    const cell_t* restrict global = mult->matrix;
    const int_t N = mult->N;
    const int_t K = mult->K;

    real_t nu = mult->l_rate;
    real_t lo = -GAMMA_NU_BOUND;
    real_t hi = GAMMA_NU_BOUND;
    gamma_moments_t g = gamma_moments(global, N, K, 0.0, nu);

    for (int_t it=0; it<GAMMA_NEWTON_ITERS; it++) {
        const real_t f = g.m_i - (real_t)N;
        if (ABS(f) <= 1e-11*((real_t)N + 1.0)) break;

        if (f > 0) hi = nu; else lo = nu;
        const real_t fp = g.v_i > 1e-300 ? g.v_i : 1e-300;
        real_t next = nu - f/fp;
        if (!(next > lo && next < hi)) next = 0.5*(lo + hi);
        if (next == nu) break;

        nu = next;
        g = gamma_moments(global, N, K, 0.0, nu);
    }

    *g_out = g;
    return nu;
}

static inline int_t poisson_log2(const int_t Q) {
    int_t b = 0;
    while (((int_t)1 << b) < Q) b++;
    return b;
}

static void poisson_bitrev(int_t* restrict rev, const int_t Q) {
    const int_t bits = poisson_log2(Q);
    for (int_t x=0; x<Q; x++) {
        int_t r = 0;
        for (int_t b=0; b<bits; b++) r |= ((x >> b) & 1) << (bits - 1 - b);
        rev[x] = r;
    }
}

// S(0) at a given Q, in plain double complex: the tilted total-count distribution folded mod Q and
// read at N. Used only to certify Q, so it favours independence from the vectorised path over speed.
static double complex poisson_s0(const double* restrict weight, const int_t N, const int_t K, const int_t Q) {
    int_t* restrict rev = malloc((size_t)Q*sizeof(int_t));
    double complex* restrict b = malloc((size_t)Q*sizeof(double complex));
    double complex* restrict p = malloc((size_t)Q*sizeof(double complex));
    double complex* restrict root = malloc((size_t)(Q/2)*sizeof(double complex));
    poisson_bitrev(rev, Q);
    for (int_t j=0; j<Q/2; j++) root[j] = cos(2.0*M_PI*(double)j/(double)Q) - I*sin(2.0*M_PI*(double)j/(double)Q);
    for (int_t q=0; q<Q; q++) p[q] = 1.0;

    for (int_t k=0; k<K; k++) {
        const int_t index_k = k*(N+1);
        for (int_t q=0; q<Q; q++) b[q] = 0.0;
        for (int_t i=0; i<=N; i++) b[rev[i & (Q-1)]] += weight[index_k + i];

        for (int_t len=2; len<=Q; len<<=1) {
            const int_t half = len >> 1;
            const int_t stride = Q/len;
            for (int_t base=0; base<Q; base+=len) {
                for (int_t j=0; j<half; j++) {
                    const double complex v = b[base+j+half]*root[j*stride];
                    const double complex u = b[base+j];
                    b[base+j] = u + v;
                    b[base+j+half] = u - v;
                }
            }
        }
        for (int_t q=0; q<Q; q++) p[q] *= b[q];
    }

    double complex s = 0;
    const int_t NQ = N % Q;
    for (int_t q=0; q<Q; q++) s += p[q]*cexp(2.0*M_PI*I*(double)(q*NQ)/(double)Q);
    s /= (double)Q;

    free(rev);
    free(b);
    free(p);
    free(root);
    return s;
}

static poisson_plan_t* build_poisson_plan(const multinomial* restrict mult, const int_t verbose) {
    const cell_t* restrict global = mult->matrix;
    const int_t N = mult->N;
    const int_t K = mult->K;

    poisson_plan_t* plan = malloc(sizeof(poisson_plan_t));
    plan->N = N;
    plan->K = K;

    gamma_moments_t g;
    const real_t nu = poisson_tilt(mult, &g);
    plan->nu = nu;
    plan->var = g.v_i > 0 ? g.v_i : 0;

    // Normalised weights and the frequency-independent magnitude.
    plan->weight = malloc((size_t)K*(N+1)*sizeof(double));
    real_t sum_scale = 0;
    for (int_t k=0; k<K; k++) {
        const int_t index_k = k*(N+1);

        real_t max_exp = -INFINITY;
        for (int_t i=0; i<=N; i++) {
            const real_t e = global[index_k+i].log_prob + nu*(real_t)i;
            if (e > max_exp) max_exp = e;
        }
        real_t z = 0;
        for (int_t i=0; i<=N; i++) z += EXP(global[index_k+i].log_prob + nu*(real_t)i - max_exp);
        const real_t scale = max_exp + LOG(z);
        sum_scale += scale;

        for (int_t i=0; i<=N; i++) plan->weight[index_k+i] = EXP(global[index_k+i].log_prob + nu*(real_t)i - scale);
    }
    plan->resN_exp = expq((quad_t)lgac[N] + (quad_t)sum_scale - (quad_t)nu*(quad_t)N);

    // Q. The product has degree K*N, so an alias N + l*Q needs l >= 1 with Q <= (K-1)*N, or l <= -1
    // with Q <= N: once Q exceeds both, no total count can alias onto N and doubling stops being useful.
    const int_t Q_need = ((K-1)*N > N ? (K-1)*N : N) + 1;
    int_t Q_exact = POISSON_Q_MIN;
    while (Q_exact < Q_need) Q_exact <<= 1;
    const int_t Q_cap = Q_exact < POISSON_Q_MAX ? Q_exact : POISSON_Q_MAX;

    int_t Q = POISSON_Q_MIN;
    while ((real_t)Q < POISSON_SIGMAS*sqrt(plan->var) && Q < Q_cap) Q <<= 1;

    double alias = 0;
    int certified = 1;
    if (Q < Q_exact) {
        certified = 0;
        double complex s_Q = poisson_s0(plan->weight, N, K, Q);
        double prev = INFINITY;
        for (;;) {
            if (2*Q > POISSON_Q_MAX) { alias = prev; break; }  // cannot afford to measure Q against 2Q

            const double complex s_2Q = poisson_s0(plan->weight, N, K, 2*Q);
            const double a = cabs(s_Q - s_2Q)/cabs(s_2Q);
            // Accept Q once its aliasing is below tolerance -- or once doubling stops helping while
            // the difference is already small, which means it has reached the rounding floor and the
            // aliasing is gone. (Below about two standard deviations the difference saturates near
            // one and also stops shrinking; the smallness test keeps that from being mistaken for
            // convergence.)
            const int floor_reached = (a < 1e-6) && !(a < 0.1*prev);
            if (a <= POISSON_ALIAS_TOL || floor_reached) { alias = a; certified = 1; break; }

            Q <<= 1;
            s_Q = s_2Q;
            prev = a;
            if (Q >= Q_exact) { alias = 0; certified = 1; break; }  // no total count can alias onto N
        }
    }
    plan->Q = Q;
    plan->alias_rel = alias;

    // Tables.
    int_t* rev = malloc((size_t)Q*sizeof(int_t));
    poisson_bitrev(rev, Q);
    plan->dest = malloc((size_t)(N+1)*sizeof(int_t));
    for (int_t i=0; i<=N; i++) plan->dest[i] = rev[i & (Q-1)];
    free(rev);

    plan->root_re = malloc((size_t)(Q/2)*sizeof(double));
    plan->root_im = malloc((size_t)(Q/2)*sizeof(double));
    for (int_t j=0; j<Q/2; j++) {
        const double a = 2.0*M_PI*(double)j/(double)Q;
        plan->root_re[j] = cos(a);
        plan->root_im[j] = -sin(a);
    }

    plan->out_re = malloc((size_t)Q*sizeof(double));
    plan->out_im = malloc((size_t)Q*sizeof(double));
    const int_t NQ = N % Q;
    for (int_t q=0; q<Q; q++) {
        const double a = 2.0*M_PI*(double)(q*NQ)/(double)Q;
        plan->out_re[q] = cos(a)/(double)Q;
        plan->out_im[q] = sin(a)/(double)Q;
    }

    if (verbose) {
        printf("\tpoisson plan: nu = %.6f, sd(total) = %.1f, Q = %lld, m=0 aliasing = %.1e\n",
               (double)nu, sqrt((double)plan->var), (long long)Q, alias);
        if (!certified) printf("\tpoisson plan: WARNING, Q capped at %lld before aliasing was certified\n", (long long)Q);
    }
    return plan;
}

static void free_poisson_plan(poisson_plan_t* plan) {
    free(plan->weight);
    free(plan->dest);
    free(plan->root_re);
    free(plan->root_im);
    free(plan->out_re);
    free(plan->out_im);
    free(plan);
}

// Iterative FFT on vec lanes, input already in bit-reversed order: X[q] = sum_j x_j exp(-2 pi i qj/Q).
// The first level multiplies by root 1 only, so it is a plain sum/difference.
static inline void poisson_fft(vec_t* restrict re, vec_t* restrict im, const poisson_plan_t* restrict plan) {
    const int_t Q = plan->Q;
    const double* restrict root_re = plan->root_re;
    const double* restrict root_im = plan->root_im;

    for (int_t base=0; base<Q; base+=2) {
        const vec_t ur = re[base], ui = im[base];
        const vec_t vr = re[base+1], vi = im[base+1];
        re[base] = ur + vr;    im[base] = ui + vi;
        re[base+1] = ur - vr;  im[base+1] = ui - vi;
    }

    for (int_t len=4; len<=Q; len<<=1) {
        const int_t half = len >> 1;
        const int_t stride = Q/len;
        for (int_t base=0; base<Q; base+=len) {
            for (int_t j=0; j<half; j++) {
                const real_t wr = root_re[j*stride];
                const real_t wi = root_im[j*stride];
                const vec_t xr = re[base+j+half];
                const vec_t xi = im[base+j+half];
                const vec_t vr = wr*xr - wi*xi;
                const vec_t vi = wr*xi + wi*xr;
                const vec_t ur = re[base+j];
                const vec_t ui = im[base+j];
                re[base+j] = ur + vr;       im[base+j] = ui + vi;
                re[base+j+half] = ur - vr;  im[base+j+half] = ui - vi;
            }
        }
    }
}

static void* eval_fourier_poisson_thread(void* args_void) {
    arguments_t* args = args_void;

    const multinomial* restrict mult = args->mult;
    const poisson_plan_t* restrict plan = args->plan;
    qcomplex_t* restrict res_arr = args->res_arr;

    const int_t N = mult->N;
    const int_t K = mult->K;
    const int_t Q = plan->Q;
    const int_t cells = K*(N+1);

    vec_t* restrict b_re = malloc((size_t)Q*sizeof(vec_t));
    vec_t* restrict b_im = malloc((size_t)Q*sizeof(vec_t));
    vec_t* restrict p_re = malloc((size_t)Q*sizeof(vec_t));
    vec_t* restrict p_im = malloc((size_t)Q*sizeof(vec_t));

    const complex_t* restrict phase = args->phase;
    const local_t* restrict lane = args->lane;
    const int_t* restrict dest = plan->dest;
    const double* restrict out_re = plan->out_re;
    const double* restrict out_im = plan->out_im;
    const real_t s2_re = M_PI*mult->gamma/mult->T;

    if (!args->warm) phase_seed(args, args->start, cells);

    for (int_t n=args->start; n<args->end; n+=args->stride_n) {
        for (int_t k=0; k<K; k++) {
            const int_t index_k = k*(N+1);
            const double* restrict w = &plan->weight[index_k];

            memset(b_re, 0, (size_t)Q*sizeof(vec_t));  // IEEE 0.0 is all-bits-zero
            memset(b_im, 0, (size_t)Q*sizeof(vec_t));

            // Fold the coefficients w_i exp(-i m theta_i) into their bins. The phase is carried as
            // exp(i n STRIDE theta) (per thread) times exp(i s theta) (per lane, shared), and the
            // conjugate is taken here, exactly as the FFT-tree evaluators do.
            for (int_t i=0; i<=N; i++) {
                const complex_t ph = phase[index_k + i];
                const real_t cr = w[i]*creal(ph);
                const real_t ci = w[i]*cimag(ph);
                const local_t l = lane[index_k + i];
                const int_t d = dest[i];
                b_re[d] += cr*l.cos - ci*l.sin;
                b_im[d] -= cr*l.sin + ci*l.cos;
            }

            poisson_fft(b_re, b_im, plan);  // phi_k at all Q points on the circle

            if (k == 0) {
                memcpy(p_re, b_re, (size_t)Q*sizeof(vec_t));
                memcpy(p_im, b_im, (size_t)Q*sizeof(vec_t));
            } else {
                for (int_t q=0; q<Q; q++) {
                    const vec_t pr = p_re[q];
                    const vec_t pi = p_im[q];
                    p_re[q] = pr*b_re[q] - pi*b_im[q];
                    p_im[q] = pr*b_im[q] + pi*b_re[q];
                }
            }
        }

        // [t^N] by the Q-point trapezoid rule; out carries exp(2 pi i qN/Q)/Q.
        vec_t s_re = VEC_ZERO;
        vec_t s_im = VEC_ZERO;
        for (int_t q=0; q<Q; q++) {
            s_re += out_re[q]*p_re[q] - out_im[q]*p_im[q];
            s_im += out_re[q]*p_im[q] + out_im[q]*p_re[q];
        }

#ifdef USE_SIMD
        for (int_t s=0; s<STRIDE; s++) {
            const int_t m = n*STRIDE + s;
            const complex_t unit = s_re[s] + I*s_im[s];
            const complex_t s2 = s2_re + I*(real_t)m*M_PI;
            complex_t sinc = (1-EXPC(-s2))/s2;
            if ((m == 0) && (ABS(s2_re)<=EPS)) sinc = 1 - s2*(0.5 - s2/6);
            res_arr[m] = plan->resN_exp*(unit*sinc);
        }
#else
        const complex_t s2 = s2_re + I*(real_t)n*M_PI;
        complex_t sinc = (1-EXPC(-s2))/s2;
        if ((n == 0) && (ABS(s2_re)<=EPS)) sinc = 1 - s2*(0.5 - s2/6);
        const complex_t unit = s_re + I*s_im;
        res_arr[n] = plan->resN_exp*(unit*sinc);
#endif
        phase_advance(args, cells);
    }

    free(b_re);
    free(b_im);
    free(p_re);
    free(p_im);
    return 0;
}
