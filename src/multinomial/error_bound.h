#pragma once

// Truncation-error diagnostics for the inversion series. See latex/error_bounds.tex for the
// derivation; the short version:
//
// The `sinc` factor the evaluators multiply in is exactly the two-sided Laplace transform of a
// uniform smoothing variable: sinc(z) = (1 - exp(-zW))/(zW) = E[exp(-zU)] for U ~ Uniform[0,W],
// independent of S. So res_arr[n] = M_S(z_n)*sinc(z_n) = M_{S+U}(z_n), and the series is a
// trapezoid-rule inverse Laplace transform recovering the DENSITY of S+U at zero:
//
//     p = P(S <= 0) = W * f_{S+U}(0),
//
// exactly, because f_{S+U}(0) = (1/W) * P(-W <= S <= 0). Convolving with U is what turns the
// atomic S into something with an honest density, spreading each point mass rightwards over a box
// of width W; W >= -s_min makes that box cover the event, and W >= s_max kills the periodization
// image at 2W (both enforced by fill_interval's W_max = max(s_max, -s_min)).
//
// So M_S(z_n) = res_arr[n]/sinc(z_n) is a deconvolution: divide out the known, data-independent
// kernel transform to recover the transform of the signal. Its decay is what governs everything
// below. The 1/n floor on |sinc| is the Fourier decay of the box kernel -- the box's own edges are
// the discontinuities that produce the ringing, not any jump in the statistic.
//
// Three things follow, and all are computed here:
//
//   1. nu >= 0 forces |nuhat(n)| <= nuhat(0) = M(gamma) for every n. That is a free, rigorous
//      a-posteriori check on the evaluator itself: any violation is a numerical bug, not a
//      convergence issue. (Verified to catch the endpoint-chord tilt bug by ~19 orders of
//      magnitude at n=0.)
//
//   2. Wiener's theorem pins the root-mean-square of nuhat(n) to M(gamma)/sqrt(N_eff), where
//      N_eff is the participation ratio of the tilted measure's atoms -- the precise form of
//      "a large support with small probabilities behaves like a regular measure". Combined with
//      sum_{n>N} |c_n|^2 this yields an estimate of the truncation error from terms already
//      computed. IMPORTANT: the RMS must be taken in the trailing window of n, not over all
//      n <= N. At small n the coefficients are still in their smooth initial decay, so averaging
//      from n=1 measures that decay rather than the plateau and makes N_eff grow spuriously with
//      the number of terms averaged.
//
//   3. A late revival of |nuhat(n)| towards nuhat(0) signals a lattice/near-lattice support, where
//      the coefficients stop cancelling and the series stalls. This is the failure mode of the
//      chi-squared (lambda=1) statistic, whose rewards are quadratic in the counts with rationally
//      related coefficients. The revival detector deliberately ignores small n, where
//      |nuhat(n)|/nuhat(0) ~ 1 trivially because nuhat is continuous and peaks at n=0.
//
// Everything here is read-only with respect to the series result: it never changes the p-value.

typedef struct {
    complex_t* restrict nu;   // nuhat(n) = M_S(z_n) for n = 0 .. n_last, complex
    int_t cap;
    int_t n_last;
} err_diag_t;

// The sinc factor the evaluators multiply into res_arr[n]; same formula, kept in one place.
static inline complex_t series_sinc(const int_t n, const real_t gamma, const real_t T) {
    const real_t s2_re = M_PI*gamma/T;
    const complex_t s2 = s2_re + I*((real_t)n*M_PI);
    if ((n == 0) && (ABS(s2_re) <= EPS)) return 1 - s2*(0.5 - s2/6);
    return (1 - EXPC(-s2))/s2;
}

static void err_diag_init(err_diag_t* restrict d, const int_t cap) {
    d->nu = malloc((size_t)cap*sizeof(complex_t));
    d->cap = cap;
    d->n_last = -1;
}

static void err_diag_free(err_diag_t* restrict d) {
    free(d->nu);
    d->nu = NULL;
}

// Accumulate over res_arr[n_lo .. n_hi-1]. Must be called on the RAW terms, before series()
// halves res_arr[0].
static void err_diag_accum(
        err_diag_t* restrict d, const qcomplex_t* restrict res_arr,
        const int_t n_lo, const int_t n_hi, const real_t gamma, const real_t T
) {
    for (int_t n=n_lo; n<n_hi && n<d->cap; n++) {
        const complex_t s = series_sinc(n, gamma, T);
        d->nu[n] = cabs(s) > 0 ? res_arr[n]/s : 0.0;
        d->n_last = n;
    }
}

// RMS of |nuhat(n)| over the trailing window [n_last/2, n_last] -- the frequencies that actually
// govern the tail, and the only ones representative of the Wiener plateau.
static double err_rms_tail(const err_diag_t* restrict d) {
    if (d->n_last < 3) return NAN;
    const int_t lo = d->n_last/2 > 1 ? d->n_last/2 : 1;
    double s = 0;
    int_t c = 0;
    for (int_t n=lo; n<=d->n_last; n++) { s += cabs(d->nu[n])*cabs(d->nu[n]); c++; }
    return c > 0 ? sqrt(s/(double)c) : NAN;
}

// Participation ratio of the tilted measure's atoms, via Wiener's theorem:
// RMS_n |nuhat(n)| = nuhat(0)/sqrt(N_eff).
static double err_n_eff(const err_diag_t* restrict d) {
    if (d->n_last < 3 || !(cabs(d->nu[0]) > 0)) return NAN;
    const double rms = err_rms_tail(d);
    if (!(rms > 0)) return INFINITY;
    return (cabs(d->nu[0])/rms)*(cabs(d->nu[0])/rms);
}

// Largest |nuhat(n)|/nuhat(0) over the upper three quarters of the computed range: a value near 1
// here means the coefficients have revived rather than continued to cancel (lattice support).
static double err_max_revival(const err_diag_t* restrict d, int_t* restrict n_at) {
    if (n_at) *n_at = -1;
    if (d->n_last < 4 || !(cabs(d->nu[0]) > 0)) return NAN;
    double m = 0;
    for (int_t n=d->n_last/4; n<=d->n_last; n++) {
        const double r = cabs(d->nu[n])/cabs(d->nu[0]);
        if (r > m) { m = r; if (n_at) *n_at = n; }
    }
    return m;
}

// sum_{n>N} |c_n|^2 with c_n = sinc_n/2. |c_n| -> 1/(2*pi*n) since W*T = pi, so the far tail is
// summed analytically rather than term by term.
static double err_tail_c2(const int_t N, const real_t gamma, const real_t T) {
    const int_t cap = N + 50000;
    double s = 0;
    for (int_t n=N+1; n<=cap; n++) {
        const complex_t c = series_sinc(n, gamma, T)/2.0;
        s += creal(c)*creal(c) + cimag(c)*cimag(c);
    }
    return s + 1.0/(4.0*M_PI*M_PI*(double)cap);
}

// Estimated relative truncation error: RMS|nuhat| over the trailing window, times
// sqrt(sum_{n>N}|c_n|^2), scaled by sqrt(2) for the conjugate-symmetric pairing, divided by p.
// This is a magnitude estimate under a random-phase model for the unseen coefficients, NOT a
// certified bound -- it is exactly the quantity that a late revival invalidates, which is why
// err_max_revival is reported alongside it.
static double err_rel_estimate(const err_diag_t* restrict d, const double p, const real_t gamma, const real_t T) {
    if (d->n_last < 3 || !(fabs(p) > 0)) return NAN;
    const double rms = err_rms_tail(d);
    if (!(rms >= 0)) return NAN;
    return M_SQRT2*rms*sqrt(err_tail_c2(d->n_last, gamma, T))/fabs(p);
}

// ---------------------------------------------------------------------------------------------
// The certified Selberg enclosure width.
//
// Selberg's majorant/minorant of an interval indicator 1_[a,b] differ by exactly a pair of Fejer
// kernels sited at the two endpoints (this follows from Vaaler's |V_N - psi| <= F_N/(2N+2) and the
// identity 1_[a,b] = (b-a) + psi(x-b) - psi(x-a)):
//
//     p_I - q_I = ( F_N(x-a) + F_N(x-b) ) / (N+1),      F_N = Fejer kernel, mean 1, F_N >= 0.
//
// Our test function f(s) = exp(gamma*s)*1{s<=0} is not an indicator, but it is a layer cake of
// them: f = INTEGRAL_0^1 1_{I_t} dt with I_t = (log(t)/gamma, 0] clipped to [-W, 0]. Averaging
// Selberg's pair over t gives an admissible pair for f, so the certified enclosure width is
//
//     width = INTEGRAL (P - Q) dnu
//           = ( Phi(0) + INTEGRAL_0^1 Phi(a(t)) dt ) / (N+1),      a(t) = max(-W, log(t)/gamma),
//
// where Phi(c) := INTEGRAL F_N(s-c) dnu(s) = sum_n (1 - |n|/(N+1)) exp(i*n*T*c) nuhat(n).
//
// Phi is computable from the stored coefficients, is real (nuhat(-n) = conj(nuhat(n))), and is
// >= 0 for every c since F_N >= 0 and nu >= 0 -- another free self-check on the computation.
// The width is the exact amount by which N moments fail to determine p: if it exceeds p, the
// certificate cannot even establish the sign, no matter how accurate the series itself is.

static double selberg_phi(const err_diag_t* restrict d, const double c, const real_t T) {
    const int_t N = d->n_last;
    if (N < 1) return NAN;
    const double Np1 = (double)(N + 1);
    double acc = creal(d->nu[0]);                    // n = 0 term, weight 1
    for (int_t n=1; n<=N; n++) {
        const double w = 1.0 - (double)n/Np1;
        const complex_t term = cexp(I*(double)n*(double)T*c)*d->nu[n];
        acc += 2.0*w*creal(term);                    // n and -n, using nuhat(-n) = conj(nuhat(n))
    }
    return acc;
}

// Width of the certified enclosure. quad_pts controls the layer-cake quadrature in t.
static double selberg_width(const err_diag_t* restrict d, const real_t gamma, const real_t T, const int_t quad_pts) {
    const int_t N = d->n_last;
    if (N < 1) return NAN;
    const double W = M_PI/(double)T;

    // INTEGRAL_0^1 Phi(a(t)) dt by the midpoint rule; a(t) = max(-W, log(t)/gamma).
    double integral = 0;
    for (int_t k=0; k<quad_pts; k++) {
        const double t = ((double)k + 0.5)/(double)quad_pts;
        double a = log(t)/(double)gamma;
        if (a < -W) a = -W;
        integral += selberg_phi(d, a, T);
    }
    integral /= (double)quad_pts;

    return (selberg_phi(d, 0.0, T) + integral)/(double)(N + 1);
}
