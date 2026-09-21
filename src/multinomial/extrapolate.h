#pragma once

// Convergence acceleration for the inversion series, by Wynn's epsilon algorithm (the Shanks
// transformation computed in place). See latex/error_bounds.tex for why the raw partial sums are
// slow; the short version:
//
// The terms are Re(res_arr[n]) = Re(A*nuhat(n)*c_n) with the kernel c_n = 1/(gamma*pi/T + i*n*pi)
// known in closed form: a Lorentzian of half-width (gamma/T) in n, falling off as 1/n beyond it.
// Once |nuhat(n)| has decayed into its Wiener plateau, the remainder is therefore governed by the
// kernel rather than by the signal, and it does not decay geometrically -- it rings. Summing
// Re(-i*nuhat(n)/(n*pi)) against a measure supported on [-W, 0] leaves a remainder of the form
//
//     S_L - p ~ (alpha*cos(L*phi) + beta*sin(L*phi))/L + O(1/L^2),
//
// the Gibbs ringing produced by the box kernel's own edges. That is an exactly extrapolable shape:
// a sequence whose error is a few damped oscillations is what the Shanks transformation is built
// for, and each even column of the epsilon table annihilates one more of them.
//
// Two practical notes:
//
//   1. This is NOT the trailing-window average (options->average_window). Averaging partial sums is
//      a linear filter: it damps the ringing but also shifts the mean, and measured against long
//      reference runs it is frequently WORSE than plain truncation. The epsilon algorithm is
//      nonlinear -- it estimates phi from the sequence instead of assuming it -- which is what makes
//      it win by one to two orders of magnitude at the same L.
//
//   2. The epsilon algorithm divides by differences of nearly equal numbers, so a naive
//      implementation is unusable. The form below is QUADPACK's DQELG: the table is capped at
//      EPSALG_LIMEXP entries and shifted, columns are abandoned when a difference collapses to
//      rounding or when the epsilon-inf test detects irregular behaviour, and the reported result is
//      the table entry with the smallest error estimate rather than the deepest column. That
//      safeguarding is the whole difference between a 100x gain and an occasional 300% blunder.
//
// The partial sums are quad_t but are fed in here scaled by a power of two to O(1) and handled in
// double: extrapolation is a relative-accuracy operation, and the enormous exponents that motivated
// quad precision in the first place carry no information for it.
//
// Stopping on this sequence (options->extrapolate >= 2 in series()) needs a third safeguard beyond
// the two above: DQELG's own abserr estimate is itself a difference of nearly-equal numbers (see
// res3la above), so counting consecutive terms where it happens to read small is exactly the kind
// of test a single unlucky rounding can flip. series() instead runs a second, shallow-table
// instance of this same algorithm (epsalg_add with a small limexp) with its raw output smoothed by
// epsalg_median5, and asks whether that smoothed value has stayed flat over a trailing window --
// a question that one outlier term cannot change the answer to, because the window it would have
// to move is checked only once several more terms have aged the outlier out of both the median and
// the window itself.

#define EPSALG_LIMEXP 50
#define EPSALG_EPMACH 2.2204460492503131e-16
#define EPSALG_OFLOW  1.7976931348623157e+308

typedef struct {
    double tab[EPSALG_LIMEXP + 3];  // 1-based; two slots of working space past the live entries
    double res3la[4];               // 1-based; the last three extrapolates, for the error estimate
    int_t n;                        // live entries in tab
    int_t nres;                     // extrapolates produced so far
    double result;                  // best extrapolate seen on the last update
    double abserr;                  // its error estimate (INFINITY until three are available)
} epsalg_t;

static inline void epsalg_init(epsalg_t* restrict e) {
    e->n = 0;
    e->nres = 0;
    e->result = 0;
    e->abserr = INFINITY;
    e->res3la[1] = 0;
    e->res3la[2] = 0;
    e->res3la[3] = 0;
}

// Append one partial sum and re-extrapolate. O(limexp) per call. limexp truncates the table to
// fewer columns than the EPSALG_LIMEXP the struct is sized for; see series() in fourier.h for why
// the stopping decision runs a second, shallower table side by side with the one used for
// reporting -- the deep columns are exactly the ones subject to catastrophic cancellation, so a
// shallow table is the noise-resistant one, at some cost in how much ringing it can cancel.
static void epsalg_add(epsalg_t* restrict e, const double s, const int_t limexp) {
    int_t n = e->n + 1;
    e->tab[n] = s;

    e->nres++;
    double abserr = EPSALG_OFLOW;
    double result = e->tab[n];

    if (n < 3) {
        e->result = result;
        e->abserr = fmax(abserr, 5.0*EPSALG_EPMACH*fabs(result));
        e->n = n;
        return;
    }

    e->tab[n+2] = e->tab[n];
    const int_t newelm = (n-1)/2;
    e->tab[n] = EPSALG_OFLOW;
    const int_t num = n;
    int_t k1 = n;
    int_t converged = 0;

    for (int_t i=1; i<=newelm; i++) {
        const int_t k2 = k1 - 1;
        const int_t k3 = k1 - 2;
        double res = e->tab[k1+2];
        const double e0 = e->tab[k3];
        const double e1 = e->tab[k2];
        const double e2 = res;
        const double e1abs = fabs(e1);

        const double delta2 = e2 - e1;
        const double err2 = fabs(delta2);
        const double tol2 = fmax(fabs(e2), e1abs)*EPSALG_EPMACH;
        const double delta3 = e1 - e0;
        const double err3 = fabs(delta3);
        const double tol3 = fmax(e1abs, fabs(e0))*EPSALG_EPMACH;

        if (err2 <= tol2 && err3 <= tol3) {  // e0, e1, e2 agree to rounding: done
            result = res;
            abserr = err2 + err3;
            converged = 1;
            break;
        }

        const double e3 = e->tab[k1];
        e->tab[k1] = e1;
        const double delta1 = e1 - e3;
        const double err1 = fabs(delta1);
        const double tol1 = fmax(e1abs, fabs(e3))*EPSALG_EPMACH;

        // Two neighbouring entries collapsed: the rest of the table is rounding noise.
        if (err1 <= tol1 || err2 <= tol2 || err3 <= tol3) { n = i+i-1; break; }

        const double ss = 1.0/delta1 + 1.0/delta2 - 1.0/delta3;
        const double epsinf = fabs(ss*e1);
        if (epsinf <= 1e-4) { n = i+i-1; break; }  // irregular behaviour: abandon the deep columns

        res = e1 + 1.0/ss;
        e->tab[k1] = res;
        k1 -= 2;

        const double error = err2 + fabs(res - e2) + err3;
        if (error <= abserr) { abserr = error; result = res; }
    }

    if (!converged) {
        if (n >= limexp) n = 2*(limexp/2) - 1;

        int_t ib = ((num/2)*2 == num) ? 2 : 1;
        const int_t ie = newelm + 1;
        for (int_t i=1; i<=ie; i++) {
            const int_t ib2 = ib + 2;
            e->tab[ib] = e->tab[ib2];
            ib = ib2;
        }
        if (num != n) {
            int_t indx = num - n + 1;
            for (int_t i=1; i<=n; i++) { e->tab[i] = e->tab[indx]; indx++; }
        }

        if (e->nres < 4) {
            e->res3la[e->nres] = result;
            abserr = EPSALG_OFLOW;
        } else {
            abserr = fabs(result - e->res3la[3]) + fabs(result - e->res3la[2])
                   + fabs(result - e->res3la[1]);
            e->res3la[1] = e->res3la[2];
            e->res3la[2] = e->res3la[3];
            e->res3la[3] = result;
        }
    }

    e->result = result;
    e->abserr = fmax(abserr, 5.0*EPSALG_EPMACH*fabs(result));
    e->n = n;
}

// Usable once three extrapolates exist and the table has not gone non-finite.
static inline int epsalg_ready(const epsalg_t* restrict e) {
    return (e->nres >= 4) && isfinite(e->result) && isfinite(e->abserr);
}

// Median of the last 5 values of x (x[0..4]), by insertion sort into a scratch copy.
static inline double epsalg_median5(const double* restrict x) {
    double w[5] = {x[0], x[1], x[2], x[3], x[4]};
    for (int a=1; a<5; a++) {
        const double v = w[a];
        int b = a - 1;
        while (b >= 0 && w[b] > v) { w[b+1] = w[b]; b--; }
        w[b+1] = v;
    }
    return w[2];
}
