#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>
#include <stdbool.h>

#include "base.h"
#include "options.h"
#include "rewards.h"
#include "multinomial/multinomial.h"
#include "multinomial/fourier.h"
#include "multinomial/exhaustive_bisection.h"

extern void init_ext(int_t fact_max, int_t verbose_ext);

static SEXP pack_result(const multinomial_result* mult_res) {
    SEXP res   = PROTECT(Rf_allocVector(VECSXP, 10));
    SEXP names = PROTECT(Rf_allocVector(STRSXP, 10));

    SET_VECTOR_ELT(res, 0, Rf_ScalarReal(mult_res->pval));
    SET_STRING_ELT(names, 0, Rf_mkChar("pval"));

    SET_VECTOR_ELT(res, 1, Rf_ScalarInteger((int)mult_res->converged));
    SET_STRING_ELT(names, 1, Rf_mkChar("converged"));

    SET_VECTOR_ELT(res, 2, Rf_ScalarInteger((int)mult_res->terms));
    SET_STRING_ELT(names, 2, Rf_mkChar("terms"));

    SET_VECTOR_ELT(res, 3, Rf_ScalarReal(mult_res->p0));
    SET_STRING_ELT(names, 3, Rf_mkChar("p0"));

    SET_VECTOR_ELT(res, 4, Rf_ScalarReal(mult_res->eval_time));
    SET_STRING_ELT(names, 4, Rf_mkChar("eval_time"));

    SET_VECTOR_ELT(res, 5, Rf_ScalarReal(mult_res->err_rel_est));
    SET_STRING_ELT(names, 5, Rf_mkChar("err_rel_est"));

    SET_VECTOR_ELT(res, 6, Rf_ScalarReal(mult_res->n_eff));
    SET_STRING_ELT(names, 6, Rf_mkChar("n_eff"));

    SET_VECTOR_ELT(res, 7, Rf_ScalarReal(mult_res->nu_max_ratio));
    SET_STRING_ELT(names, 7, Rf_mkChar("nu_max_ratio"));

    SET_VECTOR_ELT(res, 8, Rf_ScalarReal(mult_res->gamma));
    SET_STRING_ELT(names, 8, Rf_mkChar("gamma"));

    SET_VECTOR_ELT(res, 9, Rf_ScalarInteger((int)mult_res->method));
    SET_STRING_ELT(names, 9, Rf_mkChar("method"));

    Rf_setAttrib(res, R_NamesSymbol, names);
    UNPROTECT(2);
    return res;
}

SEXP c_pval_fourier(SEXP r_x, SEXP r_p, SEXP r_fft_prec, SEXP r_precompute, 
                   SEXP r_rel_eps, SEXP r_max_terms, SEXP r_B, SEXP r_threads, SEXP r_verbose) {
    const int_t K = Rf_length(r_x);
    const double* x_in = REAL(r_x);
    const double* p_in = REAL(r_p);

    int_t N = 0;
    int_t* x0 = (int_t*)R_alloc(K, sizeof(int_t));
    real_t* probs = (real_t*)R_alloc(K, sizeof(real_t));
    for (int_t k = 0; k < K; k++) {
        x0[k] = (int_t)x_in[k];
        probs[k] = (real_t)p_in[k];
        N += x0[k];
    }

    init_ext(N + K + 100, 0);

    options_t options;
    set_options_default(&options);

    options.gamma_precision = FFT_REAL;
    int prec = Rf_asInteger(r_fft_prec);
    options.fft_precision = (prec == 2) ? FFT_DD : FFT_REAL;
    options.use_fft_precompute = Rf_asLogical(r_precompute) ? 1 : 0;
    options.error_bound = 1;
    options.enum_cutoff = 0.0;

    options.eps_rel = Rf_asReal(r_rel_eps);
    options.max_iter = Rf_asInteger(r_max_terms);
    options.B = Rf_asInteger(r_B);
    options.verbose = Rf_asLogical(r_verbose) ? 1 : 0;

    int user_threads = Rf_asInteger(r_threads);
    if (user_threads > 2) user_threads = 2;
    if (user_threads < 1) user_threads = 1;
    options.threads = user_threads;

    multinomial_result mult_res;
    memset(&mult_res, 0, sizeof(multinomial_result));
    mult_res.method = 1;
    mult_res.gamma = NA_REAL;

    multinomial* mult = get_mult(N, K, probs, &options);
    fill_mult1(mult, &mult_res, x0, &options);
    fill_mult2(mult, &mult_res, x0, &options);
    mult_res.p0 = mult->p0;
    mult_res.gamma = mult->gamma;

    series(mult, &mult_res, &options);

    free_mult(mult);
    return pack_result(&mult_res);
}

SEXP c_pval_exhaustive(SEXP r_x, SEXP r_p, SEXP r_threads, SEXP r_verbose) {
    const int_t K = Rf_length(r_x);
    const double* x_in = REAL(r_x);
    const double* p_in = REAL(r_p);

    int_t N = 0;
    int_t* x0 = (int_t*)R_alloc(K, sizeof(int_t));
    real_t* probs = (real_t*)R_alloc(K, sizeof(real_t));
    for (int_t k = 0; k < K; k++) {
        x0[k] = (int_t)x_in[k];
        probs[k] = (real_t)p_in[k];
        N += x0[k];
    }

    init_ext(N + K + 100, 0);

    options_t options;
    set_options_default(&options);
    options.verbose = Rf_asLogical(r_verbose) ? 1 : 0;

    int user_threads = Rf_asInteger(r_threads);
    if (user_threads > 2) user_threads = 2;
    if (user_threads < 1) user_threads = 1;
    options.threads = user_threads;

    multinomial_result mult_res;
    memset(&mult_res, 0, sizeof(multinomial_result));
    mult_res.method = 0;
    mult_res.gamma = NA_REAL;

    multinomial* mult = get_mult(N, K, probs, &options);
    fill_mult1(mult, &mult_res, x0, &options);
    mult_res.p0 = mult->p0;

    exhaustive_bisection(mult, &mult_res, &options);

    free_mult(mult);
    return pack_result(&mult_res);
}

SEXP c_pval_flexible(SEXP r_x, SEXP r_p, SEXP r_fft_prec, SEXP r_precompute, 
                   SEXP r_rel_eps, SEXP r_max_terms, SEXP r_B, SEXP r_threads, SEXP r_verbose) {
    const int_t K = Rf_length(r_x);
    const double* x_in = REAL(r_x);
    const double* p_in = REAL(r_p);

    int_t N = 0;
    int_t* x0 = (int_t*)R_alloc(K, sizeof(int_t));
    real_t* probs = (real_t*)R_alloc(K, sizeof(real_t));
    for (int_t k = 0; k < K; k++) {
        x0[k] = (int_t)x_in[k];
        probs[k] = (real_t)p_in[k];
        N += x0[k];
    }

    init_ext(N + K + 100, 0);

    options_t options;
    set_options_default(&options);

    options.gamma_precision = FFT_REAL;
    int prec = Rf_asInteger(r_fft_prec);
    options.fft_precision = (prec == 2) ? FFT_DD : FFT_REAL;
    options.use_fft_precompute = Rf_asLogical(r_precompute) ? 1 : 0;
    options.error_bound = 1;
    options.enum_cutoff = 6.5; 

    options.eps_rel = Rf_asReal(r_rel_eps);
    options.max_iter = Rf_asInteger(r_max_terms);
    options.B = Rf_asInteger(r_B);
    options.verbose = Rf_asLogical(r_verbose) ? 1 : 0;

    int user_threads = Rf_asInteger(r_threads);
    if (user_threads > 2) user_threads = 2;
    if (user_threads < 1) user_threads = 1;
    options.threads = user_threads;

    multinomial_result mult_res;
    memset(&mult_res, 0, sizeof(multinomial_result));

    real_t supp = (lgac[N + K - 1] - lgac[N] - lgac[K - 1]) / LOG(10.0);

    multinomial* mult = get_mult(N, K, probs, &options);
    fill_mult1(mult, &mult_res, x0, &options);
    mult_res.p0 = mult->p0;

    if (supp >= options.enum_cutoff) {
        mult_res.method = 1;
        fill_mult2(mult, &mult_res, x0, &options);
        mult_res.gamma = mult->gamma;
        series(mult, &mult_res, &options);
    } else {
        mult_res.method = 0;
        mult_res.gamma = NA_REAL;
        exhaustive_bisection(mult, &mult_res, &options);
    }

    free_mult(mult);
    return pack_result(&mult_res);
}