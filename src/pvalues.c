#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>
#include <stdbool.h>
#include <string.h>

#include "base.h"
#include "portability.h"
#include "options.h"
#include "rewards.h"
#include "multinomial/multinomial.h"
#include "multinomial/fourier.h"
#include "multinomial/exhaustive_bisection.h"

extern void init_ext(int_t fact_max, int_t verbose_ext);

/* Known option keys whitelist to prevent silent typo failures */
static const char* KNOWN_KEYS[] = {
    "method", "stat", "lambda", "undersampling", "avg_window", "average_window",
    "avg_flat", "average_flat", "rel_eps", "max_terms", "B", "n_threads", "threads",
    "precision", "precompute", "max_time", "verbose", "print_freq", "enum_cutoff",
    "eps_gamma", "error_bound"
};
static const int N_KNOWN_KEYS = sizeof(KNOWN_KEYS) / sizeof(KNOWN_KEYS[0]);

static void validate_known_keys(SEXP opts) {
    if (opts == R_NilValue || !Rf_isNewList(opts)) return;
    SEXP names = Rf_getAttrib(opts, R_NamesSymbol);
    if (names == R_NilValue) return;
    int n = Rf_length(opts);
    for (int i = 0; i < n; i++) {
        const char* key = CHAR(STRING_ELT(names, i));
        if (strlen(key) == 0) continue;
        int found = 0;
        for (int j = 0; j < N_KNOWN_KEYS; j++) {
            if (strcmp(key, KNOWN_KEYS[j]) == 0) {
                found = 1;
                break;
            }
        }
        if (!found) {
            Rf_warning("multfourier: opcion desconocida ignorada: '%s' (revisa el nombre)", key);
        }
    }
}

static SEXP get_list_element(SEXP list, const char* name) {
    if (list == R_NilValue || !Rf_isNewList(list)) return R_NilValue;
    SEXP names = Rf_getAttrib(list, R_NamesSymbol);
    if (names == R_NilValue) return R_NilValue;
    int n = Rf_length(list);
    for (int i = 0; i < n; i++) {
        if (strcmp(CHAR(STRING_ELT(names, i)), name) == 0) {
            return VECTOR_ELT(list, i);
        }
    }
    return R_NilValue;
}

static double get_opt_double(SEXP list, const char* name, double def_val) {
    SEXP el = get_list_element(list, name);
    if (el == R_NilValue || Rf_length(el) == 0) return def_val;
    return Rf_asReal(el);
}

static int get_opt_int(SEXP list, const char* name, int def_val) {
    SEXP el = get_list_element(list, name);
    if (el == R_NilValue || Rf_length(el) == 0) return def_val;
    return Rf_asInteger(el);
}

static int get_opt_bool(SEXP list, const char* name, int def_val) {
    SEXP el = get_list_element(list, name);
    if (el == R_NilValue || Rf_length(el) == 0) return def_val;
    return Rf_asLogical(el) ? 1 : 0;
}

static const char* get_opt_str(SEXP list, const char* name, const char* def_val) {
    SEXP el = get_list_element(list, name);
    if (el == R_NilValue || Rf_length(el) == 0) return def_val;
    return CHAR(STRING_ELT(el, 0));
}

static SEXP pack_result(const multinomial_result* mult_res) {
    SEXP res   = PROTECT(Rf_allocVector(VECSXP, 12));
    SEXP names = PROTECT(Rf_allocVector(STRSXP, 12));

    SET_VECTOR_ELT(res, 0, Rf_ScalarReal(mult_res->pval));
    SET_STRING_ELT(names, 0, Rf_mkChar("pval"));

    SET_VECTOR_ELT(res, 1, Rf_ScalarInteger((int)mult_res->converged));
    SET_STRING_ELT(names, 1, Rf_mkChar("converged"));

    SET_VECTOR_ELT(res, 2, Rf_ScalarInteger((int)mult_res->terms));
    SET_STRING_ELT(names, 2, Rf_mkChar("terms"));

    SET_VECTOR_ELT(res, 3, Rf_ScalarReal(mult_res->eval_time));
    SET_STRING_ELT(names, 3, Rf_mkChar("eval_time"));

    SET_VECTOR_ELT(res, 4, Rf_ScalarReal(mult_res->total_time));
    SET_STRING_ELT(names, 4, Rf_mkChar("total_time"));

    SET_VECTOR_ELT(res, 5, Rf_ScalarReal(mult_res->gamma));
    SET_STRING_ELT(names, 5, Rf_mkChar("gamma"));

    SET_VECTOR_ELT(res, 6, Rf_ScalarReal(mult_res->W));
    SET_STRING_ELT(names, 6, Rf_mkChar("W"));

    SET_VECTOR_ELT(res, 7, Rf_ScalarInteger((int)mult_res->status));
    SET_STRING_ELT(names, 7, Rf_mkChar("status"));

    SET_VECTOR_ELT(res, 8, Rf_ScalarInteger((int)mult_res->method));
    SET_STRING_ELT(names, 8, Rf_mkChar("method"));

    SET_VECTOR_ELT(res, 9, Rf_ScalarReal(mult_res->err_rel_est));
    SET_STRING_ELT(names, 9, Rf_mkChar("err_rel_est"));

    SET_VECTOR_ELT(res, 10, Rf_ScalarReal(mult_res->n_eff));
    SET_STRING_ELT(names, 10, Rf_mkChar("n_eff"));

    SET_VECTOR_ELT(res, 11, Rf_ScalarReal(mult_res->nu_max_ratio));
    SET_STRING_ELT(names, 11, Rf_mkChar("nu_max_ratio"));

    Rf_setAttrib(res, R_NamesSymbol, names);
    UNPROTECT(2);
    return res;
}

SEXP c_run_multfourier(SEXP r_x, SEXP r_p, SEXP r_opts) {
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

    /* Validate options keys against known whitelist */
    validate_known_keys(r_opts);

    options_t options;
    set_options_default(&options);

    const char* method_str = get_opt_str(r_opts, "method", "fourier");

    /* Lambda parameter for power divergence statistics */
    options.lambda = get_opt_double(r_opts, "lambda", 0.0);

    /* Undersampling aliasing factor (> 0) */
    options.undersampling = get_opt_double(r_opts, "undersampling", 1.0);
    if (options.undersampling <= 0.0) options.undersampling = 1.0;

    /* Windowed averaging parameters */
    SEXP r_aw = get_list_element(r_opts, "avg_window");
    if (r_aw == R_NilValue) r_aw = get_list_element(r_opts, "average_window");
    if (r_aw != R_NilValue && Rf_length(r_aw) > 0) {
        options.average_window = Rf_asReal(r_aw);
        if (options.average_window < 0.0) options.average_window = 0.0;
        if (options.average_window > 1.0) options.average_window = 1.0;
    }

    SEXP r_af = get_list_element(r_opts, "avg_flat");
    if (r_af == R_NilValue) r_af = get_list_element(r_opts, "average_flat");
    if (r_af != R_NilValue && Rf_length(r_af) > 0) {
        options.average_flat = Rf_asLogical(r_af) ? 1 : 0;
    }

    options.eps_rel = get_opt_double(r_opts, "rel_eps", 1e-3);
    options.max_iter = get_opt_int(r_opts, "max_terms", 10000);
    options.B = get_opt_int(r_opts, "B", 30);

    /* Thread count */
    SEXP r_th = get_list_element(r_opts, "n_threads");
    if (r_th == R_NilValue) r_th = get_list_element(r_opts, "threads");
    int user_threads = (r_th != R_NilValue && Rf_length(r_th) > 0) ? Rf_asInteger(r_th) : 1;
    if (user_threads < 1) user_threads = 1;
    options.threads = user_threads;

    options.max_time = get_opt_double(r_opts, "max_time", -1.0);
    options.verbose = get_opt_bool(r_opts, "verbose", 0);
    options.use_fft_precompute = get_opt_bool(r_opts, "precompute", 1);
    options.error_bound = get_opt_bool(r_opts, "error_bound", 1);

    const char* prec_str = get_opt_str(r_opts, "precision", "double");
    if (strcmp(prec_str, "double-double") == 0 || strcmp(prec_str, "dd") == 0) {
        options.fft_precision = FFT_DD;
    } else if (strcmp(prec_str, "quad") == 0) {
#if !QUAD_AVAILABLE
        Rf_warning("multfourier: precision 'quad' no disponible en esta plataforma/compilador; utilizando 'double'.");
        options.fft_precision = FFT_REAL;
#else
        options.fft_precision = FFT_QUAD;
#endif
    } else {
        options.fft_precision = FFT_REAL;
    }
    options.gamma_precision = FFT_REAL;

    struct timespec t0, t1;
    timespec_get(&t0, TIME_UTC);

    multinomial_result mult_res;
    memset(&mult_res, 0, sizeof(multinomial_result));
    mult_res.gamma = NA_REAL;
    mult_res.W = NA_REAL;
    mult_res.status = 0;

    multinomial* mult = get_mult(N, K, probs, &options);
    fill_mult1(mult, &mult_res, x0, &options);
    mult_res.p0 = mult->p0;

    int run_fourier = 0;
    if (strcmp(method_str, "fourier") == 0) {
        run_fourier = 1;
    } else if (strcmp(method_str, "exhaustive") == 0) {
        run_fourier = 0;
    } else { /* "flexible" */
        real_t cutoff = get_opt_double(r_opts, "enum_cutoff", 6.5);
        real_t supp = (lgac[N + K - 1] - lgac[N] - lgac[K - 1]) / LOG(10.0);
        run_fourier = (supp >= cutoff) ? 1 : 0;
    }

    if (run_fourier) {
        mult_res.method = 1;
        fill_mult2(mult, &mult_res, x0, &options);
        mult_res.gamma = mult->gamma;
        mult_res.W = mult->W;
        if (mult_res.status != 3) {
            series(mult, &mult_res, &options);
        }
    } else {
        mult_res.method = 0;
        mult_res.gamma = NA_REAL;
        mult_res.W = NA_REAL;
        exhaustive_bisection(mult, &mult_res, &options);
    }

    free_mult(mult);

    timespec_get(&t1, TIME_UTC);
    mult_res.total_time = get_dt(t0, t1);

    return pack_result(&mult_res);
}