#include <R.h>
#include <Rinternals.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <tgmath.h>
#include <time.h>
#include <pthread.h>

#include "base.h"
#include "options.h"
#include "rewards.h"

#include "multinomial.h"
#include "fourier.h"
#include "exhaustive.h"

/* ---------------------------------------------------------------------- */
/* Constantes por defecto                                                  */
/* ---------------------------------------------------------------------- */
#define DEFAULT_REL_EPS      1e-2
#define DEFAULT_ENUM_CUTOFF  6.5
#define DEFAULT_UNDERSAMPLING 1.0
#define DEFAULT_VERBOSE      0
#define DEFAULT_MAX_TERMS    1000000

/* ---------------------------------------------------------------------- */
/* Inicializacion / liberacion del arreglo de log-factoriales             */
/* ---------------------------------------------------------------------- */
static void init_ext(int64_t fact_max, int64_t verbose_ext) {
  if (lgac != NULL) {
    free(lgac);
    lgac = NULL;
  }

  lgac = malloc(fact_max * sizeof(real_t));
  if (lgac == NULL) {
    error("Error: no fue posible reservar memoria para lgac.");
    return;
  }

  lgac[0] = 0;

  double s = 0;
  double d = 0;
  for (size_t i = 1; i <= (size_t) fact_max; i++) {
    double l = log((double) i);
    double y = l - d;
    double t = s + y;
    d = (t - s) - y;
    s = t;
    lgac[i] = (real_t) s;
  }
}

static void finish_ext(void) {
  if (lgac) {
    free(lgac);
    lgac = NULL;
  }
}

/* ---------------------------------------------------------------------- */
/* Nucleo de calculo (sin dependencia de R): identico a la version previa, */
/* pero sin la rama OpenCL (no portable / no soportada por CRAN) y sin    */
/* modulos multinomial_fast/multinomial_cl no utilizados.                 */
/* ---------------------------------------------------------------------- */
static void main_int(
    const int64_t N, const int64_t K,
    const int64_t* x0, const real_t* probs,
    multinomial_result* mult_res,
    options_t* options
) {
  if (x0 == NULL || probs == NULL || mult_res == NULL || options == NULL) {
    error("Error: puntero NULL recibido en main_int.");
    return;
  }

  const int64_t verbose = options->verbose;
  real_t supp = (lgac[N + K - 1] - lgac[N] - lgac[K - 1]) / LOG(10.0);

  if (verbose) {
    printf("p = [");
    for (int64_t k = 0; k < K; k++) printf("%.8f ", probs[k]);
    printf("]\n");
    printf("x0 = [");
    for (int64_t k = 0; k < K; k++) printf("%lld, ", (long long) x0[k]);
    printf("]\n");
    printf("N = %lld\n", (long long) N);
    printf("K = %lld\n", (long long) K);
    printf("log10 support = %.6f\n\n", supp);
  }

  struct timespec t0, t1, t2;
  timespec_get(&t0, TIME_UTC);

  multinomial* mult = get_mult(N, K, probs, options);
  if (mult == NULL) {
    error("Error: no fue posible crear la estructura multinomial.");
    return;
  }

  fill_mult1(mult, x0, options);
  if (supp >= options->enum_cutoff) fill_mult2(mult, x0, options);
  mult_res->p0 = mult->p0;

  timespec_get(&t1, TIME_UTC);
  if (verbose) printf("tiempo de inicializacion: %.6f s\n\n", get_dt(t0, t1));

  if (supp >= options->enum_cutoff) {
    series(mult, mult_res, options);
    timespec_get(&t2, TIME_UTC);
    if (verbose) printf("tiempo serie de Fourier: %.6f s\n", get_dt(t1, t2));
  } else {
    exhaustive(mult, mult_res, options);
    timespec_get(&t2, TIME_UTC);
    if (verbose) printf("tiempo metodo exhaustivo: %.6f s\n", get_dt(t1, t2));
  }

  if (verbose) printf("tiempo total: %.6f s\n\n", get_dt(t0, t2));

  free_mult(mult);
}

static void main_ext(
    const int64_t N, const int64_t K,
    const int64_t* x0, const real_t* probs,
    double* restrict p_value, int64_t* restrict converged,
    int64_t* restrict terms, double* restrict p0,
    real_t rel_eps, int64_t max_terms, real_t undersampling,
    int64_t verbose
) {
  if (p_value == NULL || converged == NULL || terms == NULL || p0 == NULL) {
    error("Error: puntero NULL recibido en main_ext.");
    return;
  }

  multinomial_result* mult_res = malloc(sizeof(multinomial_result));
  if (mult_res == NULL) {
    error("Error: no fue posible reservar memoria para multinomial_result.");
    return;
  }

  options_t options = {0};
  /* En c_pval_flexible() y c_pval_fourier() (llaman a main_ext / main_series): */
  set_options(&options, rel_eps, max_terms, undersampling, DEFAULT_REL_EPS,
              DEFAULT_ENUM_CUTOFF, 1, 1, verbose, 10);   /* threads = 1 */

  main_int(N, K, x0, probs, mult_res, &options);

  *p_value  = mult_res->pval;
  *converged = mult_res->converged;
  *terms    = mult_res->terms;
  *p0       = mult_res->p0;

  free(mult_res);
}

static void main_series(
    const int64_t N, const int64_t K,
    const int64_t* x0, const real_t* probs,
    double* restrict p_value, int64_t* restrict converged,
    int64_t* restrict terms, double* restrict p0,
    real_t rel_eps, int64_t max_terms, real_t undersampling,
    int64_t verbose
) {
  if (p_value == NULL || converged == NULL || terms == NULL || p0 == NULL) {
    error("Error: puntero NULL recibido en main_series.");
    return;
  }

  multinomial_result* mult_res = malloc(sizeof(multinomial_result));
  if (mult_res == NULL) {
    error("Error: no fue posible reservar memoria para multinomial_result.");
    return;
  }

  options_t options = {0};
  /* En main_series() (usada por c_pval_fourier): */
  set_options(&options, rel_eps, max_terms, undersampling, DEFAULT_REL_EPS,
              -INFINITY, 1, 1, verbose, 10);              /* threads = 1 */

  main_int(N, K, x0, probs, mult_res, &options);

  *p_value  = mult_res->pval;
  *converged = mult_res->converged;
  *terms    = mult_res->terms;
  *p0       = mult_res->p0;

  free(mult_res);
}

/* ---------------------------------------------------------------------- */
/* Utilidad interna: construye la lista de resultados con nombres,        */
/* protegiendo correctamente frente al Garbage Collector.                 */
/* ---------------------------------------------------------------------- */
static SEXP build_result_list(double p_value, int64_t converged,
                               int64_t terms, double p0) {
  SEXP result = PROTECT(allocVector(VECSXP, 4));
  SEXP names  = PROTECT(allocVector(STRSXP, 4));

  SET_VECTOR_ELT(result, 0, ScalarReal(p_value));
  SET_VECTOR_ELT(result, 1, ScalarInteger((int) converged));
  SET_VECTOR_ELT(result, 2, ScalarInteger((int) terms));
  SET_VECTOR_ELT(result, 3, ScalarReal(p0));

  SET_STRING_ELT(names, 0, mkChar("pval"));
  SET_STRING_ELT(names, 1, mkChar("converged"));
  SET_STRING_ELT(names, 2, mkChar("terms"));
  SET_STRING_ELT(names, 3, mkChar("p0"));

  setAttrib(result, R_NamesSymbol, names);

  UNPROTECT(2); /* result y names quedan protegidos por el llamador via return */
  return result;
}

/* ---------------------------------------------------------------------- */
/* c_pval_flexible: metodo flexible (exhaustivo o serie segun el soporte) */
/* ---------------------------------------------------------------------- */
SEXP c_pval_flexible(SEXP N, SEXP K, SEXP x0, SEXP probs, SEXP max_terms,
                      SEXP rel_eps, SEXP undersampling, SEXP verbose) {

  int64_t n = (int64_t) REAL(N)[0];
  int64_t k = (int64_t) REAL(K)[0];

  if (n <= 0 || k <= 0) {
    error("Error: N y K deben ser positivos.");
    return R_NilValue;
  }
  if (length(x0) != k || length(probs) != k) {
    error("Error: x0 y probs deben tener longitud K.");
    return R_NilValue;
  }

  real_t rel_eps_val       = REAL(rel_eps)[0];
  int64_t max_terms_val    = (int64_t) REAL(max_terms)[0];
  real_t undersampling_val = REAL(undersampling)[0];
  int64_t verbose_val      = (int64_t) REAL(verbose)[0];

  SEXP x0_r    = PROTECT(coerceVector(x0, REALSXP));
  SEXP probs_r = PROTECT(coerceVector(probs, REALSXP));

  int64_t* x0_ptr = (int64_t*) malloc((size_t) k * sizeof(int64_t));
  if (x0_ptr == NULL) {
    UNPROTECT(2);
    error("Error: no fue posible reservar memoria para x0.");
    return R_NilValue;
  }
  for (int64_t i = 0; i < k; i++) x0_ptr[i] = (int64_t) REAL(x0_r)[i];

  double* probs_ptr = (double*) malloc((size_t) k * sizeof(double));
  if (probs_ptr == NULL) {
    free(x0_ptr);
    UNPROTECT(2);
    error("Error: no fue posible reservar memoria para probs.");
    return R_NilValue;
  }
  for (int64_t i = 0; i < k; i++) probs_ptr[i] = REAL(probs_r)[i];

  init_ext(n + k, 1);

  double p_value; int64_t converged; int64_t terms; double p0;
  main_ext(n, k, x0_ptr, probs_ptr, &p_value, &converged, &terms, &p0,
           rel_eps_val, max_terms_val, undersampling_val, verbose_val);

  free(x0_ptr);
  free(probs_ptr);
  finish_ext();
  UNPROTECT(2); /* x0_r, probs_r */

  SEXP result = PROTECT(build_result_list(p_value, converged, terms, p0));
  UNPROTECT(1);
  return result;
}

/* ---------------------------------------------------------------------- */
/* c_pval_exhaustive: fuerza el metodo exhaustivo                         */
/* ---------------------------------------------------------------------- */
SEXP c_pval_exhaustive(SEXP N, SEXP K, SEXP x0, SEXP probs, SEXP max_time,
                        SEXP verbose) {

  if (length(N) != 1 || length(K) != 1 || length(max_time) != 1 ||
      length(verbose) != 1) {
    error("Error: N, K, max_time y verbose deben ser escalares.");
    return R_NilValue;
  }

  int64_t n = (int64_t) REAL(N)[0];
  int64_t k = (int64_t) REAL(K)[0];
  int64_t verbose_val = (int64_t) REAL(verbose)[0];

  if (n <= 0 || k <= 0) {
    error("Error: N y K deben ser positivos.");
    return R_NilValue;
  }
  if (length(x0) != k || length(probs) != k) {
    error("Error: x0 y probs deben tener longitud K.");
    return R_NilValue;
  }

  SEXP x0_r    = PROTECT(coerceVector(x0, REALSXP));
  SEXP probs_r = PROTECT(coerceVector(probs, REALSXP));

  int64_t* x0_ptr = (int64_t*) malloc((size_t) k * sizeof(int64_t));
  if (x0_ptr == NULL) {
    UNPROTECT(2);
    error("Error: no fue posible reservar memoria para x0.");
    return R_NilValue;
  }
  for (int64_t i = 0; i < k; i++) {
    x0_ptr[i] = (int64_t) REAL(x0_r)[i];
    if (x0_ptr[i] < 0) {
      free(x0_ptr);
      UNPROTECT(2);
      error("Error: todos los valores de x0 deben ser no negativos.");
      return R_NilValue;
    }
  }

  double* probs_ptr = (double*) malloc((size_t) k * sizeof(double));
  if (probs_ptr == NULL) {
    free(x0_ptr);
    UNPROTECT(2);
    error("Error: no fue posible reservar memoria para probs.");
    return R_NilValue;
  }

  double prob_sum = 0.0;
  for (int64_t i = 0; i < k; i++) {
    probs_ptr[i] = REAL(probs_r)[i];
    if (probs_ptr[i] < 0.0 || probs_ptr[i] > 1.0) {
      free(x0_ptr);
      free(probs_ptr);
      UNPROTECT(2);
      error("Error: todos los valores de probs deben estar entre 0 y 1.");
      return R_NilValue;
    }
    prob_sum += probs_ptr[i];
  }

  if (fabs(prob_sum - 1.0) > 1e-8) {
    if (verbose_val) Rprintf("Nota: las probabilidades suman %f, normalizando a 1.\n", prob_sum);
    for (int64_t i = 0; i < k; i++) probs_ptr[i] /= prob_sum;
  }

  init_ext(n + k, verbose_val);

  multinomial_result* mult_res = malloc(sizeof(multinomial_result));
  if (mult_res == NULL) {
    free(x0_ptr);
    free(probs_ptr);
    finish_ext();
    UNPROTECT(2);
    error("Error: no fue posible reservar memoria para el resultado.");
    return R_NilValue;
  }

  options_t options = {0};
  /* En c_pval_exhaustive(): */
  set_options(&options,
              DEFAULT_REL_EPS,
              INT64_MAX,
              1.0,
              DEFAULT_REL_EPS,
              1e308,
              1, 1, verbose_val, 10);                      /* threads = 1 */

  main_int(n, k, x0_ptr, probs_ptr, mult_res, &options);

  double p_value   = mult_res->pval;
  int64_t converged = mult_res->converged;
  int64_t terms     = mult_res->terms;
  double p0         = mult_res->p0;

  free(mult_res);
  free(x0_ptr);
  free(probs_ptr);
  finish_ext();
  UNPROTECT(2); /* x0_r, probs_r */

  SEXP result = PROTECT(build_result_list(p_value, converged, terms, p0));
  UNPROTECT(1);
  return result;
}

/* ---------------------------------------------------------------------- */
/* c_pval_fourier: fuerza el metodo de serie de Fourier                   */
/* ---------------------------------------------------------------------- */
SEXP c_pval_fourier(SEXP N, SEXP K, SEXP x0, SEXP probs, SEXP max_terms,
                     SEXP rel_eps, SEXP undersampling, SEXP verbose) {

  int64_t n = (int64_t) REAL(N)[0];
  int64_t k = (int64_t) REAL(K)[0];

  if (n <= 0 || k <= 0) {
    error("Error: N y K deben ser positivos.");
    return R_NilValue;
  }
  if (length(x0) != k || length(probs) != k) {
    error("Error: x0 y probs deben tener longitud K.");
    return R_NilValue;
  }

  real_t rel_eps_val       = REAL(rel_eps)[0];
  int64_t max_terms_val    = (int64_t) REAL(max_terms)[0];
  real_t undersampling_val = REAL(undersampling)[0];
  int64_t verbose_val      = (int64_t) REAL(verbose)[0];

  SEXP x0_r    = PROTECT(coerceVector(x0, REALSXP));
  SEXP probs_r = PROTECT(coerceVector(probs, REALSXP));

  int64_t* x0_ptr = (int64_t*) malloc((size_t) k * sizeof(int64_t));
  if (x0_ptr == NULL) {
    UNPROTECT(2);
    error("Error: no fue posible reservar memoria para x0.");
    return R_NilValue;
  }
  for (int64_t i = 0; i < k; i++) x0_ptr[i] = (int64_t) REAL(x0_r)[i];

  double* probs_ptr = (double*) malloc((size_t) k * sizeof(double));
  if (probs_ptr == NULL) {
    free(x0_ptr);
    UNPROTECT(2);
    error("Error: no fue posible reservar memoria para probs.");
    return R_NilValue;
  }
  for (int64_t i = 0; i < k; i++) probs_ptr[i] = REAL(probs_r)[i];

  init_ext(n + k, 1);

  double p_value; int64_t converged; int64_t terms; double p0;
  main_series(n, k, x0_ptr, probs_ptr, &p_value, &converged, &terms, &p0,
              rel_eps_val, max_terms_val, undersampling_val, verbose_val);

  free(x0_ptr);
  free(probs_ptr);
  finish_ext();
  UNPROTECT(2); /* x0_r, probs_r */

  SEXP result = PROTECT(build_result_list(p_value, converged, terms, p0));
  UNPROTECT(1);
  return result;
}