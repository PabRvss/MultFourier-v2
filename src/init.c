#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <stdlib.h>

/*
 * Forward declarations (declaraciones adelantadas) de las rutinas .Call
 * implementadas en src/pvalues.c. No se define ningun pvalues.h: init.c
 * solo necesita conocer la firma para poder registrarlas formalmente.
 */
extern SEXP c_pval_flexible(SEXP N, SEXP K, SEXP x0, SEXP probs,
                             SEXP max_terms, SEXP rel_eps,
                             SEXP undersampling, SEXP verbose);

extern SEXP c_pval_exhaustive(SEXP N, SEXP K, SEXP x0, SEXP probs,
                               SEXP max_time, SEXP verbose);

extern SEXP c_pval_fourier(SEXP N, SEXP K, SEXP x0, SEXP probs,
                            SEXP max_terms, SEXP rel_eps,
                            SEXP undersampling, SEXP verbose);

static const R_CallMethodDef CallEntries[] = {
    {"c_pval_flexible",   (DL_FUNC) &c_pval_flexible,   8},
    {"c_pval_exhaustive", (DL_FUNC) &c_pval_exhaustive, 6},
    {"c_pval_fourier",    (DL_FUNC) &c_pval_fourier,    8},
    {NULL, NULL, 0}
};

void R_init_MultFourier(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
    R_forceSymbols(dll, TRUE);
}