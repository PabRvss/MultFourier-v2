#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <stdlib.h>

extern void free_ext(void);

extern SEXP c_pval_fourier(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP c_pval_exhaustive(SEXP, SEXP, SEXP, SEXP);
extern SEXP c_pval_flexible(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);

static const R_CallMethodDef CallEntries[] = {
    {"c_pval_fourier",    (DL_FUNC) &c_pval_fourier,    9},
    {"c_pval_exhaustive", (DL_FUNC) &c_pval_exhaustive, 4},
    {"c_pval_flexible",   (DL_FUNC) &c_pval_flexible,   9},
    {NULL, NULL, 0}
};

void R_init_MultFourier(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, 0);
    R_forceSymbols(dll, 1);
}

void R_unload_MultFourier(DllInfo *dll) {
    free_ext();
}