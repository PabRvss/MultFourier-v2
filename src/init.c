#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <stdlib.h>

extern void free_ext(void);

extern SEXP c_run_multfourier(SEXP, SEXP, SEXP);

static const R_CallMethodDef CallEntries[] = {
    {"c_run_multfourier", (DL_FUNC) &c_run_multfourier, 3},
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