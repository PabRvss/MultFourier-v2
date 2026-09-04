#pragma once

#define USE_DOUBLE_PRECISION
/*
 * USE_SIMD desactivado deliberadamente.
 *
 * El modo SIMD dependia de simde/x86/sse*.h y sse_mathfun/ssemathlib.c,
 * ambos especificos de x86 (SSE/SSE2). Esto rompia la compilacion en
 * ARM64 (Apple Silicon) y anadia dependencias no portables prohibidas
 * por CRAN. El camino escalar de abajo (#ifndef USE_SIMD) ya provee
 * definiciones completas y equivalentes de vec_t / VEC_* para que
 * multinomial.h y fourier.h compilen sin cambios en cualquier
 * arquitectura, dejando el trabajo de vectorizacion al compilador
 * (auto-vectorizacion con -O2), tal como exige el punto 3 de la
 * especificacion.
 */
/* #define USE_SIMD */

#define EXP exp
#define EXPC cexp
#define LOG log

#ifdef USE_DOUBLE_PRECISION

typedef double complex complex_t;
typedef double real_t;
#define ABS fabs
#define MIN fmin
#define MAX fmax

#define MAX_EXP 700
#define CONV_COUNT 30

#else

typedef float complex complex_t;
typedef float real_t;
#define ABS fabsf
#define MIN fminf
#define MAX fmaxf

#define MAX_EXP 85
#define CONV_COUNT 40

#endif

/*
 * Modo escalar (unico soportado): vec_t es simplemente real_t y las
 * macros VEC_* colapsan a sus equivalentes escalares estandar de C.
 * Esto es matematicamente identico al camino SIMD, solo que procesando
 * un elemento a la vez en lugar de STRIDE elementos empaquetados.
 */
#define STRIDE 1
typedef real_t vec_t;
#define VEC_ZERO 0
#define VEC_ONE 1
#define VEC_MAX MAX
#define VEC_EXP EXP
#define VEC_LOG LOG

#if defined(__GNUC__)
#define EXTERNAL __attribute__((visibility("default")))
#define PACK( __Declaration__ ) __Declaration__ __attribute__((__packed__))
#elif defined(_MSC_VER)
#define EXTERNAL __declspec(dllexport)
#define PACK( __Declaration__ ) __pragma( pack(push, 1) ) __Declaration__ __pragma( pack(pop))
#endif

typedef int64_t int_t;

#define EPS 1e-10
static real_t* restrict lgac;

static inline double get_dt(struct timespec start, struct timespec end) {
    return (double)(end.tv_sec-start.tv_sec) + (double)(end.tv_nsec-start.tv_nsec)*1e-9;
}