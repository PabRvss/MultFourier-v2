#pragma once

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE
#endif

#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <complex.h>

#if defined(__GNUC__) && !defined(__clang__) && defined(ENABLE_QUADMATH)
    #include <quadmath.h>
    typedef __float128 quad_t;
    typedef __complex128 qcomplex_t;
    #define QUAD_AVAILABLE 1
    #define exp_quad expq
    #define fabsq_quad fabsq
    #define crealq_quad crealq
    #define finiteq_quad finiteq
    #define isnanq_quad isnanq
#else
    /* On Apple Silicon / macOS / Clang or when ENABLE_QUADMATH is not set:
       long double on ARM64 is identical to double (64-bit).
       We do not pretend quad precision exists; we fall back cleanly to double. */
    typedef double quad_t;
    typedef double complex qcomplex_t;
    #define QUAD_AVAILABLE 0
    #define exp_quad exp
    #define fabsq_quad fabs
    #define crealq_quad creal
    #define finiteq_quad isfinite
    #define isnanq_quad isnan

    /* Fallback macros for raw quadmath identifiers */
    #ifndef expq
    #define expq exp
    #endif
    #ifndef fabsq
    #define fabsq fabs
    #endif
    #ifndef crealq
    #define crealq creal
    #endif
    #ifndef cimagq
    #define cimagq cimag
    #endif
    #ifndef cabsq
    #define cabsq cabs
    #endif
    #ifndef finiteq
    #define finiteq isfinite
    #endif
    #ifndef isnanq
    #define isnanq isnan
    #endif
    #ifndef scalbnq
    #define scalbnq scalbn
    #endif
    #ifndef frexpq
    #define frexpq frexp
    #endif
    #ifndef quadmath_snprintf
    #define quadmath_snprintf(buf, len, fmt, val) snprintf(buf, len, "%.10e", (double)(val))
    #endif
#endif
