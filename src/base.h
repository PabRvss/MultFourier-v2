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
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif
#ifndef M_SQRT2
#define M_SQRT2 1.414213562373095048801688724209698079
#endif

#define USE_DOUBLE_PRECISION
/* Vectorizacion y OpenCL desactivados por politicas de portabilidad CRAN */
//#define USE_SIMD
//#define USE_OPENCL
#define THRESHOLD_PROD 4

#ifdef USE_DOUBLE_PRECISION

typedef double complex complex_t;
typedef double real_t;
#define ABS fabs
#define MIN fmin
#define MAX fmax
#define CSQRT csqrt
#define POW pow
#define EXP exp
#define EXPC cexp
#define LOG log

#define MAX_EXP 700
#define CONV_COUNT 50

#else

typedef float complex complex_t;
typedef float real_t;
#define ABS fabsf
#define MIN fminf
#define MAX fmaxf
#define CSQRT csqrtf
#define POW powf
#define EXP expf
#define EXPC cexpf
#define LOG logf

#define MAX_EXP 85
#define CONV_COUNT 50

#endif

#include "base_vec.h"

#if defined(__GNUC__)
#define EXTERNAL __attribute__((visibility("default")))
#define PACK( __Declaration__ ) __Declaration__ __attribute__((__packed__))
#elif defined(_MSC_VER)
#define EXTERNAL __declspec(dllexport)
#define PACK( __Declaration__ ) __pragma( pack(push, 1) ) __Declaration__ __pragma( pack(pop))
#endif

typedef int64_t int_t;

#define EPS 1e-10
extern real_t* lgac;

#ifdef USE_DOUBLE_PRECISION
static inline int real_isnan(const real_t x) {
    uint64_t bits;
    memcpy(&bits, &x, sizeof(bits));
    return (bits & 0x7fffffffffffffffULL) > 0x7ff0000000000000ULL;
}
#else
static inline int real_isnan(const real_t x) {
    uint32_t bits;
    memcpy(&bits, &x, sizeof(bits));
    return (bits & 0x7fffffffUL) > 0x7f800000UL;
}
#endif

static inline double get_dt(struct timespec start, struct timespec end) {
    return (double)(end.tv_sec-start.tv_sec) + (double)(end.tv_nsec-start.tv_nsec)*1e-9;
}