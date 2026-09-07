#pragma once

#pragma once

#if defined(USE_SIMD) && (USE_SIMD == 0)
#undef USE_SIMD
#endif

#ifdef USE_SIMD
#define SIMDE_ENABLE_NATIVE_ALIASES
#define SIMDE_FAST_CONVERSION_RANGE
#include <simde/x86/avx512.h>
#endif


#ifdef USE_DOUBLE_PRECISION

#ifdef USE_SIMD

#define STRIDE 8
typedef __m512d vec_t;
#define VEC_ZERO _mm512_setzero_pd()
#define VEC_ONE _mm512_set1_pd(1.0)
#define VEC_MAX _mm512_max_pd
#define VEC_MIN _mm512_min_pd
#define VEC_SET1 _mm512_set1_pd
#define VEC_FLOOR _mm512_floor_pd
#define VEC_SCALEF _mm512_scalef_pd
#define VEC_SQRT _mm512_sqrt_pd

static inline vec_t vec_fma_(vec_t a, vec_t b, vec_t c) {
    simde__m512d_private ap = simde__m512d_to_private(a);
    simde__m512d_private bp = simde__m512d_to_private(b);
    simde__m512d_private cp = simde__m512d_to_private(c);
    simde__m512d_private rp;
    for (int i = 0; i < 4; i++) {
        rp.m128d[i] = _mm_fmadd_pd(ap.m128d[i], bp.m128d[i], cp.m128d[i]);
    }
    return simde__m512d_from_private(rp);
}
#define VEC_FMA vec_fma_

typedef simde__mmask8 vec_mask_t;
#define VEC_CMP_GT(a, b) _mm512_cmp_pd_mask(a, b, SIMDE_CMP_GT_OQ)
#define VEC_CMP_LT(a, b) _mm512_cmp_pd_mask(a, b, SIMDE_CMP_LT_OQ)
#define VEC_CMP_LE(a, b) _mm512_cmp_pd_mask(a, b, SIMDE_CMP_LE_OQ)
#define VEC_CMP_GE(a, b) _mm512_cmp_pd_mask(a, b, SIMDE_CMP_GE_OQ)
#define VEC_CMP_EQ(a, b) _mm512_cmp_pd_mask(a, b, SIMDE_CMP_EQ_OQ)
#define VEC_SELECT(mask, t, f) _mm512_mask_blend_pd(mask, f, t)

#else
#define VEC_MAX fmax
#define VEC_MIN fmin
#define VEC_SET1(x) (x)
#define VEC_FMA fma
#define VEC_FLOOR floor
static inline real_t VEC_SCALEF(real_t a, real_t b) { return ldexp(a, (int)floor(b)); }
#define VEC_SQRT sqrt

typedef int vec_mask_t;
#define VEC_CMP_GT(a, b) ((a) > (b))
#define VEC_CMP_LT(a, b) ((a) < (b))
#define VEC_CMP_LE(a, b) ((a) <= (b))
#define VEC_CMP_GE(a, b) ((a) >= (b))
#define VEC_CMP_EQ(a, b) ((a) == (b))
#define VEC_SELECT(mask, t, f) ((mask) ? (t) : (f))
#endif

#else  // !USE_DOUBLE_PRECISION

#ifdef USE_SIMD

#define STRIDE 16
typedef __m512 vec_t;
#define VEC_ZERO _mm512_setzero_ps()
#define VEC_ONE _mm512_set1_ps(1.0)
#define VEC_MAX _mm512_max_ps

#else
#define VEC_MAX fmaxf
#endif

#endif

#ifndef USE_SIMD

#define STRIDE 1
typedef real_t vec_t;
#define VEC_ZERO 0
#define VEC_ONE 1

#endif
