#pragma once

#include <math.h>

typedef struct {
    double hi;
    double lo;
} dd_t;

static inline dd_t dd_from_double(double x) { return (dd_t){x, 0.0}; }
static inline double dd_to_double(dd_t a) { return a.hi + a.lo; }
static inline dd_t dd_neg(dd_t a) { return (dd_t){-a.hi, -a.lo}; }

static inline dd_t two_sum(double a, double b) {
    double s = a + b;
    double bb = s - a;
    double err = (a - (s - bb)) + (b - bb);
    return (dd_t){s, err};
}

static inline dd_t quick_two_sum(double a, double b) {  // requires |a| >= |b|
    double s = a + b;
    double err = b - (s - a);
    return (dd_t){s, err};
}

static inline dd_t two_prod(double a, double b) {
    double p = a * b;
    double err = fma(a, b, -p);
    return (dd_t){p, err};
}

static inline dd_t dd_add(dd_t a, dd_t b) {
    dd_t s = two_sum(a.hi, b.hi);
    dd_t t = two_sum(a.lo, b.lo);
    s.lo += t.hi;
    s = quick_two_sum(s.hi, s.lo);
    s.lo += t.lo;
    s = quick_two_sum(s.hi, s.lo);
    return s;
}

static inline dd_t dd_add_d(dd_t a, double b) {
    dd_t s = two_sum(a.hi, b);
    s.lo += a.lo;
    return quick_two_sum(s.hi, s.lo);
}

static inline dd_t dd_sub(dd_t a, dd_t b) { return dd_add(a, dd_neg(b)); }

static inline dd_t dd_mul(dd_t a, dd_t b) {
    dd_t p = two_prod(a.hi, b.hi);
    p.lo += a.hi*b.lo + a.lo*b.hi;
    return quick_two_sum(p.hi, p.lo);
}

static inline dd_t dd_mul_d(dd_t a, double b) {
    dd_t p = two_prod(a.hi, b);
    p.lo += a.lo*b;
    return quick_two_sum(p.hi, p.lo);
}

static inline dd_t dd_div(dd_t a, dd_t b) {
    double q1 = a.hi / b.hi;
    dd_t r = dd_sub(a, dd_mul_d(b, q1));
    double q2 = r.hi / b.hi;
    r = dd_sub(r, dd_mul_d(b, q2));
    double q3 = r.hi / b.hi;
    dd_t q = quick_two_sum(q1, q2);
    return dd_add_d(q, q3);
}

static inline dd_t dd_sqrt(dd_t a) {
    if (a.hi == 0.0) return (dd_t){0.0, 0.0};
    double x = 1.0/sqrt(a.hi);
    double ax = a.hi*x;
    dd_t axdd = dd_from_double(ax);
    dd_t diff = dd_sub(a, dd_mul(axdd, axdd));
    return dd_add(axdd, dd_from_double(diff.hi*(x*0.5)));
}

static const dd_t DD_LOG2 = {0.69314718055994528623, 2.3190468138462995584e-17};

static const dd_t DD_INV_FACT_3  = {0.16666666666666665741, 9.2518585385429706566e-18};
static const dd_t DD_INV_FACT_4  = {0.041666666666666664354, 2.3129646346357426642e-18};
static const dd_t DD_INV_FACT_5  = {0.0083333333333333332177, 1.1564823173178713802e-19};
static const dd_t DD_INV_FACT_6  = {0.0013888888888888889419, -5.3005439543735770591e-20};
static const dd_t DD_INV_FACT_7  = {0.00019841269841269841253, 1.7209558293420705287e-22};
static const dd_t DD_INV_FACT_8  = {2.4801587301587301566e-05, 2.1511947866775881608e-23};
static const dd_t DD_INV_FACT_9  = {2.7557319223985892511e-06, -1.858393274046472081e-22};
static const dd_t DD_INV_FACT_10 = {2.7557319223985888276e-07, 2.3767714622250297319e-23};
static const dd_t DD_INV_FACT_11 = {2.5052108385441720224e-08, -1.448814070935911966e-24};
static const dd_t DD_INV_FACT_12 = {2.0876756987868100187e-09, -1.2073450591132599717e-25};
static const dd_t DD_INV_FACT[] = {
    DD_INV_FACT_3, DD_INV_FACT_4, DD_INV_FACT_5, DD_INV_FACT_6,
    DD_INV_FACT_7, DD_INV_FACT_8, DD_INV_FACT_9, DD_INV_FACT_10,
    DD_INV_FACT_11, DD_INV_FACT_12,
};

static dd_t dd_exp(double a_d) {
    if (a_d <= -709.0) return (dd_t){0.0, 0.0};
    if (a_d >= 709.0) return (dd_t){INFINITY, 0.0};
    if (a_d == 0.0) return (dd_t){1.0, 0.0};

    const double inv_k = 1.0/512.0;

    double m = floor(a_d/DD_LOG2.hi + 0.5);
    dd_t r = dd_mul_d(dd_sub(dd_from_double(a_d), dd_mul_d(DD_LOG2, m)), inv_k);

    dd_t p = dd_mul(r, r);
    dd_t s = dd_add(r, dd_mul_d(p, 0.5));
    p = dd_mul(p, r);
    dd_t t = dd_mul(p, DD_INV_FACT[0]);
    int i = 0;
    do {
        s = dd_add(s, t);
        p = dd_mul(p, r);
        i++;
        t = dd_mul(p, DD_INV_FACT[i]);
    } while (fabs(t.hi) > inv_k*1e-32 && i < 9);
    s = dd_add(s, t);

    for (i = 0; i < 9; i++) {
        s = dd_add(dd_mul_d(s, 2.0), dd_mul(s, s));
    }
    s = dd_add_d(s, 1.0);

    return (dd_t){ldexp(s.hi, (int)m), ldexp(s.lo, (int)m)};
}

typedef struct {
    dd_t re;
    dd_t im;
} ddc_t;

static inline ddc_t ddc_add(ddc_t a, ddc_t b) { return (ddc_t){dd_add(a.re, b.re), dd_add(a.im, b.im)}; }
static inline ddc_t ddc_sub(ddc_t a, ddc_t b) { return (ddc_t){dd_sub(a.re, b.re), dd_sub(a.im, b.im)}; }
static inline ddc_t ddc_neg(ddc_t a) { return (ddc_t){dd_neg(a.re), dd_neg(a.im)}; }

static inline ddc_t ddc_mul(ddc_t a, ddc_t b) {
    return (ddc_t){
        dd_sub(dd_mul(a.re, b.re), dd_mul(a.im, b.im)),
        dd_add(dd_mul(a.re, b.im), dd_mul(a.im, b.re)),
    };
}

static inline ddc_t ddc_scale(dd_t s, ddc_t a) {
    return (ddc_t){dd_mul(s, a.re), dd_mul(s, a.im)};
}

static inline ddc_t ddc_reciprocal(ddc_t a) {
    dd_t mag2 = dd_add(dd_mul(a.re, a.re), dd_mul(a.im, a.im));
    return (ddc_t){dd_div(a.re, mag2), dd_neg(dd_div(a.im, mag2))};
}

static inline dd_t ddc_abs(ddc_t a) {
    return dd_sqrt(dd_add(dd_mul(a.re, a.re), dd_mul(a.im, a.im)));
}

static inline ddc_t ddc_sqrt(ddc_t a) {
    dd_t mag2 = dd_add(dd_mul(a.re, a.re), dd_mul(a.im, a.im));
    dd_t mag = dd_sqrt(mag2);
    if (a.re.hi > 0.0 || (a.re.hi == 0.0 && a.re.lo >= 0.0)) {
        dd_t u = dd_sqrt(dd_mul_d(dd_add(mag, a.re), 0.5));
        if (u.hi == 0.0 && u.lo == 0.0) return (ddc_t){u, u};
        dd_t v = dd_div(a.im, dd_mul_d(u, 2.0));
        return (ddc_t){u, v};
    } else {
        dd_t v = dd_sqrt(dd_mul_d(dd_sub(mag, a.re), 0.5));
        if (a.im.hi < 0.0 || (a.im.hi == 0.0 && a.im.lo < 0.0)) v = dd_neg(v);
        dd_t u = dd_div(a.im, dd_mul_d(v, 2.0));
        return (ddc_t){u, v};
    }
}

static const dd_t DD_2PI = {6.283185307179586, 2.4492935982947064e-16};

static ddc_t dd_cis(dd_t theta) {
    const double m = round(theta.hi / (2.0*M_PI));
    dd_t r = dd_sub(theta, dd_mul_d(DD_2PI, m));

    const double inv_k = 1.0/512.0;
    r = dd_mul_d(r, inv_k);

    const ddc_t ir = {{0.0, 0.0}, r};
    ddc_t p = ddc_mul(ir, ir);
    ddc_t s = ddc_add(ir, ddc_scale(dd_from_double(0.5), p));
    p = ddc_mul(p, ir);
    ddc_t t = ddc_scale(DD_INV_FACT[0], p);
    int i = 0;
    do {
        s = ddc_add(s, t);
        p = ddc_mul(p, ir);
        i++;
        t = ddc_scale(DD_INV_FACT[i], p);
    } while ((fabs(t.re.hi) > inv_k*1e-32 || fabs(t.im.hi) > inv_k*1e-32) && i < 9);
    s = ddc_add(s, t);

    for (i = 0; i < 9; i++) {
        s = ddc_add(ddc_scale(dd_from_double(2.0), s), ddc_mul(s, s));
    }
    return (ddc_t){dd_add_d(s.re, 1.0), s.im};
}

typedef struct {
    vec_t hi;
    vec_t lo;
} vdd_t;

static inline __attribute__((always_inline)) vdd_t vdd_from_vec(vec_t x) { return (vdd_t){x, VEC_ZERO}; }
static inline __attribute__((always_inline)) vdd_t vdd_from_dd(dd_t x) { return (vdd_t){VEC_SET1(x.hi), VEC_SET1(x.lo)}; }
static inline __attribute__((always_inline)) vdd_t vdd_neg(vdd_t a) { return (vdd_t){-a.hi, -a.lo}; }

static inline __attribute__((always_inline)) vdd_t vec_two_sum(vec_t a, vec_t b) {
    const vec_t s = a + b;
    const vec_t bb = s - a;
    const vec_t err = (a - (s - bb)) + (b - bb);
    return (vdd_t){s, err};
}

static inline __attribute__((always_inline)) vdd_t vec_quick_two_sum(vec_t a, vec_t b) {  // requires |a| >= |b| per-lane
    const vec_t s = a + b;
    const vec_t err = b - (s - a);
    return (vdd_t){s, err};
}

static inline __attribute__((always_inline)) vdd_t vec_two_prod(vec_t a, vec_t b) {
    const vec_t p = a * b;
    const vec_t err = VEC_FMA(a, b, -p);
    return (vdd_t){p, err};
}

static inline __attribute__((always_inline)) vdd_t vdd_add(vdd_t a, vdd_t b) {
    vdd_t s = vec_two_sum(a.hi, b.hi);
    const vdd_t t = vec_two_sum(a.lo, b.lo);
    s.lo = s.lo + t.hi;
    s = vec_quick_two_sum(s.hi, s.lo);
    s.lo = s.lo + t.lo;
    return vec_quick_two_sum(s.hi, s.lo);
}

static inline __attribute__((always_inline)) vdd_t vdd_add_vec(vdd_t a, vec_t b) {
    vdd_t s = vec_two_sum(a.hi, b);
    s.lo = s.lo + a.lo;
    return vec_quick_two_sum(s.hi, s.lo);
}

static inline __attribute__((always_inline)) vdd_t vdd_sub(vdd_t a, vdd_t b) { return vdd_add(a, vdd_neg(b)); }

static inline __attribute__((always_inline)) vdd_t vdd_mul(vdd_t a, vdd_t b) {
    vdd_t p = vec_two_prod(a.hi, b.hi);
    p.lo = p.lo + (a.hi*b.lo + a.lo*b.hi);
    return vec_quick_two_sum(p.hi, p.lo);
}

static inline __attribute__((always_inline)) vdd_t vdd_mul_vec(vdd_t a, vec_t b) {
    vdd_t p = vec_two_prod(a.hi, b);
    p.lo = p.lo + a.lo*b;
    return vec_quick_two_sum(p.hi, p.lo);
}

static inline __attribute__((always_inline)) vdd_t vdd_div(vdd_t a, vdd_t b) {
    const vec_t q1 = a.hi / b.hi;
    vdd_t r = vdd_sub(a, vdd_mul_vec(b, q1));
    const vec_t q2 = r.hi / b.hi;
    r = vdd_sub(r, vdd_mul_vec(b, q2));
    const vec_t q3 = r.hi / b.hi;
    vdd_t q = vec_quick_two_sum(q1, q2);
    return vdd_add_vec(q, q3);
}

static inline __attribute__((always_inline)) vdd_t vdd_sqrt(vdd_t a) {
    const vec_mask_t is_zero = VEC_CMP_EQ(a.hi, VEC_ZERO);
    const vec_t safe_hi = VEC_SELECT(is_zero, VEC_ONE, a.hi);
    const vec_t safe_lo = VEC_SELECT(is_zero, VEC_ZERO, a.lo);

    const vec_t x = VEC_ONE / VEC_SQRT(safe_hi);
    const vec_t ax = safe_hi * x;
    const vdd_t axdd = vdd_from_vec(ax);
    const vdd_t diff = vdd_sub((vdd_t){safe_hi, safe_lo}, vdd_mul(axdd, axdd));
    const vdd_t result = vdd_add(axdd, vdd_from_vec(diff.hi*(x*VEC_SET1(0.5))));

    return (vdd_t){VEC_SELECT(is_zero, VEC_ZERO, result.hi), VEC_SELECT(is_zero, VEC_ZERO, result.lo)};
}

static inline __attribute__((always_inline)) vdd_t vdd_log2(void) { return vdd_from_dd(DD_LOG2); }
static inline __attribute__((always_inline)) vdd_t vdd_inv_fact(int i) { return vdd_from_dd(DD_INV_FACT[i]); }

static inline vdd_t vdd_exp(vec_t a_d) {
    const vec_mask_t underflow = VEC_CMP_LE(a_d, VEC_SET1(-709.0));
    const vec_mask_t overflow = VEC_CMP_GE(a_d, VEC_SET1(709.0));
    const vec_t safe_a = VEC_SELECT(underflow, VEC_ZERO, VEC_SELECT(overflow, VEC_ZERO, a_d));

    const vec_t inv_k = VEC_SET1(1.0/512.0);
    const vdd_t log2 = vdd_log2();

    const vec_t m = VEC_FLOOR(safe_a/log2.hi + VEC_SET1(0.5));
    vdd_t r = vdd_mul_vec(vdd_sub(vdd_from_vec(safe_a), vdd_mul_vec(log2, m)), inv_k);

    vdd_t p = vdd_mul(r, r);
    vdd_t s = vdd_add(r, vdd_mul_vec(p, VEC_SET1(0.5)));
    p = vdd_mul(p, r);
    vdd_t t = vdd_mul(p, vdd_inv_fact(0));
    for (int i = 1; i < 10; i++) {
        s = vdd_add(s, t);
        p = vdd_mul(p, r);
        t = vdd_mul(p, vdd_inv_fact(i));
    }
    s = vdd_add(s, t);

    for (int i = 0; i < 9; i++) {
        s = vdd_add(vdd_mul_vec(s, VEC_SET1(2.0)), vdd_mul(s, s));
    }
    s = vdd_add_vec(s, VEC_ONE);

    const vec_t res_hi = VEC_SCALEF(s.hi, m);
    const vec_t res_lo = VEC_SCALEF(s.lo, m);

    return (vdd_t){
        VEC_SELECT(underflow, VEC_ZERO, VEC_SELECT(overflow, VEC_SET1(INFINITY), res_hi)),
        VEC_SELECT(underflow, VEC_ZERO, VEC_SELECT(overflow, VEC_ZERO, res_lo)),
    };
}

typedef struct {
    vdd_t re;
    vdd_t im;
} vddc_t;

static inline __attribute__((always_inline)) vddc_t vddc_from_ddc(ddc_t a) { return (vddc_t){vdd_from_dd(a.re), vdd_from_dd(a.im)}; }
static inline __attribute__((always_inline)) vddc_t vddc_add(vddc_t a, vddc_t b) { return (vddc_t){vdd_add(a.re, b.re), vdd_add(a.im, b.im)}; }
static inline __attribute__((always_inline)) vddc_t vddc_sub(vddc_t a, vddc_t b) { return (vddc_t){vdd_sub(a.re, b.re), vdd_sub(a.im, b.im)}; }
static inline __attribute__((always_inline)) vddc_t vddc_neg(vddc_t a) { return (vddc_t){vdd_neg(a.re), vdd_neg(a.im)}; }

static inline __attribute__((always_inline)) vddc_t vddc_mul(vddc_t a, vddc_t b) {
    return (vddc_t){
        vdd_sub(vdd_mul(a.re, b.re), vdd_mul(a.im, b.im)),
        vdd_add(vdd_mul(a.re, b.im), vdd_mul(a.im, b.re)),
    };
}

static inline __attribute__((always_inline)) vddc_t vddc_scale(vdd_t s, vddc_t a) {
    return (vddc_t){vdd_mul(s, a.re), vdd_mul(s, a.im)};
}
static inline __attribute__((always_inline)) vddc_t vddc_scale_dd(dd_t s, vddc_t a) { return vddc_scale(vdd_from_dd(s), a); }

static inline __attribute__((always_inline)) vddc_t vddc_reciprocal(vddc_t a) {
    const vdd_t mag2 = vdd_add(vdd_mul(a.re, a.re), vdd_mul(a.im, a.im));
    return (vddc_t){vdd_div(a.re, mag2), vdd_neg(vdd_div(a.im, mag2))};
}

static inline __attribute__((always_inline)) vdd_t vddc_abs(vddc_t a) {
    return vdd_sqrt(vdd_add(vdd_mul(a.re, a.re), vdd_mul(a.im, a.im)));
}

static inline __attribute__((always_inline)) vddc_t vddc_sqrt(vddc_t a) {
    const vdd_t mag2 = vdd_add(vdd_mul(a.re, a.re), vdd_mul(a.im, a.im));
    const vdd_t mag = vdd_sqrt(mag2);

    // Branch A (re >= 0): u = sqrt((mag+re)/2); v = im/(2u)  (or (0,0) if u==0)
    const vdd_t uA = vdd_sqrt(vdd_mul_vec(vdd_add(mag, a.re), VEC_SET1(0.5)));
    const vec_mask_t uA_zero = VEC_CMP_EQ(uA.hi, VEC_ZERO) & VEC_CMP_EQ(uA.lo, VEC_ZERO);
    const vec_t uA_denom_hi = VEC_SELECT(uA_zero, VEC_ONE, (uA.hi*VEC_SET1(2.0)));  // avoid 0/0
    const vdd_t vA_raw = vdd_div(a.im, (vdd_t){uA_denom_hi, uA.lo*VEC_SET1(2.0)});
    const vdd_t vA = (vdd_t){VEC_SELECT(uA_zero, VEC_ZERO, vA_raw.hi), VEC_SELECT(uA_zero, VEC_ZERO, vA_raw.lo)};

    // Branch B (re < 0): v = sign(im)*sqrt((mag-re)/2); u = im/(2v)
    vdd_t vB = vdd_sqrt(vdd_mul_vec(vdd_sub(mag, a.re), VEC_SET1(0.5)));
    const vec_mask_t im_neg = VEC_CMP_LT(a.im.hi, VEC_ZERO) | (VEC_CMP_EQ(a.im.hi, VEC_ZERO) & VEC_CMP_LT(a.im.lo, VEC_ZERO));
    vB = (vdd_t){VEC_SELECT(im_neg, -vB.hi, vB.hi), VEC_SELECT(im_neg, -vB.lo, vB.lo)};
    const vdd_t uB = vdd_div(a.im, vdd_mul_vec(vB, VEC_SET1(2.0)));

    const vec_mask_t re_nonneg = VEC_CMP_GT(a.re.hi, VEC_ZERO) | (VEC_CMP_EQ(a.re.hi, VEC_ZERO) & VEC_CMP_GE(a.re.lo, VEC_ZERO));
    return (vddc_t){
        {VEC_SELECT(re_nonneg, uA.hi, uB.hi), VEC_SELECT(re_nonneg, uA.lo, uB.lo)},
        {VEC_SELECT(re_nonneg, vA.hi, vB.hi), VEC_SELECT(re_nonneg, vA.lo, vB.lo)},
    };
}

static inline __attribute__((always_inline)) vdd_t vdd_2pi(void) { return vdd_from_dd(DD_2PI); }

static inline __attribute__((always_inline)) vddc_t vdd_cis(vdd_t theta) {
    const vdd_t two_pi = vdd_2pi();
    const vec_t m = VEC_FLOOR(theta.hi/two_pi.hi + VEC_SET1(0.5));
    vdd_t r = vdd_sub(theta, vdd_mul_vec(two_pi, m));

    r = vdd_mul_vec(r, VEC_SET1(1.0/512.0));

    const vddc_t ir = {{VEC_ZERO, VEC_ZERO}, r};
    vddc_t p = vddc_mul(ir, ir);
    vddc_t s = vddc_add(ir, vddc_scale(vdd_from_vec(VEC_SET1(0.5)), p));
    p = vddc_mul(p, ir);
    vddc_t t = vddc_scale_dd(DD_INV_FACT[0], p);
    for (int i = 1; i < 10; i++) {
        s = vddc_add(s, t);
        p = vddc_mul(p, ir);
        t = vddc_scale_dd(DD_INV_FACT[i], p);
    }
    s = vddc_add(s, t);

    for (int i = 0; i < 9; i++) {
        s = vddc_add(vddc_scale(vdd_from_vec(VEC_SET1(2.0)), s), vddc_mul(s, s));
    }
    return (vddc_t){vdd_add_vec(s.re, VEC_ONE), s.im};
}
