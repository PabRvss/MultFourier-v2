#include <stdlib.h>
#include <math.h>
#include "base.h"

real_t* lgac = NULL;
static size_t lgac_capacity = 0;

void init_ext(int_t fact_max, int_t verbose_ext) {
    if (lgac != NULL && (size_t)fact_max < lgac_capacity) {
        return; // Mantiene la memoria si la capacidad es suficiente
    }
    if (lgac != NULL) {
        free(lgac);
    }
    lgac_capacity = fact_max + 1000;
    lgac = (real_t*)malloc(lgac_capacity * sizeof(real_t));
    if (lgac == NULL) return;

    lgac[0] = 0;
    double s = 0.0;
    double d = 0.0;
    for (size_t i = 1; i < lgac_capacity; i++) {
        double l = log((double)i);
        double y = l - d;
        double t = s + y;
        d = (t - s) - y;
        s = t;
        lgac[i] = (real_t)s;
    }
}

void free_ext(void) {
    if (lgac != NULL) {
        free(lgac);
        lgac = NULL;
        lgac_capacity = 0;
    }
}