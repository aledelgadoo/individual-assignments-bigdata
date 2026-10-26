#include "kernels.h"
#include "variant.h"

#include <string.h>

void matmul_ikj_seq(int n, const double *restrict A, const double *restrict B,
                    double *restrict C) {
    memset(C, 0, (size_t)n * (size_t)n * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            double a = A[(size_t)i * (size_t)n + (size_t)k];
            for (int j = 0; j < n; j++) {
                C[(size_t)i * (size_t)n + (size_t)j] +=
                    a * B[(size_t)k * (size_t)n + (size_t)j];
            }
        }
    }
}

#ifdef USE_OMP
void matmul_ikj_par(int n, const double *restrict A, const double *restrict B,
                    double *restrict C) {
    memset(C, 0, (size_t)n * (size_t)n * sizeof(double));
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            double a = A[(size_t)i * (size_t)n + (size_t)k];
#ifdef USE_OMP_SIMD
#pragma omp simd
#endif
            for (int j = 0; j < n; j++) {
                C[(size_t)i * (size_t)n + (size_t)j] +=
                    a * B[(size_t)k * (size_t)n + (size_t)j];
            }
        }
    }
}
#endif

kernel_fn current_kernel(void) {
#ifdef USE_OMP
    return matmul_ikj_par;
#else
    return matmul_ikj_seq;
#endif
}

const char *kernel_variant(void) {
    return VARIANT_NAME;
}
