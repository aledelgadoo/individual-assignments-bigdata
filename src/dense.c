#include "dense.h"

#include <string.h>

void matmul_ijk(const double *restrict A, const double *restrict B,
                double *restrict C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++) {
                s += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = s;
        }
    }
}

void matmul_ikj(const double *restrict A, const double *restrict B,
                double *restrict C, int n) {
    memset(C, 0, (size_t)n * (size_t)n * sizeof(double));
    for (int i = 0; i < n; i++) {
        double *restrict ci = C + (size_t)i * n;
        const double *restrict ai = A + (size_t)i * n;
        for (int k = 0; k < n; k++) {
            double a = ai[k];
            const double *restrict bk = B + (size_t)k * n;
            for (int j = 0; j < n; j++) {
                ci[j] += a * bk[j];
            }
        }
    }
}

void matmul_blocked(const double *restrict A, const double *restrict B,
                    double *restrict C, int n, int bs) {
    memset(C, 0, (size_t)n * (size_t)n * sizeof(double));
    if (bs < 1) {
        bs = 1;
    }
    for (int ii = 0; ii < n; ii += bs) {
        int im = ii + bs < n ? ii + bs : n;
        for (int jj = 0; jj < n; jj += bs) {
            int jm = jj + bs < n ? jj + bs : n;
            for (int kk = 0; kk < n; kk += bs) {
                int km = kk + bs < n ? kk + bs : n;
                for (int i = ii; i < im; i++) {
                    double *restrict ci = C + (size_t)i * n;
                    const double *restrict ai = A + (size_t)i * n;
                    for (int j = jj; j < jm; j++) {
                        double s = ci[j];
                        for (int k = kk; k < km; k++) {
                            s += ai[k] * B[(size_t)k * n + j];
                        }
                        ci[j] = s;
                    }
                }
            }
        }
    }
}
