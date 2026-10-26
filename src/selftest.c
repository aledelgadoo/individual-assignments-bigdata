#include "selftest.h"

#include "common.h"
#include "kernels.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_OMP
#include <omp.h>
#endif

static int st_checks = 0;
static int st_failures = 0;

static void st_check(int cond, const char *what) {
    st_checks++;
    if (!cond) {
        st_failures++;
        emit("SELFTEST,check=FAIL,detail=\"%s\"", what);
    }
}

static uint64_t xorshift64(uint64_t *s) {
    uint64_t x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    return x;
}

static void fill_random(double *M, size_t len, uint64_t *state) {
    for (size_t i = 0; i < len; i++) {
        double u = (double)(xorshift64(state) >> 11) * (1.0 / 9007199254740992.0);
        M[i] = 2.0 * u - 1.0;
    }
}

static void ref_long_double(int n, const double *A, const double *B, double *C) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            long double s = 0.0L;
            for (int k = 0; k < n; k++) {
                s += (long double)A[(size_t)i * n + k] *
                     (long double)B[(size_t)k * n + j];
            }
            C[(size_t)i * n + j] = (double)s;
        }
    }
}

static int all_zero(const double *C, size_t len) {
    for (size_t i = 0; i < len; i++) {
        if (C[i] != 0.0) return 0;
    }
    return 1;
}

static void test_known_2x2(kernel_fn f) {
    double A[4] = {1, 2, 3, 4};
    double B[4] = {5, 6, 7, 8};
    double C[4];
    double expect[4] = {19, 22, 43, 50};
    f(2, A, B, C);
    for (int i = 0; i < 4; i++) {
        st_check(C[i] == expect[i], "known 2x2 product");
    }
}

static void test_patterns(kernel_fn f, int n, uint64_t *state) {
    size_t len = (size_t)n * (size_t)n;
    double *A = xmalloc(len * sizeof(double));
    double *B = xmalloc(len * sizeof(double));
    double *C = xmalloc(len * sizeof(double));
    double *R = xmalloc(len * sizeof(double));

    fill_random(B, len, state);

    memset(A, 0, len * sizeof(double));
    f(n, A, B, C);
    st_check(all_zero(C, len), "zero A gives zero product");

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[(size_t)i * n + j] = (i == j) ? 1.0 : 0.0;
        }
    }
    f(n, A, B, C);
    st_check(memcmp(C, B, len * sizeof(double)) == 0,
             "identity A gives B bitwise");

    fill_random(A, len, state);
    f(n, A, B, C);
    ref_long_double(n, A, B, R);
    int ok = 1;
    for (size_t i = 0; i < len; i++) {
        double tol = 1e-10 * fabs(R[i]) + 1e-12;
        if (!(fabs(C[i] - R[i]) <= tol)) {
            ok = 0;
            break;
        }
    }
    st_check(ok, "random product within rtol/atol of long-double reference");

    free(A);
    free(B);
    free(C);
    free(R);
}

#ifdef USE_OMP
static void test_thread_invariance(kernel_fn fpar, int n, uint64_t *state) {
    static const int thread_grid[] = {2, 3, 4, 8, 12};
    size_t len = (size_t)n * (size_t)n;
    double *A = xmalloc(len * sizeof(double));
    double *B = xmalloc(len * sizeof(double));
    double *Cseq = xmalloc(len * sizeof(double));
    double *Cpar = xmalloc(len * sizeof(double));

    fill_random(A, len, state);
    fill_random(B, len, state);
    matmul_ikj_seq(n, A, B, Cseq);

    omp_set_dynamic(0);
    for (size_t t = 0; t < sizeof(thread_grid) / sizeof(thread_grid[0]); t++) {
        omp_set_num_threads(thread_grid[t]);
        fpar(n, A, B, Cpar);
        char msg[96];
        snprintf(msg, sizeof(msg), "thread invariance bitwise (p=%d, n=%d)",
                 thread_grid[t], n);
        st_check(memcmp(Cseq, Cpar, len * sizeof(double)) == 0, msg);
    }

    free(A);
    free(B);
    free(Cseq);
    free(Cpar);
}
#endif

int run_selftest(void) {
    static const int sizes[] = {1, 2, 3, 5, 8, 13, 17, 32, 64};
    uint64_t state = 0x9E3779B97F4A7C15ULL;
    kernel_fn f = current_kernel();

    test_known_2x2(f);
    for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
        test_patterns(f, sizes[i], &state);
    }
#ifdef USE_OMP
    matmul_ikj_par(2, (const double[4]){1, 2, 3, 4},
                   (const double[4]){5, 6, 7, 8}, (double[4]){0, 0, 0, 0});
    for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
        test_thread_invariance(matmul_ikj_par, sizes[i], &state);
    }
#endif

    emit("SELFTEST,variant=%s,checks=%d,failures=%d", kernel_variant(),
         st_checks, st_failures);
    return st_failures == 0 ? 0 : 1;
}
