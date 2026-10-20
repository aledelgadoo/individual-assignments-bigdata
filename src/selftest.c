#include "selftest.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"
#include "dense.h"
#include "sparse.h"

#define RTOL 1e-10
#define ATOL 1e-12

static long long checks;
static long long failures;

static void check(const char *what, int ok) {
    checks++;
    if (!ok) {
        failures++;
    }
    printf("CHECK,%s,%s\n", what, ok ? "PASS" : "FAIL");
    fflush(stdout);
}

static uint64_t lcg_next(uint64_t *s) {
    *s = *s * 6364136223846793005ULL + 1442695040888963407ULL;
    return *s;
}

static void fill_rand(double *M, long count, uint64_t *seed) {
    for (long i = 0; i < count; i++) {
        uint64_t r = lcg_next(seed);
        double u = (double)(r >> 11) * (1.0 / 9007199254740992.0);
        M[i] = u * 2.0 - 1.0;
    }
}

static void ref_long_double(const double *A, const double *B, long double *C,
                            int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            long double s = 0.0L;
            for (int k = 0; k < n; k++) {
                s += (long double)A[i * n + k] * (long double)B[k * n + j];
            }
            C[i * n + j] = s;
        }
    }
}

static int close_enough(double got, long double ref) {
    double r = (double)ref;
    double diff = got > r ? got - r : r - got;
    double aref = r > 0.0 ? r : -r;
    double tol = ATOL + RTOL * aref;
    return diff <= tol;
}

static long long count_nnz(const double *A, int n) {
    long long c = 0;
    for (long long i = 0; (long long)i < (long long)n * n; i++) {
        if (A[i] != 0.0) {
            c++;
        }
    }
    return c;
}

enum pattern { P_FULL = 0, P_MASK = 1, P_ZERO = 2, P_IDENTITY = 3, P_COUNT = 4 };

static void apply_pattern(double *A, int n, enum pattern p, uint64_t seed) {
    long count = (long)n * n;
    if (p == P_FULL) {
        return;
    }
    if (p == P_ZERO) {
        memset(A, 0, (size_t)count * sizeof(double));
        return;
    }
    if (p == P_IDENTITY) {
        memset(A, 0, (size_t)count * sizeof(double));
        for (int i = 0; i < n; i++) {
            A[i * n + i] = 1.0;
        }
        return;
    }
    uint64_t s = seed;
    for (long i = 0; i < count; i++) {
        if (lcg_next(&s) & 1ULL) {
            A[i] = 0.0;
        }
    }
}

static void run_case(int n, enum pattern p, const int *bss, int nbs) {
    char name[128];
    long count = (long)n * n;
    double *A = (double *)xmalloc((size_t)count * sizeof(double));
    double *B = (double *)xmalloc((size_t)count * sizeof(double));
    double *C0 = (double *)xmalloc((size_t)count * sizeof(double));
    double *C1 = (double *)xmalloc((size_t)count * sizeof(double));
    double *C2 = (double *)xmalloc((size_t)count * sizeof(double));
    double *C3 = (double *)xmalloc((size_t)count * sizeof(double));
    long double *R = (long double *)xmalloc((size_t)count * sizeof(long double));

    uint64_t seed = 0x243F6A8885A308D3ULL ^ (uint64_t)n * 131ULL ^ (uint64_t)p;
    fill_rand(A, count, &seed);
    fill_rand(B, count, &seed);
    apply_pattern(A, n, p, 0x9E3779B97F4A7C15ULL ^ (uint64_t)n);

    matmul_ijk(A, B, C0, n);
    matmul_ikj(A, B, C1, n);
    csr *sp0 = csr_build(A, n);
    if (sp0 == NULL) {
        die("csr_build failed in selftest");
    }
    matmul_csr(sp0, B, C3, n);
    csr_free(sp0);
    ref_long_double(A, B, R, n);

    snprintf(name, sizeof(name), "n=%d,patt=%d,ijk-vs-ldouble", n, p);
    int ok_ref = 1;
    for (long i = 0; i < count; i++) {
        if (!close_enough(C0[i], R[i])) {
            ok_ref = 0;
            break;
        }
    }
    check(name, ok_ref);

    snprintf(name, sizeof(name), "n=%d,patt=%d,ikj==ijk", n, p);
    check(name, memcmp(C0, C1, (size_t)count * sizeof(double)) == 0);

    snprintf(name, sizeof(name), "n=%d,patt=%d,csr==ijk", n, p);
    check(name, memcmp(C0, C3, (size_t)count * sizeof(double)) == 0);

    for (int bi = 0; bi < nbs; bi++) {
        matmul_blocked(A, B, C2, n, bss[bi]);
        snprintf(name, sizeof(name), "n=%d,patt=%d,block%d==ijk", n, p,
                 bss[bi]);
        check(name, memcmp(C0, C2, (size_t)count * sizeof(double)) == 0);
    }

    if (p == P_IDENTITY) {
        snprintf(name, sizeof(name), "n=%d,identityA*C==B", n);
        check(name, memcmp(C0, B, (size_t)count * sizeof(double)) == 0);
    }

    csr *sp = csr_build(A, n);
    snprintf(name, sizeof(name), "n=%d,patt=%d,csr-nnz", n, p);
    check(name, sp != NULL && sp->nnz == count_nnz(A, n));
    csr_free(sp);

    snprintf(name, sizeof(name), "n=%d,patt=%d,all-ldouble", n, p);
    int ok_all = 1;
    for (long i = 0; i < count; i++) {
        if (!close_enough(C1[i], R[i]) || !close_enough(C3[i], R[i])) {
            ok_all = 0;
            break;
        }
    }
    check(name, ok_all);

    free(A);
    free(B);
    free(C0);
    free(C1);
    free(C2);
    free(C3);
    free(R);
}

static void run_known_answers(void) {
    double A4[4] = {1.0, 2.0, 3.0, 4.0};
    double B4[4] = {5.0, 6.0, 7.0, 8.0};
    double want4[4] = {19.0, 22.0, 43.0, 50.0};
    double C[4];

    matmul_ijk(A4, B4, C, 2);
    check("known2x2,ijk", memcmp(C, want4, sizeof(want4)) == 0);
    matmul_ikj(A4, B4, C, 2);
    check("known2x2,ikj", memcmp(C, want4, sizeof(want4)) == 0);
    matmul_blocked(A4, B4, C, 2, 4);
    check("known2x2,block", memcmp(C, want4, sizeof(want4)) == 0);
    csr *sp = csr_build(A4, 2);
    matmul_csr(sp, B4, C, 2);
    check("known2x2,csr", memcmp(C, want4, sizeof(want4)) == 0);
    csr_free(sp);

    double A3[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    double B3[9] = {2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    double C3[9];
    matmul_ijk(A3, B3, C3, 3);
    check("known3x3,identityA", memcmp(C3, B3, sizeof(B3)) == 0);

    double Z[9] = {0};
    double Cz[9];
    matmul_ijk(Z, B3, Cz, 3);
    int allzero = 1;
    for (int i = 0; i < 9; i++) {
        if (Cz[i] != 0.0) {
            allzero = 0;
        }
    }
    check("known3x3,zeroA", allzero);
    csr *spz = csr_build(Z, 3);
    check("known3x3,zeroA-csr-nnz0", spz != NULL && spz->nnz == 0);
    matmul_csr(spz, B3, Cz, 3);
    allzero = 1;
    for (int i = 0; i < 9; i++) {
        if (Cz[i] != 0.0) {
            allzero = 0;
        }
    }
    check("known3x3,zeroA-csr", allzero);
    csr_free(spz);
}

int run_selftest(void) {
    const int sizes[] = {1, 2, 3, 5, 8, 17, 32, 64};
    const int bss[] = {4, 7, 16, 64};

    run_known_answers();
    for (unsigned si = 0; si < sizeof(sizes) / sizeof(sizes[0]); si++) {
        for (int p = 0; p < P_COUNT; p++) {
            run_case(sizes[si], (enum pattern)p, bss,
                     (int)(sizeof(bss) / sizeof(bss[0])));
        }
    }
    printf("SELFTEST,checks=%lld,failures=%lld\n", checks, failures);
    return (int)failures;
}
