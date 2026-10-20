#include "sparse.h"

#include <stdlib.h>
#include <string.h>

csr *csr_build(const double *A, int n) {
    csr *c = (csr *)malloc(sizeof(csr));
    if (c == NULL) {
        return NULL;
    }
    c->n = n;
    c->rowptr = (int *)malloc(((size_t)n + 1) * sizeof(int));
    if (c->rowptr == NULL) {
        free(c);
        return NULL;
    }
    c->rowptr[0] = 0;
    for (int i = 0; i < n; i++) {
        int count = 0;
        const double *row = A + (size_t)i * n;
        for (int j = 0; j < n; j++) {
            if (row[j] != 0.0) {
                count++;
            }
        }
        c->rowptr[i + 1] = c->rowptr[i] + count;
    }
    int nnz = c->rowptr[n];
    c->nnz = nnz;
    c->val = (double *)malloc((size_t)nnz * sizeof(double));
    c->col = (int *)malloc((size_t)nnz * sizeof(int));
    if (c->val == NULL || c->col == NULL) {
        csr_free(c);
        return NULL;
    }
    int *cursor = (int *)malloc(((size_t)n + 1) * sizeof(int));
    if (cursor == NULL) {
        csr_free(c);
        return NULL;
    }
    memcpy(cursor, c->rowptr, ((size_t)n + 1) * sizeof(int));
    for (int i = 0; i < n; i++) {
        const double *row = A + (size_t)i * n;
        for (int j = 0; j < n; j++) {
            if (row[j] != 0.0) {
                int p = cursor[i]++;
                c->val[p] = row[j];
                c->col[p] = j;
            }
        }
    }
    free(cursor);
    return c;
}

void csr_free(csr *c) {
    if (c == NULL) {
        return;
    }
    free(c->val);
    free(c->col);
    free(c->rowptr);
    free(c);
}

void matmul_csr(const csr *a, const double *restrict B, double *restrict C,
                int n) {
    memset(C, 0, (size_t)n * (size_t)n * sizeof(double));
    for (int i = 0; i < n; i++) {
        double *restrict ci = C + (size_t)i * n;
        for (int p = a->rowptr[i]; p < a->rowptr[i + 1]; p++) {
            double av = a->val[p];
            const double *restrict bk = B + (size_t)a->col[p] * n;
            for (int j = 0; j < n; j++) {
                ci[j] += av * bk[j];
            }
        }
    }
}
