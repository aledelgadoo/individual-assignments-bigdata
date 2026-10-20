#ifndef SPARSE_H
#define SPARSE_H

typedef struct {
    int n;
    long long nnz;
    double *val;
    int *col;
    int *rowptr;
} csr;

csr *csr_build(const double *A, int n);
void csr_free(csr *c);
void matmul_csr(const csr *a, const double *restrict B, double *restrict C,
                int n);

#endif
