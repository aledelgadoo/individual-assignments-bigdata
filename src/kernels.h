#ifndef KERNELS_H
#define KERNELS_H

typedef void (*kernel_fn)(int n, const double *A, const double *B, double *C);

void matmul_ikj_seq(int n, const double *A, const double *B, double *C);
#ifdef USE_OMP
void matmul_ikj_par(int n, const double *A, const double *B, double *C);
#endif

kernel_fn current_kernel(void);
const char *kernel_variant(void);

#endif
