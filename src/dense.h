#ifndef DENSE_H
#define DENSE_H

void matmul_ijk(const double *restrict A, const double *restrict B,
                double *restrict C, int n);
void matmul_ikj(const double *restrict A, const double *restrict B,
                double *restrict C, int n);
void matmul_blocked(const double *restrict A, const double *restrict B,
                    double *restrict C, int n, int bs);

#endif
