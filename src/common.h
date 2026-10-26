#ifndef COMMON_H
#define COMMON_H

#include <stddef.h>

double now_ns(void);
long long peak_wss_bytes(void);
int set_process_affinity(int pcores_only);

void die(const char *msg);
void emit(const char *fmt, ...) __attribute__((format(printf, 1, 2)));

const char *find_arg(int argc, char **argv, const char *key);
const char *require_arg(int argc, char **argv, const char *key);
long long long_arg(int argc, char **argv, const char *key);

double *load_matrix(const char *path, int n);
void *xmalloc(size_t bytes);

#endif
