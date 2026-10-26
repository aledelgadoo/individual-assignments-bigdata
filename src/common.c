#include "common.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#else
#include <time.h>
#endif

void die(const char *msg) {
    fprintf(stderr, "ERROR: %s\n", msg);
    exit(1);
}

void emit(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    putchar('\n');
    fflush(stdout);
}

const char *find_arg(int argc, char **argv, const char *key) {
    for (int i = 2; i < argc - 1; i++) {
        if (strcmp(argv[i], key) == 0) {
            return argv[i + 1];
        }
    }
    return NULL;
}

const char *require_arg(int argc, char **argv, const char *key) {
    const char *v = find_arg(argc, argv, key);
    if (v == NULL) {
        die("missing required argument");
    }
    return v;
}

long long long_arg(int argc, char **argv, const char *key) {
    const char *v = find_arg(argc, argv, key);
    if (v == NULL) {
        return -1;
    }
    return atoll(v);
}

double now_ns(void) {
#if defined(_WIN32)
    LARGE_INTEGER counts, freq;
    QueryPerformanceCounter(&counts);
    QueryPerformanceFrequency(&freq);
    return (double)counts.QuadPart / (double)freq.QuadPart * 1e9;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
#endif
}

long long peak_wss_bytes(void) {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return (long long)pmc.PeakWorkingSetSize;
    }
    return -1;
#else
    return -1;
#endif
}

int set_process_affinity(int pcores_only) {
#if defined(_WIN32)
    DWORD_PTR mask = pcores_only ? (DWORD_PTR)0xF : (DWORD_PTR)0xFFF;
    if (SetProcessAffinityMask(GetCurrentProcess(), mask) == 0) {
        return 0;
    }
    return 1;
#else
    (void)pcores_only;
    return 1;
#endif
}

double *load_matrix(const char *path, int n) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        die("cannot open input file");
    }
    size_t want = (size_t)n * (size_t)n;
    double *M = (double *)malloc(want * sizeof(double));
    if (M == NULL) {
        fclose(f);
        die("malloc failed for input matrix");
    }
    if (fread(M, sizeof(double), want, f) != want) {
        fclose(f);
        die("short read on input file");
    }
    fclose(f);
    return M;
}

void *xmalloc(size_t bytes) {
    void *p = malloc(bytes);
    if (p == NULL) {
        die("malloc failed");
    }
    return p;
}
