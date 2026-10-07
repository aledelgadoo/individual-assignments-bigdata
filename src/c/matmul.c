/*
 * Basic dense matrix multiplication in C (double*, triple loop).
 *
 * The kernel is the classical O(n^3) i-j-k triple loop over IEEE-754 doubles;
 * no optimized library is used anywhere near the multiplication. Matrices are
 * a single contiguous row-major block of doubles, matching the shared binary
 * input format (little-endian float64, assumed on the x86-64 host).
 *
 * Modes:
 *   multiply --a A.bin --b B.bin --n n --out C.bin
 *       Load A and B, run the kernel once, write the product C and print a
 *       checksum.
 *   probe --a A.bin --b B.bin --n n
 *       Load, run the kernel exactly once (QueryPerformanceCounter around the
 *       kernel only) and print a JSON line. Used by the orchestrator as a
 *       budget gate.
 *   bench --a A.bin --b B.bin --n n --warmups W --reps R
 *         [--budget-ns N] [--min-batch-ns N]
 *       Load outside the timed region, run W warm-up multiplications (excluded
 *       from the statistics), calibrate a batch count so that each measured
 *       repetition lasts at least --min-batch-ns, then run R measured
 *       repetitions. One JSON line per repetition (flushed immediately) plus a
 *       final "done" line with the peak working set size.
 */

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

#define BATCH_CAP 1000000LL
#define LANG "c"

static void die(const char *msg) {
    fprintf(stderr, "ERROR: %s\n", msg);
    exit(1);
}

static const char *find_arg(int argc, char **argv, const char *key) {
    for (int i = 2; i < argc - 1; i++) {
        if (strcmp(argv[i], key) == 0) {
            return argv[i + 1];
        }
    }
    return NULL;
}

static const char *require_arg(int argc, char **argv, const char *key) {
    const char *v = find_arg(argc, argv, key);
    if (v == NULL) {
        die("missing required argument");
    }
    return v;
}

static long long long_arg(int argc, char **argv, const char *key) {
    const char *v = find_arg(argc, argv, key);
    if (v == NULL) {
        return -1;
    }
    return atoll(v);
}

/* Monotonic high-resolution clock in nanoseconds since an arbitrary origin. */
static double now_ns(void) {
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

/* Peak resident working set of this process in bytes (-1 if unavailable). */
static long long peak_wss_bytes(void) {
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

static void emit(const char *fmt, ...) __attribute__((format(printf, 1, 2)));

static void emit(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    putchar('\n');
    fflush(stdout);
}

static double *load_matrix(const char *path, int n) {
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

static void save_matrix(const char *path, const double *M, int n) {
    FILE *f = fopen(path, "wb");
    if (f == NULL) {
        die("cannot open output file");
    }
    size_t want = (size_t)n * (size_t)n;
    if (fwrite(M, sizeof(double), want, f) != want) {
        fclose(f);
        die("short write on output file");
    }
    if (fclose(f) != 0) {
        die("failed to close output file");
    }
}

void matmul_into(const double *A, const double *B, double *C, int n) {
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

static double checksum(const double *M, long count) {
    double s = 0.0;
    for (long i = 0; i < count; i++) {
        s += M[i];
    }
    return s;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        die("usage: matmul {multiply|probe|bench} --a A.bin --b B.bin --n n [...]");
    }
    const char *mode = argv[1];
    if (strcmp(mode, "multiply") != 0 && strcmp(mode, "probe") != 0 && strcmp(mode, "bench") != 0) {
        die("unknown mode (expected: multiply, probe or bench)");
    }
    const char *pa = require_arg(argc, argv, "--a");
    const char *pb = require_arg(argc, argv, "--b");
    const char *pn = require_arg(argc, argv, "--n");
    int n = atoi(pn);
    if (n <= 0) {
        die("invalid --n");
    }
    long count = (long)n * (long)n;

    if (strcmp(mode, "multiply") == 0) {
        const char *pout = require_arg(argc, argv, "--out");
        double *A = load_matrix(pa, n);
        double *B = load_matrix(pb, n);
        double *C = (double *)malloc((size_t)count * sizeof(double));
        if (C == NULL) {
            die("malloc failed for output matrix");
        }
        matmul_into(A, B, C, n);
        save_matrix(pout, C, n);
        printf("n=%d checksum=%.17g\n", n, checksum(C, count));
        free(A);
        free(B);
        free(C);
        return 0;
    }

    double *A = load_matrix(pa, n);
    double *B = load_matrix(pb, n);
    double *C = (double *)malloc((size_t)count * sizeof(double));
    if (C == NULL) {
        die("malloc failed for output matrix");
    }

    if (strcmp(mode, "probe") == 0) {
        double t0 = now_ns();
        matmul_into(A, B, C, n);
        double t1 = now_ns();
        emit("{\"mode\":\"probe\",\"lang\":\"" LANG "\",\"n\":%d,"
              "\"kernel_total_ns\":%lld,\"checksum\":%.17g}",
              n, (long long)(t1 - t0), checksum(C, count));
        free(A);
        free(B);
        free(C);
        return 0;
    }

    /* bench */
    long long warmups = long_arg(argc, argv, "--warmups");
    long long reps = long_arg(argc, argv, "--reps");
    long long budget_ns = long_arg(argc, argv, "--budget-ns");
    long long min_batch_ns = long_arg(argc, argv, "--min-batch-ns");
    if (warmups < 1 || reps < 1 || budget_ns < 1 || min_batch_ns < 1) {
        die("bench requires --warmups >= 1, --reps >= 1, --budget-ns >= 1, --min-batch-ns >= 1");
    }

    long long warmup_ns = 0;
    for (long long w = 0; w < warmups; w++) {
        double t0 = now_ns();
        matmul_into(A, B, C, n);
        double t1 = now_ns();
        warmup_ns = (long long)(t1 - t0);
        emit("{\"mode\":\"bench\",\"lang\":\"" LANG "\",\"n\":%d,"
              "\"phase\":\"warmup\",\"rep\":%lld,\"batch\":1,"
              "\"kernel_total_ns\":%lld,\"kernel_per_mult_ns\":%.17g,\"checksum\":%.17g}",
              n, w, warmup_ns, (double)warmup_ns, checksum(C, count));
    }
    if (warmup_ns > budget_ns) {
        emit("{\"mode\":\"timeout\",\"lang\":\"" LANG "\",\"n\":%d,"
              "\"reason\":\"warm-up exceeded budget\",\"kernel_total_ns\":%lld}",
              n, warmup_ns);
        return 3;
    }

    long long batch = (min_batch_ns + warmup_ns - 1) / (warmup_ns > 0 ? warmup_ns : 1);
    if (batch < 1) {
        batch = 1;
    }
    if (batch > BATCH_CAP) {
        batch = BATCH_CAP;
    }

    for (long long r = 0; r < reps; r++) {
        double t0 = now_ns();
        for (long long b = 0; b < batch; b++) {
            matmul_into(A, B, C, n);
        }
        double t1 = now_ns();
        long long total = (long long)(t1 - t0);
        emit("{\"mode\":\"bench\",\"lang\":\"" LANG "\",\"n\":%d,"
              "\"phase\":\"measure\",\"rep\":%lld,\"batch\":%lld,"
              "\"kernel_total_ns\":%lld,\"kernel_per_mult_ns\":%.17g,\"checksum\":%.17g}",
              n, r, batch, total, (double)total / (double)batch, checksum(C, count));
    }

    emit("{\"mode\":\"done\",\"lang\":\"" LANG "\",\"n\":%d,"
          "\"peak_wss_bytes\":%lld,\"heap_used_bytes\":null}",
          n, peak_wss_bytes());

    free(A);
    free(B);
    free(C);
    return 0;
}
