#include "common.h"
#include "kernels.h"
#include "selftest.h"
#include "variant.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_OMP
#include <omp.h>
#endif

typedef struct {
    int n;
    const double *A;
    const double *B;
    double *C;
} bench_ctx;

static double checksum(const double *C, size_t len) {
    double s = 0.0;
    for (size_t i = 0; i < len; i++) {
        s += C[i];
    }
    return s;
}

static void apply_settings(int threads, const char *affinity) {
#ifdef USE_OMP
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
    if (!set_process_affinity(strcmp(affinity, "pcore") == 0)) {
        die("SetProcessAffinityMask failed");
    }
#else
    (void)threads;
    (void)affinity;
#endif
}

static int cmd_probe(int argc, char **argv) {
    const char *a_path = require_arg(argc, argv, "--a");
    const char *b_path = require_arg(argc, argv, "--b");
    int n = (int)long_arg(argc, argv, "--n");
    int threads = (int)long_arg(argc, argv, "--threads");
    const char *affinity = find_arg(argc, argv, "--affinity");
    long long warmups = long_arg(argc, argv, "--warmups");
    long long probe_reps = long_arg(argc, argv, "--probe-reps");

    if (n <= 0) die("--n must be positive");
    if (threads <= 0) threads = 1;
    if (affinity == NULL) affinity = "all";
    if (warmups < 0) warmups = 1;
    if (probe_reps < 0) probe_reps = 2;
    if (strcmp(affinity, "all") != 0 && strcmp(affinity, "pcore") != 0) {
        die("--affinity must be all or pcore");
    }
#ifdef USE_OMP
    apply_settings(threads, affinity);
#else
    threads = 1;
    apply_settings(threads, affinity);
#endif

    double *A = load_matrix(a_path, n);
    double *B = load_matrix(b_path, n);
    double *C = xmalloc((size_t)n * (size_t)n * sizeof(double));
    kernel_fn f = current_kernel();

    for (long long w = 0; w < warmups; w++) {
        double t0 = now_ns();
        f(n, A, B, C);
        double dt = now_ns() - t0;
        emit("{\"mode\":\"warmup\",\"variant\":\"%s\",\"n\":%d,\"threads\":%d,"
             "\"affinity\":\"%s\",\"rep\":%lld,\"kernel_ns\":%.0f}",
             kernel_variant(), n, threads, affinity, w, dt);
    }

    size_t len = (size_t)n * (size_t)n;
    for (long long r = 0; r < probe_reps; r++) {
        double t0 = now_ns();
        f(n, A, B, C);
        double dt = now_ns() - t0;
        emit("{\"mode\":\"probe\",\"variant\":\"%s\",\"n\":%d,\"threads\":%d,"
             "\"affinity\":\"%s\",\"rep\":%lld,\"kernel_ns\":%.0f,"
             "\"checksum\":%.17g,\"peak_wss_bytes\":%lld}",
             kernel_variant(), n, threads, affinity, r, dt,
             checksum(C, len), peak_wss_bytes());
    }

    free(A);
    free(B);
    free(C);
    return 0;
}

static int cmd_bench(int argc, char **argv) {
    const char *a_path = require_arg(argc, argv, "--a");
    const char *b_path = require_arg(argc, argv, "--b");
    int n = (int)long_arg(argc, argv, "--n");
    int threads = (int)long_arg(argc, argv, "--threads");
    const char *affinity = find_arg(argc, argv, "--affinity");
    long long warmups = long_arg(argc, argv, "--warmups");
    long long reps = long_arg(argc, argv, "--reps");
    long long budget_ns = long_arg(argc, argv, "--budget-ns");
    long long min_batch_ns = long_arg(argc, argv, "--min-batch-ns");

    if (n <= 0) die("--n must be positive");
    if (threads <= 0) threads = 1;
    if (affinity == NULL) affinity = "all";
    if (warmups < 0) warmups = 1;
    if (reps < 0) reps = 3;
    if (budget_ns < 0) budget_ns = 60000000000LL;
    if (min_batch_ns < 0) min_batch_ns = 200000000LL;
    if (strcmp(affinity, "all") != 0 && strcmp(affinity, "pcore") != 0) {
        die("--affinity must be all or pcore");
    }
#ifdef USE_OMP
    apply_settings(threads, affinity);
#else
    threads = 1;
    apply_settings(threads, affinity);
#endif

    double *A = load_matrix(a_path, n);
    double *B = load_matrix(b_path, n);
    double *C = xmalloc((size_t)n * (size_t)n * sizeof(double));
    kernel_fn f = current_kernel();
    size_t len = (size_t)n * (size_t)n;

    for (long long w = 0; w < warmups; w++) {
        double t0 = now_ns();
        f(n, A, B, C);
        double dt = now_ns() - t0;
        emit("{\"mode\":\"warmup\",\"variant\":\"%s\",\"n\":%d,\"threads\":%d,"
             "\"affinity\":\"%s\",\"rep\":%lld,\"kernel_ns\":%.0f}",
             kernel_variant(), n, threads, affinity, w, dt);
        if (dt > (double)budget_ns) {
            emit("{\"mode\":\"timeout\",\"variant\":\"%s\",\"n\":%d,"
                 "\"threads\":%d,\"affinity\":\"%s\","
                 "\"reason\":\"warmup_over_budget\",\"kernel_ns\":%.0f,"
                 "\"budget_ns\":%lld}",
                 kernel_variant(), n, threads, affinity, dt, budget_ns);
            free(A);
            free(B);
            free(C);
            return 2;
        }
    }

    long long batch = 1;
    for (;;) {
        double t0 = now_ns();
        for (long long b = 0; b < batch; b++) {
            f(n, A, B, C);
        }
        double total = now_ns() - t0;
        if (total >= (double)min_batch_ns || batch >= (1LL << 40)) {
            break;
        }
        batch *= 2;
    }

    for (long long r = 0; r < reps; r++) {
        double t0 = now_ns();
        for (long long b = 0; b < batch; b++) {
            f(n, A, B, C);
        }
        double total = now_ns() - t0;
        double per = total / (double)batch;
        emit("{\"mode\":\"measure\",\"variant\":\"%s\",\"n\":%d,\"threads\":%d,"
             "\"affinity\":\"%s\",\"rep\":%lld,\"batch\":%lld,"
             "\"kernel_total_ns\":%.0f,\"kernel_per_mult_ns\":%.1f,"
             "\"checksum\":%.17g,\"peak_wss_bytes\":%lld}",
             kernel_variant(), n, threads, affinity, r, batch, total, per,
             checksum(C, len), peak_wss_bytes());
    }

    emit("{\"mode\":\"done\",\"variant\":\"%s\",\"n\":%d,\"threads\":%d,"
         "\"affinity\":\"%s\",\"reps\":%lld,\"batch\":%lld}",
         kernel_variant(), n, threads, affinity, reps, batch);

    free(A);
    free(B);
    free(C);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        die("usage: selftest | probe | bench  (see --help in scripts)");
    }
    if (strcmp(argv[1], "selftest") == 0) {
        return run_selftest();
    }
    if (strcmp(argv[1], "probe") == 0) {
        return cmd_probe(argc, argv);
    }
    if (strcmp(argv[1], "bench") == 0) {
        return cmd_bench(argc, argv);
    }
    die("unknown mode (expected selftest, probe or bench)");
    return 1;
}
