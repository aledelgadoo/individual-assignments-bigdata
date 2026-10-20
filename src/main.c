#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"
#include "dense.h"
#include "selftest.h"
#include "sparse.h"

#define BATCH_CAP 1000000LL

typedef struct {
    const char *method;
    const double *A;
    const double *B;
    double *C;
    csr *sp;
    int n;
    int bs;
} job;

static void run_kernel(const job *jb) {
    if (strcmp(jb->method, "ijk") == 0) {
        matmul_ijk(jb->A, jb->B, jb->C, jb->n);
    } else if (strcmp(jb->method, "ikj") == 0) {
        matmul_ikj(jb->A, jb->B, jb->C, jb->n);
    } else if (strcmp(jb->method, "block") == 0) {
        matmul_blocked(jb->A, jb->B, jb->C, jb->n, jb->bs);
    } else {
        matmul_csr(jb->sp, jb->B, jb->C, jb->n);
    }
}

static double checksum(const double *M, long count) {
    double s = 0.0;
    for (long i = 0; i < count; i++) {
        s += M[i];
    }
    return s;
}

static int method_valid(const char *m) {
    return strcmp(m, "ijk") == 0 || strcmp(m, "ikj") == 0 ||
           strcmp(m, "block") == 0 || strcmp(m, "csr") == 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        die("usage: matmul2 {selftest|probe|bench} --method m --n n [...]");
    }
    const char *mode = argv[1];
    if (strcmp(mode, "selftest") == 0) {
        return run_selftest() == 0 ? 0 : 1;
    }
    if (strcmp(mode, "probe") != 0 && strcmp(mode, "bench") != 0) {
        die("unknown mode (expected: selftest, probe or bench)");
    }

    const char *method = require_arg(argc, argv, "--method");
    if (!method_valid(method)) {
        die("invalid --method (expected: ijk, ikj, block or csr)");
    }
    long long n_arg = long_arg(argc, argv, "--n");
    if (n_arg <= 0) {
        die("missing or invalid --n");
    }
    int n = (int)n_arg;
    long long sparsity = long_arg(argc, argv, "--sparsity");
    if (sparsity < 0) {
        sparsity = 0;
    }
    long long bs_arg = long_arg(argc, argv, "--blocksize");
    int bs = bs_arg > 0 ? (int)bs_arg : 64;

    if (find_arg(argc, argv, "--pin") != NULL) {
        if (!pin_pcore()) {
            die("failed to pin thread affinity");
        }
    }

    const char *pa = require_arg(argc, argv, "--a");
    const char *pb = require_arg(argc, argv, "--b");
    long count = (long)n * (long)n;

    job jb;
    jb.method = method;
    jb.n = n;
    jb.bs = bs;
    jb.A = load_matrix(pa, n);
    jb.B = load_matrix(pb, n);
    jb.C = (double *)xmalloc((size_t)count * sizeof(double));
    jb.sp = NULL;

    if (strcmp(mode, "probe") == 0) {
        long long convert_ns = -1;
        if (strcmp(method, "csr") == 0) {
            double c0 = now_ns();
            jb.sp = csr_build(jb.A, n);
            convert_ns = (long long)(now_ns() - c0);
            if (jb.sp == NULL) {
                die("csr_build failed");
            }
        }
        double w0 = now_ns();
        run_kernel(&jb);
        long long warmup_ns = (long long)(now_ns() - w0);
        double t0 = now_ns();
        run_kernel(&jb);
        long long kernel_ns = (long long)(now_ns() - t0);
        emit("{\"mode\":\"probe\",\"method\":\"%s\",\"n\":%d,"
              "\"sparsity\":%lld,\"warmup_ns\":%lld,\"kernel_total_ns\":%lld,"
              "\"convert_ns\":%lld,\"checksum\":%.17g,"
              "\"peak_wss_bytes\":%lld}",
              method, n, sparsity, warmup_ns, kernel_ns, convert_ns,
              checksum(jb.C, count), peak_wss_bytes());
        free((void *)jb.A);
        free((void *)jb.B);
        free(jb.C);
        csr_free(jb.sp);
        return 0;
    }

    long long warmups = long_arg(argc, argv, "--warmups");
    long long reps = long_arg(argc, argv, "--reps");
    long long budget_ns = long_arg(argc, argv, "--budget-ns");
    long long min_batch_ns = long_arg(argc, argv, "--min-batch-ns");
    if (warmups < 1 || reps < 1 || budget_ns < 1 || min_batch_ns < 1) {
        die("bench requires --warmups >= 1, --reps >= 1, --budget-ns >= 1, "
            "--min-batch-ns >= 1");
    }

    if (strcmp(method, "csr") == 0) {
        for (long long r = 0; r < reps; r++) {
            csr_free(jb.sp);
            double c0 = now_ns();
            jb.sp = csr_build(jb.A, n);
            long long convert_ns = (long long)(now_ns() - c0);
            if (jb.sp == NULL) {
                die("csr_build failed");
            }
            emit("{\"mode\":\"bench\",\"phase\":\"convert\",\"method\":\"csr\","
                  "\"n\":%d,\"sparsity\":%lld,\"rep\":%lld,"
                  "\"convert_ns\":%lld,\"nnz\":%lld}",
                  n, sparsity, r, convert_ns, jb.sp->nnz);
        }
    }

    long long warmup_ns = 0;
    for (long long w = 0; w < warmups; w++) {
        double t0 = now_ns();
        run_kernel(&jb);
        warmup_ns = (long long)(now_ns() - t0);
        emit("{\"mode\":\"bench\",\"phase\":\"warmup\",\"method\":\"%s\","
              "\"n\":%d,\"sparsity\":%lld,\"rep\":%lld,\"batch\":1,"
              "\"kernel_total_ns\":%lld,\"kernel_per_mult_ns\":%lld,"
              "\"checksum\":%.17g}",
              method, n, sparsity, w, warmup_ns, warmup_ns,
              checksum(jb.C, count));
    }
    if (warmup_ns > budget_ns) {
        emit("{\"mode\":\"timeout\",\"method\":\"%s\",\"n\":%d,"
              "\"reason\":\"warm-up exceeded budget\","
              "\"kernel_total_ns\":%lld}",
              method, n, warmup_ns);
        return 3;
    }

    long long batch = (min_batch_ns + warmup_ns - 1) / (warmup_ns > 0 ? warmup_ns : 1);
    if (batch < 1) {
        batch = 1;
    }
    if (batch > BATCH_CAP) {
        batch = BATCH_CAP;
    }
    for (;;) {
        double c0 = now_ns();
        for (long long b = 0; b < batch; b++) {
            run_kernel(&jb);
        }
        double calib = now_ns() - c0;
        if (calib >= (double)min_batch_ns || batch >= BATCH_CAP) {
            break;
        }
        batch = batch > BATCH_CAP / 2 ? BATCH_CAP : batch * 2;
    }

    for (long long r = 0; r < reps; r++) {
        double t0 = now_ns();
        for (long long b = 0; b < batch; b++) {
            run_kernel(&jb);
        }
        long long total = (long long)(now_ns() - t0);
        emit("{\"mode\":\"bench\",\"phase\":\"measure\",\"method\":\"%s\","
              "\"n\":%d,\"sparsity\":%lld,\"rep\":%lld,\"batch\":%lld,"
              "\"kernel_total_ns\":%lld,\"kernel_per_mult_ns\":%.17g,"
              "\"checksum\":%.17g}",
              method, n, sparsity, r, batch, total,
              (double)total / (double)batch, checksum(jb.C, count));
    }

    emit("{\"mode\":\"done\",\"method\":\"%s\",\"n\":%d,"
          "\"peak_wss_bytes\":%lld}",
          method, n, peak_wss_bytes());

    free((void *)jb.A);
    free((void *)jb.B);
    free(jb.C);
    csr_free(jb.sp);
    return 0;
}
