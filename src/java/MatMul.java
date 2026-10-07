import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

/**
 * Basic dense matrix multiplication in Java (double[][], triple loop).
 *
 * Standard JDK only. The kernel is the classical O(n^3) i-j-k triple loop over
 * IEEE-754 doubles; no optimized library is used anywhere near the
 * multiplication.
 *
 * Modes:
 *   multiply --a A.bin --b B.bin --n n --out C.bin
 *       Load A and B (little-endian float64, row-major), run the kernel once,
 *       write the product C in the same format and print a checksum.
 *   probe --a A.bin --b B.bin --n n
 *       Load, run the kernel exactly once (System.nanoTime around the kernel
 *       only) and print a JSON line. Used by the orchestrator as a budget gate.
 *   bench --a A.bin --b B.bin --n n --warmups W --reps R
 *         [--budget-ns N] [--min-batch-ns N]
 *       Load outside the timed region, run W warm-up multiplications (excluded
 *       from the statistics, they trigger JIT compilation), calibrate a batch
 *       count so that each measured repetition lasts at least --min-batch-ns,
 *       then run R measured repetitions. One JSON line per repetition (flushed
 *       immediately) plus a final "done" line with heap usage.
 */
public class MatMul {

    private static final String LANG = "java";
    private static final long BATCH_CAP = 1_000_000L;

    private static void die(String msg) {
        System.err.println("ERROR: " + msg);
        System.exit(1);
    }

    private static String findArg(String[] args, String key) {
        for (int i = 1; i < args.length - 1; i++) {
            if (args[i].equals(key)) {
                return args[i + 1];
            }
        }
        return null;
    }

    private static String requireArg(String[] args, String key) {
        String v = findArg(args, key);
        if (v == null) {
            die("missing required argument " + key);
        }
        return v;
    }

    private static long longArg(String[] args, String key, long dflt) {
        String v = findArg(args, key);
        if (v == null) {
            return dflt;
        }
        try {
            return Long.parseLong(v);
        } catch (NumberFormatException e) {
            die("invalid " + key + ": " + v);
            return -1;
        }
    }

    private static void emit(String json) {
        System.out.println(json);
        System.out.flush();
    }

    private static double[][] loadMatrix(String path, int n) {
        byte[] bytes;
        try {
            bytes = Files.readAllBytes(Path.of(path));
        } catch (IOException e) {
            die("cannot read input file: " + e.getMessage());
            return null;
        }
        long want = 8L * n * (long) n;
        if (bytes.length != want) {
            die("input file size mismatch: expected " + want + " bytes, got " + bytes.length);
        }
        ByteBuffer buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        double[][] M = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                M[i][j] = buf.getDouble();
            }
        }
        return M;
    }

    static void matmulInto(double[][] A, double[][] B, double[][] C) {
        int n = A.length;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double s = 0.0;
                for (int k = 0; k < n; k++) {
                    s += A[i][k] * B[k][j];
                }
                C[i][j] = s;
            }
        }
    }

    static double checksum(double[][] M) {
        double s = 0.0;
        for (double[] row : M) {
            for (double v : row) {
                s += v;
            }
        }
        return s;
    }

    private static long heapUsedBytes() {
        Runtime rt = Runtime.getRuntime();
        return rt.totalMemory() - rt.freeMemory();
    }

    private static void saveMatrix(String path, double[][] M, int n) {
        ByteBuffer buf = ByteBuffer.allocate(8 * n * n).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                buf.putDouble(M[i][j]);
            }
        }
        try {
            Files.write(Path.of(path), buf.array());
        } catch (IOException e) {
            die("cannot write output file: " + e.getMessage());
        }
    }

    private static int parseIntArg(String[] args, String key) {
        String v = requireArg(args, key);
        try {
            return Integer.parseInt(v);
        } catch (NumberFormatException e) {
            die("invalid " + key + ": " + v);
            return -1;
        }
    }

    public static void main(String[] args) {
        if (args.length == 0) {
            die("usage: java MatMul {multiply|probe|bench} --a A.bin --b B.bin --n n [...]");
        }
        String mode = args[0];
        if (!mode.equals("multiply") && !mode.equals("probe") && !mode.equals("bench")) {
            die("unknown mode (expected: multiply, probe or bench)");
        }
        String pa = requireArg(args, "--a");
        String pb = requireArg(args, "--b");
        int n = parseIntArg(args, "--n");
        if (n <= 0) {
            die("invalid --n: " + n);
        }

        if (mode.equals("multiply")) {
            String pout = requireArg(args, "--out");
            double[][] A = loadMatrix(pa, n);
            double[][] B = loadMatrix(pb, n);
            double[][] C = new double[n][n];
            matmulInto(A, B, C);
            saveMatrix(pout, C, n);
            System.out.printf(Locale.ROOT, "n=%d checksum=%.17g%n", n, checksum(C));
            return;
        }

        double[][] A = loadMatrix(pa, n);
        double[][] B = loadMatrix(pb, n);
        double[][] C = new double[n][n];

        if (mode.equals("probe")) {
            // One untimed kernel first: the budget gate must see the JVM in a
            // warm state (JIT compilation), not the interpreter speed of the
            // very first run. The timed kernel is the gate measurement.
            matmulInto(A, B, C);
            long t0 = System.nanoTime();
            matmulInto(A, B, C);
            long t1 = System.nanoTime();
            emit("{\"mode\":\"probe\",\"lang\":\"" + LANG + "\"," +
                    "\"n\":" + n + ",\"kernel_total_ns\":" + (t1 - t0) +
                    ",\"checksum\":" + Double.toString(checksum(C)) + "}");
            return;
        }

        // bench
        long warmups = longArg(args, "--warmups", -1);
        long reps = longArg(args, "--reps", -1);
        long budgetNs = longArg(args, "--budget-ns", -1);
        long minBatchNs = longArg(args, "--min-batch-ns", -1);
        if (warmups < 1 || reps < 1 || budgetNs < 1 || minBatchNs < 1) {
            die("bench requires --warmups >= 1, --reps >= 1, --budget-ns >= 1, --min-batch-ns >= 1");
        }

        long warmupNs = 0;
        for (long w = 0; w < warmups; w++) {
            long t0 = System.nanoTime();
            matmulInto(A, B, C);
            long t1 = System.nanoTime();
            warmupNs = t1 - t0;
            emit("{\"mode\":\"bench\",\"lang\":\"" + LANG + "\",\"n\":" + n +
                    ",\"phase\":\"warmup\",\"rep\":" + w + ",\"batch\":1" +
                    ",\"kernel_total_ns\":" + warmupNs +
                    ",\"kernel_per_mult_ns\":" + Double.toString((double) warmupNs) +
                    ",\"checksum\":" + Double.toString(checksum(C)) + "}");
        }
        // Budget applies to the steady state only: the first warm-up of a fresh
        // JVM runs mostly interpreted and must not fail the gate.
        if (warmupNs > budgetNs) {
            emit("{\"mode\":\"timeout\",\"lang\":\"" + LANG + "\",\"n\":" + n +
                    ",\"reason\":\"warm-up exceeded budget\"," +
                    "\"kernel_total_ns\":" + warmupNs + "}");
            System.exit(3);
        }

        long batch = Math.max(1, Math.min(BATCH_CAP,
                (minBatchNs + warmupNs - 1) / Math.max(warmupNs, 1)));
        for (long r = 0; r < reps; r++) {
            long t0 = System.nanoTime();
            for (long b = 0; b < batch; b++) {
                matmulInto(A, B, C);
            }
            long t1 = System.nanoTime();
            long total = t1 - t0;
            emit("{\"mode\":\"bench\",\"lang\":\"" + LANG + "\",\"n\":" + n +
                    ",\"phase\":\"measure\",\"rep\":" + r + ",\"batch\":" + batch +
                    ",\"kernel_total_ns\":" + total +
                    ",\"kernel_per_mult_ns\":" + Double.toString((double) total / batch) +
                    ",\"checksum\":" + Double.toString(checksum(C)) + "}");
        }

        emit("{\"mode\":\"done\",\"lang\":\"" + LANG + "\",\"n\":" + n +
                ",\"peak_wss_bytes\":null,\"heap_used_bytes\":" + heapUsedBytes() + "}");
    }
}
