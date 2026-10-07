#!/usr/bin/env python3
"""Basic dense matrix multiplication in Python (list of lists, triple loop).

Standard library only (plus optional psutil for the memory self-report). The
kernel is the classical O(n^3) i-j-k triple loop over IEEE-754 doubles (Python
floats are C doubles); no optimized library is used anywhere near the
multiplication.

Modes:
    multiply --a A.bin --b B.bin --n n --out C.bin
        Load A and B (little-endian float64, row-major), run the kernel once,
        write the product C in the same format and print a checksum.
    probe --a A.bin --b B.bin --n n
        Load, run the kernel exactly once (monotonic timer around the kernel
        only) and print a JSON line. Used by the orchestrator as a budget gate.
    bench --a A.bin --b B.bin --n n --warmups W --reps R
         [--budget-ns N] [--min-batch-ns N]
        Load outside the timed region, run W warm-up multiplications (excluded
        from the statistics), calibrate a batch count so that each measured
        repetition lasts at least --min-batch-ns, then run R measured
        repetitions. One JSON line is emitted per repetition (flushed
        immediately) plus a final "done" line with the peak working set.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from array import array
from pathlib import Path

BATCH_CAP = 1_000_000
LANG = "python"


def matmul_into(A: list[list[float]], B: list[list[float]], C: list[list[float]]) -> None:
    n = len(A)
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[i][k] * B[k][j]
            C[i][j] = s


def load_matrix(path: str | Path, n: int) -> list[list[float]]:
    data = Path(path).read_bytes()
    expected = 8 * n * n
    if len(data) != expected:
        raise ValueError(f"{path}: expected {expected} bytes for n={n}, got {len(data)}")
    flat = array("d")
    flat.frombytes(data)
    return [list(flat[i * n : (i + 1) * n]) for i in range(n)]


def save_matrix(path: str | Path, M: list[list[float]]) -> None:
    flat = array("d")
    for row in M:
        flat.extend(row)
    Path(path).write_bytes(flat.tobytes())


def checksum(M: list[list[float]]) -> float:
    s = 0.0
    for row in M:
        for v in row:
            s += v
    return s


def now_ns() -> int:
    return time.perf_counter_ns()


def emit(obj: dict) -> None:
    print(json.dumps(obj), flush=True)


def peak_wss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().peak_wset)
    except Exception:
        return None


def run_multiply(args: argparse.Namespace) -> int:
    A = load_matrix(args.a, args.n)
    B = load_matrix(args.b, args.n)
    C = [[0.0] * args.n for _ in range(args.n)]
    matmul_into(A, B, C)
    save_matrix(args.out, C)
    print(f"n={args.n} checksum={checksum(C):.17g}")
    return 0


def run_probe(args: argparse.Namespace) -> int:
    A = load_matrix(args.a, args.n)
    B = load_matrix(args.b, args.n)
    C = [[0.0] * args.n for _ in range(args.n)]
    t0 = now_ns()
    matmul_into(A, B, C)
    t1 = now_ns()
    emit({"mode": "probe", "lang": LANG, "n": args.n,
          "kernel_total_ns": t1 - t0, "checksum": checksum(C)})
    return 0


def run_bench(args: argparse.Namespace) -> int:
    if args.warmups < 1:
        raise SystemExit("ERROR: --warmups must be >= 1 (needed for batch calibration)")
    A = load_matrix(args.a, args.n)
    B = load_matrix(args.b, args.n)
    C = [[0.0] * args.n for _ in range(args.n)]

    warmup_ns = 0
    for w in range(args.warmups):
        t0 = now_ns()
        matmul_into(A, B, C)
        t1 = now_ns()
        warmup_ns = t1 - t0
        emit({"mode": "bench", "lang": LANG, "n": args.n, "phase": "warmup", "rep": w,
              "batch": 1, "kernel_total_ns": warmup_ns,
              "kernel_per_mult_ns": float(warmup_ns), "checksum": checksum(C)})
    if warmup_ns > args.budget_ns:
        emit({"mode": "timeout", "lang": LANG, "n": args.n,
              "reason": "warm-up exceeded budget", "kernel_total_ns": warmup_ns})
        return 3

    batch = max(1, min(BATCH_CAP, math.ceil(args.min_batch_ns / max(warmup_ns, 1))))
    for r in range(args.reps):
        t0 = now_ns()
        for _ in range(batch):
            matmul_into(A, B, C)
        t1 = now_ns()
        total = t1 - t0
        emit({"mode": "bench", "lang": LANG, "n": args.n, "phase": "measure", "rep": r,
              "batch": batch, "kernel_total_ns": total,
              "kernel_per_mult_ns": total / batch, "checksum": checksum(C)})

    emit({"mode": "done", "lang": LANG, "n": args.n,
          "peak_wss_bytes": peak_wss_bytes(), "heap_used_bytes": None})
    return 0


def main(argv: list[str] | None = None) -> int:
    if sys.byteorder != "little":
        raise SystemExit("ERROR: little-endian host required for the shared binary format")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    def io_args(sp, out: bool) -> None:
        sp.add_argument("--a", required=True, help="input matrix A (bin)")
        sp.add_argument("--b", required=True, help="input matrix B (bin)")
        sp.add_argument("--n", type=int, required=True, help="side length")
        if out:
            sp.add_argument("--out", required=True, help="output matrix C (bin)")

    m = sub.add_parser("multiply")
    io_args(m, out=True)
    pr = sub.add_parser("probe")
    io_args(pr, out=False)
    b = sub.add_parser("bench")
    io_args(b, out=False)
    b.add_argument("--warmups", type=int, required=True, help="warm-up runs (excluded from stats)")
    b.add_argument("--reps", type=int, required=True, help="measured repetitions")
    b.add_argument("--budget-ns", type=int, required=True, help="max time for one multiplication")
    b.add_argument("--min-batch-ns", type=int, required=True, help="min timed-region length")

    args = p.parse_args(argv)
    if args.mode == "multiply":
        return run_multiply(args)
    if args.mode == "probe":
        return run_probe(args)
    return run_bench(args)


if __name__ == "__main__":
    raise SystemExit(main())
