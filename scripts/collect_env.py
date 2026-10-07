#!/usr/bin/env python3
"""Collect the execution environment into data/results/environment.json.

Run before the benchmark campaign so the report can describe the machine,
toolchains and the fixed protocol parameters.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "results" / "environment.json"

BUDGET_S = 60
SIZES = [10, 50, 100, 200, 500, 1000, 2000]
REPS = 5
MIN_BATCH_NS = 200_000_000  # >=0.2 s per timing
WARMUPS_PYTHON_C = 1
WARMUPS_JAVA = {"n_lt_50": 20000, "n_le_100": 20, "n_ge_200": 5}
JAVA_BENCH_FLAGS = ["-XX:CompileCommand=dontinline,MatMul::matmulInto"]
SEED = 42
MEM_LIMIT_BYTES = 4 * 1024**3


def _run(cmd: list[str]) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=30)
        return (r.stdout or r.stderr or "").strip()
    except Exception as exc:  # pragma: no cover - best effort diagnostics
        return f"<unavailable: {exc}>"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    cpu_name = _run(["powershell", "-NoProfile", "-Command",
                     "(Get-CimInstance Win32_Processor).Name"])
    java_version = _run(["java", "-version"])
    gcc_version = _run(["gcc", "--version"]).splitlines()
    try:
        import numpy
        numpy_version = numpy.__version__
    except Exception:
        numpy_version = None

    git_commit = _run(["git", "rev-parse", "HEAD"])
    git_dirty = bool(_run(["git", "status", "--porcelain"]))

    env = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_name": cpu_name,
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "ram_total_bytes": psutil.virtual_memory().total,
        "python_version": sys.version,
        "java_version": java_version,
        "gcc_version": gcc_version[0] if gcc_version else None,
        "numpy_version": numpy_version,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "protocol": {
            "budget_s": BUDGET_S,
            "sizes": SIZES,
            "reps": REPS,
            "min_batch_ns": MIN_BATCH_NS,
            "warmups_python_c": WARMUPS_PYTHON_C,
            "warmups_java": WARMUPS_JAVA,
            "java_bench_flags": JAVA_BENCH_FLAGS,
            "probe": "one warm-up kernel + one timed kernel; gate = timed <= budget; "
                     "one retry if the probe completed but failed the gate",
            "seed": SEED,
            "mem_limit_bytes": MEM_LIMIT_BYTES,
            "statistical_estimator": "median + IQR over reps (warm-ups excluded)",
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(env, indent=2), encoding="utf-8")
    print(f"environment written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
