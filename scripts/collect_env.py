#!/usr/bin/env python3
"""Collect the execution environment into data/results/environment.json.

Run before the benchmark campaign so the report can describe the machine,
toolchains, cache hierarchy and the fixed protocol parameters.
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
SPARSITIES = [1, 5, 10, 25, 50, 75, 90, 95, 99]
SPARSE_N = 1000
METHODS = ["ijk", "ikj", "block", "csr"]
REPS = 5
WARMUPS = 1
MIN_BATCH_NS = 200_000_000
SEED = 42
MEM_LIMIT_BYTES = 4 * 1024**3
AFFINITY = "child pinned to CPU0 (P-core) via SetThreadAffinityMask"


def _run(cmd: list[str]) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=30)
        return (r.stdout or r.stderr or "").strip()
    except Exception as exc:
        return f"<unavailable: {exc}>"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--blocksize", type=int, default=64)
    args = ap.parse_args()

    cpu_name = _run(["powershell", "-NoProfile", "-Command",
                     "(Get-CimInstance Win32_Processor).Name"])
    l2_kb = _run(["powershell", "-NoProfile", "-Command",
                  "(Get-CimInstance Win32_Processor).L2CacheSize"])
    l3_kb = _run(["powershell", "-NoProfile", "-Command",
                  "(Get-CimInstance Win32_Processor).L3CacheSize"])
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
        "cache": {
            "line_bytes": 64,
            "line_bytes_source": "Intel Raptor Lake specification (WMI returns null)",
            "l1d_per_pcore_kb": 48,
            "l1d_source": "Intel Raptor Lake specification (WMI returns null)",
            "l2_total_kb": l2_kb,
            "l3_total_kb": l3_kb,
            "l2_source": "Win32_Processor.L2CacheSize "
                         "(2x1.25 MB P-cores + 4 MB E-cluster)",
        },
        "python_version": sys.version,
        "gcc_version": gcc_version[0] if gcc_version else None,
        "numpy_version": numpy_version,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "protocol": {
            "budget_s": BUDGET_S,
            "mem_limit_bytes": MEM_LIMIT_BYTES,
            "sizes": SIZES,
            "sparsity_levels_percent": SPARSITIES,
            "sparse_n": SPARSE_N,
            "methods": METHODS,
            "reps": REPS,
            "warmups": WARMUPS,
            "min_batch_ns": MIN_BATCH_NS,
            "blocksize": args.blocksize,
            "affinity": AFFINITY,
            "compiler_flags": "-O2 -std=c11 -Wall -Wextra",
            "seed": SEED,
            "probe": "one warm-up kernel (excluded) + one timed kernel; "
                     "gate = timed <= budget; one retry if the probe "
                     "completed but failed the gate",
            "conversion": "dense-to-CSR conversions timed separately, never "
                          "inside the multiplication timing",
            "statistical_estimator": "median + IQR over reps (warm-ups excluded)",
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(env, indent=2), encoding="utf-8")
    print(f"environment written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
