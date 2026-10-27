#!/usr/bin/env python3
"""Capture the Assignment 3 execution environment into data/results/environment.json.

Records OS/CPU/RAM, GCC and its native ISA capabilities (AVX2/FMA/AVX-512),
the OpenMP runtime used, power plan, Python/numpy/psutil versions, the git
commit, and the full benchmark protocol (sizes, thread grids, experiments,
budgets, affinity masks).

Usage:
    python scripts/collect_env.py
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"


def run(cmd: list[str]) -> str:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return (p.stdout or p.stderr).strip()
    except Exception as exc:  # noqa: BLE001
        return f"(failed: {exc})"


def native_isa_flags() -> dict:
    out = run(["gcc", "-march=native", "-Q", "--help=target"])
    flags = {}
    for name in ("march", "mtune", "mavx", "mavx2", "mfma", "msse4.2",
                 "mavx512f", "mavx512bw", "mavx512vl", "mavx512dq"):
        m = re.search(rf"^\s+-{name}\s+\[([^\]]+)\]", out, re.M)
        if m:
            flags[name] = m.group(1)
    m = re.search(r"^\s+-march=([^\s]+)", out, re.M)
    march = m.group(1) if m else "?"
    m = re.search(r"^\s+-mtune=([^\s]+)", out, re.M)
    mtune = m.group(1) if m else "?"
    return {"march_native": march, "mtune_native": mtune, "flags": flags}


def power_plan() -> str:
    return run(["powercfg", "/getactivescheme"])


def cpu_topology() -> dict:
    info = psutil.cpu_freq()
    win = run(["wmic", "cpu", "get", "Name,NumberOfCores,NumberOfLogicalProcessors"
               "/format:list"])
    return {
        "psutil_logical": psutil.cpu_count(logical=True),
        "psutil_physical": psutil.cpu_count(logical=False),
        "min_mhz": info.min if info else None,
        "max_mhz": info.max if info else None,
        "current_mhz": info.current if info else None,
        "wmic": win,
        "hybrid_note": "Intel i7-1355U: 2 P-cores (2 threads each, CPU0-3) "
                       "+ 8 E-cores (1 thread each, CPU4-11); 12 logical CPUs",
    }


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    env = {
        "description": "Execution environment for Assignment 3 "
                       "(OpenMP parallel + SIMD dense ikj)",
        "collected_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "psutil": psutil.__version__,
        "cpu": cpu_topology(),
        "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 2),
        "compiler": run(["gcc", "--version"]).splitlines()[0],
        "compiler_flags": {
            "seq": "-std=c11 -O2 -march=native -Wall -Wextra (no OpenMP)",
            "scalar": "-std=c11 -O2 -march=native -fopenmp "
                      "-fno-tree-vectorize -DUSE_OMP",
            "auto": "-std=c11 -O2 -march=native -fopenmp -DUSE_OMP",
            "simd": "-std=c11 -O2 -march=native -fopenmp -DUSE_OMP "
                    "-DUSE_OMP_SIMD",
            "link": "-lpsapi -lm",
        },
        "isa_native": native_isa_flags(),
        "openmp": {
            "runtime": "GCC libgomp (MinGW-w64 ucrt64, winpthreads)",
            "api": "omp_set_dynamic(0) + omp_set_num_threads(p) per run",
            "scheduling": "schedule(static) over row index i (rows of C)",
            "wait_policy": "libgomp default (active spin during parallel "
                           "regions); OMP_WAIT_POLICY not set",
            "affinity_caveat": "libgomp on Windows has no reliable "
                               "OMP_PROC_BIND; OS scheduler places threads",
        },
        "affinity_masks": {
            "all": "SetProcessAffinityMask 0xFFF (CPU0-11, all logical CPUs)",
            "pcore": "SetProcessAffinityMask 0xF (CPU0-3, P-core threads only)",
        },
        "power_plan": power_plan(),
        "protocol": {
            "sizes": [50, 100, 200, 500, 1000, 2000],
            "scale_variant": "matmul3_simd (parallel + #pragma omp simd)",
            "scale_reference_rows": "an earlier full-grid sweep with "
                                    "matmul3_auto (compiler default, inner "
                                    "loop NOT vectorised) is kept for the "
                                    "SIMD comparison",
            "scale_threads": [1, 2, 4, 8, 12, 16],
            "simd_sizes": [200, 500, 1000, 2000],
            "simd_threads": [1, 8, 12],
            "pcore_sizes": [500, 1000, 2000],
            "pcore_threads": [1, 2, 3, 4],
            "pcore_variant": "matmul3_simd (CPU0-3 mask)",
            "warmups": 1,
            "reps": 5,
            "probe_budget_s": 60,
            "min_batch_ns": 200_000_000,
            "gate": "bench budget = min(60 s, 2 x max probe); skip if probe > 60 s",
            "timing_scope": "kernel only (matmul incl. memset of C); "
                            "load + checksum outside timed region",
            "inputs": "data/inputs/A_n.bin, B_n.bin (identical to Assignments 1-2)",
        },
        "git_commit": run(["git", "rev-parse", "--short", "HEAD"]),
        "commands": [
            "python scripts/build.py",
            "python scripts/generate_inputs.py",
            "python scripts/run_benchmark3.py --phase all",
            "python scripts/collect_env.py",
        ],
    }
    out = RESULTS / "environment.json"
    out.write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
