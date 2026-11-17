#!/usr/bin/env python3
"""Capture the Assignment 4 execution environment into data/results/environment.json.

Records OS/CPU/RAM, the Java and PySpark toolchain, the local standalone
cluster topology (1 master + N worker processes on one physical machine),
the Hadoop/winutils helper status, the power plan, Python/numpy/pandas
versions, the git commit, and the full benchmark protocol (sizes, worker
grids, experiments, budgets, timing boundaries).

Usage:
    python scripts/collect_env.py
"""

from __future__ import annotations

import json
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"


def run(cmd: list[str]) -> str:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return (p.stdout or p.stderr).strip()
    except Exception as exc:  # noqa: BLE001
        return f"(failed: {exc})"


def java_version() -> str:
    out = run(["java", "-version"])
    first = out.splitlines()[0] if out else "?"
    return first


def spark_version() -> str:
    try:
        import pyspark

        return pyspark.__version__
    except Exception as exc:  # noqa: BLE001
        return f"(failed: {exc})"


def hadoop_helpers() -> dict:
    home = ROOT / "build" / "hadoop"
    winutils = home / "bin" / "winutils.exe"
    dll = home / "bin" / "hadoop.dll"
    return {
        "HADOOP_HOME": str(home) if winutils.exists() else None,
        "winutils_present": winutils.exists(),
        "hadoop_dll_present": dll.exists(),
        "why": "Spark's event log is written through Hadoop's LocalFileSystem; "
               "on Windows that path needs %HADOOP_HOME%\\bin\\winutils.exe. "
               "Fetched by scripts/fetch_winutils.py into build/hadoop/ "
               "(git-ignored third-party binaries).",
        "scope_note": "Only Hadoop FileSystem calls need it; matrix I/O in "
                      "this study uses plain numpy file I/O.",
    }


def cluster_topology() -> dict:
    state = ROOT / "build" / "spark" / "cluster.json"
    if not state.exists():
        return {"running": False,
                "note": "no build/spark/cluster.json; start with "
                        "python scripts/spark_cluster.py start --workers N"}
    info = json.loads(state.read_text(encoding="utf-8"))
    return {
        "running": True,
        "master_url": info["master_url"],
        "workers": info["workers"],
        "worker_cores": info["worker_cores"],
        "worker_memory": info["worker_memory"],
        "pids": info["pids"],
        "started_utc": info["started_utc"],
        "physical_machines": 1,
        "topology_note": "1 driver process + 1 master process + N worker "
                         "processes (1 JVM each, 1 core each), all bound to "
                         "127.0.0.1 on a single physical laptop.",
    }


def power_plan() -> str:
    return run(["powercfg", "/getactivescheme"])


def cpu_topology() -> dict:
    info = psutil.cpu_freq()
    return {
        "psutil_logical": psutil.cpu_count(logical=True),
        "psutil_physical": psutil.cpu_count(logical=False),
        "min_mhz": info.min if info else None,
        "max_mhz": info.max if info else None,
        "current_mhz": info.current if info else None,
        "hybrid_note": "Intel i7-1355U: 2 P-cores (2 threads each, CPU0-3) "
                       "+ 8 E-cores (1 thread each, CPU4-11); 12 logical CPUs",
    }


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    env = {
        "description": "Execution environment for Assignment 4 "
                       "(distributed dense matmul, PySpark standalone)",
        "collected_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "psutil": psutil.__version__,
        "cpu": cpu_topology(),
        "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 2),
        "java": java_version(),
        "java_home": __import__("os").environ.get("JAVA_HOME"),
        "spark": {
            "pyspark": spark_version(),
            "py4j": _pkg_version("py4j"),
            "zstandard": _pkg_version("zstandard"),
            "deployment": "standalone (spark://127.0.0.1:7077)",
            "mode": "client (driver = submitting Python process)",
            "spark_home": str(_spark_home()),
            "configs": {
                "spark.driver.host": "127.0.0.1",
                "spark.driver.bindAddress": "127.0.0.1",
                "spark.pyspark.python": sys.executable,
                "spark.executor.pyspark.python": sys.executable,
                "spark.executor.memory": "1g",
                "spark.executor.memoryOverhead": "512m",
                "spark.eventLog.enabled": "true (events_*.zstd, Spark 4 v2 format)",
            },
            "windows_fixes": [
                "hostname VIVOBOOK_ALED is not a valid Spark URL host -> "
                "driver binds/advertises 127.0.0.1",
                "Windows App Execution Alias for 'python' prints the Store "
                "banner and exits -> PYSPARK_PYTHON points at the real "
                "interpreter path",
                "SPARK_LOCAL_IP=127.0.0.1 so worker daemons bind loopback",
            ],
        },
        "hadoop_helpers": hadoop_helpers(),
        "cluster": cluster_topology(),
        "compiler_baselines": {
            "note": "sequential + OpenMP baselines reused unchanged from "
                    "Assignment 3 (same source, same flags)",
            "gcc": run(["gcc", "--version"]).splitlines()[0],
            "flags": {
                "seq": "-std=c11 -O2 -march=native -Wall -Wextra (no OpenMP)",
                "simd": "-std=c11 -O2 -march=native -fopenmp -DUSE_OMP "
                        "-DUSE_OMP_SIMD",
                "link": "-lpsapi -lm",
            },
        },
        "power_plan": {
            "benchmark_plan": "High performance "
                              "(8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c); "
                              "switched by run_benchmark4.py before the "
                              "campaign and restored afterwards",
            "resting_plan": power_plan(),
            "why": "Balanced throttled the BLAS kernel by ~2-3x, which would "
                   "have confounded the Spark/C/numpy comparison",
        },
        "protocol": {
            "sizes_strong": [2000, 4000],
            "sizes_weak": {"1": 2000, "2": 2500, "4": 3200, "8": 4000},
            "weak_scaling_rule": "n_total = n_1 * p^(1/3) so that n^3/p "
                                 "(work per worker) stays constant",
            "sizes_size_scaling": [500, 1000, 2000, 4000],
            "workers_grid": [1, 2, 4, 8],
            "worker_definition": "1 worker = 1 standalone worker process "
                                 "(1 JVM, 1 core); drivers and masters are "
                                 "separate processes; 1 physical machine",
            "distributed_variant": "row-block partition of A + B replicated "
                                   "to every worker with sc.addFile; "
                                   "numpy/BLAS matmul per block",
            "distributed_reps": 3,
            "baseline_reps": 5,
            "summary_statistic": "median",
            "warmups": 1,
            "heat_soak_s": 300,
            "min_batch_ns": 0,
            "min_batch_note": "0 forces batch = 1 in the C bench mode, so "
                              "every measured batch is exactly one "
                              "multiplication",
            "timing_boundary": "additive driver phases per repetition: "
                               "t_e2e = t_stage_a (split + serialize A) + "
                               "t_action_job (dispatch, transfer, compute, "
                               "collect) + t_assemble (rebuild C)",
            "startup_excluded": "t_startup (SparkSession + executor launch) "
                                "and t_replicate (addFile + forcing job that "
                                "materialises B everywhere) are one-time "
                                "per application and reported separately; "
                                "the per-task Python-worker respawn is NOT "
                                "excluded, because on Windows a fresh worker "
                                "process is started for every task",
            "baselines": [
                "numpy_sequential: same BLAS kernel, 1 process, 1 thread "
                "(isolates distribution overhead from kernel speed)",
                "matmul3_seq: Assignment 3 sequential C kernel",
                "matmul3_simd: Assignment 3 OpenMP/SIMD C kernel at "
                "p in {1,2,4,8} threads",
            ],
            "known_limitations": [
                "sc.broadcast is unusable on Windows: PythonBroadcast."
                "readObject moves its temp file into the block manager while "
                "the FileOutputStream is still open, and Files.move fails "
                "with ERROR_SHARING_VIOLATION. Verified against the bundled "
                "JDK 21; sc.addFile is used instead (same 8n^2 bytes per "
                "worker).",
                "Python workers cannot be reused on Windows: "
                "PythonWorkerFactory.releaseWorker only requeues when "
                "useDaemon is true, which needs os.fork(). Every task pays "
                "a fresh python.exe + import pyspark.worker + numpy.",
                "The i7-1355U (2 P-cores + 8 E-cores, 15 W) caps useful "
                "parallelism at ~1.9x: measured 1-thread 65 GFLOP/s vs "
                "8-thread 121 GFLOP/s on the same BLAS kernel.",
                "All workers are processes on ONE physical machine, never "
                "separate nodes.",
            ],
            "validation": "distributed C vs matmul3_seq C on identical "
                          "inputs, rtol=1e-9, atol=1e-12 (numpy/BLAS "
                          "summation order differs from ikj)",
            "inputs": "data/inputs/A_n.bin, B_n.bin (same generator/seed as "
                      "Assignments 1-3, byte-identical where sizes overlap)",
        },
        "git_commit": run(["git", "rev-parse", "--short", "HEAD"]),
        "commands": [
            "python -m pip install -r requirements-a4.txt",
            "python scripts/fetch_winutils.py",
            "python scripts/generate_inputs.py",
            "python scripts/build.py",
            "python scripts/spark_cluster.py start --workers 8",
            "python scripts/run_benchmark4.py --phase all --soak-s 300",
            "python scripts/collect_env.py",
        ],
    }
    out = RESULTS / "environment.json"
    out.write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


def _pkg_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # noqa: BLE001
        return "unknown"


def _spark_home() -> Path:
    import pyspark

    return Path(pyspark.__file__).resolve().parent


if __name__ == "__main__":
    raise SystemExit(main())
