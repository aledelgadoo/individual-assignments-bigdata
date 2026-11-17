#!/usr/bin/env python3
"""Benchmark campaign orchestrator for Assignment 4 (PySpark distributed matmul).

One config = experiment x variant x n x p:

    strong   strong scaling: n in {2000, 4000} x p in {1, 2, 4, 8} row blocks
             (= workers, one core each).
    weak     weak scaling: (p, n) = (1,2000), (2,2500), (4,3200), (8,4000), so
             that n^3 / p (work per worker) stays constant at ~8e9 flop.
             Holding n per worker fixed would NOT be equal work for a cubic
             multiplication, so n grows as n_1 * p^(1/3).
    size     problem-size sweep at p = 4, n in {500, 1000, 2000, 4000}.
    baselines single-process references that need no cluster:
             numpy  the *same* BLAS kernel, one process, one thread -- this is
                    what isolates distribution overhead from kernel speed.
             c_seq  Assignment 3 sequential C kernel.
             c_omp  Assignment 3 OpenMP/SIMD C kernel with p threads.
    all      every experiment above.

There is no separate "compare" experiment: the headline table (Spark vs the C
and numpy references) is built in the analysis by joining ``strong`` at
p in {1, 8} with ``baselines`` on n, which would otherwise re-measure the same
four configurations twice.

Protocol per config (one process per multiplication, like Assignment 3):
    Spark / numpy  1 warmup + REPS measured repetitions inside a single child
                   process.  For Spark that is one application, so the
                   one-time ``t_startup`` and ``t_replicate`` are paid once
                   and reported separately from the per-repetition phases.
    C              ``bench --min-batch-ns 0``, which forces batch = 1, so every
                   measured batch is exactly one multiplication.

JSON lines from every child run go to ``data/results/raw_detail/``.

Outputs:
    data/results/raw_timings4.csv   per measured repetition
    data/results/configs4.csv       per config (status, medians, event log)

Resume: configs whose status is not ``running`` are skipped; ``--force`` redoes
them.  Spark configs are grouped by ``p`` so the cluster is started once per
distinct worker count; baseline configs never touch the cluster.  The power
plan is switched to *High performance* for the whole campaign and restored
afterwards (``--no-power`` skips that).

Usage:
    python scripts/run_benchmark4.py --phase strong|weak|size|baselines|all
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts", ROOT / "src" / "spark"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import spark_cluster  # noqa: E402
import sparkconf  # noqa: E402

INPUTS = ROOT / "data" / "inputs"
REFS = ROOT / "build" / "ref"
RESULTS = ROOT / "data" / "results"
DETAIL = RESULTS / "raw_detail"
EVENTLOG = ROOT / "build" / "spark" / "eventlog"
SPARK_SCRIPT = ROOT / "src" / "spark" / "distributed_matmul.py"
CFG4_PATH = RESULTS / "configs4.csv"
RAW4_PATH = RESULTS / "raw_timings4.csv"

# --------------------------------------------------------------------- grids
STRONG_N = (2000, 4000)
STRONG_P = (1, 2, 4, 8)
WEAK = ((1, 2000), (2, 2500), (4, 3200), (8, 4000))
SIZE_P = 4
SIZE_N = (500, 1000, 2000, 4000)
BASE_N = (500, 1000, 2000, 4000)
OMP_THREADS = (1, 2, 4, 8)

VARIANT_ORDER = {"spark": 0, "numpy": 1, "c_seq": 2, "c_omp": 3}

RAW_FIELDS = [
    "experiment", "variant", "n", "p", "rep", "measured_s",
    "stage_a_s", "action_job_s", "assemble_s", "load_s",
    "matmul_ms_max", "b_load_ms_max", "checksum",
]
CFG_FIELDS = [
    "experiment", "variant", "n", "p", "status", "reason", "reps",
    "exit_code", "wall_s", "measured_median_s", "measured_min_s",
    "measured_max_s", "startup_s", "replicate_s",
    "stage_a_s", "action_job_s", "assemble_s",
    "matmul_ms_max", "b_load_ms_max", "checksum",
    "valid", "max_abs_diff",
    "n_executors", "n_tasks", "task_span_max_ms", "run_sum_ms",
    "cpu_sum_ms", "shuffle_write_bytes", "shuffle_read_bytes", "app_id",
]


# ------------------------------------------------------------------- helpers
def build_configs(phase: str) -> list[tuple]:
    cfgs: list[tuple] = []
    if phase in ("strong", "all"):
        for n in STRONG_N:
            for p in STRONG_P:
                cfgs.append(("strong", "spark", n, p))
    if phase in ("weak", "all"):
        for p, n in WEAK:
            cfgs.append(("weak", "spark", n, p))
    if phase in ("size", "all"):
        for n in SIZE_N:
            cfgs.append(("size", "spark", n, SIZE_P))
    if phase in ("baselines", "all"):
        for n in BASE_N:
            cfgs.append(("baselines", "numpy", n, 1))
            cfgs.append(("baselines", "c_seq", n, 1))
            for t in OMP_THREADS:
                cfgs.append(("baselines", "c_omp", n, t))
    return cfgs


def sort_key(cfg: tuple) -> tuple:
    """Spark configs first, grouped by p (one cluster restart per group)."""
    exp, variant, n, p = cfg
    return (VARIANT_ORDER[variant], p if variant == "spark" else 0, n, p, exp)


def load_done() -> set[tuple]:
    done: set[tuple] = set()
    if not CFG4_PATH.exists():
        return done
    with CFG4_PATH.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") not in (None, "", "running"):
                done.add((row["experiment"], row["variant"],
                          int(row["n"]), int(row["p"])))
    return done


def child_env() -> dict[str, str]:
    """Environment for every child: one BLAS thread + the Windows fixes."""
    env = dict(os.environ)
    env.update(sparkconf.blas_thread_limits())
    env["PYSPARK_PYTHON"] = sparkconf.REAL_PYTHON
    env["PYSPARK_DRIVER_PYTHON"] = sparkconf.REAL_PYTHON
    env["SPARK_LOCAL_IP"] = sparkconf.LOCAL_HOST
    if sparkconf.HADOOP_HOME_STR:
        env["HADOOP_HOME"] = sparkconf.HADOOP_HOME_STR
        env["PATH"] = (str(Path(sparkconf.HADOOP_HOME_STR) / "bin") +
                       os.pathsep + env.get("PATH", ""))
    return env


class ChildResult:
    def __init__(self) -> None:
        self.lines: list[dict] = []
        self.exit_code = -1
        self.wall_s = 0.0
        self.timed_out = False


def run_child(cmd: list[str], detail_path: Path, deadline_s: float,
              env: dict[str, str] | None = None) -> ChildResult:
    """Run one child, stream its stdout JSON lines, enforce a wall deadline."""
    res = ChildResult()
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, bufsize=1, cwd=str(ROOT), env=env)

    def _kill() -> None:
        res.timed_out = True
        try:
            proc.kill()
        except OSError:
            pass

    killer = threading.Timer(deadline_s, _kill)
    killer.start()

    detail_lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        detail_lines.append(line)
        if line.startswith("{"):
            try:
                res.lines.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    proc.wait()
    killer.cancel()
    res.wall_s = time.monotonic() - t0
    res.exit_code = proc.returncode

    stderr_tail = ""
    if proc.stderr is not None:
        stderr_tail = proc.stderr.read()[-2000:]
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    with detail_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"cmd": cmd, "exit": res.exit_code,
                            "wall_s": round(res.wall_s, 3),
                            "timed_out": res.timed_out}) + "\n")
        for ln in detail_lines:
            f.write(ln + "\n")
        if stderr_tail:
            f.write(json.dumps({"stderr_tail": stderr_tail}) + "\n")
    return res


# --------------------------------------------------------------- power plan
HIGH_PERF_GUID = "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"
_prev_scheme: str | None = None


def _powercfg(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["powercfg", *args], capture_output=True, text=True)


def _scheme_guid(output: str) -> str | None:
    m = re.search(
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
        output)
    return m.group(0) if m else None


def set_high_performance() -> None:
    """Switch to High performance, remembering the plan to restore later."""
    global _prev_scheme
    cur = _scheme_guid(_powercfg("/getactivescheme").stdout or "")
    if cur and cur.lower() == HIGH_PERF_GUID.lower():
        print("[campaign] power plan already High performance")
        return
    _prev_scheme = cur
    rc = _powercfg("/setactive", HIGH_PERF_GUID)
    if rc.returncode != 0:
        print(f"[campaign] WARNING: power plan not switched: "
              f"{(rc.stderr or rc.stdout).strip()}", file=sys.stderr)
        _prev_scheme = None
    else:
        print("[campaign] power plan -> High performance")


def restore_power_plan() -> None:
    global _prev_scheme
    if not _prev_scheme:
        return
    rc = _powercfg("/setactive", _prev_scheme)
    if rc.returncode == 0:
        print("[campaign] power plan restored")
    else:
        print(f"[campaign] WARNING: could not restore power plan: "
              f"{(rc.stderr or rc.stdout).strip()}", file=sys.stderr)
    _prev_scheme = None


# ------------------------------------------------------------------ cluster
def _initial_workers() -> int | None:
    if not spark_cluster.STATE_FILE.exists():
        return None
    try:
        return int(json.loads(
            spark_cluster.STATE_FILE.read_text(encoding="utf-8")).get("workers"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None


_current_workers: int | None = _initial_workers()


def _cluster_ok() -> bool:
    if not spark_cluster.STATE_FILE.exists():
        return False
    try:
        state = json.loads(spark_cluster.STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    pids = state.get("pids", [])
    if not pids or not all(spark_cluster._alive(p) for p in pids):
        return False
    info = spark_cluster._master_json(timeout=3.0)
    return bool(info) and len(info.get("workers", [])) == int(state.get("workers", -1))


def ensure_cluster(workers: int) -> None:
    global _current_workers
    if _current_workers == workers and _cluster_ok():
        return
    if spark_cluster.STATE_FILE.exists():
        print(f"[campaign] cluster has {_current_workers} workers, "
              f"restarting for {workers} ...", flush=True)
        spark_cluster.stop()
    print(f"[campaign] starting cluster with {workers} workers ...", flush=True)
    rc = spark_cluster.start(workers)
    if rc != 0:
        raise SystemExit(f"[campaign] cluster start failed (rc={rc})")
    _current_workers = workers


# -------------------------------------------------------------------- child
def build_cmd(cfg: tuple, args: argparse.Namespace) -> tuple[list[str], float]:
    """Return (command, wall deadline in seconds) for one config."""
    exp, variant, n, p = cfg
    a_bin = INPUTS / f"A_{n}.bin"
    b_bin = INPUTS / f"B_{n}.bin"
    ref = REFS / f"C_{n}_seq.bin"
    ref_args = ["--ref", str(ref)] if ref.exists() else []
    env_cmd = [sys.executable, str(SPARK_SCRIPT)]

    if variant == "spark":
        cmd = env_cmd + [
            "--master", sparkconf.standalone_master(),
            "--n", str(n), "--a", str(a_bin), "--b", str(b_bin),
            "--slices", str(p),
            "--warmup", str(args.warmups), "--reps", str(args.reps),
            "--app-name", f"a4_{exp}_n{n}_p{p}",
            "--eventlog-dir", str(EVENTLOG),
            *ref_args,
        ]
        return cmd, args.deadline_s

    if variant == "numpy":
        cmd = env_cmd + [
            "--baseline", "numpy",
            "--n", str(n), "--a", str(a_bin), "--b", str(b_bin),
            "--warmup", str(args.warmups), "--reps", str(args.baseline_reps),
            *ref_args,
        ]
        return cmd, args.deadline_s

    exe = ROOT / "src" / ("matmul3_seq.exe" if variant == "c_seq"
                          else "matmul3_simd.exe")
    threads = 1 if variant == "c_seq" else p
    cmd = [str(exe), "bench",
           "--n", str(n), "--threads", str(threads), "--affinity", "all",
           "--a", str(a_bin), "--b", str(b_bin),
           "--warmups", str(args.warmups), "--reps", str(args.baseline_reps),
           "--budget-ns", str(int(args.budget_s * 1e9)),
           "--min-batch-ns", "0"]
    return cmd, args.budget_s * (args.baseline_reps + 3) + 60.0


def measured_s(variant: str, rec: dict) -> float:
    if variant == "spark":
        return float(rec["phases_s"]["t_e2e"])
    if variant == "numpy":
        return float(rec["wall_s"])
    return float(rec["kernel_per_mult_ns"]) / 1e9


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase",
                    choices=["strong", "weak", "size", "baselines", "all"],
                    default="all")
    ap.add_argument("--reps", type=int, default=3,
                    help="measured Spark repetitions (protocol: K = 3)")
    ap.add_argument("--baseline-reps", type=int, default=5,
                    help="measured baseline repetitions (protocol: K = 5)")
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--budget-s", type=float, default=120.0)
    ap.add_argument("--deadline-s", type=float, default=900.0,
                    help="hard wall-clock limit for one child process")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="run at most N configs (smoke testing)")
    ap.add_argument("--no-power", action="store_true",
                    help="do not switch the power plan")
    ap.add_argument("--soak-s", type=float, default=0.0,
                    help="seconds to idle at the new power plan before the "
                         "first config (thermal heat-soak)")
    ap.add_argument("--stop-cluster", action="store_true",
                    help="stop the Spark cluster when the campaign ends")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    DETAIL.mkdir(parents=True, exist_ok=True)
    EVENTLOG.mkdir(parents=True, exist_ok=True)

    configs = sorted(build_configs(args.phase), key=sort_key)
    done = set() if args.force else load_done()
    todo = [c for c in configs if c not in done]
    n_done = len(configs) - len(todo)
    if args.limit:
        todo = todo[:args.limit]
    print(f"[campaign] phase={args.phase} total={len(configs)} "
          f"done={n_done} todo={len(todo)}")
    if args.dry:
        for c in todo:
            print("  ", c)
        return 0

    missing = sorted({n for _, _, n, _ in todo
                      if not (INPUTS / f"A_{n}.bin").exists()})
    if missing:
        print(f"[campaign] missing inputs for n={missing}; "
              f"run scripts/generate_inputs.py first")
        return 1

    raw_new = not RAW4_PATH.exists()
    cfg_new = not CFG4_PATH.exists()
    raw_f = RAW4_PATH.open("a", newline="", encoding="utf-8")
    cfg_f = CFG4_PATH.open("a", newline="", encoding="utf-8")
    raw_w = csv.DictWriter(raw_f, fieldnames=RAW_FIELDS)
    cfg_w = csv.DictWriter(cfg_f, fieldnames=CFG_FIELDS)
    if raw_new:
        raw_w.writeheader()
    if cfg_new:
        cfg_w.writeheader()

    if not args.no_power:
        set_high_performance()
    if args.soak_s > 0:
        print(f"[campaign] heat-soak {args.soak_s:.0f}s ...", flush=True)
        time.sleep(args.soak_s)

    failures = 0
    try:
        for idx, cfg in enumerate(todo, 1):
            exp, variant, n, p = cfg
            tag = f"{exp}_{variant}_n{n}_p{p}"
            print(f"[{idx}/{len(todo)}] {tag}", flush=True)

            if variant == "spark":
                ensure_cluster(p)
            cmd, deadline = build_cmd(cfg, args)
            res = run_child(cmd, DETAIL / f"{tag}.jsonl", deadline,
                            env=child_env())

            if variant == "spark":
                recs = [l for l in res.lines
                        if l.get("mode") == "distributed"
                        and not l.get("warmup")]
                event = next((l for l in res.lines
                              if l.get("mode") == "eventlog"), None)
            elif variant == "numpy":
                recs = [l for l in res.lines
                        if l.get("mode") == "numpy"
                        and not l.get("warmup")]
                event = None
            else:
                recs = [l for l in res.lines if l.get("mode") == "measure"]
                event = None
                if any(l.get("mode") == "timeout" for l in res.lines):
                    res.timed_out = True

            status, reason = "ok", ""
            if res.timed_out:
                status, reason = "skipped", "timeout"
            elif res.exit_code != 0:
                status, reason = "failed", f"exit_{res.exit_code}"
            elif not recs:
                status, reason = "failed", "no_measures"

            values = [measured_s(variant, r) for r in recs] if status == "ok" else []
            first = recs[0] if recs else {}
            phases = first.get("phases_s", {})
            val = first.get("validation") or (recs[-1].get("validation")
                                              if recs else None)

            for rep, rec in enumerate(recs):
                raw_w.writerow({
                    "experiment": exp, "variant": variant, "n": n, "p": p,
                    "rep": rec.get("rep", rep),
                    "measured_s": round(measured_s(variant, rec), 6),
                    "stage_a_s": phases.get("t_stage_a", ""),
                    "action_job_s": phases.get("t_action_job", ""),
                    "assemble_s": phases.get("t_assemble", ""),
                    "load_s": phases.get("t_load", ""),
                    "matmul_ms_max": rec.get("matmul_ms_max", ""),
                    "b_load_ms_max": rec.get("b_load_ms_max", ""),
                    "checksum": rec.get("checksum", ""),
                })

            ev_ok = bool(event) and "error" not in (event or {})
            cfg_w.writerow({
                "experiment": exp, "variant": variant, "n": n, "p": p,
                "status": status, "reason": reason, "reps": len(values),
                "exit_code": res.exit_code, "wall_s": round(res.wall_s, 3),
                "measured_median_s": round(statistics.median(values), 6) if values else "",
                "measured_min_s": round(min(values), 6) if values else "",
                "measured_max_s": round(max(values), 6) if values else "",
                "startup_s": first.get("t_startup_s", ""),
                "replicate_s": first.get("t_replicate_s", ""),
                "stage_a_s": phases.get("t_stage_a", ""),
                "action_job_s": phases.get("t_action_job", ""),
                "assemble_s": phases.get("t_assemble", ""),
                "matmul_ms_max": first.get("matmul_ms_max", ""),
                "b_load_ms_max": first.get("b_load_ms_max", ""),
                "checksum": first.get("checksum", ""),
                "valid": "" if not val else str(val.get("allclose")).lower(),
                "max_abs_diff": "" if not val else val.get("max_abs_diff", ""),
                "n_executors": event.get("n_executors", "") if ev_ok else "",
                "n_tasks": event.get("n_tasks", "") if ev_ok else "",
                "task_span_max_ms": event.get("task_span_max_ms", "") if ev_ok else "",
                "run_sum_ms": event.get("run_sum_ms", "") if ev_ok else "",
                "cpu_sum_ms": event.get("cpu_sum_ms", "") if ev_ok else "",
                "shuffle_write_bytes": event.get("shuffle_write_bytes", "") if ev_ok else "",
                "shuffle_read_bytes": event.get("shuffle_read_bytes", "") if ev_ok else "",
                "app_id": first.get("app_id", ""),
            })
            raw_f.flush()
            cfg_f.flush()
            if status != "ok":
                failures += 1
            med = f" median={statistics.median(values):.4f}s" if values else ""
            print(f"    -> {status} {reason}".rstrip() + med, flush=True)
    finally:
        raw_f.close()
        cfg_f.close()
        if not args.no_power:
            restore_power_plan()
        if args.stop_cluster and spark_cluster.STATE_FILE.exists():
            spark_cluster.stop()

    print(f"[campaign] done ({failures} config(s) not ok)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
