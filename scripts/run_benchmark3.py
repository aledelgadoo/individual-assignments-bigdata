#!/usr/bin/env python3
"""Benchmark campaign orchestrator for Assignment 3 (OpenMP + SIMD).

Experiments (each config = variant x threads x n x affinity):
    seq    sequential baseline: matmul3_seq, threads=1, all cores, n in SIZES
    scale  S(p)/E(p) curve: matmul3_auto, p in {1,2,4,8,12,16}, all cores
    simd   controlled SIMD comparison: matmul3_scalar vs matmul3_simd,
           p in {1,8,12}, n in SIMD_SIZES, all cores
    pcore  P-cores-only run: matmul3_auto confined to CPU0-3 (mask 0xF),
           p in {1,2,3,4}, n in PCORE_SIZES

Protocol per config (identical to Assignment 2):
    1. `probe`: 1 unchecked warmup + 2 timed probes.
       Skip if max(probe) > budget; otherwise bench budget = min(budget, 2*max probe).
    2. `bench`: 1 warmup (budget-checked) + calibration (batch doubling until
       >= min-batch) + REPS measured batches of one multiplication each.
    JSON lines from every child run go to data/results/raw_detail/.

Outputs:
    data/results/raw_timings.csv   per measured batch
    data/results/configs.csv       per config (status, probes, timings)

Resume: already-done configs (status != running) are skipped; --force redoes them.
Set OMP_WAIT_POLICY=passive in the environment to bound spin-waiting.

Usage:
    python scripts/run_benchmark3.py --phase seq|scale|simd|pcore|all
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "data" / "inputs"
RESULTS = ROOT / "data" / "results"
DETAIL = RESULTS / "raw_detail"

SIZES = [50, 100, 200, 500, 1000, 2000]
SIMD_SIZES = [200, 500, 1000, 2000]
PCORE_SIZES = [500, 1000, 2000]
SCALE_THREADS = [1, 2, 4, 8, 12, 16]
SIMD_THREADS = [1, 8, 12]
PCORE_THREADS = [1, 2, 3, 4]

RAW_FIELDS = [
    "experiment", "variant", "affinity", "n", "threads", "rep", "batch",
    "kernel_total_ns", "kernel_per_mult_ns", "checksum",
]
CFG_FIELDS = [
    "experiment", "variant", "n", "threads", "affinity", "status", "reason",
    "probe1_ns", "probe2_ns", "gate_ns", "warmups", "reps", "batch",
    "exit_code", "wall_s", "peak_rss_mb", "sys_cpu_mean_pct", "sys_cpu_max_pct",
]


def build_configs(phase: str) -> list[tuple]:
    cfgs: list[tuple] = []
    if phase in ("seq", "all"):
        for n in SIZES:
            cfgs.append(("seq", "seq", n, 1, "all"))
    if phase in ("scale", "all"):
        for n in SIZES:
            for t in SCALE_THREADS:
                cfgs.append(("scale", "auto", n, t, "all"))
    if phase in ("simd", "all"):
        for n in SIMD_SIZES:
            for v in ("scalar", "simd"):
                for t in SIMD_THREADS:
                    cfgs.append(("simd", v, n, t, "all"))
    if phase in ("pcore", "all"):
        for n in PCORE_SIZES:
            for t in PCORE_THREADS:
                cfgs.append(("pcore", "auto", n, t, "pcore"))
    return cfgs


def exe_for(variant: str) -> Path:
    return ROOT / "src" / f"matmul3_{variant}.exe"


def load_done() -> set[tuple]:
    done = set()
    path = RESULTS / "configs.csv"
    if not path.exists():
        return done
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") not in (None, "", "running"):
                done.add((row["experiment"], row["variant"], int(row["n"]),
                          int(row["threads"]), row["affinity"]))
    return done


class ChildResult:
    def __init__(self) -> None:
        self.lines: list[dict] = []
        self.exit_code = -1
        self.wall_s = 0.0
        self.peak_rss_mb = 0.0
        self.sys_cpu_mean_pct = 0.0
        self.sys_cpu_max_pct = 0.0
        self.timed_out = False
        self.mem_killed = False


def run_child(cmd: list[str], detail_path: Path, deadline_s: float,
              mem_limit_mb: float = 4096.0) -> ChildResult:
    res = ChildResult()
    t0 = time.monotonic()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1, cwd=ROOT,
    )
    samples: list[float] = []
    peak_rss = 0

    def sample_loop() -> None:
        nonlocal peak_rss
        try:
            p = psutil.Process(proc.pid)
            p.cpu_percent(None)
            while proc.poll() is None:
                rss = p.memory_info().rss
                peak_rss = max(peak_rss, rss)
                samples.append(p.cpu_percent(None))
                if rss > mem_limit_mb * 1024 * 1024:
                    res.mem_killed = True
                    proc.kill()
                    return
                if time.monotonic() - t0 > deadline_s:
                    res.timed_out = True
                    proc.kill()
                    return
                time.sleep(0.1)
        except psutil.NoSuchProcess:
            pass

    sampler = threading.Thread(target=sample_loop, daemon=True)
    sampler.start()

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
    sampler.join(timeout=2.0)
    res.wall_s = time.monotonic() - t0
    res.exit_code = proc.returncode
    res.peak_rss_mb = peak_rss / (1024 * 1024)
    valid = [s for s in samples if s > 0.0]
    if valid:
        res.sys_cpu_mean_pct = sum(valid) / len(valid)
        res.sys_cpu_max_pct = max(valid)

    stderr_tail = ""
    if proc.stderr is not None:
        stderr_tail = proc.stderr.read()[-2000:]
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    with detail_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"cmd": cmd, "exit": res.exit_code,
                            "wall_s": res.wall_s,
                            "mem_killed": res.mem_killed,
                            "timed_out": res.timed_out}) + "\n")
        for ln in detail_lines:
            f.write(ln + "\n")
        if stderr_tail:
            f.write(json.dumps({"stderr_tail": stderr_tail}) + "\n")
    return res


def mode_values(res: ChildResult, mode: str, key: str) -> list[float]:
    return [float(l[key]) for l in res.lines
            if l.get("mode") == mode and key in l]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["seq", "scale", "simd", "pcore", "all"],
                    default="all")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--budget-s", type=float, default=60.0)
    ap.add_argument("--min-batch-ns", type=int, default=200_000_000)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    budget_ns = int(args.budget_s * 1e9)
    RESULTS.mkdir(parents=True, exist_ok=True)
    DETAIL.mkdir(parents=True, exist_ok=True)

    configs = build_configs(args.phase)
    done = set() if args.force else load_done()
    todo = [c for c in configs if c not in done]
    print(f"[campaign] phase={args.phase} total={len(configs)} "
          f"done={len(configs) - len(todo)} todo={len(todo)}")
    if args.dry:
        for c in todo:
            print("  ", c)
        return 0

    raw_path = RESULTS / "raw_timings.csv"
    cfg_path = RESULTS / "configs.csv"
    raw_new = not raw_path.exists()
    cfg_new = not cfg_path.exists()
    raw_f = raw_path.open("a", newline="", encoding="utf-8")
    cfg_f = cfg_path.open("a", newline="", encoding="utf-8")
    raw_w = csv.DictWriter(raw_f, fieldnames=RAW_FIELDS)
    cfg_w = csv.DictWriter(cfg_f, fieldnames=CFG_FIELDS)
    if raw_new:
        raw_w.writeheader()
    if cfg_new:
        cfg_w.writeheader()

    def flush() -> None:
        raw_f.flush()
        cfg_f.flush()

    env = os.environ.copy()
    env.setdefault("OMP_WAIT_POLICY", "PASSIVE")

    for idx, (exp, variant, n, threads, affinity) in enumerate(todo, 1):
        exe = exe_for(variant)
        if not exe.exists():
            print(f"[campaign] MISSING {exe}; run scripts/build.py first")
            return 1
        a_bin = INPUTS / f"A_{n}.bin"
        b_bin = INPUTS / f"B_{n}.bin"
        if not a_bin.exists() or not b_bin.exists():
            print(f"[campaign] missing inputs for n={n}; run generate_inputs.py")
            return 1

        tag = f"{exp}_{variant}_n{n}_t{threads}_{affinity}"
        base = ["--n", str(n), "--threads", str(threads),
                "--affinity", affinity,
                "--a", str(a_bin), "--b", str(b_bin)]

        print(f"[{idx}/{len(todo)}] {tag}", flush=True)

        # --- probe ---
        probe = run_child(
            [str(exe), "probe", *base, "--warmups", "1", "--probe-reps", "2"],
            DETAIL / f"{tag}_probe.jsonl",
            deadline_s=budget_ns / 1e9 * 3 + 60.0,
        )
        probe_vals = mode_values(probe, "probe", "kernel_ns")
        p1 = probe_vals[0] if len(probe_vals) > 0 else float("nan")
        p2 = probe_vals[1] if len(probe_vals) > 1 else float("nan")

        status = "ok"
        reason = ""
        gate = 0
        if probe.mem_killed:
            status, reason = "skipped", "probe_mem_kill"
        elif probe.timed_out or not probe_vals:
            status, reason = "skipped", "probe_failed"
        elif max(probe_vals) > budget_ns:
            status, reason = "skipped", "probe_over_budget"
        else:
            gate = int(min(budget_ns, 2 * max(probe_vals)))

        batch = 0
        exit_code = probe.exit_code if status != "ok" else -1
        wall = probe.wall_s
        rss = probe.peak_rss_mb
        cpu_mean = probe.sys_cpu_mean_pct
        cpu_max = probe.sys_cpu_max_pct

        if status == "ok":
            bench = run_child(
                [str(exe), "bench", *base,
                 "--warmups", "1", "--reps", str(args.reps),
                 "--budget-ns", str(gate),
                 "--min-batch-ns", str(args.min_batch_ns)],
                DETAIL / f"{tag}_bench.jsonl",
                deadline_s=gate / 1e9 * (args.reps + 3) + 90.0,
            )
            measures = [l for l in bench.lines if l.get("mode") == "measure"]
            wall = bench.wall_s
            rss = max(rss, bench.peak_rss_mb)
            cpu_mean = max(cpu_mean, bench.sys_cpu_mean_pct)
            cpu_max = max(cpu_max, bench.sys_cpu_max_pct)
            exit_code = bench.exit_code
            if bench.mem_killed:
                status, reason = "skipped", "bench_mem_kill"
            elif bench.timed_out:
                status, reason = "skipped", "bench_timeout"
            elif not measures:
                status, reason = "skipped", "bench_no_measures"
            else:
                for m in measures:
                    raw_w.writerow({
                        "experiment": exp, "variant": variant,
                        "affinity": affinity, "n": n, "threads": threads,
                        "rep": m.get("rep"), "batch": m.get("batch"),
                        "kernel_total_ns": m.get("kernel_total_ns"),
                        "kernel_per_mult_ns": m.get("kernel_per_mult_ns"),
                        "checksum": m.get("checksum"),
                    })
                batch = int(measures[0].get("batch") or 0)

        cfg_w.writerow({
            "experiment": exp, "variant": variant, "n": n,
            "threads": threads, "affinity": affinity,
            "status": status, "reason": reason,
            "probe1_ns": "" if p1 != p1 else int(p1),
            "probe2_ns": "" if p2 != p2 else int(p2),
            "gate_ns": gate, "warmups": 1 if status == "ok" else 0,
            "reps": args.reps if status == "ok" else 0, "batch": batch,
            "exit_code": exit_code, "wall_s": round(wall, 3),
            "peak_rss_mb": round(rss, 1),
            "sys_cpu_mean_pct": round(cpu_mean, 1),
            "sys_cpu_max_pct": round(cpu_max, 1),
        })
        flush()
        print(f"    -> {status} {reason}".rstrip(), flush=True)

    raw_f.close()
    cfg_f.close()
    print("[campaign] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
