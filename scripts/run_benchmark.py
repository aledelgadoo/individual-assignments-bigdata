#!/usr/bin/env python3
"""Benchmark orchestrator for Assignment 2 (dense optimisations + CSR).

Dense Part A: every (method, n) with method in {ijk, ikj, block} and the
Assignment 1 size sequence. Sparse Part B: every (method, sparsity) with
method in {ijk, ikj, block, csr}, n = 1000 fixed. Block-size tuning runs
into a separate CSV (--phase tune).

Per configuration:

1. probe  - feasibility gate: one warm-up kernel (excluded) + one timed
            kernel. The gate passes when the timed kernel <= budget. A
            completed-but-failing probe is retried once; a probe killed by
            its deadline is recorded as timeout without retry.
2. bench  - if the gate passed: warm-ups (excluded), then --reps timed
            repetitions with a batch size calibrated from the last warm-up
            so each timing lasts >= 0.2 s. CSR configurations additionally
            time --reps dense-to-CSR conversions (reported separately, never
            inside the multiplication timing).

Every child process is pinned to CPU0 (a P-core) unless --no-pin is given.

Incremental outputs:

  data/results/raw_timings.csv      one row per timed kernel event
  data/results/conversions.csv      one row per timed dense-to-CSR conversion
  data/results/configs.csv          one row per configuration
  data/results/blocksize_tuning.csv block-size sweep (--phase tune)
  data/results/raw_detail/*.json    raw stdout of every child process
"""
from __future__ import annotations

import argparse
import csv
import json
import queue
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
INPUTS = ROOT / "data" / "inputs"
RESULTS = ROOT / "data" / "results"
DETAIL_DIR = RESULTS / "raw_detail"
RAW_CSV = RESULTS / "raw_timings.csv"
CONV_CSV = RESULTS / "conversions.csv"
CFG_CSV = RESULTS / "configs.csv"
TUNE_CSV = RESULTS / "blocksize_tuning.csv"
EXE = ROOT / "src" / ("matmul2.exe" if sys.platform == "win32" else "matmul2")

BUDGET_S = 60
BUDGET_NS = BUDGET_S * 1_000_000_000
REPS = 5
MIN_BATCH_NS = 200_000_000
MEM_LIMIT_BYTES = 4 * 1024**3
WARMUPS = 1
SIZES = [10, 50, 100, 200, 500, 1000, 2000]
SPARSITIES = [1, 5, 10, 25, 50, 75, 90, 95, 99]
SPARSE_N = 1000
DENSE_METHODS = ["ijk", "ikj", "block"]
SPARSE_METHODS = ["ijk", "ikj", "block", "csr"]
TUNE_BLOCKSIZES = [16, 32, 64, 128]
TUNE_SIZES = [500, 1000, 2000]

RAW_FIELDS = ["method", "n", "sparsity", "phase", "rep", "batch",
              "kernel_total_ns", "kernel_per_mult_ns", "checksum", "attempt",
              "blocksize"]
CONV_FIELDS = ["n", "sparsity", "rep", "convert_ns", "nnz"]
CFG_FIELDS = ["phase", "method", "n", "sparsity", "status", "reason",
              "probe1_ns", "probe2_ns", "gate_ns", "warmups", "reps", "batch",
              "blocksize", "exit_code", "wall_s", "peak_rss_mb",
              "sys_cpu_mean_pct", "sys_cpu_max_pct"]
TUNE_FIELDS = ["blocksize", "n", "rep", "kernel_per_mult_ns"]


def a_path(n: int, sparsity: int) -> Path:
    if sparsity == 0:
        return INPUTS / f"A_{n}.bin"
    return INPUTS / f"A_{n}_sp{sparsity:02d}.bin"


def cmd_for(method: str, mode: str, n: int, sparsity: int, blocksize: int,
            pin: bool, extra: list[str]) -> list[str]:
    cmd = [str(EXE), mode, "--method", method,
           "--a", str(a_path(n, sparsity)), "--b", str(INPUTS / f"B_{n}.bin"),
           "--n", str(n), "--sparsity", str(sparsity),
           "--blocksize", str(blocksize)]
    if pin:
        cmd.append("--pin")
    return cmd + extra


@dataclass
class RunResult:
    lines: list[dict] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    killed: str | None = None
    exit_code: int | None = None
    wall_s: float = 0.0
    peak_rss_mb: float = 0.0
    cpu_samples: list[float] = field(default_factory=list)


def run_child(cmd: list[str], deadline_s: float, detail_path: Path) -> RunResult:
    res = RunResult()
    t0 = time.monotonic()
    proc = subprocess.Popen(
        cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    assert proc.stdout is not None
    q: queue.Queue[str | None] = queue.Queue()

    def reader() -> None:
        try:
            for line in proc.stdout:
                q.put(line)
        finally:
            q.put(None)

    threading.Thread(target=reader, daemon=True).start()

    ps = psutil.Process(proc.pid)
    stop = threading.Event()

    def sampler() -> None:
        psutil.cpu_percent(None)
        while not stop.is_set():
            try:
                rss = ps.memory_info().rss
                res.peak_rss_mb = max(res.peak_rss_mb, rss / 1024**2)
                if rss > MEM_LIMIT_BYTES:
                    res.killed = "memory"
                    proc.kill()
                    return
                res.cpu_samples.append(psutil.cpu_percent(None))
            except psutil.NoSuchProcess:
                return
            stop.wait(0.05)

    threading.Thread(target=sampler, daemon=True).start()

    deadline = t0 + deadline_s
    eof = False
    while True:
        if res.killed == "memory":
            break
        if time.monotonic() > deadline:
            res.killed = "deadline"
            proc.kill()
            break
        try:
            line = q.get(timeout=0.25)
        except queue.Empty:
            continue
        if line is None:
            eof = True
            break
        line = line.strip()
        if not line:
            continue
        try:
            res.lines.append(json.loads(line))
        except json.JSONDecodeError:
            res.log.append(line)

    if not eof and res.killed is None:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    try:
        res.exit_code = proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        res.exit_code = -1
    stop.set()
    res.wall_s = time.monotonic() - t0

    detail_path.parent.mkdir(parents=True, exist_ok=True)
    with detail_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"cmd": cmd, "deadline_s": deadline_s,
                             "killed": res.killed, "exit_code": res.exit_code,
                             "wall_s": round(res.wall_s, 3)}) + "\n")
        for ln in res.lines:
            fh.write(json.dumps(ln) + "\n")
        for ln in res.log:
            fh.write(json.dumps({"unparsed": ln}) + "\n")
    return res


class Writer:
    def __init__(self) -> None:
        RESULTS.mkdir(parents=True, exist_ok=True)
        self.raw_fh = RAW_CSV.open("a", newline="", encoding="utf-8")
        self.conv_fh = CONV_CSV.open("a", newline="", encoding="utf-8")
        self.cfg_fh = CFG_CSV.open("a", newline="", encoding="utf-8")
        self.raw = csv.DictWriter(self.raw_fh, fieldnames=RAW_FIELDS)
        self.conv = csv.DictWriter(self.conv_fh, fieldnames=CONV_FIELDS)
        self.cfg = csv.DictWriter(self.cfg_fh, fieldnames=CFG_FIELDS)
        if self.raw_fh.tell() == 0:
            self.raw.writeheader()
        if self.conv_fh.tell() == 0:
            self.conv.writeheader()
        if self.cfg_fh.tell() == 0:
            self.cfg.writeheader()
        self.raw_fh.flush()
        self.conv_fh.flush()
        self.cfg_fh.flush()

    def raw_row(self, **kw) -> None:
        self.raw.writerow({k: kw.get(k, "") for k in RAW_FIELDS})
        self.raw_fh.flush()

    def conv_row(self, **kw) -> None:
        self.conv.writerow({k: kw.get(k, "") for k in CONV_FIELDS})
        self.conv_fh.flush()

    def cfg_row(self, **kw) -> None:
        self.cfg.writerow({k: kw.get(k, "") for k in CFG_FIELDS})
        self.cfg_fh.flush()

    def close(self) -> None:
        self.raw_fh.close()
        self.conv_fh.close()
        self.cfg_fh.close()


def best_blocksize(n: int, default: int) -> int:
    """Median-best block size for n from the tuning sweep; else the fallback."""
    if not TUNE_CSV.exists():
        return default
    per: dict[int, list[float]] = {}
    with TUNE_CSV.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["n"] and int(row["n"]) == n and row["kernel_per_mult_ns"]:
                per.setdefault(int(row["blocksize"]), []).append(
                    float(row["kernel_per_mult_ns"]))
    if not per:
        return default
    med = {bs: statistics.median(v) for bs, v in per.items()}
    return min(med, key=med.get)


def fallback_blocksize(default: int) -> int:
    """Best block size at the smallest tuned n, used for untuned sizes."""
    if not TUNE_CSV.exists():
        return default
    sizes = {int(row["n"]) for row in csv.DictReader(TUNE_CSV.open(
        newline="", encoding="utf-8")) if row["n"]}
    if not sizes:
        return default
    return best_blocksize(min(sizes), default)


def load_done() -> set[tuple[str, str, int, int]]:
    done: set[tuple[str, str, int, int]] = set()
    if CFG_CSV.exists():
        with CFG_CSV.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("status") == "ok":
                    done.add((row["phase"], row["method"], int(row["n"]),
                              int(row["sparsity"])))
    return done


def run_probe(method: str, n: int, sparsity: int, blocksize: int, pin: bool,
              budget_ns: int, w: Writer) -> tuple[str, list[int], float]:
    detail = DETAIL_DIR / f"{method}_n{n}_sp{sparsity}_probe.jsonl"
    if detail.exists():
        detail.unlink()
    probes: list[int] = []
    for attempt in (1, 2):
        r = run_child(cmd_for(method, "probe", n, sparsity, blocksize, pin, []),
                      BUDGET_S * 2 + 30, detail)
        gate = None
        for ln in r.lines:
            if ln.get("mode") == "probe":
                gate = ln.get("kernel_total_ns")
        if r.killed == "memory":
            return "memory_limit", probes, float("nan")
        if gate is None:
            return "timeout", probes, float("nan")
        probes.append(int(gate))
        w.raw_row(method=method, n=n, sparsity=sparsity, phase="probe",
                  rep=len(probes) - 1, batch=1, kernel_total_ns=gate,
                  kernel_per_mult_ns=gate, checksum="", attempt=attempt,
                  blocksize=blocksize)
        if gate <= budget_ns:
            return "passed", probes, float(gate)
        if r.killed is not None:
            break
    return "timeout", probes, probes[-1] if probes else float("nan")


def run_bench(method: str, n: int, sparsity: int, blocksize: int, pin: bool,
              reps: int, w: Writer) -> tuple[str, dict]:
    detail = DETAIL_DIR / f"{method}_n{n}_sp{sparsity}_bench.jsonl"
    if detail.exists():
        detail.unlink()
    extra = ["--warmups", str(WARMUPS), "--reps", str(reps),
             "--budget-ns", str(BUDGET_NS), "--min-batch-ns", str(MIN_BATCH_NS)]
    deadline = min(BUDGET_S * (WARMUPS + reps + 1) + 90, BUDGET_S * 12 + 180)
    r = run_child(cmd_for(method, "bench", n, sparsity, blocksize, pin, extra),
                  deadline, detail)

    status = "ok"
    reason = ""
    batch = ""
    measures: list[float] = []
    for ln in r.lines:
        if ln.get("mode") == "timeout":
            status, reason = "timeout", ln.get("reason", "")
        elif ln.get("mode") == "bench":
            phase = ln.get("phase")
            if phase == "convert":
                w.conv_row(n=n, sparsity=sparsity, rep=ln.get("rep", ""),
                           convert_ns=ln.get("convert_ns", ""),
                           nnz=ln.get("nnz", ""))
            else:
                w.raw_row(method=method, n=n, sparsity=sparsity, phase=phase,
                          rep=ln.get("rep", ""), batch=ln.get("batch", ""),
                          kernel_total_ns=ln.get("kernel_total_ns", ""),
                          kernel_per_mult_ns=ln.get("kernel_per_mult_ns", ""),
                          checksum=ln.get("checksum", ""), attempt=1,
                          blocksize=blocksize)
            if phase == "measure":
                measures.append(float(ln.get("kernel_per_mult_ns", 0)))
                batch = ln.get("batch", "")
    if r.killed == "deadline" and status == "ok":
        status, reason = "timeout", "bench deadline exceeded"
    elif r.killed == "memory" and status == "ok":
        status, reason = "memory_limit", "child exceeded 4 GiB"
    elif status == "ok" and r.exit_code not in (0,):
        status, reason = "error", f"exit_code={r.exit_code}"

    cpu = r.cpu_samples or [0.0]
    meta = {
        "warmups": WARMUPS, "reps": reps, "batch": batch,
        "exit_code": r.exit_code, "wall_s": round(r.wall_s, 3),
        "peak_rss_mb": round(r.peak_rss_mb, 1),
        "sys_cpu_mean_pct": round(sum(cpu) / len(cpu), 1),
        "sys_cpu_max_pct": round(max(cpu), 1),
        "median_per_mult_ns": sorted(measures)[len(measures) // 2] if measures else "",
    }
    return status, meta


def run_tune(blocksize: int, pin: bool, reps: int) -> None:
    new = not TUNE_CSV.exists() or TUNE_CSV.stat().st_size == 0
    fh = TUNE_CSV.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=TUNE_FIELDS)
    if new:
        writer.writeheader()
    for n in TUNE_SIZES:
        extra = ["--warmups", str(WARMUPS), "--reps", str(reps),
                 "--budget-ns", str(BUDGET_NS), "--min-batch-ns", str(MIN_BATCH_NS)]
        detail = DETAIL_DIR / f"tune_block{blocksize}_n{n}.jsonl"
        if detail.exists():
            detail.unlink()
        print(f"[tune] blocksize={blocksize} n={n} ...", flush=True)
        r = run_child(cmd_for("block", "bench", n, 0, blocksize, pin, extra),
                      300.0, detail)
        for ln in r.lines:
            if ln.get("mode") == "bench" and ln.get("phase") == "measure":
                writer.writerow({"blocksize": blocksize, "n": n,
                                 "rep": ln.get("rep", ""),
                                 "kernel_per_mult_ns": ln.get("kernel_per_mult_ns", "")})
                fh.flush()
        if r.killed:
            print(f"   -> killed ({r.killed})", flush=True)
    fh.close()


def main() -> int:
    global BUDGET_NS, BUDGET_S
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["a", "b", "all", "tune"], default="all")
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--force", action="store_true",
                    help="rerun configs already ok")
    ap.add_argument("--budget-s", type=int, default=BUDGET_S)
    ap.add_argument("--blocksize", type=int, default=64)
    ap.add_argument("--no-pin", action="store_true",
                    help="do not pin child processes to CPU0")
    args = ap.parse_args()

    BUDGET_S = args.budget_s
    BUDGET_NS = BUDGET_S * 1_000_000_000
    pin = not args.no_pin

    if not EXE.exists():
        print("binary missing -> running scripts/build.py")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "build.py")],
                       check=True, cwd=ROOT)

    if args.phase == "tune":
        t0 = time.monotonic()
        for bs in TUNE_BLOCKSIZES:
            run_tune(bs, pin, max(args.reps, 3))
        print(f"tuning done in {time.monotonic() - t0:.1f}s -> {TUNE_CSV}")
        return 0

    configs: list[tuple[str, str, int, int]] = []
    if args.phase in ("a", "all"):
        configs += [("a", m, n, 0) for m in DENSE_METHODS for n in SIZES]
    if args.phase in ("b", "all"):
        configs += [("b", m, SPARSE_N, p) for m in SPARSE_METHODS
                    for p in SPARSITIES]

    for _, _, n, sp in configs:
        need = [a_path(n, sp), INPUTS / f"B_{n}.bin"]
        missing = [str(p) for p in need if not p.exists()]
        if missing:
            raise SystemExit(f"missing inputs: {missing} -> "
                             "run scripts/generate_inputs.py first")

    done = set() if args.force else load_done()
    fb_bs = fallback_blocksize(args.blocksize)
    w = Writer()
    t_campaign = time.monotonic()
    try:
        for phase, method, n, sparsity in configs:
            key = (phase, method, n, sparsity)
            if key in done:
                print(f"[skip] {phase} {method} n={n} sp={sparsity}% (already ok)")
                continue
            t0 = time.monotonic()
            bs = best_blocksize(n, fb_bs) if method == "block" else 0
            label = f"{method} n={n} sp={sparsity}%"
            if method == "block":
                label += f" bs={bs}"
            print(f"[probe] {label} ...", flush=True)
            pstatus, probes, gate = run_probe(method, n, sparsity,
                                              bs, pin, BUDGET_NS, w)
            if pstatus != "passed":
                w.cfg_row(phase=phase, method=method, n=n, sparsity=sparsity,
                          status=pstatus, reason="budget gate failed",
                          probe1_ns=probes[0] if probes else "",
                          probe2_ns=probes[1] if len(probes) > 1 else "",
                          gate_ns="", warmups="", reps="", batch="",
                          blocksize=bs, exit_code="",
                          wall_s=round(time.monotonic() - t0, 3),
                          peak_rss_mb="", sys_cpu_mean_pct="",
                          sys_cpu_max_pct="")
                print(f"   -> {pstatus} (probes={probes})", flush=True)
                continue
            print(f"   gate passed: {gate / 1e6:.3f} ms -> bench", flush=True)
            bstatus, meta = run_bench(method, n, sparsity, bs,
                                      pin, args.reps, w)
            wall = round(time.monotonic() - t0, 3)
            w.cfg_row(phase=phase, method=method, n=n, sparsity=sparsity,
                      status=bstatus, reason="",
                      probe1_ns=probes[0] if probes else "",
                      probe2_ns=probes[1] if len(probes) > 1 else "",
                      gate_ns=int(gate), exit_code=meta["exit_code"],
                      wall_s=wall, blocksize=bs,
                      **{k: meta[k] for k in
                         ("warmups", "reps", "batch", "peak_rss_mb",
                          "sys_cpu_mean_pct", "sys_cpu_max_pct")})
            med = meta["median_per_mult_ns"]
            med_s = (f"{med / 1e6:.4f} ms"
                     if isinstance(med, (int, float)) and med else "n/a")
            print(f"   -> {bstatus}, median={med_s}, wall={wall}s", flush=True)
    finally:
        w.close()
    print(f"campaign phase done in {time.monotonic() - t_campaign:.1f}s")
    print(f"results: {RAW_CSV} | {CFG_CSV} | {CONV_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
