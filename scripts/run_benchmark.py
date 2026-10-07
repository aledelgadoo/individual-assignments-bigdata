#!/usr/bin/env python3
"""Benchmark orchestrator for Assignment 1 (Phases 4-5).

For every (language, n) configuration this script runs, sequentially:

1. probe  - feasibility gate: one warm-up kernel + one timed kernel (Java
   only prints both; Python/C print a single kernel, which is already the
   warmed measurement because they have no JIT). The gate passes when the
   timed kernel <= budget. A completed-but-failing probe is retried once;
   a probe killed by its deadline is recorded as timeout without retry.
2. bench  - if the gate passed: warm-ups (excluded), then `--reps` timed
   repetitions with a batch size calibrated from the last warm-up so each
   timing lasts >= 0.2 s.

Everything is written incrementally so an interrupted campaign keeps its
partial results:

  data/results/raw_timings.csv   one row per timed kernel event
  data/results/configs.csv       one row per (lang, n) configuration
  data/results/raw_detail/*.json raw stdout of every child process
"""
from __future__ import annotations

import argparse
import csv
import json
import queue
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
CFG_CSV = RESULTS / "configs.csv"

BUDGET_S = 60
BUDGET_NS = BUDGET_S * 1_000_000_000
REPS = 5
MIN_BATCH_NS = 200_000_000
MEM_LIMIT_BYTES = 4 * 1024**3
SIZES = [10, 50, 100, 200, 500, 1000, 2000]
LANGS = ["python", "java", "c"]
JAVA_BENCH_FLAGS = ["-XX:CompileCommand=dontinline,MatMul::matmulInto"]

RAW_FIELDS = ["lang", "n", "phase", "rep", "batch", "kernel_total_ns",
              "kernel_per_mult_ns", "checksum", "attempt"]
CFG_FIELDS = ["lang", "n", "status", "reason", "probe1_ns", "probe2_ns",
              "gate_ns", "warmups", "reps", "batch", "exit_code", "wall_s",
              "peak_rss_mb", "sys_cpu_mean_pct", "sys_cpu_max_pct"]


def warmups_for(lang: str, n: int) -> int:
    if lang != "java":
        return 1
    if n < 50:
        return 20000
    if n <= 100:
        return 20
    return 5


def cmd_for(lang: str, mode: str, n: int, extra: list[str]) -> list[str]:
    a = INPUTS / f"A_{n}.bin"
    b = INPUTS / f"B_{n}.bin"
    base = ["--a", str(a), "--b", str(b), "--n", str(n), *extra]
    if lang == "python":
        return [sys.executable, str(ROOT / "src" / "python" / "matmul.py"), mode, *base]
    if lang == "c":
        return [str(ROOT / "src" / "c" / "matmul.exe"), mode, *base]
    flags = JAVA_BENCH_FLAGS if mode == "bench" else []
    return ["java", *flags, "-cp", str(ROOT / "src" / "java"), "MatMul", mode, *base]


@dataclass
class RunResult:
    lines: list[dict] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    killed: str | None = None          # deadline | memory
    exit_code: int | None = None
    wall_s: float = 0.0
    peak_rss_mb: float = 0.0
    cpu_samples: list[float] = field(default_factory=list)


def run_child(cmd: list[str], deadline_s: float, detail_path: Path) -> RunResult:
    """Run cmd, stream stdout JSON lines, enforce wall-clock and memory limits."""
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
        psutil.cpu_percent(None)  # prime
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
        # queue drained without sentinel should not happen; reap defensively
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
        self.raw_new = not RAW_CSV.exists()
        self.cfg_new = not CFG_CSV.exists()
        self.raw_fh = RAW_CSV.open("a", newline="", encoding="utf-8")
        self.cfg_fh = CFG_CSV.open("a", newline="", encoding="utf-8")
        self.raw = csv.DictWriter(self.raw_fh, fieldnames=RAW_FIELDS)
        self.cfg = csv.DictWriter(self.cfg_fh, fieldnames=CFG_FIELDS)
        if self.raw_new:
            self.raw.writeheader()
        if self.cfg_new:
            self.cfg.writeheader()
        self.raw_fh.flush()
        self.cfg_fh.flush()

    def raw_row(self, **kw) -> None:
        self.raw.writerow({k: kw.get(k, "") for k in RAW_FIELDS})
        self.raw_fh.flush()

    def cfg_row(self, **kw) -> None:
        self.cfg.writerow({k: kw.get(k, "") for k in CFG_FIELDS})
        self.cfg_fh.flush()

    def close(self) -> None:
        self.raw_fh.close()
        self.cfg_fh.close()


def load_done() -> set[tuple[str, int]]:
    done: set[tuple[str, int]] = set()
    if CFG_CSV.exists():
        with CFG_CSV.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("status") == "ok":
                    done.add((row["lang"], int(row["n"])))
    return done


def run_probe(lang: str, n: int, w: Writer) -> tuple[str, list[int], float]:
    """Returns (status, probe_ns_list, gate_ns)."""
    detail = DETAIL_DIR / f"{lang}_n{n}_probe.jsonl"
    if detail.exists():
        detail.unlink()
    probes: list[int] = []
    for attempt in (1, 2):
        r = run_child(cmd_for(lang, "probe", n, []), BUDGET_S * 2 + 30, detail)
        gate = None
        for ln in r.lines:
            if ln.get("mode") != "probe":
                continue
            if ln.get("phase") == "timed" or (ln.get("mode") == "probe" and "phase" not in ln):
                gate = ln.get("kernel_total_ns")
        if r.killed == "memory":
            return "memory_limit", probes, float("nan")
        if gate is None:
            # killed before the gate line (deadline or OOM)
            return "timeout", probes, float("nan")
        probes.append(int(gate))
        w.raw_row(lang=lang, n=n, phase="probe", rep=len(probes) - 1, batch=1,
                  kernel_total_ns=gate, kernel_per_mult_ns=gate,
                  checksum="", attempt=attempt)
        if gate <= BUDGET_NS:
            return "passed", probes, float(gate)
        # retry only if the probe actually completed (was not killed)
        if r.killed is not None:
            break
    return "timeout", probes, probes[-1] if probes else float("nan")


def run_bench(lang: str, n: int, reps: int, w: Writer) -> tuple[str, dict]:
    warmups = warmups_for(lang, n)
    detail = DETAIL_DIR / f"{lang}_n{n}_bench.jsonl"
    if detail.exists():
        detail.unlink()
    extra = ["--warmups", str(warmups), "--reps", str(reps),
             "--budget-ns", str(BUDGET_NS), "--min-batch-ns", str(MIN_BATCH_NS)]
    deadline = min(BUDGET_S * (warmups + reps + 1) + 90, BUDGET_S * 12 + 180)
    r = run_child(cmd_for(lang, "bench", n, extra), deadline, detail)

    status = "ok"
    reason = ""
    batch = ""
    measures: list[float] = []
    for ln in r.lines:
        if ln.get("mode") == "timeout":
            status, reason = "timeout", ln.get("reason", "")
        elif ln.get("mode") == "bench":
            phase = ln.get("phase")
            w.raw_row(lang=lang, n=n, phase=phase, rep=ln.get("rep", ""),
                      batch=ln.get("batch", ""),
                      kernel_total_ns=ln.get("kernel_total_ns", ""),
                      kernel_per_mult_ns=ln.get("kernel_per_mult_ns", ""),
                      checksum=ln.get("checksum", ""), attempt=1)
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
        "warmups": warmups, "reps": reps, "batch": batch,
        "exit_code": r.exit_code, "wall_s": round(r.wall_s, 3),
        "peak_rss_mb": round(r.peak_rss_mb, 1),
        "sys_cpu_mean_pct": round(sum(cpu) / len(cpu), 1),
        "sys_cpu_max_pct": round(max(cpu), 1),
        "median_per_mult_ns": sorted(measures)[len(measures) // 2] if measures else "",
    }
    return status, meta


def main() -> int:
    global BUDGET_NS, BUDGET_S
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", default=",".join(map(str, SIZES)))
    ap.add_argument("--langs", default=",".join(LANGS))
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--force", action="store_true", help="rerun configs already ok")
    ap.add_argument("--budget-s", type=int, default=BUDGET_S)
    args = ap.parse_args()

    BUDGET_S = args.budget_s
    BUDGET_NS = BUDGET_S * 1_000_000_000

    sizes = [int(x) for x in args.sizes.split(",") if x]
    langs = [x.strip() for x in args.langs.split(",") if x]

    # ensure binaries exist
    if not (ROOT / "src" / "c" / "matmul.exe").exists() or not (
            ROOT / "src" / "java" / "MatMul.class").exists():
        print("binaries missing -> running scripts/build.py")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "build.py")],
                       check=True, cwd=ROOT)

    done = set() if args.force else load_done()
    w = Writer()
    t_campaign = time.monotonic()
    try:
        for n in sizes:
            for lang in langs:
                if (lang, n) in done:
                    print(f"[skip] {lang} n={n} (already ok)")
                    continue
                t0 = time.monotonic()
                print(f"[probe] {lang} n={n} ...", flush=True)
                pstatus, probes, gate = run_probe(lang, n, w)
                if pstatus != "passed":
                    w.cfg_row(lang=lang, n=n, status=pstatus, reason="budget gate failed",
                              probe1_ns=probes[0] if probes else "",
                              probe2_ns=probes[1] if len(probes) > 1 else "",
                              gate_ns="", warmups="", reps="", batch="",
                              exit_code="", wall_s=round(time.monotonic() - t0, 3),
                              peak_rss_mb="", sys_cpu_mean_pct="", sys_cpu_max_pct="")
                    print(f"   -> {pstatus} (probes={probes})", flush=True)
                    continue
                print(f"   gate passed: {gate / 1e6:.2f} ms -> bench", flush=True)
                bstatus, meta = run_bench(lang, n, args.reps, w)
                wall = round(time.monotonic() - t0, 3)
                w.cfg_row(lang=lang, n=n, status=bstatus, reason="",
                          probe1_ns=probes[0] if probes else "",
                          probe2_ns=probes[1] if len(probes) > 1 else "",
                          gate_ns=int(gate), exit_code=meta["exit_code"],
                          wall_s=wall, **{k: meta[k] for k in
                                          ("warmups", "reps", "batch", "peak_rss_mb",
                                           "sys_cpu_mean_pct", "sys_cpu_max_pct")})
                med = meta["median_per_mult_ns"]
                med_s = f"{med / 1e6:.3f} ms" if isinstance(med, (int, float)) and med else "n/a"
                print(f"   -> {bstatus}, median={med_s}, wall={wall}s", flush=True)
    finally:
        w.close()
    print(f"campaign section done in {time.monotonic() - t_campaign:.1f}s")
    print(f"results: {RAW_CSV} | {CFG_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
