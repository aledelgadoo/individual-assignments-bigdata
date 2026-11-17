"""Distributed dense matrix multiplication for Assignment 4 (PySpark).

Design
------
* **Partitioning.**  ``A`` is split into ``num_slices`` contiguous *row
  blocks* (``A[start:end]``).  Every block keeps the full row width ``n``, so
  a block ``r x n`` multiplied by the full ``B`` (``n x n``) yields exactly the
  corresponding ``r x n`` rows of ``C``.
* **Placement.**  ``row_blocks`` produces exactly ``num_slices`` items, and
  ``sc.parallelize(..., numSlices=num_slices)`` uses a batch size of 1 in that
  case, so partition *i* holds block *i*.  On the reported topology
  (``num_slices`` = number of workers, 1 core each) each worker process runs
  one block.
* **Replication of B.**  ``B`` is replicated to every worker once per
  application with :meth:`SparkContext.addFile`, which is Spark's file
  distribution facility and has the same data volume as a broadcast
  (``8n^2`` bytes per worker).  ``sc.broadcast`` *cannot* be used here: on
  Windows ``PythonBroadcast.readObject`` (PythonRDD.scala) moves its temporary
  file into the block manager **while the ``FileOutputStream`` is still
  open**, and ``Files.move`` on Windows fails with ERROR_SHARING_VIOLATION.
  This was verified against the bundled JDK 21 and is reproducible on every
  run.  The limitation is documented in the report rather than hidden.
* **Computation.**  Each block computes ``block @ B`` with numpy/BLAS.
  ``OPENBLAS_NUM_THREADS=1`` is forced on the workers (see ``sparkconf``) so
  one worker really means one core.
* **Collection.**  Partial ``C`` blocks are ``collect()``-ed to the driver and
  assembled in row order into the full ``n x n`` result.

Timing boundary
---------------
Two families of numbers, never mixed:

1. **One-time setup per application**, measured once:
   ``t_startup`` (SparkSession + executor launch) and ``t_replicate``
   (``addFile`` + a forcing job that makes every executor materialise B).
2. **Per repetition**, additive because the phases run strictly one after the
   other::

       t_e2e = t_load + t_stage_a + t_action_job + t_assemble

   ``t_action_job`` is a single natural Spark action (dispatch, transfer of A,
   compute, transfer of C back).  It is decomposed afterwards with the
   framework's event log plus timers measured *inside* the worker, which
   separates the pure matrix multiplication from the serialization Spark adds
   around it.

The single-process numpy baseline (:func:`numpy_sequential`) runs the *same*
BLAS kernel in one process, so comparing against it isolates the cost of
distribution.  Comparing against the Assignment 3 C baselines does not: those
use a different, much slower kernel.  That difference is documented, not
hidden.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from time import perf_counter

SPARK_DIR = Path(__file__).resolve().parent
if str(SPARK_DIR) not in sys.path:
    sys.path.insert(0, str(SPARK_DIR))

import sparkconf  # noqa: E402  (must follow the sys.path insert)

# OpenBLAS reads its thread-count environment when the library loads, so the
# single-thread pin has to happen before numpy is imported.  Without it the
# single-process numpy baseline would silently use every core and the
# comparison against the distributed run would be meaningless.
sparkconf.apply_blas_thread_limits()

import numpy as np  # noqa: E402  (must follow the BLAS thread pin)


# --------------------------------------------------------------------------- #
# data helpers
# --------------------------------------------------------------------------- #
def load_matrix(path: str | Path, n: int) -> np.ndarray:
    """Read a raw little-endian float64 row-major n x n matrix."""
    arr = np.fromfile(path, dtype="<f8")
    if arr.size != n * n:
        raise ValueError(f"{path}: expected {n * n} doubles, got {arr.size}")
    return arr.reshape(n, n)


def row_blocks(A: np.ndarray, num_slices: int) -> list[tuple[int, int, np.ndarray]]:
    """Split ``A`` into ``num_slices`` contiguous row blocks."""
    n = A.shape[0]
    bounds = np.linspace(0, n, num_slices + 1, dtype=int)
    blocks = []
    for i in range(num_slices):
        start, end = int(bounds[i]), int(bounds[i + 1])
        if end > start:
            blocks.append((i, start, A[start:end]))
    return blocks


def numpy_sequential(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Single-process, single-thread BLAS baseline (isolates the BLAS gain)."""
    return A @ B


def checksum(C: np.ndarray) -> float:
    return float(np.asarray(C, dtype=np.float64).sum())


def validate(C: np.ndarray, ref_path: str | Path,
             rtol: float = 1e-9, atol: float = 1e-12) -> dict:
    """Compare against the sequential C reference produced by ``main.c dump``."""
    ref = np.fromfile(ref_path, dtype="<f8").reshape(C.shape)
    diff = np.abs(C - ref)
    return {
        "ref": str(ref_path),
        "max_abs_diff": float(diff.max()),
        "allclose": bool(np.allclose(C, ref, rtol=rtol, atol=atol)),
        "rtol": rtol,
        "atol": atol,
    }


# --------------------------------------------------------------------------- #
# the worker-side kernel (cloudpickled to every executor)
# --------------------------------------------------------------------------- #
def _make_mul(b_name: str, n: int):
    """Return a picklable mapper for one row block.

    Returns ``(idx, start, C_block, b_load_ms, matmul_ms)`` where the two
    timings are measured *inside the worker*: ``b_load_ms`` is the cost of
    reading the local copy of B, ``matmul_ms`` is the pure BLAS call.
    """
    from pyspark import SparkFiles

    def _mul(item: tuple[int, int, np.ndarray]):
        idx, start, block = item
        t0 = perf_counter()
        B = np.fromfile(SparkFiles.get(b_name), dtype="<f8").reshape(n, n)
        b_load_ms = (perf_counter() - t0) * 1e3
        t0 = perf_counter()
        out = block @ B
        matmul_ms = (perf_counter() - t0) * 1e3
        return (idx, start, out, b_load_ms, matmul_ms)

    return _mul


# --------------------------------------------------------------------------- #
# session
# --------------------------------------------------------------------------- #
def build_session(master_url: str,
                  app_name: str = "assignment4",
                  executor_memory: str = "1g",
                  event_log_dir: str | Path | None = None):
    """Create a SparkSession bound to the standalone cluster."""
    from pyspark.sql import SparkSession

    builder = SparkSession.builder.appName(app_name).master(master_url)
    for key, value in sparkconf.session_configs(
            executor_memory=executor_memory,
            event_log_dir=str(event_log_dir) if event_log_dir else None).items():
        builder = builder.config(key, value)
    return builder.getOrCreate()


def executor_count(sc) -> int:
    """Number of executors currently registered for this application."""
    try:
        return max(1, len(sc._jsc.sc().getExecutorMemoryStatus()))
    except Exception:  # noqa: BLE001
        return 1


# --------------------------------------------------------------------------- #
# replication of B
# --------------------------------------------------------------------------- #
def replicate(sc, b_path: str | Path, force_partitions: int | None = None) -> float:
    """Distribute ``b_path`` to every executor; return elapsed seconds.

    ``SparkContext.addFile`` only *registers* the file -- the actual download
    happens inside ``Executor.updateDependencies()`` when a task arrives.  A
    tiny forcing job with one partition per executor therefore materialises B
    everywhere and puts a clean wall-clock number on the transfer, instead of
    letting it hide inside the first matrix multiplication.
    """
    from pyspark import SparkFiles

    b_path = Path(b_path).resolve()
    name = b_path.name
    n_force = max(1, force_partitions or executor_count(sc))

    t0 = perf_counter()
    sc.addFile(b_path.as_uri())
    sc.parallelize(range(n_force), n_force) \
      .map(lambda _: os.path.getsize(SparkFiles.get(name))).collect()
    return perf_counter() - t0


# --------------------------------------------------------------------------- #
# the distributed multiplication
# --------------------------------------------------------------------------- #
def multiply(sc, A: np.ndarray, b_name: str, n: int, num_slices: int):
    """Run the distributed multiplication; return (C, phases, block_stats).

    The phases run strictly one after another, so they can be added:

    ``t_stage_a``  split A into row blocks and stage them for distribution.
                   PySpark serializes the blocks to a driver-side temp file
                   and the driver JVM reads them back eagerly, so this is a
                   real driver cost and not free bookkeeping.
    ``t_action_job`` one natural Spark action: dispatch, ship the blocks to
                   the executors, compute, ship C back.
    ``t_assemble`` rebuild the full n x n matrix on the driver.
    """
    phases: dict[str, float] = {}

    t0 = perf_counter()
    blocks = row_blocks(A, num_slices)
    rdd = sc.parallelize(blocks, numSlices=num_slices)
    phases["t_stage_a"] = perf_counter() - t0

    t0 = perf_counter()
    parts = rdd.map(_make_mul(b_name, n)).collect()
    phases["t_action_job"] = perf_counter() - t0

    t0 = perf_counter()
    C = np.empty((n, n), dtype=np.float64)
    b_load_ms: list[float] = []
    matmul_ms: list[float] = []
    for idx, start, block, b_ms, m_ms in sorted(parts, key=lambda p: p[0]):
        rows = block.shape[0]
        C[start:start + rows, :] = block
        b_load_ms.append(b_ms)
        matmul_ms.append(m_ms)
    phases["t_assemble"] = perf_counter() - t0

    phases["t_e2e"] = (phases["t_stage_a"] + phases["t_action_job"] +
                       phases["t_assemble"])
    stats = {
        "n_blocks": len(parts),
        "b_load_ms_sum": float(sum(b_load_ms)),
        "b_load_ms_max": float(max(b_load_ms)) if b_load_ms else 0.0,
        "matmul_ms_sum": float(sum(matmul_ms)),
        "matmul_ms_max": float(max(matmul_ms)) if matmul_ms else 0.0,
    }
    return C, phases, stats


# --------------------------------------------------------------------------- #
# event-log metrics
# --------------------------------------------------------------------------- #
def parse_eventlog(event_dir: str | Path, app_id: str) -> dict:
    """Extract per-task metrics for ``app_id`` from a Spark 4.x event log.

    Spark 4 writes ``eventlog_v2_<app>/events_1_<app>.zstd`` (zstd framed,
    one JSON object per line).  Returns sums/maxima over all finished tasks
    plus the executor inventory.
    """
    import zstandard

    base = Path(event_dir)
    candidates = sorted(base.glob(f"eventlog_v2_{app_id}/events_*_{app_id}.zstd"))
    if not candidates:
        candidates = sorted(base.glob(f"eventlog_v2_{app_id}/*"))
    if not candidates:
        raise FileNotFoundError(f"no event log for {app_id} under {base}")

    path = candidates[0]
    raw = path.read_bytes()
    if path.suffix == ".zstd" or raw[:4] == b"\x28\xb5\x2f\xfd":
        # The log is a *concatenated* zstd stream: ZstdDecompressor.decompress()
        # only returns the first frame (59 bytes here), so read the whole
        # stream with stream_reader().
        import io

        with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(raw)) as reader:
            payload = reader.read()
        lines = payload.decode("utf-8", errors="replace").splitlines()
    else:
        lines = raw.decode("utf-8", errors="replace").splitlines()

    tasks, executors, block_managers, stages = [], [], [], []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = ev.get("Event", "")
        if kind == "SparkListenerTaskEnd" and \
                (ev.get("Task End Reason") or {}).get("Reason") == "Success":
            info = ev.get("Task Info", {})
            met = ev.get("Task Metrics", {}) or {}
            acc = {a.get("Name"): a.get("Value", 0)
                   for a in (info.get("Accumulables") or [])}
            launch = info.get("Launch Time", 0)
            finish = info.get("Finish Time", 0)
            srm = met.get("Shuffle Read Metrics") or {}
            swm = met.get("Shuffle Write Metrics") or {}
            tasks.append({
                "stage": ev.get("Stage ID"),
                "partition": info.get("Partition ID"),
                "executor": info.get("Executor ID"),
                "span_ms": (finish - launch) if launch and finish else 0,
                "deserialize_ms": met.get("Executor Deserialize Time", 0),
                "run_ms": met.get("Executor Run Time", 0),
                "cpu_ms": acc.get("internal.metrics.executorCpuTime", 0) / 1e6,
                "result_ser_ms": met.get("Result Serialization Time", 0),
                "jvm_gc_ms": met.get("JVM GC Time", 0),
                "shuffle_write_bytes": swm.get("Shuffle Bytes Written", 0),
                "shuffle_read_bytes": srm.get("Local Bytes Read", 0) +
                                      srm.get("Remote Bytes Read", 0),
            })
        elif kind == "SparkListenerExecutorAdded":
            info = ev.get("Executor Info", {}) or {}
            executors.append({"executor": ev.get("Executor ID"),
                              "host": info.get("Host"),
                              "cores": info.get("Total Cores")})
        elif kind == "SparkListenerBlockManagerAdded":
            bm = ev.get("Block Manager ID", {}) or {}
            block_managers.append({
                "executor": bm.get("Executor ID"),
                "max_memory_bytes": ev.get("Maximum Memory"),
            })
        elif kind == "SparkListenerStageCompleted":
            s = ev.get("Stage Info", {}) or {}
            stages.append({"stage": s.get("Stage ID"),
                           "name": s.get("Name"),
                           "tasks": s.get("Number of Tasks")})

    if not tasks:
        return {"app_id": app_id, "n_tasks": 0, "note": "no successful tasks found"}

    def _sum(key: str) -> float:
        return float(sum(t[key] for t in tasks))

    def _max(key: str) -> float:
        return float(max(t[key] for t in tasks))

    return {
        "app_id": app_id,
        "n_tasks": len(tasks),
        "stages": stages,
        "executors": executors,
        "n_executors": len(executors),
        "block_managers": block_managers,
        "task_span_max_ms": _max("span_ms"),
        "task_span_sum_ms": _sum("span_ms"),
        "deserialize_sum_ms": _sum("deserialize_ms"),
        "deserialize_max_ms": _max("deserialize_ms"),
        "run_sum_ms": _sum("run_ms"),
        "run_max_ms": _max("run_ms"),
        "cpu_sum_ms": _sum("cpu_ms"),
        "result_ser_sum_ms": _sum("result_ser_ms"),
        "jvm_gc_sum_ms": _sum("jvm_gc_ms"),
        "shuffle_write_bytes": int(_sum("shuffle_write_bytes")),
        "shuffle_read_bytes": int(_sum("shuffle_read_bytes")),
    }


# --------------------------------------------------------------------------- #
# CLI (standalone validation / smoke runs)
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master", default=sparkconf.standalone_master())
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--a", required=True, help="path to A_n.bin (driver reads it)")
    ap.add_argument("--b", required=True, help="path to B_n.bin (distributed via addFile)")
    ap.add_argument("--slices", type=int, default=0,
                    help="row blocks = partitions (use the worker count)")
    ap.add_argument("--executor-memory", default="1g")
    ap.add_argument("--eventlog-dir", default=None)
    ap.add_argument("--out", default=None, help="optional C dump path")
    ap.add_argument("--ref", default=None, help="optional C_n_seq.bin to validate against")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--app-name", default="assignment4")
    ap.add_argument("--baseline", choices=["spark", "numpy"], default="spark",
                    help="'numpy' runs the single-process BLAS baseline "
                         "instead of the distributed multiplication")
    args = ap.parse_args()
    args.warmups = max(0, args.warmup)
    if args.baseline == "spark" and args.slices <= 0:
        ap.error("--slices must be positive for the spark baseline")

    t0 = perf_counter()
    A = load_matrix(args.a, args.n)
    t_load = perf_counter() - t0
    b_name = Path(args.b).name

    if args.baseline == "numpy":
        # Same BLAS kernel, one process, one thread: the reference that makes
        # the distribution overhead measurable instead of confounded with the
        # kernel difference against the Assignment 3 C baselines.
        B = load_matrix(args.b, args.n)
        for rep in range(args.warmups + args.reps):
            t0 = perf_counter()
            C = numpy_sequential(A, B)
            wall = perf_counter() - t0
            rec = {
                "mode": "numpy",
                "rep": rep,
                "warmup": rep < args.warmups,
                "n": args.n,
                "wall_s": round(wall, 6),
                "checksum": checksum(C),
            }
            if args.ref:
                rec["validation"] = validate(C, args.ref)
            print(json.dumps(rec), flush=True)
        return 0

    t0 = perf_counter()
    spark = build_session(args.master, app_name=args.app_name,
                          executor_memory=args.executor_memory,
                          event_log_dir=args.eventlog_dir)
    t_startup = perf_counter() - t0
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")

    t_replicate = replicate(sc, args.b)

    records = []
    for rep in range(args.warmups + args.reps):
        t0 = perf_counter()
        C, phases, stats = multiply(sc, A, b_name, args.n, args.slices)
        wall = perf_counter() - t0
        phases["t_load"] = t_load
        rec = {
            "mode": "distributed",
            "rep": rep,
            "warmup": rep < args.warmups,
            "n": args.n,
            "slices": args.slices,
            "phases_s": {k: round(v, 6) for k, v in phases.items()},
            "wall_s": round(wall, 6),
            "t_startup_s": round(t_startup, 6),
            "t_replicate_s": round(t_replicate, 6),
            "checksum": checksum(C),
            "app_id": sc.applicationId,
            **stats,
        }
        if args.ref:
            rec["validation"] = validate(C, args.ref)
        records.append(rec)
        print(json.dumps(rec), flush=True)

        if rep == args.warmups + args.reps - 1 and args.out:
            C.astype("<f8").tofile(args.out)
            print(json.dumps({"mode": "dump", "out": args.out,
                              "bytes": C.nbytes}), flush=True)

    if args.eventlog_dir:
        try:
            print(json.dumps({"mode": "eventlog",
                              **parse_eventlog(args.eventlog_dir, sc.applicationId)}),
                  flush=True)
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"mode": "eventlog", "error": str(exc)}), flush=True)

    spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
