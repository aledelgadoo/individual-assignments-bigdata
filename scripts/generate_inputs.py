#!/usr/bin/env python3
"""Generate the shared input matrices for Assignment 2.

Dense Part A: for every declared size n, two dense n x n matrices (A and B)
are written as raw little-endian IEEE-754 binary64 files in row-major order,
using exactly the generator and seed scheme of Assignment 1 so the validated
baseline and inputs are reused unchanged.

Sparse Part B: for every declared sparsity level p (percentage of zeros) at
n = 1000, a sparse variant of the same A matrix is written by zeroing the
entries selected by an independent deterministic mask stream. The dense
matrix B is shared by every sparsity level.

A meta.json manifest records the seed, generator, value range, dimensions,
non-zero counts and SHA-256 checksum of every file.

Usage:
    python scripts/generate_inputs.py             # generate everything
    python scripts/generate_inputs.py --check     # verify existing files
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTDIR = ROOT / "data" / "inputs"
DEFAULT_SEED = 42
DEFAULT_SIZES = [10, 50, 100, 200, 500, 1000, 2000]
SPARSITIES = [1, 5, 10, 25, 50, 75, 90, 95, 99]
SPARSE_N = 1000
VALUE_RANGE = (-1.0, 1.0)
MATRICES = (("A", 0), ("B", 1))


def make_matrix(n: int, matrix_index: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng([seed, n, matrix_index])
    lo, hi = VALUE_RANGE
    return rng.uniform(lo, hi, size=(n, n)).astype("<f8")


def make_sparse(n: int, sparsity: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    base = make_matrix(n, 0, seed)
    mask_rng = np.random.default_rng([seed, n, sparsity, 2])
    keep = mask_rng.random((n, n)) >= (sparsity / 100.0)
    arr = np.where(keep, base, 0.0).astype("<f8")
    return arr, keep


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_entry(outdir: Path, name: str, arr: np.ndarray, entries: list, **extra) -> None:
    path = outdir / f"{name}.bin"
    payload = arr.tobytes(order="C")
    path.write_bytes(payload)

    back = np.fromfile(path, dtype="<f8")
    if back.size != arr.size or not np.array_equal(back, arr.ravel(order="C")):
        raise RuntimeError(f"read-back verification failed for {path}")
    if len(payload) != 8 * arr.size:
        raise RuntimeError(f"unexpected byte size for {path}")

    entries.append({
        "file": path.name,
        "n": int(arr.shape[0]),
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "min": float(arr.min()),
        "max": float(arr.max()),
        **extra,
    })
    print(f"  wrote {path.name:>20}  {len(payload):>9} bytes  "
          f"sha256={entries[-1]['sha256'][:16]}...")


def generate(sizes: list[int], seed: int, outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    entries = []

    for n in sizes:
        for name, idx in MATRICES:
            arr = make_matrix(n, idx, seed)
            write_entry(outdir, f"{name}_{n}", arr, entries,
                        matrix=name, kind="dense",
                        seed_scheme=f"default_rng([{seed}, {n}, {idx}])")

    for p in SPARSITIES:
        arr, keep = make_sparse(SPARSE_N, p, seed)
        nnz = int(np.count_nonzero(arr))
        if not (0 < nnz < arr.size):
            raise RuntimeError(f"degenerate sparsity {p}% (nnz={nnz})")
        write_entry(outdir, f"A_{SPARSE_N}_sp{p:02d}", arr, entries,
                    matrix="A", kind="sparse",
                    sparsity_percent=p, nnz=nnz,
                    seed_scheme=f"default_rng([{seed}, {SPARSE_N}, {p}, 2]) mask "
                                f"over dense A default_rng([{seed}, {SPARSE_N}, 0])")

    meta = {
        "description": "Shared input matrices for Assignment 2 (dense "
                       "optimisations and CSR sparse multiplication).",
        "generator": "numpy.random.Generator.uniform (PCG64)",
        "seed": seed,
        "seed_scheme_dense": "numpy.random.default_rng([seed, n, matrix_index]) "
                             "with matrix_index: A=0, B=1 (identical to Assignment 1)",
        "seed_scheme_sparse": "numpy.random.default_rng([seed, n, sparsity, 2]) "
                              "boolean keep-mask; zeros where mask selects",
        "value_range": list(VALUE_RANGE),
        "dtype": "float64 (IEEE-754 binary64), little-endian '<f8'",
        "layout": "row-major (C order)",
        "dense_sizes": list(sizes),
        "sparsity_levels_percent": list(SPARSITIES),
        "sparse_n": SPARSE_N,
        "numpy_version": np.__version__,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/generate_inputs.py",
        "files": entries,
    }
    meta_path = outdir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote meta.json ({meta_path.stat().st_size} bytes)")
    return meta


def check(outdir: Path) -> int:
    meta_path = outdir / "meta.json"
    if not meta_path.exists():
        print(f"ERROR: {meta_path} not found; run without --check first.",
              file=sys.stderr)
        return 1
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    failures = 0
    for entry in meta["files"]:
        path = outdir / entry["file"]
        if not path.exists():
            print(f"FAIL {entry['file']}: missing")
            failures += 1
            continue
        payload = path.read_bytes()
        ok_size = len(payload) == entry["bytes"]
        ok_hash = sha256_bytes(payload) == entry["sha256"]
        status = "ok  " if (ok_size and ok_hash) else "FAIL"
        if not (ok_size and ok_hash):
            failures += 1
        print(f"{status} {entry['file']}: size={len(payload)} "
              f"sha256={sha256_bytes(payload)[:16]}...")
    if failures:
        print(f"{failures} file(s) failed verification.", file=sys.stderr)
        return 1
    print(f"All {len(meta['files'])} input files verified against meta.json.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if args.check:
        return check(args.outdir)

    print(f"Generating inputs: sizes={args.sizes} seed={args.seed} "
          f"sparsities={SPARSITIES} sparse_n={SPARSE_N} outdir={args.outdir}")
    generate(args.sizes, args.seed, args.outdir)
    print("Done. Verify anytime with: python scripts/generate_inputs.py --check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
