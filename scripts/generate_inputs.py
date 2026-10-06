#!/usr/bin/env python3
"""Generate the shared input matrices for the matrix multiplication study.

For every declared size n, two dense n x n matrices (A and B) are written as
raw little-endian IEEE-754 binary64 files in row-major (C) order, plus a
meta.json manifest recording the seed, generator, value range, dimensions and
SHA-256 checksum of every file.

The same files are read by the Python, Java and C implementations, so all
three languages multiply byte-identical logical inputs. NumPy is used only for
generation and never inside a measured kernel.

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
VALUE_RANGE = (-1.0, 1.0)
MATRICES = (("A", 0), ("B", 1))


def make_matrix(n: int, matrix_index: int, seed: int) -> np.ndarray:
    """Deterministic input matrix: uniform values in [-1, 1), float64.

    Each (seed, n, matrix_index) triple derives an independent stream, so the
    files do not depend on the order in which sizes are generated.
    """
    rng = np.random.default_rng([seed, n, matrix_index])
    lo, hi = VALUE_RANGE
    return rng.uniform(lo, hi, size=(n, n)).astype("<f8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def generate(sizes: list[int], seed: int, outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    entries = []
    for n in sizes:
        for name, idx in MATRICES:
            path = outdir / f"{name}_{n}.bin"
            arr = make_matrix(n, idx, seed)
            payload = arr.tobytes(order="C")
            path.write_bytes(payload)

            # Read-back verification: what the benchmarking programs will load.
            back = np.fromfile(path, dtype="<f8")
            if back.size != n * n or not np.array_equal(back, arr.ravel(order="C")):
                raise RuntimeError(f"read-back verification failed for {path}")
            if len(payload) != 8 * n * n:
                raise RuntimeError(f"unexpected byte size for {path}")

            entries.append(
                {
                    "file": path.name,
                    "matrix": name,
                    "n": n,
                    "bytes": len(payload),
                    "sha256": sha256_bytes(payload),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                }
            )
            print(f"  wrote {path.name:>12}  {len(payload):>9} bytes  sha256={entries[-1]['sha256'][:16]}...")

    meta = {
        "description": "Shared input matrices for Assignment 1 (basic dense matrix multiplication).",
        "generator": "numpy.random.Generator.uniform (PCG64)",
        "seed": seed,
        "seed_scheme": "numpy.random.default_rng([seed, n, matrix_index]) with matrix_index: A=0, B=1",
        "value_range": list(VALUE_RANGE),
        "dtype": "float64 (IEEE-754 binary64), little-endian '<f8'",
        "layout": "row-major (C order)",
        "dimensions": "one pair of n x n square matrices (A, B) per size",
        "sizes": list(sizes),
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
        print(f"ERROR: {meta_path} not found; run without --check first.", file=sys.stderr)
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
        print(f"{status} {entry['file']}: size={len(payload)} sha256={sha256_bytes(payload)[:16]}...")
    if failures:
        print(f"{failures} file(s) failed verification.", file=sys.stderr)
        return 1
    print(f"All {len(meta['files'])} input files verified against meta.json.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed (default: %(default)s)")
    ap.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES, help="matrix side lengths")
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="output directory")
    ap.add_argument("--check", action="store_true", help="verify existing files against meta.json")
    args = ap.parse_args()

    if args.check:
        return check(args.outdir)

    print(f"Generating inputs: sizes={args.sizes} seed={args.seed} outdir={args.outdir}")
    generate(args.sizes, args.seed, args.outdir)
    print("Done. Verify anytime with: python scripts/generate_inputs.py --check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
