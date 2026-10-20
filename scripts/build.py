#!/usr/bin/env python3
"""Build the C implementation for Assignment 2.

C: gcc -O2 -std=c11 -Wall -Wextra  ->  src/matmul2[.exe]

The flags are fixed and documented in the report; they are part of the
experimental configuration and must not change between runs.

Usage:
    python scripts/build.py           # build
    python scripts/build.py --clean   # remove build artefacts first
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
C_SRC = sorted((ROOT / "src").glob("*.c"))
C_OUT = ROOT / "src" / ("matmul2.exe" if os.name == "nt" else "matmul2")
C_FLAGS = ["-O2", "-std=c11", "-Wall", "-Wextra"]


def run(cmd: list[str]) -> None:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    proc = subprocess.run([str(c) for c in cmd], cwd=ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"build failed with exit code {proc.returncode}")


def build_c() -> None:
    if not C_SRC:
        raise SystemExit("no C sources found under src/")
    cc = os.environ.get("CC", "gcc")
    link = ["-lpsapi"] if os.name == "nt" else []
    run([cc, *C_FLAGS, "-o", C_OUT, *C_SRC, *link])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", action="store_true", help="remove build artefacts first")
    args = ap.parse_args()

    if args.clean and C_OUT.exists():
        C_OUT.unlink()
        print(f"removed {C_OUT}")
    build_c()
    print(f"Build OK -> {C_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
