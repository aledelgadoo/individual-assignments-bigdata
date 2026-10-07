#!/usr/bin/env python3
"""Build the C and Java implementations.

C:    gcc -O2 -std=c11 -Wall -Wextra  ->  src/c/matmul[.exe]
Java: javac -encoding UTF-8 -d src/java  ->  src/java/MatMul.class

The flags are fixed and documented in the report; they are part of the
experimental configuration and must not change between runs.

Usage:
    python scripts/build.py           # build everything
    python scripts/build.py --clean   # remove build artefacts first
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
C_SRC = ROOT / "src" / "c" / "matmul.c"
C_OUT = ROOT / "src" / "c" / ("matmul.exe" if os.name == "nt" else "matmul")
JAVA_SRC = ROOT / "src" / "java" / "MatMul.java"
JAVA_OUT_DIR = ROOT / "src" / "java"

C_FLAGS = ["-O2", "-std=c11", "-Wall", "-Wextra"]


def run(cmd: list[str]) -> None:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    proc = subprocess.run([str(c) for c in cmd], cwd=ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"build failed with exit code {proc.returncode}")


def build_c() -> None:
    cc = os.environ.get("CC", "gcc")
    link = ["-lpsapi"] if os.name == "nt" else []  # GetProcessMemoryInfo (peak RSS)
    run([cc, *C_FLAGS, "-o", C_OUT, C_SRC, *link])


def build_java() -> None:
    javac = os.environ.get("JAVAC", "javac")
    run([javac, "-encoding", "UTF-8", "-d", JAVA_OUT_DIR, JAVA_SRC])


def clean() -> None:
    for path in (C_OUT, JAVA_OUT_DIR / "MatMul.class"):
        if path.exists():
            path.unlink()
            print(f"removed {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", action="store_true", help="remove build artefacts first")
    ap.add_argument("--only", choices=["c", "java"], help="build only one language")
    args = ap.parse_args()

    if args.clean:
        clean()
    if args.only in (None, "c"):
        build_c()
    if args.only in (None, "java"):
        build_java()
    print("Build OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
