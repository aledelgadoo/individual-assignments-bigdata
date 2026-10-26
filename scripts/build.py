#!/usr/bin/env python3
"""Build the Assignment 3 executables from one shared C source.

Variants (same kernels.c compiled four ways):
    seq    -O2                       sequential baseline (no OpenMP)
    scalar -O2 -fopenmp -fno-tree-vectorize   parallel, SIMD disabled (control)
    auto   -O2 -fopenmp              parallel, compiler autovectorisation
    simd   -O2 -fopenmp -DUSE_OMP_SIMD       parallel, explicit #pragma omp simd

Writes src/variant.h before each compile (gitignored) and, after building,
records vectorisation evidence (opt-info lines + objdump packed-instruction
counts) in data/results/vectorisation.txt.

Usage:
    python scripts/build.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
BUILD = ROOT / "build"
RESULTS = ROOT / "data" / "results"

SOURCES = ["common.c", "kernels.c", "selftest.c", "main.c"]

VARIANTS = {
    "seq": ["-O2", "-march=native"],
    "scalar": ["-O2", "-march=native", "-fopenmp", "-fno-tree-vectorize",
               "-DUSE_OMP"],
    "auto": ["-O2", "-march=native", "-fopenmp", "-DUSE_OMP"],
    "simd": ["-O2", "-march=native", "-fopenmp", "-DUSE_OMP",
             "-DUSE_OMP_SIMD"],
}

PACKED_RE = re.compile(
    r"\b(vfmadd\w*|vfmsub\w*|vmulpd|vaddpd|vsubpd|vmovapd|vmovupd|vunpcklpd|"
    r"vpermilpd|vinsertf128|vextractf128)\b.*\b(ymm|zmm)"
)


def write_variant_h(name: str) -> None:
    (SRC / "variant.h").write_text(
        f'#ifndef VARIANT_H\n#define VARIANT_H\n\n#define VARIANT_NAME "{name}"\n\n#endif\n',
        encoding="utf-8",
    )


def build_one(name: str, extra: list[str]) -> dict:
    write_variant_h(name)
    BUILD.mkdir(exist_ok=True)
    vec_file = BUILD / f"vec_{name}.txt"
    if vec_file.exists():
        vec_file.unlink()
    flags = ["-std=c11", "-Wall", "-Wextra", *extra]
    if name in ("seq", "auto", "simd"):
        flags.append(f"-fopt-info-vec-optimized={vec_file}")
    cmd = [
        "gcc", *flags,
        *[(SRC / s).as_posix() for s in SOURCES],
        "-o", (SRC / f"matmul3_{name}.exe").as_posix(),
        "-lpsapi", "-lm",
    ]
    print(f"[build] {name}: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout.strip():
        print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(f"build failed for variant {name}")
    if proc.stderr.strip():
        print(f"[warn] {name} stderr:\n{proc.stderr}")
    return {"name": name, "vec_file": vec_file}


def objdump_packed_count(exe: Path) -> tuple[int, list[str]]:
    proc = subprocess.run(["objdump", "-d", exe.as_posix()],
                          capture_output=True, text=True, errors="replace")
    hits = [ln.strip() for ln in proc.stdout.splitlines() if PACKED_RE.search(ln)]
    return len(hits), hits[:20]


def main() -> int:
    BUILD.mkdir(exist_ok=True)
    built = []
    for name, extra in VARIANTS.items():
        built.append(build_one(name, extra))

    lines = [
        "Vectorisation evidence (Assignment 3)",
        f"generated_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
    ]
    for info in built:
        name = info["name"]
        exe = SRC / f"matmul3_{name}.exe"
        lines.append(f"=== {name} ({exe.name}) ===")
        vec = info["vec_file"]
        if vec.exists():
            text = vec.read_text(encoding="utf-8", errors="replace").strip()
            lines.append("-fopt-info-vec-optimized:")
            lines.append(text if text else "  (no vectorised loops reported)")
        else:
            lines.append("-fopt-info-vec-optimized: (not requested)")
        count, sample = objdump_packed_count(exe)
        lines.append(f"objdump packed ymm/zmm instructions (whole exe): {count}")
        for s in sample:
            lines.append(f"  {s}")
        lines.append("")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "vectorisation.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[build] wrote {out.relative_to(ROOT)}")
    print("[build] OK: " + ", ".join(VARIANTS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
