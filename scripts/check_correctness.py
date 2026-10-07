#!/usr/bin/env python3
"""Correctness validation for the three basic implementations.

Validates Python, Java and C against a trusted reference (NumPy float64 matmul,
used only for checking, never as a student implementation) on a declared set of
test cases, and cross-checks the three implementations against each other.

Test cases:
    known_3            hand-computed 3x3 product (also checks the reference)
    identity_right_64  A x I = A
    identity_left_64   I x A = A
    zero_right_64      A x 0 = 0
    zero_left_64       0 x A = 0
    random_7           ad-hoc random input (odd, prime size)
    random_33          ad-hoc random input (odd, prime size)
    shared_10          shared benchmark input A_10 x B_10
    shared_100         shared benchmark input A_100 x B_100

Tolerances for non-exact cases: rtol=1e-9, atol=1e-12.
Structured cases (known, identity, zero) must be exactly equal (value-wise).

Usage:
    python scripts/check_correctness.py
    python scripts/check_correctness.py --skip-build
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "data" / "inputs"
WORKDIR = ROOT / "data" / "results" / "correctness"
REPORT = ROOT / "data" / "results" / "correctness_report.csv"

RTOL = 1e-9
ATOL = 1e-12
CASE_SEED_BASE = 4242
RUN_TIMEOUT_S = 180

C_BIN = ROOT / "src" / "c" / ("matmul.exe" if __import__("os").name == "nt" else "matmul")


# --------------------------------------------------------------------------- #
# test cases
# --------------------------------------------------------------------------- #

def build_cases() -> list[dict]:
    rng = np.random.default_rng(CASE_SEED_BASE)
    I64 = np.eye(64, dtype=np.float64)
    Z64 = np.zeros((64, 64), dtype=np.float64)
    A64 = rng.uniform(-1.0, 1.0, size=(64, 64))
    A7 = np.random.default_rng(CASE_SEED_BASE + 7).uniform(-1.0, 1.0, size=(7, 7))
    A33 = np.random.default_rng(CASE_SEED_BASE + 33).uniform(-1.0, 1.0, size=(33, 33))

    A3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
    B3 = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]], dtype=np.float64)
    EXPECTED3 = np.array([[30, 24, 18], [84, 69, 54], [138, 114, 90]], dtype=np.float64)

    return [
        {"name": "known_3", "A": A3, "B": B3, "expected": EXPECTED3, "exact": True},
        {"name": "identity_right_64", "A": A64, "B": I64, "exact": True},
        {"name": "identity_left_64", "A": I64, "B": A64, "exact": True},
        {"name": "zero_right_64", "A": A64, "B": Z64, "exact": True},
        {"name": "zero_left_64", "A": Z64, "B": A64, "exact": True},
        {"name": "random_7", "A": A7, "B": A7.T.copy(), "exact": False},
        {"name": "random_33", "A": A33, "B": A33[::-1].copy(), "exact": False},
        {"name": "shared_10", "shared": ("A_10.bin", "B_10.bin", 10), "exact": False},
        {"name": "shared_100", "shared": ("A_100.bin", "B_100.bin", 100), "exact": False},
    ]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def write_f64(path: Path, arr: np.ndarray) -> None:
    path.write_bytes(np.ascontiguousarray(arr, dtype="<f8").tobytes())


def load_f64(path: Path, n: int) -> np.ndarray:
    data = np.fromfile(path, dtype="<f8")
    if data.size != n * n:
        raise ValueError(f"{path}: expected {n * n} doubles, got {data.size}")
    return data.reshape(n, n)


def max_rel_err(cand: np.ndarray, ref: np.ndarray) -> float:
    diff = np.abs(cand - ref)
    rel = np.empty_like(diff)
    nz = ref != 0
    rel[nz] = diff[nz] / np.abs(ref[nz])
    rel[~nz] = diff[~nz]
    return float(rel.max())


def compare(row: dict, cand: np.ndarray, ref: np.ndarray) -> None:
    exact = bool(row["exact_required"])
    row["bitwise_equal"] = bool(cand.tobytes() == ref.tobytes())
    row["max_abs_err"] = float(np.max(np.abs(cand - ref)))
    row["max_rel_err"] = max_rel_err(cand, ref)
    if exact:
        row["passed"] = bool(np.array_equal(cand, ref))
        row["criterion"] = "exact"
    else:
        row["passed"] = bool(np.allclose(cand, ref, rtol=RTOL, atol=ATOL))
        row["criterion"] = f"allclose(rtol={RTOL:g}, atol={ATOL:g})"


def run_lang(lang: str, a: Path, b: Path, n: int, out: Path) -> str:
    if lang == "python":
        cmd = [sys.executable, str(ROOT / "src" / "python" / "matmul.py"),
               "multiply", "--a", str(a), "--b", str(b), "--n", str(n), "--out", str(out)]
    elif lang == "java":
        cmd = ["java", "-cp", str(ROOT / "src" / "java"), "MatMul",
               "multiply", "--a", str(a), "--b", str(b), "--n", str(n), "--out", str(out)]
    elif lang == "c":
        cmd = [str(C_BIN), "multiply", "--a", str(a), "--b", str(b), "--n", str(n), "--out", str(out)]
    else:
        raise ValueError(lang)
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=RUN_TIMEOUT_S)
    if proc.returncode != 0:
        raise RuntimeError(f"{lang} failed on {a.name}/{b.name}: {proc.stderr.strip()}")
    return proc.stdout.strip()


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-build", action="store_true", help="do not rebuild C/Java first")
    args = ap.parse_args()

    if not (INPUTS / "meta.json").exists():
        print("ERROR: shared inputs missing; run scripts/generate_inputs.py first.", file=sys.stderr)
        return 1
    if not args.skip_build:
        proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "build.py")], cwd=ROOT)
        if proc.returncode != 0:
            return proc.returncode

    shutil.rmtree(WORKDIR, ignore_errors=True)
    WORKDIR.mkdir(parents=True, exist_ok=True)

    langs = ["python", "java", "c"]
    rows: list[dict] = []

    for case in build_cases():
        name = case["name"]
        if "shared" in case:
            fa, fb, n = case["shared"]
            path_a, path_b = INPUTS / fa, INPUTS / fb
            A = load_f64(path_a, n)
            B = load_f64(path_b, n)
        else:
            A, B = case["A"], case["B"]
            n = A.shape[0]
            path_a, path_b = WORKDIR / f"A_{name}.bin", WORKDIR / f"B_{name}.bin"
            write_f64(path_a, A)
            write_f64(path_b, B)

        ref = (A @ B).astype(np.float64)

        # sanity-check the reference itself against the hand-computed answer
        if "expected" in case:
            exp = case["expected"]
            rows.append({
                "case": name, "n": n, "comparison": "reference_vs_hand", "language": "numpy",
                "exact_required": True, "criterion": "exact", "passed": bool(np.array_equal(ref, exp)),
                "bitwise_equal": bool(ref.tobytes() == exp.tobytes()),
                "max_abs_err": float(np.max(np.abs(ref - exp))),
                "max_rel_err": max_rel_err(ref, exp), "rtol": 0.0, "atol": 0.0, "detail": "",
            })
            case["exact"] = case["exact"] and np.array_equal(ref, exp)

        outputs: dict[str, np.ndarray] = {}
        for lang in langs:
            out = WORKDIR / f"C_{name}_{lang}.bin"
            detail = run_lang(lang, path_a, path_b, n, out)
            cand = load_f64(out, n)
            outputs[lang] = cand
            row = {
                "case": name, "n": n, "comparison": "vs_reference", "language": lang,
                "exact_required": case["exact"], "rtol": 0.0 if case["exact"] else RTOL,
                "atol": 0.0 if case["exact"] else ATOL, "detail": detail,
            }
            compare(row, cand, ref)
            rows.append(row)

        # cross-language pairwise comparison
        for i in range(len(langs)):
            for j in range(i + 1, len(langs)):
                la, lb = langs[i], langs[j]
                row = {
                    "case": name, "n": n, "comparison": "pairwise",
                    "language": f"{la}~{lb}", "exact_required": False,
                    "rtol": RTOL, "atol": ATOL, "detail": "",
                }
                compare(row, outputs[la], outputs[lb])
                rows.append(row)

        print(f"  case {name:<18} n={n:<4} done")

    fields = ["case", "n", "comparison", "language", "exact_required", "criterion",
              "passed", "bitwise_equal", "max_abs_err", "max_rel_err", "rtol", "atol", "detail"]
    with REPORT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    failures = [r for r in rows if not r["passed"]]
    print(f"\n{len(rows)} comparisons written to {REPORT.relative_to(ROOT)}")
    for r in failures:
        print(f"  FAIL {r['case']} {r['comparison']} {r['language']}: "
              f"max_abs_err={r['max_abs_err']:.3e} max_rel_err={r['max_rel_err']:.3e}")
    if failures:
        print(f"{len(failures)} comparison(s) FAILED.", file=sys.stderr)
        return 1
    print("All correctness checks PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
