# Matrix Multiplication Study — Assignment 2

Optimised dense matrix multiplication (**loop reordering**, **cache
blocking**) and **CSR sparse multiplication**, all in C, benchmarked under a
controlled, reproducible protocol against the validated baseline of Part A.

Research question: *How do loop reordering, cache blocking and CSR sparse
storage affect runtime and memory compared with the baseline triple-loop
implementation?*

Course: BIG DATA, Grado en Ciencia e Ingeniería de Datos, Universidad de Las
Palmas de Gran Canaria. Academic year 2026-2027.

Author: Alejandro Delgado Valera.

Reuses from Assignment 1 (tag `assignment-1`): the validated `i-j-k` baseline
kernel and the input generator/seed scheme, so all inputs are byte-identical
where sizes overlap.

## Repository structure

```
individual-assignments-bigdata/
├── data/
│   ├── inputs/          # shared input matrices + meta.json manifest
│   └── results/         # raw and summary measurements (CSV/JSON)
├── src/
│   ├── common.[ch]      # timing, peak RSS, affinity, loading
│   ├── dense.[ch]       # ijk baseline, ikj reordering, blocked tiling
│   ├── sparse.[ch]      # CSR build + CSR x dense multiplication
│   ├── selftest.[ch]    # correctness suite (bitwise + long double reference)
│   └── main.c           # CLI: selftest | probe | bench
├── scripts/             # build, input generation, benchmark driver, env
├── notebooks/           # analysis (tables and figures from raw CSV)
├── reports/             # final report (LaTeX) and figures
└── README.md
```

## Prerequisites

- Python 3.11.9 with `numpy` 2.2.6 (input generation only, never inside a
  measured kernel), `psutil` 7.1.0, `pandas`, `matplotlib`, `jupyter`
- GCC 16.1.0 (MinGW-w64 on Windows), flags fixed at `-O2 -std=c11 -Wall -Wextra`
- A LaTeX distribution (`pdflatex`, `latexmk`) for the report

## How to reproduce

```bash
# 1. Generate the shared inputs (dense pairs + sparse variants, fixed seed)
python scripts/generate_inputs.py
python scripts/generate_inputs.py --check

# 2. Build and validate before any benchmarking
python scripts/build.py
python src/matmul2.exe selftest

# 3. (Optional) pick the cache-blocking size empirically
python scripts/run_benchmark.py --phase tune

# 4. Run the benchmark campaign (phases A and B)
python scripts/run_benchmark.py --phase all

# 5. Environment snapshot
python scripts/collect_env.py

# 6. Analyse the raw measurements
jupyter nbconvert --to notebook --execute notebooks/02_analysis.ipynb

# 7. Build the report
cd reports && latexmk -pdf Assignment02_DelgadoValera_Report.tex
```

## Measurement contract

- Kernel-only timing (`QueryPerformanceCounter`), input loading, `C`
  initialisation and checksums outside the timed region.
- Budget: 60 s per multiplication, 4 GiB peak RSS; probe gate + one retry,
  timeouts reported without an invented runtime.
- 1 warm-up (excluded) + 5 measured repetitions, batched so each timing spans
  at least 0.2 s; median and IQR.
- Every child pinned to CPU0 (a P-core); system CPU% recorded per config.
- CSR: dense-to-CSR conversion timed separately (`conversions.csv`), never
  inside the multiplication timing.

## Output files (`data/results/`)

| File | Contents |
|---|---|
| `raw_timings.csv` | one row per timed kernel event |
| `conversions.csv` | one row per timed dense-to-CSR conversion |
| `configs.csv` | one row per configuration (status, probes, RSS, CPU) |
| `blocksize_tuning.csv` | block-size sweep (16/32/64/128) |
| `environment.json` | machine, toolchain, cache, protocol |
| `raw_detail/*.json` | raw stdout of every child process |
