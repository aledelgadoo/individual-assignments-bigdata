# Matrix Multiplication Study — Assignment 1

Basic dense matrix multiplication (classical triple loop, O(n^3)) implemented
from scratch in **Python**, **Java** and **C**, benchmarked under a controlled,
reproducible protocol.

Research question: *How does matrix size affect runtime and memory use for the
three basic implementations?*

Course: BIG DATA, Grado en Ciencia e Ingeniería de Datos, Universidad de Las
Palmas de Gran Canaria. Academic year 2026-2027.

Author: Alejandro Delgado Valera.

## Repository structure

```
individual-assignments-bigdata/
├── data/
│   ├── inputs/          # shared input matrices (binaries + meta.json manifest)
│   └── results/         # raw and summary measurements (CSV/JSON)
├── src/
│   ├── python/          # list-of-lists triple-loop implementation + harness
│   ├── java/            # double[][] triple-loop implementation + harness
│   └── c/               # double* triple-loop implementation + harness
├── scripts/             # input generation, correctness checks, benchmark driver
├── notebooks/           # analysis (tables and figures from raw CSV)
├── reports/             # final report (LaTeX) and figures
└── README.md
```

## Prerequisites

- Python 3.11.9 with `numpy` 2.2.6 (input generation and correctness reference
  only, never inside the measured kernel), `psutil` 7.1.0, `pandas`, `matplotlib`,
  `jupyter`
- OpenJDK 21.0.10 (`java`, `javac`)
- GCC 16.1.0 (MinGW-w64 on Windows)
- A LaTeX distribution (`pdflatex`, `latexmk`) for the report

## How to reproduce

```bash
# 1. Generate the shared inputs (fixed seed, byte-identical for all languages)
python scripts/generate_inputs.py

# 2. Verify correctness before any benchmarking
python scripts/check_correctness.py

# 3. Run the benchmark suite                     (Phase 4)
python scripts/run_benchmark.py

# 4. Analyse the raw measurements                (Phase 6)
jupyter nbconvert --to notebook --execute notebooks/01_analysis.ipynb

# 5. Build the report                            (Phase 7)
cd reports && latexmk -pdf Assignment01_DelgadoValera_Report.tex
```
