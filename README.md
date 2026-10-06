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
matrix-multiplication-study/
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

- Python 3.11+ with `numpy` (input generation and correctness reference only,
  never inside the measured kernel), `psutil`, `pandas`, `matplotlib`, `jupyter`
- JDK 21 (`java`, `javac`)
- GCC (MinGW-w64 on Windows)
- A LaTeX distribution (`pdflatex`, `latexmk`) for the report

## How to reproduce

```bash
# 1. Generate the shared inputs (fixed seed, byte-identical for all languages)
python scripts/generate_inputs.py

# 2. Verify correctness before any benchmarking  (Phase 2, todo)
python scripts/check_correctness.py

# 3. Run the benchmark suite                     (Phase 4, todo)
python scripts/run_benchmark.py

# 4. Analyse the raw measurements                (Phase 6, todo)
jupyter notebook notebooks/01_analysis.ipynb
```

## Status

- [x] Phase 0: repository skeleton
- [x] Phase 1: shared input generation
- [ ] Phase 2: correctness validation
- [ ] Phase 3: timing harnesses
- [ ] Phase 4: benchmark orchestrator
- [ ] Phase 5: full measurement campaign
- [ ] Phase 6: analysis notebook
- [ ] Phase 7: LaTeX report
- [ ] Phase 8: final reproducibility review
