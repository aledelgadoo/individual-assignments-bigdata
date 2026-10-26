# Assignment 3 — Parallel & SIMD dense matrix multiplication

OpenMP parallelisation and SIMD vectorisation of dense matrix multiplication
on an Intel hybrid CPU (i7-1355U: 2 P-cores + 8 E-cores, 12 logical CPUs).

The kernel is the `ikj` ordering validated in Assignment 2 (row-parallel,
elementwise inner loop). The equivalent sequential baseline is the same
kernel without OpenMP, compiled with the same optimisation level.

## Build (four binaries from one shared source)

```powershell
python scripts/build.py
```

| Binary               | Flags                                                | Purpose                          |
|----------------------|------------------------------------------------------|----------------------------------|
| `matmul3_seq.exe`    | `-O2 -march=native`                                 | Sequential baseline              |
| `matmul3_scalar.exe` | `-O2 -march=native -fopenmp -fno-tree-vectorize`    | Parallel, SIMD disabled (control)|
| `matmul3_auto.exe`   | `-O2 -march=native -fopenmp`                        | Parallel, autovectorisation      |
| `matmul3_simd.exe`   | `-O2 -march=native -fopenmp -DUSE_OMP_SIMD`         | Parallel, `#pragma omp simd`     |

Vectorisation evidence: `data/results/vectorisation.txt` (opt-info + objdump).

## Inputs

```powershell
python scripts/generate_inputs.py          # dense A_n/B_n, byte-identical to A1/A2
python scripts/generate_inputs.py --check
```

## Correctness

```powershell
src\matmul3_simd.exe selftest              # known answers, long-double reference,
                                           # bitwise thread invariance (p=2..12)
```

## Campaign

```powershell
python scripts/run_benchmark3.py --phase all     # seq | scale | simd | pcore
python scripts/collect_env.py
```

- `seq`: sequential baseline, all n.
- `scale`: S(p), E(p) for p in {1,2,4,8,12,16} (16 > 12 contextos → oversubscription).
- `simd`: scalar vs simd at p in {1,8,12}, n in {200,500,1000,2000}.
- `pcore`: `matmul3_auto` confined to CPU0-3 (P-core threads only).

Protocol per config: probe (gate = min(60 s, 2×max probe)) → 1 warmup →
batch calibration (≥0.2 s per batch) → 5 measured repetitions. JSON lines
per child run land in `data/results/raw_detail/`; per-config status in
`configs.csv`; measurements in `raw_timings.csv`.

Set the power plan to **High performance** for the campaign
(`powercfg /setactive SCHEME_MIN`), restore Balanced afterwards.

## Analysis

`notebooks/03_analysis.ipynb` → figures in `reports/figures/`,
tables + narrative in `reports/Assignment03_DelgadoValera_Report.tex`.
