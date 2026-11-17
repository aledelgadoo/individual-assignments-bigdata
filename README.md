# Assignment 4 — Distributed dense matrix multiplication (PySpark)

Row-block partitioning of `A` with `B` replicated to every worker, one
numpy/BLAS kernel per block, results collected on the driver — run on a
PySpark 4.3 **standalone cluster on localhost** (1 master + N single-core
worker processes, driver = the submitting Python process, **one physical
machine**).

Research question: how does the distributed multiplication scale with
workers, how is the end-to-end time split between computation,
communication and framework overhead, and how does it compare with the
validated sequential and parallel baselines from Assignments 1–3.

Everything runs over loopback, so no inter-machine network claim is made.

## Environment

```powershell
python -m pip install -r requirements-a4.txt   # pyspark, numpy, psutil, zstandard
python scripts/fetch_winutils.py               # HADOOP_HOME\bin\winutils.exe -> build/hadoop (git-ignored)
```

`build/hadoop` is required because Spark writes its event log through
Hadoop's `LocalFileSystem`; only Hadoop FileSystem calls need it.

Windows fixes applied in `src/spark/sparkconf.py`:

| Problem | Fix |
|---------|-----|
| hostname is not a valid Spark URL host | `spark.driver.host` / `spark.driver.bindAddress` = `127.0.0.1` |
| App Execution Alias for `python` prints the Store banner | `PYSPARK_PYTHON` / `spark.pyspark.python` = real interpreter path |
| worker daemons must bind loopback | `SPARK_LOCAL_IP=127.0.0.1` |
| `sc.broadcast` fails with `ERROR_SHARING_VIOLATION` | `sc.addFile()` + `SparkFiles.get()` instead (same `8n²` bytes per worker) |
| Windows cannot reuse Python workers (`os.fork` unavailable) | every task spawns a fresh worker — measured, documented in the report |
| BLAS would use every core | `OPENBLAS_NUM_THREADS=1` set before `numpy` is imported, plus `spark.executorEnv.*` |

## Inputs

```powershell
python scripts/generate_inputs.py          # dense A_n/B_n for n in 10..4000
python scripts/generate_inputs.py --check
```

Sizes 10–2000 are byte-identical to Assignments 1–3; 2500/3200/4000 are the
Assignment 4 weak-scaling sizes (`n(p) = 2000 · p^(1/3)`).

## Correctness reference

```powershell
python scripts/build.py                    # matmul3_* binaries (Assignment 3 kernels)
src\matmul3_seq.exe dump --n 4000 --a data\inputs\A_4000.bin --b data\inputs\B_4000.bin --out build\ref\C_4000_seq.bin
```

`dump` writes the validated sequential result to `build/ref/C_<n>_seq.bin`;
every distributed run is compared against it with `rtol=1e-9`, `atol=1e-12`
(BLAS summation order differs from `ikj`, so bitwise equality is not expected).

## Cluster

```powershell
python scripts/spark_cluster.py start --workers 8
python scripts/spark_cluster.py status
python scripts/spark_cluster.py stop
```

One worker = one standalone worker process = one JVM = one core; the driver
and the master are separate processes. `environment.json` records the PIDs
and the topology actually used.

## Campaign

```powershell
python scripts/run_benchmark4.py --phase all --soak-s 300
python scripts/collect_env.py
```

Phases: `strong` (n ∈ {2000,4000} × p ∈ {1,2,4,8}), `weak`
((p,n) = (1,2000),(2,2500),(4,3200),(8,4000)), `size` (n ∈
{500,1000,2000,4000} at p=4), `baselines` (numpy BLAS, C sequential, C
OpenMP at 1/2/4/8 threads), or `all`. The script sets the High-performance
power plan, heat-soaks the machine, restarts the cluster only between
groups of `p`, and restores Balanced afterwards.

Timing boundary: `t_e2e = t_stage_a + t_action_job + t_assemble` per
repetition; `t_startup` and `t_replicate` are one-time per application and
are reported beside it. Protocol per config: 1 warm-up + 3 measured
repetitions (5 for baselines), median as the summary statistic.

Outputs: `data/results/configs4.csv` (per-config status),
`raw_timings4.csv` (168 measurements), `raw_detail/<tag>.jsonl` (JSON lines
per run incl. in-worker `matmul_ms`), and `environment.json`.

## Analysis

```powershell
jupyter nbconvert --execute --to notebook --inplace notebooks/04_analysis.ipynb
```

Writes `data/results/summary_{strong,weak,size,compare}.csv` and the figures
in `reports/figures/`. The report is
`reports/Assignment04_DelgadoValera_Report.tex` (compile with `latexmk -pdf`).

## Previous assignments

Tags `assignment-1`, `assignment-2` and `assignment-3` keep the earlier
reports, harnesses and measurements (Assignment 3: OpenMP + SIMD scaling).
