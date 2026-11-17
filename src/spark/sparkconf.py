"""Shared Spark session/cluster configuration for Assignment 4.

Three Windows-specific fixes were found by smoke testing PySpark 4.3.0 on
this machine and are applied here so every entry point behaves the same:

1. ``spark.driver.host`` / ``spark.driver.bindAddress`` = 127.0.0.1
   The Windows hostname (``VIVOBOOK_ALED``) is not a valid Spark URL host for
   this build, so the driver must bind and advertise the loopback address.
2. ``PYSPARK_PYTHON`` / ``spark.pyspark.python`` = the real interpreter path
   Windows ships an App Execution Alias for ``python`` that prints the
   Microsoft Store banner and exits, which made every executor fail with
   "Python worker failed to connect back".
3. ``SPARK_LOCAL_IP`` = 127.0.0.1 so worker daemons bind loopback as well.

``winutils.exe`` is absent (HADOOP_HOME unset). Spark logs a WARN for it but
only Hadoop FileSystem calls are affected; this study reads and writes plain
local files through numpy, so it is recorded as a limitation rather than a
blocker.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Loopback fixes (also exported as environment variables for child processes).
LOCAL_HOST = "127.0.0.1"
REAL_PYTHON = sys.executable

os.environ.setdefault("SPARK_LOCAL_IP", LOCAL_HOST)
os.environ.setdefault("PYSPARK_PYTHON", REAL_PYTHON)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", REAL_PYTHON)

# HADOOP_HOME is required for event logging on Windows (the event log is
# written through Hadoop's LocalFileSystem).  Binaries are fetched into
# build/hadoop by scripts/fetch_winutils.py; if they are missing we leave the
# variable unset and callers fall back to the Spark REST UI for metrics.
_ROOT = Path(__file__).resolve().parents[2]
_HADOOP_HOME = _ROOT / "build" / "hadoop"
HADOOP_HOME_STR: str | None = None
if (_HADOOP_HOME / "bin" / "winutils.exe").exists():
    HADOOP_HOME_STR = str(_HADOOP_HOME)
    os.environ.setdefault("HADOOP_HOME", HADOOP_HOME_STR)
    os.environ.setdefault("PATH", str(_HADOOP_HOME / "bin") + os.pathsep +
                          os.environ.get("PATH", ""))


def spark_home() -> Path:
    """Return the bundled Spark home shipped inside the pyspark package."""
    import pyspark

    return Path(pyspark.__file__).resolve().parent


def base_configs() -> dict[str, str]:
    """Config keys every Assignment 4 session must set."""
    return {
        "spark.driver.host": LOCAL_HOST,
        "spark.driver.bindAddress": LOCAL_HOST,
        "spark.pyspark.python": REAL_PYTHON,
        "spark.executor.pyspark.python": REAL_PYTHON,
        "spark.ui.showConsoleProgress": "false",
        "spark.sql.shuffle.partitions": "8",
        # Deterministic local cluster behaviour.
        "spark.python.worker.reuse": "true",
        # One worker must mean one core: numpy links scipy-openblas with
        # MAX_THREADS=24, so without this a single executor would use the
        # whole machine and the scaling study would be meaningless.  Set on
        # the executor environment (propagated to the Python worker child).
        "spark.executorEnv.OPENBLAS_NUM_THREADS": "1",
        "spark.executorEnv.OMP_NUM_THREADS": "1",
        "spark.executorEnv.MKL_NUM_THREADS": "1",
        # Executors are launched by the Worker daemon, not by this process, so
        # they do not inherit HADOOP_HOME from the driver.  Without it any
        # executor-side Hadoop FileSystem call (file distribution, local file
        # access) dies with "HADOOP_HOME and hadoop.home.dir are unset".
        **({"spark.executorEnv.HADOOP_HOME": HADOOP_HOME_STR}
           if HADOOP_HOME_STR else {}),
    }


def session_configs(workers: int | None = None,
                    executor_memory: str = "1g",
                    event_log_dir: str | None = None) -> dict[str, str]:
    """Configs for a session; ``workers`` only used by ``local[N]`` fallback."""
    cfg = dict(base_configs())
    cfg["spark.executor.memory"] = executor_memory
    cfg["spark.executor.memoryOverhead"] = "512m"
    cfg["spark.driver.memory"] = "2g"
    if workers is not None:
        cfg["spark.executor.cores"] = "1"
    if event_log_dir:
        cfg["spark.eventLog.enabled"] = "true"
        cfg["spark.eventLog.dir"] = event_log_dir
        cfg["spark.eventLog.compress"] = "true"
    return cfg


def local_master(workers: int) -> str:
    """Single-JVM fallback master (``local[N]``); not the reported topology."""
    return f"local[{workers}]"


def standalone_master(host: str = "localhost", port: int = 7077) -> str:
    """URL of the standalone master started by ``scripts/spark_cluster.py``."""
    return f"spark://{host}:{port}"


def blas_thread_limits() -> dict[str, str]:
    """Env vars that pin every BLAS to a single thread.

    Must be applied **before numpy is imported** (OpenBLAS reads its
    environment when the library loads), so entry points call this at the top
    of ``main()`` and the cluster launcher puts the same values in the worker
    process environment.
    """
    return {
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }


def apply_blas_thread_limits() -> None:
    """Pin BLAS threads in this process (call before importing numpy)."""
    for key, value in blas_thread_limits().items():
        os.environ[key] = value
