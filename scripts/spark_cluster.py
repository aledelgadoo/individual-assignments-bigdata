#!/usr/bin/env python3
"""Start/stop the local Spark standalone cluster used by Assignment 4.

Topology (the one reported in the assignment): **one physical machine**, with

  * 1 master process  (RPC 127.0.0.1:7077, web UI :8080)
  * N worker processes (1 JVM each, ``--cores 1`` so 1 worker = 1 core)
  * 1 driver process  (the Python program that submits the application)

Everything binds the loopback interface, so "workers" are *processes on the
same machine* -- never separate physical nodes.  That is exactly the local
execution the course allows, and it is labelled as such in the report.

Usage:
    python scripts/spark_cluster.py start --workers 4
    python scripts/spark_cluster.py status
    python scripts/spark_cluster.py stop

State (PIDs + ports) lives in ``build/spark/``; logs in ``build/spark/logs/``.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "build" / "spark"
LOG_DIR = STATE_DIR / "logs"
STATE_FILE = STATE_DIR / "cluster.json"

MASTER_HOST = "127.0.0.1"
MASTER_RPC_PORT = 7077
MASTER_WEBUI_PORT = 8080
WORKER_RPC_PORT_BASE = 7100
WORKER_WEBUI_PORT_BASE = 8100
WORKER_MEMORY = "1g"
STARTUP_TIMEOUT_S = 90


def spark_home() -> Path:
    import pyspark

    return Path(pyspark.__file__).resolve().parent


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["SPARK_HOME"] = str(spark_home())
    env["SPARK_LOCAL_IP"] = MASTER_HOST
    env["JAVA_HOME"] = env.get("JAVA_HOME", r"C:\Users\Aleja\.jdks\ms-21.0.10")
    env["PYSPARK_PYTHON"] = sys.executable
    env["PYSPARK_DRIVER_PYTHON"] = sys.executable
    env["SPARK_NO_DAEMON"] = "1"
    # One worker = one core.  numpy links scipy-openblas with MAX_THREADS=24,
    # so the Python worker (a child of this JVM) must be told to use a single
    # thread or every executor would consume the whole machine.  The same
    # values are also passed as spark.executorEnv.* in sparkconf.py.
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    # Executors inherit the Worker's environment; without HADOOP_HOME any
    # executor-side Hadoop FileSystem call fails on Windows.
    hadoop_home = ROOT / "build" / "hadoop"
    if (hadoop_home / "bin" / "winutils.exe").exists():
        env["HADOOP_HOME"] = str(hadoop_home)
        env["PATH"] = str(hadoop_home / "bin") + os.pathsep + env.get("PATH", "")
    return env


def _spark_class(main_class: str, args: list[str]) -> subprocess.Popen:
    """Launch a JVM entry point through the bundled ``spark-class.cmd``."""
    cmd = ["cmd", "/c", str(spark_home() / "bin" / "spark-class.cmd"), main_class, *args]
    return subprocess.Popen(cmd, env=_env(), cwd=str(ROOT),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))


def _wait_port(host: str, port: int, timeout: float) -> bool:
    import socket

    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            try:
                if s.connect_ex((host, port)) == 0:
                    return True
            except OSError:
                pass
        time.sleep(0.5)
    return False


def _master_json(timeout: float = STARTUP_TIMEOUT_S) -> dict | None:
    url = f"http://{MASTER_HOST}:{MASTER_WEBUI_PORT}/json/"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            time.sleep(0.5)
    return None


def start(workers: int) -> int:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if STATE_FILE.exists():
        prev = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        if all(_alive(p) for p in prev.get("pids", [])):
            print(f"cluster already running ({len(prev['pids'])} pids); "
                  f"use 'status' or 'stop' first.")
            return 0
        stop()

    print(f"starting master on {MASTER_HOST}:{MASTER_RPC_PORT} ...")
    master_log = open(LOG_DIR / "master.log", "wb")
    master = _spark_class("org.apache.spark.deploy.master.Master", [
        "--host", MASTER_HOST,
        "--port", str(MASTER_RPC_PORT),
        "--webui-port", str(MASTER_WEBUI_PORT),
    ])
    if not _wait_port(MASTER_HOST, MASTER_RPC_PORT, STARTUP_TIMEOUT_S):
        print("ERROR: master RPC port never opened", file=sys.stderr)
        master.terminate()
        return 1

    pids = {"master": master.pid}
    print(f"  master pid {master.pid}")

    for i in range(workers):
        rpc = WORKER_RPC_PORT_BASE + i
        webui = WORKER_WEBUI_PORT_BASE + i
        log = open(LOG_DIR / f"worker{i}.log", "wb")
        worker = _spark_class("org.apache.spark.deploy.worker.Worker", [
            f"spark://{MASTER_HOST}:{MASTER_RPC_PORT}",
            "--host", MASTER_HOST,
            "--port", str(rpc),
            "--cores", "1",
            "--memory", WORKER_MEMORY,
            "--webui-port", str(webui),
        ])
        pids[f"worker{i}"] = worker.pid
        print(f"  worker{i} pid {worker.pid} (1 core, rpc {rpc})")
        if not _wait_port(MASTER_HOST, rpc, STARTUP_TIMEOUT_S):
            print(f"ERROR: worker{i} RPC port {rpc} never opened", file=sys.stderr)
            stop()
            return 1

    # The Popen pids are cmd.exe wrappers that exit once the JVM is up, so
    # resolve the real java pids (these are what 'stop' must kill).
    java_pids = {"master": _find_java_pid("deploy.master.Master",
                                          f"--port {MASTER_RPC_PORT}")}
    for i in range(workers):
        rpc = WORKER_RPC_PORT_BASE + i
        java_pids[f"worker{i}"] = _find_java_pid("deploy.worker.Worker",
                                                 f"--port {rpc}")
    missing = [k for k, v in java_pids.items() if v is None]
    if missing:
        print(f"ERROR: could not resolve JVM pid for {missing}", file=sys.stderr)
        stop()
        return 1
    print("  jvm pids: " + ", ".join(f"{k}={v}" for k, v in java_pids.items()))

    # Workers open their RPC port before they finish registering with the
    # master, so poll the master's view instead of sampling it once.
    registered = -1
    deadline = time.time() + STARTUP_TIMEOUT_S
    while time.time() < deadline:
        info = _master_json(timeout=5.0)
        registered = len(info.get("workers", [])) if info else -1
        if registered == workers:
            break
        time.sleep(0.5)
    if registered != workers:
        print(f"ERROR: master reports {registered} workers, expected {workers}",
              file=sys.stderr)
        stop()
        return 1

    STATE_FILE.write_text(json.dumps({
        "master_url": f"spark://{MASTER_HOST}:{MASTER_RPC_PORT}",
        "workers": workers,
        "worker_cores": 1,
        "worker_memory": WORKER_MEMORY,
        # pids/pid_map hold the *JVM* pids (kill these); launcher_pids holds
        # the cmd.exe wrappers, which are already gone by now.
        "pids": [v for v in java_pids.values()],
        "pid_map": java_pids,
        "launcher_pids": pids,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2), encoding="utf-8")

    print(f"cluster ready: 1 master + {workers} worker processes "
          f"(all 1 core, loopback).")
    print(f"master_url = spark://{MASTER_HOST}:{MASTER_RPC_PORT}")
    return 0


def _alive(pid: int) -> bool:
    """True if ``pid`` exists.

    ``os.kill(pid, 0)`` is unusable here: on this Windows build it raises
    ``OSError(87)`` / ``SystemError`` for every pid, alive or not, so process
    liveness goes through psutil (already a declared dependency).
    """
    import psutil

    try:
        return psutil.pid_exists(pid) and psutil.Process(pid).is_running()
    except Exception:  # noqa: BLE001
        return False


def _find_java_pid(*markers: str) -> int | None:
    """PID of the ``java.exe`` whose command line contains every marker.

    ``spark-class.cmd`` is a ``cmd.exe`` chain: the pid returned by
    ``Popen`` is an outer wrapper that exits once the JVM is up, so the
    *launcher* pid recorded in the state file goes stale immediately.  The
    JVM that actually holds the port is found by its command line instead.
    """
    import psutil

    want = [m.lower() for m in markers]
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = (proc.info.get("name") or "").lower()
            if not name.startswith("java"):
                continue
            cmdline = " ".join(proc.info.get("cmdline") or []).lower()
            if all(m in cmdline for m in want):
                return int(proc.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return None


def status() -> int:
    if not STATE_FILE.exists():
        print("no cluster state file; cluster is not running.")
        return 1
    state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    pid_map = state.get("pid_map", {})
    # Older state files stored cmd.exe wrapper pids, which are always dead
    # while the JVM keeps running: fall back to finding the JVMs directly.
    alive = {k: _alive(p) for k, p in pid_map.items()}
    if not any(alive.values()):
        resolved = {"master": _find_java_pid("deploy.master.Master",
                                             f"--port {MASTER_RPC_PORT}")}
        for i in range(int(state.get("workers", 0))):
            resolved[f"worker{i}"] = _find_java_pid(
                "deploy.worker.Worker",
                f"--port {WORKER_RPC_PORT_BASE + i}")
        if any(v is not None for v in resolved.values()):
            pid_map = {k: v for k, v in resolved.items() if v is not None}
            alive = {k: _alive(p) for k, p in pid_map.items()}
    info = _master_json(timeout=5.0)
    registered = len(info.get("workers", [])) if info else -1
    print(json.dumps({
        "master_url": state["master_url"],
        "expected_workers": state["workers"],
        "registered_workers": registered,
        "jvm_pids": pid_map,
        "alive": alive,
    }, indent=2))
    return 0 if all(alive.values()) and registered == state["workers"] else 1


def stop() -> int:
    if not STATE_FILE.exists():
        print("no cluster state file; nothing to stop.")
        return 0
    state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    pid_map = state.get("pid_map", {})
    # Resolve by command line as well, so a stale state file still stops the
    # running JVMs instead of silently leaving them behind.
    resolved = dict(pid_map)
    if "master" in state.get("pid_map", {}):
        found = _find_java_pid("deploy.master.Master",
                               f"--port {MASTER_RPC_PORT}")
        if found:
            resolved["master"] = found
    for i in range(int(state.get("workers", 0))):
        found = _find_java_pid("deploy.worker.Worker",
                               f"--port {WORKER_RPC_PORT_BASE + i}")
        if found:
            resolved[f"worker{i}"] = found

    for name, pid in resolved.items():
        if not _alive(pid):
            continue
        try:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"  killed {name} pid {pid}")
        except OSError as exc:
            print(f"  could not kill {name} pid {pid}: {exc}", file=sys.stderr)
    STATE_FILE.unlink(missing_ok=True)
    print("cluster stopped.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_start = sub.add_parser("start", help="start master + workers")
    p_start.add_argument("--workers", type=int, required=True)
    sub.add_parser("status", help="report cluster health")
    sub.add_parser("stop", help="stop master + workers")
    args = ap.parse_args()

    if args.cmd == "start":
        return start(args.workers)
    if args.cmd == "status":
        return status()
    return stop()


if __name__ == "__main__":
    raise SystemExit(main())
