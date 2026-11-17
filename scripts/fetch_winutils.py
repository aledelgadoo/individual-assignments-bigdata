#!/usr/bin/env python3
"""Fetch the Windows Hadoop helpers Spark needs (winutils.exe, hadoop.dll).

Spark's event log is written through the Hadoop ``LocalFileSystem``, and on
Windows that path calls ``Shell.getHadoopHomeDir()``, which fails with

    HADOOP_HOME and hadoop.home.dir are unset

unless ``%HADOOP_HOME%\\bin\\winutils.exe`` exists.  Without it the event log
cannot be enabled, so per-task metrics would have to be scraped from the
ephemeral Spark UI instead.

The binaries are third-party artefacts, so they are downloaded into
``build/hadoop/`` (git-ignored) rather than committed.  Versions are pinned:

    winutils.exe / hadoop.dll  <- cdarlint/winutils, hadoop-3.3.6

Usage:
    python scripts/fetch_winutils.py           # download if missing
    python scripts/fetch_winutils.py --print   # print HADOOP_HOME to export

Then either export HADOOP_HOME, or run anything that needs it through
``scripts/with_hadoop.py`` which sets it for the child process.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HADOOP_HOME = ROOT / "build" / "hadoop"
BASE_URL = ("https://raw.githubusercontent.com/cdarlint/winutils/"
            "master/hadoop-3.3.6/bin")
FILES = ("winutils.exe", "hadoop.dll")


def ensure(verbose: bool = True) -> Path:
    bindir = HADOOP_HOME / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = bindir / name
        if dest.exists() and dest.stat().st_size > 0:
            if verbose:
                print(f"  present {dest} ({dest.stat().st_size} bytes)")
            continue
        url = f"{BASE_URL}/{name}"
        if verbose:
            print(f"  fetching {url} ...")
        req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
        dest.write_bytes(data)
        if verbose:
            print(f"  wrote {dest} ({len(data)} bytes)")
    return HADOOP_HOME


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--print", action="store_true",
                    help="print the HADOOP_HOME path (for export)")
    args = ap.parse_args()

    home = ensure(verbose=not args.print)
    if args.print:
        print(home)
        return 0

    exe = home / "bin" / "winutils.exe"
    if not exe.exists():
        print("ERROR: winutils.exe missing after download", file=sys.stderr)
        return 1
    print(f"HADOOP_HOME={home}")
    print("Set it before running Spark with event logging, e.g.:\n"
          f"  $env:HADOOP_HOME='{home}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
