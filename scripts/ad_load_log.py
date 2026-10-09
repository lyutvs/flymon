#!/usr/bin/env python3
"""AD.4: one load sample a minute (uptime + processes above 50% CPU with their ancestry) appended to a JSONL log, until
--until-pid exits (or forever). The first line records --own-pid: the process tree whose CPU is the measured work.

    uv run python scripts/ad_load_log.py --out .superpowers/sdd/2026-10-10-ad-stage2/runs/ad_bench_load.log --own-pid 0
    uv run python scripts/ad_load_log.py --out .../ad_run_load.log --own-pid $RUN_PID --until-pid $RUN_PID"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from flymon.ad import load
from flymon.ad.spec import SPEC


def alive(pid) -> bool:
    if not pid:
        return True
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--own-pid", type=int, default=0)
    ap.add_argument("--until-pid", type=int, default=0)
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--count", type=int, default=0, help="stop after N samples (0 = no limit)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        f.write(json.dumps(dict(kind="start", t=time.time(), own_pid=a.own_pid, until_pid=a.until_pid)) + "\n")
    n = 0
    while alive(a.until_pid):
        x = load.take(SPEC.busy_cpu)
        with out.open("a") as f:
            f.write(json.dumps(x) + "\n")
        n += 1
        if a.count and n >= a.count:
            break
        time.sleep(a.interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
