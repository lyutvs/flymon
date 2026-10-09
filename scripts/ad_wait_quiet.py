#!/usr/bin/env python3
"""AD.4's start condition: wait until no process (other than this one) has been above 50% CPU for 3 consecutive
one-minute samples; every sample is appended to --log (the uptime log). Exit 0 when quiet, 3 after --max-minutes.

    uv run python scripts/ad_wait_quiet.py --log .superpowers/sdd/2026-10-10-ad-stage2/runs/ad_bench_load.log"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from flymon.ad import load
from flymon.ad.spec import SPEC


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True)
    ap.add_argument("--samples", type=int, default=SPEC.busy_samples)
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--max-minutes", type=float, default=240.0)
    a = ap.parse_args(argv)
    log = Path(a.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    me, seen, t0 = {os.getpid()}, [], time.time()
    while time.time() - t0 <= a.max_minutes * 60:
        x = dict(load.take(SPEC.busy_cpu), kind="wait")
        with log.open("a") as f:
            f.write(json.dumps(x) + "\n")
        seen.append(x)
        if load.quiet(seen, me, a.samples):
            print(f"quiet: {a.samples} samples without a process above {SPEC.busy_cpu}% CPU; load {x['load']}", flush=True)
            return 0
        time.sleep(a.interval)
    print(f"not quiet after {a.max_minutes} min", flush=True)
    return 3


if __name__ == "__main__":
    sys.exit(main())
