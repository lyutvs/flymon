#!/usr/bin/env python3
"""Spec AC.7 3, last step: stage 1's extrapolation (AC.6) from the representative benchmark.

    uv run python scripts/write_ac_budget.py [--bench results/m4-bench/BRAIN/bench.json]

Writes results/summary/ac_budget.json; exit 0 (OK) or 2 (STOP_BUDGET: above 48 h; stage 1 does not run)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from flymon.ac import store
from flymon.ac.budget import BENCH, BUDGET, extrapolate
from flymon.ac.spec import LABEL


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", default=BENCH)
    a = ap.parse_args(argv)
    p = Path(a.bench)
    if not p.exists():
        raise SystemExit(f"refusing: {p} does not exist (run scripts/run_ac_m4.py --bench --arm BRAIN first)")
    bench = json.loads(p.read_text())
    doc = dict(extrapolate(bench), bench=bench, bench_path=str(p), bench_sha256=store.sha256_file(p), label=LABEL)
    store.write_json(BUDGET, doc)
    print(f"{doc['status']}: stage 1 extrapolated {doc['hours']:.1f} h (limit {doc['limit_h']} h); wrote {BUDGET}")
    return 0 if doc["status"] == "OK" else 2


if __name__ == "__main__":
    sys.exit(main())
