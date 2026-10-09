#!/usr/bin/env python3
"""Spec AD.6 4: the re-bench median and AD's extrapolation. Reads every attempt results/m4-bench/ad-<n>/bench.json (an
attempt directory without bench.json counts as an invalid attempt), judges each attempt valid from the load log
(AD.4: no outside process above 50% CPU in 3 consecutive samples inside its window), takes the median of the first 3
valid attempts and chooses T (120 with / without the new set, 80, or STOP_BUDGET).

    uv run python scripts/write_ad_budget.py --load-log .superpowers/sdd/2026-10-10-ad-stage2/runs/ad_bench_load.log

Writes results/summary/ad_budget.json. Exit 0 OK, 2 STOP_BUDGET or STOP_BENCH (6 attempts without 3 valid), 3 fewer
than 3 valid attempts so far (run the next attempt)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from flymon.ad import budget, load, store
from flymon.ad.spec import LABEL, SPEC


def attempts(root: Path, samples) -> list:
    out = []
    dirs = sorted((d for d in root.glob("ad-*") if d.is_dir() and d.name[3:].isdigit()), key=lambda d: int(d.name[3:]))
    for d in dirs:
        n, p = int(d.name[3:]), d / "bench.json"
        if not p.exists():
            out.append(dict(attempt=n, valid=False, reason="no bench.json", s_per_batch_battle=None,
                            sit_eval_s_per_eval=None, offenders=[]))
            continue
        b = json.loads(p.read_text())
        v = load.bench_validity(samples, b["t_started"], b["t_ended"], b["pid"], SPEC.busy_samples)
        out.append(dict(attempt=n, valid=v["valid"], reason=v["reason"], offenders=v["offenders"],
                        n_samples=v["n_samples"], s_per_batch_battle=float(b["s_per_batch_battle"]),
                        sit_eval_s_per_eval=float(b["sit_eval_s_per_eval"]), bench_sha256=store.sha256_file(p)))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="results/m4-bench")
    ap.add_argument("--load-log", required=True)
    a = ap.parse_args(argv)
    samples = load.read(a.load_log)
    att = attempts(Path(a.root), samples)
    summ = budget.bench_summary(att, SPEC)
    info = dict(bench=summ, attempts=att, load_log=a.load_log,
                load_log_sha256=store.sha256_file(a.load_log) if Path(a.load_log).exists() else None, label=LABEL)
    if summ["status"] != "OK":
        store.write_json(budget.BUDGET, dict(info, status=summ["status"], T=None, new_set=None))
        print(f"{summ['status']}: {len(summ['used'])} valid of {len(att)} attempts; wrote {budget.BUDGET}", flush=True)
        return 3 if summ["status"] == "MORE" else 2
    c = budget.choose(SPEC, summ["s_per_batch_battle"], summ["sit_eval_s"])
    store.write_json(budget.BUDGET, dict(c, **info))
    cases = {k: round(v, 2) for k, v in c["cases"].items()}
    print(f"{c['status']}: T {c['T']}, new set {c['new_set']}; cases {cases} h (limit {c['limit_h']:.3f} h); "
          f"bench medians {summ['s_per_batch_battle']:.1f} s/batch-battle, {summ['sit_eval_s']:.1f} s/evaluation from "
          f"attempts {summ['used']}; wrote {budget.BUDGET}", flush=True)
    return 0 if c["status"] == "OK" else 2


if __name__ == "__main__":
    sys.exit(main())
