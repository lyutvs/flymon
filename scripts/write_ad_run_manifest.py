#!/usr/bin/env python3
"""Spec AD.6 4, last step: the run manifest, from results/summary/ad_budget.json (status OK: T and the new-set flag)
and the AD input manifest (status OK), at the current commit; the code under flymon/ and scripts/run_ad.py must be
committed. --smoke writes the smoke's manifest (T = 2, no budget) under results/m4-smoke/ad/.

    uv run python scripts/write_ad_run_manifest.py
    uv run python scripts/write_ad_run_manifest.py --smoke"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from flymon.ad import inputs, run_manifest as rm, store
from flymon.ad.budget import BUDGET
from flymon.ad.spec import SPEC, smoke


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true", help="smoke only: skip the uncommitted-code refusal")
    a = ap.parse_args(argv)
    spec = smoke() if a.smoke else SPEC
    _, man_p = inputs.input_paths(spec)
    if not man_p.exists() or json.loads(man_p.read_text()).get("status") != "OK":
        raise SystemExit(f"refusing: {man_p} missing or not OK (scripts/write_ad_inputs.py"
                         f"{' --smoke' if a.smoke else ''}; STOP_REUSE stops AD)")
    if a.smoke:
        T, new_set, bp = spec.learn_battles, True, None
    else:
        bp = Path(BUDGET)
        if not bp.exists():
            raise SystemExit(f"refusing: {bp} does not exist (scripts/write_ad_budget.py)")
        b = json.loads(bp.read_text())
        if b.get("status") != "OK":
            raise SystemExit(f"refusing: {bp} status {b.get('status')!r} (STOP_BUDGET / STOP_BENCH stop AD)")
        T, new_set = int(b["T"]), bool(b["new_set"])
    commit = store.head_commit()
    dirty = store.code_changes_since(commit, rm.CODE_PATHS)
    if dirty and not (a.smoke and a.allow_dirty):
        raise SystemExit(f"refusing: uncommitted code {dirty} (the run manifest pins the code commit)")
    m = rm.build(spec, T=T, new_set=new_set, budget_path=bp, inputs_manifest_path=man_p, commit=commit)
    out = rm.manifest_path(spec)
    store.write_json(out, m)
    print(f"run manifest: T {T}, points {m['points']}, new set {new_set} at {m['new_points']}, commit {commit[:12]}; "
          f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
