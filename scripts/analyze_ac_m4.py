#!/usr/bin/env python3
"""Spec AC.7 4, last step: read M4 stage 1 (results/m4/stage1/{BRAIN,RND,MAX}/result.json, the situation records and
the brain logs) and write results/summary/ac_m4_stage1.json with AC.8's sentences and AC.5's label. No PASS / FAIL.

    uv run python scripts/analyze_ac_m4.py"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from flymon.ac import analysis, smoke, stage1, store
from flymon.ac.spec import SPEC
from flymon.rescope import blocks

M1_PILOT = "results/summary/m1_pilot.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=SPEC.out_root, help="stage-1 output root (default: the spec's)")
    ap.add_argument("--out", default=analysis.RESULT)
    ap.add_argument("--m1", default=M1_PILOT, help="the M1 pilot summary (WEAK arms, copied)")
    a = ap.parse_args(argv)
    root = Path(a.root)
    results = {}
    for arm in ("BRAIN", "RND", "MAX"):
        p = root / arm / "result.json"
        if not p.exists():
            raise SystemExit(f"refusing: {p} does not exist (every stage-1 arm must be complete)")
        results[arm] = json.loads(p.read_text())
        if not results[arm].get("complete"):
            raise SystemExit(f"refusing: {p} is not complete")
    analysis.check_digests(results)
    sits = stage1.SituationLog(root / "BRAIN" / "logs").records()
    eval_paths = sorted((root / "BRAIN" / "logs" / "eval").glob("fly*.jsonl"))
    eval_recs = [r for p in eval_paths for r in blocks.read_jsonl(p)]
    all_recs = [r for p in smoke.log_paths(root / "BRAIN") for r in blocks.read_jsonl(p)]
    m1 = json.loads(Path(a.m1).read_text())["arms"]
    weak = {k: m1[k]["win_rate"] for k in ("WEAK-RND", "WEAK-MAX")}
    doc = analysis.analyse(results, sits, eval_recs, all_recs, SPEC, weak=weak)
    doc["sources"] = {arm: dict(path=str(root / arm / "result.json"), sha256=store.sha256_file(root / arm / "result.json"))
                      for arm in results}
    store.write_json(a.out, doc)
    for s in doc["sentences"] + [doc["c1_summary"]]:
        print(s, flush=True)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
