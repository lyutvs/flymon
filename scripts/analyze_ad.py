#!/usr/bin/env python3
"""Spec AD.6 5, last step: AD's reading. Refuses unless the raw-data list verifies (AD.6 5) and the stage-1 reuse fields
are unchanged (STOP_REUSE). Reads results/m4/ad/ (result.json, situation records, learning / evaluation logs,
battles.jsonl), the stage-1 C-off and FLY records, stage 1's RND / MAX results (same evaluation digest), the run load
log; writes results/summary/ad_m4.json with AD.3's sentences and AC.5's label. No PASS / FAIL.

    uv run python scripts/analyze_ad.py --load-log .superpowers/sdd/2026-10-10-ad-stage2/runs/ad_run_load.log"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from flymon.ac.analysis import kc_ratio_by_arm
from flymon.ad import hooks, load, raw, report, reuse, run_manifest as rm, store
from flymon.ad.spec import SPEC
from flymon.rescope import blocks


def own_pid(path) -> int:
    for line in Path(path).read_text().splitlines():
        rec = json.loads(line)
        if rec.get("kind") == "start":
            return int(rec.get("own_pid") or 0)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=SPEC.out_root)
    ap.add_argument("--load-log", required=True)
    ap.add_argument("--out", default=report.RESULT)
    a = ap.parse_args(argv)
    root = Path(a.root)
    rdoc = json.loads(Path(raw.RAW).read_text()) if Path(raw.RAW).exists() else None
    if rdoc is None:
        raise SystemExit(f"refusing: {raw.RAW} does not exist (scripts/write_ad_raw_manifest.py first, AD.6 5)")
    bad = raw.verify(rdoc)
    if bad:
        raise SystemExit(f"refusing: raw files changed since {raw.RAW}: {bad[:5]}")
    man = json.loads(Path(rm.RUN_MANIFEST).read_text())
    T = int(man["T"])
    spec = dataclasses.replace(SPEC, learn_battles=T)
    result = json.loads((root / "result.json").read_text())
    if result.get("status") not in ("COMPLETE", "STOP_BUDGET"):
        raise SystemExit(f"refusing: {root / 'result.json'} status {result.get('status')!r}")
    doc = json.loads(Path("results/summary/ad_inputs.json").read_text())
    iman = json.loads(Path("results/summary/ad_inputs_manifest.json").read_text())
    now = reuse.reuse_fields(doc["digests"]["confirm"], doc["digests"]["eval"], spec)
    changed = reuse.changed_since(iman.get("reuse"), now)
    if changed:
        raise SystemExit(f"STOP_REUSE: stage-1 fields changed since the input manifest: {changed}")
    p = reuse.stage1_paths(spec)
    s1 = json.loads(p["result"].read_text())
    fly1_valid = [r["fly"] for r in s1["per_fly"] if r["arm"] == "FLY" and not r["invalid"]]
    coff = reuse.coff_records(p["init"], spec.coff_flies)
    fly1 = reuse.stage1_fly_records(p["points"], fly1_valid, spec.stage1_point)
    nob = []
    for arm in ("RND", "MAX"):
        r = json.loads((Path(spec.stage1_root) / arm / "result.json").read_text())
        if r.get("eval_digest") != result["eval_digest"]:
            raise SystemExit(f"refusing: stage-1 {arm} ran another evaluation schedule (AC.4 digest gate)")
        nob += [row for row in r["per_fly"]]
    sits = hooks.ADSituationLog(root / "logs").records()
    learn_logs = {int(q.stem[3:]): blocks.read_jsonl(q) for q in sorted((root / "logs").glob("fly*.jsonl"))}
    eval_recs = [x for q in sorted((root / "logs" / "eval").glob("fly*.jsonl")) for x in blocks.read_jsonl(q)]
    all_recs = [x for v in learn_logs.values() for x in v] + eval_recs
    lay = spec.layout()
    rows = [dict(fly=i, arm=arm, invalid=False) for i, (arm, _, _) in enumerate(lay)]
    from flymon.brain.h4_pairs import pool_vocabulary
    _, mi, _, _ = pool_vocabulary()
    sess = json.loads((root / "wall_clock.json").read_text())["sessions"]
    import datetime as dt

    def ts(s):
        return dt.datetime.fromisoformat(s).timestamp()
    start, end = ts(sess[0]["started_utc"]), max(ts(x["ended_utc"]) for x in sess if x.get("ended_utc"))
    lsum = load.summary(load.read(a.load_log), start, end, own_pid(a.load_log), SPEC.busy_samples)
    out = report.analyse(spec=spec, T=T, result=result, sits=sits, pairs=doc["confirm"]["pairs"], coff=coff, fly1=fly1,
                         battles_L=blocks.read_jsonl(root / "logs" / "battles.jsonl"), eval_records=eval_recs,
                         learn_logs=learn_logs, nobrain_rows=nob, move_info=mi,
                         kc_by_arm=kc_ratio_by_arm(all_recs, rows), load_summary=lsum)
    out["sources"] = dict(result=dict(path=str(root / "result.json"), sha256=store.sha256_file(root / "result.json")),
                          raw_manifest=dict(path=raw.RAW, sha256=store.sha256_file(raw.RAW)),
                          run_manifest=dict(path=rm.RUN_MANIFEST, sha256=store.sha256_file(rm.RUN_MANIFEST)))
    store.write_json(a.out, out)
    for s in out["sentences"] + [out["summary"]] + out["futility_and_learncheck_sentences"]:
        print(s, flush=True)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
