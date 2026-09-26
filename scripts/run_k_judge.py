#!/usr/bin/env python3
"""Spec K.8.4, stage 2: judge the top setting of a complete stage 1 whose outcome is scan_go — H.4 readout
reselection, z, the oracle on the 21 even-turn (b) and 18 (a) pairs, J.12.9's bands, D.6 on the judged engine.

    uv run python scripts/run_k_judge.py                       # ~1.5 h
    uv run python scripts/run_k_judge.py --smoke --allow-dirty --scan <a smoke scan report>

Run it from the repository root. --scan is results/summary/k_engine.json (its block "scan") or a scan report. Refused
before measuring: a scan under another K key, manifest (HASHED_FILES) or configuration; a smoke or dirty scan outside
--smoke; a scan whose outcome is not scan_go (K.8.4: no judgement); a chosen engine whose threshold file is missing or
has another sha256. Report: <out>/runs/<run id>-judge.{json,md}; block "judge" of the summary is replaced only by a
complete judgement (every pair, D.6 measured, clean, declared configuration, not aborted). Exit codes: 0, 2, 3.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import uuid
from pathlib import Path

from flymon.brain import d6a, k_store
from flymon.brain.config import Params
from flymon.brain.fly_pool import FlyPool
from flymon.brain.h3_measure import PoolMeasurer
from flymon.brain.h3_store import ROOT, MeasureCache, canonical, code_key, git_state, sha256_file
from flymon.brain.h4_measure import H4Measurer
from flymon.brain.h4_pairs import pair_key
from flymon.brain.j_measure import JMeasurer
from flymon.brain.j_rules import COMPUTE_ABORTED, SELECTED
from flymon.brain.j_setup import build
from flymon.brain.k_measure import HASHED_FILES, MEASURE_FILES
from flymon.brain.k_rules import SCAN_GO
from flymon.brain.k_runner import stage2
from flymon.brain.k_spec import SPEC, KSpec, smoke

POOL_TIMEOUT_S = 1800


def refuse(why: str) -> int:
    print(f"refused: {why}", file=sys.stderr)
    return 2


def main(argv=None, spec: KSpec | None = None, summary_spec: KSpec = SPEC, require_root: bool = True,
         pools: dict | None = None, pairs: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default="data/malecns.npz")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=None)
    ap.add_argument("--scan", default="results/summary/k_engine.json")
    ap.add_argument("--summary", default="results/summary/k_engine.json")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--pairs", type=int, default=None)       # verification only
    ap.add_argument("--no-d6", action="store_true")          # verification only
    ap.add_argument("--max-hours", type=float, default=None)
    a = ap.parse_args(argv)
    if a.pairs is not None and a.pairs < 1:
        return refuse(f"--pairs must be at least 1, got {a.pairs}")
    if require_root and Path.cwd().resolve() != ROOT:
        return refuse(f"not at the repository root {ROOT}")
    spec = spec or (smoke(SPEC) if a.smoke else SPEC)
    out = Path(a.out or ("results/m0d/k/smoke" if a.smoke else "results/m0d/k/run"))
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    git = git_state(files=HASHED_FILES)
    if git["dirty_hashed"] and not a.allow_dirty:
        return refuse(f"hashed files are dirty {git['dirty_hashed']} (commit them, or --allow-dirty)")
    code, manifest = code_key(a.npz, files=MEASURE_FILES), code_key(a.npz, files=HASHED_FILES)
    try:
        doc = json.loads(Path(a.scan).read_text())
    except (OSError, ValueError) as e:
        return refuse(f"no usable scan at {a.scan}: {e}")
    scan = doc["scan"] if "scan" in doc and "stage1" not in doc else doc
    if scan.get("measure_key") != code["key"]:
        return refuse(f"the scan was measured under K key {str(scan.get('measure_key'))[:12]}, this code's is "
                      f"{code['key'][:12]}: its engines are not this code's")
    if not a.smoke and scan.get("code", {}).get("key") != manifest["key"]:
        return refuse(f"the scan was run under another procedure code {str(scan.get('code', {}).get('key'))[:12]}, "
                      f"this code's is {manifest['key'][:12]}")
    if not a.smoke and (scan.get("smoke") or scan.get("git", {}).get("dirty_hashed")):
        why = "a smoke run" if scan.get("smoke") else f"measured with dirty hashed files {scan['git']['dirty_hashed']}"
        return refuse(f"the scan is {why}: only a --smoke judgement may use it")
    if not a.smoke and canonical(scan.get("spec")) != canonical(spec):
        return refuse("the scan was run under another K configuration (its spec differs from this judgement's)")
    st1 = scan.get("stage1", {})
    if st1.get("outcome") != SCAN_GO:
        return refuse(f"the scan's outcome is {st1.get('outcome')}: K.8.4 judges only after {SCAN_GO}")
    setting = st1["settings"][st1["order"][0]]
    p = setting["reconverge"]["adopted"]["params"]   # the JSON record; stage2 rebuilds Params from it
    tf = Path(p.get("kc_thresh_file") or "")
    if not p.get("kc_thresh_file") or not tf.is_file() or sha256_file(tf) != p.get("kc_thresh_sha256"):
        return refuse(f"the chosen engine's threshold file {tf} is missing or has another sha256")
    t0 = time.time()
    deadline = None if a.max_hours is None else t0 + 3600.0 * a.max_hours
    s = build(a.npz, spec.j, out, deadline=deadline, log=lambda x: print(x, flush=True), pools=pools, pairs=pairs)
    if pairs is None and s["pairs_digest"] != spec.j.h4.pairs_digest:
        return refuse(f"the pair list differs from the declared one ({s['pairs_digest'][:12]})")
    if a.pairs:
        s["pairs"] = [q for ax in ("a", "b") for q in [r for r in s["pairs"] if r["axis"] == ax][:a.pairs]]
        s["ctx"].expected = [pair_key(q) for q in s["pairs"]]
    res = dict(run_id=run_id, smoke=a.smoke, argv=list(sys.argv[1:] if argv is None else argv), git=git,
               code=manifest, measure_key=code["key"], spec=spec, scan_run_id=scan.get("run_id"),
               n_pairs=len(s["pairs"]), setting_g=setting["g"], even_pair_uses=list(spec.even_pair_uses) + ["K.8"])
    with FlyPool(a.npz, Params(), [{} for _ in range(a.workers)], workers=a.workers,
                 punish_type=spec.j.h4.h3.punish_type, reward_type=spec.j.h4.h3.reward_type,
                 timeout_s=POOL_TIMEOUT_S) as pool:
        cache = MeasureCache(out / "cache", code, run_id)
        m3 = PoolMeasurer(pool, spec.j.h4.h3, {"reference": s["odors"]}, s["all51"], cache, None)
        m4 = H4Measurer(pool, spec.j.h4, s["pairs"], s["ctx"].pools, cache, m3)
        jm = JMeasurer(pool, spec.j, s["all51"], cache)
        seen = [m3, m4, jm]
        res["stage2"] = stage2(m4, jm, s["ctx"], setting, with_d6=not a.no_d6,
                               d6_seeds=d6a.SEEDS[:2] if a.smoke else d6a.SEEDS)
    guard_params = [Params()] + list(dict.fromkeys(q for m in seen for q in m.params_seen if q != Params()))
    res["wall_s"] = time.time() - t0
    res["cache"] = dict(hits=cache.hits, misses=cache.misses)
    res["artifacts"] = {q: sha256_file(q) for q in sorted(cache.used) if Path(q).exists()}
    st2 = res["stage2"]
    report = k_store.write_json(out / "runs" / f"{run_id}-judge.json", res, guard_params)
    k_store.write_bytes(out / "runs" / f"{run_id}-judge.md",
                        (f"# K.8.4 judgement {run_id}\n\n**state: {st2['state']}** ({st2['name']})\n\n"
                         f"reading: {(st2.get('judge') or {}).get('reading')}\n\nD.6: {st2.get('d6')}\n").encode(),
                        guard_params)
    print(f"wrote {report}: {st2['state']} in {res['wall_s'] / 60:.1f} min", flush=True)
    blockers = dict(smoke=a.smoke, pairs=a.pairs is not None, no_d6=a.no_d6, dirty=bool(git["dirty_hashed"]),
                    spec=spec != summary_spec, aborted=st2["state"] == COMPUTE_ABORTED)
    if any(blockers.values()):
        print(f"summary not written: {[k for k, v in blockers.items() if v]}", flush=True)
    else:
        block = dict(res, report=str(report), report_sha256=sha256_file(report))
        if st2["state"] == SELECTED:
            rd = st2["judge"]["reading"]
            block["selection"] = dict(setting=st2["name"], params=st2["params"], T_b=rd["T_b"], F_a=rd["F_a"],
                                      testable_b=rd["testable_b"], n_b=rd["n_b"], confirmed=False)
        k_store.write_summary_block(a.summary, "judge", block, guard_params)
        print(f"wrote {a.summary} (block judge)", flush=True)
    return 3 if st2["state"] == COMPUTE_ABORTED else 0


if __name__ == "__main__":
    sys.exit(main())
