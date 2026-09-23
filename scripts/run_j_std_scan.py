#!/usr/bin/env python3
"""Spec J.11.3, stage 1: scan the engine's ORN->PN depression on C3 (calibration — no pairs, readouts or learning).

    uv run python scripts/run_j_std_scan.py                        # 9 settings x (8 all51 + 1 reference), ~2 h
    uv run python scripts/run_j_std_scan.py --smoke --allow-dirty  # minutes; results/m0d/j/std-smoke/, no summary

Run it from the repository root. The reference point is C3's adopted Params from block "h3" of
results/summary/m0d.json (its threshold file restored from the committed copy when missing). Every measurement is
cached by content under --out, so an interrupted run resumes by running the same command. The run report goes to
<out>/runs/<run id>-scan.{json,md}; results/summary/std_scan.json is replaced only by a complete run (all settings,
no --smoke, clean hashed files, the declared connectome and configuration, no compute abort).
Exit codes: 0 done (any outcome), 2 refused before measuring, 3 compute aborted.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import sys
import time
import uuid
from pathlib import Path

from flymon.brain.config import Params
from flymon.brain.fly_pool import FlyPool
from flymon.brain.h3_measure import PoolMeasurer
from flymon.brain.h3_store import ROOT, MeasureCache, code_key, git_state, sha256_file, write_bytes, write_json
from flymon.brain.j_measure import HASHED_FILES, MEASURE_FILES, JMeasurer
from flymon.brain.j_rules import COMPUTE_ABORTED
from flymon.brain.j_runner import stage1
from flymon.brain.j_setup import build
from flymon.brain.j_spec import SPEC, JSpec
from flymon.brain import j_store

POOL_TIMEOUT_S = 1800


def smoke_spec(spec: JSpec) -> JSpec:
    """A few-minute pass through every code path; its numbers are not judgments. The match tolerance and KC band are
    wide so the single setting is ordered and the smoke judgement has something to re-converge."""
    r = dataclasses.replace
    h3 = r(spec.h4.h3, reference=r(spec.h4.h3.reference, n=4), all51_seeds=(200,))
    return r(spec, h4=r(spec.h4, h3=h3), std_f=(0.9,), std_tau_ms=(300.0,), scale_bisect_steps=3, alpn_match_tol=0.5,
             kc_band_pct=(0.0, 100.0))


def refuse(why: str) -> int:
    print(f"refused: {why}", file=sys.stderr)
    return 2


def md_report(res: dict) -> str:
    sc = res["scan"]
    L = [f"# J.11.3 scan {res['run_id']}", "", f"commit {res['git']['commit']}, dirty hashed files "
         f"{res['git']['dirty_hashed'] or 'none'}, smoke {res['smoke']}, wall {res['wall_s'] / 60:.1f} min", "",
         f"**outcome: {sc['outcome']}**, order {sc['order']}", ""]
    if "base" in sc:
        b = sc["base"]["metrics"]
        L += [f"C3 (no depression): median ALPN {b['alpn_median']:.1f}, log10 var kc_on {b['log10_var']['kc_on']:.4f}, "
              f"reference KC {sc['base']['kc_pct']:.3f}%", "",
              "| f | tau ms | restorable | scale | KC % | feasible | kc_on log10 var | reduction | uniPN log10 var |",
              "|---|---|---|---|---|---|---|---|---|"]
        for s in sc["settings"]:
            m = s.get("metrics") or {"log10_var": {}}
            L.append(f"| {s['f']:g} | {s['tau_ms']:g} | {s['restorable']} | {s['scale'] or ''} | {s.get('kc_pct', '')} | "
                     f"{s['feasible']} | {m['log10_var'].get('kc_on', '')} | {s['reduction'] or ''} | "
                     f"{m['log10_var'].get('uni_hz', '')} |")
    return "\n".join(L) + "\n"


def main(argv=None, spec: JSpec | None = None, summary_spec: JSpec = SPEC, require_root: bool = True,
         pools: dict | None = None, pairs: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default="data/malecns.npz")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary", default="results/summary/std_scan.json")
    ap.add_argument("--m0d", default="results/summary/m0d.json")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--max-hours", type=float, default=None)
    a = ap.parse_args(argv)
    if require_root and Path.cwd().resolve() != ROOT:
        return refuse(f"not at the repository root {ROOT}")
    spec = spec or (smoke_spec(SPEC) if a.smoke else SPEC)
    out = Path(a.out or ("results/m0d/j/std-smoke" if a.smoke else "results/m0d/j/std"))
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    git = git_state(files=HASHED_FILES)
    if git["dirty_hashed"] and not a.allow_dirty:
        return refuse(f"hashed files are dirty {git['dirty_hashed']} (commit them, or --allow-dirty)")
    code, manifest = code_key(a.npz, files=MEASURE_FILES), code_key(a.npz, files=HASHED_FILES)
    npz_sha = code["files"]["npz:" + Path(a.npz).name]
    if npz_sha != spec.h4.h3.connectome_sha256:
        return refuse(f"{a.npz} is not the declared connectome ({npz_sha[:12]})")
    try:
        summary = json.loads(Path(a.m0d).read_text())
        base, _, _ = j_store.load_c3(summary, spec.h3_block, spec.c3_name)
    except (OSError, ValueError, KeyError) as e:
        return refuse(f"no usable C3 in block {spec.h3_block} of {a.m0d}: {e}")
    t0 = time.time()
    deadline = None if a.max_hours is None else t0 + 3600.0 * a.max_hours
    s = build(a.npz, spec, out, deadline=deadline, log=lambda x: print(x, flush=True), pools=pools, pairs=pairs)
    if pools is None and summary[spec.h3_block].get("pools") != s["core"]:
        return refuse(f"the core pools {s['core']} differ from block {spec.h3_block}'s")
    print(f"run {run_id}: J key {code['key'][:12]}, base C3 {base.kc_thresh:g}", flush=True)
    res = dict(run_id=run_id, smoke=a.smoke, argv=list(sys.argv[1:] if argv is None else argv), git=git,
               code=manifest, measure_key=code["key"], spec=spec, base_block=spec.h3_block,
               h3_run_id=summary[spec.h3_block].get("run_id"))
    with FlyPool(a.npz, Params(), [{} for _ in range(a.workers)], workers=a.workers,
                 punish_type=spec.h4.h3.punish_type, reward_type=spec.h4.h3.reward_type, timeout_s=POOL_TIMEOUT_S) as pool:
        cache = MeasureCache(out / "cache", code, run_id)
        m3 = PoolMeasurer(pool, spec.h4.h3, {"reference": s["odors"]}, s["all51"], cache, None)
        jm = JMeasurer(pool, spec, s["all51"], cache)
        res["scan"] = stage1(jm, m3, s["ctx"], base)
        guard_params = [Params()] + [p for p in m3.params_seen + jm.params_seen if p != Params()]
    res["wall_s"] = time.time() - t0
    res["cache"] = dict(hits=cache.hits, misses=cache.misses)
    res["artifacts"] = {p: sha256_file(p) for p in sorted(cache.used) if Path(p).exists()}
    report = write_json(out / "runs" / f"{run_id}-scan.json", res, guard_params)
    write_bytes(out / "runs" / f"{run_id}-scan.md", md_report(res).encode(), guard_params)
    aborted = res["scan"]["outcome"] == COMPUTE_ABORTED
    print(f"wrote {report}: {res['scan']['outcome']}, order {res['scan']['order']}, "
          f"{res['wall_s'] / 60:.1f} min", flush=True)
    blockers = dict(smoke=a.smoke, dirty=bool(git["dirty_hashed"]), spec=spec != summary_spec, aborted=aborted)
    if any(blockers.values()):
        print(f"summary not written: {[k for k, v in blockers.items() if v]}", flush=True)
    else:
        j_store.write_json(a.summary, dict(res, report=str(report), report_sha256=sha256_file(report)), guard_params)
        print(f"wrote {a.summary}", flush=True)
    return 3 if aborted else 0


if __name__ == "__main__":
    sys.exit(main())
