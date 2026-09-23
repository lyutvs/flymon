#!/usr/bin/env python3
"""Spec J.11.4, stage 2: re-converge the scan's best depression setting by C3's rule and judge it by the M2 bar, with
D.6 (a) and (b) on the judged engine; or one of J.11.6's self-checks.

    uv run python scripts/run_j_std_judge.py                   # the judgement (normal ~2 h; worst ~7 h per setting)
    uv run python scripts/run_j_std_judge.py --self-check i    # C3 re-measured: guard + first (b) pair = blocks h3/h4
    uv run python scripts/run_j_std_judge.py --self-check ii   # C3's rule without depression = C3's thresholds (~1 h)
    uv run python scripts/run_j_std_judge.py --smoke --allow-dirty --scan <a smoke scan report>   # minutes

Run it from the repository root. The settings and their order come from --scan (results/summary/std_scan.json: a
complete scan measured under this code's J key; outside --smoke a smoke scan, one measured with dirty hashed files,
one run under another J configuration (spec) or another procedure code (HASHED_FILES manifest key) is refused). The run
report goes to <out>/runs/<run id>-judge.{json,md} (self-checks:
<out>/selfcheck/<run id>-<i|ii>.json). results/summary/m2_engine.json is replaced only by a complete judgement (every
pair, D.6 measured, no --smoke / --pairs / --no-d6, clean hashed files, the declared configuration) whose outcome is
SELECTED, B or STOP_NO_OPERATING_POINT; a SELECTED one carries a `selection` with confirmed = false (H.5 sets it).
Exit codes: 0 done (any outcome; a self-check that matches), 2 refused before measuring, 3 compute aborted,
4 self-check (i) mismatch, 5 self-check (ii) mismatch (the divergence is recorded; stop and ask the user).
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

from flymon.brain import d6a, j_store
from flymon.brain.config import Params
from flymon.brain.fly_pool import FlyPool
from flymon.brain.h3_measure import PoolMeasurer
from flymon.brain.h3_store import (ROOT, MeasureCache, canonical, code_key, git_state, sha256_file, write_bytes,
                                   write_json)
from flymon.brain.h4_formula import pair_stats
from flymon.brain.h4_measure import H4Measurer
from flymon.brain.h4_pairs import pair_key
from flymon.brain.h4_runner import Context as H4Context, reselect
from flymon.brain.j_measure import HASHED_FILES, MEASURE_FILES, JMeasurer
from flymon.brain.j_params import params_from_json
from flymon.brain.j_rules import (B, COMPUTE_ABORTED, SCAN_COMPLETE, SELECTED, STOP_NO_OPERATING_POINT, c3_mismatch,
                                  cycle_divergence, pair_mismatch)
from flymon.brain.j_runner import params_json, reconverge, stage2
from flymon.brain.j_setup import build
from flymon.brain.j_spec import SPEC, JSpec

POOL_TIMEOUT_S = 1800
WRITES_SUMMARY = (SELECTED, B, STOP_NO_OPERATING_POINT)


def smoke_spec(spec: JSpec) -> JSpec:
    r = dataclasses.replace
    h3 = r(spec.h4.h3, reference=r(spec.h4.h3.reference, n=4), all51_seeds=(200,), kc_grid=(1.65,),
           scale_bisect_steps=2, hold_bisect_steps=2, membrane_tol_mv=5.0, design_seeds=(100, 101),
           design_extra_seeds=(102, 103), baseline_cal_seeds=(100, 101, 102, 103), baseline_gate_seeds=(116, 117),
           runaway_rest_seeds=(100, 101), runaway_odor_seeds=(100, 101), boot_draws=200, homeo_max_iter=2,
           c3_max_cycles=1)
    h4 = r(spec.h4, h3=h3, teach_seeds=(8, 9), teach_min_decreased=1, act_seeds=(500, 501), select_seeds=(600, 601),
           report_seeds=(608, 609))
    return r(spec, h4=h4, max_settings=1)


def refuse(why: str) -> int:
    print(f"refused: {why}", file=sys.stderr)
    return 2


def self_check_i(pool, s, spec, summary, base, guard, cache, seen: list) -> dict:
    """C3 re-measured under this code: H.4's reselection must reproduce block "h3"'s guard (it raises otherwise) and the
    z constants and first (b) pair of block "h4"'s C3 oracle."""
    ctx = s["ctx"]
    m3 = PoolMeasurer(pool, spec.h4.h3, {"reference": s["odors"]}, s["all51"], cache, None)
    first = [p for p in s["pairs"] if p["axis"] == "b"][:1]
    m4 = H4Measurer(pool, spec.h4, first, ctx.pools, cache, m3)
    seen += [m3, m4]
    h4ctx = H4Context(spec=spec.h4, combos={"C3": base}, pools=ctx.pools, probe_seeds=ctx.probe_seeds,
                      expected=[pair_key(p) for p in first], h3_guard={"C3": guard}, log=ctx.log)
    try:
        r = reselect(m4, h4ctx, "C3", base)
    except RuntimeError as e:
        return dict(ok=False, stage="reactivity", note=str(e))
    rec = summary["h4"]["h4"]["combos"]["C3"]
    z_ok = {k: list(v) for k, v in r["z"].items()} == {k: list(v) for k, v in rec["z"].items()}
    row = m4.oracle(base, r["readout"], r["z"])[0]
    got = pair_stats(row["report"], r["z"], spec.h4.testable_min)
    key = pair_key(first[0])
    want = next(p for p in rec["oracle"]["pairs"] if (p["axis"], p["turn"], p["x"], p["y"]) == tuple(key))
    bad = pair_mismatch(got, want)
    return dict(ok=bool(z_ok and not bad), stage=None if z_ok and not bad else ("z" if not z_ok else "pair"),
                z=r["z"], z_recorded=rec["z"], pair=list(key), got=got, want=want, mismatched=bad)


def self_check_ii(pool, s, spec, summary, cache, seen: list) -> dict:
    """C3's rule without depression (make_base None) must adopt block "h3"'s C3; otherwise the first diverging cycle."""
    ctx = s["ctx"]
    m3 = PoolMeasurer(pool, spec.h4.h3, {"reference": s["odors"]}, s["all51"], cache, None)
    seen.append(m3)
    rc = reconverge(m3, ctx, None)
    want = summary[spec.h3_block]["combos"][spec.c3_name]
    got = params_json(rc["adopted"]["params"]) if rc["adopted"] else {}
    bad = c3_mismatch(got, want["adopted"]["params"])
    return dict(ok=not bad, status=rc["status"], mismatched=bad, got=got, want=want["adopted"]["params"],
                divergence=None if not bad else cycle_divergence(rc["cells"], want["cells"]), reconverge=rc)


def main(argv=None, spec: JSpec | None = None, summary_spec: JSpec = SPEC, require_root: bool = True,
         pools: dict | None = None, pairs: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default="data/malecns.npz")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=None)
    ap.add_argument("--scan", default="results/summary/std_scan.json")
    ap.add_argument("--summary", default="results/summary/m2_engine.json")
    ap.add_argument("--m0d", default="results/summary/m0d.json")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--pairs", type=int, default=None)          # verification only: the first N pairs of each axis
    ap.add_argument("--no-d6", action="store_true")             # verification only
    ap.add_argument("--self-check", choices=["i", "ii"], default=None)
    ap.add_argument("--max-hours", type=float, default=None)
    a = ap.parse_args(argv)
    if a.pairs is not None and a.pairs < 1:
        return refuse(f"--pairs must be at least 1, got {a.pairs}")
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
        base, guard, _ = j_store.load_c3(summary, spec.h3_block, spec.c3_name)
    except (OSError, ValueError, KeyError) as e:
        return refuse(f"no usable C3 in block {spec.h3_block} of {a.m0d}: {e}")
    scan = None
    if a.self_check is None:
        try:
            scan = json.loads(Path(a.scan).read_text())
        except (OSError, ValueError) as e:
            return refuse(f"no usable scan at {a.scan}: {e}")
        if scan.get("measure_key") != code["key"]:
            return refuse(f"the scan was measured under J key {str(scan.get('measure_key'))[:12]}, this code's is "
                          f"{code['key'][:12]}: its settings are not this engine's")
        if scan.get("scan", {}).get("outcome") != SCAN_COMPLETE:
            return refuse(f"the scan's outcome is {scan.get('scan', {}).get('outcome')}, not {SCAN_COMPLETE}")
        if not a.smoke and (scan.get("smoke") or scan.get("git", {}).get("dirty_hashed")):
            why = "a smoke run" if scan.get("smoke") else f"measured with dirty hashed files {scan['git']['dirty_hashed']}"
            return refuse(f"the scan is {why}: only a --smoke judgement may use it")
        if not a.smoke and canonical(scan.get("spec")) != canonical(spec):
            return refuse("the scan was run under another J configuration (its spec differs from this judgement's): "
                          "its settings and order are not this configuration's")
        if not a.smoke and scan.get("code", {}).get("key") != manifest["key"]:
            return refuse(f"the scan was run under another procedure code {str(scan.get('code', {}).get('key'))[:12]}, "
                          f"this code's is {manifest['key'][:12]} (HASHED_FILES: grid, tolerances, rules may differ)")
    t0 = time.time()
    deadline = None if a.max_hours is None else t0 + 3600.0 * a.max_hours
    s = build(a.npz, spec, out, deadline=deadline, log=lambda x: print(x, flush=True), pools=pools, pairs=pairs)
    if pools is None and summary[spec.h3_block].get("pools") != s["core"]:
        return refuse(f"the core pools {s['core']} differ from block {spec.h3_block}'s")
    if pairs is None and (s["pairs_digest"] != spec.h4.pairs_digest):
        return refuse(f"the pair list differs from the declared one ({s['pairs_digest'][:12]})")
    if a.pairs:
        s["pairs"] = [p for ax in ("a", "b") for p in [q for q in s["pairs"] if q["axis"] == ax][:a.pairs]]
        s["ctx"].expected = [pair_key(p) for p in s["pairs"]]
    res = dict(run_id=run_id, smoke=a.smoke, argv=list(sys.argv[1:] if argv is None else argv), git=git,
               code=manifest, measure_key=code["key"], spec=spec, self_check=a.self_check,
               scan_run_id=None if scan is None else scan.get("run_id"), n_pairs=len(s["pairs"]))
    with FlyPool(a.npz, Params(), [{} for _ in range(a.workers)], workers=a.workers,
                 punish_type=spec.h4.h3.punish_type, reward_type=spec.h4.h3.reward_type, timeout_s=POOL_TIMEOUT_S) as pool:
        cache, seen = MeasureCache(out / "cache", code, run_id), []
        if a.self_check == "i":
            res["check"] = self_check_i(pool, s, spec, summary, base, guard, cache, seen)
        elif a.self_check == "ii":
            res["check"] = self_check_ii(pool, s, spec, summary, cache, seen)
        else:
            m3 = PoolMeasurer(pool, spec.h4.h3, {"reference": s["odors"]}, s["all51"], cache, None)
            m4 = H4Measurer(pool, spec.h4, s["pairs"], s["ctx"].pools, cache, m3)
            jm = JMeasurer(pool, spec, s["all51"], cache)
            seen += [m3, m4, jm]
            settings = [dict(f=x["f"], tau_ms=x["tau_ms"], scale=x["scale"]) for x in scan["scan"]["settings"]]
            seeds = d6a.SEEDS[:2] if a.smoke else d6a.SEEDS
            res["stage2"] = stage2(m3, m4, jm, s["ctx"], settings, scan["scan"]["order"], with_d6=not a.no_d6,
                                   d6_seeds=seeds)
    guard_params = [Params()] + list(dict.fromkeys(p for m in seen for p in m.params_seen if p != Params()))
    res["wall_s"] = time.time() - t0
    res["cache"] = dict(hits=cache.hits, misses=cache.misses)
    res["artifacts"] = {p: sha256_file(p) for p in sorted(cache.used) if Path(p).exists()}
    if a.self_check:
        rep = write_json(out / "selfcheck" / f"{run_id}-{a.self_check}.json", res, guard_params)
        print(f"wrote {rep}: self-check ({a.self_check}) {'MATCHES' if res['check']['ok'] else 'DIFFERS'}", flush=True)
        return 0 if res["check"]["ok"] else (4 if a.self_check == "i" else 5)
    out2 = res["stage2"]
    report = write_json(out / "runs" / f"{run_id}-judge.json", res, guard_params)
    write_bytes(out / "runs" / f"{run_id}-judge.md", (f"# J.11.4 judgement {run_id}\n\n**outcome: {out2['outcome']}**\n\n"
                                                      f"tried: {[t['name'] for t in out2['tried']]}\n\n"
                                                      f"reading: {out2.get('judge', {}).get('reading')}\n\n"
                                                      f"D.6: {out2.get('d6')}\n").encode(), guard_params)
    print(f"wrote {report}: {out2['outcome']} in {res['wall_s'] / 60:.1f} min", flush=True)
    blockers = dict(smoke=a.smoke, pairs=a.pairs is not None, no_d6=a.no_d6, dirty=bool(git["dirty_hashed"]),
                    spec=spec != summary_spec, outcome=out2["outcome"] not in WRITES_SUMMARY)
    if any(blockers.values()):
        print(f"summary not written: {[k for k, v in blockers.items() if v]}", flush=True)
    else:
        block = dict(res, report=str(report), report_sha256=sha256_file(report))
        if out2["outcome"] == SELECTED:
            t, rd = out2["tried"][out2["judged"]], out2["judge"]["reading"]
            block["selection"] = dict(setting=t["name"], params=t["reconverge"]["adopted"]["params"], T_b=rd["T_b"],
                                      F_a=rd["F_a"], testable_b=rd["testable_b"], n_b=rd["n_b"], confirmed=False)
        j_store.write_json(a.summary, block, guard_params)
        print(f"wrote {a.summary}", flush=True)
    return 3 if out2["outcome"] == COMPUTE_ABORTED else 0


if __name__ == "__main__":
    sys.exit(main())
