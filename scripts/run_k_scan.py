#!/usr/bin/env python3
"""Spec K.8.2, stage 1: for every g, re-converge C3's rule with fast KC->KC inhibition, measure the X-only MBON13
drive N on the odd-turn (b) pairs for ADOPTED engines, rank and gate (K.8.3-4); or K.8.7's self-check (i).

    uv run python scripts/run_k_scan.py                        # stage 1 (~5-7 h; resumable: rerun the same command)
    uv run python scripts/run_k_scan.py --self-check i         # g = 0 re-converged = C3's recorded thresholds (~1 h)
    uv run python scripts/run_k_scan.py --smoke --allow-dirty  # minutes

Run it from the repository root. Report: <out>/runs/<run id>-scan.{json,md} (self-check: <out>/selfcheck/<id>-i.json).
results/summary/k_engine.json block "scan" is replaced only by a complete stage 1 (no --smoke / --grid / --pairs,
clean hashed files, the declared configuration, not aborted). Exit codes: 0 done (any outcome; a matching self-check),
2 refused before measuring, 3 compute aborted, 5 self-check (i) mismatch (stop and ask the user).
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import sys
import time
import uuid
from pathlib import Path

from flymon.brain import j_store, k_store
from flymon.brain.config import Params
from flymon.brain.fly_pool import FlyPool
from flymon.brain.h3_measure import PoolMeasurer
from flymon.brain.h3_runner import Deadline
from flymon.brain.h3_store import ROOT, MeasureCache, code_key, git_state, sha256_file
from flymon.brain.h4_pairs import pairs_digest
from flymon.brain.j_rules import COMPUTE_ABORTED, c3_mismatch, cycle_divergence
from flymon.brain.j_measure import JMeasurer
from flymon.brain.j_runner import params_json, reconverge
from flymon.brain.j_setup import build
from flymon.brain.k_measure import HASHED_FILES, MEASURE_FILES, KMeasurer
from flymon.brain.k_pairs import odd_pairs, overlap_report
from flymon.brain.k_params import k_make
from flymon.brain.k_runner import arrays, stage1
from flymon.brain.k_spec import SPEC, KSpec, smoke

POOL_TIMEOUT_S = 1800


def refuse(why: str) -> int:
    print(f"refused: {why}", file=sys.stderr)
    return 2


def out_allowed(out) -> bool:
    """--out must lie under results/m0d/k/ of the repository root (normalised like k_store.guard)."""
    rel = os.path.relpath(os.path.abspath(str(out)), ROOT).replace(os.sep, "/")
    return (rel + "/").startswith(k_store.ALLOWED_DIR)


def main(argv=None, spec: KSpec | None = None, summary_spec: KSpec = SPEC, require_root: bool = True,
         pools: dict | None = None, odd: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default="data/malecns.npz")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary", default="results/summary/k_engine.json")
    ap.add_argument("--m0d", default="results/summary/m0d.json")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--grid", type=float, nargs="+", default=None)   # verification only
    ap.add_argument("--pairs", type=int, default=None)               # verification only: the first N odd pairs
    ap.add_argument("--self-check", choices=["i"], default=None)
    ap.add_argument("--max-hours", type=float, default=None)
    a = ap.parse_args(argv)
    if a.pairs is not None and a.pairs < 1:
        return refuse(f"--pairs must be at least 1, got {a.pairs}")
    if require_root and Path.cwd().resolve() != ROOT:
        return refuse(f"not at the repository root {ROOT}")
    spec = spec or (smoke(SPEC) if a.smoke else SPEC)
    if a.grid:
        spec = dataclasses.replace(spec, grid=tuple(a.grid))
    out = Path(a.out or ("results/m0d/k/smoke" if a.smoke else "results/m0d/k/run"))
    if not out_allowed(out):
        return refuse(f"--out {out} is not under {k_store.ALLOWED_DIR} of the repository root")
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    git = git_state(files=HASHED_FILES)
    if git["dirty_hashed"] and not a.allow_dirty:
        return refuse(f"hashed files are dirty {git['dirty_hashed']} (commit them, or --allow-dirty)")
    code, manifest = code_key(a.npz, files=MEASURE_FILES), code_key(a.npz, files=HASHED_FILES)
    npz_sha = code["files"]["npz:" + Path(a.npz).name]
    if npz_sha != spec.j.h4.h3.connectome_sha256:
        return refuse(f"{a.npz} is not the declared connectome ({npz_sha[:12]})")
    try:
        summary = json.loads(Path(a.m0d).read_text())
        c3, _, _ = j_store.load_c3(summary, spec.j.h3_block, spec.j.c3_name)
    except (OSError, ValueError, KeyError) as e:
        return refuse(f"no usable C3 in block {spec.j.h3_block} of {a.m0d}: {e}")
    t0 = time.time()
    deadline = None if a.max_hours is None else t0 + 3600.0 * a.max_hours
    s = build(a.npz, spec.j, out, deadline=deadline, log=lambda x: print(x, flush=True), pools=pools)
    if pools is None and summary[spec.j.h3_block].get("pools") != s["core"]:
        return refuse(f"the core pools {s['core']} differ from block {spec.j.h3_block}'s")
    odd = odd if odd is not None else odd_pairs(s["pops"])
    digest = pairs_digest(odd)
    if not a.smoke and a.pairs is None and digest != spec.odd_pairs_digest:
        return refuse(f"the odd pair list differs from the declared one ({digest[:12]})")
    if a.pairs:
        odd = odd[:a.pairs]
    print(f"run {run_id}: K key {code['key'][:12]}, {len(odd)} odd (b) pairs", flush=True)
    res = dict(run_id=run_id, smoke=a.smoke, argv=list(sys.argv[1:] if argv is None else argv), git=git,
               code=manifest, measure_key=code["key"], spec=spec, self_check=a.self_check, odd_pairs_digest=digest,
               n_odd=len(odd), overlap=overlap_report(odd, {"even_b": [p for p in s["pairs"] if p["axis"] == "b"],
                                                             "even_a": [p for p in s["pairs"] if p["axis"] == "a"]}))
    with FlyPool(a.npz, Params(), [{} for _ in range(a.workers)], workers=a.workers,
                 punish_type=spec.j.h4.h3.punish_type, reward_type=spec.j.h4.h3.reward_type,
                 timeout_s=POOL_TIMEOUT_S) as pool:
        cache = MeasureCache(out / "cache", code, run_id)
        m3 = PoolMeasurer(pool, spec.j.h4.h3, {"reference": s["odors"]}, s["all51"], cache, None)
        km = KMeasurer(pool, spec, odd, len(s["pops"].kc), cache)
        jm = JMeasurer(pool, spec.j, s["all51"], cache)
        seen = [m3, km, jm]
        if a.self_check == "i":
            rc = reconverge(Deadline(m3, s["ctx"].h3), s["ctx"], k_make(spec.j.h4.h3, 0.0))
            want = summary[spec.j.h3_block]["combos"][spec.j.c3_name]
            got = params_json(rc["adopted"]["params"]) if rc["adopted"] else {}
            bad = c3_mismatch(got, want["adopted"]["params"])
            res["check"] = dict(ok=not bad, status=rc["status"], mismatched=bad, got=got,
                                want=want["adopted"]["params"],
                                divergence=None if not bad else cycle_divergence(rc["cells"], want["cells"]))
        else:
            res["stage1"] = stage1(m3, km, s["ctx"], spec, c3, arrays(s["conn"], s["pops"], spec, c3.mv_per_synapse), jm)
    guard_params = [Params()] + list(dict.fromkeys(p for m in seen for p in m.params_seen if p != Params()))
    res["wall_s"] = time.time() - t0
    res["cache"] = dict(hits=cache.hits, misses=cache.misses)
    res["artifacts"] = {p: sha256_file(p) for p in sorted(cache.used) if Path(p).exists()}
    if a.self_check:
        rep = k_store.write_json(out / "selfcheck" / f"{run_id}-i.json", res, guard_params)
        print(f"wrote {rep}: self-check (i) {'MATCHES' if res['check']['ok'] else 'DIFFERS'} "
              f"in {res['wall_s'] / 60:.1f} min", flush=True)
        return 0 if res["check"]["ok"] else 5
    st1 = res["stage1"]
    report = k_store.write_json(out / "runs" / f"{run_id}-scan.json", res, guard_params)
    lines = [f"# K.8.2 stage 1 {run_id}\n", f"**outcome: {st1['outcome']}**\n",
             f"C3 N: {st1['base']['metrics']['N'] if st1['base'] else None}\n"]
    for i, t in enumerate(st1["settings"]):
        n = t["metrics"]["N"] if t["metrics"] else None
        lines.append(f"- g {t['g']:g}: {t['status']}, N {n}, ratio {st1['ratios'].get(i)}\n")
    lines.append(f"\norder: {[st1['settings'][i]['g'] for i in st1['order']]}\n")
    k_store.write_bytes(out / "runs" / f"{run_id}-scan.md", "".join(lines).encode(), guard_params)
    print(f"wrote {report}: {st1['outcome']} in {res['wall_s'] / 60:.1f} min", flush=True)
    blockers = dict(smoke=a.smoke, grid=a.grid is not None, pairs=a.pairs is not None,
                    dirty=bool(git["dirty_hashed"]), spec=spec != summary_spec, aborted=st1["outcome"] == COMPUTE_ABORTED)
    if any(blockers.values()):
        print(f"summary not written: {[k for k, v in blockers.items() if v]}", flush=True)
    else:
        k_store.write_summary_block(a.summary, "scan", dict(res, report=str(report), report_sha256=sha256_file(report)),
                                    guard_params)
        print(f"wrote {a.summary} (block scan)", flush=True)
    return 3 if st1["outcome"] == COMPUTE_ABORTED else 0


if __name__ == "__main__":
    sys.exit(main())
