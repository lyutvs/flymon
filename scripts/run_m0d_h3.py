#!/usr/bin/env python3
"""M0d H.3: fix each combination's parameters (spec appendix H.3a) — C0, C1 and C3, in that order.

    uv run python scripts/run_m0d_h3.py                          # the full run (typically ~2 h, worst ~8 h)
    uv run python scripts/run_m0d_h3.py --smoke --allow-dirty    # a few minutes; results/m0d/h3-smoke/, no summary

Run it from the repository root. Every measurement is cached under --out by content (the measurement's inputs, the
files a measurement depends on, the NPZ, the Python and NumPy versions), so an interrupted run resumes by running the
same command, and a verification run (--kc-cells, --stop-after, --combos) seeds the cache of the full run.
The runner stops at the first combination that is dropped (H.3a.4: the user decides) unless --continue-after-drop.
The run report goes to <out>/runs/<run id>.{json,md}. Block "h3" of results/summary/m0d.json is replaced once, and
only by a complete run: all three combinations, the full grid, records on, no --smoke / --kc-cells / --stop-after,
no compute abort or stop, clean hashed files, the declared connectome and the declared configuration.
Exit codes: 0 done, 2 refused (dirty hashed files, not at the repository root), 3 compute aborted.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import sys
import time
import uuid
from pathlib import Path

import numpy as np

from flymon.brain.circuits import Populations, compartments
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.fly_pool import FlyPool
from flymon.brain.h3_c3 import ThresholdFiles, rule_thresholds, run_c3, update_mask
from flymon.brain.h3_measure import PoolMeasurer
from flymon.brain.h3_records import collect
from flymon.brain.h3_rules import COMBO_ABORTED, COMBO_ADOPTED, COMBO_DROPPED
from flymon.brain.h3_runner import ComputeAborted, Context, Deadline, run_c0, run_c1
from flymon.brain.h3_spec import SPEC, H3Spec, all51_glomeruli, make_odors, odor_digest
from flymon.brain.h3_store import (HASHED_FILES, ROOT, MeasureCache, code_key, git_state, replace_summary_block,
                                  sha256_file, write_bytes, write_json)

C3_THRESHOLDS_COPY = "results/summary/m0d_h3_c3_thresholds.npz"


def smoke_spec(spec: H3Spec) -> H3Spec:
    """A few-minute pass through every code path; its numbers are not judgments."""
    r = dataclasses.replace
    return r(spec, reference=r(spec.reference, n=4), extended=r(spec.extended, n=8), guard_half=2,
             kc_grid=(1.6,), scale_bisect_steps=2, hold_bisect_steps=2, membrane_tol_mv=5.0,
             design_seeds=(100, 101), design_extra_seeds=(102, 103), baseline_cal_seeds=(100, 101, 102, 103),
             baseline_gate_seeds=(116, 117, 118, 119), runaway_rest_seeds=(100, 101),
             runaway_odor_seeds=(100, 101, 102, 103), all51_seeds=(200,), boot_draws=200,
             homeo_max_iter=2, c3_max_cycles=1)


def pool_types(conn, comps, dan_type) -> list:
    t = np.asarray(conn.type).astype(str)
    return sorted({str(t[int(i)]) for i in comps[dan_type].core})


def md_report(res: dict) -> str:
    L = [f"# M0d H.3 run {res['run_id']}", "", f"commit {res['git']['commit']}, dirty hashed files "
         f"{res['git']['dirty_hashed'] or 'none'}, smoke {res['smoke']}, wall {res['wall_s'] / 60:.1f} min", ""]
    for name, c in res["combos"].items():
        L += [f"## {name}: {c['status']}", "", "| cell | status | rank | score | final Params (kc, s, hold) |", "|---|---|---|---|---|"]
        for cell in c["cells"]:
            fp = cell.get("final_params")
            L.append(f"| {cell['label']} | {cell.get('status')} | {cell.get('rank', '')} | "
                     f"{cell.get('stage3', {}).get('score', '')} | "
                     + (f"{fp.kc_thresh:g}, {fp.apl_input_scale:.6g}, {fp.mbon_hold_frac:.6g}" if fp else "") + " |")
        L.append("")
    return "\n".join(L) + "\n"


def main(argv=None, spec: H3Spec | None = None, summary_spec: H3Spec = SPEC, require_root: bool = True) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default="data/malecns.npz")
    ap.add_argument("--combos", default="C0,C1,C3")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary", default="results/summary/m0d.json")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--kc-cells", type=float, nargs="+", default=None)   # verification only: restricts the grid
    ap.add_argument("--stop-after", choices=["stage1"], default=None)    # verification only: C1 stage 1
    ap.add_argument("--no-records", action="store_true")
    ap.add_argument("--continue-after-drop", action="store_true")        # only after the user decided to continue
    ap.add_argument("--max-hours", type=float, default=None)
    a = ap.parse_args(argv)

    if require_root and Path.cwd().resolve() != ROOT:
        print(f"refusing to run outside the repository root {ROOT}", file=sys.stderr)
        return 2
    spec = spec or (smoke_spec(SPEC) if a.smoke else SPEC)
    if a.kc_cells:
        spec = dataclasses.replace(spec, kc_grid=tuple(float(k) for k in a.kc_cells))
    out = Path(a.out or ("results/m0d/h3-smoke" if a.smoke else "results/m0d/h3"))
    combos = [c.strip() for c in a.combos.split(",") if c.strip()]
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    git = git_state()
    if git["dirty_hashed"] and not a.allow_dirty:
        print(f"refusing to run: hashed files are dirty {git['dirty_hashed']} (commit them, or --allow-dirty)",
              file=sys.stderr)
        return 2
    measure_code = code_key(a.npz)                                  # the cache key's code part
    code = code_key(a.npz, files=HASHED_FILES)                      # the manifest
    npz_sha = code["files"]["npz:" + Path(a.npz).name]
    t0 = time.time()
    deadline = None if a.max_hours is None else t0 + 3600.0 * a.max_hours

    conn = Connectome.load(a.npz)
    pops = Populations.from_connectome(conn)
    comps = compartments(conn, pops, Params().core_frac)
    odors = dict(reference=make_odors(pops, spec.reference), extended=make_odors(pops, spec.extended))
    gloms, c_norm = all51_glomeruli(pops, spec.reference.exclude)
    pools = dict(A=pool_types(conn, comps, spec.punish_type), P=pool_types(conn, comps, spec.reward_type))
    upd = update_mask(conn, pops, Params())
    rules = {}

    def rule_for(kc):
        if kc not in rules:
            rules[kc] = rule_thresholds(conn, pops, dataclasses.replace(Params(), kc_thresh=float(kc)))
        return rules[kc]

    diag_root = Path("results/m0d/diag/h3-smoke" if a.smoke else "results/m0d/diag/h3")
    ctx = Context(spec=spec, odors=odors["reference"], pools=pools, n_kc=len(pops.kc), deadline=deadline,
                  log=lambda s: print(s, flush=True),
                  extra=dict(update_mask=upd, rule_thresholds=rule_for, stop_after=a.stop_after,
                             diag_dir=str(diag_root / "contrasts"),
                             threshold_files=ThresholdFiles(out / "thresholds", conn.bodyId[pops.kc])))
    cache = MeasureCache(out / "cache", measure_code, run_id)
    diag_cache = MeasureCache(diag_root / "cache", measure_code, run_id)
    print(f"run {run_id}: combos {combos}, pools {pools}, {len(odors['reference'])} reference odours, "
          f"update set {int(upd.sum())} KCs, measurement key {measure_code['key'][:12]}", flush=True)

    res = dict(run_id=run_id, smoke=a.smoke, argv=list(sys.argv[1:] if argv is None else argv), git=git,
               code=code, measure_key=measure_code["key"], spec=spec,
               odor_digests={k: odor_digest(v) for k, v in odors.items()}, pools=pools,
               update_set_n=int(upd.sum()), combos={}, records={}, stopped_after=None)
    aborted = False
    with FlyPool(a.npz, Params(), [{} for _ in range(a.workers)], workers=a.workers,
                 punish_type=spec.punish_type, reward_type=spec.reward_type) as pool:
        m = PoolMeasurer(pool, spec, odors, (gloms, c_norm), cache, diag_cache)
        c1 = None
        for name in ("C0", "C1", "C3"):
            if name not in combos and not (name == "C1" and "C3" in combos):
                continue
            print(f"== {name}", flush=True)
            r = run_c0(m, ctx) if name == "C0" else run_c1(m, ctx) if name == "C1" else run_c3(m, ctx, c1)
            res["combos"][name] = r
            if name == "C1":
                c1 = r
            if r["status"] == COMBO_ABORTED:
                aborted = True
                break
            if r["status"] == COMBO_ADOPTED and not a.no_records:
                print(f"== {name} records", flush=True)
                try:
                    res["records"][name] = collect(Deadline(m, ctx), ctx, r["adopted"]["params"])
                except ComputeAborted as e:
                    res["records"][name] = dict(aborted=str(e))
                    aborted = True
                    break
            if r["status"] == COMBO_DROPPED and not a.continue_after_drop:
                res["stopped_after"] = name
                print(f"== stopped: {name} was dropped (H.3a.4: the user decides; --continue-after-drop to go on)",
                      flush=True)
                break
        guard_params = [Params()] + [p for p in m.params_seen if p != Params()]
    res["wall_s"] = time.time() - t0
    res["cache"] = dict(hits=cache.hits + diag_cache.hits, misses=cache.misses + diag_cache.misses)
    used = sorted(set(cache.used) | set(diag_cache.used) | {str(p) for p in (out / "thresholds").glob("theta-*.npz")})
    res["artifacts"] = {p: sha256_file(p) for p in used if Path(p).exists()}
    report = write_json(out / "runs" / f"{run_id}.json", res, guard_params)
    write_bytes(out / "runs" / f"{run_id}.md", md_report(res).encode(), guard_params)
    print(f"wrote {report} in {res['wall_s'] / 60:.1f} min; cache hits {res['cache']['hits']}, "
          f"misses {res['cache']['misses']}", flush=True)
    blockers = dict(smoke=a.smoke, kc_cells=a.kc_cells is not None, stop_after=a.stop_after is not None,
                    combos=set(combos) != {"C0", "C1", "C3"}, aborted=aborted, stopped=res["stopped_after"] is not None,
                    no_records=a.no_records, dirty=bool(git["dirty_hashed"]), connectome=npz_sha != spec.connectome_sha256,
                    spec=spec != summary_spec)
    if any(blockers.values()):
        print(f"summary not written: {[k for k, v in blockers.items() if v]}", flush=True)
    else:
        block = dict(res, report=str(report), report_sha256=sha256_file(report))
        c3 = res["combos"].get("C3", {})
        if c3.get("status") == COMBO_ADOPTED:
            src = Path(c3["adopted"]["params"].kc_thresh_file)
            copy = write_bytes(C3_THRESHOLDS_COPY, src.read_bytes(), guard_params)
            block["c3_thresholds"] = dict(path=str(copy), sha256=sha256_file(copy), source=str(src))
        replace_summary_block(a.summary, "h3", block, guard_params)
        print(f"replaced block 'h3' of {a.summary}", flush=True)
    return 3 if aborted else 0


if __name__ == "__main__":
    sys.exit(main())
