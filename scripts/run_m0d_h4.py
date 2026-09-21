#!/usr/bin/env python3
"""M0d H.4: select the combination (spec appendix H.4, amended by H.3a.1, H.3a.9 and H.4a) — readout reselection and
the oracle for C0, C1 and C3, then the selection.

    uv run python scripts/run_m0d_h4.py                          # the full run (~1.5 h on 16 workers)
    uv run python scripts/run_m0d_h4.py --smoke --allow-dirty    # minutes; results/m0d/h4-smoke/, no summary

Run it from the repository root, and leave the hashed files alone while it runs. The adopted Params come from block
"h3" of results/summary/m0d.json. Reference and rest measurements are H.3's cache entries (results/m0d/h3/cache); the
conditioning arms and the oracle are cached under --out by content, per pair for the oracle, so an interrupted run
resumes by running the same command. Every combination is reselected before the first oracle.
The run report goes to <out>/runs/<run id>.{json,md}. Block "h4" of results/summary/m0d.json is replaced (rerunning a
finished run replaces it again, with the same verdict from the cache) only by a complete run: all pairs, no --smoke /
--pairs, clean hashed files, the declared configuration, and an outcome that is neither INVALID nor the mid-run stop
on two passing types in a pool (the stops for the user's decision after a selection are complete results).
Exit codes: 0 done (whatever the outcome); 2 refused before measuring anything — not at the repository root, dirty
hashed files, --pairs below 1, an NPZ other than the declared connectome, no usable block "h3", core pools or an H.3
measurement key other than block "h3"'s, a pair list other than the declared one, a block "h3" without three adopted
combinations or with a C3 threshold file that fails its sha256. A reactivity that differs from block "h3"'s guard
raises: the run stops with a traceback before any oracle and writes nothing.
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

import numpy as np

from flymon.brain.circuits import Populations, compartments
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.fly_pool import FlyPool
from flymon.brain.h3_measure import PoolMeasurer
from flymon.brain.h3_spec import make_odors
from flymon.brain.h3_store import (MEASURE_FILES as H3_MEASURE_FILES, ROOT, MeasureCache, code_key, git_state,
                                  replace_summary_block, sha256_file, write_bytes, write_json)
from flymon.brain.h4_measure import HASHED_FILES, MEASURE_FILES, H4Measurer
from flymon.brain.h4_pairs import even_pairs, pair_key, pairs_digest
from flymon.brain.h4_rules import INVALID, STOP_MULTI_TYPE
from flymon.brain.h4_runner import Context, run_h4
from flymon.brain.h4_spec import SPEC, H4Spec

H3_CACHE = "results/m0d/h3/cache"
C3_COPY = "results/summary/m0d_h3_c3_thresholds.npz"
POOL_TIMEOUT_S = 1800.0          # one oracle round is ~6-8 min; a lost worker should not stall the run for hours


def refuse(msg: str) -> int:
    print(f"refusing to run: {msg}", file=sys.stderr)
    return 2


def params_from_json(d: dict) -> Params:
    d = dict(d)
    d["kc_norm_clip"] = tuple(d["kc_norm_clip"])
    d["sign_override"] = tuple(tuple(x) for x in d["sign_override"])
    return Params(**d)


def adopted(summary: dict, spec: H4Spec) -> tuple[dict, dict]:
    """{name: Params} and {name: {type: guard stats}} from block "h3". C3's threshold file must exist with its sha256;
    a missing source is restored from the committed copy (same sha256). ValueError when the block cannot be used."""
    try:
        h3 = summary[spec.h3_block]
        params, guards = {}, {}
        for name in spec.combos:
            c = h3["combos"][name]
            if c["status"] != "adopted":
                raise ValueError(f"block {spec.h3_block}: {name} is {c['status']}, not adopted")
            params[name] = params_from_json(c["adopted"]["params"])
            cell = next(x for x in c["cells"] if x.get("status") == "adopted")
            guards[name] = {t: st for k in ("A", "P") for t, st in cell["stage3"]["guard"][k]["types"].items()}
    except (KeyError, StopIteration, TypeError) as e:
        raise ValueError(f"block {spec.h3_block} is malformed: {e!r}") from None
    for name, p in params.items():
        if p.kc_thresh_mode != "homeostatic":
            continue
        src = Path(p.kc_thresh_file)
        if not src.exists():
            if not Path(C3_COPY).exists() or sha256_file(C3_COPY) != p.kc_thresh_sha256:
                raise ValueError(f"{src} is missing and {C3_COPY} is missing or does not match {name}'s sha256")
            write_bytes(src, Path(C3_COPY).read_bytes(), [p])
        if sha256_file(src) != p.kc_thresh_sha256:
            raise ValueError(f"{src}: sha256 differs from {name}'s Params")
    return params, guards


def smoke_spec(spec: H4Spec) -> H4Spec:
    """A few-minute pass through every code path (with --pairs 1); its numbers are not judgments."""
    r = dataclasses.replace
    return r(spec, teach_seeds=(8, 9), teach_min_decreased=1, act_seeds=(500, 501), select_seeds=(600, 601),
             report_seeds=(608, 609))


def md_report(res: dict) -> str:
    L = [f"# M0d H.4 run {res['run_id']}", "", f"commit {res['git']['commit']}, dirty hashed files "
         f"{res['git']['dirty_hashed'] or 'none'}, smoke {res['smoke']}, wall {res['wall_s'] / 60:.1f} min", "",
         f"**outcome: {res['h4']['outcome']}**", "", "| combination | status | readout | z | testable (b) / T_b | F_a |"
         " naive floor A / P (mean, zero share) | A only / P only testable (b) | report-seed halves |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, c in res["h4"]["combos"].items():
        agg, rec = (c.get("oracle") or {}).get("aggregate"), c.get("records")
        cells = [name, c["status"], str(c.get("readout") or ""), str(c.get("z") or ""),
                 f"{agg['testable_b']}/{agg['n_b']} / {agg['T_b']:.3f}" if agg else "", str(agg["F_a"]) if agg else ""]
        if rec:
            fl, st = rec["naive_floor"], rec["single_type"]
            cells += [" / ".join(f"({fl[k]['mean']:.1f}, {fl[k]['zero_share']:.2f})" for k in ("A", "P")),
                      " / ".join(str(st[k]["aggregate"]["testable_b"]) if st[k]["aggregate"] else "-" for k in ("A", "P")),
                      " / ".join(str(h["testable_b"]) for h in rec["report_halves"])]
        else:
            cells += ["", "", ""]
        L.append("| " + " | ".join(cells) + " |")
    L += ["", "teaching per type (reading 4; the untaught odour's decreases are recorded, not judged):", ""]
    for name, c in res["h4"]["combos"].items():
        for t, tc in c["teach"].items():
            L.append(f"- {name} {t}: taught odour {tc['odour']}, order {tc['order']}, arm {tc['arm']}, decreased "
                     f"{tc['n_decreased']}/{len(res['spec'].teach_seeds)}, teachable {tc['teachable']}, untaught "
                     f"decreased {tc['untaught_n_decreased']}")
    L += ["", f"selection: {res['h4'].get('selection')}", ""] + [f"- {n}" for n in res["notes"]]
    return "\n".join(L) + "\n"


def main(argv=None, spec: H4Spec | None = None, summary_spec: H4Spec = SPEC, require_root: bool = True,
         pairs: list | None = None, pools: dict | None = None) -> int:
    """pairs / pools replace the E0 pair list and the core-type pools in the tests only: the synthetic connectome has
    neither the pool's receptor types nor one type per pool. Every refusal comes before any measurement or write."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default="data/malecns.npz")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary", default="results/summary/m0d.json")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--pairs", type=int, default=None)          # verification only: the first N pairs of each axis
    a = ap.parse_args(argv)
    if a.pairs is not None and a.pairs < 1:                     # 0 would run every pair, a negative N drop the last
        return refuse(f"--pairs must be at least 1, got {a.pairs}")
    if a.smoke and a.pairs is None:
        a.pairs = 1

    if require_root and Path.cwd().resolve() != ROOT:
        return refuse(f"not at the repository root {ROOT}")
    spec = spec or (smoke_spec(SPEC) if a.smoke else SPEC)
    out = Path(a.out or ("results/m0d/h4-smoke" if a.smoke else "results/m0d/h4"))
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    git = git_state(files=HASHED_FILES)
    if git["dirty_hashed"] and not a.allow_dirty:
        return refuse(f"hashed files are dirty {git['dirty_hashed']} (commit them, or --allow-dirty)")
    h3_code = code_key(a.npz, files=H3_MEASURE_FILES)
    code = code_key(a.npz, files=MEASURE_FILES)
    manifest = code_key(a.npz, files=HASHED_FILES)
    npz_sha = code["files"]["npz:" + Path(a.npz).name]
    if npz_sha != spec.h3.connectome_sha256:
        return refuse(f"{a.npz} is not the declared connectome ({npz_sha[:12]})")
    try:
        summary = json.loads(Path(a.summary).read_text())
        block = summary[spec.h3_block]
    except (OSError, ValueError, KeyError) as e:
        return refuse(f"no usable block {spec.h3_block} in {a.summary}: {e!r}")

    conn = Connectome.load(a.npz)
    pops = Populations.from_connectome(conn)
    comps = compartments(conn, pops, Params().core_frac)
    types_of = lambda dan: sorted({str(np.asarray(conn.type).astype(str)[int(i)]) for i in comps[dan].core})
    core = dict(A=types_of(spec.h3.punish_type), P=types_of(spec.h3.reward_type))
    if block.get("pools") != core:
        return refuse(f"the core pools {core} differ from block {spec.h3_block}'s {block.get('pools')}")
    if block.get("measure_key") != h3_code["key"]:
        return refuse(f"H.3's measurement key is {h3_code['key'][:12]} but block {spec.h3_block} was measured with "
                      f"{str(block.get('measure_key'))[:12]}: the reactivity would not be H.3's guard")
    pools = pools or core
    pairs = even_pairs(pops) if pairs is None else pairs
    digest = pairs_digest(pairs)
    if digest != spec.pairs_digest or sum(p["axis"] == "a" for p in pairs) != spec.n_pairs_a \
            or sum(p["axis"] == "b" for p in pairs) != spec.n_pairs_b:
        return refuse(f"the pair list differs from the declared one ({digest})")
    if a.pairs:
        pairs = [p for ax in ("a", "b") for p in [q for q in pairs if q["axis"] == ax][:a.pairs]]
    try:
        combos, guards = adopted(summary, spec)                 # last: it may restore C3's threshold file
    except ValueError as e:
        return refuse(str(e))

    t0 = time.time()
    odors = make_odors(pops, spec.h3.reference)
    ctx = Context(spec=spec, combos=combos, pools=pools, probe_seeds=[int(s) for o in odors for s in o["seeds"]],
                  expected=[pair_key(p) for p in pairs], h3_guard=guards, log=lambda s: print(s, flush=True))
    cache = MeasureCache(out / "cache", code, run_id)
    h3_cache = MeasureCache(H3_CACHE, h3_code, run_id)
    print(f"run {run_id}: pools {pools}, {len(pairs)} pairs, H.3 measurement key {h3_code['key'][:12]} "
          f"(= block {spec.h3_block}'s), H.4 key {code['key'][:12]}", flush=True)

    res = dict(run_id=run_id, smoke=a.smoke, argv=list(sys.argv[1:] if argv is None else argv), git=git,
               code=manifest, measure_key=code["key"], h3_measure_key=h3_code["key"], spec=spec, pools=pools,
               pairs_digest=digest, n_pairs=len(pairs), h3_run_id=block.get("run_id"), notes=list(spec.notes))
    with FlyPool(a.npz, Params(), [{} for _ in range(a.workers)], workers=a.workers, punish_type=spec.h3.punish_type,
                 reward_type=spec.h3.reward_type, timeout_s=POOL_TIMEOUT_S) as pool:
        h3m = PoolMeasurer(pool, spec.h3, {"reference": odors}, None, h3_cache, None)
        m = H4Measurer(pool, spec, pairs, pools, cache, h3m)
        res["h4"] = run_h4(m, ctx)
        guard_params = [Params()] + [p for p in m.params_seen if p != Params()]
    res["wall_s"] = time.time() - t0
    res["cache"] = dict(hits=cache.hits + h3_cache.hits, misses=cache.misses + h3_cache.misses)
    used = sorted(set(cache.used) | set(h3_cache.used))
    res["artifacts"] = {p: sha256_file(p) for p in used if Path(p).exists()}
    report = write_json(out / "runs" / f"{run_id}.json", res, guard_params)
    write_bytes(out / "runs" / f"{run_id}.md", md_report(res).encode(), guard_params)
    print(f"wrote {report}: outcome {res['h4']['outcome']} in {res['wall_s'] / 60:.1f} min", flush=True)
    blockers = dict(smoke=a.smoke, pairs=a.pairs is not None, dirty=bool(git["dirty_hashed"]), spec=spec != summary_spec,
                    invalid=res["h4"]["outcome"] == INVALID, incomplete=res["h4"]["outcome"] == STOP_MULTI_TYPE)
    if any(blockers.values()):
        print(f"summary not written: {[k for k, v in blockers.items() if v]}", flush=True)
    else:
        replace_summary_block(a.summary, "h4", dict(res, report=str(report), report_sha256=sha256_file(report)),
                              guard_params)
        print(f"replaced block 'h4' of {a.summary}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
