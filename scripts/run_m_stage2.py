#!/usr/bin/env python3
"""Spec M.10.3 (+ M.10.5, M.10.7, plan readings 2, 12 and 16), stage 2: the joint oracle on the top-2 x 2 combinations.

- The combinations: block stage1's top per arm (m_rules.combos: top_r x top_p in top_r-major order minus the pairs whose
  core cells overlap, recorded as `excluded`).
- pre_job once per even pair (H.4's (a) 18 + (b) 21 in H.4 order; shared by every combination), then edit_job per pair
  for each combination with both arms' POPULATION readouts ({A: punish group, P: reward group}, z from block stage0,
  reward_type / punish_type = the groups' names); l_rules.pair_rows_stats (pair_stats per row) and
  h4_formula.arm_aggregate(stats, naive_max, t_b_min, f_a_min), plus m_median = the median of the defined m values.
- The choice: m_rules.choose (bar first: testable_b >= 11 and F_a >= 2, then testable_b, F_a, m_median); the winner is
  the first. Gate: STAGE2_GO iff the winner meets the M2 bar on the even pairs (m_rules.meets_bar), else STOP_NO_GAIN
  with M.10.5's sentence. No combination left after the overlap exclusion (M.10.8): STOP_NO_GAIN before the pool,
  "겹치지 않는 조합이 없었다", the block written (no winner, no odd record), exit 0.
- The odd record: the winner on k_pairs.odd_pairs (K.8.1's 20 (b) pairs), testable per pair and the count, whatever the
  gate says (a record, not a judgement).

    uv run python scripts/run_m_stage2.py                                        # ~2 h (resumable: rerun it)
    uv run python scripts/run_m_stage2.py --smoke --allow-dirty --workers 4 --summary results/m0d/m/smoke/m_readout.json

Run it from the repository root. Refusals (exit 2) before the pool starts, in order: another directory, --out outside
results/m0d/m/, dirty hashed files (unless --allow-dirty), a summary whose blocks spec_check, stage0 and oc are not
committed (block level: m_cli.blocks_committed; outside --smoke; --smoke reads only a smoke summary under results/m0d/m/)
or that lacks any of spec_check, stage0, oc, stage1 (each listed), no connectome file, any of them produced under other
code, a run-id chain break (stage0 -> spec_check, oc -> stage0, stage1 -> stage0 and oc), a later block in the summary
(outside --smoke), block stage1's status other than "done", another connectome, an m0d.json other than the one stage 0
ran on (sha256) or a C3 record differing from block stage0's, a top entry whose group is missing from block stage0, has
no z or whose cells / digest differ from block stage0's, another even pair list (H.4's digest), another odd pair list
(K's digest), other MBON cells than stage 0 measured.
Report: <out>/runs/<run id>-stage2.{json,md}. Block "stage2" (commit it before the judgement list), outside --smoke only
with clean hashed files and the declared spec; a --smoke run writes it into its smoke summary (the L convention for
chaining smoke stages). Exit codes: 0 done, 2 refused, 3 invalid oracle rows (recorded; no block).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from flymon.brain import m_store
from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.fly_pool import FlyPool
from flymon.brain.h3_store import ROOT, MeasureCache, sha256_file
from flymon.brain.h4_formula import arm_aggregate
from flymon.brain.h4_pairs import even_pairs, pair_key, pairs_digest
from flymon.brain.j_runner import params_json
from flymon.brain.k_pairs import odd_pairs
from flymon.brain.l_rules import pair_rows_stats
from flymon.brain.m_cands import cells_digest
from flymon.brain.m_cli import (POOL_TIMEOUT_S, blocks_committed, code_keys, git_state, guard_params, later_blocks,
                                load_c3_record, other_code, out_allowed, read_previous, refuse, run_id, run_id_chain,
                                same_code, stage0_inputs, write_block)
from flymon.brain.m_measure import HASHED_FILES, MMeasurer
from flymon.brain.m_rules import STAGE2_GO, STOP_NO_GAIN, choose, combos, meets_bar, sentence
from flymon.brain.m_spec import SPEC, MSpec, smoke

NEED = ("spec_check", "stage0", "oc", "stage1")
COMMITTED = ("spec_check", "stage0", "oc")   # reading 16: stage 2 needs stage1 present only
DONE = "done"
INVALID = "invalid"


def check_committed(summary, blocks) -> str | None:
    """The commit gate of reading 16 for stage 2: spec_check, stage0 and oc committed; stage1 may be uncommitted."""
    return blocks_committed(summary, COMMITTED)


def combo_name(r: str, p: str) -> str:
    return f"{r}|{p}"                                                        # predicted_joint's key (reward|punish)


def label(r: str, p: str) -> str:
    return f"{p}·{r}"                                                        # M.10.5's 〈PPLx·PAMy〉


def aggregate(rows: list, pairs: list, z: dict, h4) -> dict:
    """{reasons, pairs (per-pair stats), agg (arm_aggregate + m_median) or None when a row is invalid}."""
    st = pair_rows_stats(rows, z, [pair_key(p) for p in pairs], h4)
    if st["reasons"]:
        return dict(reasons=st["reasons"], pairs=st["pairs"], agg=None)
    agg = arm_aggregate(st["stats"], h4.naive_max, h4.t_b_min, h4.f_a_min)
    ms = [s["m"] for s in st["stats"].values() if s is not None and np.isfinite(s["m"])]
    agg["m_median"] = float(np.median(ms)) if ms else None                   # M.10.8: undefined -> null, ranked lowest
    return dict(reasons=[], pairs=st["pairs"], agg=agg)


def no_combination(a, rid, argv, git, code, manifest, spec, summary_spec, sc, s0, s1, doc, c3, readout, z_c3, m0d_path,
                   even, cells, tops, excluded, passing, out, t0) -> int:
    """M.10.8: every top-2 x 2 combination overlapped: STOP_NO_GAIN before any pool, the stage2 block written."""
    sent = sentence(STOP_NO_GAIN, dict(excluded=excluded))
    res = dict(run_id=rid, smoke=a.smoke, argv=list(sys.argv[1:] if argv is None else argv), git=git, code=manifest,
               measure_key=code["key"], spec=spec, spec_check_run_id=sc.get("run_id"), stage0_run_id=s0.get("run_id"),
               oc_run_id=doc["oc"].get("run_id"), stage1_run_id=s1.get("run_id"), c3=params_json(c3),
               readout=readout, z_c3=z_c3, inputs=dict(m0d=dict(path=m0d_path, sha256=sha256_file(m0d_path))),
               pairs=[list(pair_key(p)) for p in even], pairs_digest=pairs_digest(even),
               n_b=sum(p["axis"] == "b" for p in even),
               n_pairs=len(even), cells=cells, top={arm: [t["name"] for t in tops[arm]] for arm in tops},
               excluded=excluded, n_spec_passing=sum(len(v) for v in passing.values()),
               n_cands=len(summary_spec.reward_candidates) + len(summary_spec.punish_candidates) - 2,
               combos=[], ranked=[], winner=None, odd=None, gate=STOP_NO_GAIN, sentence=sent,
               note="no top-2 x 2 combination left after the core-overlap exclusion: closed before the pool (M.10.8)",
               status=DONE, wall_s=time.time() - t0, cache=dict(hits=0, misses=0), artifacts={})
    gp = guard_params(Params(), [])
    report = m_store.write_json(out / "runs" / f"{rid}-stage2.json", res, gp)
    lines = [f"# M.10.3 stage 2 {rid}\n\n", f"**status: {DONE}; gate {STOP_NO_GAIN}**\n\n",
             f"- excluded (overlapping cells): {excluded}\n", "- no combination left: no pool, no odd record\n",
             f"\n{sent}\n"]
    m_store.write_bytes(out / "runs" / f"{rid}-stage2.md", "".join(lines).encode(), gp)
    print(f"wrote {report}: {DONE}, gate {STOP_NO_GAIN} (no combination left)", flush=True)
    blockers = dict(dirty=bool(git["dirty_hashed"]) and not a.smoke, spec=spec != summary_spec and not a.smoke)
    if any(blockers.values()):
        print(f"summary not written: {[k for k, v in blockers.items() if v]}", flush=True)
    else:
        write_block(a.summary, "stage2", res, report, gp)
        print(f"wrote {a.summary} (block stage2{', smoke' if a.smoke else ''})", flush=True)
    print(f"{STOP_NO_GAIN}: {sent}", flush=True)
    return 0


def main(argv=None, spec: MSpec | None = None, summary_spec: MSpec = SPEC, require_root: bool = True) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default="data/malecns.npz")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary", default=m_store.SUMMARY)
    ap.add_argument("--m0d", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args(argv)
    if require_root and Path.cwd().resolve() != ROOT:
        return refuse(f"not at the repository root {ROOT}")
    spec = spec or (smoke(SPEC) if a.smoke else SPEC)
    out = Path(a.out or ("results/m0d/m/smoke" if a.smoke else "results/m0d/m/run"))
    if not out_allowed(out):
        return refuse(f"--out {out} is not under {m_store.ALLOWED_DIR} of the repository root")
    rid = run_id()
    git = git_state(files=HASHED_FILES)
    if git["dirty_hashed"] and not a.allow_dirty:
        return refuse(f"hashed files are dirty {git['dirty_hashed']} (commit them, or --allow-dirty)")
    doc, why = read_previous(a.summary, list(NEED), a.smoke, check_committed)
    if why:
        return refuse(why)
    if not Path(a.npz).exists():
        return refuse(f"{a.npz} does not exist (the declared connectome)")
    try:
        code, manifest = code_keys(a.npz)
    except SystemExit as e:                                  # code_keys named the missing hashed files
        return int(e.code or 2)
    why = (other_code(doc, list(NEED), code["key"], manifest["key"], same_code)
           or run_id_chain(doc, "stage0", {"spec_check_run_id": "spec_check"})
           or run_id_chain(doc, "oc", {"stage0_run_id": "stage0"})
           or run_id_chain(doc, "stage1", {"stage0_run_id": "stage0", "oc_run_id": "oc"})
           or (None if a.smoke else later_blocks(a.summary, "stage2", doc)))
    if why:
        return refuse(why)
    sc, s0, s1 = doc["spec_check"], doc["stage0"], doc["stage1"]
    if s1.get("status") != DONE:
        return refuse(f"block stage1's status is {s1.get('status')}: stage 2 runs only after stage 1 is {DONE!r}")
    npz_sha = code["files"].get("npz:" + Path(a.npz).name)
    if npz_sha != spec.j.h4.h3.connectome_sha256:
        return refuse(f"{a.npz} is not the declared connectome ({str(npz_sha)[:12]})")
    m0d_path = a.m0d or spec.m0d_path
    why = stage0_inputs(s0, m0d_path)
    if why:
        return refuse(why)
    try:
        c3, readout, z_c3, pools_h4 = load_c3_record(m0d_path, spec.l)
    except ValueError as e:
        return refuse(str(e))
    why = stage0_inputs(s0, m0d_path, c3=params_json(c3), readout=readout, z_c3=z_c3, pools=pools_h4)
    if why:
        return refuse(why)
    # ---- block stage1's top per arm, tied to block stage0's groups (cells, digest, z) -----------------------------
    groups = {g.get("name"): g for g in s0.get("groups") or []}
    tops = {arm: list((s1.get("top") or {}).get(arm) or []) for arm in ("reward", "punish")}
    bad = [t.get("name") for arm in tops for t in tops[arm]
           if t.get("name") not in groups or groups[t["name"]].get("z") is None
           or cells_digest(t.get("cells") or []) != t.get("digest")
           or [int(c) for c in t.get("cells") or []] != [int(c) for c in groups[t["name"]].get("cells") or []]
           or t.get("digest") != groups[t["name"]].get("digest")]
    if bad or not tops["reward"] or not tops["punish"]:
        return refuse(f"block stage1's top entries {bad or tops} are missing from block stage0, have no z, or their "
                      f"cells / digest differ from block stage0's")
    ok, excluded = combos(tops["reward"], tops["punish"])
    conn = Connectome.load(a.npz)
    pops = Populations.from_connectome(conn)
    all_even = even_pairs(pops)
    if pairs_digest(all_even) != spec.j.h4.pairs_digest:
        return refuse(f"the even pair list differs from H.4's ({pairs_digest(all_even)[:12]})")
    eb = [p for p in all_even if p["axis"] == "b"]
    ea = [p for p in all_even if p["axis"] == "a"]
    keep = {id(p) for p in (eb[:spec.n_even_b] if spec.n_even_b is not None else eb)
            + (ea[:spec.n_even_a] if spec.n_even_a is not None else ea)}
    even = [p for p in all_even if id(p) in keep]                                 # H.4 order, cut to smoke sizes
    all_odd = odd_pairs(pops)
    if pairs_digest(all_odd) != spec.k.odd_pairs_digest:
        return refuse(f"the odd pair list differs from K's ({pairs_digest(all_odd)[:12]})")
    odd = all_odd[:spec.n_odd] if spec.n_odd is not None else all_odd
    cells = sorted(int(i) for i in pops.mbon)
    if [int(c) for c in s0.get("cells") or []] != cells:
        return refuse("the MBON cells differ from the ones block stage0 measured")
    z = {n: tuple(float(v) for v in groups[n]["z"]) for n in groups if groups[n].get("z") is not None}
    cg = {n: [int(i) for i in groups[n]["cells"]] for n in groups}
    h4 = spec.j.h4
    passing = sc.get("passing") or {}

    t0 = time.time()
    if not ok:                                                   # M.10.8: every combination overlapped; no pool
        return no_combination(a, rid, argv, git, code, manifest, spec, summary_spec, sc, s0, s1, doc, c3, readout, z_c3,
                              m0d_path, even, cells, tops, excluded, passing, out, t0)
    print(f"run {rid}: M key {code['key'][:12]}, {len(even)} even pairs, {len(odd)} odd, combinations "
          f"{[combo_name(r['name'], p['name']) for r, p in ok]}, excluded {excluded}", flush=True)
    res = dict(run_id=rid, smoke=a.smoke, argv=list(sys.argv[1:] if argv is None else argv), git=git, code=manifest,
               measure_key=code["key"], spec=spec, spec_check_run_id=sc.get("run_id"), stage0_run_id=s0.get("run_id"),
               oc_run_id=doc["oc"].get("run_id"), stage1_run_id=s1.get("run_id"), c3=params_json(c3),
               readout=readout, z_c3=z_c3, inputs=dict(m0d=dict(path=m0d_path, sha256=sha256_file(m0d_path))),
               pairs=[list(pair_key(p)) for p in even], pairs_digest=pairs_digest(even),
               n_b=sum(p["axis"] == "b" for p in even),
               n_pairs=len(even), cells=cells, top={arm: [t["name"] for t in tops[arm]] for arm in tops},
               excluded=excluded, n_spec_passing=sum(len(v) for v in passing.values()),
               n_cands=len(summary_spec.reward_candidates) + len(summary_spec.punish_candidates) - 2)
    status, seen, measured = DONE, [], []
    with FlyPool(a.npz, Params(), [{} for _ in range(a.workers)], workers=a.workers,
                 punish_type=h4.h3.punish_type, reward_type=h4.h3.reward_type, timeout_s=POOL_TIMEOUT_S) as pool:
        cache = MeasureCache(out / "cache", code, rid)
        m = MMeasurer(pool, spec, cache)
        seen.append(m)
        pres = m.pre(c3, even, cells)                                                # once per pair, shared
        for r, p in ok:
            rn, pn = r["name"], p["name"]
            print(f"combination {combo_name(rn, pn)}", flush=True)
            gr, ro, zz = {rn: cg[rn], pn: cg[pn]}, {"A": pn, "P": rn}, {"A": z[pn], "P": z[rn]}
            rows = m.edit(c3, even, pres, cells, gr, ro, zz, reward_type=rn, punish_type=pn)
            ag = aggregate(rows, even, zz, h4)
            measured.append(dict(name=combo_name(rn, pn), label=label(rn, pn), reward=rn, punish=pn, readout=ro,
                                 z={k: list(v) for k, v in zz.items()}, cells={rn: gr[rn], pn: gr[pn]},
                                 digests={rn: groups[rn]["digest"], pn: groups[pn]["digest"]},
                                 edited=rows[0]["edited"] if rows else None, **ag))
        res["combos"] = measured
        if any(c["agg"] is None for c in measured) or not measured:
            status = INVALID
        else:
            ranked = choose({c["name"]: c["agg"] for c in measured}, spec.j)
            res["ranked"] = [n for n, _ in ranked]
            w = next(c for c in measured if c["name"] == ranked[0][0])
            gate = STAGE2_GO if meets_bar(w["agg"], spec.j) else STOP_NO_GAIN
            gs = lambda n: dict(n_dan_cells=groups[n].get("n_dan_cells"),
                                w_mbon_median=(groups[n].get("w_mbon_stats") or {}).get("median"))
            res["winner"] = dict(name=w["name"], label=w["label"], reward=w["reward"], punish=w["punish"],
                                 readout=w["readout"], z=w["z"], cells=w["cells"], digests=w["digests"],
                                 aggregate=w["agg"], testable_b=int(w["agg"]["testable_b"]),
                                 groups={w["reward"]: gs(w["reward"]), w["punish"]: gs(w["punish"])})
            res["gate"] = gate
            res["sentence"] = (sentence(STOP_NO_GAIN, dict(combo=w["label"], n=w["agg"]["testable_b"],
                                                           fa=w["agg"]["F_a"]))
                               if gate == STOP_NO_GAIN else None)
            # ---- the odd record (M.10.3): the winner on K.8.1's odd (b) pairs, whatever the gate says ---------------
            print(f"odd record {w['name']} on {len(odd)} odd (b) pairs", flush=True)
            opres = m.pre(c3, odd, cells)
            zz = {k: tuple(v) for k, v in w["z"].items()}
            orows = m.edit(c3, odd, opres, cells, w["cells"], w["readout"], zz, reward_type=w["reward"],
                           punish_type=w["punish"])
            ost = pair_rows_stats(orows, zz, [pair_key(p) for p in odd], h4)
            testable = [(ost["stats"].get(tuple(pair_key(p))) or {}).get("testable") for p in odd]
            res["odd"] = dict(pairs=[list(pair_key(p)) for p in odd], pairs_digest=pairs_digest(odd), n=len(odd),
                              testable=testable, n_testable=int(sum(bool(t) for t in testable)),
                              reasons=ost["reasons"], stats=ost["pairs"],
                              note="the winner on K.8.1's odd (b) pairs: a record, not a judgement (C3 2/20)")
    gp = guard_params(Params(), seen)
    res["status"] = status
    res["wall_s"] = time.time() - t0
    res["cache"] = dict(hits=cache.hits, misses=cache.misses)
    res["artifacts"] = {p: sha256_file(p) for p in sorted(cache.used) if Path(p).exists()}
    report = m_store.write_json(out / "runs" / f"{rid}-stage2.json", res, gp)
    lines = [f"# M.10.3 stage 2 {rid}\n\n", f"**status: {status}; gate {res.get('gate')}**\n\n",
             f"- excluded (overlapping cells): {excluded}\n",
             "\n| combination | testable_b | n_b | F_a | naive_a | m median | bar | reasons |\n"
             "|---|---|---|---|---|---|---|---|\n"]
    for c in measured:
        g = c["agg"] or {}
        lines.append(f"| {c['label']} | {g.get('testable_b')} | {g.get('n_b')} | {g.get('F_a')} | {g.get('naive_a')} | "
                     f"{g.get('m_median')} | {g.get('bar')} | {c['reasons']} |\n")
    if res.get("winner"):
        lines.append(f"\nwinner {res['winner']['label']}; odd record {res['odd']['n_testable']}/{res['odd']['n']}\n")
    if res.get("sentence"):
        lines.append(f"\n{res['sentence']}\n")
    m_store.write_bytes(out / "runs" / f"{rid}-stage2.md", "".join(lines).encode(), gp)
    print(f"wrote {report}: {status}, gate {res.get('gate')} in {res['wall_s'] / 60:.1f} min", flush=True)
    if status == INVALID:
        print(f"invalid oracle rows { {c['name']: c['reasons'] for c in measured} }: no block", flush=True)
        return 3
    blockers = dict(dirty=bool(git["dirty_hashed"]) and not a.smoke, spec=spec != summary_spec and not a.smoke)
    if any(blockers.values()):
        print(f"summary not written: {[k for k, v in blockers.items() if v]}", flush=True)
    else:
        write_block(a.summary, "stage2", res, report, gp)
        print(f"wrote {a.summary} (block stage2{', smoke' if a.smoke else ''})", flush=True)
    print(f"{res['gate']}: winner {res['winner']['label']} (testable_b {res['winner']['testable_b']}, F_a "
          f"{res['winner']['aggregate']['F_a']}), odd {res['odd']['n_testable']}/{res['odd']['n']}"
          + (f" — {res['sentence']}" if res["sentence"] else "; next: commit block stage2, then run_m_list.py"),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
