#!/usr/bin/env python3
"""Spec M.10.1 (+ M.10.7, plan readings 4 and 6-9), stage 0: H.3's reference presentations and rest windows with the
read counts of every MBON cell (m_jobs.reference_cells_job / rest_cells_job, through MMeasurer), then

- the self-check: C3's single readout (MBON13 / MBON05 type sums) re-measured this way reproduces block h4's C3
  reactivity (every field) and z (ddof SPEC.j.h4.z_ddof) bit for bit — else exit 5, no block;
- the guard on core-cell sums: H.4's reactivity rule and z = (mean, sd) valid only when finite with sd > 0, for both
  incumbent populations (PPL105 core, PAM08 core: the fixed-arm readouts of stages 1-2) and every candidate that passed
  block spec_check (--smoke: the smoke candidates); an incumbent failing -> STOP_INCUMBENT (exit 5, the user decides);
  an arm with no passing candidate is fixed to its incumbent; STOP_NO_CANDIDATE only when both arms are empty;
- records only: per-cell reactivity and variance shares, core w_mbon distribution, DAN cell counts, and the teach record
  of every guard-passing candidate (h4_jobs.teach_job unchanged on its core types, h4_rules.teach_choice per type).

    uv run python scripts/run_m_stage0.py                                        # ~1 h (resumable: rerun it)
    uv run python scripts/run_m_stage0.py --smoke --allow-dirty --workers 4 --summary results/m0d/m/smoke/m_readout.json

Run it from the repository root. Refusals (exit 2) before the pool starts, in order: another directory, --out outside
results/m0d/m/, dirty hashed files (unless --allow-dirty), a summary that is not git-tracked and clean at HEAD (outside
--smoke; --smoke reads only a smoke summary under results/m0d/m/) or has no block spec_check, a spec_check outcome other
than SPEC_GO, another connectome, a spec_check block produced under other code, a later block in the summary (outside
--smoke), an m0d.json other than the one spec_check ran on (sha256), no usable C3 record in it, candidates whose names
or core-cell sha256 differ from block spec_check's (M.10.7: that block pins them).
Report: <out>/runs/<run id>-stage0.{json,md}. Block "stage0" is written when the status is "done" or STOP_NO_CANDIDATE,
outside --smoke only with clean hashed files and the declared spec; a --smoke run writes it into its smoke summary (the
L convention for chaining smoke stages). Exit codes: 0 done (incl. STOP_NO_CANDIDATE), 2 refused, 5 self-check mismatch
or STOP_INCUMBENT (recorded; stop and ask the user).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from flymon.brain import m_store
from flymon.brain.circuits import Populations, compartments
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.fly_pool import FlyPool
from flymon.brain.h3_store import ROOT, MeasureCache, sha256_file
from flymon.brain.h4_rules import teach_choice
from flymon.brain.j_runner import params_json
from flymon.brain.j_setup import build
from flymon.brain.m_cands import SPEC_GO, candidates
from flymon.brain.m_cli import (POOL_TIMEOUT_S, check_committed, code_keys, git_state, guard_params, later_blocks,
                                load_c3_record, other_code, out_allowed, read_previous, refuse, run_id, same_code,
                                write_block)
from flymon.brain.m_measure import HASHED_FILES, MMeasurer, cell_records, group_react, group_rows, group_z
from flymon.brain.m_rules import STOP_INCUMBENT, STOP_NO_CANDIDATE, sentence
from flymon.brain.m_spec import SPEC, MSpec, smoke

DONE = "done"                   # stage 1 requires block stage0's status == "done"
MISMATCH = "mismatch"
ARM_OF = {"reward": "reward_only", "punish": "punish_only"}


def _norm(x):
    return json.loads(json.dumps(x))


def self_check(ref, rest, cells, conn, readout: dict, react_h4: dict, z_h4: dict, h4spec) -> dict:
    """C3's single readout re-measured through the per-cell path: reactivity (every field) and (mean, sd) of the type
    sums against block h4's record, compared exactly."""
    types = np.asarray(conn.type).astype(str)
    have = set(int(c) for c in cells)
    out, bad = {}, []
    for k, t in readout.items():
        tc = [int(i) for i in np.flatnonzero(types == t)]
        if not tc or not set(tc) <= have:
            out[t] = dict(cells=tc, note="type cells missing from the measured MBON cells")
            bad.append(t)
            continue
        react = group_react(ref, rest, cells, {t: tc}, h4spec)[t]
        x = np.array([r["types"][t] for r in group_rows(ref, cells, {t: tc})], float)
        z = [float(x.mean()), float(x.std(ddof=h4spec.z_ddof))]
        want_r, want_z = react_h4.get(t), list(z_h4.get(k) or [])
        ok = _norm(react) == _norm(want_r) and z == [float(v) for v in want_z]
        out[t] = dict(key=k, n_cells=len(tc), react=react, want_react=want_r, z=z, want_z=want_z, ok=ok)
        if not ok:
            bad.append(t)
    return dict(ok=not bad, mismatched=bad, types=out,
                note="C3's MBON13 / MBON05 type sums re-measured on the per-cell path against block h4 (M.10.1)")


def group_record(c: dict, arm: str, ref, rest, cells, h4spec, incumbent: bool) -> dict:
    g = [int(i) for i in c["cells"]]
    react = group_react(ref, rest, cells, {c["name"]: g}, h4spec)[c["name"]]
    z = group_z(ref, cells, g, h4spec.z_ddof)
    w = np.asarray(c["w_mbon"], float)
    return dict(name=str(c["name"]), arm=arm, incumbent=incumbent, dans=[str(d) for d in c["dans"]],
                cells=g, n_cells=len(g), digest=c["digest"], types=list(c["types"]), react=react,
                z=None if z is None else list(z),
                passes=bool(react["passes"] and z is not None),
                cell_records=cell_records(ref, rest, cells, g, h4spec),
                w_mbon=[float(v) for v in w], w_mbon_stats=dict(min=float(w.min()), median=float(np.median(w)),
                                                              max=float(w.max())) if len(w) else {},
                n_dan_cells=int(c["n_dan_cells"]))


def teach_record(m, c3, g: dict, spec) -> dict:
    """Reading 9: teach_job unchanged on the candidate's core types, its arm driven by the candidate, the other arm's
    DAN the incumbent; teach_choice per core type (a ValueError is recorded as its message). A record, not a gate."""
    arm = ARM_OF[g["arm"]]
    reward = g["name"] if g["arm"] == "reward" else spec.incumbent_reward
    punish = g["name"] if g["arm"] == "punish" else spec.incumbent_punish
    rows = m.teach(c3, g["types"], arm, punish, reward)
    out = {}
    for t in g["types"]:
        try:
            out[t] = teach_choice(rows, t, arm, spec.j.h4)
        except ValueError as e:
            out[t] = str(e)
    return dict(arm=arm, reward_type=reward, punish_type=punish, types=out)


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
    doc, why = read_previous(a.summary, ["spec_check"], a.smoke, check_committed)
    if why:
        return refuse(why)
    sc = doc["spec_check"]
    if sc.get("outcome") != SPEC_GO:
        return refuse(f"block spec_check's outcome is {sc.get('outcome')}: stage 0 runs only after {SPEC_GO} (M.10.2)")
    if not Path(a.npz).exists():
        return refuse(f"{a.npz} does not exist (the declared connectome)")
    try:
        code, manifest = code_keys(a.npz)
    except SystemExit as e:                                  # code_keys named the missing hashed files
        return int(e.code or 2)
    npz_sha = code["files"]["npz:" + Path(a.npz).name]
    if npz_sha != spec.j.h4.h3.connectome_sha256:
        return refuse(f"{a.npz} is not the declared connectome ({npz_sha[:12]})")
    why = other_code(doc, ["spec_check"], code["key"], manifest, same_code)
    if why:
        return refuse(why)
    why = None if a.smoke else later_blocks(a.summary, "stage0", doc)
    if why:
        return refuse(why)
    m0d_path = a.m0d or spec.m0d_path
    want = ((sc.get("inputs") or {}).get("m0d") or {}).get("sha256")
    try:
        got = sha256_file(m0d_path)
    except OSError as e:
        return refuse(f"the m0d summary {m0d_path} is not readable: {e}")
    if not want or got != want:
        return refuse(f"the m0d summary {m0d_path} (sha256 {got[:12]}) is not the one block spec_check ran on "
                      f"(inputs.m0d.sha256 {str(want)[:12]})")
    try:
        c3, readout, z_c3, pools_h4 = load_c3_record(m0d_path, spec.l)
        react_h4 = json.loads(Path(m0d_path).read_text())["h4"]["h4"]["combos"][spec.j.c3_name]["reactivity"]
    except (ValueError, KeyError, TypeError) as e:
        return refuse(str(e) if isinstance(e, ValueError) else f"block h4 of {m0d_path} has no C3 reactivity: {e!r}")
    conn = Connectome.load(a.npz)
    pops = Populations.from_connectome(conn)
    try:
        cands = candidates(conn.type, compartments(conn, pops, Params().core_frac), spec)
    except ValueError as e:
        return refuse(f"the candidates are not the declared ones: {e}")
    pinned = {arm: [(c.get("name"), c.get("digest")) for c in (sc.get("candidates") or {}).get(arm, [])]
              for arm in ("reward", "punish")}
    now = {arm: [(c["name"], c["digest"]) for c in cands[arm]] for arm in ("reward", "punish")}
    if now != pinned:
        diff = {arm: sorted(set(now[arm]) ^ set(pinned[arm])) for arm in now if now[arm] != pinned[arm]}
        return refuse(f"the candidates' names or core-cell digests differ from block spec_check's (M.10.7 pins them): "
                      f"{diff}")
    inc = {"reward": spec.incumbent_reward, "punish": spec.incumbent_punish}
    names = {"reward": spec.reward_candidates, "punish": spec.punish_candidates}
    scan = ({arm: [n for n in names[arm] if n != inc[arm]] for arm in inc} if a.smoke
            else {arm: list((sc.get("passing") or {}).get(arm, [])) for arm in inc})
    by = {arm: {c["name"]: c for c in cands[arm]} for arm in inc}
    unknown = {arm: [n for n in scan[arm] if n not in by[arm]] for arm in inc}
    if any(unknown.values()):
        return refuse(f"block spec_check passes candidates this spec does not hold: {unknown}")

    t0 = time.time()
    s = build(a.npz, spec.j, out, log=lambda x: print(x, flush=True))
    odors = s["odors"]
    seeds = [int(x) for o in odors for x in o["seeds"]]
    cells = sorted(int(i) for i in pops.mbon)
    print(f"run {rid}: M key {code['key'][:12]}, {len(odors)} odours, {len(cells)} MBON cells, scan {scan}", flush=True)
    res = dict(run_id=rid, smoke=a.smoke, argv=list(sys.argv[1:] if argv is None else argv), git=git, code=manifest,
               measure_key=code["key"], spec=spec, spec_check_run_id=sc.get("run_id"), c3=params_json(c3),
               readout=readout, z_c3=z_c3, pools=pools_h4,
               inputs=dict(m0d=dict(path=m0d_path, sha256=got)), n_presentations=sum(len(o["seeds"]) for o in odors),
               cells=cells, incumbents=inc, scanned=scan)
    status, seen = DONE, []
    with FlyPool(a.npz, Params(), [{} for _ in range(a.workers)], workers=a.workers,
                 punish_type=spec.j.h4.h3.punish_type, reward_type=spec.j.h4.h3.reward_type,
                 timeout_s=POOL_TIMEOUT_S) as pool:
        cache = MeasureCache(out / "cache", code, rid)
        m = MMeasurer(pool, spec, cache)
        seen.append(m)
        ref = m.reference(c3, odors, cells)
        rest = m.rest(c3, seeds, cells)
        h4 = spec.j.h4
        res["self_check"] = self_check(ref, rest, cells, conn, readout, react_h4, z_c3, h4)
        if not res["self_check"]["ok"]:
            status = MISMATCH
        else:
            groups = [group_record(by[arm][inc[arm]], arm, ref, rest, cells, h4, True) for arm in ("punish", "reward")]
            groups += [group_record(by[arm][n], arm, ref, rest, cells, h4, False) for arm in ("reward", "punish")
                       for n in scan[arm]]
            res["groups"] = groups
            if not all(g["passes"] for g in groups if g["incumbent"]):
                status = STOP_INCUMBENT
            else:
                arms = {arm: [g["name"] for g in groups if g["arm"] == arm and not g["incumbent"] and g["passes"]]
                        for arm in ("reward", "punish")}
                res["arms"] = arms
                res["fixed"] = {arm: inc[arm] for arm in arms if not arms[arm]}     # an empty arm: its incumbent
                if not arms["reward"] and not arms["punish"]:
                    status = STOP_NO_CANDIDATE
                for g in groups:
                    if not g["incumbent"] and g["passes"]:
                        print(f"teach record {g['name']} ({g['types']})", flush=True)
                        g["teach"] = teach_record(m, c3, g, spec)
    gp = guard_params(Params(), seen)
    res["status"] = status
    if status in (STOP_INCUMBENT, STOP_NO_CANDIDATE):
        res["sentence"] = sentence(status, {})
    res["wall_s"] = time.time() - t0
    res["cache"] = dict(hits=cache.hits, misses=cache.misses)
    res["artifacts"] = {p: sha256_file(p) for p in sorted(cache.used) if Path(p).exists()}
    report = m_store.write_json(out / "runs" / f"{rid}-stage0.json", res, gp)
    ch = res["self_check"]
    lines = [f"# M.10.1 stage 0 {rid}\n\n", f"**status: {status}**\n\n",
             f"- self-check (C3 single readout vs block h4): {ch['ok']} {ch['mismatched']}\n"]
    if res.get("groups"):
        lines.append("\n| arm | group | incumbent | core cells | median Δ | zero share | z | passes | max var share "
                     "|\n|---|---|---|---|---|---|---|---|---|\n")
        for g in res["groups"]:
            lines.append(f"| {g['arm']} | {g['name']} | {g['incumbent']} | {g['n_cells']} | "
                         f"{g['react']['median_delta']:.6g} | {g['react']['zero_share']:.3g} | {g['z']} | "
                         f"{g['passes']} | {g['cell_records']['max_var_share']:.3g} |\n")
    if res.get("arms") is not None:
        lines.append(f"\narms: {res['arms']}; fixed to the incumbent: {res['fixed']}\n")
    if res.get("sentence"):
        lines.append(f"\n{res['sentence']}\n")
    m_store.write_bytes(out / "runs" / f"{rid}-stage0.md", "".join(lines).encode(), gp)
    print(f"wrote {report}: {status} in {res['wall_s'] / 60:.1f} min", flush=True)
    if status == MISMATCH:
        print(f"self-check DIFFERS {ch['mismatched']}: stop and ask the user", flush=True)
        return 5
    if status == STOP_INCUMBENT:
        print(f"{STOP_INCUMBENT}: {res['sentence']}", flush=True)
        return 5
    if status == STOP_NO_CANDIDATE:
        print(res["sentence"], flush=True)
    blockers = dict(dirty=bool(git["dirty_hashed"]) and not a.smoke, spec=spec != summary_spec and not a.smoke)
    if any(blockers.values()):
        print(f"summary not written: {[k for k, v in blockers.items() if v]}", flush=True)
    else:
        write_block(a.summary, "stage0", res, report, gp)
        print(f"wrote {a.summary} (block stage0{', smoke' if a.smoke else ''})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
