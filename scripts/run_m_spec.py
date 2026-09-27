#!/usr/bin/env python3
"""Spec M.10.2 (+ M.10.1, M.10.7): the offline specificity pre-check. No engine, no pool: the connectome and H.4a.8's
pinned ceiling file (its C3 rows' fx / fy on the even (b) 21 pairs, act seeds 500-507, sha256 of L.11.1).

    uv run python scripts/run_m_spec.py                          # minutes; commit block spec_check before stage 0
    uv run python scripts/run_m_spec.py --smoke --allow-dirty    # the smoke candidates (PAM08 PAM10 / PPL103 PPL105)

Run it from the repository root. Refusals (exit 2), in order: another directory, --out outside results/m0d/m/, dirty
hashed files (unless --allow-dirty), a --summary already holding a later block (outside --smoke), another connectome,
no usable C3 record in block h3 / h4 of results/summary/m0d.json, a ceiling file other than the pinned one (--ceiling
replaces the path, never the hash) or not measured on C3's engine / readout / seeds, candidates other than the declared
21 (m_cands.candidates).
Computed per candidate (all 21, incumbents included): the KC weights onto its core cells (m_cands.cell_weights, edges >=
SPEC.k.min_weight), S_c per even (b) pair (k_metrics.pair_metrics' S), the median and quartiles over the pairs, the lobe
composition of the core's KC input (k_metrics.lobe_masks) and the input itself; then m_cands.spec_check (strictly above
the arm incumbent's median; STOP_NO_SPECIFICITY iff no candidate in either arm). Recorded per candidate: representative
DAN, member DANs, core cells and their sha256, core types, DAN cell count and core w_mbon distribution — the committed
block pins the candidates for every later stage (M.10.7).
Report: <out>/runs/<run id>-spec.{json,md}. Block "spec_check" of results/summary/m_readout.json is written unless
--smoke, dirty hashed files or a spec other than the declared one. Exit 0 once computed (either outcome).
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np

from flymon.brain import m_store
from flymon.brain.circuits import Populations, compartments
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.h3_store import ROOT, sha256_file
from flymon.brain.j_runner import params_json
from flymon.brain.k_metrics import lobe_masks
from flymon.brain.l_measure import load_even
from flymon.brain.m_cands import STOP_NO_SPECIFICITY, candidates, cell_weights, lobe_shares, overlap, pair_s, spec_check
from flymon.brain.m_cli import (code_keys, git_state, later_blocks, load_c3_record, out_allowed, refuse, run_id,
                                write_block)
from flymon.brain.m_measure import HASHED_FILES
from flymon.brain.m_rules import sentence
from flymon.brain.m_spec import SPEC, MSpec, smoke


def _stats(x) -> dict:
    x = np.asarray(x, float)
    return dict(min=float(x.min()), median=float(np.median(x)), max=float(x.max())) if len(x) else {}


def candidate_record(c: dict, arm: str, w, even: list, lobes: dict) -> dict:
    """One candidate's recorded values: identity (M.10.7), weights and S per even (b) pair (M.10.2)."""
    s = [pair_s(e["fx"], e["fy"], w) for e in even]
    return dict(name=str(c["name"]), arm=arm, family=str(c["family"]), dans=[str(d) for d in c["dans"]],
                cells=[int(i) for i in c["cells"]], n_cells=len(c["cells"]), digest=c["digest"], types=list(c["types"]),
                n_dan_cells=int(c["n_dan_cells"]),
                w_mbon=[float(v) for v in c["w_mbon"]], w_mbon_stats=_stats(c["w_mbon"]), kc_input=float(np.sum(w)),
                lobes=lobe_shares(w, lobes), S=[float(v) for v in s], median=float(np.median(s)),
                quartiles=[float(q) for q in np.percentile(s, [25, 75])])


def main(argv=None, spec: MSpec | None = None, summary_spec: MSpec = SPEC, require_root: bool = True) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default="data/malecns.npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary", default=m_store.SUMMARY)
    ap.add_argument("--m0d", default=None)
    ap.add_argument("--ceiling", default=None)             # another path to the pinned ceiling file (hash checked)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args(argv)
    if require_root and Path.cwd().resolve() != ROOT:
        return refuse(f"not at the repository root {ROOT}")
    spec = spec or (smoke(SPEC) if a.smoke else SPEC)
    if a.ceiling:
        spec = dataclasses.replace(spec, l=dataclasses.replace(spec.l, ceiling_path=a.ceiling))
    out = Path(a.out or ("results/m0d/m/smoke" if a.smoke else "results/m0d/m/run"))
    if not out_allowed(out):
        return refuse(f"--out {out} is not under {m_store.ALLOWED_DIR} of the repository root")
    rid = run_id()
    git = git_state(files=HASHED_FILES)
    if git["dirty_hashed"] and not a.allow_dirty:
        return refuse(f"hashed files are dirty {git['dirty_hashed']} (commit them, or --allow-dirty)")
    why = None if a.smoke else later_blocks(a.summary, "spec_check")
    if why:
        return refuse(why)
    if not Path(a.npz).exists():
        return refuse(f"{a.npz} does not exist (the declared connectome)")
    try:
        code, manifest = code_keys(a.npz)
    except SystemExit as e:                                  # code_keys named the missing hashed files
        return int(e.code or 2)
    npz_sha = code["files"]["npz:" + Path(a.npz).name]
    if npz_sha != spec.j.h4.h3.connectome_sha256:
        return refuse(f"{a.npz} is not the declared connectome ({npz_sha[:12]})")
    m0d_path = a.m0d or spec.m0d_path
    try:
        c3, readout, _, _ = load_c3_record(m0d_path, spec.l)
    except ValueError as e:
        return refuse(str(e))
    c3_json = params_json(c3)
    try:
        even = load_even(spec.l, c3_json)
    except (OSError, ValueError, KeyError) as e:
        return refuse(f"the pinned ceiling file is not usable (L.11.1 pins it by sha256): {e}")
    t0 = time.time()
    conn = Connectome.load(a.npz)
    pops = Populations.from_connectome(conn)
    try:
        cands = candidates(conn.type, compartments(conn, pops, Params().core_frac), spec)
    except ValueError as e:
        return refuse(f"the candidates are not the declared ones: {e}")
    lobes = lobe_masks(conn, pops)
    recs = {arm: [candidate_record(c, arm, cell_weights(conn, pops, c["cells"], spec.k.min_weight), even, lobes)
                  for c in cands[arm]] for arm in ("reward", "punish")}
    chk = spec_check({r["name"]: r["S"] for arm in recs for r in recs[arm]}, spec)
    inc = {"reward": spec.incumbent_reward, "punish": spec.incumbent_punish}
    by = {r["name"]: r for arm in recs for r in recs[arm]}
    other = {"reward": "punish", "punish": "reward"}
    for arm in recs:                                         # reading 2: overlap with the other arm's incumbent
        for r in recs[arm]:
            r["overlaps_other_incumbent"] = overlap(r, by[inc[other[arm]]])
    res = dict(run_id=rid, smoke=a.smoke, argv=list(sys.argv[1:] if argv is None else argv), git=git, code=manifest,
               measure_key=code["key"], spec=spec, c3=c3_json, readout=readout,
               inputs=dict(ceiling=dict(path=spec.l.ceiling_path, sha256=spec.l.ceiling_sha256),
                           m0d=dict(path=m0d_path, sha256=sha256_file(m0d_path))),
               min_weight=int(spec.k.min_weight), n_pairs=len(even), pairs=[list(e["key"]) for e in even],
               incumbents=inc, candidates=recs, **chk,
               incumbent_values={arm: dict(name=n, median=by[n]["median"], quartiles=by[n]["quartiles"],
                                           kc_input=by[n]["kc_input"], lobes=by[n]["lobes"], n_cells=by[n]["n_cells"],
                                           digest=by[n]["digest"]) for arm, n in inc.items()},
               sentence=sentence(STOP_NO_SPECIFICITY, {}) if chk["outcome"] == STOP_NO_SPECIFICITY else None)
    res["wall_s"] = time.time() - t0
    gp = [Params()] + ([c3] if c3 != Params() else [])
    report = m_store.write_json(out / "runs" / f"{rid}-spec.json", res, gp)
    lines = [f"# M.10.2 specificity pre-check {rid}\n\n", f"**outcome: {chk['outcome']}**\n\n",
             f"{len(even)} even (b) pairs; incumbent medians reward {inc['reward']} "
             f"{chk['incumbent_median']['reward']:.6g}, punish {inc['punish']} "
             f"{chk['incumbent_median']['punish']:.6g}\n\n",
             "| arm | candidate | DANs | core cells | median S | q25 | q75 | KC input | passes |\n"
             "|---|---|---|---|---|---|---|---|---|\n"]
    for arm in ("reward", "punish"):
        for r in recs[arm]:
            lines.append(f"| {arm} | {r['name']} | {' '.join(r['dans'])} | {r['n_cells']} | {r['median']:.6g} | "
                         f"{r['quartiles'][0]:.6g} | {r['quartiles'][1]:.6g} | {r['kc_input']:.6g} | "
                         f"{'incumbent' if r['name'] == inc[arm] else r['name'] in chk['passing'][arm]} |\n")
    if res["sentence"]:
        lines.append(f"\n{res['sentence']}\n")
    m_store.write_bytes(out / "runs" / f"{rid}-spec.md", "".join(lines).encode(), gp)
    print(f"wrote {report}: {chk['outcome']} (passing {chk['passing']})", flush=True)
    if res["sentence"]:
        print(res["sentence"], flush=True)
    blockers = dict(smoke=a.smoke, dirty=bool(git["dirty_hashed"]), spec=spec != summary_spec)
    if any(blockers.values()):
        print(f"summary not written: {[k for k, v in blockers.items() if v]}", flush=True)
    else:
        write_block(a.summary, "spec_check", res, report, gp)
        print(f"wrote {a.summary} (block spec_check)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
