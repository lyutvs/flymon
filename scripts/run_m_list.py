#!/usr/bin/env python3
"""Spec M.10.4 (plan readings 13 and 16): the judgement list, written before stage 3 runs. No pool, no engine.

l_pairs.new_pairs regenerated with a_turns = n_turns (the generator and its order unchanged), then m_rules.judgement_set:
the first 21 (b) pairs with turn >= 4 in declared order and every (a) pair of their turns — (b) 21 / (a) 18 over turns
4, 5, 7-12, both digests pinned (SPEC.judge_b_digest / judge_a_digest). Turns 0-3 were measured by L's R0 smoke and are
excluded; turns 13 on stay unmeasured (the F v4 confirmation candidates, M.10.6).

    uv run python scripts/run_m_list.py                                          # seconds
    uv run python scripts/run_m_list.py --smoke --allow-dirty --summary results/m0d/m/smoke/m_readout.json

Run it from the repository root. Refusals (exit 2), in order: another directory, --out outside results/m0d/m/, dirty
hashed files (unless --allow-dirty), a summary that is not git-tracked and clean at HEAD (outside --smoke; --smoke reads
only a smoke summary under results/m0d/m/) or has no block stage2, block stage2's gate other than STAGE2_GO, a later
block in the summary (outside --smoke), no connectome file, block stage2 produced under other code, another connectome,
a judgement list whose digests are not the pinned ones, or whose sizes are not (b) 21 / (a) 18 (outside --smoke).
Report: <out>/runs/<run id>-list.{json,md}. Block "list" (commit it before stage 3), outside --smoke only with clean
hashed files and the declared spec; a --smoke run (turns >= 13, no digests: never the judgement set) writes it into its
smoke summary (the L convention for chaining smoke stages). Exit codes: 0 written, 2 refused.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from flymon.brain import m_store
from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.h3_store import ROOT
from flymon.brain.h4_pairs import pair_key
from flymon.brain.l_pairs import new_pairs
from flymon.brain.m_cli import (check_committed, code_keys, git_state, guard_params, later_blocks, other_code,
                                out_allowed, read_previous, refuse, run_id, same_code, write_block)
from flymon.brain.m_measure import HASHED_FILES
from flymon.brain.m_rules import STAGE2_GO, judgement_set
from flymon.brain.m_spec import SPEC, MSpec, smoke

NEED = ("stage2",)
EXPOSURE = ("turns 0-3 of the L set were measured by L's R0 smoke (naive; part of them by C3's oracle); excluded. "
            "Turns 13 on stay unmeasured (F v4 confirmation candidates, M.10.6).")


def diversity(b: list) -> dict:
    """The (b) pairs' distinct moves (x's part before " vs "), the top move and its share."""
    moves = [p["x"].split(" vs ")[0] for p in b]
    top = max(sorted(set(moves)), key=moves.count) if moves else None
    n = moves.count(top) if top else 0
    return dict(n_moves=len(set(moves)), top_move=top, top_count=n, top_move_share=n / len(moves) if moves else 0.0,
                top_move_text=f"{top} {n}/{len(moves)}" if top else "")


def main(argv=None, spec: MSpec | None = None, summary_spec: MSpec = SPEC, require_root: bool = True) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default="data/malecns.npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary", default=m_store.SUMMARY)
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
    s2 = doc["stage2"]
    if s2.get("gate") != STAGE2_GO:
        return refuse(f"block stage2's gate is {s2.get('gate')}, not {STAGE2_GO}: no judgement list follows")
    why = None if a.smoke else later_blocks(a.summary, "list", doc)
    if why:
        return refuse(why)
    if not Path(a.npz).exists():
        return refuse(f"{a.npz} does not exist (the declared connectome)")
    try:
        code, manifest = code_keys(a.npz)
    except SystemExit as e:                                  # code_keys named the missing hashed files
        return int(e.code or 2)
    why = other_code(doc, list(NEED), code["key"], manifest["key"], same_code)
    if why:
        return refuse(why)
    npz_sha = code["files"].get("npz:" + Path(a.npz).name)
    if npz_sha != spec.j.h4.h3.connectome_sha256:
        return refuse(f"{a.npz} is not the declared connectome ({str(npz_sha)[:12]})")
    pops = Populations.from_connectome(Connectome.load(a.npz))
    new_all = new_pairs(pops, dataclasses.replace(spec.l, a_turns=spec.l.n_turns))
    try:
        js = judgement_set(new_all, spec)
    except ValueError as e:
        return refuse(str(e))
    if not a.smoke and (len(js["b"]), len(js["a"])) != (spec.judge_n_b, spec.judge_n_a):
        return refuse(f"the judgement list has (b) {len(js['b'])} / (a) {len(js['a'])}, not the declared "
                      f"{spec.judge_n_b} / {spec.judge_n_a} (M.10.4)")
    turns = sorted({int(p["turn"]) for p in js["b"]})
    res = dict(run_id=rid, smoke=a.smoke, argv=list(sys.argv[1:] if argv is None else argv), git=git, code=manifest,
               measure_key=code["key"], spec=spec, stage2_run_id=s2.get("run_id"),
               b=[list(pair_key(p)) for p in js["b"]], a=[list(pair_key(p)) for p in js["a"]],
               n_b=len(js["b"]), n_a=len(js["a"]), turns=turns, from_turn=spec.judge_from_turn,
               b_digest=js["b_digest"], a_digest=js["a_digest"],
               pinned=dict(b=spec.judge_b_digest, a=spec.judge_a_digest), exposure_note=EXPOSURE,
               diversity=diversity(js["b"]))
    gp = guard_params(Params(), [])
    report = m_store.write_json(out / "runs" / f"{rid}-list.json", res, gp)
    dv = res["diversity"]
    lines = [f"# M.10.4 judgement list {rid}\n\n",
             f"- (b) {res['n_b']} / (a) {res['n_a']}, turns {turns} (from {spec.judge_from_turn})\n",
             f"- digests b {res['b_digest']}, a {res['a_digest']} (pinned: {bool(spec.judge_b_digest)})\n",
             f"- moves {dv['n_moves']}, top {dv['top_move_text']}\n", f"- {EXPOSURE}\n"]
    m_store.write_bytes(out / "runs" / f"{rid}-list.md", "".join(lines).encode(), gp)
    print(f"wrote {report}: (b) {res['n_b']} / (a) {res['n_a']}, turns {turns}", flush=True)
    blockers = dict(dirty=bool(git["dirty_hashed"]) and not a.smoke, spec=spec != summary_spec and not a.smoke)
    if any(blockers.values()):
        print(f"summary not written: {[k for k, v in blockers.items() if v]}", flush=True)
    else:
        write_block(a.summary, "list", res, report, gp)
        print(f"wrote {a.summary} (block list{', smoke' if a.smoke else ''}); next: commit it, then run_m_stage3.py",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
