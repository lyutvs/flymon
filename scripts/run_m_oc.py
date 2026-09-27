#!/usr/bin/env python3
"""Spec M.10.6 (+ M.10.7, plan reading 16): the stage-3 operating characteristics, recorded after stage 0 and before
stage 1 (they never change a judgement). Pure: no pool, seconds.

    uv run python scripts/run_m_oc.py
    uv run python scripts/run_m_oc.py --smoke --allow-dirty --summary results/m0d/m/smoke/m_readout.json

Run it from the repository root. Refusals (exit 2), in order: another directory, --out outside results/m0d/m/, dirty
hashed files (unless --allow-dirty), a summary without block stage0 (reading 16: stage0 need only be present; --smoke
reads only a smoke summary under results/m0d/m/), no connectome file, a stage0 block produced under other code, a later
block in the summary (outside --smoke).
Computed (flymon/brain/m_oc.py): u = 0, q_b in SPEC.oc_q_b x q_a/q_b in SPEC.oc_ratio x naive_a in SPEC.oc_naive_a x
c in SPEC.oc_c, each (n, F_a) cell read by m_rules.stage3_reading (M.10.7). The table always has the declared n_b
(SPEC.judge_n_b = 21): a smoke spec's n_b 2 is no judgement list, so stage3_reading would give no outcome — the smoke
run computes the declared table and records that it did.
Report: <out>/runs/<run id>-oc.{json,md}; block "oc" (with J.12.9's provenance: the inputs' sha256, commit, dirty,
arguments), outside --smoke only with clean hashed files and the declared spec; a --smoke run writes it into its smoke
summary (the L convention for chaining smoke stages). Exit codes: 0 written, 2 refused.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

from flymon.brain import l_oc, m_oc, m_store
from flymon.brain.config import Params
from flymon.brain.h3_store import ROOT, sha256_file
from flymon.brain.m_cli import (code_keys, git_state, later_blocks, other_code, out_allowed, provenance, read_previous,
                                refuse, run_id, same_code, write_block)
from flymon.brain.m_measure import HASHED_FILES
from flymon.brain.m_oc import ASSUMPTION, stage3_table
from flymon.brain.m_rules import SELECTED
from flymon.brain.m_spec import SPEC, MSpec, smoke

NEED = ("stage0",)
PRINT_AT = dict(ratio=1.0, naive_a=4, c=4)          # the printed line per q_b (the report carries every row)


def table_spec(spec: MSpec) -> MSpec:
    """spec with the declared judge_n_b: the OC is of the declared 21-pair reading (a smoke n_b reads nothing)."""
    return spec if spec.judge_n_b == SPEC.judge_n_b else dataclasses.replace(spec, judge_n_b=SPEC.judge_n_b)


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
    doc, why = read_previous(a.summary, list(NEED), a.smoke, lambda *_: None)   # reading 16: stage0 present only
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
    why = None if a.smoke else later_blocks(a.summary, "oc", doc)
    if why:
        return refuse(why)
    t0 = time.time()
    ts = table_spec(spec)
    rows = stage3_table(ts)
    argv_ = list(sys.argv[1:] if argv is None else argv)
    prov = provenance([m_oc.__file__, l_oc.STAGE2_OC_PATH, a.summary], vars(a), argv_)
    res = dict(run_id=rid, smoke=a.smoke, argv=argv_, git=git, code=manifest, measure_key=code["key"], spec=spec,
               stage0_run_id=doc["stage0"].get("run_id"), assumption=ASSUMPTION, n_b=ts.judge_n_b,
               n_b_note=None if ts is spec else (f"the spec's judge_n_b {spec.judge_n_b} is not the declared "
                                                 f"{SPEC.judge_n_b}: the table is the declared one (smoke)"),
               rows=rows, provenance=prov, wall_s=time.time() - t0)
    gp = [Params()]
    report = m_store.write_json(out / "runs" / f"{rid}-oc.json", res, gp)
    at = [r for r in rows if r["ratio"] == PRINT_AT["ratio"] and r["naive_a"] == PRINT_AT["naive_a"]
          and r["c"] == PRINT_AT["c"]]
    lines = [f"# M.10.6 stage-3 operating characteristics {rid}\n\n", f"{ASSUMPTION}\n\n",
             f"n_b {ts.judge_n_b}; {len(rows)} rows; P(outcome) at q_a/q_b {PRINT_AT['ratio']}, naive_a "
             f"{PRINT_AT['naive_a']}, c {PRINT_AT['c']}:\n\n",
             *(f"- q_b {r['q_b']}: " + ", ".join(f"{k} {v:.4f}" for k, v in r["P"].items()) + "\n" for r in at)]
    m_store.write_bytes(out / "runs" / f"{rid}-oc.md", "".join(lines).encode(), gp)
    for r in at:
        print(f"q_b {r['q_b']}: P({SELECTED}) {r['P'][SELECTED]:.4f} at q_a/q_b {PRINT_AT['ratio']}, naive_a "
              f"{PRINT_AT['naive_a']}, c {PRINT_AT['c']}", flush=True)
    print(f"wrote {report} (sha256 {sha256_file(report)[:12]}) in {res['wall_s']:.1f} s", flush=True)
    blockers = dict(dirty=bool(git["dirty_hashed"]) and not a.smoke, spec=spec != summary_spec and not a.smoke)
    if any(blockers.values()):
        print(f"summary not written: {[k for k, v in blockers.items() if v]}", flush=True)
    else:
        write_block(a.summary, "oc", res, report, gp)
        print(f"wrote {a.summary} (block oc{', smoke' if a.smoke else ''})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
