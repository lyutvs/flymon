#!/usr/bin/env python3
"""Spec O.2 / O.7.1 / O.7.2 (O1: C3's two-state map on the real-odour rig; a characterisation, no learning claim).

Steps:
1. Load the pinned Hallem data; a mismatch is a refusal (exit 2, nothing written).
2. Present every O1 stimulus at every g of o_spec.o1_g_grid (c_δ = o1_c_delta on δ-DL only) on the O1 seeds under the
   four conditions (on, APL->KC block, APL->non-KC block, all-output block). One presentation per job and one cache
   file per (O1, condition, cell, seed) under results/o/run/cache/o1_pres/: rerunning the command resumes.
   Before the pool: the APL->non-KC block must zero at least one APL -> MBON05 edge on the real connectome (o_cli.
   nonkc_edit, recorded in the block); none zeroed ends INVALID with no presentation run (READOUT_PATH would not
   name the readout path).
3. o_rules.o1_judge: the validity gate (INVALID), then STOP_NO_SILENT_STATE or the three judgements, with the records.

    uv run python scripts/run_o1.py                                        # the controller only (R1; resumable)
    uv run python scripts/run_o1.py --smoke --allow-dirty --workers 4      # results/o/smoke/

Exit 0 for JUDGED, 5 for STOP_NO_SILENT_STATE or INVALID (block written; reported to the user), 2 for a refusal."""
from __future__ import annotations

import sys

from flymon.brain import n_cli
from flymon.brain.n_rules import wall_per_step
from flymon.brain.o_cli import (code_keys, git_state, hooks, load_c3, m0d_sha, main_stage,  # noqa: F401
                                make_measurer, model_types, n_oracle, nonkc_edit, nview, spec_commit)
from flymon.brain.o_rules import INVALID, JUDGED, o1_judge, sentence


def items_o1(spec, st: dict) -> list:
    """Every declared (condition, g, stimulus, seed), grouped by condition (a worker rebuilds its rig only on a
    switch)."""
    return [dict(cond=cond, edit=edit, g=float(g), stim=s, seed=int(seed), odor=st[g][s]["odor"])
            for cond, edit in spec.o1_conditions for g in spec.o1_g_grid for s in spec.o1_stimuli
            for seed in spec.o1_seeds]


def body(ctx) -> tuple:
    spec, nv = ctx.spec, nview(ctx)
    _, glom = n_cli.glomeruli(nv)
    st = {g: n_cli.stimuli_at(nv, spec.o1_stimuli, g, spec.o1_c_delta, glom) for g in spec.o1_g_grid}
    items = items_o1(spec, st)
    edit = ctx.hooks["nonkc_edit"](ctx)
    if edit["apl_to_p_zeroed"] < 1:
        res = dict(outcome=INVALID, nonkc_edit=edit, n_presentations=0,
                   reasons=[f"the {edit['edit']} rig zeroed no APL -> {edit['readout_p']} edge "
                            f"({edit['apl_to_p_edges']} such edges): the non-KC block would not touch the readout"])
        return dict(res, sentence=sentence("o1", res)), 5
    m, pool = ctx.hooks["make_measurer"](ctx)
    try:
        rows = m.o1_presentations(ctx.c3, items, ctx.readout)
    finally:
        pool.close()
    res = o1_judge(rows, spec)
    res.update(nonkc_edit=edit, drives={f"{float(g):g}": n_cli.drives(st[g]) for g in spec.o1_g_grid},
               n_presentations=len(rows),
               wall_s_per_step=wall_per_step(rows) if rows else None)
    return dict(res, sentence=sentence("o1", res)), (0 if res["outcome"] == JUDGED else 5)


def main(argv=None, spec=None, require_root: bool = True) -> int:
    return main_stage("o1", argv, body, hooks(sys.modules[__name__]), spec=spec, require_root=require_root,
                      doc_help=__doc__)


if __name__ == "__main__":
    sys.exit(main())
