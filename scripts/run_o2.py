#!/usr/bin/env python3
"""Spec O.3 / O.7.3 / O.7.4 (O2: presentation-evoked depression at N.8a's point; a characterisation, no learning
claim — its result is only grounds for declaring a later test, O.3).

Steps:
1. Refuse unless block n1 of the committed N summary agrees with O.7.4's declared oracle o (reading 13).
2. At o_spec.o2_point, for each X of o2_pairs and each arm of o2_arms (1 plastic, 2 frozen, 3 punish, 4 da_zero) on the
   O2 seeds: o_jobs.arm_job (N2's absolute-arm procedure with a punish flag). One cache file per (O2, X, arm, seed)
   under results/o/run/cache/o2_arm/: rerunning the command resumes.
3. o_rules.o2_judge: the validity gate (INVALID), then per X the plumbing check, the three judgements, the state-flip
   guard and the records.

    uv run python scripts/run_o2.py                                        # the controller only (R2; resumable)
    uv run python scripts/run_o2.py --smoke --allow-dirty --workers 4

Exit 0 for a judged stage with no X INVALID, 5 otherwise (block written; reported to the user), 2 for a refusal."""
from __future__ import annotations

import sys

from flymon.brain import n_cli
from flymon.brain.o_cli import (code_keys, git_state, hooks, load_c3, m0d_sha, main_stage,  # noqa: F401
                                make_measurer, model_types, n_oracle, nonkc_edit, nview, spec_commit)
from flymon.brain.o_rules import INVALID, JUDGED, o2_judge, sentence


def items_o2(spec, st: dict) -> list:
    """Every declared (X, arm, seed) on the APL-on rig, grouped by X then arm."""
    return [dict(x=x, y=y, edit=spec.on_edit, arm=a, punish=pu, plastic=pl, da_zero=dz, odor_x=st[x]["odor"],
                 odor_y=st[y]["odor"], seed=int(s))
            for x, y, _ in spec.o2_pairs for a, pu, pl, dz in spec.o2_arms for s in spec.o2_seeds]


def body(ctx) -> tuple:
    spec, nv = ctx.spec, nview(ctx)
    g, c = spec.o2_point
    names = list(dict.fromkeys(s for x, y, _ in spec.o2_pairs for s in (x, y)))
    st = n_cli.stimuli_at(nv, names, g, c)
    items = items_o2(spec, st)
    m, pool = ctx.hooks["make_measurer"](ctx)
    try:
        rows = m.o2_arms(ctx.c3, items, ctx.readout, spec.n.h3.punish_type)
    finally:
        pool.close()
    res = o2_judge(rows, ctx.z, ctx.extra["declared"], spec)        # z: C3's frozen z, as run_n2_judge's deltas
    res.update(oracle=ctx.extra, drives=n_cli.drives(st), point=dict(g=float(g), c_delta=float(c)), n_arms=len(rows))
    ok = res["outcome"] == JUDGED and all(v["outcome"] != INVALID for v in res["x"].values())
    return dict(res, sentence=sentence("o2", res)), (0 if ok else 5)


def main(argv=None, spec=None, require_root: bool = True) -> int:
    return main_stage("o2", argv, body, hooks(sys.modules[__name__]), spec=spec, require_root=require_root,
                      doc_help=__doc__, prepare=lambda sp, smoke, hk: hk["n_oracle"](sp, smoke))


if __name__ == "__main__":
    sys.exit(main())
