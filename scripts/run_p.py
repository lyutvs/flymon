#!/usr/bin/env python3
"""Spec P.2 / P.6.2 / P.6.5 (the first learning judgement: punishment learning on the real-odour dissimilar pair, both
directions, new seeds 23_000_000 + i; a confirmatory test declared after O2, P.0).

Steps:
1. Refuse (outside --smoke) unless the committed P summary holds block oc (P.6.4: the operating characteristic is
   recorded before the judgement run), and unless block n1's dissimilar-pair o agrees with -2.433 within 5e-4 (c1).
2. At the operating point, for each direction (r1 X = 4:1 / Y = δ-DL, r2 X = δ-DL / Y = 4:1) and each arm (plastic ①,
   frozen ②, punish ③) on the P seeds: o_jobs.arm_job. One cache file per (P, direction, arm, seed) under
   results/p/run/cache/p_arm/: rerunning the command resumes.
3. p_rules.p_judge: the gate and arm ②'s plumbing (INVALID), the per-direction rule and the label, the records.

    uv run python scripts/run_p.py                                        # the controller only (R1; resumable)
    uv run python scripts/run_p.py --rerun-after-invalid                  # P.6.5: once, after a technical INVALID
    uv run python scripts/run_p.py --smoke --allow-dirty --workers 4

Exit 0 for a judged stage (any label), 5 for INVALID (block written; reported to the user), 2 for a refusal."""
from __future__ import annotations

import sys

from flymon.brain import n_cli
from flymon.brain.p_cli import (c1_source, check_committed, code_keys, git_state, hooks, load_c3,  # noqa: F401
                                m0d_sha, main_stage, make_measurer, model_types, nview, o2_source, spec_commit)
from flymon.brain.p_rules import JUDGED, p_judge, sentence


def items_p(spec, st: dict) -> list:
    """Every declared (direction, arm, seed) on the APL-on rig at the operating point, grouped by direction then arm."""
    point = tuple(float(v) for v in spec.o.o2_point)
    flags = spec.flags()
    return [dict(direction=d, x=x, y=y, edit=spec.o.on_edit, arm=a, punish=flags[a][0], plastic=flags[a][1],
                 da_zero=flags[a][2], odor_x=st[x]["odor"], odor_y=st[y]["odor"], seed=int(s), point=point)
            for d, (x, y) in spec.pairs().items() for a in spec.arms for s in spec.seeds]


def prepare(spec, smoke: bool, hk: dict, doc: dict, summary, z) -> tuple:
    """P.6.4 before P.6.2 (outside --smoke: block oc present and committed), then c1 from block n1 (P.6.5)."""
    if not smoke:
        if "oc" not in doc:
            return None, f"{summary} holds no block oc: record the operating characteristic first (P.6.4)"
        why = hk["check_committed"](summary, ["oc"])
        if why:
            return None, why
    c1, why = hk["c1_source"](spec, smoke)
    if why:
        return None, why
    return dict(c1=c1, oc_run_id=(doc.get("oc") or {}).get("run_id")), None


def body(ctx) -> tuple:
    spec, nv = ctx.spec, nview(ctx)
    g, c = spec.o.o2_point
    names = list(dict.fromkeys(s for x, y in spec.pairs().values() for s in (x, y)))
    st = n_cli.stimuli_at(nv, names, g, c)
    m, pool = ctx.hooks["make_measurer"](ctx)
    try:
        rows = m.p_arms(ctx.c3, items_p(spec, st), ctx.readout, spec.o.n.h3.punish_type)
    finally:
        pool.close()
    res = p_judge(rows, ctx.z, ctx.extra["c1"]["c1"], spec)        # z: C3's frozen z, as O2's
    res.update(c1_source=ctx.extra["c1"], oc_run_id=ctx.extra["oc_run_id"], drives=n_cli.drives(st),
               point=dict(g=float(g), c_delta=float(c)), n_arms=len(rows))
    return dict(res, sentence=sentence(res)), (0 if res["outcome"] == JUDGED else 5)


def main(argv=None, spec=None, require_root: bool = True) -> int:
    return main_stage("p", argv, body, hooks(sys.modules[__name__]), spec=spec, require_root=require_root,
                      doc_help=__doc__, prepare=prepare, rerun_once=True)


if __name__ == "__main__":
    sys.exit(main())
