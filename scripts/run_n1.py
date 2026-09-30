#!/usr/bin/env python3
"""Spec N.3 as amended by N.8.5 (N1: the punish-only oracle, APL on only).

Steps:
1. For each pair (similar 4:1 / 1:4, dissimilar 4:1 / δ-DL) at block n0f's operating point, run
   n_jobs.punish_only_oracle_job: α on the select seeds, pre and P on the report seeds. The engine is the unedited one:
   N1 never runs an APL block and never asks for a KC vector (N.8 blinding).
2. Per pair: p0 = d'(dV_P - dV_pre), testable iff p0 <= -testable_min and sd > 0, and the raw mean o (N.8.6's c1).
3. N1_GO iff both pairs are testable, else STOP_UNTESTABLE (exit 5), with the note when only the dissimilar pair is
   testable.

    uv run python scripts/run_n1.py                                           # the controller only (R3)
    uv run python scripts/run_n1.py --smoke --allow-dirty --workers 4

Refusals (exit 2) are n_cli.main_stage's: blocks n0f and n0 committed, n0f OPERATING_POINT, n0 SIMILARITY_GO, and the
committed **N.8a paragraph citing block n0f's run id (outside --smoke). A missing pair or a report block that is not on
every report seed raises, so nothing is written on it."""
from __future__ import annotations

import sys

from flymon.brain.n_cli import (check_committed, code_keys, git_state, head_spec, hooks, load_c3,  # noqa: F401
                                m0d_sha, main_stage, make_measurer, model_types, operating_point, stimuli_at)
from flymon.brain.n_rules import N1_GO, OPERATING_POINT, SIMILARITY_GO, n1_outcome, n1_pair, sentence


def body(ctx) -> tuple:
    spec = ctx.spec
    g, c = operating_point(ctx.doc, spec, ctx.smoke)
    st = stimuli_at(ctx, spec.judged_stimuli, g, c)
    pairs = [dict(name=n, odor_x=st[x]["odor"], odor_y=st[y]["odor"]) for n, (x, y) in spec.pair_stimuli().items()]
    m, pool = ctx.hooks["make_measurer"](ctx)
    try:
        rows = m.punish_oracle(ctx.c3, pairs, ctx.readout, ctx.z, spec.h3.punish_type)
    finally:
        pool.close()
    seen = sorted({len(r["report"][k][f]) for r in rows for k in ("pre", "P") for f in ("A", "P")})
    if len(rows) != len(pairs) or seen != [len(spec.report_seeds)]:
        raise ValueError(f"oracle rows are missing: {len(rows)} of {len(pairs)} pairs, report seeds per probe {seen} "
                         f"for {len(spec.report_seeds)}")
    per = {p["name"]: n1_pair(r, ctx.z, spec) for p, r in zip(pairs, rows)}
    out = n1_outcome(per)
    res = dict(outcome=out["outcome"], note=out["note"], point=dict(g=g, c_delta=c), pairs=per)
    return dict(res, sentence=sentence("n1", res)), (0 if out["outcome"] == N1_GO else 5)


def main(argv=None, spec=None, require_root: bool = True) -> int:
    return main_stage("n1", argv, body, hooks(sys.modules[__name__]), need=("n0f", "n0"),
                      outcomes={"n0f": (OPERATING_POINT,), "n0": (SIMILARITY_GO,)}, note=("N.8a", "n0f"), spec=spec,
                      require_root=require_root, doc_help=__doc__)


if __name__ == "__main__":
    sys.exit(main())
