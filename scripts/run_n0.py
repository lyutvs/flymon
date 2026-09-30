#!/usr/bin/env python3
"""Spec N.8.4 (N0's similarity gate). At block n0f's operating point, with APL on (the unedited engine — a blocked
condition's KC vector is read only after the N2 verdict, N.8), it presents the three judged stimuli on the activity
seeds and builds the KC firing-probability vectors. It then computes Δr = r(4:1, 1:4) - r(4:1, δ-DL) with a
common-seed bootstrap 95% CI: SIMILARITY_GO iff the lower bound > 0, else STOP_SIMILARITY_ORDER (exit 5). Records:
r(4:1, 1:4), per-stimulus KC activity and sub-window, the drive tables.

    uv run python scripts/run_n0.py                                           # the controller only (R2)
    uv run python scripts/run_n0.py --smoke --allow-dirty --workers 4

Refusals (exit 2) are n_cli.main_stage's, in particular: block n0f not committed, not OPERATING_POINT, or no committed
**N.8a paragraph citing block n0f's run id (outside --smoke). Missing or incomplete KC vectors raise (n_rules), so
nothing is written on them."""
from __future__ import annotations

import sys

from flymon.brain.n_cli import (UNEDITED, check_committed, code_keys, drives, git_state, head_spec,  # noqa: F401
                                hooks, kc_blocks, load_c3, m0d_sha, main_stage, make_measurer, model_types,
                                operating_point, stimuli_at)
from flymon.brain.n_rules import OPERATING_POINT, SIMILARITY_GO, cell_stats, sentence, similarity


def body(ctx) -> tuple:
    spec = ctx.spec
    g, c = operating_point(ctx.doc, spec, ctx.smoke)
    st = stimuli_at(ctx, spec.judged_stimuli, g, c)
    m, pool = ctx.hooks["make_measurer"](ctx)
    try:
        rows = m.kc_vectors(ctx.c3, [(UNEDITED, st[s]["odor"]) for s in spec.judged_stimuli], spec.act_seeds,
                            ctx.readout)
    finally:
        pool.close()
    by, n_kc = kc_blocks(rows, spec.judged_stimuli, spec.act_seeds)      # UNEDITED: the only edit N0 asks a vector for
    sim = similarity({s: [r["kc_fired"] for r in rr] for s, rr in by.items()}, n_kc, spec)
    res = dict(outcome=sim["outcome"], point=dict(g=g, c_delta=c), similarity=sim,
               kc={s: cell_stats(rr, spec) for s, rr in by.items()}, drives=drives(st, spec.judged_stimuli))
    return dict(res, sentence=sentence("n0", res)), (0 if sim["outcome"] == SIMILARITY_GO else 5)


def main(argv=None, spec=None, require_root: bool = True) -> int:
    return main_stage("n0", argv, body, hooks(sys.modules[__name__]), need=("n0f",),
                      outcomes={"n0f": (OPERATING_POINT,)}, note=("N.8a", "n0f"), spec=spec,
                      require_root=require_root, doc_help=__doc__)


if __name__ == "__main__":
    sys.exit(main())
