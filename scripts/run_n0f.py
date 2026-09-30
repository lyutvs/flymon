#!/usr/bin/env python3
"""Spec N.8.3 (N0f, record only, no learning; with N.8.2's data stop).

Steps:
1. Load data/odor/ against its pins. A defect ends STOP_DATA_MISMATCH (block written, exit 5, no pool).
2. Record the Lin 2014 totals under both conventions.
3. Present every N0f stimulus of every (g, c_δ) on the N0f seeds, APL on / APL->KC block / all-output block. This is
   the presentation measurement: scalars only. N0f never asks for a KC vector under any condition (N.8 blinding).
4. Apply the operating-point rule, which ends in OPERATING_POINT or STOP_NO_OPERATING_POINT (exit 5).

    uv run python scripts/run_n0f.py                                          # the controller only (R1)
    uv run python scripts/run_n0f.py --smoke --allow-dirty --workers 4        # results/n/smoke/

Refusals (exit 2) are n_cli.main_stage's. Block "n0f" is written for every outcome; missing presentation rows raise
(n_rules), so nothing is written on them."""
from __future__ import annotations

import sys

from flymon.brain.h3_store import canonical
from flymon.brain.n_cli import (check_committed, code_keys, drives, git_state, glomeruli, head_spec,  # noqa: F401
                                hooks, load_c3, m0d_sha, main_stage, make_measurer, model_types, stimuli_at)
from flymon.brain.n_rules import (OPERATING_POINT, STOP_DATA_MISMATCH, budget_hours, cell_stats, point_key,
                                  select_point, sentence, state_flags, wall_per_step)
from flymon.brain.odor_real import DataMismatch, lin_record


def body(ctx) -> tuple:
    spec = ctx.spec
    try:
        table, glom = glomeruli(ctx)
    except DataMismatch as e:
        res = dict(outcome=STOP_DATA_MISMATCH, reason=str(e))
        return dict(res, sentence=sentence("n0f", res)), 5
    grid = {(g, c): stimuli_at(ctx, spec.n0f_stimuli, g, c, glom) for g in spec.g_grid for c in spec.c_delta_grid}
    odours = {}
    for st in grid.values():                       # an odour is its drive: the non-δ stimuli are shared across c_δ
        for rec in st.values():
            odours.setdefault(canonical(rec["odor"]), rec["odor"])
    items = [(edit, o) for _, edit in spec.conditions for o in odours.values()]
    m, pool = ctx.hooks["make_measurer"](ctx)
    try:
        rows = m.presentations(ctx.c3, items, spec.n0f_seeds, ctx.readout)
    finally:
        pool.close()
    if len(rows) != len(items) or any(len(rr) != len(spec.n0f_seeds) for rr in rows):
        raise ValueError(f"presentation rows are missing: {len(rows)} blocks for {len(items)} (edit, odour) items, "
                         f"seeds per block {sorted({len(rr) for rr in rows})} for {len(spec.n0f_seeds)}")
    by = {(e, canonical(o)): r for (e, o), r in zip(items, rows)}
    cells = {gc: {s: {cond: cell_stats(by[(edit, canonical(rec["odor"]))], spec) for cond, edit in spec.conditions}
                  for s, rec in st.items()} for gc, st in grid.items()}
    sel = select_point(cells, spec)
    wps = wall_per_step([r for rr in rows for r in rr])
    res = dict(outcome=sel["outcome"], selected=sel["selected"], checks=sel["checks"],
               lin_totals=lin_record(table, dict(spec.lin_totals), spec.lin_tol),
               grid={point_key(*gc): v for gc, v in cells.items()}, wall_s_per_step=wps,
               budget_estimate_h={str(n): budget_hours(n, wps, spec, ctx.c3.dt) for n in spec.n_grid})
    if sel["selected"]:
        gc = (sel["selected"]["g"], sel["selected"]["c_delta"])
        at = cells[gc]
        on = dict(spec.conditions)
        base = next(iter(on))                      # the first declared condition is the APL-on one
        res.update(state_shares={s: {cond: at[s][cond]["firing_share"] for cond in on} for s in at},
                   validity=sel["checks"][point_key(*gc)], state_flags=state_flags(at, spec),
                   apl_shift={s: {cond: at[s][cond]["apl_out_mean"] - at[s][base]["apl_out_mean"]
                                  for cond in on if cond != base} for s in at},
                   drives=drives(grid[gc]))
    return dict(res, sentence=sentence("n0f", res)), (0 if sel["outcome"] == OPERATING_POINT else 5)


def main(argv=None, spec=None, require_root: bool = True) -> int:
    return main_stage("n0f", argv, body, hooks(sys.modules[__name__]), need=(), spec=spec, require_root=require_root,
                      doc_help=__doc__)


if __name__ == "__main__":
    sys.exit(main())
