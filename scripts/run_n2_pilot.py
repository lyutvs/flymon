#!/usr/bin/env python3
"""Spec N.8.6, decision ⑥ (N2.0: calibration before any APL-block learning data).

Steps:
1. The absolute-conditioning arm with APL on, for both pairs, on the pilot seeds (n_jobs.absolute_arm_job). Only the
   unedited conditions of n_spec's N2 list are run: no block arm and no KC vector (N.8 blinding).
2. Per-seed Δ = dV_post - dV_pre.
3. c1 = c1_frac |o| (block n1); δ_min = delta_min_frac ℓ̂_sim,on; ε = eps_frac ℓ̂_dis,on.
4. The joint-rule OC (n_oc.simulate) over every declared pairing x spread, and n. A partial table (a subset of the
   pairings) is refused before any outcome: it raises, nothing is written.
5. The judgement-block budget from block n0f's wall clock.
Blinding: block n0f is read only through n_rules.design_view (operating point, wall clock, state shares, validity),
block n1 only for each pair's o, block n0 not at all.
Outcomes: N2_0_GO, STOP_POWER (an uncalibratable pilot or no n) or STOP_BUDGET (over the budget), both exit 5. The
numbers go into N.8b (scripts/write_n_notes.py), which the controller commits before the judge. The block keeps the
whole OC: every (pairing, spread, hypothesis, n) row, each scenario's own minimal n, and the bootstrap draws the OC
used beside the judgement's.

    uv run python scripts/run_n2_pilot.py                                     # the controller only (R4)
    uv run python scripts/run_n2_pilot.py --smoke --allow-dirty --workers 4

Refusals (exit 2) are n_cli.main_stage's: blocks n0f, n0 and n1 committed with GO outcomes, and the committed **N.8a
paragraph (outside --smoke). Missing pilot rows raise, so nothing is written on them."""
from __future__ import annotations

import sys

from flymon.brain.n_cli import (UNEDITED, arm_items, by_condition, check_committed, code_keys,  # noqa: F401
                                firing_shares, git_state, head_spec, hooks, load_c3, m0d_sha, main_stage,
                                make_measurer, model_types, operating_point, stimuli_at)
from flymon.brain.n_oc import simulate
from flymon.brain.n_rules import (N1_GO, N2_0_GO, OPERATING_POINT, SIMILARITY_GO, budget_hours, deltas, design_view,
                                  n2_0_outcome, sentence, thresholds)


def body(ctx) -> tuple:
    spec = ctx.spec
    view = design_view(ctx.doc["n0f"])                  # blinding: nothing else of block n0f is read
    g, c = operating_point({"n0f": view}, spec, ctx.smoke)
    st = stimuli_at(ctx, spec.judged_stimuli, g, c)
    conds = [cnd for cnd in spec.n2_conditions if cnd[2] == UNEDITED]          # APL on only (decision ⑥)
    names = {pair: name for name, pair, _ in conds}
    m, pool = ctx.hooks["make_measurer"](ctx)
    try:
        rows = m.arms(ctx.c3, arm_items(st, spec, conds, spec.pilot_seeds, True), ctx.readout, spec.h3.punish_type)
    finally:
        pool.close()
    by = by_condition(rows, names.values(), spec.pilot_seeds, "N2.0 pilot")
    d = deltas(by, names.values(), ctx.z)
    pilot = {pair: d[name] for pair, name in names.items()}
    th = thresholds({pair: ctx.doc["n1"]["pairs"][pair]["o"] for pair in names}, pilot, spec)
    oc = simulate(pilot, th["c1"], th["delta_min"], th["eps"], spec, spec.oc_pairings) if th["calibratable"] else None
    if oc is not None and oc["partial"]:
        raise ValueError(f"the OC is partial (pairings {oc['pairings']} of {list(spec.oc_pairings)}): a design n needs "
                         "every declared pairing; refusing to report an outcome")
    n = oc["n"] if oc else None
    # with no n the outcome is STOP_POWER whatever the budget; the hours recorded are then the grid's largest n's
    budget = budget_hours(n if n else max(spec.n_grid), view["wall_s_per_step"], spec, ctx.c3.dt)
    out = n2_0_outcome(th, oc, budget, spec)
    res = dict(outcome=out["outcome"], reason=out["reason"], n=n, c1=th["c1"], delta_min=th["delta_min"],
               eps=th["eps"], ell_hat=th["ell_hat"], pilot_sd=th["pilot_sd"], alt_D_sim=th["alt_D_sim"],
               pilot={k: v.tolist() for k, v in pilot.items()}, pilot_seeds=list(spec.pilot_seeds), oc=oc,
               scenario_n=oc["scenario_n"] if oc else None, oc_boot=oc["boot"] if oc else None,
               judge_boot=spec.boot_draws, budget_h=budget, design_view=view, pilot_states=firing_shares(by, spec))
    return dict(res, sentence=sentence("n2_0", res)), (0 if out["outcome"] == N2_0_GO else 5)


def main(argv=None, spec=None, require_root: bool = True) -> int:
    return main_stage("n2_0", argv, body, hooks(sys.modules[__name__]), need=("n0f", "n0", "n1"),
                      outcomes={"n0f": (OPERATING_POINT,), "n0": (SIMILARITY_GO,), "n1": (N1_GO,)},
                      note=("N.8a", "n0f"), spec=spec, require_root=require_root, doc_help=__doc__)


if __name__ == "__main__":
    sys.exit(main())
