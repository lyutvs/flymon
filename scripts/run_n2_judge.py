#!/usr/bin/env python3
"""Spec N.4 / N.8.6 / N.8.7 (N2: the judgement).

Steps:
1. On the n judge seeds of block n2_0, run the four judged conditions (similar / dissimilar x APL on /
   APL->KC block) and the two all-output-block record arms (n_jobs.absolute_arm_job).
2. Rerun each condition's first plumbing_seeds seeds with plasticity off (the plumbing check). The rerun count is
   checked against n_spec before the check: a missing rerun raises, it is not a verdict.
3. Checkpoint: every (condition, seed) is one cache entry under results/n/run/cache/, so rerunning the command resumes.
   Those entries hold the per-presentation rows (and so each presentation's state); the block holds aggregates.
4. INVALID first: plumbing, block validity, CSC sha256 checks. Then the raw-unit statistics and the verdict with c1,
   δ_min and ε from block n2_0: SUPPORTED / NOT_REPLICATED / NO_LEARNING.
5. Records: d'-based L and I, the all-output-block table (same seeds), the state-conditional ℓ, the state shares and
   the per-seed Δ.
6. The block with the verdict is written. Only then (n_cli.main_stage's `after`) are the block conditions' KC vectors
   asked for, on the activity seeds, and their correlations added to the block. An INVALID run asks for none: its
   judgement is not over, so the block KC correlations stay unread.

    uv run python scripts/run_n2_judge.py                                     # the controller only (R5; resumable)
    uv run python scripts/run_n2_judge.py --smoke --allow-dirty --workers 4

Refusals (exit 2) are n_cli.main_stage's: blocks n0f, n0, n1 and n2_0 committed with GO outcomes, and the committed
**N.8b paragraph citing block n2_0's run id (outside --smoke). Exit 0 for every verdict (the verdict goes to the
user). Missing arm rows, reruns, thresholds or n raise: no verdict is written on them."""
from __future__ import annotations

import sys

from flymon.brain.n_cli import (UNEDITED, arm_items, by_condition, check_committed, code_keys,  # noqa: F401
                                firing_shares, git_state, head_spec, hooks, kc_blocks, load_c3, m0d_sha, main_stage,
                                make_measurer, model_types, operating_point, stimuli_at)
from flymon.brain.n_rules import (INVALID, N1_GO, N2_0_GO, N2_ORDER, OPERATING_POINT, SIMILARITY_GO, block_validity,
                                  csc_checks, deltas, dprime_record, n2_stats, plumbing, sentence, similarity,
                                  state_conditional, verdict)

KC_RECORD = ("r_sim", "r_dis", "delta_r", "ci95")


def body(ctx) -> tuple:
    spec, b0 = ctx.spec, ctx.doc["n2_0"]
    n = b0.get("n")
    if n is None:
        if not ctx.smoke:
            raise ValueError("block n2_0 holds no n: N2 cannot be judged")
        n = spec.n_grid[0]                              # a --smoke run past a stopped smoke pilot
    seeds = spec.judge_seeds(n)
    rerun = seeds[:spec.plumbing_seeds]
    g, c = operating_point(ctx.doc, spec, ctx.smoke)
    st = stimuli_at(ctx, spec.judged_stimuli, g, c)
    conds = spec.n2_conditions + spec.n2_record_conditions
    names = [name for name, _, _ in conds]
    items = arm_items(st, spec, conds, seeds, True) + arm_items(st, spec, conds, rerun, False)
    m, pool = ctx.hooks["make_measurer"](ctx)
    try:
        rows = m.arms(ctx.c3, items, ctx.readout, spec.h3.punish_type)
    finally:
        pool.close()
    by = by_condition([r for r in rows if r["plastic"]], names, seeds, "N2 arms")
    off = by_condition([r for r in rows if not r["plastic"]], names, rerun, "N2 plumbing reruns")
    reruns = [r for name in names for r in off[name]]
    if len(reruns) != len(conds) * spec.plumbing_seeds:
        raise ValueError(f"plumbing reruns are missing: {len(reruns)} for {len(conds)} conditions x "
                         f"{spec.plumbing_seeds} seeds")
    judged = {k: by[k] for k in N2_ORDER}
    invalid = plumbing(reruns) + block_validity(judged, spec) + csc_checks(judged)
    d = deltas(by, names, ctx.z)
    stats = n2_stats({k: d[k] for k in N2_ORDER}, spec)
    v = verdict(stats, b0["c1"], b0["delta_min"], b0["eps"], invalid)
    record = {f"{pair}_off": name for name, pair, _ in spec.n2_record_conditions}   # all-output arm as the block arm
    all_block = n2_stats({k: d[record.get(k, k)] for k in N2_ORDER}, spec)
    res = dict(outcome=v["verdict"], verdict=v, stats=stats, invalid=invalid,
               thresholds=dict(n=int(n), c1=b0["c1"], delta_min=b0["delta_min"], eps=b0["eps"]),
               per_seed={cnd: [dict(seed=int(s), delta=float(x)) for s, x in zip(seeds, d[cnd])] for cnd in d},
               dprime_record=dprime_record({k: d[k] for k in N2_ORDER}),
               all_output_block=dict(all_block, arms=record, note="기록만: 판정에 쓰지 않는다 (N.8.1)"),
               state_conditional=state_conditional(by, ctx.z, spec), state_shares=firing_shares(by, spec),
               csc_sha256={cnd: sorted({r["csc_sha256"] for r in rs}) for cnd, rs in by.items()},
               post_verdict_kc=None, n0_r_sim=(ctx.doc.get("n0") or {}).get("similarity", {}).get("r_sim"))
    return dict(res, sentence=sentence("n2", res)), 0


def kc_record(ctx, res: dict) -> dict:
    """N.8.6: the block conditions' KC correlations. main_stage calls this only after the verdict block is written."""
    if res["outcome"] == INVALID:
        return dict(post_verdict_kc=None,
                    post_verdict_kc_note="INVALID: 판정이 끝나지 않아 차단 KC 상관을 읽지 않는다")
    spec = ctx.spec
    g, c = operating_point(ctx.doc, spec, ctx.smoke)
    st = stimuli_at(ctx, spec.judged_stimuli, g, c)
    edits = [edit for _, edit in spec.conditions if edit != UNEDITED]
    k = len(spec.judged_stimuli)
    m, pool = ctx.hooks["make_measurer"](ctx)
    try:
        rows = m.kc_vectors(ctx.c3, [(e, st[s]["odor"]) for e in edits for s in spec.judged_stimuli], spec.act_seeds,
                            ctx.readout)
    finally:
        pool.close()
    if len(rows) != len(edits) * k:
        raise ValueError(f"KC vectors are missing: {len(rows)} blocks for {len(edits)} edits x {k} stimuli")
    out = {}
    for i, e in enumerate(edits):
        part, n_kc = kc_blocks(rows[i * k:(i + 1) * k], spec.judged_stimuli, spec.act_seeds)
        sim = similarity({s: [r["kc_fired"] for r in rr] for s, rr in part.items()}, n_kc, spec)
        out[e] = {x: sim[x] for x in KC_RECORD}
    return dict(post_verdict_kc=out)


def main(argv=None, spec=None, require_root: bool = True) -> int:
    return main_stage("n2", argv, body, hooks(sys.modules[__name__]), need=("n0f", "n0", "n1", "n2_0"),
                      outcomes={"n0f": (OPERATING_POINT,), "n0": (SIMILARITY_GO,), "n1": (N1_GO,),
                                "n2_0": (N2_0_GO,)},
                      note=("N.8b", "n2_0"), spec=spec, require_root=require_root, doc_help=__doc__, after=kc_record)


if __name__ == "__main__":
    sys.exit(main())
