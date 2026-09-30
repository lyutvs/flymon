"""Spec M.10.6: the stage-3 operating characteristics, recorded before stage 1 and never changing a judgement. u = 0
(the pairs independent — the new judgement set's turn effects are unknown before it is measured): testable_b ~
Poisson-binomial(21 x q_b), F_a ~ Poisson-binomial(naive_a x q_a) with q_a = ratio x q_b, the same-set C3 count c fixed
at each declared value; every (n, F_a) cell is read by m_rules.stage3_reading itself (so M.10.7's n >= 11 with c >= n
reads B_NO_CONCLUSION here too). The Poisson-binomial is J.12.9's script, loaded through l_oc.stage2_oc (not copied)."""
from __future__ import annotations

from .l_oc import stage2_oc
from .m_rules import B_FA, B_NO_CONCLUSION, B_TB, SELECTED, stage3_reading

OUTCOMES = (SELECTED, B_TB, B_NO_CONCLUSION, B_FA)
ASSUMPTION = ("u = 0: pairs independent with testable probability q_b ((b)) and q_a = ratio x q_b ((a)); the same-set C3 "
              "count c fixed; each cell read by m_rules.stage3_reading. A record, not a judgement.")


def stage3_table(spec) -> list:
    """One row per (q_b, ratio, naive_a, c) of spec's OC grid: {q_b, ratio, q_a, naive_a, c, P: {outcome: prob}}."""
    pb_fn = stage2_oc().poisson_binomial
    n_b = spec.judge_n_b
    rows = []
    for q_b in spec.oc_q_b:
        pb = pb_fn([q_b] * n_b)                                   # P(testable_b = n), n = 0..n_b
        for ratio in spec.oc_ratio:
            q_a = ratio * q_b
            for naive_a in spec.oc_naive_a:
                pa = pb_fn([q_a] * naive_a) if naive_a else [1.0]  # F_a = 0 surely when no naive (a) pair
                for c in spec.oc_c:
                    c3 = dict(testable_b=c, F_a=0, n_b=n_b, naive_a=0)
                    P = {o: 0.0 for o in OUTCOMES}
                    for n, p_n in enumerate(pb):
                        for fa, p_f in enumerate(pa):
                            agg = dict(testable_b=n, F_a=fa, n_b=n_b, naive_a=naive_a)
                            P[stage3_reading(agg, c3, spec)["outcome"]] += float(p_n) * float(p_f)
                    rows.append(dict(q_b=q_b, ratio=ratio, q_a=q_a, naive_a=naive_a, c=c, P=P))
    return rows
