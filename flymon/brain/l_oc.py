"""Spec L.11.4: operating characteristics of the whole L procedure, recorded before stage 1 (never changes a judgement).
Stage 1's gate is simulated on the 41 real feature vectors and turn structure with Bernoulli labels (q for G-passing
pairs, q_fail otherwise); stage 2's COVERAGE_SHORT is the binomial tail (21st pass after position 84 <=> fewer than 21
passes in 84); stage 3 is J.12.9's exact Poisson-binomial table on 21 (b) pairs and the fixed (a) sample, calibrated by
J.12.9's own script (docs/superpowers/specs/j-diag/stage2_oc.py, loaded by importlib — not copied) and read cell by
cell by the judgement's own function (l_rules.stage3_reading). The whole-procedure product assumes the three stages
independent (stated in the record, ASSUMPTION)."""
from __future__ import annotations

import importlib.util
from functools import lru_cache
from math import comb
from pathlib import Path

from .j_rules import B_FA, B_NO_CONCLUSION, B_TB, SELECTED
from .l_rules import COVERAGE_SHORT, stage3_reading
from .l_screen import SCREEN_FEW, SCREEN_GO, SCREEN_IMPRECISE, SCREEN_NO_RULE, gate

BANDS = (SELECTED, B_TB, B_NO_CONCLUSION, B_FA)
GATE_OUTCOMES = (SCREEN_GO, SCREEN_FEW, SCREEN_IMPRECISE, SCREEN_NO_RULE)
STAGE2_OC_PATH = Path(__file__).resolve().parents[2] / "docs/superpowers/specs/j-diag/stage2_oc.py"
ASSUMPTION = ("P(outcome | q, q_fail, c, naive_a, ratio, u) = gate x stage 2 x stage 3 with the three stages taken as "
              "independent: the gate's P(SCREEN_GO | q, q_fail) (simulated on the 41 real feature vectors and turns), "
              "P(COVERAGE_SHORT | c) (binomial tail, n_pass by max_screened) and the exact stage-3 band table at "
              "q_b = q. The same q is the testable rate of G-passing calibration pairs and of screened-passing new "
              "pairs; the LOTO-precision q_b row is added at stage 1 (reading 15). A record, not a judgement.")


@lru_cache(maxsize=1)
def stage2_oc():
    """J.12.9's script as a module (the same importlib load the tests use for scripts)."""
    sp = importlib.util.spec_from_file_location("stage2_oc", STAGE2_OC_PATH)
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


def c3_offsets(rec: dict) -> tuple:
    """(ub, ua): C3's per-pair turn effects u (J.12.9) on H.4's recorded C3 oracle block — 21 (b) and 4 naive (a)
    offsets; stage2_oc.structure refuses a record that does not reproduce the recorded aggregate."""
    s2 = stage2_oc()
    turns = s2.structure(rec)["turns"]
    return tuple(tuple(x) for x in s2.pair_offsets(turns, s2.turn_effects(turns)))


# ================================================================ stage 2
def coverage_short_prob(c: float, n_pass: int, max_screened: int) -> float:
    """P(fewer than n_pass passes among max_screened pairs | true coverage c) = P(the n_pass-th pass after position
    max_screened), the negative-binomial tail written as a binomial one."""
    return float(sum(comb(max_screened, k) * c ** k * (1 - c) ** (max_screened - k) for k in range(n_pass)))


# ================================================================ stage 3
def _cells(pb, pa, n_b: int, naive_a: int, spec) -> dict:
    P = {b: 0.0 for b in BANDS}
    for tb, w_b in enumerate(pb):
        for fa, w_a in enumerate(pa):
            rd = stage3_reading(dict(testable_b=tb, n_b=n_b, T_b=tb / n_b, F_a=fa, naive_a=naive_a), spec)
            P[rd["band"]] += float(w_b * w_a)
    return P


def stage3_table(q_bs, naive_as, ratios, spec, u_offsets=None) -> list:
    """Rows {q_b, q_a, ratio, naive_a, n_b, u, P: {band: prob}} for q_b x naive_a x q_a/q_b. u_offsets None: u = 0 (pairs
    independent, n_b = spec.j.stage2_n_b, naive_a from naive_as). u_offsets (ub, ua) from c3_offsets: C3's turn effects,
    ub on the 21 (b) pairs and ua on the (a) pairs, so naive_a is len(ua) and naive_as must be exactly (len(ua),).
    Every row is stage2_oc.calibrate per axis (mean pair probability = q) -> poisson_binomial -> stage3_reading."""
    s2, n_b = stage2_oc(), spec.j.stage2_n_b
    if u_offsets is None:
        cases = [(0, [0.0] * n_b, [0.0] * na) for na in naive_as]
    else:
        ub, ua = (list(x) for x in u_offsets)
        if len(ub) != n_b:
            raise ValueError(f"l_oc: {len(ub)} (b) offsets, the declared n_b is {n_b}")
        if tuple(naive_as) != (len(ua),):
            raise ValueError(f"l_oc: C3 rows carry the offsets' naive_a {len(ua)}, not naive_as {tuple(naive_as)}")
        cases = [("C3", ub, ua)]
    rows = []
    for q_b in q_bs:
        for u, ub_, ua_ in cases:
            pb = s2.poisson_binomial(s2.calibrate(ub_, q_b)[1])
            for ratio in ratios:
                q_a = q_b * ratio
                pa = s2.poisson_binomial(s2.calibrate(ua_, q_a)[1])
                rows.append(dict(q_b=q_b, q_a=q_a, ratio=ratio, naive_a=len(ua_), n_b=n_b, u=u,
                                 P=_cells(pb, pa, n_b, len(ua_), spec)))
    return rows


# ================================================================ stage 1
def gate_oc(pairs: list, q: float, q_fail: float, spec, rng, draws: int | None = None) -> dict:
    """{gate outcome: share of draws}: each draw labels every pair Bernoulli(q if G else q_fail) (pair order, one
    uniform each) and runs l_screen.gate on the real features and turns. The input records are not changed."""
    n = draws or spec.oc_draws
    out = {k: 0 for k in GATE_OUTCOMES}
    for _ in range(n):
        u = rng.random(len(pairs))
        lab = [bool(x < (q if p["feat"]["G"] else q_fail)) for p, x in zip(pairs, u)]
        out[gate([dict(p, testable=l) for p, l in zip(pairs, lab)], spec)["outcome"]] += 1
    return {k: v / n for k, v in out.items()}


# ================================================================ the whole procedure
def whole(gate_p: float, short_p: float, sel_p: float) -> float:
    """P(SELECTED) = P(SCREEN_GO) x (1 - P(COVERAGE_SHORT)) x P(stage-3 SELECTED), stages independent."""
    return float(gate_p * (1.0 - short_p) * sel_p)


def whole_table(gate_ps: dict, short_ps: dict, stage3_rows: list) -> dict:
    """gate_ps {(q, q_fail): gate_oc(...)}, short_ps {c: coverage_short_prob}, stage3_rows from stage3_table. One row per
    (q, q_fail, c, stage-3 row with q_b == q) — every naive_a / ratio / u sensitivity row gets its own product — with
    P over every result code: the gate's non-GO codes, COVERAGE_SHORT and the four bands (sums to 1)."""
    rows = []
    for (q, q_fail), g in gate_ps.items():
        go = g[SCREEN_GO]
        for c, sp in short_ps.items():
            for s in (r for r in stage3_rows if r["q_b"] == q):
                P = {k: g[k] for k in GATE_OUTCOMES if k != SCREEN_GO}
                P[COVERAGE_SHORT] = go * sp
                P.update({b: go * (1.0 - sp) * s["P"][b] for b in BANDS})
                rows.append(dict(q=q, q_fail=q_fail, c=c, naive_a=s["naive_a"], ratio=s["ratio"], q_a=s["q_a"],
                                 u=s["u"], P=P))
    return dict(assumption=ASSUMPTION, rows=rows)
