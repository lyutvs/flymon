"""Spec L.11.4: the COVERAGE_SHORT probability, the exact stage-3 table and the simulated gate."""
import dataclasses
import json
from math import comb
from pathlib import Path

import numpy as np
import pytest

from flymon.brain import l_oc
from flymon.brain.j_rules import B_FA, B_NO_CONCLUSION, B_TB, SELECTED
from flymon.brain.l_screen import SCREEN_FEW, SCREEN_GO, SCREEN_IMPRECISE, SCREEN_NO_RULE
from flymon.brain.l_spec import SPEC

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def c3_rec():
    return json.loads((ROOT / "results/summary/m0d.json").read_text())["h4"]["h4"]["combos"]["C3"]["oracle"]


def _feats(n=24):
    return [dict(turn=t, set="even", feat=dict(G=t % 3 != 0, S=0.1 * t, f2=float(t), f3=0.5)) for t in range(n)]


# ================================================================ stage 2: the coverage tail
def test_coverage_short_is_the_binomial_tail():
    want = sum(comb(84, k) * 0.25 ** k * 0.75 ** (84 - k) for k in range(21))
    assert l_oc.coverage_short_prob(0.25, 21, 84) == pytest.approx(want)
    assert l_oc.coverage_short_prob(0.25, 21, 84) == pytest.approx(0.46, abs=0.02)


def test_coverage_short_matches_the_spec_reference_values_and_falls_with_c():
    # L.11.4's reference says "about 0.18" at c 0.30; the exact tail is 0.1305 (the 0.25 -> 0.46 reference holds)
    want = sum(comb(84, k) * 0.3 ** k * 0.7 ** (84 - k) for k in range(21))
    assert l_oc.coverage_short_prob(0.30, 21, 84) == pytest.approx(want) == pytest.approx(0.1305, abs=1e-3)
    ps = [l_oc.coverage_short_prob(c, SPEC.n_pass, SPEC.max_screened) for c in SPEC.oc_c]
    assert all(a > b for a, b in zip(ps, ps[1:]))


# ================================================================ stage 3: the exact table
def test_stage3_table_sums_to_one_and_moves_with_q():
    rows = l_oc.stage3_table((0.5, 0.7), (4,), (1.0,), SPEC)
    for r in rows:
        assert sum(r["P"].values()) == pytest.approx(1.0)
    lo, hi = rows
    assert hi["P"][SELECTED] > lo["P"][SELECTED] and hi["P"][B_TB] < lo["P"][B_TB]


def test_stage3_table_covers_every_cell_and_names_it():
    rows = l_oc.stage3_table(SPEC.oc_q, SPEC.oc_naive_a, SPEC.oc_ratio, SPEC)
    assert len(rows) == len(SPEC.oc_q) * len(SPEC.oc_naive_a) * len(SPEC.oc_ratio)
    assert {(r["q_b"], r["naive_a"], r["ratio"]) for r in rows} == {
        (q, n, k) for q in SPEC.oc_q for n in SPEC.oc_naive_a for k in SPEC.oc_ratio}
    for r in rows:
        assert set(r["P"]) == {SELECTED, B_TB, B_NO_CONCLUSION, B_FA} and r["u"] == 0
        assert r["q_a"] == pytest.approx(r["q_b"] * r["ratio"]) and r["n_b"] == SPEC.j.stage2_n_b


def test_u0_row_is_the_plain_binomial_read_by_the_bands():
    """u = 0, naive_a 2, q_a = q_b: SELECTED = P(Bin(21,q) >= 11) * q^2, B_Tb = P(Bin <= 7)."""
    q = 0.6
    (r,) = l_oc.stage3_table((q,), (2,), (1.0,), SPEC)
    b = [comb(21, k) * q ** k * (1 - q) ** (21 - k) for k in range(22)]
    assert r["P"][SELECTED] == pytest.approx(sum(b[11:]) * q * q)
    assert r["P"][B_FA] == pytest.approx(sum(b[11:]) * (1 - q * q))
    assert r["P"][B_TB] == pytest.approx(sum(b[:8]))
    assert r["P"][B_NO_CONCLUSION] == pytest.approx(sum(b[8:11]))


def test_stage3_table_reads_with_the_spec_it_is_given():
    """A spec whose J bar needs 21/21 makes SELECTED all but impossible: the table reads through the argument."""
    strict = dataclasses.replace(SPEC, j=dataclasses.replace(SPEC.j, stage2_select_testable_b=21))
    (r,) = l_oc.stage3_table((0.6,), (4,), (1.0,), strict)
    (d,) = l_oc.stage3_table((0.6,), (4,), (1.0,), SPEC)
    assert r["P"][SELECTED] < d["P"][SELECTED] * 1e-3


def test_c3_rows_equal_j129s_oc_row(c3_rec):
    """The C3 turn-effect rows are stage2_oc's calibration on H.4's structure: same numbers as its own oc_row."""
    ub, ua = l_oc.c3_offsets(c3_rec)
    assert len(ub) == SPEC.j.stage2_n_b and len(ua) == 4
    rows = l_oc.stage3_table((0.5, 0.7), (len(ua),), (0.5, 1.0), SPEC, u_offsets=(ub, ua))
    s2 = l_oc.stage2_oc()
    for r in rows:
        assert r["u"] == "C3" and r["naive_a"] == 4 and sum(r["P"].values()) == pytest.approx(1.0)
        want = s2.oc_row(ub, ua, r["q_b"], r["q_a"])
        assert r["P"][SELECTED] == pytest.approx(want["p_selected"])
        assert r["P"][B_TB] == pytest.approx(want["p_b_tb"])
        assert r["P"][B_NO_CONCLUSION] == pytest.approx(want["p_b_no_conclusion"])
        assert r["P"][B_FA] == pytest.approx(want["p_b_fa"])


def test_c3_rows_refuse_a_naive_a_other_than_the_offsets(c3_rec):
    ub, ua = l_oc.c3_offsets(c3_rec)
    with pytest.raises(ValueError, match="naive_a"):
        l_oc.stage3_table((0.5,), (2, 4), (1.0,), SPEC, u_offsets=(ub, ua))
    with pytest.raises(ValueError, match="n_b"):
        l_oc.stage3_table((0.5,), (4,), (1.0,), SPEC, u_offsets=(ub[:-1], ua))


def test_stage2_oc_is_loaded_from_the_j_diag_script_not_copied():
    s2 = l_oc.stage2_oc()
    assert Path(s2.__file__).resolve() == (ROOT / "docs/superpowers/specs/j-diag/stage2_oc.py").resolve()
    assert not hasattr(l_oc, "_pb")


# ================================================================ stage 1: the simulated gate
def test_gate_oc_is_seeded_and_returns_probabilities():
    feats = _feats()
    a = l_oc.gate_oc(feats, 0.6, 0.0, SPEC, np.random.default_rng(1), draws=30)
    b = l_oc.gate_oc(feats, 0.6, 0.0, SPEC, np.random.default_rng(1), draws=30)
    assert a == b and sum(a.values()) == pytest.approx(1.0)
    assert set(a) == {SCREEN_GO, SCREEN_FEW, SCREEN_IMPRECISE, SCREEN_NO_RULE}


def test_gate_oc_extremes_hit_one_outcome():
    """q = 1, q_fail = 0: G alone is perfect on every fold (16 G-passing held-out pairs) -> GO every draw.
    q = 0: no pair testable, every rule has precision 0 -> IMPRECISE every draw."""
    feats = _feats()
    assert l_oc.gate_oc(feats, 1.0, 0.0, SPEC, np.random.default_rng(2), draws=5)[SCREEN_GO] == 1.0
    assert l_oc.gate_oc(feats, 0.0, 0.0, SPEC, np.random.default_rng(2), draws=5)[SCREEN_IMPRECISE] == 1.0


def test_gate_oc_does_not_mutate_the_features():
    feats = _feats()
    before = json.dumps(feats, sort_keys=True)
    l_oc.gate_oc(feats, 0.6, 0.1, SPEC, np.random.default_rng(3), draws=3)
    assert json.dumps(feats, sort_keys=True) == before


# ================================================================ the whole procedure
def test_whole_is_the_product():
    assert l_oc.whole(0.5, 0.2, 0.3) == pytest.approx(0.5 * 0.8 * 0.3)


def test_whole_table_one_row_per_cell_with_every_outcome_summing_to_one():
    gate = {(0.6, 0.0): {SCREEN_GO: 0.5, SCREEN_FEW: 0.2, SCREEN_IMPRECISE: 0.2, SCREEN_NO_RULE: 0.1}}
    short = {0.25: 0.46, 0.3: 0.18}
    s3 = l_oc.stage3_table((0.6,), (2, 3), (1.0,), SPEC)
    out = l_oc.whole_table(gate, short, s3)
    assert "independent" in out["assumption"]
    assert len(out["rows"]) == 1 * 2 * 2
    for r in out["rows"]:
        assert sum(r["P"].values()) == pytest.approx(1.0)
        s = next(x for x in s3 if x["naive_a"] == r["naive_a"])
        assert r["P"][SELECTED] == pytest.approx(l_oc.whole(0.5, short[r["c"]], s["P"][SELECTED]))
        assert r["P"]["COVERAGE_SHORT"] == pytest.approx(0.5 * short[r["c"]])
        assert r["P"][SCREEN_FEW] == 0.2 and (r["q"], r["q_fail"], r["u"]) == (0.6, 0.0, 0)
