"""Spec M.10.6: u = 0 (independent pairs) exact table; every cell read by M's own reading (so M.10.7 follows);
probabilities sum to 1."""
import dataclasses
from math import comb

import pytest

from flymon.brain import m_oc, m_rules
from flymon.brain.j_rules import B_FA, B_NO_CONCLUSION, B_TB, SELECTED
from flymon.brain.m_spec import SPEC


def _pb(q, n):
    return [comb(n, k) * q ** k * (1 - q) ** (n - k) for k in range(n + 1)]


def test_rows_cover_the_declared_grid_and_sum_to_one():
    rows = m_oc.stage3_table(SPEC)
    assert len(rows) == 4 * 2 * 9 * 3
    assert m_oc.OUTCOMES == (SELECTED, B_TB, B_NO_CONCLUSION, B_FA)
    for r in rows:
        assert set(r["P"]) == set(m_oc.OUTCOMES)
        assert sum(r["P"].values()) == pytest.approx(1.0)
        assert r["q_a"] == pytest.approx(r["ratio"] * r["q_b"])
    assert {(r["q_b"], r["ratio"], r["naive_a"], r["c"]) for r in rows} == {
        (q, x, a, c) for q in SPEC.oc_q_b for x in SPEC.oc_ratio for a in SPEC.oc_naive_a for c in SPEC.oc_c}
    assert "u = 0" in m_oc.ASSUMPTION


def test_a_cell_matches_the_binomial_by_hand():
    spec = dataclasses.replace(SPEC, oc_q_b=(0.5,), oc_ratio=(1.0,), oc_naive_a=(2,), oc_c=(7,))
    r = m_oc.stage3_table(spec)[0]
    pb = [comb(21, k) * 0.5 ** 21 for k in range(22)]
    pa2 = 0.25                                                  # both naive (a) pairs testable
    want = sum(pb[k] for k in range(11, 22)) * pa2              # n >= 11 > 7 = c and F_a >= 2
    assert r["P"]["SELECTED"] == pytest.approx(want)
    assert r["P"]["B_Tb"] == pytest.approx(sum(pb[k] for k in range(0, 8)))
    assert r["P"][B_FA] == pytest.approx(sum(pb[k] for k in range(11, 22)) * (1 - pa2))
    assert r["P"][B_NO_CONCLUSION] == pytest.approx(sum(pb[k] for k in range(8, 11)))   # c < n < 11


def test_naive_a_zero_can_never_select():
    spec = dataclasses.replace(SPEC, oc_naive_a=(0,))
    assert all(r["P"]["SELECTED"] == 0.0 for r in m_oc.stage3_table(spec))


def test_n_at_least_11_with_c_at_least_n_is_no_conclusion_not_b_tb():
    # M.10.7: c = 15 puts n in 11..15 under c >= n >= 11 -> B_NO_CONCLUSION; B_Tb only for n <= 10 (n <= c and n < 11)
    spec = dataclasses.replace(SPEC, oc_q_b=(0.6,), oc_ratio=(1.0,), oc_naive_a=(4,), oc_c=(15,))
    r = m_oc.stage3_table(spec)[0]
    pb, pa = _pb(0.6, 21), _pb(0.6, 4)
    assert r["P"][B_TB] == pytest.approx(sum(pb[:11]))
    assert r["P"][B_NO_CONCLUSION] == pytest.approx(sum(pb[11:16]))
    assert r["P"][B_NO_CONCLUSION] > 0.3                          # the overlap carries real mass at q_b = 0.6
    assert r["P"][SELECTED] == pytest.approx(sum(pb[16:]) * sum(pa[2:]))
    assert r["P"][B_FA] == pytest.approx(sum(pb[16:]) * sum(pa[:2]))


def test_every_cell_is_read_by_m_rules_stage3_reading(monkeypatch):
    calls = []
    real = m_rules.stage3_reading

    def spy(agg, agg_c3, spec):
        calls.append((agg["testable_b"], agg["F_a"], agg_c3["testable_b"]))
        return real(agg, agg_c3, spec)

    monkeypatch.setattr(m_oc, "stage3_reading", spy)
    spec = dataclasses.replace(SPEC, oc_q_b=(0.5,), oc_ratio=(1.0,), oc_naive_a=(3,), oc_c=(4,))
    m_oc.stage3_table(spec)
    assert sorted(calls) == sorted((n, f, 4) for n in range(22) for f in range(4))
