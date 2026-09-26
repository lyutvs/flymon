# tests/brain/test_l_screen.py
"""Spec L.11.2: features, candidate rules, choice with coverage >= 0.35, leave-one-turn-out gate, result codes."""
import numpy as np
import pytest

from flymon.brain import l_screen as ls
from flymon.brain.l_spec import SPEC


def P(turn, testable, G=True, S=0.5, f2=10.0, f3=0.3, set_="even"):
    return {"turn": turn, "set": set_, "testable": testable, "feat": dict(G=G, S=S, f2=f2, f3=f3)}


def test_features_guard_s_f2_f3():
    fx = np.array([1.0, 0.5, 0.0, 0.25]); fy = np.array([0.0, 0.5, 0.5, 0.0])
    w13 = np.array([2.0, 1.0, 0.0, 2.0]); w05 = np.zeros(4)
    pre = {"MBON13": [[6, 9], [5, 7]], "MBON05": [[5, 5], [8, 6]]}
    f = ls.features(fx, fy, pre, w13, w05, SPEC)
    assert f["G"] is True
    assert f["S"] == pytest.approx((2 * 1.0 + 2 * 0.25) / (2 * 1.0 + 1 * 0.5 + 2 * 0.25))
    assert f["f2"] == 5.5                                   # min(median(6, 5), median(9, 7)) = min(5.5, 8)
    assert f["f3"] == pytest.approx(1 / 4)                  # both fire on KC 1; union {0, 1, 2, 3}
    pre_low = {"MBON13": [[6, 9], [4, 7]], "MBON05": [[5, 5], [8, 6]]}
    assert ls.features(fx, fy, pre_low, w13, w05, SPEC)["G"] is False


def test_passes_respects_direction():
    feat = dict(G=True, S=0.4, f2=12.0, f3=0.2)
    assert ls.passes({"family": "S", "op": ">", "t": 0.3}, feat)
    assert not ls.passes({"family": "f3", "op": "<", "t": 0.2}, feat)       # strict
    assert ls.passes({"family": "f3", "op": "<", "t": 0.25}, feat)
    assert not ls.passes({"family": "G"}, dict(feat, G=False))
    assert not ls.passes(None, feat)


def test_choose_prefers_precision_then_coverage_then_family():
    tr = [P(0, True, f2=20), P(0, True, f2=18), P(2, False, f2=3), P(2, False, f2=4), P(4, True, f2=16),
          P(4, False, f2=15), P(6, False, G=False)]
    c = ls.choose(tr, SPEC)
    assert c["rule"] == {"family": "f2", "op": ">", "t": 15.5} and c["precision"] == 1.0
    assert c["coverage"] == pytest.approx(3 / 7)


def test_choose_returns_none_below_coverage():
    tr = [P(t, False, G=(t == 0)) for t in range(10)]
    assert ls.choose(tr, SPEC) is None                      # G alone covers 1/10 < 0.35


def test_loto_holds_out_whole_turns_and_empty_folds_predict_fail():
    pairs = [P(0, True, f2=20), P(0, True, f2=19), P(2, False, f2=3), P(4, True, f2=17), P(6, False, f2=2)]
    r = ls.loto(pairs, SPEC)
    assert [f["turn"] for f in r["folds"]] == [0, 2, 4, 6]
    assert all(len(f["predicted"]) == sum(p["turn"] == f["turn"] for p in pairs) for f in r["folds"])
    few = [P(0, True, G=True), P(2, False, G=False), P(4, False, G=False)]
    r2 = ls.loto(few, SPEC)
    assert r2["folds"][0]["rule"] is None and r2["folds"][0]["predicted"] == [False]


@pytest.mark.parametrize("n_true,n_false,expect", [(12, 2, ls.SCREEN_GO), (4, 4, ls.SCREEN_FEW),
                                                   (7, 9, ls.SCREEN_IMPRECISE)])
def test_gate_outcomes(n_true, n_false, expect):
    pairs = [P(2 * i, True, f2=20) for i in range(n_true)] + [P(2 * i + 1, False, f2=20) for i in range(n_false)]
    pairs += [P(100 + i, False, G=False) for i in range(3)]
    assert ls.gate(pairs, SPEC)["outcome"] == expect


def test_gate_no_rule():
    pairs = [P(i, False, G=False) for i in range(10)]
    g = ls.gate(pairs, SPEC)
    assert g["outcome"] == ls.SCREEN_NO_RULE and g["final"] is None


def test_auc_mann_whitney():
    assert ls.auc([6, 7, 5], [True, True, False], higher_passes=True) == 1.0
    assert ls.auc([6, 7, 5], [True, True, False], higher_passes=False) == 0.0
    assert ls.auc([6, 5, 5], [True, True, False], higher_passes=True) == 0.75


@pytest.mark.parametrize("fam,op,below,at,above", [("S", ">", False, False, True), ("f2", ">", False, False, True),
                                                   ("f3", "<", True, False, False)])
def test_passes_boundaries_are_strict_in_the_declared_direction(fam, op, below, at, above):
    t = {"S": 0.4, "f2": 12.0, "f3": 0.2}[fam]
    rule = {"family": fam, "op": op, "t": t}
    base = dict(G=True, S=0.4, f2=12.0, f3=0.2)
    assert ls.passes(rule, dict(base, **{fam: t - 0.01})) is below
    assert ls.passes(rule, dict(base, **{fam: t})) is at
    assert ls.passes(rule, dict(base, **{fam: t + 0.01})) is above
    assert not ls.passes(rule, dict(base, G=False, **{fam: t + 0.01 if op == ">" else t - 0.01}))


def test_rules_for_uses_declared_directions_and_g_passing_midpoints():
    tr = [P(0, True, S=0.2, f3=0.5), P(2, False, S=0.6, f3=0.1), P(4, False, G=False, S=0.9, f3=0.9)]
    rs = ls.rules_for(tr, SPEC)
    assert rs[0] == {"family": "G"}
    assert {"family": "S", "op": ">", "t": pytest.approx(0.4)} in rs
    assert {"family": "f3", "op": "<", "t": pytest.approx(0.3)} in rs
    assert not any(r.get("t") == pytest.approx(0.75) for r in rs)          # the G-failing pair adds no threshold
    f3 = [r["t"] for r in ls.rules_for([P(i, True, f3=v) for i, v in enumerate([0.1, 0.5, 0.3])], SPEC)
          if r["family"] == "f3"]
    assert f3 == sorted(f3, reverse=True)                                     # f3 (<): descending t


def test_zero_loto_passes_give_precision_none_and_few():
    pairs = [P(0, True, G=True), P(2, False, G=False), P(4, False, G=False)]
    r = ls.loto(pairs, SPEC)
    assert r["n_pass"] == 0 and r["precision"] is None
    import dataclasses
    g = ls.gate(pairs, dataclasses.replace(SPEC, gate_min_passes=0, cov_min_stage1=0.0))
    assert g["loto"]["precision"] is None and g["outcome"] == ls.SCREEN_FEW


def test_gate_records_fold_without_rule_as_none():
    pairs = [P(0, True, f2=20), P(1, True, f2=20)] + [P(2 + i, False, G=False) for i in range(3)]
    g = ls.gate(pairs, SPEC)
    assert len(g["loto"]["folds"]) == 5
    assert g["records"]["loto_rule_counts"] == {"G": 3, "none": 2}
