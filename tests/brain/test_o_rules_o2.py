# tests/brain/test_o_rules_o2.py
"""Spec O.7.4 as amended by O.7.8: depression with its b band and INCONCLUSIVE; dopamine dependence is the record
DA_DEPENDENT_BY_CONSTRUCTION after DEPRESSION_PRESENT (m4 and its CI recorded, never a label); the associative
component with e = 0.25 |o|; the state-flip guard (> 1/8 of seeds -> the same judgement without them, changed labels
suffixed); plumbing by sha256 on arms 2 and 4 (that X INVALID); the gate before anything (the stage INVALID)."""
from dataclasses import replace

import pytest

from flymon.brain import o_rules as R
from flymon.brain.o_spec import SPEC

S0 = 500
Z = {"A": [0.0, 1.0], "P": [0.0, 1.0]}                     # dV = (A_x - P_x) - (A_y - P_y)
O = dict(SPEC.oracle_o)


def _spec(n=16, draws=500):
    return replace(SPEC, o2_seed0=S0, o2_n_seeds=n, boot_draws=draws)


def _p(A, P=40):
    return dict(seed=0, A=A, P=P, kc_frac=0.05, kc_spikes=100, kc_max_win_hz=50.0, apl_out_per_step=0.1, wall_s=0.01,
                steps=1400)


def _rows(spec, post_A, pre_A=20.0, post_P=lambda x, a, i: 40, moved=lambda x, a, i: a in ("plastic", "punish")):
    out = []
    for x, y, _ in spec.o2_pairs:
        for a, pu, pl, dz in spec.o2_arms:
            for i, s in enumerate(spec.o2_seeds):
                out.append(dict(seed=s, edit="none", arm=a, punish=pu, plastic=pl, da_zero=dz, csc_sha256="sha-none",
                                pre={"x": _p(pre_A), "y": _p(20.0)},
                                post={"x": _p(post_A(x, a, i), post_P(x, a, i)), "y": _p(20.0)},
                                weights_frac=0.9, weights_frac_A=0.8, weights_frac_P=1.0, w0_sha256="w0",
                                w_post_sha256="w1" if moved(x, a, i) else "w0",
                                da_integral={"PPL105": 1.0, "PAM08": 0.1}, wall_s=1.0, x=x, y=y))
    return out


def _A(plastic, punish=None, da_zero=0.0):
    """post A_X: 20 minus the arm's drop; plastic and punish carry a 0.5 jitter on odd seeds."""
    drop = {"plastic": plastic, "frozen": 0.0, "punish": plastic if punish is None else punish, "da_zero": da_zero}
    return lambda x, a, i: 20.0 - drop[a] - (0.5 * (i % 2) if a in ("plastic", "punish") else 0.0)


def _x(res, x="4:1"):
    return res["x"][x]


def test_depression_present_and_dopamine_dependence_recorded_by_construction():
    sp = _spec()
    res = R.o2_judge(_rows(sp, _A(10.0)), Z, O, sp)
    v = _x(res)
    assert res["outcome"] == R.JUDGED and v["outcome"] == R.JUDGED
    assert v["labels"] == dict(depression=R.DEPRESSION_PRESENT, separable=R.NOT_SEPARABLE)   # O.7.8: no da label
    assert v["record"]["da"]["label"] == R.DA_DEPENDENT_BY_CONSTRUCTION
    assert v["record"]["da"]["m4"] == 0.0 and v["record"]["da"]["m4_ci"] == [0.0, 0.0]
    assert v["stats"]["b"] == 5.0 and v["stats"]["m"] == pytest.approx(-10.25) and v["stats"]["m4_ci"] == [0.0, 0.0]
    assert v["record"]["frozen_dA_all_zero"] and v["record"]["da_zero_weights_unmoved"]
    assert v["guard"] is None and v["o"] == -2.348 and res["x"]["dDL"]["o"] == -2.433
    assert not hasattr(R, "DA_DEPENDENT") and not hasattr(R, "DA_INDEPENDENT")


def test_da_zero_arm_moving_its_weights_invalidates_that_x():
    """O.7.8: arm 4's post-training sha256 == w0's is an implementation check (that X INVALID)."""
    sp = _spec()
    moved = lambda x, a, i: a in ("plastic", "punish") or (x == "4:1" and a == "da_zero" and i == 5)
    res = R.o2_judge(_rows(sp, _A(10.0, da_zero=10.0), moved=moved), Z, O, sp)
    assert res["outcome"] == R.JUDGED
    v = _x(res)
    assert v["outcome"] == R.INVALID and v["labels"] is None
    assert "plumbing" in v["reasons"][0] and "da_zero" in v["reasons"][0] and f"seed {S0 + 5}" in v["reasons"][0]
    assert _x(res, "dDL")["outcome"] == R.JUDGED


def test_no_depression_and_no_da_record():
    v = _x(R.o2_judge(_rows(_spec(), _A(0.0)), Z, O, _spec()))
    assert v["labels"]["depression"] == R.NO_DEPRESSION and v["record"]["da"]["label"] is None
    assert "da" not in v["labels"]


def test_inconclusive_when_the_ci_straddles_b():
    post = lambda x, a, i: 20.0 - (9.0 if a in ("plastic", "punish") and i % 2 == 0 else 0.0)
    v = _x(R.o2_judge(_rows(_spec(), post), Z, O, _spec()))
    assert v["stats"]["m"] == pytest.approx(-4.5) and v["labels"]["depression"] == R.INCONCLUSIVE
    assert v["record"]["da"]["label"] is None


def test_separable_and_its_inconclusive_band():
    sp = _spec()
    v = _x(R.o2_judge(_rows(sp, _A(10.0, punish=20.0)), Z, O, sp))
    assert v["labels"]["separable"] == R.SEPARABLE and v["stats"]["D"] == pytest.approx(-10.0)
    assert v["stats"]["D_sign"] == -1 and v["stats"]["e"] == pytest.approx(0.25 * 2.348)
    post = lambda x, a, i: 20.0 - 10.0 - (1.0 if a == "punish" and i % 2 == 0 else 0.0) if a in ("plastic", "punish") \
        else 20.0
    w = _x(R.o2_judge(_rows(sp, post), Z, O, sp))
    assert w["stats"]["D"] == pytest.approx(-0.5) and w["labels"]["separable"] == R.INCONCLUSIVE


def test_plumbing_failure_invalidates_only_that_x():
    sp = _spec()
    moved = lambda x, a, i: a in ("plastic", "punish") or (x == "4:1" and a == "frozen" and i == 3)
    res = R.o2_judge(_rows(sp, _A(10.0), moved=moved), Z, O, sp)
    assert res["outcome"] == R.JUDGED
    assert _x(res)["outcome"] == R.INVALID and _x(res)["labels"] is None and "plumbing" in _x(res)["reasons"][0]
    assert "frozen" in _x(res)["reasons"][0]
    assert _x(res, "dDL")["outcome"] == R.JUDGED


def test_flip_guard_suffixes_the_labels_that_change():
    sp = _spec()                                                          # 16 seeds: limit 2
    post_A = lambda x, a, i: 0.0 if a in ("plastic", "punish") and i < 4 else 20.0
    post_P = lambda x, a, i: 0 if a in ("plastic", "punish") and i < 4 else 40
    v = _x(R.o2_judge(_rows(sp, post_A, post_P=post_P), Z, O, sp))
    assert v["stats"]["m"] == -5.0 and v["flip_limit"] == 2.0 and len(v["flips"]["plastic"]) == 4
    assert v["labels"]["depression"] == R.DEPRESSION_PRESENT + R.FLIP_SUFFIX
    assert set(v["labels"]) == {"depression", "separable"}                 # O.7.8: the da record is not compared
    assert v["labels"]["separable"] == R.NOT_SEPARABLE                     # unchanged without the flipped seeds
    assert v["guard"]["dropped"] == [S0, S0 + 1, S0 + 2, S0 + 3]
    assert v["guard"]["alt"]["labels"]["depression"] == R.NO_DEPRESSION
    assert v["record"]["da"]["label"] == R.DA_DEPENDENT_BY_CONSTRUCTION    # the record reads judgement 1 as computed


def test_flips_at_the_limit_do_not_trigger_the_guard():
    sp = _spec()
    post_P = lambda x, a, i: 0 if a == "plastic" and i < 2 else 40
    v = _x(R.o2_judge(_rows(sp, _A(10.0), post_P=post_P), Z, O, sp))
    assert v["guard"] is None and v["labels"]["depression"] == R.DEPRESSION_PRESENT


def test_a_guard_leaving_fewer_than_two_seeds_marks_every_label():
    sp = _spec(n=4)
    post_P = lambda x, a, i: 0 if a == "plastic" and i < 3 else 40
    v = _x(R.o2_judge(_rows(sp, _A(10.0), post_P=post_P), Z, O, sp))
    assert v["guard"]["alt"] is None
    assert all(l.endswith(R.FLIP_SUFFIX) for l in v["labels"].values() if l is not None)


def test_zero_baseline_is_judged_not_crashed():
    sp = _spec()
    v = _x(R.o2_judge(_rows(sp, lambda x, a, i: 0.0, pre_A=0.0), Z, O, sp))
    assert v["stats"]["b"] == 0.0 and v["labels"]["depression"] == R.NO_DEPRESSION


def test_state_conditional_record_keeps_all_firing_seeds():
    sp = _spec()
    post_P = lambda x, a, i: 0 if a == "punish" and i == 0 else 40
    v = _x(R.o2_judge(_rows(sp, _A(10.0), post_P=post_P), Z, O, sp))
    assert v["record"]["state_conditional"]["n"] == 15
    assert v["record"]["da_integral"]["punish"] == {"PPL105": 1.0, "PAM08": 0.1}
    wf = v["record"]["weights_frac"]["plastic"]
    assert wf == {"all": 0.9, "A_punish_core": 0.8, "P_reward_core": 1.0}
    assert "core" in v["record"]["weights_frac_note"]


def test_the_gate_judges_nothing():
    sp = _spec()
    rows = _rows(sp, _A(10.0))
    for bad, why in ((rows[1:], "no row"), ([dict(rows[0], seed=1)] + rows[1:], "undeclared seed"),
                     ([dict(rows[0], weights_frac=float("inf"))] + rows[1:], "non-finite"),
                     ([dict(rows[0], edit="apl_to_kc_zero", csc_sha256="sha-kc")] + rows[1:], "does not declare")):
        res = R.o2_judge(bad, Z, O, sp)
        assert res["outcome"] == R.INVALID and res["x"] is None and any(why in r for r in res["reasons"])


def test_a_missing_or_non_finite_oracle_judges_nothing():
    sp = _spec()
    rows = _rows(sp, _A(10.0))
    for o, why in (({"sim": -2.348}, "dis"), ({"sim": -2.348, "dis": float("nan")}, "dis")):
        res = R.o2_judge(rows, Z, o, sp)
        assert res["outcome"] == R.INVALID and res["x"] is None and any(why in r for r in res["reasons"])


def test_too_few_seeds_is_invalid_not_judged():
    sp = _spec(n=1)
    res = R.o2_judge(_rows(sp, _A(10.0)), Z, O, sp)
    assert res["outcome"] == R.JUDGED and _x(res)["outcome"] == R.INVALID and _x(res)["labels"] is None


def test_sentences():
    sp = _spec()
    s = R.sentence("o2", R.o2_judge(_rows(sp, _A(10.0)), Z, O, sp))
    assert s.startswith("O2: X = 4:1: DEPRESSION_PRESENT, NOT_SEPARABLE") and "학습 주장이 아니다" in s
    assert "DA_DEPENDENT_BY_CONSTRUCTION" in s and "기록" in s
    assert "INVALID" in R.sentence("o1", dict(outcome=R.INVALID, reasons=["x"]))
    with pytest.raises(ValueError):
        R.sentence("o3", dict(outcome=R.JUDGED))
