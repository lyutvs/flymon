"""AC.11 record-only diagnosis helpers (flymon/ac/diag.py): fixtures only, no brain, no data files."""
import math

import numpy as np
import pytest

from flymon.ac import diag

PAIRS = [dict(cands=["A", "B"], best1="A", best2="B", mults1=[2.0, 1.0], mults2=[0.5, 2.0]),
         dict(cands=["A", "B", "C"], best1="C", best2="A", mults1=[0.5, 1.0, 2.0], mults2=[2.0, 0.0, 1.0])]


def rec(v00, v01, v10, v11, fly=0):
    sits = []
    for (i, side), v in (((0, 0), v00), ((0, 1), v01), ((1, 0), v10), ((1, 1), v11)):
        p = PAIRS[i]
        pick = int(np.argmax(v))
        best = p["best1"] if side == 0 else p["best2"]
        sits.append(dict(pair=i, side=side, v=v, a=[1] * len(v), p=[side] * len(v), kc_active=[10] * len(v),
                         pick=pick, correct=p["cands"][pick] == best))
    return dict(fly=fly, situations=sits)


def test_unique_argmax_and_margin():
    assert diag.unique_argmax([1, 3, 2]) == 1 and diag.unique_argmax([3, 3, 1]) is None
    assert diag.margin([1.0, 3.0, 2.5]) == pytest.approx(0.5) and diag.margin([4.0]) == 0.0


def test_chance_switch_is_inverse_square_of_candidates():
    assert diag.chance_switch(PAIRS) == [0.25, pytest.approx(1 / 9)]


def test_best_contrast_sign_follows_opponent_conditioning():
    aligned = rec([1.0, 0.0], [0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0])
    blind = rec([1.0, 0.0], [1.0, 0.0], [0.0, 0.5, 1.0], [0.0, 0.5, 1.0])
    assert diag.best_contrast(aligned, PAIRS) == [2.0, 2.0]
    assert diag.best_contrast(blind, PAIRS) == [0.0, 0.0]
    assert diag.pick_changed(aligned, 2) == [1, 1] and diag.pick_changed(blind, 2) == [0, 0]
    assert diag.side_correct(aligned) == [1, 1, 1, 1]


def test_abs_side_delta_centres_within_a_side():
    r = rec([1.0, 0.0], [2.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    assert diag.abs_side_delta(r, 2, "v") == [1.0, 1.0, 0.0, 0.0, 0.0]
    assert diag.abs_side_delta(r, 2, "v", centre=True) == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert diag.abs_side_delta(r, 2, "p") == [1.0] * 5


def test_rule_agreement_and_conflict_split():
    items = [(0, diag.rule_picks([100, 80], [0.5, 2.0])),   # power says 0, mult says 1, chose power
             (1, diag.rule_picks([100, 80], [0.5, 2.0])),   # chose mult
             (0, diag.rule_picks([90, 90], [2.0, 1.0]))]    # power tied -> undefined
    out = diag.rule_agreement(items)
    assert out["power"] == dict(n=2, agree=0.5)
    assert out["mult"] == dict(n=3, agree=pytest.approx(2 / 3))
    assert out["conflict"]["n"] == 2 and out["conflict"]["power_share"] == 0.5 and out["conflict"]["mult_share"] == 0.5


def test_mult_class_and_log2_floor():
    assert [diag.mult_class(m) for m in (0, 0.25, 1, 4)] == ["immune", "not_very", "neutral", "super"]
    assert diag.log2_mult(0) == -3 and diag.log2_mult(4) == 2


def test_pulse_ms_and_join_turns():
    assert diag.pulse_ms([["PAM08", 100.0], ["PPL105", 200.0]]) == (100.0, 200.0)
    recs = [dict(kind="decision", decider="fly", battle_id="b", turn=1, chosen="x"),
            dict(kind="decision", decider="coach", battle_id="b", turn=2),
            dict(kind="outcome", battle_id="b", turn=1, outcome={}),
            dict(kind="reinforce", battle_id="b", turn=1, pulses=[])]
    j = diag.join_turns(recs)
    assert len(j) == 1 and set(j[0]) >= {"decision", "outcome", "reinforce"}


def test_spearman_handles_ties_and_constants():
    assert diag.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert diag.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert diag.spearman([1, 1, 1], [1, 2, 3]) is None


def test_glom_jaccard_from_codebook():
    class CB:
        words = {("W", "X"): (1, 2), ("W", "Y"): (3, 4), ("W", "Z"): (1, 5)}

        def word(self, m, t):
            return self.words[(m, t)]
    assert diag.glom_jaccard(CB(), "W", ["X"], ["Y"]) == 0.0
    assert diag.glom_jaccard(CB(), "W", ["X", "Y"], ["X", "Z"]) == pytest.approx(2 / 5)


def test_weight_summary_counts_changes_and_floor():
    w0 = np.array([1.0, 1.0, 1.0, 1.0])
    s = diag.weight_summary(np.array([1.0, 0.1, 0.5, 1.0]), w0)
    assert s["n_changed"] == 2 and s["frac_changed"] == 0.5 and s["frac_floor_of_changed"] == 0.5
    assert s["frac_up_of_changed"] == 0.0 and math.isclose(s["median_ratio_changed"], 0.3)


def test_noise_spread_same_input():
    a = rec([1.0, 0.0], [1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], fly=0)
    b = rec([0.0, 1.0], [1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], fly=1)
    s = diag.noise_spread([a, b], 2)
    assert s["n_flies"] == 2 and s["modal_pick_share"] == pytest.approx((0.5 + 1 + 1 + 1) / 4)
    assert s["v_sd_across_flies_same_input"] > 0


def test_empirical_null_uses_each_sides_pick_distribution():
    a = rec([1.0, 0.0], [0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], fly=0)   # both pairs switched
    b = rec([1.0, 0.0], [1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], fly=1)   # neither
    out = diag.empirical_null([a, b], PAIRS)
    assert out["per_pair"] == [pytest.approx(1.0 * 0.5), pytest.approx(1.0 * 0.5)] and out["mean"] == 0.5


def test_crossed_boot_shares_pairs_and_pairs_flies():
    ua = {0: [1, 0, 1, 0], 1: [1, 0, 1, 0]}
    ub = {0: [1, 0, 1, 0], 1: [1, 0, 1, 0]}
    res = diag.crossed_boot(ua, ub, 200, 1, paired=True)
    assert res["diff"] == 0 and res["lo"] == 0 and res["hi"] == 0     # shared pairs + shared flies -> zero spread
    res = diag.crossed_boot(ua, {5: [0, 0, 0, 0]}, 200, 1)
    assert res["diff"] == 0.5 and res["lo"] < 0.5 < res["hi"]           # pair resampling alone gives spread
    with pytest.raises(ValueError):
        diag.crossed_boot(ua, {5: [0, 0, 0, 0]}, 10, 1, paired=True)


def test_fisher_ci_brackets_r():
    lo, hi = diag.fisher_ci(-0.27, 20)
    assert lo == pytest.approx(-0.636, abs=0.01) and hi == pytest.approx(0.195, abs=0.01)
    assert diag.fisher_ci(0.5, 3) == (None, None)
