"""AD.11 record-only diagnosis helpers (flymon/ad/diag.py): fixtures only, no brain, no data files."""
import math

import numpy as np
import pytest

from flymon.ad import diag

PAIRS = [dict(cands=["A", "B"], best1="A", best2="B", mults1=[2.0, 1.0], mults2=[0.5, 2.0]),
         dict(cands=["A", "B", "C"], best1="C", best2="A", mults1=[0.5, 1.0, 2.0], mults2=[2.0, 0.0, 1.0])]
Z = {"A": (10.0, 2.0), "P": (20.0, 4.0)}


def rec(v00, v01, v10, v11, fly=0, a=None, p=None):
    sits = []
    for (i, side), v in (((0, 0), v00), ((0, 1), v01), ((1, 0), v10), ((1, 1), v11)):
        sits.append(dict(pair=i, side=side, v=list(v), a=list(a or [10] * len(v)) if a is None else a[(i, side)],
                         p=[20] * len(v) if p is None else p[(i, side)], pick=int(np.argmax(v)), correct=False))
    return dict(fly=fly, situations=sits)


def test_right_flags_follow_best_contrast_sign():
    right = rec([1.0, 0.0], [0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0])
    wrong = rec([0.0, 1.0], [1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    flat = rec([1.0, 0.0], [1.0, 0.0], [0.0, 0.5, 1.0], [0.0, 0.5, 1.0])
    assert diag.right_flags(right, PAIRS) == [1, 1]
    assert diag.right_flags(wrong, PAIRS) == [0, 0]
    assert diag.right_flags(flat, PAIRS) == [0, 0]          # d = 0 is not the right direction


def test_contrast_signs_counts_and_symmetry():
    s = diag.contrast_signs([1.0, -1.0, 0.0, 2.0])
    assert s["n"] == 4 and s["pos"] == 0.5 and s["neg"] == 0.25 and s["zero"] == 0.25
    assert s["mean"] == pytest.approx(0.5) and s["median"] == pytest.approx(0.5)
    # exchangeable sides: d = X0 - X1 with X0, X1 iid -> P(d > 0) close to 1/2
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 20000))
    assert diag.contrast_signs(x[0] - x[1])["pos"] == pytest.approx(0.5, abs=0.02)
    assert diag.contrast_signs([]) == dict(n=0)


def test_transitions_count_flips_between_consecutive_points():
    t = diag.transitions({0: [0, 0, 1, 1], 20: [1, 0, 1, 0], 40: [1, 1, 0, 0]})
    assert t["steps"]["0-20"] == dict(r2w=1, w2r=1, same=2)
    assert t["steps"]["20-40"] == dict(r2w=1, w2r=1, same=2)
    assert t["total"] == dict(r2w=2, w2r=2, same=4)
    with pytest.raises(ValueError):
        diag.transitions({0: [0, 1], 20: [1]})


def test_majority_disagreement():
    assert diag.majority_disagreement([[1, 0], [1, 0], [1, 1], [0, 0]]) == pytest.approx((1 + 1) / 8)
    assert diag.majority_disagreement([[1, 1], [1, 1]]) == 0.0


def test_readout_variants():
    a, p = [12, 10], [20, 28]
    assert diag.readout_v(a, p, Z, "zv") == pytest.approx([1.0, -2.0])
    assert diag.readout_v(a, p, Z, "a_only") == pytest.approx([1.0, 0.0])
    assert diag.readout_v(a, p, Z, "p_only") == pytest.approx([0.0, -2.0])
    assert diag.readout_v(a, p, Z, "raw") == pytest.approx([-8.0, -18.0])
    with pytest.raises(ValueError):
        diag.readout_v(a, p, Z, "nope")
    r = rec([0, 0], [0, 0], [0, 0, 0], [0, 0, 0], a={k: [12, 10] if k[0] == 0 else [10, 10, 10] for k in
                                                     ((0, 0), (0, 1), (1, 0), (1, 1))},
            p={k: [20, 28] if k[0] == 0 else [20, 20, 20] for k in ((0, 0), (0, 1), (1, 0), (1, 1))})
    assert diag.readout_variant(r, Z, "zv")["situations"][0]["v"] == pytest.approx([1.0, -2.0])


def test_delta_record_removes_the_naive_preference():
    r0 = rec([0.0, 2.0], [2.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0])     # naive: wrong at pair 0
    rt = rec([1.0, 2.0], [2.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0])     # learned: +1 on the right moves
    assert diag.right_flags(rt, PAIRS) == [0, 0]
    d = diag.delta_record(rt, r0)
    assert d["situations"][0]["v"] == pytest.approx([1.0, 0.0])
    assert diag.right_flags(d, PAIRS) == [1, 0]
    assert rt["situations"][0]["v"] == [1.0, 2.0]                          # input untouched


def test_class_means_and_class_model():
    r = rec([1.0, 0.0], [0.0, 1.0], [0.0, 0.0, 3.0], [3.0, 0.0, 0.0])
    m = diag.class_means([r], PAIRS)
    # pair 0 side 0: A super +0.5, B neutral -0.5; side 1: A not_very -0.5, B super +0.5; pair 1 side 0: C super +2
    assert m["super"] == pytest.approx((0.5 + 0.5 + 2.0 + 2.0) / 4)
    assert m["immune"] == pytest.approx(-1.0)
    # a readout that prefers super-effective moves predicts the right direction on both pairs; the reverse, wrong
    assert [d > 0 for d in diag.class_model_contrast(PAIRS, dict(super=1.0))] == [True, True]
    assert [d < 0 for d in diag.class_model_contrast(PAIRS, dict(super=-1.0))] == [True, True]
    assert diag.class_of(PAIRS[1], 1, 1) == "immune"
    pc = diag.per_class_contrast(r, PAIRS)
    assert pc == {"super|super": [1, 1]}


def test_amplify_and_keep_only():
    w0 = np.array([1.0, 1.0, 2.0, 1.0], np.float32)
    w = np.array([1.0, 0.8, 1.0, 0.5], np.float32)
    assert diag.amplify(w, w0, 1.0) == pytest.approx(w)
    assert diag.amplify(w, w0, 2.0) == pytest.approx([1.0, 0.6, 0.4, 0.2])      # 1 - 2*0.5 = 0 -> floor 0.2
    assert diag.amplify(w, w0, 2.0).dtype == np.float32
    assert diag.keep_only(w, w0, [True, False, True, False]) == pytest.approx([1.0, 1.0, 1.0, 1.0])
    assert diag.keep_only(w, w0, [False, True, False, True]) == pytest.approx([1.0, 0.8, 2.0, 0.5])


def test_ratio_summary():
    w0 = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    w = np.array([1.0, 0.2, 0.5, 0.9, 1.0])
    s = diag.ratio_summary(w, w0)
    assert s["n"] == 5 and s["floor"] == pytest.approx(0.2) and s["at_w0"] == pytest.approx(0.4)
    assert s["above_w0"] == 0.0 and s["mean_deficit"] == pytest.approx(1 - 0.72)
    assert diag.ratio_summary(w, w0, [False] * 5) == dict(n=0)
    assert diag.ratio_summary(w, w0, [True, True, False, False, False])["floor"] == 0.5


def test_recovery_helpers():
    assert diag.retention(0.001, 0) == 1.0
    assert diag.retention(0.001, 1000) == pytest.approx(math.exp(-1), rel=1e-3)
    # small r n: no difference; large r n: the deficit without recovery grows like r n
    assert diag.no_recovery_factor(0.001, 1) == pytest.approx(1.0, abs=1e-3)
    assert diag.no_recovery_factor(0.001, 3000) == pytest.approx(3.0 / (1 - math.exp(-3.0)))
    assert diag.no_recovery_factor(0.0, 100) == 1.0
    assert diag.use_gaps(["a", "b", "a", "a", "c", "b"]) == [2, 1, 4]


def test_pulse_rows_and_class_table():
    turns = [dict(battle_id="DL-f24-b000", turn=1, decision={}, outcome=dict(outcome=dict(effectiveness="super")),
                  reinforce=dict(pulses=[["PAM08", 400.0]])),
             dict(battle_id="DL-f24-b001", turn=1, decision={}, outcome=dict(outcome=dict(effectiveness="resisted")),
                  reinforce=dict(pulses=[["PAM08", 100.0], ["PPL105", 200.0]])),
             dict(battle_id="DL-f24-b025", turn=1, decision={}, outcome=dict(outcome=dict(effectiveness="super")),
                  reinforce=dict(pulses=[["PAM08", 200.0]])),
             dict(battle_id="DL-f24-b002", turn=2, decision={})]                 # no reinforce: skipped
    rows = diag.pulse_rows(turns)
    assert [r["b"] for r in rows] == [0, 1, 25]
    assert rows[1] == dict(b=1, eff="resisted", reward_ms=100.0, punish_ms=200.0, n_reward=1, n_punish=1)
    t = diag.pulse_class_table(rows, [0, 20, 40])
    assert t["0-20"]["super"]["reward_share"] == 1.0 and t["0-20"]["super"]["net_ms_per_turn"] == 400.0
    assert t["0-20"]["resisted"]["reward_share"] == pytest.approx(1 / 3)
    assert t["20-40"]["super"]["turns"] == 1


def test_jaccard():
    assert diag.jaccard([1, 2, 3], [2, 3, 4]) == pytest.approx(0.5)
    assert diag.jaccard([], []) == 1.0


def test_floor_tolerance_sees_floored_edges_after_a_recovery_pulse():
    assert diag.floor_tol(0.2, 0.001) == pytest.approx(0.2008, rel=1e-5)
    w0 = np.ones(4)
    w = np.array([0.2 + 0.001 * 0.8, 0.2, 0.2016, 1.0])         # floored then recovered once; floored; above; w0
    s = diag.ratio_summary(w, w0, r=0.001)
    assert s["floor_n"] == 2 and s["floor"] == 0.5 and s["min"] == pytest.approx(0.2)
    assert diag.ratio_summary(w, w0, r=0.0)["floor_n"] == 1


def test_fit_tau_recovers_a_known_time_constant():
    t = np.array([5.0, 15.0, 25.0, 35.0])
    w_eq = 0.75
    w = 1 - (1 - w_eq) * (1 - np.exp(-t / 14.5))
    f = diag.fit_tau(t, w, w_eq)
    assert f["tau"] == pytest.approx(14.5, abs=0.05) and f["sse"] < 1e-6
    assert f["frac"] == pytest.approx(list(1 - np.exp(-t / 14.5)))


def test_binom_two_sided():
    assert diag.binom_two_sided(15, 21) == pytest.approx(2 * sum(math.comb(21, i) for i in range(7)) / 2 ** 21)
    assert diag.binom_two_sided(15, 21) == pytest.approx(0.0784, abs=1e-3)
    assert diag.binom_two_sided(5, 10) == 1.0
