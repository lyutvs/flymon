# tests/brain/test_p_rules.py
"""Spec P.3 / P.6.2 / P.6.5: the gate (O.7.5's validity + the declared pair, point and arm flags + a positive c1), arm 2's
sha256 plumbing, the per-direction rule (i)-(iii) and the labels in P.6.2's order, one seed bootstrap for both
directions; mutation tests (one direction zeroed, one reversed, arm 2's weights moved)."""
import json
import random
from dataclasses import replace

import numpy as np
import pytest

from flymon.brain.h4_formula import dv
from flymon.brain.n_rules import INVALID, NO_LEARNING, _ci
from flymon.brain.o_rules import INCONCLUSIVE, JUDGED, boot_weights
from flymon.brain.p_rules import (CONFIRMED, DIRECTION_DEPENDENT, EXCLUDED, LEARNS_CONFIRMATORY, UNDECIDED,
                                  arm_vectors, group, judge_vectors, label, p_judge, v_of)
from flymon.brain.p_spec import SPEC

T = replace(SPEC, o=replace(SPEC.o, o2_n_seeds=16, boot_draws=2000))
ZU = {"A": [0.0, 1.0], "P": [0.0, 1.0]}
C1 = 0.608


def _probe(seed, A=20.0, P=10):
    return dict(seed=seed, A=A, P=P, kc_frac=0.05, kc_spikes=100, kc_max_win_hz=50.0, apl_out_per_step=0.1,
                wall_s=0.01, steps=1400)


def make_rows(effect=None, y_effect=None, spec=T, nonassoc=2.0, jitter=0.2, flip=(), silent_pre=()):
    """With ZU (V = A - P): the plastic arm lowers A_X by `nonassoc`; the punish arm by nonassoc + effect[d] + j_i;
    y_effect[d] lowers the punish arm's A_Y by y_effect[d] + j_i. So ds = -(effect + j), dt = -(y_effect + j) and
    ℓ_d = effect - y_effect. `flip`: seed indices whose plastic-arm X post P is 0; `silent_pre`: seed indices whose
    probes are all silent (P 0)."""
    effect = effect or {"r1": 2.4, "r2": 2.4}
    y_effect = y_effect or {}
    out = []
    for d, (x, y) in spec.pairs().items():
        for a in spec.arms:
            pu, pl, dz = spec.flags()[a]
            for i, s in enumerate(spec.seeds):
                j = jitter * ((i % 4) - 1.5)
                ax = {"plastic": nonassoc, "frozen": 0.0, "punish": nonassoc + effect[d] + j}[a]
                ay = y_effect[d] + j if (a == "punish" and d in y_effect) else 0.0
                p0 = 0 if i in silent_pre else 10
                px = 0 if (a == "plastic" and i in flip) else p0
                out.append(dict(direction=d, x=x, y=y, seed=s, arm=a, punish=pu, plastic=pl, da_zero=dz, edit="none",
                                csc_sha256="sha-none", point=list(spec.o.o2_point),
                                pre={"x": _probe(s, P=p0), "y": _probe(s, P=p0)},
                                post={"x": _probe(s, A=20.0 - ax, P=px), "y": _probe(s, A=20.0 - ay, P=p0)},
                                w0_sha256="w0", w_post_sha256="w0" if a == "frozen" else "w1",
                                weights_frac=1.0 if a == "frozen" else 0.9, weights_frac_A=0.8, weights_frac_P=1.0,
                                da_integral={"PPL105": 1.0 if pu else 0.0}, wall_s=1.0))
    return out


def test_v_of_is_h4_formulas_v():
    z = {"A": [10.0, 4.0], "P": [20.0, 5.0]}
    x, y = _probe(1, A=30.0, P=12), _probe(1, A=7.0, P=40)
    assert v_of(x, z) - v_of(y, z) == pytest.approx(dv({"A": [[30.0, 7.0]], "P": [[12, 40]]}, z)[0])


def test_vectors_are_the_declared_differences():
    by = group(make_rows(), T)
    v = arm_vectors(by["r1"], ZU, T)
    j = 0.2 * (np.arange(16) % 4 - 1.5)
    assert v["seeds"] == list(T.seeds)
    assert np.allclose(v["ds"], -(2.4 + j)) and np.allclose(v["dt"], 0.0)
    assert np.allclose(v["dl"], v["dt"] - v["ds"])                          # ℓ = t - s per seed


def test_both_directions_learn():
    res = p_judge(make_rows(), ZU, C1, T)
    assert res["outcome"] == JUDGED and res["label"] == LEARNS_CONFIRMATORY
    for d in ("r1", "r2"):
        s = res["directions"][d]
        assert s["outcome"] == CONFIRMED and s["i"] and s["ii"] and s["iii"]
        assert s["ell"] == pytest.approx(2.4) and s["s"] == pytest.approx(-2.4) and s["t"] == pytest.approx(0.0)
    assert res["mean_ell"]["ell"] == pytest.approx(2.4) and res["mean_ell"]["record"] is True


def test_mutation_one_direction_zeroed_is_direction_dependent():
    res = p_judge(make_rows(effect={"r1": 2.4, "r2": 0.0}), ZU, C1, T)
    assert res["directions"]["r2"]["outcome"] == EXCLUDED and res["label"] == DIRECTION_DEPENDENT


def test_mutation_one_direction_reversed_never_passes():
    res = p_judge(make_rows(effect={"r1": 2.4, "r2": -2.4}), ZU, C1, T)
    assert res["directions"]["r2"]["outcome"] == EXCLUDED and res["label"] == DIRECTION_DEPENDENT
    assert res["mean_ell"]["ell"] == pytest.approx(0.0, abs=1e-9)          # the average alone hides the reversal


def test_no_effect_anywhere_is_no_learning():
    assert p_judge(make_rows(effect={"r1": 0.0, "r2": 0.0}), ZU, C1, T)["label"] == NO_LEARNING


def test_a_small_noisy_effect_is_inconclusive():
    res = p_judge(make_rows(effect={"r1": 0.5, "r2": 0.5}, jitter=1.0), ZU, C1, T)
    assert {s["outcome"] for s in res["directions"].values()} == {UNDECIDED} and res["label"] == INCONCLUSIVE


def test_a_change_spread_onto_y_fails_concentration():
    res = p_judge(make_rows(effect={"r1": 3.0, "r2": 2.4}, y_effect={"r1": 2.0}), ZU, C1, T)
    s = res["directions"]["r1"]
    assert s["ell"] == pytest.approx(1.0) and s["i"] and s["ii"] and not s["iii"]
    assert s["outcome"] == UNDECIDED and res["label"] == INCONCLUSIVE


def test_x_not_lowered_itself_fails_criterion_ii():
    res = p_judge(make_rows(effect={"r1": 0.0, "r2": 2.4}, y_effect={"r1": -2.0}), ZU, C1, T)
    s = res["directions"]["r1"]
    assert s["ell"] == pytest.approx(2.0) and s["i"] and not s["ii"] and s["outcome"] == UNDECIDED
    assert res["label"] == INCONCLUSIVE


def test_mutation_frozen_weights_moved_is_invalid():
    rows = make_rows()
    for r in rows:
        if r["direction"] == "r1" and r["arm"] == "frozen" and r["seed"] == T.seeds[3]:
            r["w_post_sha256"] = "moved"
    res = p_judge(rows, ZU, C1, T)
    assert res["outcome"] == res["label"] == INVALID and res["invalid_directions"] == ["r1"]
    assert "plumbing" in res["reasons"][0] and str(T.seeds[3]) in res["reasons"][0]


@pytest.mark.parametrize("mutate, word", [
    (lambda rows: rows.pop(), "no row"),
    (lambda rows: rows.append(dict(rows[0])), "more than once"),
    (lambda rows: rows[0]["post"]["x"].update(A=float("nan")), "non-finite"),
    (lambda rows: rows[0].update(seed=99), "undeclared seed"),
    (lambda rows: rows[0].update(x="1:4"), "(X, Y)"),
    (lambda rows: rows[0].update(punish=not rows[0]["punish"]), "flags"),
    (lambda rows: rows[0].update(point=[1.0, 1.0]), "operating point"),
    (lambda rows: rows[0].update(edit="apl_to_kc_zero"), "edit"),
])
def test_the_gate_names_each_defect(mutate, word):
    rows = make_rows()
    mutate(rows)
    res = p_judge(rows, ZU, C1, T)
    assert res["outcome"] == INVALID and word in " ".join(res["reasons"])


@pytest.mark.parametrize("c1", [float("nan"), 0.0, -0.6, None])
def test_a_missing_or_nonpositive_c1_is_invalid(c1):
    res = p_judge(make_rows(), ZU, c1, T)
    assert res["outcome"] == INVALID and "c1" in " ".join(res["reasons"])


def test_row_order_does_not_change_the_result():
    rows = make_rows(effect={"r1": 2.4, "r2": 0.9}, jitter=0.8)
    shuffled = list(rows)
    random.Random(7).shuffle(shuffled)
    a, b = p_judge(rows, ZU, C1, T), p_judge(shuffled, ZU, C1, T)
    assert json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


def test_label_order():
    assert label({"r1": CONFIRMED, "r2": CONFIRMED}) == LEARNS_CONFIRMATORY
    assert label({"r1": CONFIRMED, "r2": EXCLUDED}) == label({"r1": EXCLUDED, "r2": CONFIRMED}) == DIRECTION_DEPENDENT
    assert label({"r1": EXCLUDED, "r2": EXCLUDED}) == NO_LEARNING
    for pair in ((CONFIRMED, UNDECIDED), (UNDECIDED, EXCLUDED), (UNDECIDED, UNDECIDED)):
        assert label(dict(zip(("r1", "r2"), pair))) == INCONCLUSIVE


def test_one_bootstrap_for_both_directions():
    rows = make_rows(effect={"r1": 2.4, "r2": 1.5}, jitter=0.9)
    res = p_judge(rows, ZU, C1, T)
    by = group(rows, T)
    vec = {d: arm_vectors(by[d], ZU, T) for d in T.directions}
    W = boot_weights(16, T.o.boot_draws, T.o.boot_seed)
    for d in T.directions:
        assert res["directions"][d]["ell_ci"] == _ci(W @ vec[d]["dl"], T.o.ci_level)
    assert res["mean_ell"]["ci"] == _ci(W @ np.mean([vec["r1"]["dl"], vec["r2"]["dl"]], axis=0), T.o.ci_level)


def test_directions_must_share_their_seeds():
    v = dict(seeds=[1, 2], dl=np.ones(2), ds=-np.ones(2), dt=np.zeros(2))
    with pytest.raises(ValueError):
        judge_vectors({"r1": v, "r2": dict(v, seeds=[1, 3])}, C1, T)


def test_zero_variance_effects_are_judged_not_crashed():
    assert p_judge(make_rows(jitter=0.0), ZU, C1, T)["label"] == LEARNS_CONFIRMATORY
    res = p_judge(make_rows(effect={"r1": 0.0, "r2": 0.0}, jitter=0.0), ZU, C1, T)
    assert res["label"] == NO_LEARNING and res["directions"]["r1"]["ell_ci"] == [0.0, 0.0]
