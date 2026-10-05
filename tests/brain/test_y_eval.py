"""Y's generator and evaluation (Y.6.1, Y.6.3, Y.5 code, Y.6.6, Y.9.2 P1-4 / P1-5 / P3-14): the bit-identity test
against x_oc.evaluate, the near shift keeps balance, fills and their exclusion, the experiment-level corner draw,
qualify over both k ranges and every k, selection in budget, pick ties, the synthetic validation, and the mutations."""
import dataclasses

import numpy as np
import pytest

from flymon.brain import w_oc
from flymon.brain import w_verdict as WV
from flymon.brain import x_oc
from flymon.brain import y_oc as O
from flymon.brain.x_spec import SPEC as XS
from flymon.brain.y_spec import SPEC as Y

Z = {"A": (16.917, 12.484), "P": (80.167, 29.775)}
ONES = [1.0] * Y.f_max


@pytest.fixture(scope="module")
def theta():
    return w_oc.fit(w_oc.synthetic_pilot(np.random.default_rng(3), base=(40.0, 90.0), sd=4.0, corr=0.8,
                                         learn=(20.0, 10.0)))


@pytest.fixture(scope="module")
def ab(theta):
    idx = np.random.default_rng(1).integers(0, len(theta["resid"]), XS.cal_reps)
    c = x_oc.calibrate_x(theta, XS.d_power, "min", idx, Z, XS)
    return c["a"]["value"], c["b"]["value"]


def test_bit_identity_with_x_evaluate(theta, ab):
    out = O.bit_identity(theta, Z, Y, XS, 20, *ab)
    assert out["equal"] and set(out["by_g"]) == {"g0.0", "g0.5", "g1.0"}


def test_evaluate_shape_fill_and_p_set_order(theta, ab):
    r = O.evaluate_y(theta, O.rng(1, 2), O.rng(1, 3), 30, [O.one_corner(*ab)] * Y.k_cap, ONES, ONES, 0.5, Z, Y)
    p = np.asarray(r["p"])
    assert p.shape == (5, 3, 2, 25, 7) and len(r["fill_by_k"]) == 7 and max(r["fill_by_k"]) == 0.0
    assert np.all(np.diff(p, axis=0) <= 1e-12)                      # a larger p_set never passes more


def test_near_shift_keeps_balance_and_sets_levels(theta):
    base, filled = O.pair_bases_y(theta, O.rng(5, 1), 40, Y.k_cap, 0.0, Z, Y, "near")
    keep = ~filled
    assert np.allclose(base[keep][:, WV.A, WV.X], Y.c_a) and np.allclose(base[keep][:, WV.P, WV.X], Y.c_p)
    assert (base[keep][:, WV.A, WV.Y] >= 0).all() and (base[keep][:, WV.P, WV.Y] >= 0).all()
    raw, _ = x_oc.pair_bases(theta, O.rng(5, 2), 40, Y.k_cap, 0.0, Z, Y, Y.tries, O.accept_y(Y))
    sdp = w_oc.sd_pre(theta, Z)
    assert np.abs(WV.dv(O.near_shift(raw, Y), Z) - WV.dv(raw, Z)).max() / sdp <= 1e-12


def test_fill_exclusion_and_near_rejection(theta, ab):
    never = dataclasses.replace(Y, c_a=1e6, tries=3)
    r = O.evaluate_y(theta, O.rng(1, 4), O.rng(1, 5), 4, [O.one_corner(*ab)] * Y.k_cap, ONES, ONES, 0.0, Z, never)
    assert r["fill_by_k"] == [1.0] * 7
    fill_bad = np.round(np.asarray(r["fill_by_k"]) - Y.fill_max, 9) > 0
    ok = O.range_ok(np.ones((5, 3, 2, 25, 7)), np.zeros((5, 3, 2, 25, 7)), fill_bad, Y, (4, 8), envelope=False)
    assert not ok.any()
    hard = dataclasses.replace(Y, c_a=Y.c_a + 1e3, tries=2)         # the shift drives A(Y) far below zero
    _b, filled = O.pair_bases_y(theta, O.rng(6, 1), 10, Y.k_cap, 0.0, Z, hard, "near")
    assert filled.all()


def test_generator_fixtures_pass():
    fx = O.generator_fixtures(Y, Z)
    assert fx["ok"], {k: v for k, v in fx.items() if k != "ok" and not v["ok"]}
    assert set(fx) == {"ok", "f8b_mixture_mc", "f9_near_shift", "accept", "qualify_ranges", "select_budget",
                       "set_rule"}


def test_mutation_corner_per_pair(monkeypatch):
    def per_pair(mix_rng, n, n_pair, mixes, which):
        a, b = np.empty((n, n_pair)), np.empty((n, n_pair))
        for j in range(n_pair):
            m = mixes[which[j]]
            w = np.asarray([c[2] for c in m])
            pk = mix_rng.choice(len(m), size=n, p=w / w.sum())
            a[:, j] = np.asarray([c[0] for c in m])[pk]
            b[:, j] = np.asarray([c[1] for c in m])[pk]
        return a, b
    monkeypatch.setattr(O, "_corner_effects", per_pair)
    assert not O.generator_fixtures(Y, Z)["f8b_mixture_mc"]["ok"]


@pytest.mark.parametrize("name,fn,fixture", [
    ("accept_y", lambda ys, c_a=None, c_p=None: (lambda base: np.asarray(base)[..., 0, 0] >= ys.c_a), "accept"),
    ("_qual_ranges", lambda k_ranges: list(k_ranges)[:1], "qualify_ranges"),
    ("_range_ks", lambda kr, ys: [k - ys.k_min for k in range(kr[0] + 1, kr[1] + 1)], "qualify_ranges"),
    ("_in_budget", lambda r, elapsed_h, ys: True, "select_budget")])
def test_generator_mutations(monkeypatch, name, fn, fixture):
    monkeypatch.setattr(O, name, fn)
    assert not O.generator_fixtures(Y, Z)[fixture]["ok"]


def test_synthetic_validation_passes_every_p_set_and_range():
    out = O.synthetic_validation(Y, Z, n_rep=300)
    assert out["ok"], {k: v for k, v in out.items() if isinstance(v, dict) and not v["ok"]}
    assert set(out["big_effect"]["by_range"]) == {"[4, 8]", "[6, 10]"}


def test_pick_y_rules_and_ties():
    pw = np.zeros((5, 3, 2, 25, 7))
    fp = np.ones((5, 3, 2, 25, 7))
    pw[0, 0, 1, 0] = 0.9                       # p_set 0.5, q 0.5, K 16, F 8: power 0.9 at every k
    pw[4, 2, 0, 3] = 0.9                       # p_set 1.0, q 0.75, K 8, F 11: equal power → larger p_set wins
    b = O.pick_y(pw, fp, Y, [(4, 8), (6, 10)], require_false=False)
    assert (b["rule"], b["p_set"], b["q"], b["K"], b["F"], b["k_range"]) == ("max_power", 1.0, 0.75, 8, 11, [4, 8])
    fp[0, 0, 1, 0, 2:] = 0.0                   # false ok only on k 6..10 → [6, 10] qualifies for the false rule
    r = O.pick_y(pw, fp, Y, [(4, 8), (6, 10)], require_false=True)
    assert (r["rule"], r["p_set"], r["k_range"]) == ("false_ok_max_power", 0.5, [6, 10])
    t = O.table_at_f_y(pw, fp, Y, [(4, 8)], 32)
    assert len(t) == 30 and t[0]["k"] == [4, 5, 6, 7, 8] and t[0]["F"] == 32


def test_near_rejects_a_shift_that_drives_y_negative(theta):
    """c_A kept, c_P lowered to 0: the accept predicate holds for every base, but the shift moves P's Y cell by
    0 − base[P, X], below zero for about half the pairs — those must be rejected (filled), never kept."""
    low = dataclasses.replace(Y, c_p=0.0, tries=50)
    raw, f0 = x_oc.pair_bases(theta, O.rng(6, 1), 40, Y.k_cap, 0.0, Z, low, low.tries, O.accept_y(low))
    s = O.near_shift(raw, low)
    neg = ((s[..., WV.A, WV.Y] < 0) | (s[..., WV.P, WV.Y] < 0))[~f0]
    assert neg.mean() > 0.3                                          # the case is real: the shift makes Y < 0
    base, filled = O.pair_bases_y(theta, O.rng(6, 1), 40, Y.k_cap, 0.0, Z, low, "near")
    keep = ~filled
    assert keep.any() and (base[keep][:, :, WV.Y] >= 0).all()
    one = dataclasses.replace(low, tries=1)                         # one round: a rejected pair is a fill
    _b, f1 = O.pair_bases_y(theta, O.rng(6, 2), 40, Y.k_cap, 0.0, Z, one, "near")
    _r, fb = O.pair_bases_y(theta, O.rng(6, 2), 40, Y.k_cap, 0.0, Z, one, "base")
    assert f1.mean() > 0.3 and fb.mean() < 0.05


def _qual(*cells):
    ok = np.zeros((5, 3, 2, 25), bool)
    for c in cells:
        ok[c] = True
    return {(4, 8): ok}


def test_select_prefers_larger_p_set_over_cost():
    qual = _qual((0, 0, 0, 0), (2, 0, 0, 0), (4, 0, 0, 24))         # p_set 0.5 / 0.75 / 1.0 (F 32)
    cost = {0.5: 1.0, 0.75: 10.0, 1.0: 100.0}                      # 1.0 over budget, 0.75 in budget but dearer
    s = O.select_y(qual, lambda d: cost[d["p_set"]], 0.0, Y)
    assert s["outcome"] == "PASS" and s["n_in_budget"] == 2 and s["n_qualifying"] == 3
    assert (s["selected"]["p_set"], s["selected"]["cost_h"]) == (0.75, 10.0)
    assert [r["p_set"] for r in s["ranking"]] == [0.75, 0.5]
    same = O.select_y(_qual((1, 0, 0, 0), (1, 2, 0, 0)), lambda d: 5.0 if d["q"] == 0.75 else 1.0, 0.0, Y)
    assert same["selected"]["q"] == 0.5                              # equal p_set: the cheaper first, before q


def test_range_ok_envelope_f_plus_3_and_solo():
    pw = np.ones((1, 1, 1, 25, 7))
    fp = np.zeros((1, 1, 1, 25, 7))
    no_fill = np.zeros(7, bool)
    fi = lambda F: F - Y.f_min  # noqa: E731
    pw[..., fi(15), 0] = 0.0                                         # only F 15 fails (k 4)
    ok = O.range_ok(pw, fp, no_fill, Y, (4, 8), envelope=True)[0, 0, 0]
    assert ok[fi(11)] and not ok[fi(12)] and not ok[fi(15)] and ok[fi(16)]   # F 12 fails only through F + 3
    pw[..., fi(15), 0] = 1.0
    fp[..., fi(30), 2] = 1.0                                         # only F 30 fails (false at k 6)
    ok = O.range_ok(pw, fp, no_fill, Y, (4, 8), envelope=True)[0, 0, 0]
    assert ok[fi(26)] and not ok[fi(27)] and not ok[fi(28)]
    assert ok[fi(29)] and not ok[fi(30)] and ok[fi(31)] and ok[fi(32)]       # F ≥ 29 alone


def test_fill_by_k_is_cumulative_per_k(theta, ab, monkeypatch):
    real = O.pair_bases_y
    pattern = np.array([0, 0, 0, 1, 1, 0, 0, 1, 0, 1], bool)        # slots 4, 5, 8, 10 filled

    def fixed(*a, **k):
        base, _f = real(*a, **k)
        return base, np.broadcast_to(pattern, _f.shape).copy()
    monkeypatch.setattr(O, "pair_bases_y", fixed)
    r = O.evaluate_y(theta, O.rng(1, 6), O.rng(1, 7), 6, [O.one_corner(*ab)] * Y.k_cap, ONES, ONES, 0.0, Z,
                     dataclasses.replace(Y, oc_chunk=4))
    want = [pattern[:k].mean() for k in range(Y.k_min, Y.k_cap + 1)]
    assert np.allclose(r["fill_by_k"], want, atol=1e-12) and len(set(np.round(want, 9))) > 1


def test_pick_y_tie_prefers_larger_q():
    pw = np.zeros((5, 3, 2, 25, 7))
    fp = np.ones((5, 3, 2, 25, 7))
    pw[2, 0, 0, 0] = 0.9                       # p_set 0.75, q 0.5, K 8, F 8
    pw[2, 2, 1, 5] = 0.9                       # p_set 0.75, q 0.75, K 16, F 13: equal power and p_set → larger q
    b = O.pick_y(pw, fp, Y, [(4, 8)], require_false=False)
    assert (b["p_set"], b["q"], b["K"], b["F"]) == (0.75, 0.75, 16, 13)
