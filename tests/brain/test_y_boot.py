"""Y.6.4 / Y.5 / Y.9.2 P2-8 / Y.6.5 in y_oc (phase B): one parametric draw on the whole grid, its streams, the
bootstrap limits (simultaneous over the range's k, worst over g · scenario, fills, success-only), qualification, the
reconfirmation's one-design spec and decision, the records target without a design, and the records' root."""
import dataclasses

import numpy as np
import pytest

from flymon.brain import w_oc, x_oc
from flymon.brain import y_oc as O
from flymon.brain.y_spec import SPEC as Y

Z = {"A": (16.917, 12.484), "P": (80.167, 29.775)}
YS = dataclasses.replace(Y, boot_reps=4, reconfirm_reps=4, oc_chunk=4)


@pytest.fixture(scope="module")
def th():
    kw = dict(base=(40.0, 90.0), sd=4.0, corr=0.8, learn=(20.0, 10.0))
    keys = list(Y.pilot_w_pairs) + list(Y.pilot_v_pairs)[:3]
    r = O.r_v0(w_oc.fit(w_oc.synthetic_pilot(np.random.default_rng(4), n_pair=16, **kw)))
    return O.fit_y(w_oc.synthetic_pilot(np.random.default_rng(3), n_pair=6, **kw), keys, r), r


def test_boot_draw_shape_determinism_and_streams(th, monkeypatch):
    t, r = th
    a = O.boot_draw(t, Z, YS, r, 0)
    assert a["arrays"]["hits"].shape == (3, 2, 2) + x_oc.grid_shape(YS) and a["arrays"]["hits"].dtype == np.uint16
    assert a["arrays"]["fills"].shape == (3, 2, 2, YS.k_cap - YS.k_min + 1) and a["meta"]["n_rep"] == 4
    assert np.array_equal(a["arrays"]["hits"], O.boot_draw(t, Z, YS, r, 0)["arrays"]["hits"])
    b = O.boot_draw(t, Z, YS, r, 1)
    assert not np.array_equal(a["arrays"]["hits"], b["arrays"]["hits"])
    d = dict(p_set=0.5, q=0.5, K=8, F=8, k_range=[4, 8])
    seen, real = [], O.evaluate_y

    def spy(theta, rng_, mix_rng, *a, **k):
        seen.append((rng_.bit_generator.state["state"]["state"], mix_rng.bit_generator.state["state"]["state"]))
        return real(theta, rng_, mix_rng, *a, **k)
    monkeypatch.setattr(O, "evaluate_y", spy)
    rc0 = O.reconfirm_draw(t, Z, YS, r, d, 0, 0)
    assert rc0["arrays"]["hits"].shape == (3, 2, 2, 1, 1, 1, 4, 5)        # F 8–11, k 4–8
    s0, seen[:] = list(seen), []
    O.reconfirm_draw(t, Z, YS, r, d, 1, 0)
    s1, seen[:] = list(seen), []
    O.boot_draw(t, Z, YS, r, 0)
    sb = list(seen)
    assert len(s0) == len(s1) == len(sb) == 12 and len(set(s0) | set(s1) | set(sb)) == 36   # no stream reused
    for st in (0, 1):                                           # main and mix streams each unique on their own
        assert len({x[st] for x in s0}) == len({x[st] for x in sb}) == 12
        assert len({x[st] for x in s0 + s1 + sb}) == 36
    assert set(a["meta"]["cal"]) == {"min", "max"} and {"ok", "a", "b", "failure"} <= set(a["meta"]["cal"]["min"])


def test_a_failed_side_fills_without_simulating(th, monkeypatch):
    t, r = th
    real = O.calibrate_y

    def cal(theta, target, mode, idx, z, ys):
        c = real(theta, target, mode, idx, z, ys)
        return dict(c, ok=False, corners=None, fill=0.0, failure=dict(handle="b", status=O.FLOOR)) if mode == "min" \
            else c
    monkeypatch.setattr(O, "calibrate_y", cal)
    a = O.boot_draw(t, Z, YS, r, 0)
    assert (a["arrays"]["hits"][:, 0] == 0).all() and np.isnan(a["arrays"]["fills"][:, 0]).all()
    assert not np.isnan(a["arrays"]["fills"][:, 1]).any()



@pytest.mark.parametrize("status", [O.FLOOR, O.ABOVE])
def test_a_failed_false_side_fills_one_and_power_side_reads_the_fill(th, monkeypatch, status):
    t, r = th
    real = O.calibrate_y

    def cal(theta, target, mode, idx, z, ys):        # the false (max) side fails with its real fill, x_oc.fill_value
        c = real(theta, target, mode, idx, z, ys)
        return dict(c, ok=False, corners=None, fill=x_oc.fill_value("max"),
                    failure=dict(handle="a", status=status)) if mode == "max" else c
    monkeypatch.setattr(O, "calibrate_y", cal)
    a = O.boot_draw(t, Z, YS, r, 0)
    assert x_oc.fill_value("max") == 1.0
    assert (a["arrays"]["hits"][:, 1] == YS.boot_reps).all() and np.isnan(a["arrays"]["fills"][:, 1]).all()
    assert not np.isnan(a["arrays"]["fills"][:, 0]).any()

    def cal_min(theta, target, mode, idx, z, ys):    # power side: the 0 is the fill's value, not a default
        c = real(theta, target, mode, idx, z, ys)
        return dict(c, ok=False, corners=None, fill=0.5, failure=dict(handle="b", status=status)) if mode == "min" \
            else c
    monkeypatch.setattr(O, "calibrate_y", cal_min)
    b = O.boot_draw(t, Z, YS, r, 0)
    assert (b["arrays"]["hits"][:, 0] == round(0.5 * YS.boot_reps)).all() and np.isnan(b["arrays"]["fills"][:, 0]).all()

def _draw(pw, fp, fills=0.0, ok=(True, True), n=10):
    """A fake draw on YS with power / false P(PASS) arrays broadcast to [g, scen, *grid]."""
    shp = (3, 2) + x_oc.grid_shape(YS)
    h = np.zeros((3, 2, 2) + x_oc.grid_shape(YS), np.uint16)
    h[:, 0] = np.rint(np.broadcast_to(pw, shp) * n).astype(np.uint16)
    h[:, 1] = np.rint(np.broadcast_to(fp, shp) * n).astype(np.uint16)
    f = np.full((3, 2, 2, YS.k_cap - YS.k_min + 1), fills, float)
    knob = dict(status="ok", coarse_step=False, stage=None)
    cal = {m: dict(ok=o, failure=None if o else dict(handle="a", status="floor"), corners=None, a=knob, b=[knob])
           for m, o in zip(("min", "max"), ok)}
    return dict(arrays=dict(hits=h, fills=f), meta=dict(cal=cal, n_rep=n))


def test_limits_are_simultaneous_and_worst_over_g_and_scenario():
    nk = YS.k_cap - YS.k_min + 1
    draws = []
    for i in range(20):               # each draw is weak at one k (cycling), so per-k limits stay high
        pw = np.full(nk, 0.9)
        pw[i % nk] = 0.5
        draws.append(_draw(pw, 0.0))
    lim = O.limits(draws, YS, [(4, 8)])
    assert np.allclose(lim["power_lo_by_k"], np.percentile([0.9] * 17 + [0.5] * 3, 5))     # ≥ 2/20 weak per k
    assert np.allclose(lim["sim"][(4, 8)]["power"], 0.5)       # every draw's min over k 4–8 is 0.5
    q = O.qualify_boot(lim, YS, [(4, 8)])
    assert not q[(4, 8)].any()                                  # the per-k limits alone would qualify
    worst = [_draw(0.9, 0.0) for _ in range(20)]
    for i in range(3):
        worst[i]["arrays"]["hits"][2, 0, 1] = 0                 # g 1.0, power, near: three draws at 0
    lim2 = O.limits(worst, YS, [(4, 8)])
    assert np.allclose(lim2["power_lo_by_k"], 0.0) and np.allclose(lim2["sim"][(4, 8)]["power"], 0.0)
    assert lim2["counts"]["min"]["fails"] == 0



def test_false_limits_are_simultaneous_upper_and_worst_over_g_and_scenario():
    nk = YS.k_cap - YS.k_min + 1
    draws = []
    for i in range(20):               # draws 0–6 are high at one k (cycling), only at g 0.5 · near: per-k limits low
        fp = np.zeros((3, 2) + x_oc.grid_shape(YS))
        if i < nk:
            fp[1, 1, ..., i] = 0.5
        draws.append(_draw(0.9, fp))
    lim = O.limits(draws, YS, [(4, 8), (6, 10)])
    assert np.allclose(lim["false_hi_by_k"], np.percentile([0.0] * 19 + [0.5], 95))       # 0.025 ≤ p_false
    assert np.allclose(lim["sim"][(4, 8)]["false"], 0.5)        # 5/20 draws' max over k 4–8 is 0.5: 95th pct 0.5
    assert np.allclose(lim["sim"][(6, 10)]["false"], 0.5)
    assert np.allclose(lim["sim"][(4, 8)]["power"], 0.9)
    q = O.qualify_boot(lim, YS, [(4, 8), (6, 10)])
    assert not q[(4, 8)].any() and not q[(6, 10)].any()         # the per-k limits alone would qualify
    assert O.qualify_y(lim["power_lo_by_k"], lim["false_hi_by_k"], lim["fill_bad"], YS, [(4, 8)])[(4, 8)].all()


def test_limits_take_the_percentile_per_g_and_scenario_then_the_worst():
    draws = []
    for i in range(20):               # draws 0–5 are weak in one (g, scenario) cell each
        pw = np.full((3, 2) + x_oc.grid_shape(YS), 0.9)
        if i < 6:
            pw[i // 2, i % 2] = 0.5
        draws.append(_draw(pw, 0.0))
    lim = O.limits(draws, YS, [(4, 8)])
    per_cell = np.percentile([0.5] + [0.9] * 19, 5)              # 0.88; worst-then-percentile would give 0.5
    assert np.allclose(lim["power_lo_by_k"], per_cell) and np.allclose(lim["sim"][(4, 8)]["power"], per_cell)

def test_fills_success_only_and_counts():
    d = [_draw(0.9, 0.0, fills=0.0) for _ in range(9)] + [_draw(0.0, 1.0, fills=np.nan, ok=(False, False))]
    d[0]["arrays"]["fills"][1, 0, 0, 2] = 0.2                   # one cell, k 6: share 0.2 / 9 > 0.01
    lim = O.limits(d, YS, [(4, 8), (6, 10)])
    assert lim["fill_bad"].tolist() == [False, False, True, False, False, False, False]
    assert np.allclose(lim["success_only"]["min"], 0.9) and lim["counts"]["min"]["fails"] == 1
    assert not O.qualify_boot(lim, YS, [(4, 8)])[(4, 8)].any()   # k 6 is fill-excluded



def test_fill_share_is_over_the_simulated_draws_only():
    d = [_draw(0.9, 0.0, fills=0.0) for _ in range(5)] + [_draw(0.0, 1.0, fills=np.nan, ok=(False, False))
                                                          for _ in range(5)]
    d[0]["arrays"]["fills"][1, 0, 0, 2] = 0.06                  # k 6: 0.06 / 5 = 0.012 > 0.01 (over 10 draws: 0.006)
    lim = O.limits(d, YS, [(4, 8)])
    assert np.isclose(lim["fill_share"][1, 0, 0, 2], 0.012)
    assert lim["fill_bad"].tolist() == [False, False, True, False, False, False, False]

def test_qualify_boot_envelope():
    nk = YS.k_cap - YS.k_min + 1
    pw = np.full(x_oc.grid_shape(YS), 0.9)
    pw[..., 10, :] = 0.5                                        # F 18 fails for every design
    lim = O.limits([_draw(pw, 0.0) for _ in range(5)], YS, [(4, 8)])
    q = O.qualify_boot(lim, YS, [(4, 8)])[(4, 8)]
    assert not q[..., 7:11].any() and q[..., 6].all() and q[..., 11].all() and nk == 7     # F 15–18 hit by F 18


def test_reconfirm_spec_and_decide():
    for F, f_hi in ((8, 11), (28, 31), (29, 29), (32, 32)):
        yd = O.reconfirm_spec(Y, dict(p_set=1.0, q=0.75, K=16, F=F, k_range=[6, 10]))
        assert (yd.f_min, yd.f_max, yd.k_cap, yd.k_grid, yd.p_set_grid) == (F, f_hi, 10, (16,), (1.0,))
    d = dict(p_set=1.0, q=0.75, K=8, F=8, k_range=[4, 8])
    yd = O.reconfirm_spec(YS, d)

    def dr(p):
        shp = (3, 2, 2) + x_oc.grid_shape(yd)
        h = np.zeros(shp, np.uint16)
        h[:, 0] = int(p * 4)
        knob = dict(status="ok", coarse_step=False, stage=None)
        return dict(arrays=dict(hits=h, fills=np.zeros((3, 2, 2, 5))),
                    meta=dict(cal={m: dict(ok=True, failure=None, corners=None, a=knob, b=[knob]) for m in ("min", "max")},
                              n_rep=4))
    ok = O.reconfirm_decide([dr(1.0)] * 5, d, 0, YS)
    assert ok["ok"] and ok["k"] == [4, 5, 6, 7, 8] and ok["power_by_k"] == [1.0] * 5 and ok["root"] == Y.reconfirm_seed
    assert ok["false_by_k"] == [0.0] * 5 and ok["false_sim"] == [0.0] * len(ok["false_sim"])
    fa = [dr(1.0) for _ in range(5)]
    for x in fa:
        x["arrays"]["hits"][1, 1, 1] = 1                         # false pass 0.25 at g 0.5 · near, every k
    bad = O.reconfirm_decide(fa, d, 0, YS)
    assert not bad["ok"] and bad["false_by_k"] == [0.25] * 5 and bad["power_by_k"] == [1.0] * 5
    assert not O.reconfirm_decide([dr(0.75)] * 5, d, 2, YS)["ok"]


def test_records_target_b_and_root(monkeypatch):
    shp = x_oc.grid_shape(YS)[:-1]
    lim = dict(sim={(4, 8): dict(power=np.full(shp, 0.5), false=np.full(shp, 0.2)),
                    (6, 10): dict(power=np.full(shp, 0.6), false=np.full(shp, 0.2))})
    lim["sim"][(4, 8)]["false"][0, 0, 0, 0] = 0.04
    t = O.records_target_b(lim, YS, [(4, 8), (6, 10)], lambda d: d["F"])
    assert (t["rule"], t["p_set"], t["F"], t["k_range"]) == ("false_ok_max_power", 0.5, 8, [4, 8])
    lim["sim"][(4, 8)]["false"][0, 0, 0, 0] = 0.2
    lim["sim"][(6, 10)]["false"][4, 2, 1, 3] = 0.1
    t = O.records_target_b(lim, YS, [(4, 8), (6, 10)], lambda d: d["F"])
    assert (t["rule"], t["p_set"], t["q"], t["K"], t["F"], t["k_range"]) == ("min_false", 1.0, 0.75, 16, 11, [6, 10])
    seen = {}
    monkeypatch.setattr(O, "point_records_y", lambda th, tg, z, ys, cell=None, log=None: seen.update(
        root=ys.precheck_seed, extra=ys.records_seed) or {})
    O.records_b({}, dict(p_set=1.0, q=0.75, K=8, F=8), Z, Y)
    assert seen == dict(root=Y.oc_seed, extra=Y.records_seed)


def test_records_target_b_ties_and_then_power():
    shp = x_oc.grid_shape(YS)[:-1]
    kr = [(4, 8), (6, 10)]
    lim = dict(sim={k: dict(power=np.full(shp, 0.5), false=np.full(shp, 0.0)) for k in kr})
    t = O.records_target_b(lim, YS, kr, lambda d: 100 * d["q"] + d["F"])    # cost before q: smallest q is cheapest
    assert (t["rule"], t["p_set"], t["q"], t["K"], t["F"], t["k_range"]) == ("false_ok_max_power", 1.0, 0.5, 8, 8,
                                                                               [4, 8])
    t = O.records_target_b(lim, YS, kr, lambda d: 0.0)          # equal cost: larger q
    assert (t["p_set"], t["q"], t["K"], t["F"], t["k_range"]) == (1.0, 0.75, 8, 8, [4, 8])
    lim = dict(sim={k: dict(power=np.full(shp, 0.5), false=np.full(shp, 0.3)) for k in kr})
    lim["sim"][(6, 10)]["false"][4, 2, 0, 0] = 0.1              # min false 0.1 twice: the larger power wins
    lim["sim"][(4, 8)]["false"][0, 0, 1, 5] = 0.1
    lim["sim"][(4, 8)]["power"][0, 0, 1, 5] = 0.7
    t = O.records_target_b(lim, YS, kr, lambda d: 0.0)
    assert (t["rule"], t["p_set"], t["q"], t["K"], t["F"], t["k_range"]) == ("min_false", 0.5, 0.5, 16, 13, [4, 8])
    lim["sim"][(4, 8)]["power"][0, 0, 1, 5] = 0.5               # equal power: the tie order (larger p_set) decides
    t = O.records_target_b(lim, YS, kr, lambda d: 0.0)
    assert (t["rule"], t["p_set"], t["q"], t["F"], t["k_range"]) == ("min_false", 1.0, 0.75, 8, [6, 10])
