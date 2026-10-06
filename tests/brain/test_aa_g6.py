"""AA.9.1: the precheck wrapper replaces only d_power and precheck_seed, agrees with y_oc.precheck_y's own passing list,
reports calibration failures as statuses, and the heterogeneity wrapper is deterministic."""
import dataclasses

import numpy as np

from flymon.brain import aa_estimate as E
from flymon.brain import w_oc, x_oc, y_oc
from flymon.brain.aa_spec import SPEC
from flymon.brain.y_spec import SPEC as Y

YS = dataclasses.replace(Y, precheck_reps=2, oc_chunk=2)
COSTS = dict(trial_s=0.001, presentation_s=0.0001, oracle_round_s=1.0, workers=16)
Z = {"A": (16.916666666666668, 12.483878492769072), "P": (80.16666666666667, 29.775432639827233)}


def theta():
    kw = dict(base=(40.0, 90.0), sd=4.0, corr=0.8, learn=(20.0, 10.0))
    pil = w_oc.synthetic_pilot(np.random.default_rng(3), n_pair=6, **kw)
    keys = [f"a|{i}|O{i % 4}|y{i}" for i in range(6)]
    return y_oc.fit_y(pil, keys, np.eye(4))


def test_g6_spec_replaces_two_fields():
    yd = E.g6_spec(1.2, SPEC, Y)
    diff = {k for k, v in dataclasses.asdict(yd).items() if v != dataclasses.asdict(Y)[k]}
    assert diff == {"d_power", "precheck_seed"} and yd.precheck_seed == 87_200_000 and yd.d_power == 1.2


def test_g6_row_agrees_with_precheck_passing():
    th = theta()
    row = E.g6_row(th, Z, 1.5, COSTS, 31, SPEC, YS)
    pc = y_oc.precheck_y(th, Z, E.g6_spec(1.5, SPEC, YS), [tuple(k) for k in YS.k_ranges])
    if row["status"] == "ok":
        assert row["n_pass"] == len(pc["passing"]) and set(row["target_design_power"]) == {4, 5, 6, 7, 8}
        assert row["first"] is None or row["first"] == min(
            row["designs"], key=lambda r: (-r["p_set"], r["cost_h"], -r["q"], r["K"], r["F"], r["k_range"][0]))
    else:
        assert row["status"] == pc["calibration"]["min"]["failure"]["status"]


def test_g6_unreachable_target_is_a_status():
    row = E.g6_row(theta(), Z, 40.0, COSTS, 31, SPEC, YS)
    assert row["status"] in (y_oc.FLOOR, y_oc.ABOVE)


def test_g6_hetero_deterministic():
    th = theta()
    ref = E.g6_row(th, Z, 1.5, COSTS, 31, SPEC, YS)
    if ref["status"] != "ok":
        return
    a = E.g6_hetero(th, Z, 1.5, 0.3, ref, COSTS, 31, SPEC, YS)
    assert a == E.g6_hetero(th, Z, 1.5, 0.3, ref, COSTS, 31, SPEC, YS) and "n_pass" in a


def _fake_hetero(monkeypatch, fill_by_k=None):
    calls = dict(cal=[], mix=[])

    def cal(theta, t, mode, idx, z, yd):
        calls["cal"].append((round(t, 12), mode))
        return dict(ok=True, corners=("corner", round(t, 12)))

    def ev(theta, rng_, mix_rng, n_rep, pair_mix, fly_a, fly_b, g, z, yd, sc="base", accept=None):
        calls["mix"].append(list(pair_mix))
        fb = [0.0] * (yd.k_cap - yd.k_min + 1) if fill_by_k is None else fill_by_k
        return dict(p=np.ones(x_oc.grid_shape(yd)), fill_by_k=fb)
    monkeypatch.setattr(y_oc, "calibrate_y", cal)
    monkeypatch.setattr(y_oc, "evaluate_y", ev)
    return calls


def _ref(yd, bad_ks=(), false=0.0):
    fb = np.zeros(yd.k_cap - yd.k_min + 1, bool)
    for k in bad_ks:
        fb[k - yd.k_min] = True
    return dict(fill_bad=fb.tolist(), false=np.full(x_oc.grid_shape(yd), false).tolist())


def test_g6_hetero_mix_alternates_the_two_calibrations(monkeypatch):
    calls = _fake_hetero(monkeypatch)
    yd = E.g6_spec(1.5, SPEC, YS)
    out = E.g6_hetero({"resid": [0.0] * 5}, Z, 1.5, 0.3, _ref(yd), COSTS, 31, SPEC, YS)
    assert calls["cal"] == [(1.2, "min"), (1.8, "min")]
    lo, hi = ("corner", 1.2), ("corner", 1.8)
    assert len(calls["mix"]) == len(YS.cluster_grid) * len(y_oc.SCENARIOS)
    assert all(m == [lo if j % 2 == 0 else hi for j in range(YS.k_cap)] for m in calls["mix"])
    assert out["odd_k_low_extra"] is True and out["n_pass"] > 0


def test_g6_hetero_uses_ref_fill_exclusion_and_false(monkeypatch):
    _fake_hetero(monkeypatch)
    yd = E.g6_spec(1.5, SPEC, YS)
    th = {"resid": [0.0] * 5}
    full = E.g6_hetero(th, Z, 1.5, 0.3, _ref(yd), COSTS, 31, SPEC, YS)["n_pass"]
    cut = E.g6_hetero(th, Z, 1.5, 0.3, _ref(yd, bad_ks=(9,)), COSTS, 31, SPEC, YS)
    assert full > 0 and cut["n_pass"] == full // 2 and cut["first"]["k_range"] == [4, 8]   # ref's k 9 kills [6, 10]
    assert E.g6_hetero(th, Z, 1.5, 0.3, _ref(yd, false=1.0), COSTS, 31, SPEC, YS)["n_pass"] == 0


def test_g6_hetero_adds_its_own_fill_exclusion(monkeypatch):
    yd = E.g6_spec(1.5, SPEC, YS)
    fb = [0.0] * (yd.k_cap - yd.k_min + 1)
    fb[5 - yd.k_min] = 0.5                                          # k 5 over fill_max → [4, 8] out
    _fake_hetero(monkeypatch, fb)
    out = E.g6_hetero({"resid": [0.0] * 5}, Z, 1.5, 0.3, _ref(yd), COSTS, 31, SPEC, YS)
    assert out["n_pass"] > 0 and out["first"]["k_range"] == [6, 10]
