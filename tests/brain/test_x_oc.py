"""x_oc core (X.4.2, X.4.6, X.9.1.2, X.9.1.3 P2-6, X.4.3 diagnosis): W's brackets and steps reproduce w_oc.calibrate
exactly; X's brackets are ×4; floor fills power 0 / false pass 1; no_convergence retries with 160 steps; X's
simulator copy equals w_oc's at W's rounds; set_hits equals the per-k loop; X's evaluate at p_set 1.0 equals
w_oc.evaluate bit for bit; P(PASS) never rises with p_set; the synthetic fixtures pass at every p_set; the W diagnosis
reproduces w_oc.run's stored bootstrap calibrations and flags a tampered one."""
import dataclasses
import json

import numpy as np
import pytest

from flymon.brain import w_oc
from flymon.brain import w_verdict as WV
from flymon.brain import x_oc
from flymon.brain import x_verdict as XV
from flymon.brain.h3_store import canonical
from flymon.brain.w_spec import SPEC as W
from flymon.brain.x_spec import SPEC as XS

Z = {"A": (16.917, 12.484), "P": (80.167, 29.775)}


@pytest.fixture(scope="module")
def pilot():
    return w_oc.synthetic_pilot(np.random.default_rng(3), base=(40.0, 90.0), sd=4.0, corr=0.8, learn=(20.0, 10.0))


@pytest.fixture(scope="module")
def theta(pilot):
    return w_oc.fit(pilot)


@pytest.fixture(scope="module")
def idx(theta):
    return np.random.default_rng(1).integers(0, len(theta["resid"]), XS.cal_reps)


@pytest.mark.parametrize("target,mode", [(1.5, "min"), (0.5, "max"), (40.0, "min")])
def test_w_brackets_reproduce_w_calibrate(theta, idx, target, mode):
    x = x_oc.calibrate(theta, target, mode, idx, Z, XS, XS.w_bracket_mult, XS.cal_iter)
    assert x == w_oc.calibrate(theta, target, mode, idx, Z, W)


def test_brackets_times_four(theta):
    a1, b1 = x_oc.brackets(theta, 1.0)
    a4, b4 = x_oc.brackets(theta, XS.bracket_mult)
    assert (a4, b4) == (a1 * 4.0, b1 * 4.0)


def test_status_fill(theta, idx):
    p = x_oc.calibrate_x(theta, 40.0, "min", idx, Z, XS)
    assert not p["ok"] and p["fill"] == 0.0 and p["failure"]["status"] in ("floor", "no_convergence")
    f = x_oc.calibrate_x(theta, 40.0, "max", idx, Z, XS)
    assert not f["ok"] and f["fill"] == 1.0           # false-pass floor counts as a pass, never as 0
    ok = x_oc.calibrate_x(theta, 1.5, "min", idx, Z, XS)
    assert ok["ok"] and ok["fill"] is None and ok["failure"] is None


def test_no_convergence_is_retried_with_more_steps(theta, idx):
    one = dataclasses.replace(XS, cal_iter=1, cal_retry_iter=60)
    c = x_oc.calibrate_x(theta, 1.5, "min", idx, Z, one)
    assert c["first_failure"] == dict(handle="a", status="no_convergence") and c["retried"] and c["ok"]
    never = dataclasses.replace(XS, cal_iter=1, cal_retry_iter=1)
    c = x_oc.calibrate_x(theta, 1.5, "min", idx, Z, never)
    assert c["retried"] and not c["ok"] and c["fill"] == 0.0


def test_simulator_copy_equals_w_at_w_rounds(theta):
    sdp = w_oc.sd_pre(theta, Z)
    b_x, filled = x_oc.pair_bases(theta, np.random.default_rng(7), 30, 8, 0.5, Z, XS, XS.w_rejection_tries)
    b_w = w_oc._pair_bases(theta, np.random.default_rng(7), 30, 8, 0.5, Z, XS.naive_max, sdp)
    assert np.array_equal(b_x, b_w) and filled.shape == (30, 8)
    ab = ([3.0] * 8, [2.0] * 8, [1.0] * 12, [1.0] * 12)
    sx, _, _ = x_oc.simulate(theta, np.random.default_rng(9), 6, 8, 12, 16, *ab, 1.0, Z, XS, XS.w_rejection_tries)
    sw = w_oc.simulate(theta, np.random.default_rng(9), 6, 8, 12, 16, *ab, 1.0, Z, XS.naive_max)
    assert all(np.array_equal(sx[s], sw[s]) for s in sw)


def test_set_hits_equals_the_per_k_loop():
    rng = np.random.default_rng(2)
    fin = rng.choice([WV.P_PASS, WV.P_FAIL], size=(200, 8), p=[0.8, 0.2])
    mach = rng.random((200, 8)) < 0.01
    kk = list(range(4, 9))
    got = x_oc.set_hits(fin, mach, kk, XS)
    for j, k in enumerate(kk):
        for pi, p in enumerate(XS.p_set_grid):
            want = (XV.set_code(fin[:, :k], mach[:, :k].any(-1), p, XS) == XV.S_PASS).sum()
            assert got[pi, j] == want


def test_p2_6_bit_identity_synthetic(theta, idx):
    c = x_oc.calibrate_x(theta, XS.d_power, "min", idx, Z, XS)
    out = x_oc.bit_identity(theta, Z, XS, W, 60, c["a"]["value"], c["b"]["value"])
    assert out["equal"], out
    assert out["max_p"] > 0.0                         # the comparison is not between two zero arrays


def test_evaluate_shape_and_p_set_order(theta, idx):
    c = x_oc.calibrate_x(theta, XS.d_power, "min", idx, Z, XS)
    p = x_oc.evaluate(theta, x_oc.rng(1, 2), 40, *w_oc._uniform(XS, c["a"]["value"], c["b"]["value"]), 0.5, Z, XS)
    assert p.shape == x_oc.grid_shape(XS) == (5, 3, 2, 25, 5)
    assert np.all(np.diff(p, axis=0) <= 0)            # a larger p_set never passes more
    assert np.all(np.diff(p[-1], axis=-1) <= 1e-12)   # p_set 1.0 (all pairs PASS) never rises with k; a ratio
                                                      # rule may (3/4 FAIL at 0.875 → 4/5 PASS), so no k test there


def test_synthetic_validation_at_every_p_set():
    r = x_oc.synthetic_validation(XS, Z, n_rep=300)
    assert r["ok"], r
    assert r["zero_effect"]["max_p"] <= 0.02 and r["big_effect"]["min_p"] >= 0.98
    assert r["one_gate"]["max_p"] <= 0.02 and r["negative_correlation"]["max_p"] <= 0.02


def test_w_cal_diagnosis_reproduces_w_run(pilot, theta):
    doc = json.loads(canonical(w_oc.run(pilot, Z, W, lambda K, F: K * F, n_boot=3, n_rep=2, n_boot_rep=2)))
    d = x_oc.w_cal_diagnosis(theta, Z, W, XS, doc["boot_calibration"])
    assert d["reproduced"] and d["diffs"] == [] and d["n_draws"] == 3 and len(d["rows"]) == 3
    assert set(d["rows"][0]) == {"draw", "min", "max"} and "x_rule" in d["rows"][0]["min"]
    bad = json.loads(json.dumps(doc["boot_calibration"]))
    bad[1]["max"]["ok"] = not bad[1]["max"]["ok"]
    d2 = x_oc.w_cal_diagnosis(theta, Z, W, XS, bad)
    assert not d2["reproduced"] and d2["diffs"][0]["draw"] == 1 and d2["diffs"][0]["side"] == "max"
