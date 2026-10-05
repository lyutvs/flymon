"""W's operating characteristic (W.9.3, W.9.8 H1 / H4, W.9.9 P0-1 / P1-2 / P1-3 / P1-6 / P2-8 / P2-11): the fit
recovers a known synthetic pilot; the calibration meets the gate-vector targets within ±0.02 (min = 1.5 power, max =
0.5 false pass) and reports floor / above-at-zero; the evaluation runs the W verdict code itself (a verdict that
always fails gives P(PASS) 0) on nested experiments (P(PASS) never rises with k); an RN1 bit flipped in the simulator
stops the machine; the run is deterministic; the envelope / solo rule and the cost-then-q-then-K order; the synthetic
validation's tolerances (W.9.9 P2-11)."""
import dataclasses

import numpy as np
import pytest

from flymon.brain import w_oc
from flymon.brain import w_verdict as WV
from flymon.brain.w_spec import SPEC

Z = {"A": (16.917, 12.484), "P": (80.167, 29.775)}
SP = dataclasses.replace(SPEC, oc_chunk=50)


@pytest.fixture(scope="module")
def theta():
    rng = np.random.default_rng(3)
    return w_oc.fit(w_oc.synthetic_pilot(rng, base=(40.0, 90.0), sd=4.0, corr=0.8, learn=(20.0, 10.0)))


def test_fit_recovers_the_synthetic_structure():
    rng = np.random.default_rng(3)
    theta = w_oc.fit(w_oc.synthetic_pilot(rng, base=(40.0, 90.0), sd=4.0, corr=0.8, learn=(20.0, 10.0), drift_ax=4.0))
    assert theta["m0s"].shape == (2, 2) and np.allclose(theta["m0s"][0], 40.0, atol=1.0)
    assert np.allclose(theta["m0s"][1], 90.0, atol=1.0) and theta["m0s"][0, 0] == theta["m0s"][0, 1]
    n1, n2, rn2 = theta["drift"]
    assert abs(n2[WV.A, WV.X] + 4.0) < 0.5 and abs(rn2[WV.A, WV.X] + 4.0) < 0.5 and abs(n1[WV.A, WV.X]) < 0.5
    assert theta["spill"] == (0.0, 0.0) or max(theta["spill"]) < 0.05
    assert len(theta["resid"]) == 16 * 8 * 8 and theta["resid"].shape[1:] == (6, 2, 2)
    assert np.all(np.linalg.eigvalsh(theta["fly_cov"]) >= -1e-9)
    tb = w_oc.boot(theta, np.random.default_rng(0))
    assert tb["pair_cov"].shape == (4, 4) and len(tb["resid_blocks"]) == 16 and not np.allclose(tb["m0"], theta["m0"])


def test_calibration_meets_the_gate_vector_targets(theta):
    idx = np.random.default_rng(1).integers(0, len(theta["resid"]), SP.cal_reps)
    c = w_oc.calibrate(theta, 1.5, "min", idx, Z, SP)
    td = np.array(list(c["true_dprime"].values()))
    assert c["ok"] and abs(min(td[0], td[2]) - 1.5) <= SP.cal_tol and abs(min(td[1], td[3]) - 1.5) <= SP.cal_tol
    f = w_oc.calibrate(theta, 0.5, "max", idx, Z, SP)
    td = np.array(list(f["true_dprime"].values()))
    assert f["ok"] and abs(max(td[0], td[2]) - 0.5) <= SP.cal_tol and abs(max(td[1], td[3]) - 0.5) <= SP.cal_tol
    hi = w_oc.calibrate(theta, 40.0, "min", idx, Z, SP)
    assert not hi["ok"] and hi["a"]["status"] in ("floor", "no_convergence")


def test_w910_zero_injection_null_when_drift_already_exceeds_half():
    """W.9.10 2: a drift that alone puts the punishment drop above 0.5 at b = 0 keeps b at zero injection (the null
    "DAN 주입 0"), the reward handle is still bisected to 0.5, and the design is not unreachable; the literal rule
    ("unreachable") stays available through the switch."""
    assert SP.cal_floor_rule == "zero"
    rng = np.random.default_rng(4)
    pilot = w_oc.synthetic_pilot(rng, base=(40.0, 90.0), sd=4.0, corr=0.8, learn=(20.0, 10.0), drift_ax=15.0)
    th = w_oc.fit(pilot)
    idx = rng.integers(0, len(th["resid"]), SP.cal_reps)
    td0 = w_oc.true_dprimes(th, 0.0, 0.0, idx, Z)
    assert max(td0[1], td0[3]) >= 0.5 and max(td0[0], td0[2]) < 0.5 - SP.cal_tol
    f = w_oc.calibrate(th, 0.5, "max", idx, Z, SP)
    assert f["ok"] and f["b"]["status"] == "zero_floor" and f["b"]["value"] == 0.0
    assert f["a"]["status"] == "ok" and f["a"]["value"] > 0.0
    td = np.array(list(f["true_dprime"].values()))
    assert abs(max(td[0], td[2]) - 0.5) <= SP.cal_tol and max(td[1], td[3]) >= 0.5
    lit = w_oc.calibrate(th, 0.5, "max", idx, Z, dataclasses.replace(SP, cal_floor_rule="unreachable"))
    assert not lit["ok"] and lit["b"]["status"] == "above_at_zero"
    doc = w_oc.run(pilot, Z, SP, lambda K, F: K * F / 64, n_boot=0, n_rep=20)
    assert doc["reachable"] and doc["calibration"]["max"]["ok"]
    assert doc["calibration"]["max"]["b"]["status"] == "zero_floor"


def test_w910_normal_case_bisects_both_handles(theta):
    """W.9.10 2: with no drift above 0.5 at zero injection both handles are bisected to max = 0.5 (±0.02)."""
    idx = np.random.default_rng(1).integers(0, len(theta["resid"]), SP.cal_reps)
    td0 = w_oc.true_dprimes(theta, 0.0, 0.0, idx, Z)
    assert max(td0[0], td0[2]) < 0.5 and max(td0[1], td0[3]) < 0.5
    f = w_oc.calibrate(theta, 0.5, "max", idx, Z, SP)
    assert f["ok"] and f["a"]["status"] == "ok" and f["b"]["status"] == "ok"
    assert f["a"]["value"] > 0.0 and f["b"]["value"] > 0.0
    td = np.array(list(f["true_dprime"].values()))
    assert abs(max(td[0], td[2]) - 0.5) <= SP.cal_tol and abs(max(td[1], td[3]) - 0.5) <= SP.cal_tol


def test_w910_drift_dprime_record():
    """W.9.10 2: the drift-only gate d′ (zero injection) is recorded per pair and per gate — the population pair
    (what the calibration sees) and every pilot pair (its own drift on its X/Y-averaged naive mean)."""
    rng = np.random.default_rng(4)
    pilot = w_oc.synthetic_pilot(rng, base=(40.0, 90.0), sd=4.0, corr=0.8, learn=(20.0, 10.0), drift_ax=15.0)
    th = w_oc.fit(pilot)
    idx = rng.integers(0, len(th["resid"]), SP.cal_reps)
    rec = w_oc.drift_dprimes(th, idx, Z)
    assert set(rec) == {"population", "pilot_pairs", "level_gates"}
    assert set(rec["population"]) == set(WV.GATES) and len(rec["pilot_pairs"]) == 16
    assert all(set(r) == set(WV.GATES) for r in rec["pilot_pairs"])
    assert rec["level_gates"] == ["reward_level", "punish_drop"]
    assert np.allclose(list(rec["population"].values()), w_oc.true_dprimes(th, 0.0, 0.0, idx, Z))
    assert rec["population"]["punish_drop"] >= 0.5
    assert np.median([r["punish_drop"] for r in rec["pilot_pairs"]]) >= 0.5
    doc = w_oc.run(pilot, Z, SP, lambda K, F: K * F / 64, n_boot=0, n_rep=20)
    assert set(doc["drift_dprime"]) == set(rec) and doc["drift_dprime"]["population"]["punish_drop"] >= 0.5
    assert len(doc["drift_dprime"]["pilot_pairs"]) == 16


def test_evaluation_runs_the_verdict_code(theta, monkeypatch):
    rng = np.random.default_rng(2)
    args = w_oc._uniform(SP, 30.0, 15.0)
    p = w_oc.evaluate(theta, rng, 100, *args, 0.0, Z, SP)
    assert p.shape == (3, 2, 25, 5) and p.max() > 0.5
    assert np.all(np.diff(p, axis=-1) <= 1e-12)                    # nested pairs: more pairs never pass more
    monkeypatch.setattr(WV, "pair_final", lambda *a: np.full(np.asarray(a[0]).shape, WV.P_FAIL))
    assert w_oc.evaluate(theta, np.random.default_rng(2), 100, *args, 0.0, Z, SP).max() == 0.0


def test_rn1_bit_in_the_simulator_stops_the_machine(theta, monkeypatch):
    real = w_oc.to_stages

    def flipped(c):
        d = real(c)
        d["RN1"] = d["RN1"].copy()
        d["RN1"][..., 0, 0, 0, 0] += 1
        return d
    args = w_oc._uniform(SP, 30.0, 15.0)
    assert w_oc.evaluate(theta, np.random.default_rng(2), 50, *args, 0.0, Z, SP).max() > 0.5
    monkeypatch.setattr(w_oc, "to_stages", flipped)
    assert w_oc.evaluate(theta, np.random.default_rng(2), 50, *args, 0.0, Z, SP).max() == 0.0


def test_determinism(theta):
    args = w_oc._uniform(SP, 25.0, 12.0)
    a = w_oc.evaluate(theta, np.random.default_rng(9), 60, *args, 0.5, Z, SP)
    b = w_oc.evaluate(theta, np.random.default_rng(9), 60, *args, 0.5, Z, SP)
    assert np.array_equal(a, b)


def test_qualify_envelope_and_solo():
    nf = SP.f_max - SP.f_min + 1
    lo = np.zeros((3, 2, nf))
    hi = np.ones((3, 2, nf))
    ok_f = [10, 11, 12, 13, 15, 29]                                 # F 14 fails: 10 needs 10-13 → qualifies
    for F in ok_f:
        lo[2, 0, F - SP.f_min], hi[2, 0, F - SP.f_min] = 0.9, 0.0
    q = w_oc.qualify(lo, hi, SP)
    got = [SP.f_min + i for i in np.flatnonzero(q[2, 0])]
    assert got == [10, 29]                                          # 11: 11-14 has 14; 29 alone (solo rule)
    lo[2, 0, 30 - SP.f_min], hi[2, 0, 30 - SP.f_min] = 0.9, 0.06    # false pass above 0.05
    assert not w_oc.qualify(lo, hi, SP)[2, 0, 30 - SP.f_min]


def test_select_cost_then_q_then_k():
    nf = SP.f_max - SP.f_min + 1
    qual = np.zeros((3, 2, nf), bool)
    qual[0, 0, 2] = qual[2, 0, 2] = qual[2, 1, 0] = True
    sel, rank = w_oc.select(qual, lambda K, F: K * F / 64, SP)
    assert sel == dict(q=0.75, K=8, F=10, cost_h=1.25) and [(r["q"], r["K"], r["F"]) for r in rank] == [
        (0.75, 8, 10), (0.5, 8, 10), (0.75, 16, 8)]
    assert w_oc.select(np.zeros_like(qual), lambda K, F: 1.0, SP) == (None, [])


def test_run_small_is_deterministic(theta):
    rng = np.random.default_rng(3)
    pilot = w_oc.synthetic_pilot(rng, base=(40.0, 90.0), sd=4.0, corr=0.8, learn=(20.0, 10.0))
    a = w_oc.run(pilot, Z, SP, lambda K, F: K * F / 64, n_boot=2, n_rep=60, n_boot_rep=30)
    b = w_oc.run(pilot, Z, SP, lambda K, F: K * F / 64, n_boot=2, n_rep=60, n_boot_rep=30)
    for k in ("power_lo", "false_hi", "qualified", "selected", "ranking", "point"):
        assert a[k] == b[k], k
    assert set(a["timing"]) == {"point_s", "boot_s", "records_s", "total_s"} and a["reachable"]


def test_per_k_bootstrap_limits_w910_item3():
    """W.9.10 item 3: per-k limits [q, K, F, k] (power 5th percentile, false pass 95th, worst over g); nested in k, so
    min_k / max_k of them equal the combined limits; limits_at_f gives the F = 32 per-k limits per (q, K)."""
    rng = np.random.default_rng(3)
    pilot = w_oc.synthetic_pilot(rng, base=(40.0, 90.0), sd=4.0, corr=0.8, learn=(20.0, 10.0))
    doc = w_oc.run(pilot, Z, SP, lambda K, F: K * F / 64, n_boot=3, n_rep=40, n_boot_rep=40)
    nf, nk = SP.f_max - SP.f_min + 1, SP.k_cap - SP.k_min + 1
    pk, fk = np.asarray(doc["power_lo_by_k"]), np.asarray(doc["false_hi_by_k"])
    assert pk.shape == fk.shape == (len(SP.q_grid), len(SP.k_grid), nf, nk)
    assert doc["by_k_axes"] == ["q", "K", "F", "k"] and doc["k_values"] == [4, 5, 6, 7, 8]
    assert np.allclose(pk.min(-1), doc["power_lo"]) and np.allclose(fk.max(-1), doc["false_hi"])
    rows = w_oc.limits_at_f(doc, 32)
    assert len(rows) == len(SP.q_grid) * len(SP.k_grid)
    r = rows[0]
    assert r["F"] == 32 and r["k"] == [4, 5, 6, 7, 8] and len(r["power_lo"]) == len(r["false_hi"]) == nk
    assert np.allclose(r["power_lo"], pk[0, 0, 32 - SP.f_min]) and np.allclose(r["false_hi"], fk[0, 0, 32 - SP.f_min])
    assert len(doc["calibration_notes"]) == 2 and "above_at_zero" in doc["calibration_notes"][1]
    empty = w_oc.run(pilot, Z, SP, lambda K, F: K * F / 64, n_boot=0, n_rep=20)
    assert np.asarray(empty["power_lo_by_k"]).shape == pk.shape


def test_synthetic_validation_meets_p2_11():
    r = w_oc.synthetic_validation(SP, Z, n_rep=300)
    assert r["ok"], r
    assert r["zero_effect"]["max_p"] <= 0.02 and r["big_effect"]["min_p"] >= 0.98
    assert r["one_gate"]["max_p"] <= 0.02 and r["negative_correlation"]["max_p"] <= 0.02
    sn = r["simple_normal"]["p_by_F"]
    p = np.array([sn[k] for k in ("8", "16", "24", "32")])
    assert np.all(np.diff(p) <= w_oc.SIMPLE_NORMAL_TOL)            # never rises at any step 8 -> 16 -> 24 -> 32
    assert r["simple_normal"]["diffs"] == np.diff(p).tolist()
