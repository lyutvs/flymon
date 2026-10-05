"""X.9.1.4's record-only diagnostics: per-pair true d′ = w_oc.true_dprimes on that pair's base, the floor flags at
0.10, the filters (F1 Σ from the balanced pairs, F2(ℓ) percentile thresholds on MBON13(X) / MBON05(X)), rejection that
cannot be met fills and is counted, the pair-level counters add up, the minimum-change ranking order, the full run's
layout and that it never changes the gate."""
import numpy as np
import pytest

from flymon.brain import w_oc
from flymon.brain import w_verdict as WV
from flymon.brain import x_oc
from flymon.brain.x_spec import SPEC as XS

Z = {"A": (16.917, 12.484), "P": (80.167, 29.775)}
ND = np.array([0.08, -0.05, -0.32, 1.09, 1.43, 2.72, 4.13, 5.9, 6.0, 7.98, -9.89, 10.7, -12.88, 13.3, 14.5, -16.07])


@pytest.fixture(scope="module")
def theta():
    return w_oc.fit(w_oc.synthetic_pilot(np.random.default_rng(3), base=(40.0, 90.0), sd=4.0, corr=0.8,
                                         learn=(20.0, 10.0)))


@pytest.fixture(scope="module")
def cal(theta):
    idx = x_oc.rng(XS.diag_seed, x_oc.TAG_CAL).integers(0, len(theta["resid"]), XS.cal_reps)
    return idx, x_oc.calibrate_x(theta, XS.d_power, "min", idx, Z, XS)


def test_pair_true_dprimes_equal_w(theta, cal):
    idx, c = cal
    a, b = c["a"]["value"], c["b"]["value"]
    bases, _ = x_oc.pair_bases(theta, np.random.default_rng(4), 3, 5, 1.0, Z, XS, XS.w_rejection_tries)
    bases = bases.reshape(-1, 2, 2)
    td, fl = x_oc.pair_true_dprimes(theta, bases, a, b, idx, Z, XS)
    for i, bs in enumerate(bases):
        assert np.allclose(td[i], w_oc.true_dprimes(dict(theta, m0s=bs), a, b, idx, Z), rtol=0, atol=1e-12)
        mu = w_oc.slot_means(theta, bs, np.zeros((2, 2)), a, b)
        r = theta["resid"][idx]
        low_r = (np.rint(mu[w_oc.SI["R1"], WV.P, WV.X] + r[:, w_oc.SI["R1"], WV.P, WV.X]) < 0).mean()
        low_p = (np.rint(mu[w_oc.SI["R2"], WV.A, WV.X] + r[:, w_oc.SI["R2"], WV.A, WV.X]) < 0).mean()
        assert fl[i].tolist() == [bool(low_r >= 0.10), bool(low_p >= 0.10)]


def test_pair_bases_reject_all_fills(theta):
    never = lambda b: np.zeros(b.shape[:-2], bool)          # noqa: E731
    bases, filled = x_oc.pair_bases(theta, np.random.default_rng(0), 4, 3, 0.0, Z, XS, 5, never)
    assert filled.all() and np.allclose(bases, theta["m0s"])
    r = x_oc.evaluate_grid(theta, np.random.default_rng(1), 2, 0.0, 0.0, 0.0, Z, XS, never)
    assert r["fill_share"] == 1.0 and r["p"].shape == (5, 3, 2, 57, 13)


def test_filters(theta):
    f = x_oc.filters(theta, ND, XS)
    assert list(f) == ["none", "F1", "F2(0)", "F2(25)", "F2(50)", "F2(75)"]
    assert f["none"]["accept"] is None and f["F1"]["accept"] is None
    assert np.allclose(f["F1"]["theta"]["pair_cov"], x_oc.pair_cov_variant(theta["pair_means"], ND, "F1", XS))
    pm = theta["pair_means"]
    c = f["F2(50)"]["pass_counts"]
    assert c["c_A"] == np.percentile(pm[:, WV.A, WV.X], 50.0) and c["c_P"] == np.percentile(pm[:, WV.P, WV.X], 50.0)
    keep = (pm[:, WV.A, WV.X] >= c["c_A"]) & (pm[:, WV.P, WV.X] >= c["c_P"])
    assert c["pilot"] == int(keep.sum()) and c["balanced"] == int((keep & (np.abs(ND) < 0.5)).sum())
    assert f["F1"]["pass_counts"] == dict(pilot=3, balanced=3)


def test_pair_diag_counts_add_up(theta, cal):
    idx, c = cal
    out = x_oc.pair_diag(theta, c, idx, XS.diag_seed, 2, Z, XS)
    for g in XS.cluster_grid:
        cells = out[f"g{g}"]["cells"]
        assert set(cells) == {f"{q}|{K}|{F}" for q in XS.q_grid for K in XS.k_grid for F in XS.diag_fs}
        for v in cells.values():
            n = v["counts"]
            assert n["n"] == 2 * 16 and n["n_pass"] + n["n_fail"] == n["n"]
            assert sum(sum(r) for r in n["table"]) == n["n_fail"]
            assert n["mech"] + n["joint_only"] <= n["n_fail"]
    assert set(out["worst_g"]) == set(out["g0.0"]["cells"])


def test_min_change_order():
    fs, kk = list(range(8, 65)), list(range(4, 17))
    shp = (5, 3, 2, len(fs), len(kk))
    zero, one = np.zeros(shp), np.ones(shp)
    pw = zero.copy()
    pw[:, :, :, fs.index(40):, :] = 0.9                       # F ≥ 40 passes for "none" at every k
    pw2 = zero.copy()
    pw2[:, :, :, fs.index(20):, kk.index(4):kk.index(8) + 1] = 0.9    # F1: F 20 at k 4-8
    out = x_oc.min_change({"none": (pw, zero), "F1": (pw2, zero), "F2(0)": (zero, one)}, XS, fs, kk)
    r1 = out["rank1"]                                         # knobs: F1 (filter) = 1, none at F 40 (F > 32) = 1
    assert (r1["filter"], r1["k_lo"], r1["F"], r1["knobs"]) == ("F1", 4, 20, 1)
    assert [(e["filter"], e["k_lo"], e["F"]) for e in out["same_knobs"]] == [("F1", 4, 20), ("none", 4, 40)]
    assert r1["p_set"] == 1.0 and r1["q"] == 0.75 and r1["K"] == 8      # ties: larger p_set, larger q, smaller K
    none6 = [e for e in out["entries"] if e["filter"] == "none" and e["k_lo"] == 6][0]
    assert none6["knobs"] == 2 and not [e for e in out["entries"] if e["filter"] == "F2(0)" and e["found"]]


def test_diagnostics_layout(theta):
    d = x_oc.diagnostics(theta, ND, Z, XS, n_rep=2)
    assert set(d) >= {"calibration", "filters", "ii", "iii", "i", "arrays", "axes", "seed", "n_rep", "timing_s"}
    assert d["seed"] == 43_300_000 and list(d["ii"]) == list(d["filters"])
    assert len(d["iii"]["entries"]) == 6 * 5
    ii = d["ii"]["F2(25)"]
    assert set(ii) >= {"best", "best_g0", "fill", "fill_flag", "pass_counts", "acceptance"}
    assert 0.0 <= ii["acceptance"] <= 1.0 and d["ii"]["none"]["acceptance"] is None
    s = x_oc.diag_summary(d)
    assert set(s) == {"ii", "iii", "i", "calibration", "note"}
