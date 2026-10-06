"""Z's order-0 computations on Y's code: the 0b gate helpers, the record functions' definitions, the 0c′ tiling /
shift / streams / limits (Z.9.2 P1-3, T8 · T13)."""
import dataclasses

import numpy as np
import pytest

from flymon.brain import w_oc, y_oc
from flymon.brain import w_verdict as WV
from flymon.brain import z_oc as O
from flymon.brain import z_store
from flymon.brain.y_spec import SPEC as Y
from flymon.brain.z_spec import SPEC as Z

ZV = {"A": (16.917, 12.484), "P": (80.167, 29.775)}


@pytest.fixture(scope="module")
def theta():
    kw = dict(base=(40.0, 90.0), sd=4.0, corr=0.8, learn=(20.0, 10.0))
    th = w_oc.fit(w_oc.synthetic_pilot(np.random.default_rng(3), n_pair=6, **kw))
    r = np.eye(4)
    keys = list(Z.y_admitted)
    return y_oc.fit_y(w_oc.synthetic_pilot(np.random.default_rng(3), n_pair=6, **kw), keys, r), r


def test_design_index_is_the_records_target():
    i, ks = O.design_index(Z)
    assert i == (0, 0, 1, 0) and ks == [0, 1, 2, 3, 4]
    p = np.arange(np.prod(y_oc.x_oc.grid_shape(Y))).reshape(y_oc.x_oc.grid_shape(Y)).astype(float)
    assert O.design_k(p, Z).tolist() == p[0, 0, 1, 0, :5].tolist()
    q = np.stack([p, p + 1])
    assert O.design_k(q, Z).shape == (2, 5)


def test_tiled_shift_moves_only_the_mbon13x_n1_centre(theta):
    th, _r = theta
    for n in Z.sens_ns:
        t0 = O.tiled(th, n, 0.0, Z.sigma_share)
        t1 = O.tiled(th, n, -2.41, Z.sigma_share)
        assert len(t0["pair_means"]) == n and t0["n_sigma"] == round(n * 5 / 6)
        assert np.allclose(t1["drift"][0][WV.A, WV.X] - t0["drift"][0][WV.A, WV.X], -2.41)
        d = t1["drift"] - t0["drift"]
        d[0][WV.A, WV.X] = 0.0
        assert np.allclose(d, 0.0)
        assert np.isclose(np.std(t1["drift_pair"][:, 0, WV.A, WV.X]), np.std(t0["drift_pair"][:, 0, WV.A, WV.X]))
        assert np.array_equal(t1["pair_cov"], th["pair_cov"]) and np.array_equal(t1["m0"], t0["m0"])
    assert not np.shares_memory(O.tiled(th, 12, -2.41, Z.sigma_share)["drift_pair"], th["drift_pair"])


def test_sens_job_streams_are_86m(theta, monkeypatch):
    th, r = theta
    seen = []
    real = y_oc.rng

    def spy(*a):
        seen.append(a)
        return real(*a)
    monkeypatch.setattr(O.y_oc, "rng", spy)
    monkeypatch.setattr(O.y_oc, "calibrate_y", lambda *a, **k: dict(ok=False, failure=dict(status="floor")))
    out = O.sens_job(th, r, ZV, 12, 1, -2.41, 7, Z.sens_seed, Z.sigma_share)
    assert seen[0] == (Z.sens_seed, y_oc.TAG_BOOT, 12, 1, 7)
    assert out["meta"]["ok"] is False and not out["arrays"]["p"].any()


def test_power_job_streams_with_a_calibration(theta, monkeypatch):
    th, r = theta
    seen = []
    real = y_oc.rng
    monkeypatch.setattr(O.y_oc, "rng", lambda *a: (seen.append(a), real(*a))[1])
    monkeypatch.setattr(O.y_oc, "calibrate_y", lambda *a, **k: dict(ok=True, failure=None,
                                                                     corners=[dict(a=5.0, b=1.0, w=1.0)]))
    monkeypatch.setattr(O.y_oc, "evaluate_y", lambda *a, **k: dict(p=np.full(y_oc.x_oc.grid_shape(Y), 0.5)))
    out = O.sens_job(th, r, ZV, 24, 2, -4.82, 3, Z.sens_seed, Z.sigma_share)
    sims = [s for s in seen if s[1] == y_oc.TAG_BOOT_SIM]
    mixes = [s for s in seen if s[1] == y_oc.tag("mix")]
    assert len(sims) == len(mixes) == len(Y.cluster_grid) * len(y_oc.SCENARIOS)
    assert sims[0] == (Z.sens_seed, y_oc.TAG_BOOT_SIM, 24, 2, 3, 0, 0) and mixes[-1][2:] == (24, 2, 3, 2, 1)
    assert out["meta"]["a"] == 5.0 and np.allclose(out["arrays"]["p"], 0.5)


def test_sens_job_reps_come_from_sens_reps(theta, monkeypatch):
    th, r = theta
    reps = []
    monkeypatch.setattr(O, "Z_SPEC", dataclasses.replace(Z, sens_reps=7))
    monkeypatch.setattr(O.y_oc, "calibrate_y", lambda *a, **k: dict(ok=True, failure=None,
                                                                     corners=[dict(a=5.0, b=1.0, w=1.0)]))
    monkeypatch.setattr(O.y_oc, "evaluate_y", lambda tb, s, m, n_rep, *a, **k: (
        reps.append(n_rep), dict(p=np.zeros(y_oc.x_oc.grid_shape(Y))))[1])
    O.sens_job(th, r, ZV, 12, 0, 0.0, 0, Z.sens_seed, Z.sigma_share)
    assert reps == [7] * (len(Y.cluster_grid) * len(y_oc.SCENARIOS))


def test_sim_limit_and_sens_summary():
    rng = np.random.default_rng(0)
    a = rng.uniform(0.5, 1.0, (100, 3, 2, 5))
    lim = O.sim_limit(a)
    assert np.isclose(lim["sim"], min(np.percentile(a[:, g, s].min(-1), 5) for g in range(3) for s in range(2)))
    assert np.isclose(lim["base_only"], min(np.percentile(a[:, g, 0].min(-1), 5) for g in range(3)))
    lo = np.percentile(a, 5, axis=0).reshape(-1, a.shape[-1]).min(0)
    assert np.allclose(lim["by_k"], lo) and not np.allclose(lim["by_k"], np.median(a, 0).reshape(-1, 5).min(0))
    s = O.sens_summary(a, [dict(ok=True, a=1.0, drift_ax=-11.0)] * 100, Z)
    assert s["n_below_bar"] == int((a.min(axis=(1, 2, 3)) < 0.8).sum()) and s["n_cal_fail"] == 0
    b = np.full((4, 3, 2, 5), 0.9)
    b[0, 1, 1, 2] = Z.plan_bar                                    # worst exactly at the bar: not below (strict <)
    b[1, 0, 0, 0] = 0.5
    assert O.sens_summary(b, [dict(ok=True, a=1.0, drift_ax=0.0)] * 4, Z)["n_below_bar"] == 1


def _draws(n, rng, n_rep=400):
    shp = (3, 2, 2) + y_oc.x_oc.grid_shape(Y)
    cal = dict(ok=True, failure=None, corners=[dict(a=30.0, b=6.0, w=1.0)],
               a=dict(status="ok", coarse_step=False, stage=1), b=[dict(status="ok", coarse_step=False, stage=1)])
    return [dict(arrays=dict(hits=rng.integers(0, n_rep, shp).astype(np.uint16),
                             fills=np.zeros((3, 2, 2, Y.k_cap - Y.k_min + 1))),
                 meta=dict(cal=dict(min=cal, max=cal), n_rep=n_rep)) for _ in range(n)]


def test_gate_limits_against_its_own_json(monkeypatch):
    draws = _draws(20, np.random.default_rng(1))
    lim = y_oc.limits(draws, Y, [tuple(k) for k in Y.k_ranges])
    det = z_store.to_json(O.limits_json(lim))
    i, ks = O.design_index(Z)
    zs = dataclasses.replace(Z, gate_by_k=tuple(np.round(lim["power_lo_by_k"][i][ks], 3).tolist()),
                             gate_sim=round(float(lim["sim"][(4, 8)]["power"][i]), 3),
                             gate_false=float(np.round(lim["false_hi_by_k"][i][ks], 3).max()))
    g = O.gate_limits(draws, det, zs, z_store.to_json)
    assert g["ok"] and g["equal_limits"] and g["equal_sim"] and g["printed"]
    det2 = dict(det, limits=dict(det["limits"], n_rep=1))
    assert not O.gate_limits(draws, det2, zs, z_store.to_json)["ok"]
    assert not O.gate_limits(draws, det, dataclasses.replace(zs, gate_sim=0.5), z_store.to_json)["ok"]
    k0 = next(iter(det["sim"]))
    det3 = dict(det, sim=dict(det["sim"], **{k0: dict(det["sim"][k0], power=[-1.0])}))
    g3 = O.gate_limits(draws, det3, zs, z_store.to_json)
    assert g3["equal_limits"] and not g3["equal_sim"] and g3["printed"] and not g3["ok"]
    g4 = O.gate_limits(draws, det, dataclasses.replace(zs, gate_false=zs.gate_false + 0.5), z_store.to_json)
    assert g4["equal_limits"] and g4["equal_sim"] and not g4["printed"] and not g4["ok"]


def test_redraw_equal():
    d = _draws(1, np.random.default_rng(2))[0]
    same = dict(arrays={k: v.copy() for k, v in d["arrays"].items()}, meta=dict(d["meta"]))
    assert O.redraw_equal(same, d, z_store.to_json)["ok"]
    other = dict(same, arrays=dict(same["arrays"], hits=same["arrays"]["hits"] + 1))
    assert O.redraw_equal(other, d, z_store.to_json) == dict(hits=False, fills=True, meta=True, ok=False)
    meta = dict(same, meta=dict(same["meta"], n_rep=d["meta"]["n_rep"] + 1))
    assert O.redraw_equal(meta, d, z_store.to_json) == dict(hits=True, fills=True, meta=False, ok=False)
    fn = d["arrays"]["fills"].copy()
    fn[0, 0, 0, :2] = np.nan
    a = dict(same, arrays=dict(same["arrays"], fills=fn.copy()))
    b = dict(d, arrays=dict(d["arrays"], fills=fn.copy()))
    assert O.redraw_equal(a, b, z_store.to_json) == dict(hits=True, fills=True, meta=True, ok=True)


def test_draw_power_reads_the_power_side():
    rng = np.random.default_rng(5)
    draws = _draws(3, rng)
    for d in draws:
        d["arrays"]["hits"][:, 1] = 0                             # the other side: nothing
    i, ks = O.design_index(Z)
    pw = O.draw_power(draws, Z)
    want = np.stack([d["arrays"]["hits"][:, 0][(slice(None), slice(None)) + i][..., ks] for d in draws]) / 400.0
    assert pw.shape == (3, 3, 2, len(ks)) and np.allclose(pw, want) and pw.any()


def test_rec_distribution_definitions():
    rng = np.random.default_rng(4)
    pw = rng.uniform(0, 1, (200, 3, 2, 5))
    out = O.rec_distribution(pw, Z)
    w = pw.min(axis=(1, 2, 3))
    assert np.isclose(out["draw_q5"], np.percentile(w, 5)) and out["n_below_0.8"] == int((w < 0.8).sum())
    assert np.isclose(out["near_q5"], np.percentile(pw[:, :, 1].min(axis=(1, 2)), 5))
    assert set(k for k in out if k.startswith("low_near_g")) == {"low_near_g0", "low_near_g0.5", "low_near_g1"}
    low = w < Z.low_power
    for gi, g in enumerate(Y.cluster_grid):
        assert np.isclose(out[f"low_base_g{g:g}"], np.median(pw[low][:, gi, 0].min(-1)))
        assert np.isclose(out[f"low_near_g{g:g}"], np.median(pw[low][:, gi, 1].min(-1)))
    printed = {n for n, _, _ in Z.printed}
    assert {"draw_q0", "draw_q75", "n_low", "low_base_g1"} <= set(out) & printed


def test_drift_facts_definitions(theta):
    th, _r = theta
    f = O.drift_facts(th, th)
    d = th["drift_pair"][:, 0, WV.A, WV.X]
    assert np.isclose(f["y_drift_se"], d.std(ddof=1) / np.sqrt(len(d)))
    assert np.isclose(f["w_drift_corr"], np.corrcoef(d, th["pair_means"][:, WV.A, WV.X])[0, 1])


def test_unit_timing_records_seconds(theta):
    th, _r = theta
    zs = dataclasses.replace(Z, timing_reps=2)
    t = O.unit_timing(th, ZV, zs)
    assert t["evaluate_y_s_per_rep"] > 0 and t["evaluate_pre_s_per_rep"] > 0
    assert np.isclose(t["rpre_reconfirm_s_per_design"],
                      t["evaluate_pre_s_per_rep"] * Z.rpre_reps * Y.boot_draws * len(Y.cluster_grid) * 2)
