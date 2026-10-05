"""Y's pilot θ and order-5 computations (Y.4 θ, Y.6.1, Y.6.5 on STOP, Y.9.2 P1-3 / P1-4 / P1-5 / P2-10 / P2-12):
Σ = diagonal of the merged pairs + V0 correlation, the merge and Σ-rule mutations, boot_y, the precheck's layout,
scenario × g worst, fill exclusion, kept k ranges only, the two-stair record, the STOP records with every variant
(R-pre reusing the filtered pre), the small bootstrap and the threshold recomputation."""
import dataclasses
import json

import numpy as np
import pytest

from flymon.brain import w_oc
from flymon.brain import y_oc as O
from flymon.brain import y_rules as R
from flymon.brain.h3_store import canonical
from flymon.brain.y_spec import SPEC as Y

Z = {"A": (16.917, 12.484), "P": (80.167, 29.775)}
KEYS7 = list(Y.pilot_w_pairs) + list(Y.pilot_v_pairs)
TINY = dataclasses.replace(Y, precheck_reps=4, oc_reps=4, small_boot_draws=2, oc_chunk=4)


@pytest.fixture(scope="module")
def pilot7():
    return w_oc.synthetic_pilot(np.random.default_rng(11), n_pair=7, base=(40.0, 90.0), sd=4.0, corr=0.8,
                                learn=(20.0, 10.0))


@pytest.fixture(scope="module")
def theta_w():
    return w_oc.fit(w_oc.synthetic_pilot(np.random.default_rng(12), n_pair=16, base=(40.0, 90.0), sd=6.0, corr=0.5,
                                         learn=(20.0, 10.0)))


def test_fit_y_sigma_rule(pilot7, theta_w):
    r = O.r_v0(theta_w)
    th = O.fit_y(pilot7, KEYS7, r)
    assert th["n_sigma"] == 6 and th["groups"][4] == ["a|218|Earthquake|Strength", "a|305|Earthquake|Rock Slide"]
    s = np.sqrt(np.diag(th["pair_cov"]))
    assert np.allclose(th["pair_cov"] / np.outer(s, s), r, atol=1e-12)
    pm = th["pair_means"].reshape(7, -1)
    merged = np.stack([pm[g].mean(0) for g in R.merge_groups(KEYS7)])
    assert np.allclose(np.diag(th["pair_cov"]), merged.var(0, ddof=1), atol=1e-12)
    un = O.fit_y(pilot7, KEYS7, r, merge=False)
    assert un["n_sigma"] == 7 and np.allclose(np.diag(un["pair_cov"]), pm.var(0, ddof=1), atol=1e-12)
    assert w_oc.summary(th)["n_pairs"] == 7


def test_theta_fixtures_and_mutations(monkeypatch):
    assert O.theta_fixtures(Y)["ok"]
    monkeypatch.setattr(O.y_rules, "merge_groups", lambda keys: [[i] for i in range(len(keys))])
    assert not O.theta_fixtures(Y)["merge"]["ok"]
    monkeypatch.undo()
    monkeypatch.setattr(O, "sigma_diag_v0", lambda pm, groups, r: np.cov(np.asarray(pm).reshape(len(pm), -1),
                                                                         rowvar=False))
    assert not O.theta_fixtures(Y)["sigma_rule"]["ok"]


def test_boot_y_keeps_the_rule(pilot7, theta_w):
    r = O.r_v0(theta_w)
    th = O.fit_y(pilot7, KEYS7, r)
    tb = O.boot_y(th, np.random.default_rng(0), r)
    s = np.sqrt(np.diag(tb["pair_cov"]))
    assert tb["n_sigma"] == th["n_sigma"] and np.allclose(tb["pair_cov"] / np.outer(s, s), r, atol=1e-12)


def test_precheck_layout_worst_and_ranges(pilot7, theta_w):
    th = O.fit_y(pilot7, KEYS7, O.r_v0(theta_w))
    pc = O.precheck_y(th, Z, TINY, [(4, 8)])
    assert O.SCENARIOS == ("base", "near") and pc["seed"] == Y.precheck_seed
    assert set(pc["point"]) == {f"g{g}|{m}|{s}" for g in Y.cluster_grid for m in ("min", "max")
                                for s in ("base", "near")}
    pw = np.min([pc["point"][k] for k in pc["point"] if "|min|" in k], 0)
    assert np.array_equal(np.asarray(pc["power"]), pw)
    assert pc["best_power"]["k_range"] == [4, 8] and pc["records_target"]["k_range"] == [4, 8]
    assert all(r["k_range"] == [4, 8] for r in pc["at_f32"]) and pc["k_ranges"] == [[4, 8]]
    assert set(pc["stairs"]) == {"lo", "hi"} and len(pc["fill_bad"]) == 7
    json.loads(canonical(pc))


def test_precheck_respects_kept_ranges_and_fill(pilot7, theta_w, monkeypatch):
    th = O.fit_y(pilot7, KEYS7, O.r_v0(theta_w))
    easy = dataclasses.replace(TINY, p_power=0.0, p_false=1.0)          # every design meets the targets
    pc = O.precheck_y(th, Z, easy, [(6, 10)])
    assert pc["passed"] and {tuple(d["k_range"]) for d in pc["passing"]} == {(6, 10)}
    real = O.evaluate_y

    def filled(*a, **k):
        out = real(*a, **k)
        return dict(out, fill_by_k=[0.0] * 2 + [0.5] * 5)               # k 6.. over 1 %: [6, 10] excluded
    monkeypatch.setattr(O, "evaluate_y", filled)
    pc2 = O.precheck_y(th, Z, dataclasses.replace(easy, precheck_seed=easy.precheck_seed + 1), [(4, 8), (6, 10)])
    assert not pc2["passed"] and pc2["fill_bad"] == [False, False, True, True, True, True, True]


def test_precheck_calibration_failure_fills(pilot7, theta_w):
    th = O.fit_y(pilot7, KEYS7, O.r_v0(theta_w))
    pc = O.precheck_y(th, Z, dataclasses.replace(TINY, d_power=400.0), [(4, 8)])
    assert not pc["passed"] and pc["calibration"]["min"]["ok"] is False
    assert np.asarray(pc["power"]).max() == 0.0
    assert "보정 불가 — min: a floor" in R.precheck_decision(pc, 7, 6, Y)["sentence"]


def test_small_bootstrap_and_thresholds(pilot7, theta_w):
    r = O.r_v0(theta_w)
    th = O.fit_y(pilot7, KEYS7, r)
    sb = O.small_bootstrap(th, Z, TINY, r)
    assert sb["n_draws"] == 2 and sb["seed"] == Y.small_boot_seed and set(sb["counts"]) == {"min", "max"}
    idx = O.rng(Y.precheck_seed, O.TAG_CAL).integers(0, len(th["resid"]), Y.cal_reps)
    cal = O.calibrate_y(th, Y.d_power, "min", idx, Z, Y)
    t = O.threshold_recompute(th, cal, Y)
    assert t["available"] and t["lo"]["n_resid"] == Y.floor_resid_n and set(t) >= {"lo", "hi", "note"}
    assert O.threshold_recompute(th, dict(ok=False, failure=dict(handle="a", status="floor")), Y)["available"] is False


def test_pre_filter_generator_uses_the_real_filter(pilot7, theta_w):
    th = O.fit_y(pilot7, KEYS7, O.r_v0(theta_w))
    d, filled = O.simulate_pre(th, O.rng(1, 1), 3, 4, 8, 16, 8, np.zeros((3, 4)), np.zeros((3, 4)), [1.0] * 8,
                               [1.0] * 8, 0.0, Z, Y)
    pre = d["pre"][..., :8, :, :]
    dv, la, lp = R.filter_values(pre, Z)
    assert (R.filter_passes(dv, la, lp, Y) | filled).all()


def test_point_records_layout(pilot7, theta_w):
    r = O.r_v0(theta_w)
    th = O.fit_y(pilot7, KEYS7, r)
    cands = dict(zip(KEYS7, pilot7))
    thetas, notes = O.record_thetas(th, theta_w, cands, KEYS7, r, Y)
    assert set(thetas) == {"Y", "R-W", "R-Σ2", "R-un", "R-V", "R-pre"} and thetas["R-V"]["n_sigma"] == 3
    assert np.allclose(thetas["R-Σ2"]["pair_cov"], th["pair_cov"] * Y.sigma2_scale)
    target = dict(p_set=0.5, q=0.5, K=8, F=8, k_range=[4, 8])
    rec = O.point_records_y(thetas, target, Z, dataclasses.replace(TINY, oc_reps=2))
    y = rec["variants"]["Y"]["g0.0"]
    assert set(y) >= {"true_dprime", "false_pass", "mixed_flies", "heterogeneous_w", "pair_cov_scaled", "stairs"}
    assert set(rec["variants"]["R-pre"]["g0.0"]) >= {"true_dprime", "false_pass", "fill"}
    assert rec["roots"] == {"Y": Y.precheck_seed, "R-W": Y.precheck_seed, "R-Σ2": Y.precheck_seed,
                            "R-un": Y.precheck_seed, "R-V": Y.records_seed, "R-pre": Y.records_seed}
    json.loads(canonical(rec))
    one_v = dict(zip(KEYS7[:3] + KEYS7[4:6], pilot7[:5]))
    thetas2, notes2 = O.record_thetas(th, theta_w, one_v, KEYS7[:3] + KEYS7[4:6], r, Y)
    assert thetas2["R-V"] is None and "R-V" in notes2                 # the Earthquake pair merges into 1 < 2


def test_at_f32_rows_hold_the_worst_over_g_and_scenarios(pilot7, theta_w):
    """Y.8's STOP_OC_UNREACHABLE(precheck) parenthesis: F = 32 k-wise point power / false are "군집 최악" — the min
    power / max false over every g and scenario, not any single (g, scenario) cell."""
    th = O.fit_y(pilot7, KEYS7, O.r_v0(theta_w))
    pc = O.precheck_y(th, Z, TINY, [(4, 8), (6, 10)])
    fi = Y.f_max - Y.f_min
    pt = {k: np.asarray(v) for k, v in pc["point"].items()}
    labs = [(g, s) for g in Y.cluster_grid for s in O.SCENARIOS]
    assert len(pc["at_f32"]) == 2 * len(Y.p_set_grid) * len(Y.q_grid) * len(Y.k_grid)
    differs = False
    for r in pc["at_f32"]:
        assert r["F"] == Y.f_max == 32
        i = (Y.p_set_grid.index(r["p_set"]), Y.q_grid.index(r["q"]), Y.k_grid.index(r["K"]), fi)
        ks = [k - Y.k_min for k in r["k"]]
        pw = np.stack([pt[f"g{g}|min|{s}"][i][ks] for g, s in labs])
        fp = np.stack([pt[f"g{g}|max|{s}"][i][ks] for g, s in labs])
        assert np.array_equal(np.asarray(r["power"]), pw.min(0))
        assert np.array_equal(np.asarray(r["false"]), fp.max(0))
        differs |= bool((pw != pw.min(0)).any() or (fp != fp.max(0)).any())
    assert differs                                       # the worst is a real reduction, not a single cell
    para = R.precheck_paren(pc)
    r0 = pc["at_f32"][0]
    assert f"k {r0['k'][0]} 점 검정력 {r0['power'][0]:.3f} / 점 거짓 통과 {r0['false'][0]:.3f}" in para


def test_records_never_reuse_a_precheck_mixture_stream(pilot7, theta_w, monkeypatch):
    """The precheck's mixture streams stay SeedSequence([precheck_seed, "mix", 0, g, target, scenario]) (bit for bit
    as before); the records under the same root add their variant tags, so no record cell reuses one of them."""
    r = O.r_v0(theta_w)
    th = O.fit_y(pilot7, KEYS7, r)
    seen = []
    real = O.rng

    def spy(root, *tags):
        if tags and tags[0] == O.tag("mix"):
            seen.append((int(root),) + tuple(int(t) for t in tags))
        return real(root, *tags)
    monkeypatch.setattr(O, "rng", spy)
    O.precheck_y(th, Z, TINY, [(4, 8)])
    pre = set(seen)
    root, mix = Y.precheck_seed, O.tag("mix")
    assert pre == {(root, mix, 0, gi, mi, si) for gi in range(len(Y.cluster_grid)) for mi in range(2)
                   for si in range(2)}
    seen.clear()
    thetas, _ = O.record_thetas(th, theta_w, dict(zip(KEYS7, pilot7)), KEYS7, r, Y)
    target = dict(p_set=0.5, q=0.5, K=8, F=8, k_range=[4, 8])
    O.point_records_y(thetas, target, Z, dataclasses.replace(TINY, oc_reps=2))
    rec = set(seen)
    assert {k[0] for k in rec} == {Y.precheck_seed, Y.records_seed} and not (rec & pre)
