# tests/brain/test_n_verdict_oc.py
"""Spec N.8.6: raw-unit statistics with common-seed bootstrap CIs; each verdict branch pinned by a mutation of a
SUPPORTED base (INVALID first, then ①, then ② ③ ④), clause bounds inclusive; validity checks; N2.0's thresholds,
budget, blinding view and outcome order; the OC simulator (block arm = shifted mean, deviations x1 / x2, built from
the on arm's own resample AND from an independent one) picks the smallest n passing every spread x pairing scenario.
No rule passes on an empty or missing input: it raises or stops."""
from dataclasses import replace

import numpy as np
import pytest

from flymon.brain import n_rules as R
from flymon.brain.n_oc import arms, choose_n, off_arm, scenario_key, scenario_n, simulate
from flymon.brain.n_spec import SPEC

E = np.linspace(-0.1, 0.1, 16)
C1 = {"sim": 0.25, "dis": 0.25}
ZU = {"A": (0.0, 1.0), "P": (0.0, 1.0)}


def _d(sim_on=-1.0, sim_off=-0.2, dis_on=-1.0, dis_off=-1.0, spread=E):
    return {"sim_on": sim_on + spread, "sim_off": sim_off + spread, "dis_on": dis_on + spread,
            "dis_off": dis_off + spread}


def _v(d, c1=C1, invalid=()):
    return R.verdict(R.n2_stats(d, SPEC, draws=2000, seed=1), c1, 0.5, 0.25, list(invalid))


def test_base_case_is_supported():
    v = _v(_d())
    assert v["verdict"] == R.SUPPORTED and v["clauses"] == dict(one=True, two=True, three=True, four=True)


def test_invalid_comes_first():
    v = _v(_d(), invalid=["plumbing check failed: sim_on seed 1"])
    assert v["verdict"] == R.INVALID and v["reasons"] == ["plumbing check failed: sim_on seed 1"]


def test_small_on_learning_is_no_learning():
    v = _v(_d(sim_on=-0.2))
    assert v["verdict"] == R.NO_LEARNING and not v["clauses"]["one"]


def test_on_learning_whose_ci_reaches_zero_is_no_learning():
    d = _d(); d["sim_on"] = -0.5 + np.linspace(-3, 3, 16)
    st = R.n2_stats(d, SPEC, draws=2000, seed=1)
    assert st["ell"]["sim_on"] >= 0.25 and st["ell_ci95"]["sim_on"][0] <= 0
    assert R.verdict(st, C1, 0.5, 0.25)["verdict"] == R.NO_LEARNING


def test_a_small_block_effect_fails_clause_two():
    v = _v(_d(sim_off=-0.8))
    assert v["verdict"] == R.NOT_REPLICATED and v["clauses"] == dict(one=True, two=False, three=True, four=True)


def test_a_dissimilar_block_effect_fails_clause_three():
    v = _v(_d(dis_off=-0.5))
    assert v["verdict"] == R.NOT_REPLICATED and v["clauses"] == dict(one=True, two=True, three=False, four=True)


def test_lost_dissimilar_learning_fails_clause_four():
    v = _v(_d(dis_off=-0.9), c1={"sim": 0.25, "dis": 0.95})
    assert v["verdict"] == R.NOT_REPLICATED and v["clauses"] == dict(one=True, two=True, three=True, four=False)


def _st(**over):
    st = dict(n=16, ell={"sim_on": 0.25, "sim_off": 0.0, "dis_on": 0.25, "dis_off": 0.25},
              ell_ci95={k: [0.01, 1.0] for k in R.N2_ORDER}, D_sim=0.6, D_sim_ci95=[0.5, 0.7], D_dis=0.0,
              D_dis_ci90=[-0.25, 0.25])
    st.update(over)
    return st


def test_clause_bounds_are_inclusive_and_ci_lower_is_strict():
    assert R.verdict(_st(), C1, 0.5, 0.25)["verdict"] == R.SUPPORTED
    zero = _st(ell_ci95={k: [0.0, 1.0] for k in R.N2_ORDER})
    assert R.verdict(zero, C1, 0.5, 0.25)["verdict"] == R.NO_LEARNING
    assert R.verdict(_st(D_sim_ci95=[0.4999, 0.7]), C1, 0.5, 0.25)["clauses"]["two"] is False
    assert R.verdict(_st(D_dis_ci90=[-0.2501, 0.1]), C1, 0.5, 0.25)["clauses"]["three"] is False


# ---------------------------------------------------------------- no vacuous pass (Task 7 review, carried to N2)
def test_verdict_never_passes_on_missing_or_undefined_input():
    nan = float("nan")
    for bad in (_st(D_sim_ci95=[nan, nan]), _st(D_dis_ci90=[nan, nan]),
                _st(ell={"sim_on": 0.25, "sim_off": 0.0, "dis_on": 0.25, "dis_off": nan}),
                _st(ell_ci95={k: [nan, nan] for k in R.N2_ORDER})):
        assert R.verdict(bad, C1, 0.5, 0.25)["verdict"] != R.SUPPORTED
    st = _st(); del st["ell"]["dis_off"]
    with pytest.raises(KeyError):
        R.verdict(st, C1, 0.5, 0.25)
    with pytest.raises(KeyError):
        R.verdict(_st(), {"sim": 0.25}, 0.5, 0.25)
    for c1, dm, eps in ((C1, 0.0, 0.25), (C1, 0.5, 0.0), (C1, nan, 0.25), (C1, 0.5, nan), ({"sim": 0.0, "dis": 0.25},
                                                                                         0.5, 0.25)):
        with pytest.raises(ValueError, match="threshold"):                    # an unset threshold cannot judge
            R.verdict(_st(), c1, dm, eps)


def test_n2_stats_refuses_a_missing_condition_and_fewer_than_two_seeds():
    d = _d(); del d["dis_off"]
    with pytest.raises(KeyError):
        R.n2_stats(d, SPEC, draws=50, seed=1)
    with pytest.raises(ValueError, match="seeds"):
        R.n2_stats({k: np.array([]) for k in R.N2_ORDER}, SPEC, draws=50, seed=1)
    with pytest.raises(ValueError, match="seeds"):
        R.n2_stats(dict(_d(), sim_off=E[:8]), SPEC, draws=50, seed=1)


def test_n1_outcome_needs_both_pairs():
    t = {"testable": True}
    for pairs in ({}, {"sim": t}, {"dis": t}):
        with pytest.raises(KeyError):
            R.n1_outcome(pairs)


def test_n2_stats_resamples_seeds_in_common():
    st = R.n2_stats(_d(), SPEC, draws=500, seed=3)
    assert st["D_sim"] == pytest.approx(0.8) and st["D_sim_ci95"] == pytest.approx([0.8, 0.8])
    assert st["D_dis_ci90"] == pytest.approx([0.0, 0.0], abs=1e-12) and st["n"] == 16


def test_n2_stats_defaults_are_the_judgement_bootstrap(monkeypatch):
    seen = []
    real = R.boot_index
    monkeypatch.setattr(R, "boot_index", lambda n, draws, seed: seen.append((n, draws, seed)) or real(n, 10, seed))
    R.n2_stats(_d(), SPEC)
    assert seen == [(16, SPEC.boot_draws, SPEC.boot_seed)]


def _probe(A=10.0, P=20.0, kc=0.05, hz=50.0):
    return dict(seed=0, A=A, P=P, kc_frac=kc, kc_spikes=5, kc_max_win_hz=hz, apl_out_per_step=0.1, wall_s=0.01,
                steps=1400)


def _arm(cond, seed, k=0.0, plastic=True, sha=None, **pre_kw):
    return dict(cond=cond, seed=seed, plastic=plastic, csc_sha256=sha or cond.split("_")[1], weights_frac=1.0,
                pre={"x": _probe(**pre_kw), "y": _probe(**pre_kw)}, post={"x": _probe(A=10.0 - k), "y": _probe()})


def test_delta_is_dv_post_minus_pre_and_deltas_pair_by_seed():
    assert R.delta(_arm("sim_on", 1, k=2.5), ZU) == pytest.approx(-2.5)
    by = {"sim_on": [_arm("sim_on", 2), _arm("sim_on", 1, k=1.0)], "sim_off": [_arm("sim_off", 1), _arm("sim_off", 2)]}
    got = R.deltas(by, ["sim_on", "sim_off"], ZU)
    assert got["sim_on"].tolist() == [-1.0, 0.0]                     # seed order
    by["sim_off"][1]["seed"] = 3
    with pytest.raises(ValueError, match="seeds"):
        R.deltas(by, ["sim_on", "sim_off"], ZU)


def test_deltas_refuses_empty_missing_and_repeated_seeds():
    with pytest.raises(ValueError, match="no condition"):
        R.deltas({}, [], ZU)
    with pytest.raises(KeyError):
        R.deltas({"sim_on": [_arm("sim_on", 1)]}, ["sim_on", "sim_off"], ZU)
    with pytest.raises(ValueError, match="no rows"):
        R.deltas({"sim_on": [], "sim_off": []}, ["sim_on", "sim_off"], ZU)
    with pytest.raises(ValueError, match="repeats"):
        R.deltas({"sim_on": [_arm("sim_on", 1), _arm("sim_on", 1)]}, ["sim_on"], ZU)


def test_plumbing_flags_changed_probes_and_moved_weights():
    ok = _arm("sim_on", 1, plastic=False)
    moved = dict(_arm("sim_on", 2, plastic=False), weights_frac=0.99)
    changed = _arm("dis_off", 3, k=1.0, plastic=False)
    assert R.plumbing([ok]) == []
    bad = R.plumbing([ok, moved, changed])
    assert len(bad) == 2 and "sim_on seed 2" in bad[0] and "dis_off seed 3" in bad[1]


def test_plumbing_without_plasticity_off_reruns_fails():
    assert len(R.plumbing([])) == 1 and "no plasticity-off rerun" in R.plumbing([])[0]
    plastic = R.plumbing([_arm("sim_on", 1, plastic=True)])
    assert len(plastic) == 1 and "sim_on seed 1" in plastic[0] and "plasticity on" in plastic[0]


def test_block_validity_reads_the_off_conditions_pre_probes():
    good = {"sim_off": [_arm("sim_off", s) for s in range(8)], "dis_off": [_arm("dis_off", s) for s in range(8)]}
    assert R.block_validity(good, SPEC) == []
    hot = {"sim_off": [_arm("sim_off", s, kc=0.4) for s in range(8)], "dis_off": good["dis_off"]}
    assert any("KC median" in b for b in R.block_validity(hot, SPEC))
    floor = {"sim_off": good["sim_off"], "dis_off": [_arm("dis_off", s, P=0.0) for s in range(3)]
             + [_arm("dis_off", s) for s in range(3, 8)]}
    assert any("dDL" in b and "P" in b for b in R.block_validity(floor, SPEC))
    run = {"sim_off": [_arm("sim_off", s, hz=151.0) for s in range(2)] + [_arm("sim_off", s) for s in range(2, 8)],
           "dis_off": good["dis_off"]}
    assert any("sub-window" in b for b in R.block_validity(run, SPEC))


def test_block_validity_without_block_probes_is_invalid():
    good = [_arm("dis_off", s) for s in range(8)]
    assert any("no block" in b for b in R.block_validity({"sim_off": [], "dis_off": good}, SPEC))
    with pytest.raises(KeyError):
        R.block_validity({"dis_off": good}, SPEC)


def test_csc_checks():
    by = {c: [_arm(c, 1)] for c in R.N2_ORDER}
    assert R.csc_checks(by) == []
    by["sim_off"] = [_arm("sim_off", 1, sha="on")]
    assert any("sim" in b and "changed no weight" in b for b in R.csc_checks(by))
    by["sim_off"] = [_arm("sim_off", 1), _arm("sim_off", 2, sha="x")]
    assert any("2 CSC" in b for b in R.csc_checks(by))


def test_csc_checks_need_every_judged_arm():
    assert len(R.csc_checks({})) == 4 and all("missing" in b for b in R.csc_checks({}))
    by = {c: [_arm(c, 1)] for c in R.N2_ORDER}
    del by["dis_off"]
    assert any("dis_off" in b and "missing" in b for b in R.csc_checks(by))
    by["dis_off"] = []
    assert any("dis_off" in b and "0 CSC" in b for b in R.csc_checks(by))


def test_dprime_record_and_state_conditional_are_records():
    rec = R.dprime_record(_d())
    assert rec["L"]["sim_on"] == pytest.approx(-float(np.mean(_d()["sim_on"]) / np.std(_d()["sim_on"], ddof=1)))
    by = {"sim_on": [_arm("sim_on", 1, k=1.0), _arm("sim_on", 2, k=3.0, P=4.0)]}
    sc = R.state_conditional(by, ZU, SPEC)
    assert sc["sim_on"] == dict(n=1, n_all=2, ell=pytest.approx(1.0))


def test_thresholds_and_an_uncalibratable_pilot():
    th = R.thresholds({"sim": -4.0, "dis": -2.0}, {"sim": np.array([-1.0, -1.2]), "dis": np.array([-0.8, -0.8])},
                      SPEC)
    assert th["c1"] == {"sim": 1.0, "dis": 0.5} and th["delta_min"] == pytest.approx(0.55)
    assert th["eps"] == pytest.approx(0.2) and th["alt_D_sim"] == pytest.approx(0.825) and th["calibratable"]
    bad = R.thresholds({"sim": -4.0, "dis": -2.0}, {"sim": np.array([0.1, 0.2]), "dis": np.array([-1.0, -1.0])}, SPEC)
    assert not bad["calibratable"]
    nan = R.thresholds({"sim": -4.0, "dis": -2.0}, {"sim": np.array([np.nan, -1.0]), "dis": np.array([-1.0, -1.0])},
                       SPEC)
    assert not nan["calibratable"]
    flat = R.thresholds({"sim": 0.0, "dis": -2.0}, {"sim": np.array([-1.0, -1.2]), "dis": np.array([-0.8, -0.8])}, SPEC)
    assert not flat["calibratable"]                                   # c1 = 0 would make clauses ① ④ empty


def test_thresholds_read_the_fractions_from_the_spec():
    sp = replace(SPEC, c1_frac=0.5, delta_min_frac=0.25, eps_frac=0.125, alt_frac=1.0)
    th = R.thresholds({"sim": -4.0, "dis": -2.0}, {"sim": np.array([-1.0, -3.0]), "dis": np.array([-0.8, -0.8])}, sp)
    assert th["c1"] == {"sim": 2.0, "dis": 1.0} and th["delta_min"] == pytest.approx(0.5)
    assert th["eps"] == pytest.approx(0.1) and th["alt_D_sim"] == pytest.approx(2.0)


def test_budget_counts_every_arm_and_the_plumbing_reruns():
    assert R.steps_per_arm(SPEC, 1.0) == 4 * 1400 + 12 * 1800 == 27200
    assert R.budget_hours(16, 1e-3, SPEC) == pytest.approx((16 * 6 + 2 * 6) * 27200 * 1e-3 / 16 / 3600)


def test_design_view_is_the_whitelist():
    block = dict(outcome="OPERATING_POINT", run_id="r", selected={"g": 1.0, "c_delta": 1.0}, wall_s_per_step=1e-3,
                 state_shares={}, validity={}, state_flags=[], grid={"secret": 1}, apl_shift={}, drives={})
    assert set(R.design_view(block)) == set(R.DESIGN_KEYS) == {"outcome", "run_id", "selected", "wall_s_per_step",
                                                               "state_shares", "validity", "state_flags"}


def _oc(n, spec=SPEC, **over):
    sn = {scenario_key(p, m): n for p in spec.oc_pairings for m in spec.sd_mults}
    sn.update(over)
    return {"n": n, "scenario_n": sn}


def test_n2_0_outcome_order():
    th = dict(calibratable=True, ell_hat={"sim": 1.0, "dis": 1.0})
    assert R.n2_0_outcome(dict(th, calibratable=False), None, 1.0, SPEC)["outcome"] == R.STOP_POWER
    assert R.n2_0_outcome(th, _oc(None), 100.0, SPEC)["outcome"] == R.STOP_POWER       # power before budget
    assert R.n2_0_outcome(th, None, 1.0, SPEC)["outcome"] == R.STOP_POWER
    assert R.n2_0_outcome(th, _oc(16), 48.5, SPEC)["outcome"] == R.STOP_BUDGET
    assert R.n2_0_outcome(th, _oc(16), 48.0, SPEC) == {"outcome": R.N2_0_GO, "reason": None}


def test_n2_0_never_goes_on_an_incomplete_oc_or_an_undefined_budget():
    th = dict(calibratable=True, ell_hat={"sim": 1.0, "dis": 1.0})
    assert R.n2_0_outcome(th, _oc(16), float("nan"), SPEC)["outcome"] == R.STOP_BUDGET
    with pytest.raises(ValueError, match="scenario"):                 # a bare n, no per-scenario table
        R.n2_0_outcome(th, {"n": 16}, 1.0, SPEC)
    one = _oc(16); del one["scenario_n"][scenario_key("same", 2.0)]
    with pytest.raises(ValueError, match="scenario"):                 # one of the four scenarios never simulated
        R.n2_0_outcome(th, one, 1.0, SPEC)
    with pytest.raises(ValueError, match="scenario"):                 # n below a scenario's own minimal n
        R.n2_0_outcome(th, _oc(16, **{scenario_key("independent", 2.0): 24}), 1.0, SPEC)
    with pytest.raises(ValueError, match="scenario"):
        R.n2_0_outcome(th, _oc(16, **{scenario_key("independent", 2.0): None}), 1.0, SPEC)
    with pytest.raises(ValueError, match="grid"):
        R.n2_0_outcome(th, _oc(20), 1.0, SPEC)


def test_sentences_name_the_outcome():
    assert "n = 16" in R.sentence("n2_0", dict(outcome=R.N2_0_GO, n=16, budget_h=3.0))
    assert "STOP_POWER" in R.sentence("n2_0", dict(outcome=R.STOP_POWER, reason="x"))
    v = R.verdict(_st(D_sim_ci95=[0.1, 0.7]), C1, 0.5, 0.25)
    assert "['two']" in R.sentence("n2", dict(outcome=v["verdict"], verdict=v))
    assert "SUPPORTED" in R.sentence("n2", dict(outcome=R.SUPPORTED, verdict={}))
    inv = R.verdict(_st(), C1, 0.5, 0.25, ["a", "b"])
    assert R.sentence("n2", dict(outcome=R.INVALID, verdict=inv)).endswith("a; b")
    with pytest.raises(ValueError, match="unknown"):
        R.sentence("n9", {})
    with pytest.raises(ValueError, match="unknown"):                  # an unknown outcome is never worded as a pass
        R.sentence("n2", dict(outcome=None, verdict={}))


# ---------------------------------------------------------------- the OC (N.8.6, rulings G1 / G3)
def test_off_arm_shifts_the_mean_and_scales_deviations():
    v = np.array([-1.0, -2.0, -3.0])
    got = off_arm(v, [0, 2], -2.0, 0.5, 2.0)
    assert got.tolist() == [-2.0 + 0.5 + 2.0 * 1.0, -2.0 + 0.5 + 2.0 * -1.0]


def test_the_two_pairings_build_different_block_arms():
    sim = np.array([-1.0, -2.0, -3.0, -6.0]); dis = sim / 2
    mu = {"sim": float(sim.mean()), "dis": float(dis.mean())}
    on, off = np.array([0, 1, 3]), np.array([2, 2, 0])
    same = arms(sim, dis, mu, on, off, 0.5, 1.0, "same")
    ind = arms(sim, dis, mu, on, off, 0.5, 1.0, "independent")
    assert same["sim_on"].tolist() == ind["sim_on"].tolist() == sim[on].tolist()
    assert same["dis_on"].tolist() == dis[on].tolist()                       # the same seed indices for both pairs
    assert (same["sim_off"] - same["sim_on"]).tolist() == [0.5] * 3          # the on arm's own resample, shifted
    assert ind["sim_off"].tolist() == (sim[off] + 0.5).tolist() != same["sim_off"].tolist()
    assert ind["dis_off"].tolist() == dis[off].tolist() and same["dis_off"].tolist() == dis[on].tolist()
    wide = arms(sim, dis, mu, on, off, 0.0, 2.0, "same")
    assert (wide["sim_off"] - wide["sim_on"]).tolist() == (sim[on] - mu["sim"]).tolist()   # deviations x2
    with pytest.raises(ValueError, match="pairing"):
        arms(sim, dis, mu, on, off, 0.0, 1.0, "paired")


SMALL = replace(SPEC, n_grid=(4, 8), oc_draws=40, oc_boot=200)


def test_every_declared_pairing_is_implemented_and_both_are_declared():
    import flymon.brain.n_oc as O
    assert set(SPEC.oc_pairings) == set(O._BLOCK_IDX) and len(SPEC.oc_pairings) == 2


def test_a_clear_pilot_picks_the_smallest_n():
    clear = {"sim": -1.0 + 0.05 * np.linspace(-1, 1, 16), "dis": -1.0 + 0.05 * np.linspace(-1, 1, 16)}
    oc = simulate(clear, C1, 0.5, 0.25, SMALL)
    assert oc["n"] == 4 and len(oc["rows"]) == 2 * 2 * 2 * 2                     # pairing x spread x hypothesis x n
    assert {(r["pairing"], r["sd_mult"]) for r in oc["rows"]} == {(p, m) for p in SPEC.oc_pairings
                                                                  for m in SPEC.sd_mults}
    assert all(r["p_supported"] == 0.0 for r in oc["rows"] if r["hyp"] == "null")
    assert all(r["p_supported"] == 1.0 for r in oc["rows"] if r["hyp"] == "alt")
    assert oc["scenario_n"] == {scenario_key(p, m): 4 for p in SPEC.oc_pairings for m in SPEC.sd_mults}
    assert oc == simulate(clear, C1, 0.5, 0.25, SMALL)                           # deterministic


def test_the_oc_records_its_bootstrap_resolution_as_an_approximation():
    clear = {"sim": -1.0 + 0.05 * np.linspace(-1, 1, 16), "dis": -1.0 + 0.05 * np.linspace(-1, 1, 16)}
    oc = simulate(clear, C1, 0.5, 0.25, SMALL)
    assert (oc["boot"], oc["judge_boot"], oc["draws"]) == (SMALL.oc_boot, SMALL.boot_draws, SMALL.oc_draws)
    assert oc["boot"] != oc["judge_boot"] and str(SMALL.oc_boot) in oc["boot_note"]
    assert oc["pairings"] == list(SPEC.oc_pairings) and oc["alt_D_sim"] == pytest.approx(SPEC.alt_frac * 1.0)


def test_the_oc_uses_oc_boot_draws_per_experiment(monkeypatch):
    import flymon.brain.n_oc as O
    seen = set()
    real = O.n2_stats
    monkeypatch.setattr(O, "n2_stats", lambda d, spec, draws=None, seed=None: seen.add(draws) or real(d, spec, draws,
                                                                                                    seed))
    sp = replace(SMALL, n_grid=(4,), oc_draws=3, oc_boot=37)
    simulate({"sim": -1.0 + E, "dis": -1.0 + E}, C1, 0.5, 0.25, sp)
    assert seen == {37}


def test_the_two_pairings_give_different_operating_characteristics():
    pilot = {"sim": -1.0 + 0.6 * np.linspace(-1, 1, 16), "dis": -1.0 + 0.6 * np.linspace(-1, 1, 16)}
    oc = simulate(pilot, C1, 0.5, 0.25, replace(SMALL, n_grid=(8,)))
    p = {(r["pairing"], r["sd_mult"], r["hyp"]): r["p_supported"] for r in oc["rows"]}
    assert p[("same", 1.0, "alt")] > p[("independent", 1.0, "alt")]              # a constant shift has no spread
    assert p[("same", 2.0, "alt")] != p[("independent", 2.0, "alt")]
    only = simulate(pilot, C1, 0.5, 0.25, replace(SMALL, n_grid=(8,)), pairings=("independent",))
    assert {r["pairing"] for r in only["rows"]} == {"independent"} and only["pairings"] == ["independent"]
    with pytest.raises(ValueError, match="pairing"):
        simulate(pilot, C1, 0.5, 0.25, SMALL, pairings=("paired",))
    with pytest.raises(ValueError, match="pairing"):
        simulate(pilot, C1, 0.5, 0.25, SMALL, pairings=())


def test_a_noisy_pilot_has_no_n():
    noisy = {"sim": -0.1 + 3.0 * np.linspace(-1, 1, 16), "dis": -0.1 + 3.0 * np.linspace(-1, 1, 16)}
    oc = simulate(noisy, C1, 0.05, 0.025, SMALL)
    assert oc["n"] is None and None in oc["scenario_n"].values()


def test_simulate_refuses_a_degenerate_pilot():
    with pytest.raises(ValueError, match="pilot"):
        simulate({"sim": np.array([-1.0]), "dis": np.array([-1.0])}, C1, 0.5, 0.25, SMALL)
    with pytest.raises(ValueError, match="pilot"):
        simulate({"sim": -1.0 + E, "dis": -1.0 + E[:8]}, C1, 0.5, 0.25, SMALL)
    with pytest.raises(ValueError, match="pilot"):
        simulate({"sim": np.full(16, np.nan), "dis": -1.0 + E}, C1, 0.5, 0.25, SMALL)


def test_choose_n_needs_both_spreads_and_both_hypotheses():
    rows = [dict(sd_mult=m, hyp=h, n=n, p_supported=p) for n in (4, 8) for m in (1.0, 2.0)
            for h, p in (("null", 0.0), ("alt", 0.9 if (n, m) != (4, 2.0) else 0.7))]
    assert choose_n(rows, SMALL) == 8


def _rows(min_n: dict, grid=(4, 8, 16)):
    """An OC table in which scenario (pairing, mult) first has power at min_n[...] (None: never)."""
    out = []
    for (p, m), first in min_n.items():
        for n in grid:
            ok = first is not None and n >= first
            out += [dict(pairing=p, sd_mult=m, hyp="null", n=n, p_supported=0.05),
                    dict(pairing=p, sd_mult=m, hyp="alt", n=n, p_supported=0.8 if ok else 0.79)]
    return out


WIDE = replace(SPEC, n_grid=(4, 8, 16))


def test_n_is_the_largest_of_the_four_scenarios_minimal_n():
    mins = {("same", 1.0): 4, ("same", 2.0): 8, ("independent", 1.0): 4, ("independent", 2.0): 16}
    rows = _rows(mins)
    per = scenario_n(rows, WIDE)
    assert per == {scenario_key(p, m): n for (p, m), n in mins.items()}
    assert choose_n(rows, WIDE) == max(per.values()) == 16
    for worst in mins:                                                 # whichever scenario is the binding one
        shifted = {**{k: 4 for k in mins}, worst: 8}
        assert choose_n(_rows(shifted), WIDE) == 8
    never = _rows({**mins, ("same", 2.0): None})
    assert choose_n(never, WIDE) is None and scenario_n(never, WIDE)[scenario_key("same", 2.0)] is None


def test_a_null_rate_above_the_bound_fails_like_low_power():
    rows = _rows({("same", 1.0): 4, ("independent", 1.0): 4}, grid=(4, 8))
    for r in rows:
        if (r["pairing"], r["hyp"], r["n"]) == ("independent", "null", 4):
            r["p_supported"] = 0.0501
    assert choose_n(rows, replace(SPEC, n_grid=(4, 8), sd_mults=(1.0,))) == 8


def test_choose_n_is_the_smallest_n_passing_every_scenario_at_that_n():
    rows = _rows({("same", 1.0): 4, ("same", 2.0): 8, ("independent", 1.0): 4, ("independent", 2.0): 8})
    for r in rows:                                                     # a scenario that passes at 4 but not at 8
        if (r["pairing"], r["sd_mult"], r["hyp"], r["n"]) == ("same", 1.0, "alt", 8):
            r["p_supported"] = 0.5
    assert max(scenario_n(rows, WIDE).values()) == 8 and choose_n(rows, WIDE) == 16


def test_choose_n_never_passes_on_a_missing_cell():
    assert choose_n([], WIDE) is None
    full = _rows({("same", 1.0): 4, ("same", 2.0): 4, ("independent", 1.0): 4, ("independent", 2.0): 4})
    no_alt = [r for r in full if not (r["hyp"] == "alt" and r["pairing"] == "same" and r["sd_mult"] == 2.0)]
    with pytest.raises(ValueError, match="incomplete"):
        choose_n(no_alt, WIDE)
    no_spread = [r for r in full if r["sd_mult"] != 2.0]
    with pytest.raises(ValueError, match="incomplete"):
        choose_n(no_spread, WIDE)
    no_pairing = [r for r in full if r["pairing"] != "same"]
    assert choose_n(no_pairing, WIDE) == 4                             # rows alone: the pairings they hold
    with pytest.raises(ValueError, match="incomplete"):                # asked for both: a missing reading is refused
        choose_n(no_pairing, WIDE, pairings=SPEC.oc_pairings)
    nan = [dict(r, p_supported=float("nan")) for r in full]
    assert choose_n(nan, WIDE) is None
