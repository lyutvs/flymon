import os

import pytest

from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.fly_pool import FlyPool, FlySpec
from flymon.brain.pool_bench import (DECISIONS, EVAL_DECISIONS, budget_hours, budget_hours_e2e, budget_table, exact_match,
                        m0b_gate, m0c_gate, match_sparsity_row, refuse_modified_engine_output, refuse_old_engine_output,
                        throughput_row)

A = {"ORN_DM1": 1.0, "ORN_DA1": 1.0}


def test_budget_formulas():
    # step-based: 16 workers, 1.9 / 2.6 ms: training 52,000 x (5,600 x 1.9 + 1,600 x 2.6) ms + eval 81,600 x 5,600 x 1.9 ms, / 16
    h = budget_hours(1.9, 2.6, 16)
    train = DECISIONS * (4 * 1400 * 1.9 + 1600 * 2.6)
    ev = EVAL_DECISIONS * 4 * 1400 * 1.9
    assert h == pytest.approx((train + ev) / 16 / 3.6e6)
    assert budget_hours(1.9, 2.6, 16, eval_decisions=0) < h
    assert budget_hours(1.9, 2.6, 8) == pytest.approx(2 * h)
    # end to end: a 16-fly batch taking 11 s to decide and 4 s to reinforce
    e = budget_hours_e2e(11.0, 4.0, 16)
    assert e == pytest.approx((DECISIONS * 15.0 + EVAL_DECISIONS * 11.0) / 16 / 3600)
    assert budget_hours_e2e(11.0, 4.0, 16, eval_decisions=0) < e


def _row(workers, s_dec, s_rein, ms_dec=1.9, ms_rein=2.6):
    return {"workers": workers, "n_flies_batch": workers, "ms_decision_median": ms_dec, "ms_decision_max": ms_dec,
            "ms_reinforce_median": ms_rein, "ms_reinforce_max": ms_rein,
            "s_decide_batch_median": s_dec, "s_decide_batch_max": s_dec, "s_reinforce_batch_median": s_rein, "s_reinforce_batch_max": s_rein}


def test_budget_table_picks_the_best_configuration_and_gates_on_it():
    rows = [_row(8, 8.0, 3.0), _row(16, 11.0, 4.0)]
    t = budget_table(rows)
    assert [b["workers"] for b in t["by_workers"]] == [8, 16]
    assert t["workers"] == 16 and t["gate_ok"] is True and 20 < t["baseline_hours"] < 40
    assert t["rows"][0]["hours"] == pytest.approx(budget_hours_e2e(11.0, 4.0, 16))
    assert t["rows"][1]["hours"] < t["rows"][0]["hours"]
    contended = budget_table([_row(8, 8.0, 3.0), _row(16, 30.0, 10.0)])       # 16 workers thrash: 8 is the better configuration
    assert contended["workers"] == 8
    slow = budget_table([_row(16, 60.0, 20.0)])
    assert slow["gate_ok"] is False


def test_throughput_row_on_a_synthetic_pool(synthetic_npz):
    p = Params(noise_mv=0.15, min_weight=1, balance_hemispheres=False, kc_thresh=0.5, learn_rate=0.05)
    pops = Populations.from_connectome(Connectome.load(synthetic_npz))
    with FlyPool(synthetic_npz, p, [FlySpec(), FlySpec()], workers=2, timeout_s=120) as pool:
        row = throughput_row(pool, A, 1.0, steps=5, warm=2, repeats=2, idx=pops.mbon, settle_decision_ms=5, read_ms=5,
                             settle_reinforce_ms=5, pulse_ms=5, gap_ms=5)
        assert row["workers"] == 2 and row["n_flies_batch"] == 2 and row["repeats"] == 2
        assert row["ms_decision_max"] >= row["ms_decision_median"] > 0
        assert row["s_decide_batch_max"] >= row["s_decide_batch_median"] > 0 and row["s_reinforce_batch_max"] > 0
        assert all(pool.weights_frac(f) == pytest.approx(1.0) for f in range(2))     # learning undone


def _res(dD, a=0):
    return {"D_pre": -0.9, "D_post": -0.9 + dD, "dD": dD, "D_pre_disc": -1.0, "D_post_disc": -1.0, "dD_disc": 0.0,
            "weights_frac": 0.98, "w_frac_a_core": 0.9, "w_frac_p_core": 0.99,
            "counts": {"pre_plus": {"A": a, "P": 33}, "pre_minus": {"A": 31, "P": 35}, "post_plus": {"A": 0, "P": 22}, "post_minus": {"A": 14, "P": 0}}}


def test_exact_match_and_gate():
    ref = {"0": {"both": _res(0.2), "reversed": _res(1.4)}, "1": {"both": _res(0.3), "reversed": _res(1.2)}}
    same = {0: {"both": _res(0.2), "reversed": _res(1.4)}, 1: {"both": _res(0.3), "reversed": _res(1.2)}}
    m = exact_match(same, ref)
    assert m["ok"] and m["n_results"] == 4 and m["n_equal"] == 4 and m["max_abs_diff"] == 0.0
    off = {0: {"both": _res(0.2 + 1e-9), "reversed": _res(1.4)}, 1: {"both": _res(0.3, a=1), "reversed": _res(1.2)}}
    m = exact_match(off, ref)
    assert not m["ok"] and m["n_equal"] == 2 and m["max_abs_diff"] == pytest.approx(1e-9)
    m = exact_match({0: {"both": _res(0.2)}}, ref)
    assert m["n_missing"] == 3 and not m["ok"]
    budget = {"gate_ok": True}
    assert m0b_gate(budget, {"ok": True}, {"ok": True}, True)["passed"] is True
    assert m0b_gate(budget, {"ok": False}, {"ok": True}, True)["passed"] is False
    assert m0b_gate({"gate_ok": False}, {"ok": True}, {"ok": True}, True)["passed"] is False


def _m0c_inputs():
    row = {"frac_active_A": 0.064, "frac_active_B": 0.049, "jaccard": 0.025, "chance": 0.028}
    baseline = {"mbon_hz_rest_trimmed": 3.2}
    runaway = {"rest_n_kc_over_sat_per_seed": [0] * 8, "odor_B_n_kc_over_sat_per_seed": [0] * 64}
    equivalence = {"old_conditioning": {"ok": True}, "old_sparsity": {"ok": True}, "arm_equal": {"ok": True}, "decide_equal": True}
    budget = {"gate_ok": True, "limit_hours": 60.0}
    cond = {"n_seeds": 8, "n_flip": 5, "channel_specific_seeds": 7,
            "arms": {"both": {"mean_dD": 0.0}, "reversed": {"mean_dD": 1.49}}}
    return row, baseline, runaway, equivalence, budget, cond


def test_m0c_gate_terms_and_composition():
    row, baseline, runaway, equivalence, budget, cond = _m0c_inputs()
    g = m0c_gate(row, baseline, runaway, equivalence, budget, cond)
    assert set(g) == {"sparsity_ok", "baseline_ok", "runaway_ok", "equivalence_ok", "throughput_ok",
                      "conditioning_index_flip_ok", "channel_specific_seeds", "passed"}
    assert g["passed"] is True and g["conditioning_index_flip_ok"] is False and g["channel_specific_seeds"] == 7
    # the conditioning criterion is recorded, never gated on
    cond8 = dict(cond, n_flip=8, arms={"both": {"mean_dD": -0.4}, "reversed": {"mean_dD": 1.0}})
    assert m0c_gate(row, baseline, runaway, equivalence, budget, cond8)["conditioning_index_flip_ok"] is True
    # every gate term fails the composite on its own
    assert not m0c_gate(dict(row, frac_active_B=0.02), baseline, runaway, equivalence, budget, cond)["passed"]
    assert not m0c_gate(dict(row, jaccard=0.03), baseline, runaway, equivalence, budget, cond)["passed"]
    assert not m0c_gate(row, {"mbon_hz_rest_trimmed": 2.94}, runaway, equivalence, budget, cond)["baseline_ok"]
    assert not m0c_gate(row, baseline, dict(runaway, odor_B_n_kc_over_sat_per_seed=[0] * 63 + [52]), equivalence, budget, cond)["runaway_ok"]
    assert not m0c_gate(row, baseline, dict(runaway, rest_n_kc_over_sat_per_seed=[0, 52, 0]), equivalence, budget, cond)["runaway_ok"]
    # the sample sizes are part of the pre-registered definition: a short run cannot pass
    assert not m0c_gate(row, baseline, dict(runaway, rest_n_kc_over_sat_per_seed=[0] * 3), equivalence, budget, cond)["runaway_ok"]
    assert not m0c_gate(row, baseline, dict(runaway, odor_B_n_kc_over_sat_per_seed=[0] * 32), equivalence, budget, cond)["runaway_ok"]
    assert not m0c_gate(row, baseline, runaway, dict(equivalence, arm_equal={"ok": False}), budget, cond)["equivalence_ok"]
    assert not m0c_gate(row, baseline, runaway, equivalence, {"gate_ok": False, "limit_hours": 60.0}, cond)["throughput_ok"]
    assert not m0c_gate(row, baseline, runaway, equivalence, {"gate_ok": True, "limit_hours": 80.0}, cond)["throughput_ok"]


def test_refuse_old_engine_output_guards_the_immutable_references(capsys):
    """Spec D.5: results/m0*, results/summary/m0.json and m0b.json are the old engine's (kc_kc_scale 1.0) bit-exact
    references and git-ignored; no script writes there with another engine."""
    for out in ("results/m0/sparsity.json", "results/m0/x.json", "./results/m0/x.json", "results/m0b/throughput.json",
                "results/summary/m0.json", "results/summary/m0b.json", os.path.abspath("results/summary/m0b.json")):
        with pytest.raises(SystemExit) as e:                     # spec D.5: exit code 2, message on stderr
            refuse_old_engine_output(out, 0.0)
        assert e.value.code == 2
        assert "old engine" in capsys.readouterr().err
        refuse_old_engine_output(out, 1.0)                       # the old engine may write its own files
    for out in ("results/m0c/sparsity.json", "results/summary/m0c.json", "/tmp/x.json", ""):
        refuse_old_engine_output(out, 0.0)                       # new paths, empty (unused) paths: fine


def test_refuse_modified_engine_output_guards_every_pre_m0d_reference(capsys):
    """Spec H.2: an engine with any M0d mode on never writes under the M0/M0b/M0c reference trees or summaries."""
    modified = (Params(apl_mode="graded"), Params(orn_std=True),
                Params(kc_thresh_mode="homeostatic", kc_thresh_file="x.npz", kc_thresh_sha256="0" * 64))
    for p in modified:
        for out in ("results/m0/x.json", "results/m0b/x.json", "results/m0c/sparsity.json", "./results/m0c/x.json",
                    "results/summary/m0.json", "results/summary/m0b.json", "results/summary/m0c.json",
                    "results/summary/compartments.json", os.path.abspath("results/summary/m0c.json")):
            with pytest.raises(SystemExit) as e:
                refuse_modified_engine_output(out, p)
            assert e.value.code == 2
            assert "M0d" in capsys.readouterr().err
        for out in ("results/m0d/sparsity.json", "results/summary/m0d.json", "/tmp/x.json", ""):
            refuse_modified_engine_output(out, p)
    for out in ("results/m0c/sparsity.json", "results/summary/m0c.json"):
        refuse_modified_engine_output(out, Params())         # the M0c engine may still write its own files


def _sp_row(**kw):
    r = {"kc_thresh": Params().kc_thresh, "apl_scale": Params().apl_scale, "mbon_hold_frac": Params().mbon_hold_frac,
         "kc_kc_scale": 0.0, "frac_active_A": 0.064, "frac_active_B": 0.049, "jaccard": 0.025, "chance": 0.028,
         "mbon_hz_A": 15.7, "mbon_hz_B": 18.1, "mbon_hz_rest_trimmed": 3.3}
    r.update(kw)
    return r


def _pool_sparsity():
    return {"frac_active_A": 0.064, "frac_active_B": 0.049, "jaccard": 0.025, "chance": 0.028,
            "mbon_hz_A": 15.7, "mbon_hz_B": 18.1, "per_seed": [{"frac_active_A": 0.064}]}


def test_match_sparsity_row_never_raises_on_a_run_that_skipped_a_measurement():
    """`--rest-seeds 0` (no trimmed baseline) or `--sparsity-seeds 0` (no sparsity rows) is a legitimate run shape;
    the comparison reports the shortfall and is never the thing that loses a multi-hour pool run's results."""
    p = Params()
    ref = {"grid": [_sp_row()]}
    full = match_sparsity_row(_pool_sparsity(), {"mbon_hz_rest_trimmed": 3.3}, ref, p)
    assert full["ok"] is True and full["max_abs_diff"] == 0.0 and set(full["diffs"]) >= {"jaccard", "mbon_hz_rest_trimmed"}
    no_rest = match_sparsity_row(_pool_sparsity(), {"mbon_hz_rest_trimmed": None, "per_seed": []}, ref, p)
    assert no_rest["ok"] is False and "mbon_hz_rest_trimmed" not in no_rest["diffs"] and "rest seeds" in no_rest["note"]
    assert no_rest["max_abs_diff"] == 0.0                                   # the terms it did measure still agree
    no_sp = match_sparsity_row({"per_seed": []}, {"mbon_hz_rest_trimmed": 3.3}, ref, p)
    assert no_sp == {"ok": False, "note": "no sparsity seeds in this run"}
    assert match_sparsity_row(_pool_sparsity(), {"mbon_hz_rest_trimmed": 3.3}, None, p)["note"] == "no reference file"
    # the M0 files predate the key: they are the 1.0 engine and never match the new engine's Params()
    old_only = {"grid": [{k: v for k, v in _sp_row().items() if k != "kc_kc_scale"}]}
    assert "no reference grid row" in match_sparsity_row(_pool_sparsity(), {"mbon_hz_rest_trimmed": 3.3}, old_only, p)["note"]
    assert match_sparsity_row(_pool_sparsity(), {"mbon_hz_rest_trimmed": 3.3}, old_only, Params(kc_kc_scale=1.0))["ok"] is True


def test_refuse_modified_engine_output_covers_apl_input_scale(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    refuse_modified_engine_output("results/m0c/sparsity.json", Params())          # default: allowed
    with pytest.raises(SystemExit) as e:
        refuse_modified_engine_output("results/m0c/sparsity.json", Params(apl_input_scale=0.5))
    assert e.value.code == 2
    assert "apl_input_scale=0.5" in capsys.readouterr().err
    refuse_modified_engine_output("results/m0d/diag/x.json", Params(apl_input_scale=0.5))   # m0d path: allowed
