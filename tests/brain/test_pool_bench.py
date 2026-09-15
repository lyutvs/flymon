import pytest

from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.fly_pool import FlyPool, FlySpec
from flymon.brain.pool_bench import (DECISIONS, EVAL_DECISIONS, budget_hours, budget_hours_e2e, budget_table, exact_match, m0b_gate,
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
