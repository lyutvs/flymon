"""Spec J.12.9: docs/superpowers/specs/j-diag/stage2_oc.py's exact operating characteristic of stage 2's reading."""
import copy
import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("stage2_oc", ROOT / "docs/superpowers/specs/j-diag/stage2_oc.py")
oc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oc)

SUMMARY = ROOT / "results/summary/m0d.json"
QS = (0.33, 0.5, 0.6, 0.7)


@pytest.fixture(scope="module")
def rec():
    return json.loads(SUMMARY.read_text())["h4"]["h4"]["combos"]["C3"]["oracle"]


@pytest.fixture(scope="module")
def st(rec):
    return oc.structure(rec)


def test_the_fixed_pair_list_is_c3_s(st):
    t = st["turns"]
    assert [t[k]["b"] for k in sorted(t)] == [3, 2, 3, 3, 3, 3, 2, 2]
    assert {k: v["a_naive"] for k, v in t.items() if v["a_naive"]} == {0: 1, 4: 3}
    assert (st["aggregate"]["testable_b"], st["aggregate"]["naive_a"], st["aggregate"]["F_a"]) == (7, 4, 2)


@pytest.mark.parametrize("q", QS)
def test_the_calibrated_mean_is_q_on_both_axes(st, q):
    ub, ua = oc.pair_offsets(st["turns"], oc.turn_effects(st["turns"]))
    for off in (ub, ua):
        _, probs = oc.calibrate(off, q)
        assert abs(sum(probs) / len(probs) - q) < 1e-9


@pytest.mark.parametrize("model", ["turn_effects", "independent"])
@pytest.mark.parametrize("q", QS)
def test_exact_distributions_and_bands_sum_to_one(st, q, model):
    u = oc.turn_effects(st["turns"]) if model == "turn_effects" else None
    ub, ua = oc.pair_offsets(st["turns"], u)
    r = oc.oc_row(ub, ua, q, q)
    assert math.isclose(sum(r["testable_b_dist"]), 1.0, abs_tol=1e-12)
    assert math.isclose(sum(r["f_a_dist"]), 1.0, abs_tol=1e-12)
    assert math.isclose(r["p_selected"] + r["p_b_tb"] + r["p_b_no_conclusion"] + r["p_b_fa"], 1.0, abs_tol=1e-12)
    assert math.isclose(r["p_selected"] + r["p_b"], 1.0, abs_tol=1e-12)


@pytest.mark.parametrize("q", QS)
def test_the_independent_model_is_the_binomial(st, q):
    ub, ua = oc.pair_offsets(st["turns"], None)
    r = oc.oc_row(ub, ua, q, q)
    binom = lambda n, k, p: math.comb(n, k) * p ** k * (1 - p) ** (n - k)
    assert math.isclose(r["p_testable_b_ge_select"], sum(binom(21, k, q) for k in range(11, 22)), abs_tol=1e-12)
    assert math.isclose(r["p_testable_b_le_close"], sum(binom(21, k, q) for k in range(0, 8)), abs_tol=1e-12)
    assert math.isclose(r["p_f_a_ge_min"], sum(binom(4, k, q) for k in range(2, 5)), abs_tol=1e-12)
    # SELECTED = Tb >= 11 and Fa >= 2, independent axes
    assert math.isclose(r["p_selected"], r["p_testable_b_ge_select"] * r["p_f_a_ge_min"], abs_tol=1e-12)


def test_refuses_when_the_recorded_aggregate_differs(rec):
    bad = copy.deepcopy(rec)
    bad["aggregate"]["testable_b"] = 8
    with pytest.raises(SystemExit, match="testable_b"):
        oc.structure(bad)
    bad = copy.deepcopy(rec)
    flip = next(p for p in bad["pairs"] if p["axis"] == "b" and not p["testable"])
    flip["testable"] = True                         # the pairs no longer give the recorded aggregate
    with pytest.raises(SystemExit, match="refusing"):
        oc.structure(bad)


def test_refuses_a_pair_list_other_than_the_declared_one(rec):
    bad = copy.deepcopy(rec)
    drop = next(p for p in bad["pairs"] if p["axis"] == "b" and not p["testable"])
    bad["pairs"].remove(drop)
    bad["aggregate"]["n_b"] = 20
    with pytest.raises(SystemExit, match="stage2_n_b"):
        oc.structure(bad)


def test_refuses_an_untracked_summary():
    p = ROOT / "results" / "j" / "diag" / "_untracked_summary_for_test.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{}")
    try:
        with pytest.raises(SystemExit, match="not tracked"):
            oc.check_tracked(p)
    finally:
        p.unlink()
