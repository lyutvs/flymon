"""The M0d H.3 decision rules (spec H.3a.4-H.3a.6), one synthetic test per rule, plus reproductions of the values the
committed diagnostics recorded for the adopted operating point (tests/brain/fixtures/h3_recorded.json, spec H.3a.7)."""
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from flymon.brain import h3_rules as R
from flymon.brain.h3_spec import SPEC

FX = json.loads((Path(__file__).parent / "fixtures" / "h3_recorded.json").read_text())


# ---- bisections ---------------------------------------------------------------------------------------------------
def _replay(points, key):
    """An evaluator that must be queried at exactly the recorded x values, in order, and returns the recorded values."""
    queue = list(points)

    def ev(x):
        want, val = queue.pop(0)
        assert x == want, f"queried {x!r}, the recorded run queried {want!r}"
        return {key: val}
    return ev, queue


def test_stage1_bisection_reproduces_the_recorded_scale_bit_for_bit():
    rec = FX["stage1_kc1_6"]
    ev, left = _replay(rec["points"], "median_mv")
    b = R.bisect_log(ev, *SPEC.scale_bracket, SPEC.scale_bisect_steps, SPEC.membrane_target_mv, SPEC.membrane_tol_mv)
    assert not left
    assert b["accepted"]["x"] == rec["accepted_x"] == 0.11863011807986042
    assert b["accepted"]["median_mv"] == rec["accepted_mv"]
    assert b["final_bracket"] == rec["final_bracket"]
    assert b["bracketed"] and b["converged"]


def test_stage1_accepts_the_closest_evaluated_point_not_the_interval_midpoint():
    # value(x) = 10 log10(x) + 20: the target 11 sits at x = 10**-0.9 ~ 0.126, never evaluated exactly
    b = R.bisect_log(lambda x: {"median_mv": 10 * math.log10(x) + 20}, 0.005, 1.0, 3, 11.0, 1.0)
    evaluated = [e["x"] for e in b["endpoints"]] + [t["x"] for t in b["trace"]]
    assert b["accepted"]["x"] in evaluated
    assert b["accepted"]["x"] == min(evaluated, key=lambda x: abs(10 * math.log10(x) + 9))


def test_stage1_convergence_is_the_tolerance_on_the_accepted_point():
    ok = R.bisect_log(lambda x: {"median_mv": 12.0 if x > 0.1 else 9.95}, 0.005, 1.0, 2, 11.0, 1.0)
    assert ok["converged"] and ok["error"] == 1.0 and ok["accepted"]["x"] == 1.0     # |error| == tol converges
    far = R.bisect_log(lambda x: {"median_mv": 13.0 if x > 0.1 else 8.9}, 0.005, 1.0, 4, 11.0, 1.0)
    assert not far["converged"]


def test_stage1_records_an_unbracketed_interval():
    b = R.bisect_log(lambda x: {"median_mv": 20.0 + x}, 0.005, 1.0, 2, 11.0, 1.0)
    assert not b["bracketed"] and not b["converged"]
    assert b["accepted"]["x"] == 0.005


def test_stage2_bisection_reproduces_the_recorded_hold_and_never_stops_early():
    rec = FX["hold_G1_6"]
    ev, left = _replay(rec["points"], "mean_hz")
    b = R.bisect_mid(ev, *SPEC.hold_bracket, SPEC.hold_bisect_steps, SPEC.baseline_target_hz, key="mean_hz")
    assert not left and len(b["trace"]) == SPEC.hold_bisect_steps
    assert b["accepted"] == rec["accepted"] == 0.8408203125


def test_stage2_accepts_the_final_midpoint_even_after_an_exact_hit():
    b = R.bisect_mid(lambda h: {"mean_hz": 3.5 if h == 0.75 else (2.0 if h < 0.75 else 5.0)}, 0.5, 1.0, 3, 3.5,
                     key="mean_hz")
    assert b["trace"][0]["x"] == 0.75 and len(b["trace"]) == 3
    assert b["accepted"] == 0.5 * sum(b["final_bracket"]) != 0.75


# ---- reference statistics -----------------------------------------------------------------------------------------
def test_reference_stats_reproduce_the_recorded_point():
    rec = FX["reference_kc1_6_hold085"]
    st = R.reference_stats(rec["rows"], SPEC.release_quasi_linear)
    for k in ("median_mv", "median_kc_pct", "q1_mv", "q3_mv", "median_release_frac", "rel_share_in"):
        assert st[k] == rec[k], k


def test_reference_stats_leave_release_out_for_the_spiking_engine():
    rows = [dict(apl_v_mean=1.0, kc_active_frac=0.05, release_frac=None)] * 4
    assert "median_release_frac" not in R.reference_stats(rows, SPEC.release_quasi_linear)


def test_membrane_check_is_inclusive_at_the_tolerance():
    assert R.membrane_check({"median_mv": 12.0}, 11.0, 1.0)["ok"]
    assert not R.membrane_check({"median_mv": 12.0001}, 11.0, 1.0)["ok"]
    assert not R.membrane_check({"median_mv": 9.9}, 11.0, 1.0)["ok"]


# ---- D.4 --------------------------------------------------------------------------------------------------------
def test_d4_stats_reproduce_the_recorded_eight_seed_margins():
    d = R.d4_stats(FX["d4"]["rows"][:8], SPEC.d4_band)
    for k, v in FX["d4"]["by8"].items():
        assert d[k] == v, k
    assert d["seeds"] == list(range(100, 108))


def test_overlap_bootstrap_reproduces_the_recorded_cis():
    for n, key in ((8, "ci8"), (16, "ci16")):
        d = R.d4_stats(FX["d4"]["rows"][:n], SPEC.d4_band)
        ci = R.boot_mean_ci(d["overlap_margin_per_seed"], SPEC.boot_draws, SPEC.boot_seed)
        assert ci["ci"] == FX["d4"][key], n
    c0 = R.d4_stats(FX["d4"]["c0_rows"][:8], SPEC.d4_band)
    assert R.boot_mean_ci(c0["overlap_margin_per_seed"], SPEC.boot_draws, SPEC.boot_seed)["ci"] == FX["d4"]["c0_ci8"]


def test_sparsity_margin_is_the_distance_to_the_nearer_band_edge():
    row = dict(seed=1, frac_active_A=0.069, frac_active_B=0.031, jaccard=0.0, chance=0.0, kc_hz_A=0, kc_hz_B=0,
               mbon_hz_A=0, mbon_hz_B=0)
    d = R.d4_stats([row], SPEC.d4_band)
    assert d["margin_pp"] == pytest.approx(0.1)
    d = R.d4_stats([dict(row, frac_active_A=0.0701)], SPEC.d4_band)
    assert d["margin_pp"] < 0 and not d["band_ok"]


@pytest.mark.parametrize("ci,clause", [([1e-6, 0.01], "pass"), ([-0.01, -1e-6], "fail"), ([0.0, 0.01], "holds_zero"),
                                       ([-0.01, 0.0], "holds_zero"), ([-0.001, 0.001], "holds_zero")])
def test_overlap_clause(ci, clause):
    assert R.overlap_clause(ci) == clause


# ---- guard ------------------------------------------------------------------------------------------------------
def _guard_inputs():
    g = FX["guard"]
    return g["stim"], {r["seed"]: r for r in g["rest"]}, sorted({p["odor"] for p in g["stim"]})


def test_guard_type_stats_reproduce_the_recorded_point_estimates():
    stim, rest, _ = _guard_inputs()
    for n, want in FX["guard"]["expected"].items():
        st = R.mbon_type_stats(stim, rest, n, SPEC.guard_med_delta_min, SPEC.guard_zero_share_max)
        assert {k: st[k] for k in want} == want, n


def test_guard_single_passing_type_is_reproduced_on_both_halves():
    stim, rest, names = _guard_inputs()
    a = R.guard_pool(stim, rest, FX["guard"]["pools"]["A"], names, SPEC)
    assert a["passing"] == FX["guard"]["pass_A"] == ["MBON13"]
    assert [h["zero_share"] for h in a["halves"]] == [0.1875, 0.1875] and a["verdict"] == "pass"


def _pool_rows(zeros_first: int, zeros_second: int, n_odors: int = 48):
    """One type, 2 presentations per odour, stimulated count 10 or 0, rest 0."""
    stim, rest = [], []
    for j in range(n_odors):
        for s in (0, 1):
            seed = 1000 + 2 * j + s
            z = (zeros_first if j < n_odors // 2 else zeros_second)
            count = 0 if (2 * (j % (n_odors // 2)) + s) < z else 10
            stim.append(dict(odor=f"R{j:02d}", seed=seed, types={"M1": count, "M2": 0}))
            rest.append(dict(seed=seed, types={"M1": 0, "M2": 0}))
    return stim, {r["seed"]: r for r in rest}, [f"R{j:02d}" for j in range(n_odors)]


def test_guard_halves_that_split_are_indeterminate():
    stim, rest, names = _pool_rows(zeros_first=4, zeros_second=20)       # overall 24/96 = 0.25 -> passes
    g = R.guard_pool(stim, rest, ["M1", "M2"], names, SPEC)
    assert g["passing"] == ["M1"]
    assert [h["zero_share"] for h in g["halves"]] == [4 / 48, 20 / 48]
    assert g["verdict"] == "indeterminate"


def test_guard_needs_a_passing_type_and_skips_halves_when_two_pass():
    stim, rest, names = _pool_rows(0, 0)
    for p in stim:
        p["types"]["M2"] = 7
    g = R.guard_pool(stim, rest, ["M1", "M2"], names, SPEC)
    assert g["verdict"] == "pass" and g["halves"] is None
    stim, rest, names = _pool_rows(30, 30)                                # 60/96 zero
    assert R.guard_pool(stim, rest, ["M1", "M2"], names, SPEC)["verdict"] == "fail"


def test_guard_point_rule_is_inclusive_at_both_limits():
    stim, rest, names = _pool_rows(12, 12)                                # 24/96 = 0.25 exactly
    for p in stim:
        p["types"]["M1"] = 5 if p["types"]["M1"] else 0                   # median delta 5 exactly
    st = R.mbon_type_stats(stim, rest, "M1", SPEC.guard_med_delta_min, SPEC.guard_zero_share_max)
    assert st["zero_share"] == 0.25 and st["median_delta"] == 5.0 and st["passes"]


# ---- stage 4 ----------------------------------------------------------------------------------------------------
def test_baseline_ci_reproduces_the_recorded_gate_seeds_and_passes_by_overlap():
    st = R.baseline_stats(FX["baseline_gate32"]["per_seed"])
    assert st["mean_hz"] == FX["baseline_gate32"]["mean"] and st["se_hz"] == FX["baseline_gate32"]["se"]
    v = R.baseline_ci_verdict(st, SPEC.baseline_ci_z, SPEC.baseline_band_hz)
    assert v["ok"] and round(v["ci"][0], 2) == 2.44 and round(v["ci"][1], 2) == 3.23


@pytest.mark.parametrize("mean,se,ok", [(2.7, 0.2, True), (2.6, 0.2, False), (4.3, 0.2, True), (4.4, 0.2, False),
                                        (3.5, 0.0, True)])
def test_baseline_ci_overlap(mean, se, ok):
    assert R.baseline_ci_verdict(dict(mean_hz=mean, se_hz=se), 1.96, (3.0, 4.0))["ok"] is ok


def test_runaway_needs_the_declared_sample_sizes_and_zero_kcs():
    assert R.runaway_verdict([0] * 8, [0] * 64, 8, 64)["ok"]
    assert not R.runaway_verdict([0] * 7, [0] * 64, 8, 64)["ok"]
    assert not R.runaway_verdict([0] * 8, [0] * 63 + [1], 8, 64)["ok"]
    assert not R.runaway_verdict([], [], 0, 0)["ok"]


# ---- ranking and the qualification verdict ------------------------------------------------------------------------
def test_rank_score_is_the_normalised_minimum():
    assert R.rank_score(1.11, 0.00299, 0.00349) == pytest.approx(0.00299 / 0.00349)
    assert R.rank_score(0.44, 0.0045, 0.00349) == 0.44


@pytest.mark.parametrize("args,status,reasons", [
    ((True, "pass", [("A", "pass"), ("P", "pass")], True), None, []),
    ((True, "pass", [("A", "pass"), ("P", "pass")], None), None, []),
    ((False, "pass", [("A", "pass"), ("P", "pass")], True), R.QUAL_FAILED, ["sparsity"]),
    ((True, "indeterminate", [("A", "pass"), ("P", "pass")], True), R.INDETERMINATE, ["overlap"]),
    ((True, "pass", [("A", "indeterminate"), ("P", "pass")], True), R.INDETERMINATE, ["guard_A"]),
    ((True, "indeterminate", [("A", "pass"), ("P", "fail")], True), R.QUAL_FAILED, ["guard_P", "overlap"]),
    ((True, "pass", [("A", "pass"), ("P", "pass")], False), R.QUAL_FAILED, ["membrane"]),
])
def test_qualification_verdict(args, status, reasons):
    assert R.qualification_verdict(*args) == (status, reasons)
