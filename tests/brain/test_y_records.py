"""Y's stage0 records (Y.0 / Y.6.6 threshold reproduction, Y.3.4 comparison, fixture 6, OC timing): the 0.25-grid
floor threshold on a constructed θ, the integer rule, every candidate's predicate and pass counts, the comparison's
layout and best design per k range, fixture 6 on W's draw order, the timing scale."""
import dataclasses
import json

import numpy as np
import pytest

from flymon.brain import w_oc
from flymon.brain import w_verdict as WV
from flymon.brain import y_oc as O
from flymon.brain.h3_store import canonical
from flymon.brain.w_spec import SPEC as W
from flymon.brain.y_spec import SPEC as Y

Z = {"A": (16.917, 12.484), "P": (80.167, 29.775)}


@pytest.fixture(scope="module")
def theta():
    return w_oc.fit(w_oc.synthetic_pilot(np.random.default_rng(3), base=(40.0, 90.0), sd=4.0, corr=0.8,
                                         learn=(20.0, 10.0)))


def test_floor_threshold_definition(theta):
    a, b = 10.0, 3.0
    for cell, slot, ci in (("A", w_oc.SI["R2"], WV.A), ("P", w_oc.SI["R1"], WV.P)):
        L = O.floor_threshold(theta, a, b, cell, Y)
        off = w_oc.slot_means(theta, np.zeros((2, 2)), np.zeros((2, 2)), a, b)[slot, ci, WV.X]
        r = theta["resid"][:, slot, ci, WV.X]
        share = lambda x: float((np.rint(x + off + r) < 0).mean())  # noqa: E731
        assert L % Y.thr_step == 0 and share(L) < Y.floor_share
        assert L == 0.0 or share(L - Y.thr_step) >= Y.floor_share


def test_thresholds_integer_rule(theta, monkeypatch):
    vals = iter([19.25, 42.0])
    monkeypatch.setattr(O, "floor_threshold", lambda *a, **k: next(vals))
    t = O.thresholds(theta, 1.0, 1.0, Y)
    assert (t["c_A_calc"], t["c_P_calc"], t["c_A_int"], t["c_P_int"]) == (19.25, 42.0, 20, 43)
    assert t["matches_declared"] is True and t["matches_expected"] is True


@pytest.mark.parametrize("name,vals,want", [
    ("a", (0.4, 1.0, 1.0), True), ("b", (0.4, 20.0, 43.0), True), ("b", (0.4, 19.0, 43.0), False),
    ("c", (0.4, 8.875, 51.640625), True), ("d", (0.4, 30.0, 94.5), False), ("e", (0.4, 31.0, 1.0), True),
    ("f", (-0.9, 31.0, 1.0), True), ("f", (-1.0, 31.0, 1.0), False), ("f", (5.0, 30.0, 1.0), False)])
def test_candidate_passes(name, vals, want):
    assert O.candidate_passes(name, *vals, Y) is want


def test_candidate_predicates_on_bases(theta):
    b = np.zeros((2, 2, 2))
    b[0, WV.A, WV.X], b[0, WV.P, WV.X] = 31.0, 50.0
    b[1, WV.A, WV.X], b[1, WV.P, WV.X] = 19.0, 50.0
    assert O.candidate("b", theta, Z, Y)[1](b).tolist() == [True, False]
    assert O.candidate("e", theta, Z, Y)[1](b).tolist() == [True, False]
    spec_f, acc_f, _ = O.candidate("f", theta, Z, Y)
    assert spec_f.naive_max == float("inf") and acc_f(b).shape == (2,)


def test_compare_filters_layout(theta):
    ys = dataclasses.replace(Y, compare_reps=4)
    vals = {"p0": (0.1, 40.0, 90.0), "p1": (2.0, 10.0, 90.0), Y.pilot_w_pairs[0]: (0.0, 31.0, 107.0)}
    out = O.compare_filters(theta, vals, Z, ys)
    assert set(out["candidates"]) == set(O.CANDIDATES)
    c = out["candidates"]["b"]
    assert set(c["by_range"]) == {"[4, 8]", "[6, 10]"} and c["pass_counts"] == dict(pilot=2, balanced=1)
    assert set(c["fill"]) == {f"g{g}|{m}" for g in Y.cluster_grid for m in ("min", "max")}
    assert 0.0 <= c["acceptance"] <= 1.0 and out["candidates"]["a"]["acceptance"] <= 1.0
    s = O.compare_summary(out)
    assert set(s["candidates"]["b"]["by_range"]["[4, 8]"]) >= {"best", "best_g0", "fill_flag"}
    json.loads(canonical(s))


def test_w_failed_recal_reads_ws_draw_order(theta):
    oc = json.loads(canonical(w_oc.run(w_oc.synthetic_pilot(np.random.default_rng(5), base=(40.0, 90.0), sd=4.0,
                                                            corr=0.8, learn=(20.0, 10.0)), Z, W,
                                       lambda K, F: K * F, n_boot=3, n_rep=2, n_boot_rep=2)))
    cal = oc["boot_calibration"]
    cal[1] = dict(cal[1], min=dict(cal[1]["min"], ok=False))           # mark draw 1 failed
    out = O.w_failed_recal(theta, cal, Z, W, Y)
    assert out["failed"] == [i for i, c in enumerate(cal) if not (c["min"]["ok"] and c["max"]["ok"])]
    assert 1 in out["failed"] and out["n_failed"] == len(out["rows"])
    assert set(out["counts"]) == {"min", "max"}


def test_oc_timing_scales(theta):
    t = O.oc_timing(theta, Z, dataclasses.replace(Y, boot_reps=4), 10.0, 3.0)
    assert t["evaluate_s"] > 0 and t["precheck_point_s"] > t["evaluate_s"] and t["phase_b_boot_s"] > 0
