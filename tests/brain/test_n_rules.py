# tests/brain/test_n_rules.py
"""Spec N.8.3 / N.8.4 / N.8.5 / N.8.8: the state threshold, N0f's per-cell numbers and inclusive clause bounds, the
operating-point choice (nearest 5.54%, then smaller g, then smaller c_δ; IA / EB alone never count), the state flags,
the similarity gate's bootstrap (an undefined r stops, never passes), N1's testable rule (sd = 0 is untestable)."""
import numpy as np
import pytest

from flymon.brain import n_rules as R
from flymon.brain.n_spec import SPEC


def _row(kc=0.05, hz=50.0, A=10, P=20, apl=0.1, wall=1.4, steps=1400):
    return dict(seed=0, A=A, P=P, kc_frac=kc, kc_spikes=1, kc_max_win_hz=hz, apl_out_per_step=apl, wall_s=wall,
                steps=steps)


def _cells(on=0.05, block=0.08, **kw):
    st = lambda kc: R.cell_stats([_row(kc=kc, **kw)] * 8, SPEC)
    return {s: {"on": st(on), "block": st(block), "all": st(block)} for s in SPEC.n0f_stimuli}


def test_state_threshold_is_p_below_5_silent():
    assert (R.state(4, SPEC), R.state(5, SPEC)) == ("silent", "firing")


def test_cell_stats():
    rows = [_row(hz=h, A=a, P=p) for h, a, p in [(151, 0, 4), (150, 3, 5), (10, 0, 30), (10, 2, 0)]]
    c = R.cell_stats(rows, SPEC)
    assert c["runaway_share"] == 0.25 and c["A_zero_share"] == 0.5 and c["P_zero_share"] == 0.25
    assert c["firing_share"] == 0.5 and c["max_win_hz_max"] == 151.0 and c["n"] == 4
    assert R.wall_per_step([_row(wall=1.4), _row(wall=2.8)]) == pytest.approx(0.0015)


@pytest.mark.parametrize("on, ok", [(0.03, True), (0.15, True), (0.0299, False), (0.1501, False)])
def test_band_bounds_are_inclusive(on, ok):
    assert R.point_checks(_cells(on=on), SPEC)["band"] is ok


@pytest.mark.parametrize("block, ok", [(0.30, True), (0.3001, False)])
def test_block_kc_bound(block, ok):
    assert R.point_checks(_cells(block=block), SPEC)["runaway"] is ok


def test_runaway_share_and_zero_share_bounds():
    def with_rows(rows_block):
        c = _cells()
        for s in SPEC.judged_stimuli:
            c[s]["block"] = R.cell_stats(rows_block, SPEC)
        return R.point_checks(c, SPEC)
    one = [_row(hz=151)] + [_row()] * 7; two = [_row(hz=151)] * 2 + [_row()] * 6
    assert with_rows(one)["runaway"] and not with_rows(two)["runaway"]
    zeros2 = [_row(A=0)] * 2 + [_row()] * 6; zeros3 = [_row(P=0)] * 3 + [_row()] * 5
    assert with_rows(zeros2)["floor"] and not with_rows(zeros3)["floor"]


@pytest.mark.parametrize("stim", SPEC.judged_stimuli)
def test_clauses_two_and_three_also_fail_on_the_on_condition(stim):
    def with_on(rows_on):
        c = _cells()
        c[stim]["on"] = R.cell_stats(rows_on, SPEC)
        return R.point_checks(c, SPEC)
    one = [_row(hz=151)] + [_row()] * 7; two = [_row(hz=151)] * 2 + [_row()] * 6
    assert with_on(one)["ok"]
    got = with_on(two)                                   # (2) on: sub-window > 150 Hz in 2/8 presentations
    assert (got["band"], got["runaway"], got["floor"], got["ok"]) == (True, False, True, False)
    hot = with_on([_row(kc=0.31)] * 8)                   # (2) on: KC median above 30% (it also leaves the band)
    assert not hot["runaway"] and not hot["ok"]
    for k in ("A", "P"):                                 # (3) on: a readout's zero share 3/8 > 25%
        got = with_on([_row(**{k: 0})] * 3 + [_row()] * 5)
        assert (got["band"], got["runaway"], got["floor"], got["ok"]) == (True, True, False, False)
        assert with_on([_row(**{k: 0})] * 2 + [_row()] * 6)["ok"]


def test_ia_and_eb_alone_never_count():
    c = _cells()
    c["IA"]["on"] = R.cell_stats([_row(kc=0.9, hz=900, A=0, P=0)] * 8, SPEC)
    c["EB"]["block"] = c["IA"]["on"]
    assert R.point_checks(c, SPEC)["ok"]


def test_select_nearest_then_smaller_g_then_smaller_c():
    got = R.select_point({(0.5, 1.0): _cells(on=0.05), (1.0, 1.0): _cells(on=0.06), (0.25, 1.0): _cells(on=0.20)},
                         SPEC)
    assert got["outcome"] == R.OPERATING_POINT and got["selected"] == {"g": 1.0, "c_delta": 1.0}   # |.06-.0554| < |.05-.0554|
    tie = R.select_point({(1.0, 2.0): _cells(), (2.0, 1.0): _cells(), (1.0, 1.0): _cells()}, SPEC)
    assert tie["selected"] == {"g": 1.0, "c_delta": 1.0}
    stop = R.select_point({(0.25, 1.0): _cells(on=0.2)}, SPEC)
    assert stop["outcome"] == R.STOP_NO_OPERATING_POINT and stop["selected"] is None
    assert set(stop["checks"]) == {"0.25|1"} and R.point_key(0.125, 8.0) == "0.125|8"


def test_state_flags_are_strictly_above_a_quarter():
    cells = {"4:1": {"on": {"firing_share": 1.0}, "block": {"firing_share": 0.75}, "all": {"firing_share": 0.7}}}
    assert R.state_flags(cells, SPEC) == [dict(stimulus="4:1", condition="all", on=1.0, share=0.7)]


def _fired(x, y_sim, y_dis, n=4):
    return {"4:1": [x + [90 + s] for s in range(n)], "1:4": [y_sim + [90 + s] for s in range(n)],
            "dDL": [y_dis + [95 - s] for s in range(n)]}


def test_similarity_gate_passes_and_stops():
    same, near, far = list(range(20)), list(range(2, 22)), list(range(60, 80))
    go = R.similarity(_fired(same, near, far), 100, SPEC, draws=300)
    assert go["outcome"] == R.SIMILARITY_GO and go["r_sim"] > go["r_dis"] and go["ci95"][0] > 0
    stop = R.similarity(_fired(same, far, near), 100, SPEC, draws=300)
    assert stop["outcome"] == R.STOP_SIMILARITY_ORDER and stop["delta_r"] < 0


def test_similarity_with_silent_kcs_stops_without_crashing():
    empty = {s: [[] for _ in range(4)] for s in ("4:1", "1:4", "dDL")}
    got = R.similarity(empty, 100, SPEC, draws=50)
    assert got["outcome"] == R.STOP_SIMILARITY_ORDER and np.isnan(got["r_sim"]) and np.isnan(got["ci95"][0])
    with pytest.raises(ValueError, match="same"):
        R.similarity({"4:1": [[1]] * 4, "1:4": [[1]] * 3, "dDL": [[1]] * 4}, 10, SPEC, draws=10)


def _oracle(d):
    n = len(d)
    pre = {"A": [[10.0, 10.0]] * n, "P": [[5.0, 5.0]] * n}
    post = {"A": [[10.0 + x, 10.0] for x in d], "P": [[5.0, 5.0]] * n}
    return {"alpha_punish": 0.5, "select": {"punish": {"0.5": {"change": -1.0}}}, "report": {"pre": pre, "P": post},
            "kc": {"jaccard": 0.2}}


ZU = {"A": (0.0, 1.0), "P": (0.0, 1.0)}          # dV = (A_x - P_x) - (A_y - P_y)


def test_n1_testable_needs_p0_at_most_minus_2_and_a_spread():
    ok = R.n1_pair(_oracle([-3.0, -4.0, -5.0]), ZU, SPEC)
    assert ok["testable"] and ok["p0"] == pytest.approx(-4.0) and ok["o"] == pytest.approx(-4.0)
    weak = R.n1_pair(_oracle([-1.0, -2.0, -3.0, -0.5]), ZU, SPEC)
    assert not weak["testable"] and weak["p0"] > -2
    flat = R.n1_pair(_oracle([-3.0, -3.0, -3.0]), ZU, SPEC)
    assert flat["p0"] == -np.inf and flat["sd"] == 0.0 and not flat["testable"]      # N.8.5: sd = 0 is untestable
    assert ok["floor"] == {"A_zero_share": 0.0, "P_zero_share": 0.0} and ok["jaccard"] == 0.2


def test_n1_outcome():
    t, f = {"testable": True}, {"testable": False}
    assert R.n1_outcome({"sim": t, "dis": t}) == {"outcome": R.N1_GO, "note": None}
    dis_only = R.n1_outcome({"sim": f, "dis": t})
    assert dis_only["outcome"] == R.STOP_UNTESTABLE and "비슷한 쌍" in dis_only["note"]
    assert R.n1_outcome({"sim": t, "dis": f}) == {"outcome": R.STOP_UNTESTABLE, "note": None}
    assert R.n1_outcome({"sim": f, "dis": f}) == {"outcome": R.STOP_UNTESTABLE, "note": None}
    for missing in ({}, {"sim": t}, {"dis": t}):         # a missing pair raises: all() over nothing is never N1_GO
        with pytest.raises(KeyError):
            R.n1_outcome(missing)
