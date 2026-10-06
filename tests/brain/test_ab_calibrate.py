"""AB.5 calibration (AB.7 0 보정 함수 시험): structure-only input, bit reproducibility and resume, grid monotonicity,
the truth of the winsorized estimator, CP bounds, selection / verification on separate streams, the cells, bench and
the synthetic validation. Small scale (every rule number unchanged)."""
import inspect

import numpy as np
import pytest
from scipy import stats

from flymon.brain import aa_estimate as AE
from flymon.brain import ab_estimate as E
from flymon.brain.ab_spec import SPEC, small

S = small(SPEC, boot_b=200, boot_chunk=100, n_sel=30, n_ver=40, truth_pairs=4_000, truth_chunk=1_000, cal_chunk=7,
          bench_reps=2, synth_reps=2)
ST = E.rep_structure((3, 2, 2, 1, 1), S)                               # 9 pairs, 5 X groups, 9 type sets
TR = dict(D=1.0, Dfin=1.0, R=1.0)


def test_structure_only():
    assert list(inspect.signature(E.calibrate).parameters) == ["st", "tag", "s", "cell"]
    assert list(inspect.signature(E.calibrate_many).parameters) == ["items", "s", "cell"]
    with pytest.raises(TypeError):
        E.calibrate_many([(ST, "g5"), ([(0, 0)], "g6")], S)
    for bad in ([(0, 0)], ((0.0, 1),), ((0, 1, 2),), {"a": 1}, np.array([[0, 0]]), (), ((0, -1),), ((True, 0),)):
        with pytest.raises(TypeError):
            E.calibrate(bad, "g5", S)
    assert E.check_structure(ST) is ST


def test_rep_structures_and_cells():
    assert dict(SPEC.rep_structures) == {"g5": (5, 4, 3, 3, 2), "g6": (5, 4, 3, 3, 2, 1), "g7": (7, 6, 4, 3, 2, 2, 1)}
    st = E.rep_structure((5, 4, 3, 3, 2), SPEC)
    assert len(st) == 17 and [t for _, t in st[:10]] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 0] and st[5] == (1, 5)
    assert len(SPEC.cells) == 24 and SPEC.cell(1) == ("none", 0.0) and SPEC.cell(24) == ("floor", -1.0)
    assert SPEC.cell(18) == ("skX", 1.0) and SPEC.cell(19) == ("skX", -1.0) and SPEC.cell(22) == ("fly", 1.0)
    kinds = [c for c in SPEC.cells]
    for k in ("skX", "skT", "floor"):                                      # mirror-closed (해석 21)
        assert (k, 1.0) in kinds and (k, -1.0) in kinds
    with pytest.raises(ValueError):
        SPEC.cell(0)


def test_cp_bounds_match_scipy_and_spec_counts():
    assert E.cp_upper(5, 100, S) == pytest.approx(stats.beta.ppf(0.975, 6, 95))
    assert E.cp_upper(100, 100, S) == 1.0
    for n, target, xmax in ((10_000, 0.025, 219), (10_000, 0.00625, 47), (5_000, 0.025, 103), (5_000, 0.00625, 20)):
        assert E.cp_upper(xmax, n, S) <= target < E.cp_upper(xmax + 1, n, S)


def test_cal_run_reproducible_resumable_and_monotone():
    a = E.cal_run(ST, "g5", "sel", 4, TR, S)
    b = E.cal_run(ST, "g5", "sel", 4, TR, S)
    assert a["counts"] == b["counts"] and a["done"] == S.n_sel
    seen = []
    E.cal_run(ST, "g5", "sel", 4, TR, S, on_chunk=lambda p: seen.append(p))
    first = seen[0]
    assert first["done"] == S.cal_chunk
    c = E.cal_run(ST, "g5", "sel", 4, TR, S, resume=first)
    assert c["counts"] == a["counts"] and c["state"] == a["state"]
    for n in E.STATS:
        for side in (("P", "F") if n == "D" else ("P",)):
            v = a["counts"][n][side]
            assert all(x >= y for x, y in zip(v, v[1:])), (n, side, v)          # smaller α → fewer misses
            assert all(j <= min(t, g) for j, t, g in zip(v, a["counts"][n][side + "_TS"], a["counts"][n][side + "_CG"]))


def test_joint_miss_is_the_and_of_both_methods():
    r = AE.stream(S.cal_seed, "cal", "t", "sel", 21)
    tru = dict(D=0.9, Dfin=0.9, R=0.8)                                   # low truths: single-method misses happen
    seen_split = False
    for _ in range(40):
        row = E.rep_misses(ST, S.cell(21), r, tru, S)
        for n in E.STATS:
            ts, cg = list(row[n]["P_TS"]), list(row[n]["P_CG"])
            assert list(row[n]["P"]) == [a and b for a, b in zip(ts, cg)]
            seen_split |= any(a != b for a, b in zip(ts, cg))
        assert list(row["D"]["F"]) == [a and b for a, b in zip(row["D"]["F_TS"], row["D"]["F_CG"])]
    assert seen_split


def test_cal_run_uses_boot_b_and_its_own_stream():
    a = E.cal_run(ST, "g5", "sel", 2, TR, S, n=3)
    assert a["counts"] == E.cal_run(ST, "g5", "sel", 2, TR, S, n=3, B=S.boot_b)["counts"]
    assert a["state"] != E.cal_run(ST, "g5", "sel", 2, TR, S, n=3, B=S.boot_b // 2)["state"]
    assert a["state"] != E.cal_run(ST, "g5", "ver", 2, TR, S, n=3)["state"]
    assert a["state"] != E.cal_run(ST, "g6", "sel", 2, TR, S, n=3)["state"]


def test_truth_is_the_winsorized_expectation():
    t1 = E.truth(ST, "g5", 1, S)
    assert abs(t1["D"] - E.delta(S) * AE.c_k(8)) < 0.05 and abs(t1["R"] - E.delta(S)) < 0.05
    t23 = E.truth(ST, "g5", 23, S)
    assert t23["D"] > t23["Dfin"] + 1.0                                     # two +∞ flies → +10 in D only
    assert abs(t23["D"] - (2 * 10 + 6 * E.delta(S) * AE.c_k(8)) / 8) < 0.05  # the winsorized mean, not δ × c(7)
    assert abs(t23["R"] - (2 * 0.05 + 6 * E.delta(S)) / 8) < 0.05
    assert E.truth(ST, "g5", 23, S) == t23 and E.truth(ST, "g6", 23, S) != t23


def test_select_largest_alpha_and_none():
    n = 5_000
    good = {ci: {"D": {"P": [200, 100, 50, 20, 5, 2, 1, 0], "F": [30, 20, 10, 5, 2, 1, 0, 0, 0]},
                 "Dfin": {"P": [200, 103, 0, 0, 0, 0, 0, 0]}, "R": {"P": [104] * 8}} for ci in (1, 2)}
    ch = E.select(good, n, S)
    assert ch["D"]["P"] == 0.0125 and ch["Dfin"]["P"] == 0.0125 and ch["R"]["P"] is None
    assert ch["D"]["F"] == 0.0025                                            # 20 ≤ 20, 30 > 20


def test_verification_counts_only_the_chosen_level():
    only = {"D": {"P": S.p_grid[3]}, "Dfin": {}, "R": {"P": S.p_grid[0]}}
    v = E.cal_run(ST, "g5", "ver", 3, TR, S, n=5, only=only)
    d = v["counts"]["D"]
    assert all(x == 0 for i, x in enumerate(d["P"]) if i != 3) and all(x == 0 for x in d["F"])
    assert all(x == 0 for x in v["counts"]["Dfin"]["P"])


def test_calibrate_phases_and_verdict_of_the_gate():
    calls = []
    s2 = small(S, p_target=0.2, f_target=0.2)                          # test scale: some α is chosen at n_sel 30

    def cell(kind, args):
        calls.append((kind, args[1], args[2] if kind != "truth" else "truth"))
        return E._serial_cell(kind, args)
    out = E.calibrate(ST, "g5", s2, cell)
    kinds = [c[0] for c in calls]
    assert kinds == ["truth"] * 24 + ["sel"] * 24 + ["ver"] * 24
    assert all(c[2] == c[0] for c in calls if c[0] != "truth")           # verification runs on its own stream
    assert set(out["alpha"]) == {"D", "Dfin", "R"} and set(out["alpha"]["D"]) == {"P", "F"}
    calls.clear()
    two = E.calibrate_many([(ST, "g5"), (E.rep_structure((2, 2, 2, 1, 1), s2), "g6")], s2, cell)
    assert [c[0] for c in calls][:48] == ["truth"] * 48 and [o["tag"] for o in two] == ["g5", "g6"]
    assert two[0]["alpha"] == out["alpha"] and two[0]["sel_counts"] == out["sel_counts"]   # per-item streams
    assert out["pass_ok"] == all(out["alpha"][n]["P"] is not None for n in E.STATS)


def test_bench_and_synth_rep():
    b = E.bench(S)
    assert set(b["per_rep_s"]) == {"17", "23"} and b["max_s"] > 0
    st = E.rep_structure(SPEC.rep_sizes("g7"), S)
    lab = E.synth_rep(st, S.cell(1), (3.0,) * 4, dict(D=0.001, Dfin=0.001, R=0.001, F=0.0001),
                      AE.stream(S.synth_seed, "synth", "eff3.5", 1), S)
    assert lab in (E.PASS, E.FAIL, E.UNDECIDED)
    cfg = dict(E.synth_configs(S))
    assert list(cfg) == ["bar_one", "bar_all", "eff1.5", "eff2.5", "eff3.5"]
    assert cfg["bar_one"] == (E.delta(S), 3.0, 3.0, 3.0) and cfg["eff2.5"] == (2.5,) * 4
