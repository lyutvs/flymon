"""The H.4 decision rules on synthetic inputs (spec H.4 as amended by H.3a.1, H.3a.9 and H.4a)."""
import dataclasses

import numpy as np
import pytest

from flymon.brain import h4_rules as R
from flymon.brain.h4_spec import SPEC

SEEDS = list(SPEC.teach_seeds)


# ---- reactivity ----------------------------------------------------------------------------------------------------
def test_reactivity_is_the_h3_guard_statistic_with_h4s_thresholds():
    stim = [dict(odor=f"R{j:02d}", seed=s, types={"T": c}) for j, (s, c) in enumerate([(1, 7), (2, 0), (3, 6), (4, 6)])]
    rest = {s: dict(types={"T": 1}) for s in (1, 2, 3, 4)}
    got = R.reactivity(stim, rest, ["T"], SPEC)["T"]
    assert got["median_delta"] == 5.0 and got["zero_share"] == 0.25 and got["passes"]          # both edges inclusive
    strict = R.reactivity(stim, rest, ["T"], dataclasses.replace(SPEC, react_zero_share_max=0.24))["T"]
    assert not strict["passes"]


# ---- teachability -------------------------------------------------------------------------------------------------
def _rows(arm, order, pre_plus, pre_minus, post_plus, post_minus, t="T"):
    return [dict(seed=s, arm=arm, order=order, pre={"plus": {t: a}, "minus": {t: b}},
                 post={"plus": {t: c}, "minus": {t: d}})
            for s, a, b, c, d in zip(SEEDS, pre_plus, pre_minus, post_plus, post_minus)]


def test_teach_type_counts_strict_decreases_against_six_of_eight():
    pre, post = [10] * 8, [9] * 6 + [10, 11]
    r = R.teach_type(_rows("punish_only", "ab", pre, [0] * 8, post, [0] * 8), "T", "plus", SPEC)
    assert r["n_decreased"] == 6 and r["teachable"]
    r = R.teach_type(_rows("punish_only", "ab", pre, [0] * 8, [9] * 5 + [10] * 3, [0] * 8), "T", "plus", SPEC)
    assert r["n_decreased"] == 5 and not r["teachable"]
    r = R.teach_type(_rows("punish_only", "ab", [0] * 8, [0] * 8, [0] * 8, [0] * 8), "T", "plus", SPEC)
    assert r["n_decreased"] == 0 and not r["teachable"]                                     # silent: 0 -> 0 is no decrease
    with pytest.raises(ValueError):
        R.teach_type(_rows("punish_only", "ab", pre, pre, post, post)[:7], "T", "plus", SPEC)


@pytest.mark.parametrize("arm, naive_a, naive_b, odour, order, slot", [
    ("punish_only", 0, 25, "b", "ba", "plus"),       # MBON13 in C0: silent to a -> taught on b, CS+ = b
    ("punish_only", 15, 5, "a", "ab", "plus"),
    ("punish_only", 7, 7, "a", "ab", "plus"),        # tie -> M0c's assignment (punish on odour a)
    ("reward_only", 40, 3, "a", "ba", "minus"),      # reward on the CS- slot: odour a is CS- in order "ba"
    ("reward_only", 3, 40, "b", "ab", "minus"),
    ("reward_only", 9, 9, "b", "ab", "minus"),       # tie -> M0c's assignment (reward on odour b)
])
def test_teach_choice_judges_the_odour_the_type_answers(arm, naive_a, naive_b, odour, order, slot):
    rows = []
    for o in ("ab", "ba"):
        na, nb = (naive_a, naive_b) if o == "ab" else (naive_b, naive_a)      # "plus" is odour a in "ab", b in "ba"
        post_plus, post_minus = ([0] * 8, [nb] * 8) if slot == "plus" else ([na] * 8, [0] * 8)
        rows += _rows(arm, o, [na] * 8, [nb] * 8, post_plus, post_minus)
    got = R.teach_choice(rows, "T", arm, SPEC)
    assert (got["odour"], got["order"], got["slot"]) == (odour, order, slot)
    assert got["untaught_n_decreased"] == 0                                    # the other slot's counts never fall here
    assert got["naive_median"] == {"a": float(naive_a), "b": float(naive_b)}
    assert got["teachable"] == (max(naive_a, naive_b) > 0 if (naive_a != naive_b) else
                                (naive_a if slot == "plus" else naive_b) > 0)


def test_teach_choice_refuses_naive_rows_that_miss_the_ab_order_or_a_seed():
    rows = _rows("punish_only", "ab", [0] * 8, [25] * 8, [0] * 8, [20] * 8) + _rows("punish_only", "ba", [25] * 8,
                                                                                     [0] * 8, [20] * 8, [0] * 8)
    assert R.teach_choice(rows, "T", "punish_only", SPEC)["odour"] == "b"
    with pytest.raises(ValueError, match="naive"):                          # no "ab" rows: no silent NaN -> odour a
        R.teach_choice([r for r in rows if r["order"] == "ba"], "T", "punish_only", SPEC)
    with pytest.raises(ValueError, match="naive"):
        R.teach_choice([r for r in rows if not (r["order"] == "ab" and r["seed"] == SEEDS[0])], "T", "punish_only", SPEC)
    with pytest.raises(ValueError, match="naive"):
        R.teach_choice(rows, "T", "reward_only", SPEC)                     # no rows of this arm


# ---- readout ---------------------------------------------------------------------------------------------------------
POOLS = {"A": ["MA1", "MA2"], "P": ["MP1", "MP2"]}


def _flags(react, teach):
    return ({t: {"passes": t in react} for t in ("MA1", "MA2", "MP1", "MP2")},
            {t: {"teachable": t in teach} for t in ("MA1", "MA2", "MP1", "MP2")})


@pytest.mark.parametrize("react, teach, status, readout", [
    ({"MA1", "MP1"}, {"MA1", "MA2", "MP1", "MP2"}, R.READOUT_SELECTED, {"A": "MA1", "P": "MP1"}),
    ({"MA1", "MA2", "MP2"}, {"MA2", "MP2"}, R.READOUT_SELECTED, {"A": "MA2", "P": "MP2"}),
    ({"MA1", "MP1"}, {"MP1"}, R.DROPPED_NO_READOUT, None),                  # A empty: reactive but not teachable
    ({"MA1"}, {"MA1", "MP1"}, R.DROPPED_NO_READOUT, None),                  # P empty
    ({"MA1", "MA2", "MP1"}, {"MA1", "MA2", "MP1"}, R.STOP_MULTI_TYPE, None),
])
def test_pick_readout(react, teach, status, readout):
    got = R.pick_readout(*_flags(react, teach), POOLS)
    assert got["status"] == status and got["readout"] == readout


def test_z_constants_are_population_moments_of_the_type_count():
    rows = [dict(types={"MA1": a, "MP1": p}) for a, p in [(1, 10), (3, 10), (5, 40), (7, 40)]]
    z = R.z_constants(rows, {"A": "MA1", "P": "MP1"}, SPEC.z_ddof)
    assert z == {"A": (4.0, float(np.sqrt(5.0))), "P": (25.0, 15.0)}
    with pytest.raises(ValueError):
        R.z_constants([dict(types={"MA1": 2, "MP1": 1})] * 3, {"A": "MA1", "P": "MP1"}, 0)


# ---- oracle rows -----------------------------------------------------------------------------------------------------
Z = {"A": (0.0, 1.0), "P": (0.0, 1.0)}


def report(testable=True, naive=True, n=8):
    """pre: dV alternates +-1 (d_pre 0) or +5 +-1 (not naive); R1 lowers P(X) by 20 + (s % 2), R2 lowers A(X) the same."""
    step = [20 + (s % 2) for s in range(n)] if testable else [0] * n
    off = 0 if naive else 5
    pre = {"A": [[20 + off + (1 if s % 2 else -1), 20] for s in range(n)], "P": [[40, 40]] * n}
    r1 = {"A": pre["A"], "P": [[40 - d, 40] for d in step]}
    r2 = {"A": [[a - d, b] for (a, b), d in zip(pre["A"], step)], "P": r1["P"]}
    return {"pre": pre, "R1": r1, "R2": r2}


def rows_for(n_b_testable, n_a_testable_naive, n_a=4, n_b=5):
    rows = [dict(axis="a", turn=2 * (j % 4), x=f"ax{j}", y=f"ay{j}", report=report(j < n_a_testable_naive))
            for j in range(n_a)]
    rows += [dict(axis="b", turn=2 * (j % 4), x=f"bx{j}", y=f"by{j}", report=report(j < n_b_testable))
             for j in range(n_b)]
    return rows


def test_the_report_helper_hits_the_intended_statistics():
    from flymon.brain.h4_formula import pair_stats
    assert pair_stats(report(True), Z, 2.0)["testable"] and abs(pair_stats(report(True), Z, 2.0)["d_pre"]) < 0.5
    assert not pair_stats(report(False), Z, 2.0)["testable"]
    assert abs(pair_stats(report(True, naive=False), Z, 2.0)["d_pre"]) >= 0.5


def test_combo_stats_aggregates_clean_rows():
    rows = rows_for(3, 2)
    got = R.combo_stats(rows, Z, [(r["axis"], r["turn"], r["x"], r["y"]) for r in rows], SPEC)
    assert got["reasons"] == [] and got["aggregate"]["testable_b"] == 3 and got["aggregate"]["F_a"] == 2
    assert len(got["pairs"]) == len(rows)


def test_combo_stats_names_every_invalid_row_set():
    rows = rows_for(3, 2)
    exp = [(r["axis"], r["turn"], r["x"], r["y"]) for r in rows]
    assert "duplicate pair rows" in R.combo_stats(rows + rows[:1], Z, exp, SPEC)["reasons"]
    odd = [dict(rows[0], turn=3)] + rows[1:]
    got = R.combo_stats(odd, Z, exp, SPEC)["reasons"]
    assert "odd-turn rows present" in got and any("differ from the declared list" in r for r in got)
    short = [dict(r, report={k: {"A": v["A"][:1], "P": v["P"][:1]} for k, v in r["report"].items()}) for r in rows]
    got = R.combo_stats(short, Z, exp, SPEC)
    assert got["aggregate"] is None and got["reasons"] == [f"{len(rows)} pairs whose report probes are not the 8 report seeds"]
    seven = [dict(r, report={k: {"A": v["A"][:7], "P": v["P"][:7]} for k, v in r["report"].items()}) for r in rows]
    got = R.combo_stats(seven, Z, exp, SPEC)                                    # d' defined, still not the declared seeds
    assert got["aggregate"] is None and got["reasons"] == [f"{len(rows)} pairs whose report probes are not the 8 report seeds"]
    nanz = {"A": (0.0, float("nan")), "P": (0.0, 1.0)}
    assert any("NaN" in r for r in R.combo_stats(rows, nanz, exp, SPEC)["reasons"])


def test_combo_stats_reports_mismatched_a_and_p_probes_as_invalid_not_a_crash():
    rows = rows_for(3, 2)
    exp = [(r["axis"], r["turn"], r["x"], r["y"]) for r in rows]
    bad = dict(rows[0], report=dict(rows[0]["report"], R1={"A": rows[0]["report"]["R1"]["A"][:7],
                                                           "P": rows[0]["report"]["R1"]["P"]}))
    got = R.combo_stats([bad] + rows[1:], Z, exp, SPEC)                        # dv would raise on the 7 x 8 probe
    assert got["aggregate"] is None and got["reasons"] == ["1 pairs whose report probes are not the 8 report seeds"]


# ---- records ---------------------------------------------------------------------------------------------------------
def test_combo_records_floor_single_types_and_halves():
    rows = rows_for(3, 2)
    rec = R.combo_records(rows, Z, SPEC)
    a = [c for r in rows for pair in r["report"]["pre"]["A"] for c in pair]
    assert rec["naive_floor"]["A"] == dict(mean=float(np.mean(a)), zero_share=0.0, n=len(a))
    assert rec["naive_floor"]["P"] == dict(mean=40.0, zero_share=0.0, n=len(a))
    # report(): reward lowers P(X), punishment lowers A(X): alone, the P term sees only the reward (p = 0 -> m <= 0), the
    # A term only the punishment (r = 0 -> m <= 0), so neither type alone makes a pair testable
    assert rec["single_type"]["A"]["aggregate"]["testable_b"] == 0 and rec["single_type"]["P"]["aggregate"]["testable_b"] == 0
    pa = next(x for x in rec["single_type"]["A"]["pairs"] if x["x"] == "bx0")
    pp = next(x for x in rec["single_type"]["P"]["pairs"] if x["x"] == "bx0")
    assert pa["r"] == 0.0 and pa["p"] < -2 and pp["r"] > 2 and pp["p"] == 0.0
    assert [h["seeds"] for h in rec["report_halves"]] == [list(SPEC.report_seeds[:4]), list(SPEC.report_seeds[4:])]
    assert [h["testable_b"] for h in rec["report_halves"]] == [3, 3]


# ---- selection -------------------------------------------------------------------------------------------------------
def agg(testable_b, f_a, n_b=21):
    return dict(testable_b=testable_b, n_b=n_b, T_b=testable_b / n_b, F_a=f_a)


@pytest.mark.parametrize("aggs, outcome, winner, near", [
    ({"C0": agg(12, 2), "C1": agg(15, 3), "C3": agg(9, 5)}, R.SELECTED, "C1", ["C1"]),
    ({"C0": agg(13, 2), "C1": agg(15, 3), "C3": agg(14, 5)}, R.SELECTED, "C0", ["C0", "C1", "C3"]),   # within 2 pairs
    ({"C0": agg(12, 2), "C1": agg(15, 3), "C3": agg(13, 5)}, R.SELECTED, "C1", ["C1", "C3"]),         # 3 pairs: out
    ({"C0": agg(20, 1), "C1": agg(15, 3), "C3": agg(14, 2)}, R.SELECTED, "C1", ["C1", "C3"]),         # C0 ineligible
    ({"C0": None, "C1": agg(11, 2), "C3": agg(10, 2)}, R.SELECTED, "C1", ["C1", "C3"]),              # 11/21 >= 0.5
    ({"C0": agg(10, 2), "C1": agg(10, 3), "C3": None}, R.STOP_LOW_T_B, None, []),                   # 10/21 < 0.5
    ({"C0": agg(15, 1), "C1": agg(15, 0), "C3": None}, R.STOP_NO_ELIGIBLE, None, []),
    ({"C0": None, "C1": None, "C3": None}, R.STOP_NO_ELIGIBLE, None, []),
])
def test_select(aggs, outcome, winner, near):
    got = R.select(aggs, SPEC)
    assert (got["outcome"], got["winner"], got["near"]) == (outcome, winner, near)
    if winner:
        assert got["engine_unchanged"] == (winner == "C0")


def test_select_keeps_a_winner_whose_own_t_b_is_below_half_when_the_top_clears_it():
    """H.4 step 3 as written (H.4a.3-5, user decision): the stop looks at the best T_b; the tie band may pick an
    earlier combination below 0.5, which is flagged."""
    got = R.select({"C0": agg(9, 2), "C1": agg(11, 2), "C3": None}, SPEC)
    assert got["outcome"] == R.SELECTED and got["winner"] == "C0" and got["top_T_b"] == 11 / 21
    assert got["winner_below_bar"]
    assert not R.select({"C0": agg(11, 2), "C1": agg(12, 2), "C3": None}, SPEC)["winner_below_bar"]
