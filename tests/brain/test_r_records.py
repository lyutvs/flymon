# tests/brain/test_r_records.py
"""R's records (R.4, R.6, R.9.5, R.9.7): pair rows from h4_formula.pair_stats; raw_check finds duplicates, missing
pairs, short probes, a wrong edit, broken per-cell sums and several CSC shas without any pair statistic; saturation is
per cell against 1000 / (refrac_steps × dt); z renormalisation is a record; transitions count C -> L per criterion;
gate ①'s record sits beside encoder ③'s values; pre-read validity flags L edges != 2 as INVALID and every other
defect as a reason, including a raw entry whose stored inputs are not its condition's, pair's and judgement seeds'
(R.9.7 completeness over condition × pair × seed)."""
import copy

import pytest

from flymon.brain import r_records as RR
from flymon.brain.r_spec import SPEC
from tests.brain.r_fixtures import Z, fake_measurer, fake_oracle, fake_rows, got_for, key_of, write_raws

L, C, E0 = SPEC.conditions()
JS = SPEC.judge_seeds()
ROWS = fake_rows()
KEYS = [key_of(r) for r in ROWS]
WANT = RR.judge_inputs(fake_measurer(SPEC), ROWS, SPEC)


def test_fixture_flags_fix_the_pair_statistics():
    from flymon.brain.h4_formula import pair_stats
    for r_ok in (False, True):
        for p_ok in (False, True):
            for bal in (False, True):
                st = pair_stats(fake_oracle(r_ok, p_ok, bal)["report"], Z, 2.0)
                row = RR.pair_row("b|64|x|y", fake_oracle(r_ok, p_ok, bal), st, SPEC)
                assert (row["reward_pass"], row["punish_pass"], row["naive"]) == (r_ok, p_ok, bal)
                assert row["testable"] == (r_ok and p_ok) and row["axis"] == "b"


def test_strip_job_removes_timing_and_r_keys():
    x = dict(seed=1, wall_s=2.0, pre={"x": {"A": 1, "wall_s": 3.0}}, r={"edit_edges": 0}, direction="r1", point=[1])
    assert RR.strip_job(x) == dict(seed=1, pre={"x": {"A": 1}})
    assert RR.strip_job(dict(q=1, r=2, report=3), top=("q", "r")) == dict(report=3)


def test_raw_check_clean_and_its_defects():
    got = got_for(ROWS, n_rep=8, n_act=8)
    rc = RR.raw_check(got, C, KEYS, JS)
    assert rc["reasons"] == [] and rc["edit_edges"] == [0] and rc["csc_sha256"] == "sha-C" and rc["n_pairs"] == 53
    assert "pairs" not in rc and "aggregate" not in rc                       # no pair statistic before the seal
    dup = got + [got[0]]
    assert any("duplicate" in m for m in RR.raw_check(dup, C, KEYS, JS)["reasons"])
    assert any("missing 1" in m for m in RR.raw_check(got[1:], C, KEYS, JS)["reasons"])
    short = copy.deepcopy(got)
    short[0]["result"]["report"]["pre"]["A"].pop()
    assert any("probes" in m for m in RR.raw_check(short, C, KEYS, JS)["reasons"])
    cells = copy.deepcopy(got)
    cells[0]["result"]["r"]["p_cells"]["x"][0][0] += 1
    assert any("per-cell" in m for m in RR.raw_check(cells, C, KEYS, JS)["reasons"])
    edited = copy.deepcopy(got)
    edited[0]["result"]["q"]["edit"] = SPEC.lever_edit
    assert any("ran edit" in m for m in RR.raw_check(edited, C, KEYS, JS)["reasons"])
    two = copy.deepcopy(got)
    two[0]["result"]["q"]["csc_sha256"] = "other"
    assert any("differs between rows" in m for m in RR.raw_check(two, C, KEYS, JS)["reasons"])


def test_cond_summary_counts_and_aggregate():
    plan = {k: (True, True, False) for k in KEYS[:12]}                          # 12 (b) testable
    plan.update({k: (True, True, True) for k in KEYS[21:24]})                   # 3 (a) testable and naive
    s = RR.cond_summary(got_for(ROWS, plan), C, SPEC, Z, KEYS, JS)
    a = s["aggregate"]
    assert (a["testable_b"], a["F_a"], a["naive_a"], a["n_b"], a["n_a"]) == (12, 3, 3, 21, 32)
    assert s["counts"]["b"] == dict(n=21, testable=12, reward_pass=12, punish_pass=21, naive=0)
    assert s["kc_median"] == 0.05 and s["d6a_over_share"] == 0.0 and s["apl_out_median"] == 0.2
    assert s["alpha_punish"] == {"0.8": 53} and s["reasons"] == []


def test_naive_x_medians_are_column_0_of_the_naive_report_probe():
    """R.6 "순진 P_X·A_X 중앙값": X's column only; Y's naive A (20) would pull a pooled median away from X's 40."""
    s = RR.cond_summary(got_for(ROWS), C, SPEC, Z, KEYS, JS)
    assert s["naive_A_X_median"] == 40.0 and s["naive_P_X_median"] == 27.0
    assert s["naive_A_median"] != s["naive_A_X_median"]
    cmp = RR.compare(s, s, s, SPEC)["conditions"]
    assert {n: (c["naive_A_X_median"], c["naive_P_X_median"]) for n, c in cmp.items()} == {
        n: (40.0, 27.0) for n in SPEC.cond_names}


def test_saturation_is_per_cell_against_the_single_cell_ceiling():
    got = got_for(ROWS[:1], n_rep=8)            # P 26 / 28 per type -> cells 13 / 13 and 14 / 14 per 0.6 s
    s = RR.saturation(got, SPEC)
    assert s["cap_hz"] == 500.0 and s["n"] == 2 * 2 * 8
    assert s["median"] == pytest.approx(13.5 / 0.6) and s["max"] == pytest.approx(14 / 0.6)
    assert s["share"] == {"0.5": 0.0, "0.8": 0.0}
    hot = copy.deepcopy(got)
    hot[0]["result"]["r"]["p_cells"]["x"] = [[300, 0]] * 8                # 500 Hz on one cell, 8 presentations
    s2 = RR.saturation(hot, SPEC)
    assert s2["max"] == pytest.approx(500.0) and s2["share"]["0.8"] == pytest.approx(8 / 32)


def test_z_renorm_is_a_record_beside_block_h4s_z():
    s = RR.z_renorm(got_for(ROWS, {k: (True, True, True) for k in KEYS}), SPEC, Z)
    assert set(s) >= {"z", "testable_b", "F_a", "naive_a", "abs_d_pre_median", "abs_d_pre_median_h4"}
    assert s["abs_d_pre_median_h4"] == pytest.approx(0.0, abs=1e-9)


def test_transitions_count_c_to_l():
    cp = [dict(key=k, axis="b", testable=False, reward_pass=False, punish_pass=True, p=-3.0) for k in "abc"]
    lp = [dict(cp[0], punish_pass=False, p=-1.0), dict(cp[1], reward_pass=True), dict(cp[2])]
    t = RR.transitions(lp, cp)["b"]
    assert t["punish_pass"] == {"pp": 2, "pf": 1, "fp": 0, "ff": 0}
    assert t["reward_pass"] == {"pp": 0, "pf": 0, "fp": 1, "ff": 2}
    assert t["minus_p"]["a"] == [3.0, 1.0]


def test_gate1_record_beside_encoder_3():
    act = {"s1": dict(frac=[0.04] * 8, max_win=[5] * 8, csc_sha256=["sha-L"], edit_edges=[2]),
           "d1": dict(frac=[0.06] * 8, max_win=[40] * 8, csc_sha256=["sha-L"], edit_edges=[2])}
    rec = RR.gate1_record(act, ["s1"], ["d1"], {"s1": 0.05, "d1": 0.05}, SPEC)
    assert rec["median"] == pytest.approx(0.05) and rec["n_odours"] == 2 and rec["n_seeds"] == [8]
    assert rec["decreased"] == 1 and rec["d6a_over"] == 8 and rec["d6a_presentations"] == 16
    assert rec["edit_edges"] == [2] and rec["csc_sha256"] == ["sha-L"] and "ok" in rec["original"]


def _blocks(code="K"):
    return {n: dict(code_key=code, seeds=JS) for n in SPEC.cond_names}


@pytest.fixture
def raws(tmp_path):
    def mk(l_edges=2, c_edit="none", c_edges=0, c_sha="sha-C"):
        return write_raws({"L": got_for(ROWS, sha="sha-L", edges=l_edges, edit=SPEC.lever_edit),
                           "C": got_for(ROWS, sha=c_sha, edges=c_edges, edit=c_edit),
                           "E0": got_for(ROWS, sha="sha-C", edges=0, edit="none")}, ROWS, WANT, tmp_path)
    return mk


def test_preread_validity_clean(raws):
    v = RR.preread_validity(_blocks(), raws(), SPEC, Z, KEYS, "K", "sha-C", WANT)
    assert v["reasons"] == [] and v["invalid"] == []


@pytest.mark.parametrize("edges", [1, 3])
def test_preread_validity_l_edges_other_than_2_is_invalid(raws, edges):
    v = RR.preread_validity(_blocks(), raws(l_edges=edges), SPEC, Z, KEYS, "K", "sha-C", WANT)
    assert v["invalid"] and "L changed" in v["invalid"][0]


def test_preread_validity_an_edit_in_c_is_a_reason(raws):
    v = RR.preread_validity(_blocks(), raws(c_edit=SPEC.lever_edit, c_edges=2, c_sha="sha-L"), SPEC, Z, KEYS, "K",
                            "sha-C", WANT)
    assert v["invalid"] == [] and any("C" in m for m in v["reasons"])


def test_preread_validity_code_key_and_sha_relations(raws):
    assert RR.preread_validity(_blocks(code="old"), raws(), SPEC, Z, KEYS, "K", "sha-C", WANT)["reasons"]
    assert RR.preread_validity(_blocks(), raws(), SPEC, Z, KEYS, "K", "sha-other", WANT)["reasons"]


def test_preread_validity_e0_pointed_at_c_files_is_a_reason(raws):
    """Same edit (none), same sha, same probe lengths: only the stored inputs (condition, odours, strength) differ."""
    r = raws()
    r["E0"] = [dict(g) for g in r["C"]]
    v = RR.preread_validity(_blocks(), r, SPEC, Z, KEYS, "K", "sha-C", WANT)
    assert v["invalid"] == [] and v["reasons"] and all(m.startswith("E0:") for m in v["reasons"])
    assert "stored inputs" in v["reasons"][0] and "53 raw" in v["reasons"][0]


def test_preread_validity_inputs_of_another_block_or_seeds_are_a_reason(raws):
    r = raws()
    other = RR.judge_inputs(fake_measurer(SPEC), ROWS, SPEC, block="even")
    assert RR.preread_validity(_blocks(), r, SPEC, Z, KEYS, "K", "sha-C", other)["reasons"]
    seeds = {n: {k: dict(v, report_seeds=v["report_seeds"][:-1]) for k, v in w.items()} for n, w in WANT.items()}
    assert RR.preread_validity(_blocks(), r, SPEC, Z, KEYS, "K", "sha-C", seeds)["reasons"]
    gone = {n: dict(w) for n, w in WANT.items()}
    gone["L"].pop(KEYS[0])
    v = RR.preread_validity(_blocks(), r, SPEC, Z, KEYS, "K", "sha-C", gone)
    assert [m[:3] for m in v["reasons"]] == ["L: "]
