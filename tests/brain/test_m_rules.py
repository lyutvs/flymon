"""Spec M.10.3-M.10.5, M.10.7: arm counts from pair_stats, ranking and top-k, predicted joint counts, overlap-free
combinations, bar-first choice, the judgement list with its pinned digests, the reading with the same-set C3 (n >= 11
with c >= n reads B_NO_CONCLUSION), and the closing sentences with their scope."""
import dataclasses

import pytest

from flymon.brain import m_rules as R
from flymon.brain.m_cands import STOP_NO_SPECIFICITY
from flymon.brain.m_spec import SPEC, smoke


def _st(r, p):
    return {"d_pre": 0.0, "r": r, "p": p, "m": min(r, -p), "testable": min(r, -p) >= 2.0}


def test_arm_counts_use_r_for_reward_and_minus_p_for_punish(monkeypatch):
    stats = [_st(3, -1), _st(1, -3), None, _st(2, -2)]
    monkeypatch.setattr(R, "pair_stats", lambda rep, z, tmin: rep)
    got = R.arm_rows(stats, {}, "reward", SPEC.j.h4)
    assert (got["n"], got["med"], got["defined"]) == (2, 2.0, 3) and got["r"] == [3, 1, None, 2]
    got = R.arm_rows(stats, {}, "punish", SPEC.j.h4)
    assert (got["n"], got["med"]) == (2, 2.0) and got["p"] == [-1, -3, None, -2]
    with pytest.raises(ValueError, match="arm"):
        R.arm_rows(stats, {}, "both", SPEC.j.h4)


def test_arm_counts_all_undefined(monkeypatch):
    monkeypatch.setattr(R, "pair_stats", lambda rep, z, tmin: None)
    got = R.arm_rows([{"x": 1}, None], {}, "reward", SPEC.j.h4)
    assert (got["n"], got["defined"], got["med"]) == (0, 0, float("-inf"))


def _e(name, n, med, w):
    return dict(name=name, n=n, med=med, kc_input=w)


def test_rank_and_top():
    ranked = R.rank([_e("A", 3, 1.0, 5), _e("B", 5, 0.5, 1), _e("C", 3, 2.0, 1), _e("D", 3, 2.0, 9)], "reward")
    assert [e["name"] for e in ranked] == ["B", "D", "C", "A"]
    assert [e["name"] for e in R.top(ranked, 2)] == ["B", "D"] and len(R.top(ranked[:1], 2)) == 1


def test_predicted_joint_counts_the_same_pair():
    got = R.predicted_joint({"PAM10": [2, 3, None, 1]}, {"PPL103": [-3, -1, -4, -2]}, 2.0)
    assert got == {"PAM10|PPL103": 1}


def test_combos_drop_overlaps():
    r = [dict(name="PAM03", cells=[1, 2]), dict(name="PAM10", cells=[5])]
    p = [dict(name="PPL107", cells=[2]), dict(name="PPL103", cells=[9])]
    ok, bad = R.combos(r, p)
    assert [(a["name"], b["name"]) for a, b in ok] == [("PAM03", "PPL103"), ("PAM10", "PPL107"), ("PAM10", "PPL103")]
    assert bad == [["PAM03", "PPL107"]]


def test_combos_with_one_arm_fixed_to_the_incumbent():
    r = [dict(name="PAM08", cells=[1])]                                     # reward arm empty: fixed to the incumbent
    p = [dict(name="PPL103", cells=[9]), dict(name="PPL107", cells=[4])]
    ok, bad = R.combos(r, p)
    assert [(a["name"], b["name"]) for a, b in ok] == [("PAM08", "PPL103"), ("PAM08", "PPL107")] and bad == []


def _agg(tb, fa, m=0.0, n_b=21, naive_a=4):
    return {"testable_b": tb, "F_a": fa, "n_b": n_b, "T_b": tb / n_b, "naive_a": naive_a, "m_median": m}


def test_choose_is_bar_first():
    aggs = {"X": _agg(14, 0), "Y": _agg(13, 3), "Z": _agg(9, 5)}
    assert [k for k, _ in R.choose(aggs, SPEC.j)] == ["Y", "X", "Z"]
    assert R.meets_bar(aggs["Y"], SPEC.j) and not R.meets_bar(aggs["X"], SPEC.j)
    assert R.meets_bar(_agg(11, 2), SPEC.j) and not R.meets_bar(_agg(10, 9), SPEC.j)


def test_choose_ties_go_to_f_a_then_median_m():
    aggs = {"P": _agg(12, 2, m=1.0), "Q": _agg(12, 3, m=0.0), "S": _agg(12, 2, m=3.0)}
    assert [k for k, _ in R.choose(aggs, SPEC.j)] == ["Q", "S", "P"]


def _pairs():
    od = {"o": 1.0}
    b = [dict(axis="b", turn=t, x=f"m{t}{k} vs o", y=f"m{t}{k} vs q", odor_x=od, odor_y=od)
         for t in range(14) for k in range(3)]
    a = [dict(axis="a", turn=t, x=f"u{t}", y=f"v{t}", odor_x=od, odor_y=od) for t in range(14)]
    return dict(b=b, a=a)


def test_judgement_set_takes_21_from_turn_4_and_checks_the_digests():
    spec = dataclasses.replace(SPEC, judge_b_digest="", judge_a_digest="")
    got = R.judgement_set(_pairs(), spec)
    assert len(got["b"]) == 21 and got["b"][0]["turn"] == 4 and {p["turn"] for p in got["a"]} == set(range(4, 11))
    assert len(got["b_digest"]) == 64 and len(got["a_digest"]) == 64
    with pytest.raises(ValueError, match="digest"):
        R.judgement_set(_pairs(), SPEC)


def test_judgement_set_pinned_digests_pass():
    free = R.judgement_set(_pairs(), dataclasses.replace(SPEC, judge_b_digest="", judge_a_digest=""))
    spec = dataclasses.replace(SPEC, judge_b_digest=free["b_digest"], judge_a_digest=free["a_digest"])
    assert R.judgement_set(_pairs(), spec) == free
    with pytest.raises(ValueError, match="digest"):                        # the (a) digest is checked too
        R.judgement_set(_pairs(), dataclasses.replace(spec, judge_a_digest="0" * 64))


NOTE = "C3 자체가 기준을 넘어 판독 확장의 효과로 말할 수 없다"


@pytest.mark.parametrize("n,fa,c,out", [(11, 2, 10, R.SELECTED), (11, 2, 11, R.B_NO_CONCLUSION), (12, 1, 3, R.B_FA),
                                        (5, 3, 5, R.B_TB), (4, 3, 7, R.B_TB), (8, 3, 4, R.B_NO_CONCLUSION),
                                        (14, 3, 12, R.SELECTED), (12, 3, 13, R.B_NO_CONCLUSION),
                                        (10, 3, 10, R.B_TB), (10, 3, 12, R.B_TB), (11, 1, 11, R.B_NO_CONCLUSION),
                                        (21, 9, 21, R.B_NO_CONCLUSION), (0, 0, 0, R.B_TB)])
def test_stage3_reading_edges(n, fa, c, out):
    got = R.stage3_reading(_agg(n, fa), _agg(c, 0), SPEC)
    assert got["outcome"] == out and (got["n"], got["c"], got["F_a"]) == (n, c, fa)
    assert (got["note"] == NOTE) == (c >= SPEC.j.stage2_select_testable_b)


def test_stage3_reading_b_fa_carries_naive_a():
    got = R.stage3_reading(_agg(12, 1, naive_a=1), _agg(3, 0), SPEC)
    assert got["outcome"] == R.B_FA and got["naive_a"] == 1 and got["f_a_possible"] is False
    got = R.stage3_reading(_agg(12, 1, naive_a=5), _agg(3, 0), SPEC)
    assert got["f_a_possible"] is True


def test_no_reading_off_21_pairs():
    got = R.stage3_reading(_agg(11, 2, n_b=20), _agg(3, 0), SPEC)
    assert got["outcome"] is None and "not a judgement" in got["note"]
    got = R.stage3_reading(_agg(11, 2), _agg(3, 0, n_b=20), SPEC)
    assert got["outcome"] is None


def test_no_reading_under_smoke():
    sm = smoke(SPEC)
    got = R.stage3_reading(_agg(2, 2, n_b=sm.judge_n_b), _agg(0, 0, n_b=sm.judge_n_b), sm)
    assert got["outcome"] is None


_SEL = dict(combo="PPL103·PAM10", n=12, c=5, fa=3, h=4, n_cands=19, m=6, w="PPL103 3.0·PAM10 2.5", dan=(13, 15),
            n_b=21, moves=7, top_move="Surf 6/21", k=13)


def test_sentences_carry_their_scope():
    s = R.sentence(R.B_TB, dict(combo="PPL103·PAM10", n=5, c=6, k=12, h=3, n_b=21))
    assert "팔별 상위 2 × 2" in s and "5/21" in s and "C3 6/21" in s and "짝수 선택 12/21" in s and "홀수 기록 3/20" in s
    s = R.sentence(R.SELECTED, _SEL)
    for part in ("학습 시험이 아니다", "19후보", "특이성 통과 6개", "J·K·L·M", "G·H·J·K·L·M", "팔별 상위 2 × 2", "12/21 대 5/21",
                 "같은 세트 C3", "F_a 3", "홀수 기록 4/20", "기술 7종·Surf 6/21", "DAN 세포 수 PPL 13·PAM 15",
                 "core w_mbon PPL103 3.0·PAM10 2.5", "F v4"):
        assert part in s, part
    s = R.sentence(R.STOP_NO_GAIN, dict(combo="PPL103·PAM10", n=9, fa=1))
    assert "팔별 상위 2 × 2" in s and "최대 9/21·F_a 1" in s
    assert "PPL105·PAM08" in R.sentence(STOP_NO_SPECIFICITY, {})
    assert "반응 가드" in R.sentence(R.STOP_NO_CANDIDATE, {})


def test_open_readings_go_to_the_user():
    s = R.sentence(R.B_NO_CONCLUSION, dict(combo="PPL103·PAM10", n=12, c=13, fa=3, k=13, h=4, n_b=21, note=NOTE))
    assert "사용자" in s and NOTE in s and "팔별 상위 2 × 2" in s and "12/21" in s and "C3 13/21" in s
    s = R.sentence(R.B_NO_CONCLUSION, dict(combo="PPL103·PAM10", n=9, c=4, fa=3, k=13, h=4, n_b=21, note=None))
    assert NOTE not in s and "None" not in s
    s = R.sentence(R.B_FA, dict(combo="PPL103·PAM10", n=12, c=4, fa=1, k=13, h=4, n_b=21, naive_a=1,
                                f_a_possible=False))
    assert "사용자" in s and "F_a 1" in s and "naive_a 1" in s and "팔별 상위 2 × 2" in s
