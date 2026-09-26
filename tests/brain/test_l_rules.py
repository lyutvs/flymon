"""Spec L.5 / L.11.3: row checks without the even-turn rule, the screening stop and coverage, the lift draw, the reading
in J.12.9's bands on 21 pairs, and the closing sentences."""
import dataclasses

import pytest

from flymon.brain import l_rules as lr
from flymon.brain.j_rules import B_FA, B_NO_CONCLUSION, B_TB, SELECTED
from flymon.brain.l_screen import SCREEN_FEW, SCREEN_IMPRECISE, SCREEN_NO_RULE
from flymon.brain.l_spec import SPEC

RULE = {"family": "G"}
F = lambda g: dict(G=g, S=0.5, f2=10.0, f3=0.3)


def test_screen_stops_at_the_nth_pass_and_ignores_the_rest_of_the_batch():
    feats = [F(i % 2 == 0) for i in range(10)]                # passes at 0, 2, 4, 6, 8
    r = lr.screen_order(feats, RULE, n_pass=3, max_screened=10)
    assert r["outcome"] == lr.SCREENED and r["passed"] == [0, 2, 4] and r["n_star"] == 5
    assert r["screened"] == [0, 1, 2, 3, 4] and r["coverage"] == pytest.approx(3 / 5)


def test_mid_batch_passes_after_the_nth_are_neither_counted_nor_in_the_denominator():
    # batch of 4 measured at positions 4-7: the 3rd pass lands at 5; the passes at 6 and 7 are cached, not screened
    feats = [F(i in (1, 3, 5, 6, 7)) for i in range(8)]
    r = lr.screen_order(feats, RULE, n_pass=3, max_screened=84)
    assert r["passed"] == [1, 3, 5] and r["n_star"] == 6 and r["screened"] == list(range(6))
    assert r["coverage"] == pytest.approx(3 / 6)


def test_the_nth_pass_exactly_at_the_cap_is_screened():
    feats = [F(i in (0, 4, 7)) for i in range(10)]
    r = lr.screen_order(feats, RULE, n_pass=3, max_screened=8)
    assert r["outcome"] == lr.SCREENED and r["n_star"] == 8 and r["coverage"] == pytest.approx(3 / 8)


def test_coverage_short_at_the_cap_or_the_end():
    feats = [F(i % 4 == 0) for i in range(12)]
    r = lr.screen_order(feats, RULE, n_pass=3, max_screened=8)                 # 2 passes by 8
    assert r["outcome"] == lr.COVERAGE_SHORT and r["n_star"] is None and r["screened"] == list(range(8))
    assert r["passed"] == [0, 4]
    assert lr.screen_order(feats[:6], RULE, n_pass=3, max_screened=84)["outcome"] == lr.COVERAGE_SHORT


def test_no_rule_passes_nothing():
    r = lr.screen_order([F(True)] * 5, None, n_pass=1, max_screened=5)
    assert r["outcome"] == lr.COVERAGE_SHORT and r["passed"] == []


def test_lift_draw_is_seeded_and_capped():
    a = lr.lift_draw(list(range(30)), 10, SPEC.lift_seed)
    assert a == lr.lift_draw(list(range(30)), 10, SPEC.lift_seed) and len(set(a)) == 10 and set(a) <= set(range(30))
    assert a != lr.lift_draw(list(range(30)), 10, SPEC.lift_seed + 1)
    assert sorted(lr.lift_draw([3, 5], 10, SPEC.lift_seed)) == [3, 5]
    assert lr.lift_draw([], 10, SPEC.lift_seed) == []


@pytest.mark.parametrize("tb,fa,band", [(7, 3, B_TB), (8, 3, B_NO_CONCLUSION), (10, 3, B_NO_CONCLUSION),
                                        (11, 2, SELECTED), (12, 1, B_FA)])
def test_reading_bands_on_21(tb, fa, band):
    agg = dict(testable_b=tb, n_b=21, T_b=tb / 21, F_a=fa, naive_a=4)
    assert lr.stage3_reading(agg, SPEC)["band"] == band


def test_reading_off_21_is_not_a_judgement():
    rd = lr.stage3_reading(dict(testable_b=2, n_b=2, T_b=1.0, F_a=0, naive_a=0), SPEC)
    assert rd["band"] is None


def _rep():
    # pre dV = [-1, 1], R1 dV = [2, 3], R2 dV = [0, -1]: every d' is defined (non-zero sd)
    return {"pre": {"A": [[1, 2], [2, 1]], "P": [[1, 1], [1, 1]]},
            "R1": {"A": [[3, 1], [4, 1]], "P": [[1, 1], [1, 1]]},
            "R2": {"A": [[1, 1], [1, 2]], "P": [[1, 1], [1, 1]]}}


SPEC4 = dataclasses.replace(SPEC.j.h4, report_seeds=(608, 609))
Z = {"A": (0.0, 1.0), "P": (0.0, 1.0)}


def test_row_checks_allow_any_turn():
    rows = [dict(axis="b", turn=3, x="m", y="n", report=_rep())]
    out = lr.pair_rows_stats(rows, Z, [("b", 3, "m", "n")], SPEC4)
    assert out["reasons"] == []
    assert set(out["stats"]) == {("b", 3, "m", "n")} and out["stats"][("b", 3, "m", "n")] is not None
    assert out["pairs"][0]["turn"] == 3 and "testable" in out["pairs"][0]


def test_row_checks_catch_duplicates_missing_short_and_undefined():
    rows = [dict(axis="b", turn=3, x="m", y="n", report=_rep())]
    assert "duplicate pair rows" in lr.pair_rows_stats(rows + rows, Z, [("b", 3, "m", "n")], SPEC4)["reasons"]
    out = lr.pair_rows_stats(rows, Z, [("b", 3, "m", "n"), ("a", 0, "p", "q")], SPEC4)
    assert any("missing 1" in r for r in out["reasons"])
    out = lr.pair_rows_stats(rows, Z, [("b", 3, "m", "n")], dataclasses.replace(SPEC4, report_seeds=(608, 609, 610)))
    assert any("not the 3 report seeds" in r for r in out["reasons"])
    flat = {ph: {"A": [[1, 1], [1, 1]], "P": [[1, 1], [1, 1]]} for ph in ("pre", "R1", "R2")}
    one = {ph: {"A": [[1, 2]], "P": [[1, 1]]} for ph in ("pre", "R1", "R2")}
    out = lr.pair_rows_stats([dict(axis="b", turn=5, x="m", y="n", report=one)], Z, [("b", 5, "m", "n")],
                             dataclasses.replace(SPEC4, report_seeds=(608,)))
    assert any("undefined d'" in r for r in out["reasons"])
    assert lr.pair_rows_stats([dict(axis="b", turn=5, x="m", y="n", report=flat)], Z, [("b", 5, "m", "n")],
                              SPEC4)["reasons"] == []                        # sd 0 is a limit, not undefined


def test_diversity_counts_distinct_species_moves_and_types():
    ps = [dict(x="Surf vs Chansey", y="Surf vs Rhydon", me="Blastoise", opp_types=["NORMAL"]),
          dict(x="Surf vs Hypno", y="Surf vs X", me="Blastoise", opp_types=["PSYCHIC"]),
          dict(x="Earthquake vs Lapras", y="Earthquake vs Y", me="Rhydon", opp_types=["WATER", "ICE"])]
    d = lr.diversity(ps)
    assert d == dict(n_me=2, n_moves=2, n_opp_types=4, top_move_share=pytest.approx(2 / 3))


LIFT = dict(m=1, n_lift=7)


def test_stage1_and_coverage_sentences():
    assert "8/41" in lr.sentence(SCREEN_FEW, dict(n_pass=8, n=41))
    s = lr.sentence(SCREEN_IMPRECISE, dict(precision=0.5, k=6, n_pass=12))
    assert "LOTO 0.5(6/12)" in s
    assert "0.35" in lr.sentence(SCREEN_NO_RULE, dict(cov=0.35))
    s = lr.sentence(lr.COVERAGE_SHORT, dict(rule="G and f2 > 13.5", n_pass=15, n_screened=84, floor=0.25))
    assert "커버리지 0.25에 못 미쳤다(15/84)" in s


def test_selected_sentence_names_the_numbers_the_scope_and_the_declaration():
    s = lr.sentence(SELECTED, dict(rule="G and f2 > 13.5", coverage=0.4, n=12, n_b=21, k=3, **LIFT,
                                   p=9, q=11, n_opp_types=8, top_move_share=0.25))
    assert "오라클 시험 가능성이며 학습 시험이 아니다" in s and "12/21" in s
    assert "통과 12/21 대 불통과 표본 1/7" in s and "/10" not in s
    assert "내 포켓몬 9종·기술 11개" in s and "상대 타입 8개" in s
    assert "F_a 3는 새 세트 앞 8턴의 고정 (a) 표본" in s and "엔진 전체의 요건이며, 선별된 상황의 요건이 아니다" in s
    assert "I(M2 no-go) 뒤 J·K·L의 네 번째 선언" in s


def test_every_stage3_sentence_carries_the_lift_phrase():
    base = dict(rule="G", coverage=0.5, n=9, n_b=21, k=1, **LIFT, naive_a=1, f_a_possible=False,
                p=9, q=11, n_opp_types=8, top_move_share=0.25)
    for st in (B_TB, B_NO_CONCLUSION, B_FA, SELECTED):
        assert "통과 9/21 대 불통과 표본 1/7" in lr.sentence(st, base), st
    assert "7/21" in lr.sentence(B_TB, base)
    s = lr.sentence(B_FA, base)
    assert "naive_a 1" in s and "f_a_possible False" in s and "선별된 상황의 요건이 아니다" in s
    assert "사용자가 판단한다" in lr.sentence(B_NO_CONCLUSION, base)


def test_unknown_state_raises():
    with pytest.raises(ValueError):
        lr.sentence("NOPE", {})
