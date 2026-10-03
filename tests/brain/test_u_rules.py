# tests/brain/test_u_rules.py
"""U's decisions: the bands, the guard and gate ② are R's / S's functions themselves (U.4 = T.4); the reuse condition
(U.3 1: R's repro + T's unedited z); the endpoint reproduction (U.9.1); the scan's candidates and cap (U.3 3 (a)(b), no
monotonicity, no substitutes); the KC band per checked f (U.3 3 (c)); the even reproduction (U.3 4); the f choice
(U.3 5 with U.9.2: (b) < 3, (a) ≤ 1, max testable_b, ties → larger f, f* ≥ 11) at its boundaries; the contrast
readings (U.9.3) at their boundaries; every sentence of U.7 / U.9.2 / U.9.4 fills; the OC notes and T's cluster table."""
import dataclasses

import pytest

from flymon.brain import r_rules, s_rules, t_rules
from flymon.brain import u_rules as U
from flymon.brain.u_spec import SPEC, at

Z_H4 = {"A": [10.78125, 9.412096743243064], "P": [26.25, 19.30889259728101]}
G_OK = {"passes": True, "median_delta": 6.0, "zero_share": 0.1}


def test_the_bands_guard_and_gate2_are_rs_and_ss_functions():
    assert U.read_band is r_rules.read_band and U.g_fail_s is s_rules.g_fail_s and U.gate2 is s_rules.gate2
    assert U.oc_cluster is t_rules.oc_cluster and U.set_outcome is t_rules.set_outcome
    assert SPEC.n_a == 43 and SPEC.g_fail_drop == 3


@pytest.mark.parametrize("n,c,f_a,g,band", [
    (14, 11, 5, False, U.B_NC), (8, 8, 5, False, U.B_TB), (10, 7, 5, False, U.B_NC), (12, 11, 5, False, U.B_NC),
    (11, 10, 5, False, U.B_NC), (12, 9, 5, True, U.B_PG), (12, 9, 1, False, U.B_FA), (12, 9, 2, False, U.SELECTED)])
def test_band_order_on_us_numbers(n, c, f_a, g, band):
    assert U.read_band(n, c, f_a, g, 21, 43, 5, SPEC)["band"] == band
    assert U.read_band(n, c, f_a, g, 21, 42, 5, SPEC)["band"] == U.NOT_READ


def _r_doc():
    k = SPEC.r_shared_key
    return dict(repro=dict(passed=True, code_key=k, csc_sha256_none=SPEC.sha_none))


def _t_doc(**z):
    none = dict(z=Z_H4, csc_sha256=[SPEC.sha_none], edit_edges=[0], guard={"MBON13": G_OK, "MBON05": G_OK})
    return dict(z=dict(dict(t_measure_key=SPEC.t_measure_key_t, code_key=SPEC.r_shared_key,
                            detail_sha256=SPEC.t_z_detail_sha256, none=none), **z))


GIT = dict(tracked=True, dirty=False)


def test_reuse_passes_and_names_every_break():
    ok = U.reuse(_r_doc(), GIT, _t_doc(), GIT, SPEC.r_shared_key, SPEC.t_measure_key_t, SPEC)
    assert ok["outcome"] == U.PASS
    r = U.reuse(_r_doc(), GIT, _t_doc(), GIT, "x" * 64, "y" * 64, SPEC)
    assert r["outcome"] == U.STOP_REUSE and len(r["reasons"]) >= 3 and r["sentence"].startswith("R·T 재사용 조건")
    bad_none = _t_doc(none=dict(_t_doc()["z"]["none"], guard={"MBON13": dict(G_OK, passes=False), "MBON05": G_OK}))
    assert U.reuse(_r_doc(), GIT, bad_none, GIT, SPEC.r_shared_key, SPEC.t_measure_key_t, SPEC)["outcome"] == U.STOP_REUSE
    z2 = _t_doc(none=dict(_t_doc()["z"]["none"], z={"A": [10.78125, 9.41209674324307], "P": Z_H4["P"]}))
    assert U.reuse(_r_doc(), GIT, z2, GIT, SPEC.r_shared_key, SPEC.t_measure_key_t, SPEC)["outcome"] == U.STOP_REUSE
    for td, tg in ((_t_doc(), dict(tracked=True, dirty=True)), ({}, GIT), (_t_doc(detail_sha256="0" * 64), GIT),
                   (_t_doc(t_measure_key="1" * 64), GIT)):
        assert U.reuse(_r_doc(), GIT, td, tg, SPEC.r_shared_key, SPEC.t_measure_key_t, SPEC)["outcome"] == U.STOP_REUSE
    no_repro = dict(repro=dict(passed=False, code_key=SPEC.r_shared_key))
    assert U.reuse(no_repro, GIT, _t_doc(), GIT, SPEC.r_shared_key, SPEC.t_measure_key_t, SPEC)["outcome"] == \
        U.STOP_REUSE


def test_path_stop_invalid_and_pass():
    ok = [dict(f=1.0, ref="T 편집 없는 엔진 기준 집합 행", diffs=[], invalid=[]),
          dict(f=0.0, ref="R 짝수 L 원자료 3쌍", diffs=[], invalid=[])]
    assert U.path(ok, SPEC)["outcome"] == U.PASS
    bad = [ok[0], dict(ok[1], diffs=["R 짝수 L: 1 pair(s) differ: ['b|0|x|y']"])]
    r = U.path(bad, SPEC)
    assert r["outcome"] == U.STOP_U_PATH_REPRO and r["failed"][0]["f"] == 0.0
    assert r["sentence"] == "U 부분 편집 경로가 끝점 0.0에서 R 짝수 L 원자료 3쌍을 재현하지 못했다(R 짝수 L: 1 pair(s) differ: " \
                            "['b|0|x|y'])."
    inv = [dict(ok[0], invalid=["f 1.0: 95 reference / 96 rest, declared 96"]), bad[1]]
    assert U.path(inv, SPEC)["outcome"] == U.INVALID


def _pt(ok, med13=6.0, zero13=0.1):
    g = {"MBON13": dict(passes=ok, median_delta=med13, zero_share=zero13),
         "MBON05": dict(passes=True, median_delta=70.0, zero_share=0.0)}
    return dict(outcome=U.PASS if ok else t_rules.STOP_Z_DEGENERATE, reasons=[], guard=g)


def test_scan_candidates_are_every_passing_f_and_the_smallest_three_are_checked():
    pts = {f: _pt(f in (0.2, 0.5, 0.6, 0.9)) for f in SPEC.f_grid}
    r = U.scan(pts, SPEC)
    assert r["outcome"] == U.PASS and r["candidates"] == [0.2, 0.5, 0.6, 0.9]
    assert r["checked"] == [0.2, 0.5, 0.6] and r["unchecked"] == [0.9]
    two = U.scan({f: _pt(f in (0.7, 0.3)) for f in SPEC.f_grid}, SPEC)
    assert two["checked"] == [0.3, 0.7] and two["unchecked"] == []


def test_scan_stop_and_invalid():
    r = U.scan({f: _pt(False, 1.0, 0.302) for f in SPEC.f_grid}, SPEC)
    assert r["outcome"] == U.STOP_NO_QUALIFIED_F and r["kind"] == U.NQ_GUARD
    assert r["sentence"].startswith("f ∈ {0.1, …, 0.9} 9점 모두 판독 반응성 가드에서 떨어졌다(f 0.1: MBON05 Δ 70.0·0 비율 "
                                    "0.000, MBON13 Δ 1.0·0 비율 0.302; f 0.2:")
    pts = {f: _pt(True) for f in SPEC.f_grid}
    pts[0.4] = dict(outcome=U.INVALID, reasons=["lever: edges [3], declared 2"], guard={})
    assert U.scan(pts, SPEC)["outcome"] == U.INVALID
    assert U.scan({f: _pt(True) for f in SPEC.f_grid[:8]}, SPEC)["outcome"] == U.INVALID


def _rec112(median=0.05, edges=(2,)):
    return dict(median=median, n_odours=112, n_seeds=[8], edit_edges=list(edges), csc_sha256=["s"])


def _rec55(vals, edges=(2,)):
    return dict(per_odour=vals, n_odours=55, n_seeds=[8], edit_edges=list(edges), edit_edges_none=[0])


def test_kc_point_is_rs_gate1_on_112_and_ts_supplement_on_55():
    spf = at(SPEC, 0.3)
    ok55 = {f"o{i}": 0.05 for i in range(55)}
    assert U.kc_point(_rec112(), True, _rec55(ok55), spf)["outcome"] == U.PASS
    assert U.kc_point(_rec112(0.16), True, _rec55(ok55), spf)["outcome"] == "FAIL"
    assert U.kc_point(_rec112(), False, _rec55(ok55), spf)["outcome"] == "FAIL"
    one = dict(ok55, o7=0.029)
    r = U.kc_point(_rec112(), True, _rec55(one), spf)
    assert r["outcome"] == "FAIL" and r["failed"] == ["T 세트 o7"]
    assert U.kc_point(_rec112(0.15), True, _rec55(dict(ok55, o1=0.03)), spf)["outcome"] == U.PASS
    assert U.kc_point(_rec112(edges=(0,)), True, _rec55(ok55), spf)["outcome"] == U.INVALID


def test_kc_qualified_and_stop_sentence():
    scan = dict(candidates=[0.2, 0.5, 0.6, 0.9], checked=[0.2, 0.5, 0.6], unchecked=[0.9])
    pts = {0.2: dict(outcome="FAIL", reasons=[]), 0.5: dict(outcome=U.PASS, reasons=[]),
           0.6: dict(outcome=U.PASS, reasons=[])}
    r = U.kc(pts, scan, SPEC)
    assert r["outcome"] == U.PASS and r["qualified"] == [0.5, 0.6]
    r = U.kc({f: dict(outcome="FAIL", reasons=[]) for f in scan["checked"]}, scan, SPEC)
    assert r["outcome"] == U.STOP_NO_QUALIFIED_F and r["kind"] == U.NQ_KC
    assert r["sentence"] == ("가드 스캔 통과 후보 {0.2, 0.5, 0.6, 0.9} 가운데 작은 f부터 검사한 {0.2, 0.5, 0.6}이 모두 KC 대역에서 "
                             "떨어졌다(검사하지 않은 후보 {0.9}).")
    assert U.kc({0.2: pts[0.2]}, scan, SPEC)["outcome"] == U.INVALID


def test_even_repro():
    assert U.even_repro(dict(reasons=[], aggregate=dict(testable_b=7)), SPEC, 16) is None
    r = U.even_repro(dict(reasons=[], aggregate=dict(testable_b=6)), SPEC, 16)
    assert r["outcome"] == U.STOP_EVEN_REPRO and r["sentence"] == "R 짝수 원자료가 h4 z에서 관문 ③ 값을 재현하지 못했다(L 16, C 6)."


def _er(f, tb, db, da, fa=3):
    return dict(f=f, testable_b=tb, F_a=fa, naive_a=5, drops=dict(b=dict(net_drop=db), a=dict(net_drop=da)))


def test_choose_filter_boundaries_ties_and_bar():
    """U.9.2: (b) net drop 2 passes and 3 fails; (a) net drop 1 passes and 2 fails; the largest testable_b wins, ties
    go to the larger f; f* needs ≥ 11 (bar_b)."""
    r = U.choose({0.3: _er(0.3, 14, 2, 1), 0.4: _er(0.4, 14, 0, 0), 0.5: _er(0.5, 13, 0, 0)}, 7, SPEC)
    assert r["outcome"] == U.PASS and r["f_star"] == 0.4 and r["kept"] == [0.3, 0.4, 0.5]
    r = U.choose({0.3: _er(0.3, 16, 3, 0), 0.4: _er(0.4, 15, 0, 2), 0.5: _er(0.5, 11, 2, 1)}, 7, SPEC)
    assert r["outcome"] == U.PASS and r["f_star"] == 0.5 and r["kept"] == [0.5] and r["testable_b"] == 11
    r = U.choose({0.3: _er(0.3, 16, 3, 0), 0.4: _er(0.4, 15, 0, 2)}, 7, SPEC)
    assert r["outcome"] == U.STOP_EVEN_PUNISH
    assert r["sentence"] == ("자격 f 모두 짝수 쌍에서 처벌 순감소가 거름 기준((b) < 3, (a) ≤ 1)을 넘었다(f 0.3: (b) 3·(a) 0; "
                             "f 0.4: (b) 0·(a) 2).")
    r = U.choose({0.3: _er(0.3, 10, 0, 0), 0.6: _er(0.6, 9, 0, 0)}, 7, SPEC)
    assert r["outcome"] == U.STOP_EVEN_LOW_LEVER and r["f_star"] == 0.3
    assert r["sentence"].endswith("(10/21, 지렛대 없는 같은 실행 7/21, F_a 기록 3/18). (f* = 0.3, 검사한 f {0.3, 0.6})")


def test_choose_is_not_a_g_fail_rule_change():
    """The filter reads S's per-axis net drop (s_rules.g_fail_s's formula) but the judgement's guard stays ≥ 3 on
    both axes."""
    assert SPEC.g_fail_drop == 3 and SPEC.even_drop_b_lt == 3 and SPEC.even_drop_a_le == 1
    assert dataclasses.replace(SPEC, even_drop_a_le=2).g_fail_drop == 3


def test_contrast_readings_at_their_boundaries():
    assert U.contrast_block(2.5, 5.0, SPEC) == U.CHAIN_FOR
    assert U.contrast_block(2.6, 5.0, SPEC) == U.NO_CONCLUSION
    assert U.contrast_block(4.0, 5.0, SPEC) == U.CHAIN_AGAINST
    assert U.contrast_block(3.9, 5.0, SPEC) == U.NO_CONCLUSION
    assert U.contrast_entry(dict(median_delta=5.0, zero_share=0.25), 1.0, SPEC) == U.CHAIN_FOR
    assert U.contrast_entry(dict(median_delta=5.0, zero_share=0.26), 1.0, SPEC) == U.NO_CONCLUSION
    assert U.contrast_entry(dict(median_delta=1.9, zero_share=0.3), 1.0, SPEC) == U.KC_SIDE
    assert U.contrast_entry(dict(median_delta=2.0, zero_share=0.3), 1.0, SPEC) == U.NO_CONCLUSION
    assert U.contrast_entry(dict(median_delta=0.1, zero_share=0.5), 1.0, SPEC) == U.KC_SIDE


def test_every_sentence_fills():
    base = dict(n=12, c=8, f_a=3, naive_a=5, T=103, k_even=12, f="0.4", pb_L=10, pb_C=12, pa_L=20, pa_C=21, d_b=2,
                d_a=1, rho1="0.700", rho2="0.800")
    for band in (U.B_TB, U.B_PG, U.B_FA, U.SELECTED):
        s = U.sentence(band, dict(base, reason="x"))
        assert "{" not in s.replace("{0.1, …, 0.9}", "")
    for reason in (U.C_ABOVE_BAR, U.BELOW_BAR, U.MARGIN):
        assert U.sentence(U.B_NC, dict(base, reason=reason)).endswith("(12/21 대 8/21, F_a 3/43, f = 0.4)")
    assert U.sentence(U.B_TB, dict(base, reason="")).startswith(
        "APL→MBON05 부분 제거(f = 0.4)가 넓힌 상대 풀 판정 세트(생성원 턴 0–103)에서 12/21로")
    assert U.sentence(U.B_PG, dict(base, reason="")).endswith("순감소 (b) 2, (a) 1). (f = 0.4)")
    assert U.sentence(U.B_FA, dict(base, reason="")).endswith("다음 병목은 F_a(순진 균형 (a) 쌍)다. (f = 0.4)")
    sel = U.sentence(U.SELECTED, dict(base, reason=""))
    assert "f = 0.4배로" in sel and "판정 시드 24_500_xxx" in sel and "네 번째 시도다(U.0)" in sel
    assert "f는 {0.1, …, 0.9} 가드 스캔을" in sel and sel.endswith("이것만으로 정당화하지 않는다.")
    with pytest.raises(KeyError):
        U.sentence(U.SELECTED, {"n": 1})


def test_oc_notes_and_cluster_table():
    oc = U.oc(SPEC)
    assert oc["notes"][-1] == "작동 특성은 U 세트 조건부 값이며 Q → R → S → T → U 전체 절차의 오선택률이 아니다."
    assert any("P3-10" in n for n in oc["notes"])
    g = {(r["kind"], r["a_pf"], r["a_fp"]): round(r["p"], 3) for r in oc["g_fail"]}
    assert g[("null", 0.02, 0.02)] == 0.034 and g[("null", 0.15, 0.15)] == 0.362
    assert g[("harm", 0.1, 0.02)] == 0.752 and g[("harm", 0.1, 0.05)] == 0.549
