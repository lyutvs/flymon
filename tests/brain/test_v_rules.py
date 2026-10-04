"""V's decisions (V.3, V.4, V.7, V.9): the bands, G_fail_S and gate ②'s order are R's / S's functions; reuse names
every break of R, T and U; path / kc_input / kc_band reproduction at their boundaries; z_V with the SD in its sentence;
the even gate's ① filter ((b) < 3, (a) ≤ 1) before ② testable_b ≥ 11; gate ②'s second scale; every sentence verbatim
with V.9.3's POOL condition and V.9.4's cluster values; the OCs (independent with V's notes, T's cluster method on V's
clusters)."""
import json

import pytest

from flymon.brain import r_rules, s_rules, t_rules, u_rules, v_rules
from flymon.brain.v_spec import SPEC
from tests.brain.v_fixtures import CANDS, r_doc, t_doc, u_doc


def test_the_bands_guard_and_gate2_are_rs_and_ss_functions():
    assert v_rules.read_band is r_rules.read_band and v_rules.g_fail_s is s_rules.g_fail_s
    assert v_rules.even_repro is u_rules.even_repro and v_rules.even_validity is u_rules.even_validity


@pytest.mark.parametrize("n,c,f_a,g,band", [(14, 11, 3, False, "B_결론없음"), (8, 8, 3, False, "B_Tb"),
                                            (10, 8, 3, False, "B_결론없음"), (12, 11, 3, False, "B_결론없음"),
                                            (14, 8, 3, True, "B_처벌가드"), (14, 8, 1, False, "B_Fa"),
                                            (14, 8, 2, False, "SELECTED"), (13, 11, 2, False, "B_결론없음")])
def test_band_order_on_vs_numbers(n, c, f_a, g, band):
    assert v_rules.read_band(n, c, f_a, g, 21, 43, 5, SPEC)["band"] == band
    assert v_rules.read_band(n, c, f_a, g, 21, 42, 5, SPEC)["band"] == "NOT_READ"


def _ok():
    return dict(tracked=True, dirty=False, judge_commits=[])


def _reuse(u=None, u_git=None, u_key=None, **kw):
    import dataclasses
    sp = dataclasses.replace(SPEC, sha_none="sha-C", sha_combined="sha-V", t_measure_key_t="t" * 64,
                             t_z_detail_sha256="d" * 64, u_measure_key_u="u" * 64, u_path_detail_sha256="e" * 64,
                             u_scan_detail_sha256="f" * 64, z_h4=(("A", (10.0, 9.0)), ("P", (26.0, 19.0))))
    return v_rules.reuse(r_doc(), _ok(), t_doc(), _ok(), u if u is not None else u_doc(), u_git or _ok(),
                         sp.r_shared_key, "t" * 64, u_key or "u" * 64, sp)


def test_reuse_passes_and_names_every_u_break():
    assert _reuse()["outcome"] == "PASS"
    out = _reuse(u_key="v" * 64)
    assert out["outcome"] == "STOP_REUSE" and "V 측정 키" in out["reasons"][0]
    assert "미커밋" in _reuse(u_git=dict(tracked=True, dirty=True))["reasons"][0]
    for mut, want in ((lambda d: d.pop("kc"), "U 블록 kc 없음"),
                      (lambda d: d["path"].__setitem__("u_measure_key", "x"), "U 블록 path의 키"),
                      (lambda d: d["path"].__setitem__("detail_sha256", "x"), "U 블록 path PASS / 원자료 sha x"),
                      (lambda d: d["scan"].__setitem__("detail_sha256", "x"), "U 블록 scan 원자료 sha x"),
                      (lambda d: d["scan"]["contrast"]["entry_f0"].__setitem__("invalid", ["bad"]), "entry_f0 행 무효"),
                      (lambda d: d["kc"]["points"]["0.7"]["record_set"].__setitem__("per_odour_none", {}),
                       "per_odour_none 읽기 실패")):
        d = u_doc()
        mut(d)
        out = _reuse(u=d)
        assert out["outcome"] == "STOP_REUSE" and any(want in x for x in out["reasons"]), (want, out["reasons"])


def test_path_invalid_stop_and_pass():
    ok = [dict(engine="편집 없는 엔진", ref="T 편집 없는 엔진 기준 집합 행", diffs=[], invalid=[])]
    assert v_rules.path(ok, [], SPEC)["outcome"] == "PASS"
    assert v_rules.path(ok, ["13 edges"], SPEC)["outcome"] == "INVALID"
    bad = ok + [dict(engine="조합 엔진", ref="U entry_f0 기준 집합 행", diffs=["기준 집합: 1 row(s) differ"], invalid=[])]
    out = v_rules.path(bad, [], SPEC)
    assert out["outcome"] == "STOP_V_PATH_REPRO" and out["sentence"] == (
        "V 편집 경로가 조합 엔진에서 U entry_f0 기준 집합 행을 재현하지 못했다(기준 집합: 1 row(s) differ).")


def _rec(n=20, edges=(0, 2), sha=("sha-C", "sha-V"), seeds=8, under=True):
    return dict(none=dict(n_odours=n, n_seeds=[seeds], edit_edges=[edges[0]], csc_sha256=[sha[0]]),
                lever=dict(n_odours=n, n_seeds=[seeds], edit_edges=[edges[1]], csc_sha256=[sha[1]]),
                cap=dict(all_under=under))


def test_kc_input_invalid_stop_and_pass():
    import dataclasses
    sp = dataclasses.replace(SPEC, sha_none="sha-C", sha_combined="sha-V")
    cmp_ok = dict(diffs=[])
    assert v_rules.kc_input(_rec(), cmp_ok, 20, sp)["outcome"] == "PASS"
    for r in (_rec(n=19), _rec(edges=(0, 3)), _rec(sha=("sha-C", "sha-L0")), _rec(seeds=7), _rec(under=False)):
        assert v_rules.kc_input(r, cmp_ok, 20, sp)["outcome"] == "INVALID"
    out = v_rules.kc_input(_rec(), dict(diffs=["A|B 0.05 ≠ U 0.0501"]), 20, sp)
    assert out["outcome"] == "STOP_V_PATH_REPRO" and out["sentence"] == (
        "V 편집 경로가 편집 없는 엔진에서 U KC 블록(c0d09a7) per_odour_none을 재현하지 못했다(A|B 0.05 ≠ U 0.0501).")


def test_set_outcome_sentence():
    assert v_rules.set_outcome(dict(status="OK"))["outcome"] == "PASS"
    assert v_rules.set_outcome(dict(status="STOP_SET_SHORT", n_b=21, n_a=40))["sentence"] == (
        "KC 입력으로 거른 넓힌 상대 풀 생성원 턴 0–1985에서 (b) 21·(a) 43을 채우지 못했다(21·40쌍).")


def _side(a_med=6.0, a_zero=0.1, p_med=40.0, z=True, block=None, sha="sha-V"):
    g = {"MBON13": dict(passes=a_med >= 5 and a_zero <= 0.25, median_delta=a_med, zero_share=a_zero),
         "MBON05": dict(passes=True, median_delta=p_med, zero_share=0.0)}
    return dict(guard=g, z={"A": [16.9, 12.48], "P": [80.2, 29.8]} if z else None, why=None if z else "zero SD",
                zero_sd=[] if z else ["MBON13"], n_ref=96, n_rest=96, edit_edges=[2], csc_sha256=[sha],
                block_edges=[block or json.dumps(dict(SPEC.contrast_declared()["chain_entry"]), sort_keys=True,
                                                 separators=(",", ":"))])


def test_z_v_pass_stop_with_sd_and_invalid():
    import dataclasses
    sp = dataclasses.replace(SPEC, sha_combined="sha-V")
    ro = {"A": "MBON13", "P": "MBON05"}
    assert v_rules.z_v(_side(), 96, ro, sp)["outcome"] == "PASS"
    out = v_rules.z_v(_side(a_med=4.0), 96, ro, sp)
    assert out["sentence"] == ("조합 지렛대 아래 기준 집합에서 판독 MBON13이 반응성 가드를 넘지 못했다(Δ 중앙값 4.0, 0 비율 0.100, "
                               "SD 12.480) — z를 정할 수 없다.")
    out = v_rules.z_v(_side(z=False), 96, ro, sp)
    assert out["outcome"] == "STOP_Z_DEGENERATE" and "SD 0)" in out["sentence"]
    assert v_rules.z_v(_side(block='{"MBON05->MBON09":7}'), 96, ro, sp)["outcome"] == "INVALID"
    assert v_rules.z_v(_side(sha="sha-L0"), 96, ro, sp)["outcome"] == "INVALID"
    assert v_rules.z_v(_side(), 95, ro, sp)["outcome"] == "INVALID"


def _g1(median=0.05, edges=2, n=112):
    return dict(median=median, edit_edges=[edges], csc_sha256=["sha-V"], n_odours=n, n_seeds=[8])


def test_kc_band_order():
    import dataclasses
    sp = dataclasses.replace(SPEC, sha_combined="sha-V")
    srec = dict(per_odour={o: 0.05 for o in CANDS[:5]}, edit_edges=[2], csc_sha256=["sha-V"])
    assert v_rules.kc_band(_g1(), True, srec, [], sp)["outcome"] == "PASS"
    assert v_rules.kc_band(_g1(edges=3), True, srec, [], sp)["outcome"] == "INVALID"
    assert v_rules.kc_band(_g1(), True, dict(srec, csc_sha256=["sha-C"]), [], sp)["outcome"] == "INVALID"
    out = v_rules.kc_band(_g1(median=0.2), True, srec, ["1 odour(s) differ: X"], sp)
    assert out["outcome"] == "STOP_V_PATH_REPRO"                         # the re-check comes before the band
    out = v_rules.kc_band(_g1(median=0.2), False, srec, [], sp)
    assert out["outcome"] == "STOP_STRENGTH_LEVER" and out["sentence"] == (
        "조합 지렛대 아래에서 E-grid k2-norm s 1.0이 KC 유효 대역을 잃었다(냄새별 KC 활성 중앙값 0.2000 ∉ [0.03, 0.15]; "
        "ORN 상한(s 1.0)).")
    out = v_rules.kc_band(_g1(), True, dict(srec, per_odour={"A|B": 0.0299}), [], sp)
    assert out["outcome"] == "STOP_STRENGTH_LEVER" and "V 세트 냄새 A|B 0.0299 ∉ [0.03, 0.15]" in out["sentence"]


def _even(db, da, tb, fa=0):
    return dict(drops={"b": dict(net_drop=db), "a": dict(net_drop=da)}, testable_b=tb, F_a=fa)


def test_even_gate_boundaries():
    assert v_rules.even(_even(2, 1, 11), 7, SPEC)["outcome"] == "PASS"
    assert v_rules.even(_even(3, 0, 15), 7, SPEC)["outcome"] == "STOP_EVEN_PUNISH"
    assert v_rules.even(_even(0, 2, 15), 7, SPEC)["outcome"] == "STOP_EVEN_PUNISH"
    out = v_rules.even(_even(3, 2, 5), 7, SPEC)                           # ① before ②
    assert out["sentence"] == ("조합 지렛대가 짝수 쌍에서 처벌 순감소 거름 기준((b) < 3, (a) ≤ 1)을 넘었다((b) 3, (a) 2). "
                               "(POOL 안 짝수 쌍 조건부)")
    out = v_rules.even(_even(-1, -2, 10, 4), 7, SPEC)
    assert out["outcome"] == "STOP_EVEN_LOW_LEVER" and out["sentence"] == (
        "조합 지렛대 아래 짝수 (b) 21쌍에서 testable_b가 M2 기준(11)에 못 미쳤다(10/21, 지렛대 없는 같은 실행 7/21, F_a 기록 "
        "4/18). (POOL 안 짝수 쌍 조건부)")


def _res(label="LEARNS_CONFIRMATORY", l1=1.0, l2=1.0):
    return dict(outcome="PASS", label=label, directions={"r1": dict(ell=l1), "r2": dict(ell=l2)})


def _ratio(r1, r2, ell_l=0.5, ell_c=1.0):
    return {d: dict(ratio=v, ell_L=ell_l, ell_C=ell_c) for d, v in (("r1", r1), ("r2", r2))}


_WEAK = "조합 지렛대 아래 같은 시드 P에서 처벌 학습량 비가 0.5에 못 미쳤다"


def test_gate2_both_scales_in_order():
    """V.9.2 order (h4 ratio, then z_V ratio) with V.9.7 1's V-only sentence: both scales' ratios always, and every
    failing scale·direction named."""
    assert v_rules.gate2(_res(), _res(), _ratio(0.6, 0.7), _ratio(0.5, 0.9), SPEC)["outcome"] == "PASS"
    out = v_rules.gate2(_res(), _res(), _ratio(0.49, 0.7), _ratio(0.9, 0.9), SPEC)
    assert (out["outcome"], out["scale"], out["low"]) == ("STOP_PUNISH_WEAKENED", "h4", ["r1"])
    assert out["failed_scales"] == [["h4", "r1"]]
    assert out["sentence"] == (_WEAK + "(h4 z 비 ℓ_r1 0.490·ℓ_r2 0.700, z_V 비 ℓ_r1 0.900·ℓ_r2 0.900; "
                               "미달: h4 z ℓ_r1).")
    out = v_rules.gate2(_res(), _res(), _ratio(0.6, 0.7), _ratio(0.9, 0.499, 0.4, 0.81), SPEC)
    assert (out["outcome"], out["scale"], out["low"]) == ("STOP_PUNISH_WEAKENED", "z_V", ["r2"])
    assert out["sentence"] == (_WEAK + "(h4 z 비 ℓ_r1 0.600·ℓ_r2 0.700, z_V 비 ℓ_r1 0.900·ℓ_r2 0.499; "
                               "미달: z_V ℓ_r2).")
    out = v_rules.gate2(_res(), _res(), _ratio(0.49, 0.7), _ratio(0.3, 0.4), SPEC)   # both scales fail: h4 first
    assert (out["outcome"], out["scale"], out["low"]) == ("STOP_PUNISH_WEAKENED", "h4", ["r1"])
    assert out["failed_scales"] == [["h4", "r1"], ["z_V", "r1"], ["z_V", "r2"]]
    assert out["sentence"] == (_WEAK + "(h4 z 비 ℓ_r1 0.490·ℓ_r2 0.700, z_V 비 ℓ_r1 0.300·ℓ_r2 0.400; "
                               "미달: h4 z ℓ_r1, z_V ℓ_r1, z_V ℓ_r2).")
    out = v_rules.gate2(_res(), _res(), _ratio(0.5, 0.5), _ratio(0.5, 0.5), SPEC)       # 0.5 itself passes
    assert (out["outcome"], out["scale"]) == ("PASS", None)
    assert v_rules.gate2(_res("NOT_LEARNING"), _res(), None, None, SPEC)["outcome"] == "STOP_PUNISH_BROKEN"
    assert v_rules.gate2(_res(), _res("NOT_LEARNING"), None, None, SPEC)["outcome"] == "STOP_P_REFERENCE"
    assert v_rules.gate2(dict(outcome="INVALID", reasons=["x"]), _res(), None, None, SPEC)["outcome"] == "INVALID"


F = dict(n=14, c=8, f_a=3, naive_a=4, T=296, k_even=13, k_cl=11, pb_L=20, pb_C=21, pa_L=40, pa_C=41, d_b=1, d_a=1,
         rho1="0.700", rho2="0.800", rho1_zv="0.650", rho2_zv="0.550", cl_null="0.401", cl_harm1="0.685",
         cl_harm2="0.571")

_SELECTED = (
    "C3에서 APL→MBON05 2간선 제거와 MBON05→MBON09/MBON11/MBON01 11간선 제거를 함께 한 모델 변형(커넥톰 간선 13개 제거)과 "
    "E-grid k2-norm(s 1.0)에서, 엔진 변형마다 자기 기준 집합 z로 읽었을 때, 편집 없는 엔진과 조합 엔진 둘 다에서 KC 입력으로 "
    "거른 판정 세트(상대를 1세대 기본 폼으로 넓힌 풀의 생성원 턴 0–296, 기존 세트·키·사구체 중복 제외, 모든 행이 POOL 밖 상대 "
    "타입 조합 포함) (b) 21쌍의 H.4 오라클 시험 가능성이 M2 기준을 만족했고 지렛대 없는 같은 세트보다 높았다(14/21 대 8/21, "
    "여유 ≥ 2, F_a 3/43, 처벌 순감소 가드(축마다 ≥ 3) 통과, 같은 시드 P 비 h4 z ℓ_r1 0.700·ℓ_r2 0.800, z_V ℓ_r1 0.650·ℓ_r2 "
    "0.550 ≥ 0.5(약화 정도 기록), 짝수 13/21, 판정 시드 24_600_xxx). 판정 세트의 상대는 1세대 기본 폼으로 넓힌 풀이며 새 "
    "키는 POOL에 없는 상대 타입 조합에서 나온다 — 과제 풀 안의 시험 가능성은 R·S가 마지막이다. (b) 21쌍은 11개 상대 타입 "
    "조합 쌍에 몰려 있어 쌍끼리 독립이 아니다. 지렛대는 U 기전 기록(사슬 진입부 차단에서 MBON13 회복)을 보고 골랐고 같은 "
    "측정의 출력 차단 읽기는 '사슬 비지지'였다. 이것은 Q 결과로 고른 지렛대 계열의 R·S·T·U에 이은 다섯 번째 시도다(V.0). "
    "V 세트 군집 상관 모형에서 G_fail_S null 0.401, 오선택(harm) 0.685 / 0.571. 이 결과는 이 64쌍 조건부 기술 결과다. "
    "작동 특성은 V 세트 조건부 값이며 Q → R → S → T → U → V 전체 절차의 오선택률이 아니다. 실제 커넥톰 간선을 지운 모델이며 "
    "오라클 시험 가능성이지 학습 시험이 아니다. 귀결은 넓힌 풀에서 엔진별 z로 M2 시험 가능성이 섰다는 것까지이며, POOL 배틀 "
    "과제의 F v4 학습 시험을 이것만으로 정당화하지 않는다.")


def test_every_sentence_fills():
    for o in ("B_Tb", "B_처벌가드", "B_Fa", "SELECTED"):
        assert v_rules.sentence(o, dict(F, reason="PASS"))
    for reason in ("C_ABOVE_BAR", "BELOW_BAR", "MARGIN"):
        assert v_rules.sentence("B_결론없음", dict(F, reason=reason)).endswith("(14/21 대 8/21, F_a 3/43, 조합 지렛대)")
    assert v_rules.sentence("B_Tb", F) == (                                # V.9.7 3: filtered on both engines
        "APL→MBON05 2간선 제거와 MBON05→MBON09/MBON11/MBON01 11간선 제거를 함께 한 모델 변형(커넥톰 간선 13개 제거)이 편집 "
        "없는 엔진과 조합 엔진 둘 다에서 KC 입력으로 거른 넓힌 상대 풀 판정 세트(생성원 턴 0–296)에서 14/21로 지렛대 없는 같은 "
        "세트 8/21보다 오르지 않았다(엔진마다 자기 기준 집합 z). → 넓힌 풀·엔진별 z에서의 이 조합 지렛대 주장을 닫는다.")
    assert v_rules.sentence("B_처벌가드", F).endswith(" (조합 지렛대)") and v_rules.sentence("B_Fa", F).endswith(" (조합 지렛대)")
    s = v_rules.sentence("SELECTED", F)
    assert s == _SELECTED                                                  # V.7 + V.9.4 + V.9.7 2 / 3
    assert "(b) 21쌍은 11개 상대 타입 조합 쌍에 몰려" in s and "판정 시드 24_600_xxx" in s and "생성원 턴 0–296" in s
    assert ("V 세트 군집 상관 모형에서 G_fail_S null 0.401, 오선택(harm) 0.685 / 0.571. 이 결과는 이 64쌍 조건부 기술 결과다. "
            "작동 특성은 V 세트 조건부 값이며") in s
    assert "같은 시드 P 비 h4 z ℓ_r1 0.700·ℓ_r2 0.800, z_V ℓ_r1 0.650·ℓ_r2 0.550 ≥ 0.5" in s
    assert all("KC 입력 유효성" not in v_rules.SENTENCES[o] for o in ("B_Tb", "SELECTED"))
    with pytest.raises(KeyError):                                          # the z_V ratios are required fields
        v_rules.sentence("SELECTED", {k: v for k, v in F.items() if k != "rho2_zv"})
    assert s.endswith("POOL 배틀 과제의 F v4 학습 시험을 이것만으로 정당화하지 않는다.")
    assert v_rules.sentence("STOP_REUSE", dict(why="x")).startswith("R·T·U 재사용 조건(V.3 1)이 깨졌다(x).")
    assert v_rules.SENTENCES["STOP_EVEN_REPRO"] == t_rules.SENTENCES["STOP_EVEN_REPRO"]
    for o in ("STOP_PUNISH_BROKEN", "STOP_P_REFERENCE"):
        assert v_rules.SENTENCES[o] == s_rules.SENTENCES[o]
    assert v_rules.sentence("STOP_PUNISH_WEAKENED", dict(h1="0.4", h2="0.6", v1="0.7", v2="0.8", miss="h4 z ℓ_r1")) == (
        _WEAK + "(h4 z 비 ℓ_r1 0.4·ℓ_r2 0.6, z_V 비 ℓ_r1 0.7·ℓ_r2 0.8; 미달: h4 z ℓ_r1).")   # V.9.7 1: V's own


def test_oc_notes_and_cluster_values():
    oc = v_rules.oc(SPEC)
    assert oc["notes"] == list(v_rules.OC_NOTES) and len(oc["sha256"]) == 64
    assert "Q → R → S → T → U → V" in oc["notes"][-1]
    table = dict(g_fail=[dict(kind="null", p=0.40069), dict(kind="harm", p=0.68484), dict(kind="harm", p=0.57129)])
    assert v_rules.cluster_values(table) == dict(cl_null="0.401", cl_harm1="0.685", cl_harm2="0.571")


def test_cluster_model_on_ts_clusters_reproduces_ts_fixture():
    """V.6: T's method — on T's own clusters V's oc_cluster reproduces T's committed fixture's g rows and n_dist (only
    the notes and therefore the sha differ)."""
    from pathlib import Path

    from flymon.brain.t_spec import SPEC as T
    root = Path(__file__).resolve().parents[2]
    fx = json.loads((root / T.oc_cluster_fixture).read_text())
    got = v_rules.oc_cluster([list(c) for c in T.clusters_b], [list(c) for c in T.clusters_a])
    assert got["g_fail"] == fx["g_fail"] and got["n_dist"] == fx["n_dist"] and got["rows"] == fx["rows"]
    assert got["notes"] == list(v_rules.OC_NOTES[1:]) and got["seed"] == 20261005 and got["icc"] == 0.3
