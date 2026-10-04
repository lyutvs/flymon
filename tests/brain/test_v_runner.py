"""The V stage chain up to gate ② (V.3 as ordered by V.9.6): reuse -> path -> kc_input -> set -> z -> kc_band -> even ->
smoke -> oc -> gate2_oc -> gate2; each stage refuses (exit 2, nothing written) when an earlier block is missing, a
later one exists, its own block exists (gate ②'s one INVALID rerun excepted), the summary is uncommitted or a hashed
file is dirty; STOP_REUSE, STOP_V_PATH_REPRO (path, kc_input, kc_band), STOP_SET_SHORT, STOP_Z_DEGENERATE (SD in the
sentence), STOP_STRENGTH_LEVER, STOP_EVEN_REPRO (L_V then unmeasured), STOP_EVEN_PUNISH / STOP_EVEN_LOW_LEVER (POOL
condition appended), a smoke problem and gate ② STOPs (both scales) block every later stage; the reuse condition broken
after `reuse` refuses with exit 7; L_V's oracle runs on z_V and C / E0's on block h4's z; the set is regenerated from
block kc_input and checked against block set; the judgement set is touched only through its odours before jm."""
import json
from pathlib import Path

import pytest

from flymon.brain import v_rules
from flymon.brain import v_runner as VR
from flymon.brain.v_spec import LEVER_V
from tests.brain.v_world import APL, CANDS, SPEC, U_IDS, ZV, World, ZScripted, doc, through


@pytest.fixture
def w(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def _to(w, m, zm, last):
    r = w.runner(m, zm)
    for s in VR.ORDER[:VR.ORDER.index(last)]:
        getattr(r, f"stage_{s}")()
    return r


def test_chain_to_even(w):
    m, zm = w.scripted(), ZScripted()
    through(w, m, last="even", zm=zm)
    d = doc()
    assert set(d) == set(VR.ORDER[:VR.ORDER.index("smoke")])
    assert all(d[b]["pipeline_key"] == "p" * 64 and d[b]["u_measure_key"] == "u" * 64 for b in d)
    assert d["reuse"]["records"]["u"]["entry_reading"] == "사슬 지지"
    pa = d["path"]
    assert [(c["engine"], c["diffs"]) for c in pa["checks"]] == [
        ("편집 없는 엔진", []), ("APL→MBON05 제거 단독 엔진", []), ("조합 엔진", []), ("APL→MBON05 제거 단독 엔진", []),
        ("편집 없는 엔진", [])]
    assert [c[1] for c in zm.calls if c[0] == "reference"] == ["none", APL, LEVER_V]
    assert pa["sides"]["none"]["z"] == {"A": [10.0, 9.0], "P": [26.0, 19.0]} and Path(SPEC.path_detail).exists()
    assert pa["sides"]["lever"]["block_edges"] == [json.dumps(dict(SPEC.contrast_declared()["chain_entry"]),
                                                              separators=(",", ":"), sort_keys=True)]
    assert pa["records"]["lever"] == dict(p_mean_ratio=80.0 / 26.0, sd_ratio={"A": 3.0 / 9.0, "P": 20.0 / 19.0})
    assert set(pa["sides"]["lever"]["mech"]["types"]) == set(w.ctx["rec_types"])
    pc = [c for c in m.calls if c[0] == "oracle" and c[1] == "path"]
    assert pc == [("oracle", "path", "L", APL, 3, (10.0, 9.0)), ("oracle", "path", "C", "none", 3, (10.0, 9.0))]
    ki = d["kc_input"]
    assert ki["n_candidates"] == 20 and ki["u_compare"]["n_shared"] == 5 and ki["u_compare"]["differ"] == []
    assert ki["record"]["none"]["seed_range"][CANDS[0]] == [0.05, 0.05] and ki["record"]["cap"]["all_under"]
    acts = [c for c in m.calls if c[0] == "activity"]
    assert acts[:2] == [("activity", "kc_input", "none", 20), ("activity", "kc_input", LEVER_V, 20)]
    st = d["set"]
    assert st["set"]["n_b"] == 21 and st["set"]["n_a"] == 43 and st["set"]["digest_keys"] == "k" * 64
    assert "odour_ids" in st and "kc_record" in st
    z = d["z"]
    assert z["z_V"] == {"A": [6.0, 3.0], "P": [80.0, 20.0]} and z["z_none_reused"] is True
    kb = d["kc_band"]
    assert kb["calib"]["outcome"] == "PASS" and kb["record_set"]["n_odours"] == 10 and kb["record_e0"]["n_odours"] == 3
    assert [c[1:3] for c in acts[2:]] == [("kc_band", LEVER_V), ("kc_band", LEVER_V), ("kc_band_e0", "none")]
    ev = d["even"]
    assert ev["c_even"] == 7 and ev["L_h4_R"] == dict(testable_b=16, matches_r_gate3=True)
    assert ev["record"]["testable_b"] == 13 and ev["record"]["drops"]["b"]["net_drop"] == 0
    evc = [c for c in m.calls if c[0] == "oracle" and c[1] == "even"]
    assert evc == [("oracle", "even", "L", LEVER_V, 39, ZV["A"])]
    assert w.judgement_calls == 0 and w.odour_calls == 1


def test_reuse_stop_names_u_and_later_stages_refuse(w):
    w.u["scan"]["detail_sha256"] = "x" * 64
    out = w.runner(w.scripted(), ucode={"key": "v" * 64}).stage_reuse()
    assert out["outcome"] == v_rules.STOP_REUSE and "V 측정 키" in out["sentence"] and "scan 원자료" in out["sentence"]
    assert out["sentence"].startswith("R·T·U 재사용 조건(V.3 1)이 깨졌다(")
    with pytest.raises(SystemExit) as e:
        w.runner(w.scripted()).stage_path()
    assert e.value.code == 2


def test_reuse_broken_after_reuse_refuses_with_7(w):
    r = w.runner(w.scripted())
    r.stage_reuse()
    w.u_git["dirty"] = True
    with pytest.raises(SystemExit) as e:
        r.stage_path()
    assert e.value.code == VR.EXIT_KEY and "path" not in doc()


def test_path_stop_on_the_unedited_rows(w):
    a, p = [1, 19] * 47 + [1, 20], [7, 45] * 48
    out = _to(w, w.scripted(), ZScripted(counts={"none": (a, p)}), "path").stage_path()
    assert out["outcome"] == v_rules.STOP_V_PATH_REPRO and out["failed"][0]["engine"] == "편집 없는 엔진"
    assert out["sentence"].startswith("V 편집 경로가 편집 없는 엔진에서 T 편집 없는 엔진 기준 집합 행을 재현하지 못했다(기준 "
                                      "집합: 1 row(s) differ")


def test_path_mutation_the_chain_entry_never_reached_the_engine(w):
    """V.3 2 unit test: an L_V whose chain-entry cut is not applied (its rows are the APL-only rows) fails against U."""
    zm = ZScripted(counts={LEVER_V: ([0, 2] * 48, [60, 100] * 48)})
    out = _to(w, w.scripted(), zm, "path").stage_path()
    assert out["outcome"] == v_rules.STOP_V_PATH_REPRO and out["failed"][0]["engine"] == "조합 엔진"
    assert "U entry_f0 기준 집합 행" in out["sentence"]


def test_path_mutation_an_oracle_on_the_unedited_csc_fails(w):
    m = w.scripted(override={("path", "L"): dict(sha="sha-C")})
    out = _to(w, m, ZScripted(), "path").stage_path()
    assert out["outcome"] == v_rules.STOP_V_PATH_REPRO and out["failed"][0]["ref"] == "R 짝수 L 원자료 3쌍"


@pytest.mark.parametrize("bad", [dict(changed=12), dict(csc_sha256="sha-x"),
                                 dict(mbon05_apl=dict(n=2, before=[-7.86, -12.93], after=[0.0, -12.93],
                                                      same_bits=False)),
                                 dict(block_edges={"MBON05->MBON01": 2, "MBON05->MBON09": 6, "MBON05->MBON11": 2})])
def test_path_invalid_on_the_static_edit_facts(w, bad):
    w.facts.update(bad)
    out = _to(w, w.scripted(), ZScripted(), "path").stage_path()
    assert out["outcome"] == v_rules.INVALID and out["reasons"]
    with pytest.raises(SystemExit):
        w.runner(w.scripted()).stage_kc_input()


def test_path_invalid_without_96_presentations(w):
    out = _to(w, w.scripted(), ZScripted(n=95), "path").stage_path()
    assert out["outcome"] == v_rules.INVALID and any("95 reference" in x for x in out["reasons"])


def test_path_invalid_on_other_chain_entry_edges(w):
    zm = ZScripted(blocks={LEVER_V: {"MBON05->MBON09": 6}})
    out = _to(w, w.scripted(), zm, "path").stage_path()
    assert out["outcome"] == v_rules.INVALID and any("chain entry edges" in x for x in out["reasons"])


def test_kc_input_stop_when_us_none_value_differs(w):
    w.u["kc"]["points"] = {k: dict(record_set=dict(per_odour_none={U_IDS[0]: 0.0501}))
                           for k in SPEC.u_kc_points}
    out = _to(w, w.scripted(), ZScripted(), "kc_input").stage_kc_input()
    assert out["outcome"] == v_rules.STOP_V_PATH_REPRO
    assert out["sentence"].startswith("V 편집 경로가 편집 없는 엔진에서 U KC 블록(c0d09a7) per_odour_none을 재현하지 "
                                      f"못했다({U_IDS[0]} 0.05 ≠ U 0.0501)")


def test_kc_input_invalid_on_edges(w):
    m = w.scripted(arm_edges={LEVER_V: 3})
    out = _to(w, m, ZScripted(), "kc_input").stage_kc_input()
    assert out["outcome"] == v_rules.INVALID and any("lever: edges [3]" in x for x in out["reasons"])


def test_set_filters_by_both_engines(w):
    m = w.scripted(kc={("kc_input", LEVER_V): {CANDS[0]: 0.2}})
    r = _to(w, m, ZScripted(), "set")
    st = r.stage_set()
    assert st["outcome"] == "PASS" and CANDS[0] not in st["odour_ids"] and st["set"]["skipped"]["kc_input"] == 1


def test_set_short_sentence(w):
    w.short = True
    out = _to(w, w.scripted(), ZScripted(), "set").stage_set()
    assert out["outcome"] == v_rules.STOP_SET_SHORT and out["sentence"] == (
        "KC 입력으로 거른 넓힌 상대 풀 생성원 턴 0–1985에서 (b) 21·(a) 43을 채우지 못했다(0·0쌍).")
    with pytest.raises(SystemExit):
        w.runner(w.scripted()).stage_z()


def test_z_stop_names_the_sd(w):
    zm = ZScripted(counts={LEVER_V: ([0, 2] * 48, [60, 100] * 48)})
    w.u_rows = dict(w.u_rows, ref=zm.reference(LEVER_V, [None] * 48, 0.35, 800.0, 600))
    out = _to(w, w.scripted(), zm, "z").stage_z()
    assert out["outcome"] == v_rules.STOP_Z_DEGENERATE and out["sentence"] == (
        "조합 지렛대 아래 기준 집합에서 판독 MBON13이 반응성 가드를 넘지 못했다(Δ 중앙값 1.0, 0 비율 0.500, SD 1.000) — z를 "
        "정할 수 없다.")


def test_kc_band_recheck_and_strength(w):
    m = w.scripted(kc={("kc_band", LEVER_V): {f"K{i}|T": 0.0499 for i in range(15)}})
    out = _to(w, m, ZScripted(), "kc_band").stage_kc_band()
    assert out["outcome"] == v_rules.STOP_V_PATH_REPRO and out["sentence"].startswith(
        "V 편집 경로가 조합 엔진에서 KC 입력 블록의 V 세트 냄새 값을 재현하지 못했다(")


def test_kc_band_strength_stop_on_the_calibration_median(w):
    calib = list(w.ctx["calib"][0])
    m = w.scripted(kc={("kc_band", LEVER_V): {o: 0.2 for o in calib}})
    out = _to(w, m, ZScripted(), "kc_band").stage_kc_band()
    assert out["outcome"] == v_rules.STOP_STRENGTH_LEVER and out["sentence"].startswith(
        "조합 지렛대 아래에서 E-grid k2-norm s 1.0이 KC 유효 대역을 잃었다(냄새별 KC 활성 중앙값 0.2000 ∉ [0.03, 0.15])")


def test_even_repro_stop_leaves_l_unmeasured(w):
    w.r_even_plan["C"] = {k: v for k, v in list(w.r_even_plan["C"].items())[:6]}
    m = w.scripted()
    out = _to(w, m, ZScripted(), "even").stage_even()
    assert out["outcome"] == v_rules.STOP_EVEN_REPRO and out["sentence"] == (
        "R 짝수 원자료가 h4 z에서 관문 ③ 값을 재현하지 못했다(L 16, C 6).")
    assert not [c for c in m.calls if c[0] == "oracle" and c[1] == "even"]


def test_even_punish_filter_boundaries(w):
    ea = w.keys(w.even, "a")
    m = w.scripted(plan={("even", LEVER_V): {**{k: (True, True, False) for k in w.keys(w.even, "b")[:13]},
                                            **{k: (False, False, False) for k in ea[:2]}}})
    out = _to(w, m, ZScripted(), "even").stage_even()
    assert out["outcome"] == v_rules.STOP_EVEN_PUNISH and out["sentence"] == (
        "조합 지렛대가 짝수 쌍에서 처벌 순감소 거름 기준((b) < 3, (a) ≤ 1)을 넘었다((b) 0, (a) 2). (POOL 안 짝수 쌍 조건부)")
    with pytest.raises(SystemExit):
        w.runner(w.scripted())._require("smoke")                         # STOP blocks every later stage


def test_even_punish_filter_passes_at_b_2_and_a_1(w):
    eb, ea = w.keys(w.even, "b"), w.keys(w.even, "a")
    m = w.scripted(plan={("even", LEVER_V): {**{k: (True, True, False) for k in eb[:13]},
                                            **{k: (True, False, False) for k in eb[19:21]},
                                            **{k: (False, False, False) for k in ea[:1]}}})
    out = _to(w, m, ZScripted(), "even").stage_even()
    assert out["outcome"] == "PASS" and (out["record"]["drops"]["b"]["net_drop"],
                                         out["record"]["drops"]["a"]["net_drop"]) == (2, 1)


def test_even_low_lever_sentence(w):
    m = w.scripted(plan={("even", LEVER_V): {k: (True, True, False) for k in w.keys(w.even, "b")[:10]}})
    out = _to(w, m, ZScripted(), "even").stage_even()
    assert out["outcome"] == v_rules.STOP_EVEN_LOW_LEVER and out["sentence"] == (
        "조합 지렛대 아래 짝수 (b) 21쌍에서 testable_b가 M2 기준(11)에 못 미쳤다(10/21, 지렛대 없는 같은 실행 7/21, F_a 기록 "
        "0/18). (POOL 안 짝수 쌍 조건부)")


def test_even_invalid_on_wrong_edges(w):
    m = w.scripted(override={("even", "L"): dict(edges=3)})
    assert _to(w, m, ZScripted(), "even").stage_even()["outcome"] == v_rules.INVALID


def test_refusals_own_block_order_and_dirty(w, monkeypatch):
    m = w.scripted()
    r = _to(w, m, ZScripted(), "kc_input")
    with pytest.raises(SystemExit):
        r.stage_path()                                                   # own block exists
    with pytest.raises(SystemExit):
        r.stage_set()                                                    # kc_input missing
    monkeypatch.setattr(VR, "summary_git", lambda p: dict(tracked=True, dirty=True, judge_commits=[]))
    with pytest.raises(SystemExit):
        r.stage_kc_input()
    monkeypatch.setattr(VR, "summary_git", lambda p: dict(tracked=True, dirty=False, judge_commits=[]))
    monkeypatch.setattr(VR, "git_state", lambda: dict(commit="c" * 40, dirty_hashed=["flymon/brain/v_rules.py"],
                                                      dirty_other=[]))
    with pytest.raises(SystemExit):
        r.stage_kc_input()


def test_a_set_that_no_longer_matches_block_set_refuses_kc_band(w):
    r = _to(w, w.scripted(), ZScripted(), "kc_band")
    d = doc()
    d["set"]["set"]["digest_keys"] = "0" * 64
    Path(SPEC.summary).write_text(json.dumps(d))
    with pytest.raises(SystemExit) as e:
        r.stage_kc_band()
    assert e.value.code == 2 and "kc_band" not in doc()
