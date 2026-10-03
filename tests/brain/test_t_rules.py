"""T.1-T.4 / T.6 / T.7 as replaced by T.9: every cell n, c ∈ 0..21 × F_a ∈ 0..43 × G_fail_S reads exactly one band
(R.3's order, r_rules.read_band itself) and G_fail_S / gate ② are S's functions; the reuse gate (R's repro and
gate ①), the set outcome, the z gates in their order (STOP_Z_REPRO on any bit of difference from block h4's z, then
STOP_Z_DEGENERATE on H.4's guard at its boundaries 5 / 0.25 or a zero SD), the gate-① supplement on the T set's
odours, gate ③ with R's C reproduction first; the sentences verbatim; the independent OC equal to S.6's values with
T's notes; the cluster-correlated OC equal to its committed fixture and to the values recorded here."""
import dataclasses
import itertools
import json
from pathlib import Path

import pytest

from flymon.brain import r_rules, s_rules
from flymon.brain import t_records as TR
from flymon.brain import t_rules as R
from flymon.brain.h3_store import canonical
from flymon.brain.t_spec import SPEC
from tests.brain.t_fixtures import SHA_L, SHA_NONE, counts, ref_rows, rest_rows

ROOT = Path(__file__).resolve().parents[2]
READOUT = {"A": "MBON13", "P": "MBON05"}


def _ref_band(n, c, f, g, nb=21, na=43):
    lines = [(nb != 21 or na != 43, R.NOT_READ), (c >= 11, R.B_NC), (n <= c, R.B_TB), (n < 11, R.B_NC),
             (n - c < 2, R.B_NC), (g, R.B_PG), (f < 2, R.B_FA), (True, R.SELECTED)]
    return next(b for cond, b in lines if cond)


def test_every_cell_reads_exactly_the_transcribed_band():
    seen = set()
    for n, c, f, g in itertools.product(range(22), range(22), range(44), (False, True)):
        b = R.read_band(n, c, f, g, 21, 43, f, SPEC)["band"]
        assert b == _ref_band(n, c, f, g), (n, c, f, g)
        seen.add(b)
    assert seen == set(R.BANDS)
    assert R.read_band is r_rules.read_band and R.g_fail_s is s_rules.g_fail_s and R.gate2 is s_rules.gate2


@pytest.mark.parametrize("n,c,f,g,band", [
    (13, 10, 2, False, R.SELECTED), (13, 11, 2, False, R.B_NC), (8, 8, 5, False, R.B_TB), (10, 8, 5, False, R.B_NC),
    (11, 8, 5, False, R.SELECTED), (11, 10, 5, False, R.B_NC), (12, 10, 5, False, R.SELECTED),
    (13, 8, 1, False, R.B_FA), (13, 8, 2, False, R.SELECTED), (13, 8, 1, True, R.B_PG), (11, 10, 5, True, R.B_NC)])
def test_boundaries(n, c, f, g, band):
    assert R.read_band(n, c, f, g, 21, 43, max(f, 4), SPEC)["band"] == band


@pytest.mark.parametrize("nb,na", [(20, 43), (22, 43), (21, 42), (21, 44), (21, 32)])
def test_counts_other_than_21_43_are_not_read(nb, na):
    assert R.read_band(13, 8, 5, False, nb, na, 5, SPEC)["band"] == R.NOT_READ


def test_net_drop_2_vs_3_on_t():
    cp = [dict(key=f"a|{i}", axis="a", punish_pass=i < 6) for i in range(12)]
    lp2 = [dict(key=f"a|{i}", axis="a", punish_pass=i < 4) for i in range(12)]
    lp3 = [dict(key=f"a|{i}", axis="a", punish_pass=i < 3) for i in range(12)]
    assert R.g_fail_s(lp2, cp, SPEC)["g_fail"] is False and R.g_fail_s(lp3, cp, SPEC)["g_fail"] is True


def _blk(**kw):
    return dict(dict(code_key=SPEC.r_shared_key), **kw)


R_DOC = dict(repro=_blk(passed=True), gate1=_blk(outcome="PASS"))
CLEAN = dict(tracked=True, dirty=False)


def test_reuse_needs_rs_repro_and_gate1_only():
    assert R.reuse(R_DOC, CLEAN, SPEC.r_shared_key, SPEC)["outcome"] == R.PASS
    out = R.reuse(R_DOC, CLEAN, "f" * 64, SPEC)
    assert out["outcome"] == R.STOP_REUSE and "공유 측정 키" in out["sentence"] and "T는 R 관문을" in out["sentence"]
    assert R.reuse(R_DOC, dict(tracked=True, dirty=True), SPEC.r_shared_key, SPEC)["outcome"] == R.STOP_REUSE
    for b, bad in (("repro", _blk(passed=False)), ("gate1", _blk(outcome="STOP_STRENGTH_LEVER")),
                   ("gate1", _blk(outcome="PASS", code_key="x")), ("repro", None)):
        doc = dict(R_DOC, **{b: bad}) if bad is not None else {k: v for k, v in R_DOC.items() if k != b}
        out = R.reuse(doc, CLEAN, SPEC.r_shared_key, SPEC)
        assert out["outcome"] == R.STOP_REUSE and any(b in w for w in out["reasons"]), b


def test_set_outcome():
    assert R.set_outcome(dict(status="OK", n_b=21, n_a=43))["outcome"] == R.PASS
    out = R.set_outcome(dict(status=R.STOP_SET_SHORT, n_b=21, n_a=40))
    assert out["outcome"] == R.STOP_SET_SHORT
    assert out["sentence"] == "넓힌 상대 풀 생성원 턴 0–1985에서 (b) 21·(a) 43을 채우지 못했다(21·40쌍)."


def _side(a, p, rest=0, edges=0, sha=SHA_NONE):
    return TR.z_side(ref_rows(a, p, edges, sha), rest_rows(len(a), rest, edges, sha), READOUT, SPEC)


def test_z_repro_needs_every_bit_of_h4s_z():
    side = _side(counts(96, 8), counts(96, 20))
    h4 = SPEC.z_h4_dict()
    exact = dict(side, z={k: [v[0], v[1]] for k, v in h4.items()})
    assert R.z_repro(exact, 96, SHA_NONE, SPEC)["outcome"] == R.PASS
    off = dict(side, z=dict(exact["z"], A=[h4["A"][0], h4["A"][1] + 1e-15]))
    out = R.z_repro(off, 96, SHA_NONE, SPEC)
    assert out["outcome"] == R.STOP_Z_REPRO
    assert out["sentence"].startswith("편집 없는 엔진에서 기준 집합 z가 블록 h4 값을 재현하지 못했다(A 10.78125 / ")
    assert out["sentence"].endswith(") — z 절차 결함.")
    none_z = dict(exact, z=None, why="readout MBON13: zero SD")
    assert R.z_repro(none_z, 96, SHA_NONE, SPEC)["outcome"] == R.STOP_Z_REPRO
    assert R.z_repro(exact, 96, "sha-other", SPEC)["outcome"] == R.INVALID
    assert R.z_repro(dict(exact, edit_edges=[2]), 96, SHA_NONE, SPEC)["outcome"] == R.INVALID
    assert R.z_repro(dict(exact, n_rest=95), 96, SHA_NONE, SPEC)["outcome"] == R.INVALID


@pytest.mark.parametrize("a,p,outcome,failed", [
    (counts(96, 5), counts(96, 30), R.PASS, []),                          # MBON13 median Δ 7 (5..9)
    (counts(96, 3), counts(96, 30), R.PASS, []),                          # median Δ exactly 5
    (counts(96, 2), counts(96, 30), R.STOP_Z_DEGENERATE, ["MBON13"]),     # median Δ 4
    (counts(96, 10, zeros=24), counts(96, 30), R.PASS, []),               # zero share exactly 0.25
    (counts(96, 10, zeros=25), counts(96, 30), R.STOP_Z_DEGENERATE, ["MBON13"]),
    (counts(96, 10), [0] * 96, R.STOP_Z_DEGENERATE, ["MBON05"]),          # silent P type: guard and zero SD
])
def test_z_lever_guard_boundaries(a, p, outcome, failed):
    side = _side(a, p, edges=2, sha=SHA_L)
    out = R.z_lever(side, 96, READOUT, SPEC)
    assert out["outcome"] == outcome and out.get("failed", []) == failed


def test_z_lever_zero_sd_and_sentence():
    side = _side([0] * 96, counts(96, 30), edges=2, sha=SHA_L)
    assert side["z"] is None and "zero SD" in side["why"]
    out = R.z_lever(side, 96, READOUT, SPEC)
    assert out["outcome"] == R.STOP_Z_DEGENERATE
    assert out["sentence"] == ("APL→MBON05 제거 아래 기준 집합에서 판독 MBON13이 반응성 가드를 넘지 못했다(Δ 중앙값 0.0, "
                               "0 비율 1.000) — z를 정할 수 없다.")
    assert R.z_lever(dict(side, edit_edges=[0]), 96, READOUT, SPEC)["outcome"] == R.INVALID


def _act(per: dict, edges: int, sha: str, n_seeds: int = 8) -> dict:
    return {o: dict(frac=[v] * n_seeds, max_win=[3] * n_seeds, edit_edges=[edges], csc_sha256=[sha])
            for o, v in per.items()}


def test_gate1s_band_and_sentence():
    ods = {f"M{i}|T": 0.05 for i in range(55)}
    rec = TR.gate1s_record(_act(ods, 2, SHA_L), _act(ods, 0, SHA_NONE), SPEC)
    assert R.gate1s(rec, SPEC)["outcome"] == R.PASS
    edge = dict(ods, **{"M0|T": 0.03, "M1|T": 0.15})
    assert R.gate1s(TR.gate1s_record(_act(edge, 2, SHA_L), _act(ods, 0, SHA_NONE), SPEC), SPEC)["outcome"] == R.PASS
    bad = dict(ods, **{"M3|T": 0.029, "M4|T": 0.2})
    out = R.gate1s(TR.gate1s_record(_act(bad, 2, SHA_L), _act(ods, 0, SHA_NONE), SPEC), SPEC)
    assert out["outcome"] == R.STOP_STRENGTH_LEVER and out["failed"] == ["M3|T", "M4|T"]
    assert out["sentence"] == ("APL→MBON05 제거 아래에서 E-grid k2-norm s 1.0이 KC 활성 자격(T 세트 냄새 M3|T 0.0290; "
                               "M4|T 0.2000 ∉ [0.03, 0.15])을 잃었다.")
    assert R.gate1s(TR.gate1s_record(_act(ods, 1, SHA_L), _act(ods, 0, SHA_NONE), SPEC), SPEC)["outcome"] == R.INVALID
    few = {o: v for o, v in list(ods.items())[:54]}
    assert R.gate1s(TR.gate1s_record(_act(few, 2, SHA_L), _act(few, 0, SHA_NONE), SPEC), SPEC)["outcome"] == R.INVALID
    assert R.gate1s(TR.gate1s_record(_act(ods, 2, SHA_L, 7), _act(ods, 0, SHA_NONE, 7), SPEC),
                    SPEC)["outcome"] == R.INVALID


def _summ(tb, edges, sha, reasons=()):
    return dict(reasons=list(reasons), edit_edges=[edges], csc_sha256=sha,
                aggregate=dict(testable_b=tb, F_a=1, naive_a=2, n_b=21, n_a=18))


def test_gate3_order():
    L, C = _summ(12, 2, SHA_L), _summ(7, 0, SHA_NONE)
    assert R.gate3(L, C, SPEC, SHA_NONE, 16)["outcome"] == R.PASS
    out = R.gate3(None, _summ(6, 0, SHA_NONE), SPEC, SHA_NONE, 16)
    assert out["outcome"] == R.STOP_EVEN_REPRO and out["sentence"] == ("R 짝수 원자료가 h4 z에서 관문 ③ 값을 재현하지 "
                                                                     "못했다(L 16, C 6).")
    assert R.gate3(L, _summ(7, 0, SHA_NONE, ["x"]), SPEC, SHA_NONE, 16)["outcome"] == R.STOP_EVEN_REPRO
    out = R.gate3(_summ(10, 2, SHA_L), C, SPEC, SHA_NONE, 16)
    assert out["outcome"] == R.STOP_EVEN_LOW_LEVER and "(10/21, 지렛대 없는 같은 실행 7/21, F_a 기록 1/18)" in out["sentence"]
    assert R.gate3(_summ(11, 2, SHA_L), C, SPEC, SHA_NONE, 16)["outcome"] == R.PASS
    assert R.gate3(_summ(12, 1, SHA_L), C, SPEC, SHA_NONE, 16)["outcome"] == R.INVALID


F = dict(n=14, c=8, f_a=3, naive_a=4, T=103, k_even=12, pb_L=15, pb_C=18, pa_L=20, pa_C=22, d_b=3, d_a=2,
         rho1="0.700", rho2="0.900", why="x", b=21, a=40, val="A 1 / 2", type="MBON13", med="3.0", zero="0.250",
         cond="x", l=16, tb=10, c_even=7, k=1, label="NO_LEARNING", l1="0.100", l2="0.200", l1L="1", l1C="2", l2L="3",
         l2C="4")


def test_sentences_verbatim():
    for o in (R.STOP_REUSE, R.STOP_SET_SHORT, R.STOP_Z_REPRO, R.STOP_Z_DEGENERATE, R.STOP_STRENGTH_LEVER,
              R.STOP_EVEN_REPRO, R.STOP_EVEN_LOW_LEVER, R.STOP_C_EVEN_MISMATCH, R.STOP_PUNISH_BROKEN,
              R.STOP_P_REFERENCE, R.STOP_PUNISH_WEAKENED, R.B_TB, R.B_PG, R.B_FA, R.SELECTED):
        assert "{" not in R.sentence(o, F), o
    assert R.sentence(R.B_TB, F) == (
        "APL→MBON05 제거가 넓힌 상대 풀 판정 세트(생성원 턴 0–103)에서 14/21로 지렛대 없는 같은 세트 8/21보다 오르지 않았다"
        "(엔진마다 자기 기준 집합 z). → 넓힌 풀·엔진별 z에서의 T 주장을 닫는다. 지렛대 전체를 닫지 않는다(POOL 안 결과는 "
        "R·S 그대로).")
    assert R.sentence(R.B_NC, dict(F, reason=R.MARGIN)) == (
        "기준은 넘었으나 지렛대 없는 쪽 대비 여유가 2 미만이다 (14/21 대 8/21, F_a 3/43)")
    assert R.sentence(R.B_NC, dict(F, reason=R.C_ABOVE_BAR)).endswith(" (14/21 대 8/21, F_a 3/43)")
    assert R.sentence(R.B_PG, F) == s_rules.sentence(s_rules.B_PG, F)
    assert R.sentence(R.B_FA, F) == ("M2 (b) 기준·여유·처벌 가드는 넘었지만 F_a가 기준에 못 미쳤다(14/21 대 8/21, F_a 3/43, "
                                     "naive_a 4). 다음 병목은 F_a(순진 균형 (a) 쌍)다.")
    assert R.sentence(R.SELECTED, F) == (
        "C3에서 APL→MBON05 2간선을 지운 모델 변형과 E-grid k2-norm(s 1.0)에서, 엔진 변형마다 자기 기준 집합 z로 읽었을 때, "
        "판정 세트(상대를 1세대 기본 폼으로 넓힌 풀의 생성원 턴 0–103, 기존 키·사구체 중복 제외, 모든 행이 POOL 밖 상대 타입 "
        "조합 포함) (b) 21쌍의 H.4 오라클 시험 가능성이 M2 기준을 만족했고 지렛대 없는 같은 세트보다 높았다(14/21 대 8/21, "
        "여유 ≥ 2, F_a 3/43, 처벌 순감소 가드(축마다 ≥ 3) 통과, 같은 시드 P 비 ℓ_r1 0.700·ℓ_r2 0.900 ≥ 0.5(약화 정도 기록), "
        "짝수 재판독 12/21, 판정 시드 24_400_xxx). 판정 세트의 상대는 1세대 기본 폼으로 넓힌 풀이며, 새 키는 POOL에 없는 "
        "상대 타입 조합(12개 중)에서 나온다 — 과제 풀 안의 시험 가능성은 R·S가 마지막이다. (b) 21쌍은 9개 상대 타입 조합 "
        "쌍에 몰려 있어 쌍끼리 독립이 아니다. 지렛대는 Q 결과를 보고 골랐고, z 규칙은 R·S 판정 결과를 본 뒤 정했다(T.0). "
        "이 판정은 R(뒤에 가드 변경)·S(뒤에 z 규칙)에 이은 같은 지렛대의 세 번째 판정이다. 작동 특성은 T 세트 조건부 값이며 "
        "Q → R → S → T 전체 절차의 오선택률이 아니다. 실제 커넥톰 간선을 지운 모델이며 오라클 시험 가능성이지 학습 시험이 "
        "아니다. 귀결은 넓힌 풀에서 엔진별 z로 M2 시험 가능성이 섰다는 것까지이며, POOL 배틀 과제의 F v4 학습 시험을 "
        "이것만으로 정당화하지 않는다.")
    with pytest.raises(KeyError):
        R.sentence(R.B_FA, {})


DECLARED_S6 = {("null", 0.02, 0.02): 0.034, ("null", 0.05, 0.05): 0.145, ("null", 0.1, 0.1): 0.281,
               ("null", 0.15, 0.15): 0.362, ("harm", 0.1, 0.02): 0.752, ("harm", 0.15, 0.02): 0.949,
               ("harm", 0.2, 0.05): 0.961, ("harm", 0.1, 0.05): 0.549}


def test_independent_oc_is_s6s_with_ts_notes():
    oc = R.oc(SPEC)
    got = {(g["kind"], g["a_pf"], g["a_fp"]): round(g["p"], 3) for g in oc["g_fail"]}
    assert got == DECLARED_S6
    assert len(oc["notes"]) == 3 and "9개 상대 타입 조합 쌍" in oc["notes"][1] and "Q → R → S → T" in oc["notes"][2]
    assert oc["sha256"] == R.oc(SPEC)["sha256"] and len(oc["rows"]) == 4 * 3 * 13 * 9


# Recorded while planning (T.9.7: the fixture is the declaration; these lines name its headline values).
CLUSTER_G = {("null", 0.1, 0.1): 0.401, ("harm", 0.1, 0.02): 0.685, ("harm", 0.1, 0.05): 0.571}
CLUSTER_P_N_GE_11 = {"0.33": 0.097, "0.5": 0.498, "0.6": 0.776, "0.7": 0.939}


@pytest.fixture(scope="module")
def cluster():
    return R.oc_cluster(SPEC)


def test_cluster_oc_equals_its_fixture(cluster):
    fx = json.loads((ROOT / SPEC.oc_cluster_fixture).read_text())
    assert canonical(fx) == canonical(cluster) and fx["sha256"] == cluster["sha256"]


def test_cluster_oc_headline_values(cluster):
    assert (cluster["seed"], cluster["draws"], cluster["icc"]) == (20261005, 100_000, 0.3)
    assert cluster["clusters_b"] == [1, 3, 3, 3, 3, 1, 1, 2, 4] and cluster["clusters_a"] == [7, 8, 2, 1, 5, 9, 9, 2]
    got = {(g["kind"], g["a_pf"], g["a_fp"]): round(g["p"], 3) for g in cluster["g_fail"]}
    assert got == CLUSTER_G
    assert {(g["kind"], g["a_pf"], g["a_fp"]): round(g["p_independent"], 3) for g in cluster["g_fail"]} == {
        k: DECLARED_S6[k] for k in CLUSTER_G}
    for q, d in cluster["n_dist"].items():
        assert sum(d["cluster"]) == pytest.approx(1.0) and round(sum(d["cluster"][11:]), 3) == CLUSTER_P_N_GE_11[q]
    assert len(cluster["rows"]) == 4 * 3 * 13 * 4
    for row in cluster["rows"][::37]:
        assert sum(row["P"].values()) == pytest.approx(1.0) and set(row["P"]) == set(R.BANDS)
    assert all(r["P"][R.B_PG] == 0 for r in cluster["rows"] if r["g_kind"] == "none")


def test_cluster_oc_depends_on_icc_and_seed():
    small = dataclasses.replace(SPEC, oc_cluster_draws=2000)
    a, b = R.oc_cluster(small), R.oc_cluster(dataclasses.replace(small, oc_cluster_seed=1))
    assert a["sha256"] != b["sha256"] and a["sha256"] == R.oc_cluster(small)["sha256"]
    lo = R.oc_cluster(dataclasses.replace(small, oc_cluster_icc=0.01))
    assert abs(lo["g_fail"][0]["p"] - lo["g_fail"][0]["p_independent"]) < 0.05


def test_t_rules_literals_are_only_0_and_1():
    import ast
    src = (ROOT / "flymon/brain/t_rules.py").read_text()
    nums = {n.value for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Constant) and type(n.value) in (int, float)}
    assert nums <= {0, 1}, nums


def test_z_lever_stops_when_z_is_none_without_a_failing_guard():
    side = _side(counts(96, 5), counts(96, 30), edges=2, sha=SHA_L)
    out = R.z_lever(dict(side, z=None, why="readout MBON05: z undefined"), 96, READOUT, SPEC)
    assert out["outcome"] == R.STOP_Z_DEGENERATE and out["failed"] == ["MBON05"]
    assert out["reasons"] == ["readout MBON05: z undefined"]
    out = R.z_lever(dict(side, z=None, why="no counts"), 96, READOUT, SPEC)
    assert out["outcome"] == R.STOP_Z_DEGENERATE and out["failed"] == list(READOUT.values())


def test_gate3_refuses_a_missing_l_after_c_reproduced():
    with pytest.raises(ValueError, match="L was not measured"):
        R.gate3(None, _summ(7, 0, SHA_NONE), SPEC, SHA_NONE, 16)
