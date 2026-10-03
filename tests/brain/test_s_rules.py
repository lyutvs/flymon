# tests/brain/test_s_rules.py
"""S.2 / S.3 / S.4 / S.6 / S.7 / S.9: every cell n, c ∈ 0..21 × F_a ∈ 0..43 × G_fail_S reads exactly one band, equal
to an independent transcription of S.4's ordered lines; the boundary fixtures (c 10/11, n = c / c+1 / c+2, n 10/11,
F_a 1/2, net drop 2/3 on each axis, a symmetric swap 3/3 -> no G_fail, counts 20/22 and 42/44); the reuse gate, the
set outcome, gate ②'s order (INVALID, L, C, ratio 0.5 per direction); the sentences verbatim; the OC exact and equal
to S.6's declared values beside R's guard on the same pair counts."""
import dataclasses
import itertools

import pytest

from flymon.brain import r_rules
from flymon.brain import s_rules as R
from flymon.brain.s_spec import SPEC


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
    assert seen == set(R.BANDS) and R.read_band is r_rules.read_band


@pytest.mark.parametrize("n,c,f,g,band,reason", [
    (13, 10, 2, False, R.SELECTED, "PASS"),
    (13, 11, 2, False, R.B_NC, R.C_ABOVE_BAR),            # c 10 / 11
    (8, 8, 5, False, R.B_TB, "NO_GAIN"),                   # n = c
    (9, 8, 5, False, R.B_NC, R.BELOW_BAR),                 # n = c + 1, below 11
    (10, 8, 5, False, R.B_NC, R.BELOW_BAR),                # n = 10
    (11, 8, 5, False, R.SELECTED, "PASS"),                 # n = 11
    (11, 10, 5, False, R.B_NC, R.MARGIN),                  # n = c + 1 at the bar
    (12, 10, 5, False, R.SELECTED, "PASS"),                # n = c + 2
    (13, 8, 1, False, R.B_FA, "F_A"),                      # F_a 1
    (13, 8, 2, False, R.SELECTED, "PASS"),                 # F_a 2
    (13, 8, 1, True, R.B_PG, "PUNISH_GUARD"),              # G_fail_S before F_a
    (11, 10, 5, True, R.B_NC, R.MARGIN),                   # margin before G_fail_S
])
def test_boundaries(n, c, f, g, band, reason):
    out = R.read_band(n, c, f, g, 21, 43, max(f, 4), SPEC)
    assert (out["band"], out["reason"]) == (band, reason)


@pytest.mark.parametrize("nb,na", [(20, 43), (22, 43), (21, 42), (21, 44), (21, 32)])
def test_counts_other_than_21_43_are_not_read(nb, na):
    assert R.read_band(13, 8, 5, False, nb, na, 5, SPEC)["band"] == R.NOT_READ


def _pp(ax, c_pass, l_pass, n=12):
    cp = [dict(key=f"{ax}|{i}", axis=ax, punish_pass=i in c_pass) for i in range(n)]
    lp = [dict(key=f"{ax}|{i}", axis=ax, punish_pass=i in l_pass) for i in range(n)]
    return lp, cp


@pytest.mark.parametrize("ax", ["b", "a"])
def test_net_drop_2_vs_3(ax):
    lp, cp = _pp(ax, set(range(6)), set(range(4)))           # 6 -> 4: net drop 2
    out = R.g_fail_s(lp, cp, SPEC)
    assert out["g_fail"] is False and out["axes"][ax]["net_drop"] == 2
    lp, cp = _pp(ax, set(range(6)), set(range(3)))           # 6 -> 3: net drop 3
    out = R.g_fail_s(lp, cp, SPEC)
    assert out["g_fail"] is True and out["axes"][ax]["fail"] and out["axes"][ax]["net_drop"] == 3
    other = "a" if ax == "b" else "b"
    assert out["axes"][other]["n"] == 0 and out["axes"][other]["fail"] is False


@pytest.mark.parametrize("ax", ["b", "a"])
def test_a_symmetric_swap_never_fails(ax):
    lp, cp = _pp(ax, set(range(5)), {3, 4, 5, 6, 7})         # pass->fail 3, fail->pass 3 (R's (a) at R.10)
    out = R.g_fail_s(lp, cp, SPEC)["axes"][ax]
    assert (out["pass_to_fail"], out["fail_to_pass"], out["net_drop"], out["fail"]) == (3, 3, 0, False)
    lp, cp = _pp(ax, set(range(10)), set(range(2, 10)) | {10, 11}, n=20)   # 2 -> fail, 2 -> pass: big swap, drop 0
    assert R.g_fail_s(lp, cp, SPEC)["g_fail"] is False


def test_only_pairs_in_both_conditions_count():
    lp, cp = _pp("a", set(range(6)), set())
    out = R.g_fail_s(lp[:2], cp, SPEC)["axes"]["a"]
    assert (out["n"], out["pun_C"], out["net_drop"]) == (2, 2, 2)


def _blk(**kw):
    return dict(dict(code_key=SPEC.r_shared_key), **kw)


R_DOC = dict(repro=_blk(passed=True), gate1=_blk(outcome="PASS"), gate3=_blk(outcome="PASS"))
CLEAN = dict(tracked=True, dirty=False)


def test_reuse():
    assert R.reuse(R_DOC, CLEAN, SPEC.r_shared_key, SPEC)["outcome"] == R.PASS
    out = R.reuse(R_DOC, CLEAN, "f" * 64, SPEC)
    assert out["outcome"] == R.STOP_REUSE and "공유 측정 키" in out["sentence"]
    assert R.reuse(R_DOC, dict(tracked=True, dirty=True), SPEC.r_shared_key, SPEC)["outcome"] == R.STOP_REUSE
    assert R.reuse(R_DOC, dict(tracked=False, dirty=False), SPEC.r_shared_key, SPEC)["outcome"] == R.STOP_REUSE
    for b, bad in (("repro", _blk(passed=False)), ("gate1", _blk(outcome="STOP_STRENGTH_LEVER")),
                   ("gate3", _blk(outcome="PASS", code_key="x")), ("gate1", None)):
        doc = dict(R_DOC, **{b: bad}) if bad is not None else {k: v for k, v in R_DOC.items() if k != b}
        out = R.reuse(doc, CLEAN, SPEC.r_shared_key, SPEC)
        assert out["outcome"] == R.STOP_REUSE and any(b in w for w in out["reasons"]), b


def test_set_outcome():
    assert R.set_outcome(dict(status="OK", n_b=21))["outcome"] == R.PASS
    out = R.set_outcome(dict(status=R.STOP_SET_SHORT, n_b=19))
    assert out["outcome"] == R.STOP_SET_SHORT
    assert out["sentence"] == "L 생성기 턴 104–209에서 E-grid 키 중복을 뺀 (b) 쌍이 21개에 못 미쳤다(19쌍)."


def _res(label="LEARNS_CONFIRMATORY", l1=1.8, l2=2.2):
    return dict(outcome="JUDGED", label=label, reasons=[], directions={"r1": {"ell": l1}, "r2": {"ell": l2}})


def _ratio(r1, r2):
    return {"r1": dict(ratio=r1), "r2": dict(ratio=r2)}


def test_gate2_order_and_ratio_boundary():
    assert R.gate2(_res(), _res(), _ratio(0.5, 0.5), SPEC)["outcome"] == R.PASS          # 0.5 passes
    out = R.gate2(_res(l1=0.89), _res(l1=1.8), _ratio(0.4999, 0.9), SPEC)
    assert out["outcome"] == R.STOP_PUNISH_WEAKENED and out["low"] == ["r1"]
    assert out["sentence"] == ("APL→MBON05 제거 아래 같은 시드의 처벌 학습량이 지렛대 없는 쪽의 절반에 못 미쳤다"
                               "(ℓ_r1 0.890 대 1.800, ℓ_r2 2.200 대 2.200).")
    assert R.gate2(_res(), _res(), _ratio(0.9, 0.49), SPEC)["low"] == ["r2"]
    out = R.gate2(_res("NO_LEARNING", 0.1, 0.2), _res(), _ratio(0.1, 0.1), SPEC)
    assert out["outcome"] == R.STOP_PUNISH_BROKEN                                        # L before C and ratio
    assert out["sentence"] == ("APL→MBON05 제거 아래에서 P의 처벌 학습 확인이 재현되지 않았다"
                               "(NO_LEARNING, ℓ_r1 0.100, ℓ_r2 0.200).")
    out = R.gate2(_res(), _res("DIRECTION_DEPENDENT"), _ratio(0.1, 0.1), SPEC)
    assert out["outcome"] == R.STOP_P_REFERENCE                                          # C before ratio
    assert out["sentence"] == ("지렛대 없는 P가 새 시드 블록에서 처벌 학습 확인을 재현하지 못했다(DIRECTION_DEPENDENT) — "
                               "비교 기준이 없다.")
    inv = dict(outcome="INVALID", label="INVALID", reasons=["broken"])
    assert R.gate2(inv, _res(), None, SPEC)["outcome"] == R.INVALID                      # INVALID first
    assert R.gate2(_res("NO_LEARNING"), inv, None, SPEC)["reasons"] == ["C: broken"]


F = dict(n=14, c=8, f_a=3, naive_a=4, T=177, k_even=16, pb_L=15, pb_C=18, pa_L=20, pa_C=22, d_b=3, d_a=2,
         rho1="0.700", rho2="0.900", why="x", m=19, label="NO_LEARNING", l1="0.100", l2="0.200", l1L="1", l1C="2",
         l2L="3", l2C="4")


def test_sentences_verbatim():
    for o in (R.STOP_REUSE, R.STOP_SET_SHORT, R.STOP_PUNISH_BROKEN, R.STOP_P_REFERENCE, R.STOP_PUNISH_WEAKENED,
              R.B_TB, R.B_PG, R.B_FA, R.SELECTED):
        assert "{" not in R.sentence(o, F), o
    assert R.sentence(R.B_TB, F) == ("APL→MBON05 제거가 마지막 판정 세트(L 생성기 턴 104–177)에서 14/21로 지렛대 없는 같은 "
                                     "세트 8/21보다 오르지 않았다. → 이 지렛대를 닫는다.")
    assert R.sentence(R.B_NC, dict(F, reason=R.C_ABOVE_BAR)) == (
        "지렛대 없는 E-grid가 이 세트에서 이미 기준을 넘어 지렛대 효과로 말할 수 없다. (14/21 대 8/21, F_a 3/43)")
    assert R.sentence(R.B_NC, dict(F, reason=R.BELOW_BAR)) == (
        "지렛대 없는 쪽보다 올랐지만 M2 기준에 못 미쳤다. (14/21 대 8/21, F_a 3/43)")
    assert R.sentence(R.B_NC, dict(F, reason=R.MARGIN)) == (
        "기준은 넘었으나 지렛대 없는 쪽 대비 여유가 2 미만이다 (14/21 대 8/21, F_a 3/43)")
    assert R.sentence(R.B_PG, F) == ("M2 (b) 기준과 여유는 넘었지만 지렛대 아래 처벌 통과가 순감소했다(처벌 통과 (b) 15 대 18, "
                                     "(a) 20 대 22; 순감소 (b) 3, (a) 2).")
    assert R.sentence(R.B_FA, F) == ("M2 (b) 기준·여유·처벌 가드는 넘었지만 F_a가 기준에 못 미쳤다(14/21 대 8/21, F_a 3/43, "
                                     "naive_a 4). 다음 병목은 F_a(순진 균형 (a) 쌍)다.")
    assert R.sentence(R.SELECTED, F) == (
        "C3에서 APL→MBON05 2간선을 지운 모델 변형과 E-grid k2-norm(s 1.0)에서, 마지막 판정 세트(L 생성기 턴 104–177, "
        "E-grid 키 중복 제외) (b) 21쌍의 H.4 오라클 시험 가능성이 M2 기준을 만족했고 지렛대 없는 같은 세트보다 높았다"
        "(14/21 대 8/21, 여유 ≥ 2, F_a 3/43, 처벌 순감소 가드(축마다 ≥ 3) 통과, 같은 시드 P 비 ℓ_r1 0.700·ℓ_r2 0.900 ≥ "
        "0.5(약화 정도 기록), 짝수 16/21, 판정 시드 24_300_xxx). 지렛대는 Q 결과를 보고 골랐고, 이 판정은 R 판정 결과를 본 뒤 "
        "가드를 바꾼 두 번째 판정이다(S.0). 작동 특성은 S 세트 조건부 값이며 Q → R → S 전체 절차의 오선택률이 아니다. "
        "실제 커넥톰 간선을 지운 모델이며 오라클 시험 가능성이지 학습 시험이 아니다.")
    with pytest.raises(KeyError):
        R.sentence(R.B_FA, {})


def test_g_axis_matches_brute_force_on_a_small_axis():
    a_pf, a_fp, tot = 0.2, 0.1, 0.0
    for states in itertools.product(("pf", "fp", "same"), repeat=5):
        w = 1.0
        for s in states:
            w *= {"pf": a_pf, "fp": a_fp, "same": 1 - a_pf - a_fp}[s]
        if states.count("pf") - states.count("fp") >= 3:
            tot += w
    assert R.g_axis(5, a_pf, a_fp, SPEC) == pytest.approx(tot)


DECLARED_S = {("null", 0.02, 0.02): 0.034, ("null", 0.05, 0.05): 0.145, ("null", 0.1, 0.1): 0.281,
              ("null", 0.15, 0.15): 0.362, ("harm", 0.1, 0.02): 0.752, ("harm", 0.15, 0.02): 0.949,
              ("harm", 0.2, 0.05): 0.961, ("harm", 0.1, 0.05): 0.549}
DECLARED_R = {("null", 0.02, 0.02): 0.163, ("null", 0.05, 0.05): 0.489, ("null", 0.1, 0.1): 0.888,
              ("null", 0.15, 0.15): 0.988, ("harm", 0.1, 0.02): 0.938, ("harm", 0.15, 0.02): 0.994,
              ("harm", 0.2, 0.05): 0.999, ("harm", 0.1, 0.05): 0.906}


def test_oc_reproduces_s6s_declared_values_and_rs_guard_row():
    oc = R.oc(SPEC)
    got = {(g["kind"], g["a_pf"], g["a_fp"]): g for g in oc["g_fail"]}
    assert set(got) == set(DECLARED_S)
    for k, v in DECLARED_S.items():
        assert round(got[k]["p"], 3) == v, k
        assert round(got[k]["r_guard"], 3) == DECLARED_R[k], k
    assert round(R.g_prob(0.1, 0.1, dataclasses.replace(SPEC, g_fail_drop=2)), 3) == 0.461    # S.9.1: ≥ 2 would be


def test_oc_rows_are_distributions():
    oc = R.oc(SPEC)
    n_na = min(SPEC.n_a, SPEC.oc_naive_max) + 1
    assert len(oc["rows"]) == len(SPEC.oc_q) * len(SPEC.oc_c) * n_na * (1 + len(oc["g_fail"]))
    for row in oc["rows"][::41]:
        assert sum(row["P"].values()) == pytest.approx(1.0) and set(row["P"]) == set(R.BANDS)
    assert all(r["P"][R.B_PG] == 0 for r in oc["rows"] if r["g_kind"] == "none")
    assert len(oc["notes"]) == 2 and "pun_C(b) − n ≥ 3" in oc["notes"][0] and "오선택률" in oc["notes"][1]
    assert len(oc["sha256"]) == 64 and oc["sha256"] == R.oc(SPEC)["sha256"]


def test_s_rules_literals_are_only_0_and_1():
    import ast
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2] / "flymon/brain/s_rules.py").read_text()
    nums = {n.value for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Constant) and type(n.value) in (int, float)}
    assert nums <= {0, 1}, nums
