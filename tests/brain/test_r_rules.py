# tests/brain/test_r_rules.py
"""R.3 / R.7 / R.9.2-R.9.5: the judgement code is the authoritative source. Every cell n, c ∈ 0..21 × F_a ∈ 0..32 ×
G_fail reads exactly one band, equal to an independent transcription of R.3's ordered lines; the boundary fixtures (c
10/11, n = c / c+1 / c+2, n 10/11, F_a 1/2, G_fail before F_a, counts 20/22 and 31/33, the R.9.4 counterexample);
G_fail's drop 1/2 and pass->fail 2/3 boundaries on either axis; the gates; the sentences verbatim; the OC exact."""
import itertools

import pytest

from flymon.brain import r_rules as R
from flymon.brain.r_spec import SPEC


def _ref_band(n, c, f, g, nb=21, na=32):
    lines = [(nb != 21 or na != 32, R.NOT_READ), (c >= 11, R.B_NC), (n <= c, R.B_TB), (n < 11, R.B_NC),
             (n - c < 2, R.B_NC), (g, R.B_PG), (f < 2, R.B_FA), (True, R.SELECTED)]
    return next(b for cond, b in lines if cond)


def test_every_cell_reads_exactly_the_transcribed_band():
    seen = set()
    for n, c, f, g in itertools.product(range(22), range(22), range(33), (False, True)):
        b = R.read_band(n, c, f, g, 21, 32, f, SPEC)["band"]
        assert b == _ref_band(n, c, f, g), (n, c, f, g)
        seen.add(b)
    assert seen == set(R.BANDS)


@pytest.mark.parametrize("n,c,f,g,band,reason", [
    (13, 10, 2, False, R.SELECTED, "PASS"),
    (13, 11, 2, False, R.B_NC, R.C_ABOVE_BAR),            # c 10 / 11
    (13, 11, 2, True, R.B_NC, R.C_ABOVE_BAR),             # R.9.4: c 11 · F_a_C 0 · n 13 · F_a 2 -> B_결론없음
    (8, 8, 5, False, R.B_TB, "NO_GAIN"),                   # n = c
    (9, 8, 5, False, R.B_NC, R.BELOW_BAR),                 # n = c + 1, below 11
    (10, 8, 5, False, R.B_NC, R.BELOW_BAR),                # n = 10
    (11, 8, 5, False, R.SELECTED, "PASS"),                 # n = 11
    (11, 10, 5, False, R.B_NC, R.MARGIN),                  # n = c + 1 at the bar
    (12, 10, 5, False, R.SELECTED, "PASS"),                # n = c + 2
    (13, 8, 1, False, R.B_FA, "F_A"),                      # F_a 1
    (13, 8, 2, False, R.SELECTED, "PASS"),                 # F_a 2
    (13, 8, 1, True, R.B_PG, "PUNISH_GUARD"),              # G_fail before F_a (R.3 line 6, user addition ii)
    (13, 8, 5, True, R.B_PG, "PUNISH_GUARD"),
    (11, 10, 5, True, R.B_NC, R.MARGIN),                   # margin before G_fail
])
def test_boundaries(n, c, f, g, band, reason):
    out = R.read_band(n, c, f, g, 21, 32, max(f, 4), SPEC)
    assert (out["band"], out["reason"]) == (band, reason)


@pytest.mark.parametrize("nb,na", [(20, 32), (22, 32), (21, 31), (21, 33)])
def test_counts_other_than_21_32_are_not_read(nb, na):
    assert R.read_band(13, 8, 5, False, nb, na, 5, SPEC)["band"] == R.NOT_READ


def test_b_fa_carries_naive_a_and_possibility():
    out = R.read_band(13, 8, 1, False, 21, 32, 1, SPEC)
    assert out["naive_a"] == 1 and out["f_a_possible"] is False
    assert R.read_band(13, 8, 1, False, 21, 32, 4, SPEC)["f_a_possible"] is True


def _pp(ax, c_pass, l_pass, n=10):
    cp = [dict(key=f"{ax}|{i}", axis=ax, punish_pass=i in c_pass) for i in range(n)]
    lp = [dict(key=f"{ax}|{i}", axis=ax, punish_pass=i in l_pass) for i in range(n)]
    return lp, cp


@pytest.mark.parametrize("ax", ["b", "a"])
def test_g_fail_drop_1_vs_2(ax):
    lp, cp = _pp(ax, set(range(5)), set(range(4)))           # 5 -> 4: drop 1, pass->fail 1
    assert R.g_fail(lp, cp, SPEC)["g_fail"] is False
    lp, cp = _pp(ax, set(range(5)), set(range(3)))           # 5 -> 3: drop 2
    out = R.g_fail(lp, cp, SPEC)
    assert out["g_fail"] is True and out["axes"][ax]["drop"] and not out["axes"][ax]["flips"]


@pytest.mark.parametrize("ax", ["b", "a"])
def test_g_fail_pass_to_fail_2_vs_3_with_no_drop(ax):
    lp, cp = _pp(ax, set(range(5)), {2, 3, 4, 5, 6})         # pass->fail 2, fail->pass 2, drop 0
    out = R.g_fail(lp, cp, SPEC)
    assert out["g_fail"] is False and out["axes"][ax]["pass_to_fail"] == 2
    lp, cp = _pp(ax, set(range(5)), {3, 4, 5, 6, 7})         # pass->fail 3, fail->pass 3, drop 0
    out = R.g_fail(lp, cp, SPEC)
    assert out["g_fail"] is True and out["axes"][ax]["flips"] and not out["axes"][ax]["drop"]
    assert (out["axes"][ax]["pun_L"], out["axes"][ax]["pun_C"]) == (5, 5)


def _rec(**kw):
    return dict(dict(median=0.045, n_odours=112, n_seeds=[8], edit_edges=[2], csc_sha256=["s"]), **kw)


def test_gate1():
    assert R.gate1(_rec(), True, SPEC)["outcome"] == R.PASS
    assert R.gate1(_rec(median=0.03), True, SPEC)["outcome"] == R.PASS            # band inclusive
    assert R.gate1(_rec(median=0.15), True, SPEC)["outcome"] == R.PASS
    out = R.gate1(_rec(median=0.0281), True, SPEC)
    assert out["outcome"] == R.STOP_STRENGTH_LEVER and "0.0281" in out["sentence"]
    assert R.gate1(_rec(median=0.1501), True, SPEC)["outcome"] == R.STOP_STRENGTH_LEVER
    out = R.gate1(_rec(), False, SPEC)
    assert out["outcome"] == R.STOP_STRENGTH_LEVER and "ORN 상한" in out["sentence"]
    assert R.gate1(_rec(edit_edges=[1]), True, SPEC)["outcome"] == R.INVALID
    assert R.gate1(_rec(csc_sha256=["a", "b"]), True, SPEC)["outcome"] == R.INVALID
    assert R.gate1(_rec(n_odours=111), True, SPEC)["outcome"] == R.INVALID


def test_gate2():
    ok = dict(outcome="JUDGED", label="LEARNS_CONFIRMATORY", directions={"r1": {"ell": 1.8}, "r2": {"ell": 2.2}},
              reasons=[])
    assert R.gate2(ok, SPEC)["outcome"] == R.PASS
    out = R.gate2(dict(ok, label="DIRECTION_DEPENDENT"), SPEC)
    assert out["outcome"] == R.STOP_PUNISH_BROKEN
    assert out["sentence"] == ("APL→MBON05 제거 아래에서 P의 처벌 학습 확인이 재현되지 않았다"
                               "(DIRECTION_DEPENDENT, ℓ_r1 1.800, ℓ_r2 2.200).")
    assert R.gate2(dict(outcome="INVALID", label="INVALID", reasons=["x"]), SPEC)["outcome"] == R.INVALID


def _sum(tb, fa=0, edges=(2,), sha="sha-L", reasons=()):
    return dict(reasons=list(reasons), edit_edges=list(edges), csc_sha256=sha,
                aggregate=dict(testable_b=tb, F_a=fa, naive_a=2, n_b=21, n_a=18))


def test_gate3():
    C = _sum(7, edges=(0,), sha="sha-C")
    out = R.gate3(_sum(11), C, SPEC, "sha-C")
    assert out["outcome"] == R.PASS and (out["testable_b"], out["c_even"]) == (11, 7)
    out = R.gate3(_sum(10), C, SPEC, "sha-C")
    assert out["outcome"] == R.STOP_EVEN_LOW_LEVER
    assert out["sentence"] == ("APL→MBON05 제거 아래 짝수 (b) 21쌍에서 testable_b가 M2 기준(11)에 못 미쳤다"
                               "(10/21, 지렛대 없는 같은 실행 7/21, F_a 기록 0/18).")
    out = R.gate3(_sum(16), _sum(8, edges=(0,), sha="sha-C"), SPEC, "sha-C")
    assert out["outcome"] == R.STOP_C_EVEN_MISMATCH and "8/21" in out["sentence"]   # checked before testable_b
    assert R.gate3(_sum(16, edges=(1,)), C, SPEC, "sha-C")["outcome"] == R.INVALID
    assert R.gate3(_sum(16), C, SPEC, "sha-other")["outcome"] == R.INVALID
    assert R.gate3(_sum(16, sha="sha-C"), C, SPEC, "sha-C")["outcome"] == R.INVALID
    assert R.gate3(_sum(16, reasons=("dup",)), C, SPEC, "sha-C")["outcome"] == R.INVALID


F = dict(n=14, c=8, f_a=3, naive_a=4, T=103, k_even=12, pb_L=15, pb_C=17, pa_L=20, pa_C=22, cond="x",
         label="NO_LEARNING", l1="0.100", l2="0.200", tb=9, c_even=7, k=0, k_b=3, k_a=1)


def test_sentences_verbatim():
    for o in (R.STOP_STRENGTH_LEVER, R.STOP_PUNISH_BROKEN, R.STOP_EVEN_LOW_LEVER, R.STOP_C_EVEN_MISMATCH, R.B_TB,
              R.B_PG, R.B_FA, R.SELECTED):
        assert "{" not in R.sentence(o, F), o
    assert R.sentence(R.B_TB, F) == ("APL→MBON05 제거가 판정 세트(L 생성기 턴 64–103)에서 14/21로 지렛대 없는 같은 세트 "
                                     "8/21보다 오르지 않았다. → 이 지렛대를 닫는다.")
    assert R.sentence(R.B_NC, dict(F, reason=R.C_ABOVE_BAR)) == (
        "지렛대 없는 E-grid가 이 세트에서 이미 기준을 넘어 지렛대 효과로 말할 수 없다. (14/21 대 8/21, F_a 3/32)")
    assert R.sentence(R.B_NC, dict(F, reason=R.BELOW_BAR)).startswith("지렛대 없는 쪽보다 올랐지만 M2 기준에 못 미쳤다.")
    assert R.sentence(R.B_NC, dict(F, reason=R.MARGIN)).startswith("기준은 넘었으나 지렛대 없는 쪽 대비 여유가 2 미만이다")
    assert R.sentence(R.B_PG, F) == ("M2 (b) 기준과 여유는 넘었지만 지렛대가 처벌 통과를 줄이거나 바꿨다(처벌 통과 (b) "
                                     "15 대 17, (a) 20 대 22; pass→fail (b) 3, (a) 1).")   # R.9.5 (D10)
    assert R.sentence(R.B_FA, F) == ("M2 (b) 기준·여유·처벌 가드는 넘었지만 F_a가 기준에 못 미쳤다(14/21 대 8/21, F_a 3/32, "
                                     "naive_a 4). 다음 병목은 F_a(순진 균형 (a) 쌍)다.")
    sel = R.sentence(R.SELECTED, F)
    assert sel.startswith("C3에서 APL→MBON05 2간선을 지운 모델 변형과 E-grid k2-norm(s 1.0)에서, 판정 세트(L 생성기 턴 64–103")
    assert "(14/21 대 8/21, 여유 ≥ 2, F_a 3/32, 처벌 가드 통과, 짝수 12/21, P 재현 `LEARNS_CONFIRMATORY`, 판정 시드 24_100_xxx)" in sel
    assert sel.endswith("실제 커넥톰 간선을 지운 모델이며 오라클 시험 가능성이지 학습 시험이 아니다.")
    assert R.sentence(R.STOP_STRENGTH_LEVER, F) == ("APL→MBON05 제거 아래에서 E-grid k2-norm s 1.0이 KC 활성 자격(x)을 "
                                                    "잃었다.")
    with pytest.raises(KeyError):
        R.sentence(R.B_FA, {})


def test_g_axis_matches_brute_force_on_a_small_axis():
    a_pf, a_fp, tot = 0.2, 0.1, 0.0
    for states in itertools.product(("pf", "fp", "same"), repeat=4):
        w = 1.0
        for s in states:
            w *= {"pf": a_pf, "fp": a_fp, "same": 1 - a_pf - a_fp}[s]
        k, j = states.count("pf"), states.count("fp")
        if k - j >= 2 or k >= 3:
            tot += w
    assert R.g_axis(4, a_pf, a_fp, SPEC) == pytest.approx(tot)


def test_g_prob_null_and_harm():
    assert R.g_prob(0.0, 0.0, SPEC) == 0.0
    vals = [R.g_prob(a, a, SPEC) for a in SPEC.oc_discord]
    assert all(0 < v < 1 for v in vals) and vals == sorted(vals)
    assert R.g_prob(0.2, 0.05, SPEC) > R.g_prob(0.05, 0.05, SPEC)


def test_oc_rows_are_distributions_and_carry_the_g_rows():
    oc = R.oc(SPEC)
    assert len(oc["g_fail"]) == len(SPEC.oc_discord) + len(SPEC.oc_harm)
    n_na = min(SPEC.n_a, SPEC.oc_naive_max) + 1
    assert len(oc["rows"]) == len(SPEC.oc_q) * len(SPEC.oc_c) * n_na * (1 + len(oc["g_fail"]))
    for row in oc["rows"][::37]:
        assert sum(row["P"].values()) == pytest.approx(1.0)
    assert all(r["P"][R.B_PG] == 0 for r in oc["rows"] if r["g_kind"] == "none")
    assert len(oc["sha256"]) == 64 and oc["sha256"] == R.oc(SPEC)["sha256"]


def test_r_rules_literals_are_only_0_and_1():
    import ast
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2] / "flymon/brain/r_rules.py").read_text()
    nums = {n.value for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Constant) and type(n.value) in (int, float)}
    assert nums <= {0, 1}, nums
