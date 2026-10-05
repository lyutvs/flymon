"""Y's phase-A decisions (Y.3.1–Y.3.3 final filter, Y.4 admission / merge / 4a, Y.7 1 X facts, Y.6.1 precheck
sentence, Y.9.2 P1-4 / P1-6 records): one filter function equal to W's naive d′ and pilot medians; bounds and
rounding; the Earthquake merge; STOP_PILOT_FEW after the merge; H6; X reuse reasons; the precheck sentence."""
import numpy as np
import pytest

from flymon.brain import w_records
from flymon.brain import w_verdict as WV
from flymon.brain import y_rules as R
from flymon.brain.w_spec import SPEC as W
from flymon.brain.y_spec import SPEC as Y

Z = {"A": (16.917, 12.484), "P": (80.167, 29.775)}
KEYS7 = list(Y.pilot_w_pairs) + list(Y.pilot_v_pairs)


def _pre(rng, a=40.0, p=90.0, shape=(8, 8)):
    x = np.empty(shape + (2, 2))
    x[..., WV.A, :], x[..., WV.P, :] = a, p
    return np.clip(np.rint(x + rng.normal(0, 3.0, x.shape)), 0, None)


def test_filter_values_equal_w_naive_and_pilot_medians():
    rng = np.random.default_rng(0)
    pre = _pre(rng)
    d, la, lp = R.filter_values(pre, Z)
    assert float(d) == WV.naive_dprime(pre, Z)
    pair = {s: pre for s in WV.STAGES}
    rec = w_records.pilot_record({"k": pair}, Z, {}, W)["pairs"]["k"]["naive_median"]
    assert (float(la), float(lp)) == (rec["MBON13_X"], rec["MBON05_X"])


def test_filter_values_batch_equals_single():
    rng = np.random.default_rng(1)
    pres = np.stack([np.stack([_pre(rng) for _ in range(3)]) for _ in range(2)])     # [2, 3, F, K, 2, 2]
    d, la, lp = R.filter_values(pres, Z)
    assert d.shape == (2, 3)
    for i in range(2):
        for j in range(3):
            s = R.final_filter(pres[i, j], Z, Y)
            assert (s["d"], s["L_A"], s["L_P"]) == (float(d[i, j]), float(la[i, j]), float(lp[i, j]))


@pytest.mark.parametrize("d,la,lp,want", [
    (0.499999999, 20.0, 43.0, True), (0.5, 20.0, 43.0, False), (-0.5, 30.0, 60.0, False),
    (0.0, 19.9999999999, 43.0, True), (0.0, 19.99999, 43.0, False), (0.0, 20.0, 42.9999, False),
    (float("nan"), 30.0, 60.0, False), (float("inf"), 30.0, 60.0, False)])
def test_filter_bounds_and_rounding(d, la, lp, want):
    assert bool(R.filter_passes(d, la, lp, Y)) is want


@pytest.mark.parametrize("d,la,lp,want", [
    (0.4999999999, 20.0, 43.0, False),     # |d′| rounds to 0.5 → strict bound fails (removing d′'s rounding passes it)
    (0.0, 20.0, 42.9999999999, True)])     # L_P rounds to 43 → bound included (removing L_P's rounding fails it)
def test_filter_rounding_is_load_bearing(d, la, lp, want):
    assert bool(R.filter_passes(d, la, lp, Y)) is want
    assert Y.round_digits == 9


def test_merge_groups_on_the_seven_candidates():
    g = R.merge_groups(KEYS7)
    assert len(g) == 6 and [KEYS7[i] for i in g[4]] == ["a|218|Earthquake|Strength", "a|305|Earthquake|Rock Slide"]
    assert R.merge_groups(KEYS7[:3]) == [[0], [1], [2]]


def _adm(passed: dict):
    return {k: dict(d=0.1, L_A=30.0, L_P=60.0, passed=passed.get(k, True)) for k in KEYS7}


def _rec(meds):
    return dict(pairs={f"p{i}": dict(median=m) for i, m in enumerate(meds)}, floor_both_share=0.0)


GOOD = dict(reward_level=2.0, punish_drop=-2.0, reward_assoc=2.0, punish_assoc=-2.0)


def test_pilot_gates_few_after_merge():
    adm = _adm({KEYS7[3]: False, KEYS7[6]: False})        # b|17 and a|223 fail; W 3 + the Earthquake pair pass
    out = R.pilot_gates(adm, None, [], Y, W)
    assert out["outcome"] == R.STOP_PILOT_FEW and out["n_admitted"] == 5 and out["n_sigma"] == 4
    assert "통과한 쌍이 5개" in out["sentence"] and "묶은 뒤 4개로 최소 5에" in out["sentence"]
    assert "a|223|Surf|Earthquake d′ 0.100 · MBON13(X) 30 · MBON05(X) 60" in out["sentence"]
    assert out["records_unavailable"] is True


def test_pilot_gates_order_invalid_then_few_then_h6_then_pass():
    adm = _adm({})
    assert R.pilot_gates(adm, _rec([GOOD] * 7), ["fly 0: N pre ≠ R pre"], Y, W)["outcome"] == R.INVALID
    bad = dict(GOOD, reward_assoc=0.1, punish_assoc=0.1)
    out = R.pilot_gates(adm, _rec([bad] * 7), [], Y, W)
    assert out["outcome"] == R.STOP_PILOT_NO_EFFECT and out["n_sigma"] == 6
    assert out["sentence"].startswith("Y 파일럿(거름 통과 7쌍)에서 조합 지렛대의 F.2 학습 효과가 (i)")
    assert R.pilot_gates(adm, _rec([GOOD] * 7), [], Y, W)["outcome"] == R.PASS


def test_pilot_gates_machine_reason_wins_over_few():
    adm = _adm({KEYS7[3]: False, KEYS7[6]: False})        # n_Σ 4 < 5 and a machine reason at once
    out = R.pilot_gates(adm, None, ["fly 0: N pre ≠ R pre"], Y, W)
    assert out["outcome"] == R.INVALID and out["reasons"] == ["fly 0: N pre ≠ R pre"]
    assert out["n_sigma"] == 4 and "sentence" not in out and "records_unavailable" not in out


def test_sentences_a_numbers_from_spec():
    s = R.sentences_a(Y)[(R.STOP_OC_UNREACHABLE, "precheck")]
    assert "F ≤ 32로" in s and "점 검정력 ≥ 0.80과 점 거짓 통과 ≤ 0.05를" in s
    assert s.endswith("부트스트랩 작동 특성은 계산하지 않았고, 주 세트 학습 측정 없이 멈춘다.")


def _facts(**kw):
    f = dict(x_doc={"stage0": {"decision_files": {"a.py": "1"}}, "precheck": {}}, ancestors={"868771a": True,
             "4b81035": True}, last="4b81035" + "0" * 33, git=dict(tracked=True, dirty=False),
             diag_sha=Y.x_precheck_diag_sha256, x_files={"a.py": "1"},
             w_ancestors={c: True for _, c in Y.w_commits}, w_blocks={b: Y.w_measure_key for b, _ in Y.w_commits})
    f.update(kw)
    return f


def test_w_block_reasons():
    assert R.w_block_reasons(_facts(), Y) == []
    f = _facts(w_ancestors={"5fbc4c8": True, "f30ae35": True})
    assert sorted(w for w in R.w_block_reasons(f, Y)) == ["W 커밋 5620f95(path)가 HEAD 이력에 없음",
                                                       "W 커밋 744d1bc(reuse)가 HEAD 이력에 없음"]
    f = _facts(w_blocks={"reuse": Y.w_measure_key, "pilot": Y.w_measure_key, "oc": Y.w_measure_key})
    assert R.w_block_reasons(f, Y) == ["W 블록 path의 W 측정 키 None"]


@pytest.mark.parametrize("kw,needle", [
    (dict(git=dict(tracked=True, dirty=True)), "미커밋"), (dict(ancestors={"868771a": True}), "4b81035(precheck)"),
    (dict(last="ffffff"), "마지막 커밋"), (dict(diag_sha="0" * 64), "precheck_diag"),
    (dict(x_files={"a.py": "2"}), "x_* 파일"), (dict(x_doc={"stage0": {"decision_files": {"a.py": "1"}}}), "X 블록 precheck")])
def test_x_reasons(kw, needle):
    assert R.x_reasons(_facts(), Y) == []
    assert any(needle in w for w in R.x_reasons(_facts(**kw), Y))


def _pc(passed=False, cal_ok=True):
    c = dict(ok=cal_ok, a=dict(status="ok" if cal_ok else "floor"), b=[dict(status="ok")] if cal_ok else None)
    return dict(passed=passed, passing=[{}] if passed else [], calibration=dict(min=c, max=dict(c)),
                at_f32=[dict(p_set=0.5, q=0.5, K=8, F=32, k_range=[4, 8], k=[4, 5], power=[0.7, 0.6],
                             false=[0.0, 0.01])],
                best_power=dict(p_set=0.5, q=0.5, K=16, F=8, k_range=[4, 8], k=[4, 5], power_by_k=[0.71, 0.62]))


def test_precheck_decision_and_paren():
    assert R.precheck_decision(_pc(True), 7, 6, Y)["outcome"] == R.PASS
    out = R.precheck_decision(_pc(False), 7, 6, Y)
    assert out["outcome"] == R.STOP_OC_UNREACHABLE
    assert "Y 파일럿(거름 통과 7쌍, Σ 추정 6쌍)" in out["sentence"]
    assert "p_set 0.5·q 0.5·K 8·F 32·k 범위 [4, 8]: k 4 점 검정력 0.700 / 점 거짓 통과 0.000" in out["sentence"]
    assert "점 검정력 최대 설계 p_set 0.5 · q 0.5 · K 16 · F 8 · k 범위 [4, 8]의 k별 점 검정력 k 4 0.710, k 5 0.620" in \
        out["sentence"]
    fl = R.precheck_decision(_pc(False, cal_ok=False), 7, 6, Y)["sentence"]
    assert "보정 불가 — min: a floor, b 미시도; max: a floor, b 미시도" in fl


def test_level_compare_and_fact_flags():
    adm = {"a": dict(L_A=20.0, L_P=50.0, passed=True), "b": dict(L_A=40.0, L_P=90.0, passed=True),
           "c": dict(L_A=1.0, L_P=1.0, passed=False)}
    out = R.level_compare({"n": 9}, adm, Y)
    assert out["oracle_lenient"] == {"n": 9} and out["pilot"]["n"] == 2 and out["pilot"]["A"] == [20.0, 25.0, 30.0,
                                                                                                    35.0, 40.0]
    f = R.fact_flags({"x": (0.07747, 31.0, 107.0), "y": (0.1, 1.0, 1.0)},
                     {"x": (0.077, 31.0, 107.0), "y": (0.2, 1.0, 1.0), "z": (0.0, 0.0, 0.0)}, Y)
    assert f == {"x": True, "y": False, "z": None}


def test_flip_record_on_the_spec_facts():
    rec = lambda d, a, p: dict(value=True, testable=True, d_pre=d, L_A=a, L_P=p)  # noqa: E731
    k10, k2, k4, k1 = (Y.pilot_w_pairs[2], Y.pilot_w_pairs[1], "b|4|Rock Slide vs Venusaur|Rock Slide vs Charizard",
                       Y.pilot_w_pairs[0])
    oracle = {k10: rec(-0.762, 35.0, 121.0), k2: rec(1.005, 38.0, 114.5), k4: rec(0.706, 29.5, 105.0),
              k1: rec(-0.009, 29.5, 105.0), "x": rec(5.0, 10.0, 10.0)}
    pilot = {k10: (-0.318, 34.5, 121.0), k2: (-0.052, 38.0, 114.5), k4: (1.088, 31.0, 108.0), k1: (0.077, 31.0, 107.0),
             "x": (9.0, 10.0, 10.0)}
    out = R.flip_record(oracle, pilot, Y)
    assert sorted(out["strict_flips"]) == sorted([k10, k2]) and sorted(out["lenient_mismatch"]) == sorted([k4, k2])
    assert out["recall"] == [2, 3] and out["n"] == 5 and out["matches_spec"] is True
    assert out["yield_basis"] == dict(lenient=3, strict_of_lenient=2, c=1.5, matches_spec=True)
