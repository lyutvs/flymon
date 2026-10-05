"""Y's phase-B numbers and decisions (Y.5, Y.7 6–12, Y.8, Y.9, Y.9.2 P2-8 · P2-9): sentences verbatim, W's cost
model with Y's k and N_len, the ×1.3 budget rule, the order-6 decision, the final-filter STOP and the set verdict."""
import pytest

from flymon.brain import y_rules as R
from flymon.brain.w_spec import SPEC as W
from flymon.brain.y_spec import SPEC as Y

PILOT_COSTS = dict(trial_s=3.71789051508822, presentation_s=3.46501247351019, oracle_round_s=689.9133451848684,
                   workers=16)


def test_phase_b_numbers():
    assert (Y.oc_detail, Y.gates_detail, Y.judge_detail) == ("results/y/oc.json", "results/y/gates.json",
                                                             "results/y/judge.json")
    assert Y.smoke_cache_dir == "results/y/smoke/cache" and "smoke" in Y.smoke_cache_dir.split("/")
    assert (Y.smoke_pair, Y.smoke_flies, Y.smoke_block) == (0, 1, (77_100_000, 77_200_000))
    assert Y.smoke_block[0] == Y.smoke_probe_seed0 and Y.smoke_block[0] <= Y.smoke_oracle_seed0 < Y.smoke_block[1]
    assert (Y.pct_scale, Y.s_per_h, Y.cost_margin, Y.reconfirm_reps, Y.reconfirm_max) == (100.0, 3600.0, 1.3, 1600, 5)
    assert Y.cost_fields == ("trial_s", "presentation_s", "oracle_round_s")


def test_sentences_b_are_the_specs():
    s = R.sentences_b(Y)
    assert s[(R.STOP_OC_UNREACHABLE, "boot")].format(paren="P") == (
        "Y 파일럿 잡음에서 F ≤ 32로 G.6 작동 특성 목표를 맞출 수 없다 — p_set · q · K · k 범위 어느 설계도(P) — 주 세트 "
        "학습 측정 없이 멈춘다.")
    assert s[(R.STOP_OC_UNREACHABLE, "reconfirm")].format(r=4, paren="P") == (
        "Y 파일럿 잡음에서 선택 설계와 대체 설계 4개가 독립 시드 재확인(200 × 1600)에서 G.6 작동 특성 목표를 다시 "
        "만족하지 못했다(P) — 주 세트 학습 측정 없이 멈춘다.")
    assert s[R.STOP_BUDGET].format(h="누적 1.00 h + 남은 2.00 h = 3.00 h", where="순서 7a") == (
        "남은 추정 비용 누적 1.00 h + 남은 2.00 h = 3.00 h가 Y 상한 24 h를 넘는다(순서 7a).")
    assert s[(R.STOP_FEW_PAIRS, "final")].format(n=249, n_pre=31, k=3, k_lo=4) == (
        "Y 주 세트 249쌍 중 오라클 사전 거름을 통과한 31쌍에서 판정 시드 순진 거름(|d′| < 0.5 ∧ 순진 MBON13(X) ≥ 20 ∧ "
        "순진 MBON05(X) ≥ 43)을 통과한 관문 쌍이 3개로 설계의 최소 4에 못 미쳤다 — 설계를 바꾸지 않고, 주 세트 학습 "
        "측정 없이 멈춘다.")
    assert s[R.STOP_MACHINE].format(why="W") == (
        "Y 학습 측정에서 기계 검사가 맞지 않았다(W) — 같은 관문 쌍으로 다시 돌리지 않으며(Y.2, W.5 · W.9.8 H8), "
        "INVALID_RUN 여부는 사용자 몫이다.")


def test_design_cost_y_is_ws_model_with_ys_k_and_n_len():
    c = PILOT_COSTS
    job8 = 2 * W.trials * c["trial_s"] + W.job_stages * 2 * 8 * c["presentation_s"]
    want = dict(oracle=0.0, naive=16 * 2 * 8 * c["presentation_s"], learn=12 * job8, band=12 * job8, c=12 * job8,
                noplast=1 * job8)                       # 31 × 8 = 248 naive jobs / 16 → 16 rounds; 8 × 8 × 3 = 192 → 12
    got = R.design_cost_y(c, dict(K=8, F=8, k_range=[4, 8]), Y, W, 31, 8, True)
    assert got["parts_h"] == pytest.approx({k: v / 3600 for k, v in want.items()})
    no_c = R.design_cost_y(c, dict(K=8, F=8, k_range=[4, 8]), Y, W, 31, 8, False)
    assert no_c["total_h"] == pytest.approx(got["total_h"] - want["c"] / 3600) and no_c["parts_h"]["c"] == 0.0
    k10 = R.design_cost_y(c, dict(K=16, F=32, k_range=[6, 10]), Y, W, 31, 10, False)
    job16 = 2 * W.trials * c["trial_s"] + W.job_stages * 2 * 16 * c["presentation_s"]
    assert k10["parts_h"]["learn"] == pytest.approx(60 * job16 / 3600)        # 10 × 32 × 3 = 960 jobs → 60 rounds
    assert k10["parts_h"]["naive"] == pytest.approx(62 * 2 * 16 * c["presentation_s"] / 3600)   # 31 × 32 / 16


def test_costs_max_and_margin():
    sm = dict(trial_s=4.0, presentation_s=1.0, oracle_round_s=700.0, workers=4)
    m = R.costs_max(PILOT_COSTS, sm, Y)
    assert m["trial_s"] == 4.0 and m["presentation_s"] == PILOT_COSTS["presentation_s"] and m["workers"] == 16
    assert R.costs_max(PILOT_COSTS, None, Y) == PILOT_COSTS
    assert R.in_budget(11.0, 10.0, Y) and not R.in_budget(11.0 + 1e-6, 10.0, Y)     # 11 + 1.3 × 10 = 24 exactly
    assert not R.in_budget(0.0, 24.0 / 1.3 + 1e-6, Y) and R.in_budget(23.0, 0.0, Y)
    st = R.budget_stop(20.0, [dict(total_h=5.0), dict(total_h=4.0)], "순서 7a", Y)
    assert st["outcome"] == R.STOP_BUDGET and "누적 20.00 h + 남은 5.20 h = 25.20 h" in st["sentence"]
    assert st["sentence"].endswith("(순서 7a).")


def test_oc_decision():
    at_f = [dict(p_set=0.5, q=0.5, K=8, F=32, k_range=[4, 8], k=[4, 5], power=[0.7, 0.6], false=[0.01, 0.02])]
    counts = dict(min=dict(handle_status={"a|ok": 3}), max=dict(handle_status={"a|ok": 3}))
    d = R.oc_decision(dict(outcome=R.STOP_OC_UNREACHABLE), at_f, counts, Y)
    assert d["outcome"] == R.STOP_OC_UNREACHABLE and "k 4 0.700 / 0.010" in d["sentence"] and "a|ok 3" in d["sentence"]
    d = R.oc_decision(dict(outcome=R.STOP_BUDGET, budget_text="누적 1.00 h + 남은 30.00 h = 31.00 h"), at_f, counts, Y)
    assert d["outcome"] == R.STOP_BUDGET and d["sentence"].endswith("(순서 6 선택).")
    assert R.oc_decision(dict(outcome=R.PASS), at_f, counts, Y)["outcome"] == R.PASS
    rs = [dict(design=dict(p_set=1.0, q=0.75, K=8, F=8, k_range=[4, 8]), k=[4], power_by_k=[0.7], false_by_k=[0.0])] * 5
    st = R.reconfirm_stop(rs, Y)
    assert st["stop_kind"] == "reconfirm" and "대체 설계 4개" in st["sentence"] and "k 4 0.700 / 0.000" in st["sentence"]


def test_few_pairs_final():
    assert R.few_pairs_final(249, 31, 4, 4, Y)["outcome"] == R.PASS
    d = R.few_pairs_final(249, 31, 5, 6, Y)
    assert d["outcome"] == R.STOP_FEW_PAIRS and "관문 쌍이 5개로 설계의 최소 6에" in d["sentence"]
    assert d["main_set"].startswith("미사용")


def _v(statuses, machine=(), mech=()):
    pairs = {f"p{i}": dict(status=s, reasons=["마리별 동시 충족 미달"] if s == "FAIL" else [],
                           failing_gates={"punish_assoc": 2} if s == "FAIL" else {}) for i, s in enumerate(statuses)}
    return dict(pairs=pairs, machine=list(machine), failing=[k for k, p in pairs.items() if p["status"] == "FAIL"],
                mech_fail=list(mech), invalid=[k for k, p in pairs.items() if p["status"] == "INVALID"])


D = dict(p_set=0.75, q=0.75, K=8, F=8, k_range=[4, 8])


def test_set_verdict_pass_fail_machine():
    v = R.set_verdict(_v(["PASS"] * 7 + ["FAIL"]), D, 8, Y)
    assert v["verdict"] == R.PASS and (v["n"], v["m"], v["b"]) == (8, 7, 0)
    assert "p_set 0.75 이상에서 섰다" in v["sentence"] and "PASS 7(7/8 = 0.875 ≥ p_set 0.75)" in v["sentence"]
    assert "p7: 마리별 동시 충족 미달(punish_assoc 2마리)" in v["sentence"] and "k 범위 [4, 8]" in v["sentence"]
    v = R.set_verdict(_v(["PASS"] * 8), dict(D, p_set=1.0), 8, Y)
    assert v["verdict"] == R.PASS and "중 전부에서 섰다" in v["sentence"] and "FAIL 쌍 없음" in v["sentence"]
    v = R.set_verdict(_v(["PASS"] * 5 + ["FAIL"] * 3), D, 8, Y)
    assert v["verdict"] == R.FAIL and "m/n 0.625 < p_set 0.75" in v["sentence"]
    v = R.set_verdict(_v(["PASS"] * 2 + ["FAIL"] * 2), dict(D, p_set=0.5), 4, Y)
    assert v["verdict"] == R.FAIL and "PASS 쌍 2 < 3" in v["sentence"] and "m/n" not in v["sentence"]
    v = R.set_verdict(_v(["PASS"] * 8, machine=["p1: RN1 ≠ R1"]), D, 8, Y)
    assert v["verdict"] == R.STOP_MACHINE and "p1: RN1 ≠ R1" in v["sentence"]
    v = R.set_verdict(_v(["PASS"] * 7 + ["INVALID"]), D, 8, Y)
    assert v["verdict"] == R.STOP_MACHINE and "INVALID 쌍 p7" in v["sentence"]
    v = R.set_verdict(_v(["PASS"] * 2 + ["BAND"] * 2), D, 4, Y)
    assert v["verdict"] == R.UNDECIDED and v["b"] == 2 and "판정 가능 쌍 < 4" in v["sentence"]
    assert set(R.CONSEQUENCE) == {R.PASS, R.FAIL, R.UNDECIDED, R.STOP_MACHINE}
