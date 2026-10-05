"""W's gate decisions and sentences (W.3, W.7, W.9.3-W.9.9): reuse (V's blocks, keys, commits, band, z_V), the path
gate's INVALID / STOP order, H6's three conditions, the OC's outcome (W.9.10 3), the budget order (C first, then the
alternatives), STOP_FEW_PAIRS (W.9.10 4), and every closing sentence verbatim (W.9.10 5-7)."""
import dataclasses

from flymon.brain import w_rules as R
from flymon.brain.w_spec import SPEC

KEYS = dict(u_measure_key=SPEC.u_measure_key_u, code_key=SPEC.r_shared_key, t_measure_key=SPEC.t_measure_key_t)
ZV = {"A": [16.916666666666668, 12.483878492769072], "P": [80.16666666666667, 29.775432639827233]}


def v_doc():
    return dict(z=dict(KEYS, z_V=ZV), kc_input=dict(KEYS), set=dict(KEYS), judge=dict(KEYS, band="SELECTED"))


def reuse(**kw):
    a = dict(v_dec=dict(outcome="PASS", reasons=[]), v_doc=v_doc(), v_git=dict(tracked=True, dirty=False),
             ancestors={c: True for _, c in SPEC.v_commits}, last_v_commit="a279a56" + "f" * 33,
             u_key=SPEC.u_measure_key_u, code_key=SPEC.r_shared_key, t_key=SPEC.t_measure_key_t, spec=SPEC)
    a.update(kw)
    return R.reuse(**a)


def test_reuse():
    assert reuse()["outcome"] == "PASS"
    d = v_doc()
    d["z"]["z_V"] = {"A": [16.9, 12.484], "P": ZV["P"]}
    for kw, word in ((dict(v_dec=dict(outcome="STOP_REUSE", reasons=["R 재현"])), "R 재현"),
                     (dict(u_key="x"), "U 측정 키 x"), (dict(v_git=dict(tracked=True, dirty=True)), "미커밋"),
                     (dict(last_v_commit="b" * 40), "마지막 커밋"),
                     (dict(ancestors={c: c != "cf0b3b2" for _, c in SPEC.v_commits}), "cf0b3b2(set)"),
                     (dict(v_doc=d), "V z_V")):
        out = reuse(**kw)
        assert out["outcome"] == R.STOP_REUSE and word in out["sentence"], (kw, out)
    d = v_doc()
    d["judge"]["band"] = "B_Tb"
    assert "V 판정 B_Tb" in reuse(v_doc=d)["sentence"]
    assert reuse(v_doc=d)["sentence"].startswith("W 재사용 조건(W.3 1)이 깨졌다(")


def test_path_order():
    ok = dict(ref="a", diffs=[], invalid=[])
    assert R.path([ok, ok])["outcome"] == "PASS"
    out = R.path([ok, dict(ref="V 관문 ② 처벌 팔 행(L_V r1 1)", diffs=["x pre"], invalid=[]),
                  dict(ref="b", diffs=["y"], invalid=[])])
    assert out["outcome"] == R.STOP_W_PATH_REPRO and len(out["failed"]) == 2
    assert out["sentence"] == "W 학습 경로가 V 관문 ② 처벌 팔 행(L_V r1 1)을 재현하지 못했다(x pre)."
    assert R.path([dict(ref="a", diffs=["d"], invalid=["edges 3"])])["outcome"] == R.INVALID


def rec(pairs, floor=0.0):
    return dict(pairs={f"p{i}": dict(median=dict(reward_assoc=r, punish_assoc=p)) for i, (r, p) in enumerate(pairs)},
                floor_both_share=floor)


def test_pilot_h6():
    assert R.pilot(rec([(1.0, -1.0)] * 4), [], SPEC)["outcome"] == "PASS"
    assert R.pilot(rec([(1.0, -1.0)] * 4), ["fly 0: N pre ≠ R pre"], SPEC)["outcome"] == R.INVALID
    out = R.pilot(rec([(1.0, -0.4)] * 3 + [(1.0, -1.0)]), [], SPEC)                  # punish share 0.25
    assert out["outcome"] == R.STOP_PILOT_NO_EFFECT and out["reasons"][0].startswith("(i)")
    assert R.pilot(rec([(1.0, -1.0)] * 2 + [(0.5, -0.5)] * 2), [], SPEC)["outcome"] == "PASS"   # 0.5 / −0.5 count
    out = R.pilot(rec([(1.0, -1.0)] * 2 + [(-0.1, -1.0)] * 3), [], SPEC)            # reversed 0.6 > 0.5
    assert any(r.startswith("(ii)") for r in out["reasons"])
    out = R.pilot(rec([(1.0, -1.0)] * 4, floor=0.51), [], SPEC)
    assert out["reasons"] == ["(iii) X·Y 함께 바닥인 프로브 비율 0.510(> 0.5)"]
    assert out["sentence"] == ("파일럿에서 조합 지렛대의 F.2 학습 효과가 (iii) X·Y 함께 바닥인 프로브 비율 0.510(> 0.5) — 주 "
                               "세트를 쓰지 않고 멈춘다.")
    assert R.pilot(rec([(1.0, -1.0)] * 4, floor=0.5), [], SPEC)["outcome"] == "PASS"


def oc_doc(**kw):
    """A STOP_OC_UNREACHABLE doc: combined limits over k (must NOT be printed) and per-k limits on [q, K, F, k]; F = 32
    is picked by value (axes F), not by position."""
    import numpy as np
    fs = list(range(SPEC.f_min, SPEC.f_max + 1))
    ks = list(range(SPEC.k_min, SPEC.k_cap + 1))
    nq, nk, nf, nkk = len(SPEC.q_grid), len(SPEC.k_grid), len(fs), len(ks)
    pk = np.full((nq, nk, nf, nkk), 0.9)
    fk = np.full((nq, nk, nf, nkk), 0.2)
    f32 = fs.index(32)
    for i in range(nkk):
        pk[:, :, f32, i] = 0.40 + 0.01 * i
        fk[:, :, f32, i] = 0.010 + 0.001 * i
    pk[0, 1, f32] += 0.1
    d = dict(selected=None, reachable=True, power_lo=np.full((nq, nk, nf), 0.111).tolist(),
             false_hi=np.full((nq, nk, nf), 0.999).tolist(), power_lo_by_k=pk.tolist(), false_hi_by_k=fk.tolist(),
             k_values=ks, axes=dict(q=list(SPEC.q_grid), K=list(SPEC.k_grid), F=fs, k=ks))
    d.update(kw)
    return d


def test_oc_outcome_and_open3_text():
    assert R.oc(dict(selected=dict(q=0.75, K=8, F=10)), SPEC)["outcome"] == "PASS"
    out = R.oc(oc_doc(), SPEC)
    assert out["outcome"] == R.STOP_OC_UNREACHABLE
    q0, K0, K1 = SPEC.q_grid[0], SPEC.k_grid[0], SPEC.k_grid[1]
    first = (f"q {q0}·K {K0}·F 32: k 4 검정력 하한 0.400 / 거짓 통과 상한 0.010, k 5 검정력 하한 0.410 / 거짓 통과 "
             "상한 0.011, k 6 검정력 하한 0.420 / 거짓 통과 상한 0.012, k 7 검정력 하한 0.430 / 거짓 통과 상한 0.013, "
             "k 8 검정력 하한 0.440 / 거짓 통과 상한 0.014; ")
    assert out["sentence"].startswith("파일럿 잡음에서 F ≤ 32로 G.6 작동 특성 목표를 맞출 수 없다(" + first)
    assert f"q {q0}·K {K1}·F 32: k 4 검정력 하한 0.500 / " in out["sentence"]
    assert out["sentence"].count("F 32:") == len(SPEC.q_grid) * len(SPEC.k_grid)
    assert out["sentence"].count("검정력 하한") == len(SPEC.q_grid) * len(SPEC.k_grid) * 5
    assert "0.111" not in out["sentence"] and "0.999" not in out["sentence"] and "0.900" not in out["sentence"]
    un = R.oc(dict(selected=None, reachable=False, calibration=dict(min=dict(ok=False, a=dict(status="floor"), b=None),
                                                                  max=dict(ok=True, a=dict(status="ok"),
                                                                           b=dict(status="ok")))), SPEC)
    assert "보정 불가 — min: a floor, b None; max: a ok, b ok" in un["sentence"]


def test_budget_order():
    import pytest
    opts = [dict(design="s", with_c=True, total_h=10.0), dict(design="s", with_c=False, total_h=7.0),
            dict(design="alt", with_c=False, total_h=6.0)]
    assert R.budget(10.0, opts, SPEC, "5a")["plan"]["with_c"] is True
    assert R.budget(15.0, opts, SPEC, "5a")["plan"] == opts[1]
    assert R.budget(17.5, opts, SPEC, "5a")["plan"] == opts[2]
    out = R.budget(19.0, opts, SPEC, "5a")
    assert out["outcome"] == R.STOP_BUDGET and out["sentence"] == (        # the cheapest tried option, not opts[0]
        "남은 추정 비용 누적 19.00 h + 남은 6.00 h = 25.00 h가 W 상한 24 h를 넘는다.")
    assert R.budget(14.0, opts, dataclasses.replace(SPEC, budget_h=24.0), "5a")["plan"]["with_c"] is True
    # stage 8 (W.9.9 P1-5): only [with C, without C]; C dropped, then stop
    assert R.budget(15.0, opts[:2], SPEC, "8")["plan"] == opts[1]
    assert R.budget(10.0, opts[:1], SPEC, "8")["plan"] == opts[0]
    s8 = R.budget(18.0, opts[:2], SPEC, "8")
    assert s8["outcome"] == R.STOP_BUDGET and "남은 7.00 h = 25.00 h" in s8["sentence"]
    for bad in (opts, [opts[1], opts[0]], [opts[0], opts[2]], [opts[1]]):
        with pytest.raises(ValueError):
            R.budget(0.0, bad, SPEC, "8")
    with pytest.raises(ValueError):
        R.budget(0.0, opts, SPEC, "6")


def test_few_pairs():
    assert R.gate_pairs(249, 4, SPEC)["outcome"] == "PASS"
    out = R.gate_pairs(249, 3, SPEC)
    assert out["outcome"] == R.STOP_FEW_PAIRS
    assert out["sentence"] == "W 주 세트 249쌍에서 순진 프로브 균형·오라클 시험 가능 쌍이 3개로 최소 4에 못 미쳤다."   # W.9.10 4


def test_verdict_sentences():
    pairs = {"b|1|x|y": dict(status="FAIL", reasons=["마리별 동시 충족 미달"], failing_gates=dict(punish_assoc=5)),
             "a|2|x|y": dict(status="FAIL", reasons=["기계 대조 실패"], failing_gates={})}
    v = dict(verdict="FAIL", pairs=pairs, failing=list(pairs), mech_fail=["a|2|x|y"], machine=[], invalid=[],
             n_pairs=8, n_judgeable=8, undecided_cause="")
    s = R.verdict_sentence(v, dict(q=0.75, K=8, F=12), SPEC)
    assert s["sentence"] == ("F.7대로 M2 no-go를 기록한다 — 이 조합 지렛대·이 세트·오라클 거름 조건부(b|1|x|y: 마리별 동시 "
                             "충족 미달(punish_assoc 5마리); a|2|x|y: 기계 대조 실패; 기계 대조 실패 1쌍) (마리별 동시 충족 집계).")
    assert s["consequence"].startswith("STD 재설계 여부는 사용자 몫")
    p = R.verdict_sentence(dict(v, verdict="PASS", failing=[], mech_fail=[]), dict(q=0.625, K=16, F=12), SPEC)
    assert p["sentence"] == (
        "조합 지렛대 모델(C3, APL→MBON05 2간선 + MBON05→MBON09/MBON11/MBON01 11간선 제거, E-grid k2-norm s 1.0, "
        "엔진별 z)에서 실제 학습 규칙으로 M2 학습 단위가 섰다 — 넓힌 풀(생성원 턴 306–1985), 오라클 순진·시험 가능 쌍 "
        "조건부(관문 쌍 8개, 판정 가능 8개, 설계 q 0.625 · K 16 · F 12 · k 상한 8, 마리별 동시 충족).")
    assert p["consequence"] == "POOL 배틀 과제(M3)는 별도 선언이 필요하다(T.9.5 좁힘 유지)."
    u = R.verdict_sentence(dict(v, verdict="UNDECIDED", undecided_cause="BAND 잔존"), {}, SPEC)
    assert u["sentence"] == "원인(BAND 잔존)을 기록하고 사용자 판단."
    u4 = R.verdict_sentence(dict(v, verdict="UNDECIDED", undecided_cause="판정 가능 쌍 < 4"), {}, SPEC)   # W.9.10 5
    assert u4["sentence"] == "원인(판정 가능 쌍 < 4)을 기록하고 사용자 판단."
    # W.9.10 6: RN1 ≠ R1 / INVALID pairs with their reasons
    m = R.verdict_sentence(dict(v, verdict="STOP_MACHINE", machine=["p: RN1 ≠ R1"], invalid=["q"],
                                pairs=dict(pairs, q=dict(status="INVALID", reasons=["NaN 관문 통계"]))), {}, SPEC)
    assert m["sentence"] == ("W 학습 측정에서 기계 검사가 맞지 않았다(INVALID 쌍 q(NaN 관문 통계); p: RN1 ≠ R1) — 같은 "
                             "관문 쌍으로 다시 돌리지 않으며(W.5·W.9.8 H8), INVALID_RUN 여부는 사용자 몫이다.")
    # W.9.10 6: INVALID pairs (with reasons) first, then machine reasons, nothing truncated (> 6 entries)
    inv = [f"i{j}" for j in range(5)]
    big = R.verdict_sentence(dict(v, verdict="STOP_MACHINE", machine=[f"m{j}" for j in range(4)], invalid=inv,
                                  pairs={k: dict(status="INVALID", reasons=[f"r{k}"]) for k in inv}), {}, SPEC)
    assert big["sentence"].startswith("W 학습 측정에서 기계 검사가 맞지 않았다(INVALID 쌍 i0(ri0); INVALID 쌍 i1(ri1); "
                                      "INVALID 쌍 i2(ri2); INVALID 쌍 i3(ri3); INVALID 쌍 i4(ri4); m0; m1; m2; m3) — ")
