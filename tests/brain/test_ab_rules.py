"""AB.8 sentences (verbatim slots), the candidate tail and closure rule, reuse / KC / oracle reasons, the budget
arithmetic (24 h, × 2.0)."""
import dataclasses

from flymon.brain import ab_rules as R
from flymon.brain.ab_spec import SPEC
from flymon.brain.y_spec import SPEC as Y

C = dict(untouched=150, oracled=10, screened=20, training=0, trained=4)


def test_tail_and_every_stop_ends_with_counts_and_closure():
    t = R.tail(C)
    assert t == ("(후보 상태: `untouched` 150 · `oracled` 10 · `screened` 20 · `training` 0 · `trained` 4) "
                 "이 지렛대의 M2 판정 트랙은 닫힌다(AB.9 닫힘 규칙).")
    stops = [R.reuse_stop(["x"], C), R.set_mismatch_stop("0c", "n_combos: 1 · 2", C), R.set_mismatch_stop("4", "d", C),
             R.calibration_stop("g7", (7, 6, 4, 3, 2, 2, 1), "R", "sel", 19, 0.031, 120, 5000, None, C),
             R.calibration_stop("g7", (7, 6, 4, 3, 2, 2, 1), "D", "ver", 19, 0.026, 230, 10000, 0.0001, C),
             R.futile_stop(_futile_fill(), C),
             R.kc_repro_stop(list(SPEC.kc_repro), ["none · ELECTRIC|GROUND: 0.1 대 0.2"], C),
             R.budget_stop("0e 전", "0.30 h + 2.0 × 12.00 h = 24.30 h", "2세대 측정 없음", C),
             R.machine_stop("오라클", "단위 수 1 ≠ 2", C, "6"), R.few_pairs_oracle(150, 7, 4, C),
             R.few_pairs_screen(30, 7, 5, 6, dict(balance=10, a_floor=12, p_floor=1), C),
             R.coverage_stop(20, (5, 4, 3, 3, 2, 2, 1), (3,) * 6 + (2,), "R", "ver", 21, 0.026, C),
             R.protocol_stop("a|3|Surf|Earthquake", [0.7, 0.9], C)]
    for s in stops:
        assert s["sentence"].endswith(R.CLOSURE) and s["outcome"] in R.STOPS and s["candidates_count"] == C
    assert "선택: 격자 끝 α 0.0001에서도 칸 19의 CP 97.5% 상한 0.031 > 0.025(120/5 000)" in stops[3]["sentence"]
    assert "검증: 고른 α 0.0001에서 칸 19의 CP 상한 0.026 > 0.025(230/10 000)" in stops[4]["sentence"]
    assert stops[1]["sentence"].startswith("AB 생성원(AB.3)이 선언값 n_combos: 1 · 2과 다르다 — 측정 없음.")
    assert "KC 입력만 측정(냄새 값, L_V 판정 결과 아님)" in stops[2]["sentence"]


def _futile_fill(pareto=None):
    return dict(aD=0.0005, aF=0.00025, aR=0.0005, p_hat=0.1985, x=397, n=2000, cp=[0.181, 0.217], g5=0.25, g7=0.12,
                pair=[0.45, 0.39, 0.3], group=[0.08, 0.07, 0.05],
                worst=dict(gate="punish_drop", stat="D", method="CG", share=0.61),
                pareto=pareto if pareto is not None else {"all": [[7, 28], [9, 20]], "noskew": [[6, 24]], "sd2": [],
                                                          "sd1": None})


def test_futile_sentence_slots():
    t = R.futile_stop(_futile_fill(), C)
    assert t["outcome"] == R.STOP_FUTILE and t["stop_stage"] == "0f" and R.STOP_FUTILE in R.STOPS
    s = t["sentence"]
    assert "(α_P^D 0.0005 · α_P^F 0.00025 · α_P^R 0.0005)" in s and "0.199(397/2 000, CP 95% [0.181, 0.217]) < 0.5다" in s
    assert "쌍 배분 0.45 · 0.39 · 0.3, 묶음 배분 0.08 · 0.07 · 0.05; 가장 자주 막은 조건 처벌 하락 · D · CG 0.61" in s
    assert "AB를 바꾸지 않음): all (7, 28) · (9, 20); noskew (6, 24); sd2 없음; sd1 records 상한. 사용자 몫." in s


def test_reuse_reasons_futility_source():
    f = _facts()
    f["fut_src"] = dict(n_units=383, n_pairs=16, missing=["k"], bad_sha=[])
    why = R.reuse_reasons(f, SPEC)
    assert len(why) == 2 and why[0].startswith("가망 관문 원천 단위 383개")


def _pass_fill():
    return dict(k=20, gx=6, gt=9, D_lim=[3.1, -2.9, 3.4, -2.5], Dfin_lim=[3.0, -2.8, 3.3, -2.4],
                R_lim=[1.2, -0.9, 1.5, -0.8], D_pt=[5.0, -4.0, 5.5, -3.9], aD=0.0001, aF=0.00025, aR=0.0001, k_fin=20,
                df=5, share=[0.8, 0.7, 0.8, 0.6], n10=[30, 12, 25, 10], ninf=[4, 1, 3, 0], n_flies=160, phi_R=0.0,
                phi_P=0.125)


def test_pass_fail_undecided_sentences():
    p = R.pass_sentence(_pass_fill())
    assert p.startswith("조합 지렛대 모델(C3, APL→MBON05 2간선 + MBON05→MBON09/MBON11/MBON01 11간선 제거, E-grid k2-norm "
                        "s 1.0, z_V)에서 실제 학습 규칙(F.2 R · N + G.5 RN)은 2세대 넓힌 풀 AB 세트의 Y 거름 통과 20쌍(X "
                        "라벨 묶음 6개 · 상대 타입 집합 묶음 9개, F 8 × K 8)에서 네 관문 모두 효과 크기 기준을 넘었다")
    assert "보상 수준 3.1 · 보상 연합 3.4 > +1, 처벌 하락 -2.9 · 처벌 연합 -2.5 < −1" in p
    assert "POOL 안 확인이 아니며 M3를 자동으로 열지 않는다(T.9.5, AB.9)." in p and "df 5" in p
    f = R.fail_sentence(dict(k=20, gx=6, gt=9, gate="reward_assoc", sign=1.0, vals=[0.4, 0.5], aF=0.00025,
                             past_zero=False))
    assert "관문 보상 연합에서 두 방식 모두의 상한 0.4 · 0.5 < +1" in f and "0은 기대 방향으로 넘었다" in f
    u = R.undecided_sentence(dict(k=20, gx=6, gt=9, causes=["관문 punish_assoc의 R 구간이 ±0.25 z를 걸침"]))
    assert "PASS도 FAIL도 아니다 — 관문 punish_assoc의 R 구간이 ±0.25 z를 걸침." in u and R.CLOSURE in u


def _facts():
    return dict(v_anc={c: True for _, c in SPEC.v_commits}, aa_anc={c: True for _, c in SPEC.aa_blocks},
                kc_sha=SPEC.kc_input_sha256, w_digest=dict(digest_keys=SPEC.w_digest_keys, n_b=167, n_a=82,
                                                         last_turn=1967),
                aa_doc={b: dict(outcome="PASS") for b, _ in SPEC.aa_blocks}, lv=dict(reasons=[]), v_check=[],
                keys=dict(w_measure_key=Y.w_measure_key, u_measure_key=Y.u_measure_key),
                want_keys=dict(w_measure_key=Y.w_measure_key, u_measure_key=Y.u_measure_key),
                decl_sha={"flymon/brain/w_spec.py": "a"}, now_sha={"flymon/brain/w_spec.py": "a"},
                fut_src=dict(n_units=SPEC.fut_src_units, n_pairs=SPEC.fut_src_pairs, missing=[], bad_sha=[]))


def test_reuse_reasons():
    assert R.reuse_reasons(_facts(), SPEC) == []
    f = _facts()
    f["aa_doc"]["learn"] = dict(outcome="STOP_BUDGET")
    f["kc_sha"] = "x"
    f["now_sha"] = {"flymon/brain/w_spec.py": "b"}
    f["lv"] = dict(reasons=["lv_odour 냄새 107개 ≠ 108"])
    why = R.reuse_reasons(f, SPEC)
    assert len(why) == 4 and any("AA 블록 learn" in w for w in why) and any("lv_odour" in w for w in why)
    assert R.reuse_reasons(dict(_facts(), decl_sha={}), SPEC)[0].startswith("ea408a2 시점")


def test_kc_repro_and_budget():
    v = {"none": {"A|B": 0.05}, "lever": {"A|B": 0.04}}
    assert R.kc_repro_diffs(["A|B"], v, v) == []
    assert R.kc_repro_diffs(["A|B"], v, {"none": {"A|B": 0.05}, "lever": {"A|B": 0.0400001}}) == \
        ["lever · A|B: 0.0400001 대 0.04"]
    assert R.core_ok(4.0, 10.0, SPEC) and not R.core_ok(4.0, 10.001, SPEC)
    assert R.budget_text(0.3, 10.3, SPEC) == "0.30 h + 2.0 × 10.30 h = 20.90 h"
    assert R.reason_counts([dict(d=0.6, L_A=19.9999999999, L_P=50)], Y) == dict(balance=1, a_floor=0, p_floor=0)


# ---------------------------------------------------------------- the AB.8 text itself (located by content)
import re  # noqa: E402
from pathlib import Path  # noqa: E402

SPEC_MD = Path(__file__).resolve().parents[2] / "docs/superpowers/specs/2026-09-14-flymon-design.md"


def _pattern(t: str) -> str:
    t = t.replace("**", "").replace("… 같은 앞부분 … ", "")
    out, depth, buf = "", 0, ""
    for ch in t:
        if ch == "〈":
            if depth == 0:
                out, buf = out + re.escape(buf) + ".*?", ""
            depth += 1
        elif ch == "〉":
            depth -= 1
        elif depth == 0:
            buf += ch
    return out + re.escape(buf)


def _quoted(prefix: str) -> str:
    lines = [ln for ln in SPEC_MD.read_text().splitlines() if ln.startswith(prefix)]
    assert len(lines) == 1, prefix
    return re.search(r'"(.*)"', lines[0]).group(1)


def test_every_sentence_matches_ab8_verbatim():
    strip = lambda s: s.split(" (후보 상태")[0]                       # noqa: E731
    cases = [("- **`STOP_REUSE`**(0b):", strip(R.reuse_stop(["x"], C)["sentence"])),
             ("- **`STOP_SET_MISMATCH`**(0c · 4):", strip(R.set_mismatch_stop("0c", "a: 1 · 2", C)["sentence"])),
             ("- **`STOP_CALIBRATION`**〈0e〉:",
              strip(R.calibration_stop("g7", (7, 6), "R", "sel", 19, 0.031, 120, 5000, None, C)["sentence"])),
             ("- **`STOP_FUTILE`**〈0f〉:", strip(R.futile_stop(_futile_fill(), C)["sentence"])),
             ("- **`STOP_KC_REPRO`**(3):", strip(R.kc_repro_stop(["A"], ["d"], C)["sentence"])),
             ("- **`STOP_BUDGET`**(0e 전", strip(R.budget_stop("0e 전", "a", "b", C)["sentence"])),
             ("- **`STOP_MACHINE`**(6 · 7 · 8):", strip(R.machine_stop("오라클", "x", C, "6")["sentence"])),
             ("- **`STOP_FEW_PAIRS`〈오라클〉**(6):", strip(R.few_pairs_oracle(150, 7, 4, C)["sentence"])),
             ("- **`STOP_FEW_PAIRS`〈거름〉**(7):",
              strip(R.few_pairs_screen(30, 7, 5, 6, dict(balance=1, a_floor=2, p_floor=0), C)["sentence"])),
             ("- **`STOP_COVERAGE`**(7b):", strip(R.coverage_stop(20, (5,), (3,), "R", "ver", 21, 0.03, C)["sentence"])),
             ("- **`STOP_PROTOCOL`**(9, F.7 행):", strip(R.protocol_stop("a|3", [0.7, 0.9], C)["sentence"])),
             ("- **`PASS` 문장(AB.10에 그대로)**:", R.pass_sentence(_pass_fill()))]
    for prefix, text in cases:
        assert re.fullmatch(_pattern(_quoted(prefix)), text.replace("**", ""), re.S), prefix
    for prefix, text in (("- **`FAIL` 문장**:", R.fail_sentence(dict(k=20, gx=6, gt=9, gate="reward_assoc", sign=1.0,
                                                                   vals=[0.4, 0.5], aF=0.00025, past_zero=True))),
                         ("- **`UNDECIDED` 문장**:", R.undecided_sentence(dict(k=20, gx=6, gt=9, causes=["c"])))):
        body = text.replace("**", "").split("F 8 × K 8)에서 ", 1)[1]
        assert re.fullmatch(_pattern(_quoted(prefix)), body, re.S), prefix


# ---------------------------------------------------------------- full strings, typed from AB.8 (slots filled by hand)
TAIL = ("(후보 상태: `untouched` 150 · `oracled` 10 · `screened` 20 · `training` 0 · `trained` 4) "
        "이 지렛대의 M2 판정 트랙은 닫힌다(AB.9 닫힘 규칙).")
HEAD = ("조합 지렛대 모델(C3, APL→MBON05 2간선 + MBON05→MBON09/MBON11/MBON01 11간선 제거, E-grid k2-norm s 1.0, z_V)에서 "
        "실제 학습 규칙(F.2 R · N + G.5 RN)은 2세대 넓힌 풀 AB 세트의 Y 거름 통과 20쌍(X 라벨 묶음 6개 · 상대 타입 집합 묶음 "
        "9개, F 8 × K 8)에서")


def test_stop_sentences_full_strings():
    cases = [
        (R.reuse_stop(["x", "y"], C), R.STOP_REUSE, "0b",
         "AB 재사용 조건(AB.2)이 깨졌다(x; y). AB는 V · W · Y · AA의 블록 · 캐시를 다시 재거나 고치는 경로를 갖지 않으므로 "
         "2세대 측정 없이 멈춘다 — 사용자 몫."),
        (R.set_mismatch_stop("0c", "n_combos: 1 · 2", C), R.STOP_SET_MISMATCH, "0c",
         "AB 생성원(AB.3)이 선언값 n_combos: 1 · 2과 다르다 — 측정 없음. 사용자 몫."),
        (R.set_mismatch_stop("4", "d", C), R.STOP_SET_MISMATCH, "4",
         "AB 생성원(AB.3)이 같은 코드의 두 번 생성이 다르다 — KC 입력만 측정(냄새 값, L_V 판정 결과 아님). 사용자 몫."),
        (R.calibration_stop("g7", (7, 6, 4, 3, 2, 2, 1), "R", "sel", 19, 0.031, 120, 5000, None, C), R.STOP_CALIBRATION,
         "0e",
         "측정 전 보정 관문(AB.5)에서 대표 구조 g 7, X 라벨 묶음 크기 (7, 6, 4, 3, 2, 2, 1)의 R PASS 쪽 동시 놓침이 선택: "
         "격자 끝 α 0.0001에서도 칸 19의 CP 97.5% 상한 0.031 > 0.025(120/5 000) — 모의 모형 범위 안에서 거짓 PASS를 "
         "통제할 수준이 없으므로 2세대 측정 없이 멈춘다(AB 세트 미소비). 사용자 몫."),
        (R.calibration_stop("g5", (5, 5), "D", "ver", 3, 0.026, 230, 10000, 0.0001, C), R.STOP_CALIBRATION, "0e",
         "측정 전 보정 관문(AB.5)에서 대표 구조 g 5, X 라벨 묶음 크기 (5, 5)의 D PASS 쪽 동시 놓침이 검증: 고른 α 0.0001에서 "
         "칸 3의 CP 상한 0.026 > 0.025(230/10 000) — 모의 모형 범위 안에서 거짓 PASS를 통제할 수준이 없으므로 2세대 측정 "
         "없이 멈춘다(AB 세트 미소비). 사용자 몫."),
        (R.futile_stop(_futile_fill(), C), R.STOP_FUTILE, "0f",
         "측정 전 가망 관문(AB.7 0f, 사용자 지시 2026-10-07)에서 AA 수준 효과와 쌍 간 SD(AA S1 16쌍의 마리 자료를 옮겨 다시 "
         "표집, μ · τ · ρ · C는 `futility` 블록)와 0e가 g 6 대표 구조에서 고른 수준(α_P^D 0.0005 · α_P^F 0.00025 · α_P^R "
         "0.0005)으로 낸 예상 P(PASS — 네 관문 × 세 통계 × 두 방식 동시)가 0.199(397/2 000, CP 95% [0.181, 0.217]) < 0.5다"
         "(기록: g 5 0.25 · g 7 0.12; 쌍 배분 0.45 · 0.39 · 0.3, 묶음 배분 0.08 · 0.07 · 0.05; 가장 자주 막은 조건 처벌 하락 · "
         "D · CG 0.61). AA 크기의 효과가 2세대로 그대로 옮겨 가도 UNDECIDED가 PASS보다 흔하다고 예측되므로, UNDECIDED가 "
         "유력한 시험에 예산을 쓰지 않고 2세대 측정 없이 멈춘다(AB 세트 미사용). 학습 단위에 대한 증거가 아니다(측정이 없음). "
         "P̂ ≥ 0.5에 닿는 변경(기록 격자, AB를 바꾸지 않음): all (7, 28) · (9, 20); noskew (6, 24); sd2 없음; sd1 records "
         "상한. 사용자 몫."),
        (R.kc_repro_stop(["A|B", "C|D"], ["none · A|B: 0.1 대 0.2", "lever · C|D: 0.3 대 0.4"], C), R.STOP_KC_REPRO, "3",
         "V 냄새 4개(A|B, C|D)의 KC 활성 중앙값을 다시 잰 값이 V `kc_input`과 none · A|B: 0.1 대 0.2; lever · C|D: 0.3 대 "
         "0.4에서 다르다 — KC 경로가 V와 같지 않으므로 2세대 KC 값으로 세트를 정하지 않고 멈춘다. 사용자 몫."),
        (R.budget_stop("0e 전", "0.30 h + 2.0 × 12.00 h = 24.30 h", "2세대 측정 없음", C), R.STOP_BUDGET, "0e 전",
         "AB 0e 전에서 0.30 h + 2.0 × 12.00 h = 24.30 h가 핵심 상한 24 h를 넘는다 — 2세대 측정 없음. 사용자 몫."),
        (R.machine_stop("오라클", "단위 수 1 ≠ 2", C, "6"), R.STOP_MACHINE, "6",
         "AB 오라클 측정에서 기계 검사가 맞지 않았다(단위 수 1 ≠ 2) — 같은 쌍으로 다시 돌리지 않으며(W.9.8 H8), "
         "`INVALID_RUN` 여부는 사용자 몫이다."),
        (R.few_pairs_oracle(150, 7, 4, C), R.STOP_FEW_PAIRS, "6",
         "AB 세트 150행의 오라클에서 관대 사전 거름 통과가 7쌍(X 라벨 묶음 4개)이라 하한(N_len ≥ 8, X 라벨 묶음 ≥ 5)에 못 "
         "미친다 — 판정 시드를 뽑지 않고 멈춘다. 사용자 몫."),
        (R.few_pairs_screen(30, 7, 5, 6, dict(balance=10, a_floor=12, p_floor=1), C), R.STOP_FEW_PAIRS, "7",
         "AB 순진 거름(판정 시드 순진 pre, F 8 × K 8)에서 관대 통과 30쌍 가운데 Y 최종 거름 통과가 7쌍(X 라벨 묶음 5개 · "
         "상대 타입 집합 묶음 6개; 탈락 사유별 균형 10 · MBON13(X) 바닥 12 · MBON05(X) 바닥 1)이라 하한(k ≥ 8, 두 묶음 정의 "
         "모두 g ≥ 5)에 못 미친다 — 학습 없이 멈춘다. 사용자 몫."),
        (R.coverage_stop(20, (5, 4, 3, 3, 2, 2, 1), (3,) * 6 + (2,), "R", "ver", 21, 0.026, C), R.STOP_COVERAGE, "7b",
         "S1 구조(20쌍, X 라벨 묶음 크기 (5, 4, 3, 3, 2, 2, 1), 상대 타입 집합 묶음 크기 (3, 3, 3, 3, 3, 3, 2))에서 R의 PASS "
         "쪽 동시 놓침이 검증 흐름에서 칸 21의 CP 상한 0.026 > 0.025 — 막대에서의 거짓 PASS를 통제할 수준이 없으므로 학습 "
         "없이 멈춘다. 측정 전 관문(0e)은 대표 구조에서 통과했고, AB 세트는 오라클(순서 6)부터 소비되었다. 사용자 몫."),
        (R.coverage_stop(20, (5, 4), (3, 2), "D_fin", "sel", 2, 0.031, C), R.STOP_COVERAGE, "7b",
         "S1 구조(20쌍, X 라벨 묶음 크기 (5, 4), 상대 타입 집합 묶음 크기 (3, 2))에서 D_fin의 PASS 쪽 동시 놓침이 격자 끝 "
         "α 0.0001에서도 칸 2의 CP 97.5% 상한 0.031 > 0.025 — 막대에서의 거짓 PASS를 통제할 수준이 없으므로 학습 없이 "
         "멈춘다. 측정 전 관문(0e)은 대표 구조에서 통과했고, AB 세트는 오라클(순서 6)부터 소비되었다. 사용자 몫."),
        (R.protocol_stop("a|3|Surf|Earthquake", [0.7, 0.9], C), R.STOP_PROTOCOL, "9",
         "AB S1 a|3|Surf|Earthquake에서 기계 대조(두 부호 비율의 마리 중앙값 0.7 · 0.9)가 0.75에 못 미친다 — 프로토콜 · "
         "판독 결함이며 새 사전 등록이 필요하다(F.7)."),
    ]
    for got, label, where, body in cases:
        assert got["sentence"] == body + " " + TAIL, label
        assert got["outcome"] == label and got["stop_stage"] == where and got["candidates_count"] == C


def test_verdict_sentences_full_strings():
    assert R.head(20, 6, 9) == HEAD
    assert R.pass_sentence(_pass_fill()) == (
        HEAD + " 네 관문 모두 효과 크기 기준을 넘었다 — **±10으로 잘라낸 마리 d′의 S1 쌍 평균**(Hedges 척도, × J(8) 0.8889 — "
        "막대 척도 관례)의 보정 신뢰 한계(두 방식 중 불리한 끝)가 보상 수준 3.1 · 보상 연합 3.4 > +1, 처벌 하락 -2.9 · 처벌 "
        "연합 -2.5 < −1(점 5 · 5.5 · -4 · -3.9; 한쪽 α_P^D 0.0001), ±∞ 마리를 뺀 유한 마리판도 3 · 3.3 · -2.8 · -2.4로 "
        "막대를 넘었고(α_P^F 0.00025, 남은 쌍 20), 네 비표준화 대조가 1.2 · 1.5 · -0.9 · -0.8로 ±0.25 z를 넘었다(α_P^R "
        "0.0001). 두 방식 = X 라벨 묶음 두 단계 부트스트랩(B 10 000) · 두 방향 군집 강건 t(df 5). 관문별 S1 마리 가운데 "
        "기대 방향 |d′| ≥ 1인 몫 0.8 · 0.7 · 0.8 · 0.6, ±10 마리 30 · 12 · 25 · 10(그 가운데 ±∞ 4 · 1 · 3 · 0) / "
        "160마리. 기계 대조 · noplast · RN1 = R1 통과. → **M2 학습 단위 PASS — 2세대 넓힌 영역(상대 · 대체 상대가 2세대 "
        "기본 종), L_V, S1 쌍 평균 한정.** 스펙 5의 '보상 20회 뒤 d′ ≥ 1, 처벌 20회 뒤 하락'을 부록 AB의 개정(쌍 평균 "
        "Hedges 척도의 보정 신뢰 한계, 연합 관문 · 유한 마리판 · 비표준화 최소 효과 포함)으로 판정했다 — 마리마다 d′ ≥ 1이라는 "
        "뜻이 아니다. 오류율 보장은 AB.5의 모의 모형 범위 안에서다. 가르친 세포 바닥 몫 중앙 φ_R 0 · φ_P 0.125(편향 부호 "
        "미정), Y 거름 통과(같은 프로브 실현값에서) 쌍 조건부. 기준은 AA의 여유를 알고 정했다(AB.0 1 · 15). POOL 안 확인이 "
        "아니며 M3를 자동으로 열지 않는다(T.9.5, AB.9).")
    fail_rest = (" — ±10으로 잘라낸 마리 d′의 S1 쌍 평균(Hedges 척도)이 막대에 확실히 못 미친다; {}. → **M2 학습 단위 no-go — "
                 "2세대 넓힌 영역, L_V.** D.6 (c) 충족을 넓힌 영역 범위로 기록한다. 원인을 인코더 · STD · 지렛대에 귀속하지 "
                 "않는다 — 대안 설명: 판독이 단일 타입, 바닥 절단(부호 미정), 쌍 조건부 대상, 2세대 냄새가 POOL 밖, (a) 묶음 "
                 "7개.")
    assert R.fail_sentence(dict(k=20, gx=6, gt=9, gate="reward_assoc", sign=1.0, vals=[0.4, 0.5], aF=0.00025,
                                past_zero=False)) == (
        HEAD + " 관문 보상 연합에서 두 방식 모두의 상한 0.4 · 0.5 < +1(FAIL 쪽 한쪽 α_F 0.00025 — 관문마다 거짓 FAIL ≤ "
        "0.00625, 본페로니)" + fail_rest.format("0은 기대 방향으로 넘었다"))
    assert R.fail_sentence(dict(k=20, gx=6, gt=9, gate="punish_drop", sign=-1.0, vals=[-0.3, 0.2], aF=0.00025,
                                past_zero=True)) == (
        HEAD + " 관문 처벌 하락에서 두 방식 모두의 하한 -0.3 · 0.2 > −1(FAIL 쪽 한쪽 α_F 0.00025 — 관문마다 거짓 FAIL ≤ "
        "0.00625, 본페로니)" + fail_rest.format("그 구간은 0의 잘못된 쪽까지 갔다"))
    assert R.undecided_sentence(dict(k=20, gx=6, gt=9, causes=["a", "b"])) == (
        HEAD + " PASS도 FAIL도 아니다 — a; b. 같은 세트로 다시 재지 않는다. 학습이 없다는 증거가 아니다(AB.0 15). 이 지렛대의 "
        "M2 판정 트랙은 닫힌다(AB.9 닫힘 규칙). 사용자 판단.")


def test_labels_and_states():
    assert R.STATES == ("untouched", "oracled", "screened", "training", "trained")
    assert R.VERDICTS == ("PASS", "FAIL", "UNDECIDED", "STOP_MACHINE", "STOP_PROTOCOL")
    assert set(R.STOPS) == {"STOP_REUSE", "STOP_SET_MISMATCH", "STOP_CALIBRATION", "STOP_FUTILE", "STOP_KC_REPRO",
                            "STOP_BUDGET", "STOP_MACHINE", "STOP_FEW_PAIRS", "STOP_COVERAGE", "STOP_PROTOCOL"}
    assert R.counts([dict(state="untouched"), dict(state="trained"), dict(state="trained")]) == \
        dict(untouched=1, oracled=0, screened=0, training=0, trained=2)
    assert R.tail({}) == ("(후보 상태: `untouched` 0 · `oracled` 0 · `screened` 0 · `training` 0 · `trained` 0) "
                          "이 지렛대의 M2 판정 트랙은 닫힌다(AB.9 닫힘 규칙).")
    assert R.spent_ok(24.0, SPEC) and not R.spent_ok(24.000001, SPEC) and R.spent_text(3.456) == "실측 누적 3.46 h"
    assert R.env_diff is not None and R.env_reasons is not None
