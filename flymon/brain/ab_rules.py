"""AB's labels and sentences (AB.8 6838–6882, verbatim with the 〈…〉 slots filled), the candidate-state tail with the
closure rule (AB.9), the reuse / KC / oracle / machine reasons and the budget arithmetic (AB.7 예산: 24 h · × 2.0).
The env comparison is aa_rules.env_reasons (imported; Z.9.2 P2-11 form)."""
from __future__ import annotations

import numpy as np

from .aa_rules import env_diff, env_reasons  # noqa: F401  (re-exported for ab_runner)

PASS, FAIL, UNDECIDED, INVALID = "PASS", "FAIL", "UNDECIDED", "INVALID"
STOP_REUSE, STOP_SET_MISMATCH, STOP_CALIBRATION = "STOP_REUSE", "STOP_SET_MISMATCH", "STOP_CALIBRATION"
STOP_FUTILE = "STOP_FUTILE"                                                   # AB.7 0f (AB.9.3)
STOP_KC_REPRO, STOP_BUDGET, STOP_MACHINE = "STOP_KC_REPRO", "STOP_BUDGET", "STOP_MACHINE"
STOP_FEW_PAIRS, STOP_COVERAGE, STOP_PROTOCOL = "STOP_FEW_PAIRS", "STOP_COVERAGE", "STOP_PROTOCOL"
STOPS = (STOP_REUSE, STOP_SET_MISMATCH, STOP_CALIBRATION, STOP_FUTILE, STOP_KC_REPRO, STOP_BUDGET, STOP_MACHINE, STOP_FEW_PAIRS,
         STOP_COVERAGE, STOP_PROTOCOL)
VERDICTS = (PASS, FAIL, UNDECIDED, STOP_MACHINE, STOP_PROTOCOL)
STATES = ("untouched", "oracled", "screened", "training", "trained")
CLOSURE = "이 지렛대의 M2 판정 트랙은 닫힌다(AB.9 닫힘 규칙)."
RECORDS_REASON = "records 상한"
MODEL = ("조합 지렛대 모델(C3, APL→MBON05 2간선 + MBON05→MBON09/MBON11/MBON01 11간선 제거, E-grid k2-norm s 1.0, z_V)에서 "
         "실제 학습 규칙(F.2 R · N + G.5 RN)은 2세대 넓힌 풀 AB 세트의 Y 거름 통과 {k}쌍(X 라벨 묶음 {gx}개 · 상대 타입 "
         "집합 묶음 {gt}개, F 8 × K 8)에서")
GATE_KO = dict(reward_level="보상 수준", punish_drop="처벌 하락", reward_assoc="보상 연합", punish_assoc="처벌 연합")


def n3(x) -> str:
    return "null" if x is None else f"{round(float(x), 3) + 0.0:g}"


def counts(cands: list) -> dict:
    return {s: sum(1 for x in cands if x["state"] == s) for s in STATES}


def tail(c: dict) -> str:
    """AB.8: every STOP sentence ends with the candidate-state counts and the closure rule."""
    return ("(후보 상태: " + " · ".join(f"`{s}` {int(c.get(s, 0))}" for s in STATES) + ") " + CLOSURE)


def _stop(label: str, why: list, text: str, c: dict, where: str | None = None) -> dict:
    out = dict(outcome=label, reasons=list(why), sentence=f"{text} {tail(c)}", candidates_count=dict(c))
    if where is not None:
        out["stop_stage"] = where
    return out


def reuse_stop(why: list, c: dict) -> dict:
    return _stop(STOP_REUSE, why, f"AB 재사용 조건(AB.2)이 깨졌다({'; '.join(why)}). AB는 V · W · Y · AA의 블록 · "
                 "캐시를 다시 재거나 고치는 경로를 갖지 않으므로 2세대 측정 없이 멈춘다 — 사용자 몫.", c, "0b")


def set_mismatch_stop(where: str, detail: str, c: dict) -> dict:
    what = (f"선언값 {detail}과 다르다" if where == "0c" else "같은 코드의 두 번 생성이 다르다")
    meas = "측정 없음" if where == "0c" else "KC 입력만 측정(냄새 값, L_V 판정 결과 아님)"
    return _stop(STOP_SET_MISMATCH, [detail], f"AB 생성원(AB.3)이 {what} — {meas}. 사용자 몫.", c, where)


def calibration_stop(tag: str, sizes, stat: str, phase: str, cell: int, cp: float, x: int, n: int,
                     alpha=None, c: dict | None = None) -> dict:
    if phase == "sel":
        how = f"선택: 격자 끝 α 0.0001에서도 칸 {cell}의 CP 97.5% 상한 {n3(cp)} > 0.025({x}/{n:,})".replace(",", " ")
    else:
        how = f"검증: 고른 α {alpha}에서 칸 {cell}의 CP 상한 {n3(cp)} > 0.025({x}/{n:,})".replace(",", " ")
    g = tag[1:] if tag.startswith("g") else tag
    return _stop(STOP_CALIBRATION, [f"{tag} {stat} {phase} 칸 {cell}"],
                 f"측정 전 보정 관문(AB.5)에서 대표 구조 g {g}, X 라벨 묶음 크기 {tuple(sizes)}의 {stat} PASS 쪽 동시 "
                 f"놓침이 {how} — 모의 모형 범위 안에서 거짓 PASS를 통제할 수준이 없으므로 2세대 측정 없이 멈춘다(AB 세트 "
                 "미소비). 사용자 몫.", c or {}, "0e")


def _pareto_text(par: dict) -> str:
    """〈시나리오 집합마다 파레토 최소 (g, k) 목록 / 없음〉; a scenario cut by the records cap reads RECORDS_REASON."""
    out = []
    for tag, cells in par.items():
        if cells is None:
            out.append(f"{tag} {RECORDS_REASON}")
        else:
            out.append(f"{tag} " + (" · ".join(f"({g}, {k})" for g, k in cells) if cells else "없음"))
    return "; ".join(out)


def futile_stop(f: dict, c: dict) -> dict:
    """AB.8 STOP_FUTILE〈0f〉. f: aD, aF, aR (0e's g 6 levels), p_hat, x, n, cp ([lo, hi]), g5, g7 (P̂ of the ρ rows),
    pair (3: g 5, 6, 7), group (3), worst {gate, stat, method, share}, pareto {scenario: [[g, k], …] | [] | None}."""
    w, lo, hi = f["worst"], f["cp"][0], f["cp"][1]
    n = f"{int(f['n']):,}".replace(",", " ")
    rec = (f"기록: g 5 {n3(f['g5'])} · g 7 {n3(f['g7'])}; 쌍 배분 {' · '.join(n3(v) for v in f['pair'])}, 묶음 배분 "
           f"{' · '.join(n3(v) for v in f['group'])}; 가장 자주 막은 조건 {GATE_KO[w['gate']]} · {w['stat']} · "
           f"{w['method']} {n3(w['share'])}")
    return _stop(STOP_FUTILE, [f"P̂ {n3(f['p_hat'])} < 0.5"],
                 "측정 전 가망 관문(AB.7 0f, 사용자 지시 2026-10-07)에서 AA 수준 효과와 쌍 간 SD(AA S1 16쌍의 마리 자료를 "
                 "옮겨 다시 표집, μ · τ · ρ · C는 `futility` 블록)와 0e가 g 6 대표 구조에서 고른 수준(α_P^D "
                 f"{f['aD']} · α_P^F {f['aF']} · α_P^R {f['aR']})으로 낸 예상 P(PASS — 네 관문 × 세 통계 × 두 방식 동시)가 "
                 f"{n3(f['p_hat'])}({f['x']}/{n}, CP 95% [{n3(lo)}, {n3(hi)}]) < 0.5다({rec}). AA 크기의 효과가 2세대로 "
                 "그대로 옮겨 가도 UNDECIDED가 PASS보다 흔하다고 예측되므로, UNDECIDED가 유력한 시험에 예산을 쓰지 않고 "
                 "2세대 측정 없이 멈춘다(AB 세트 미사용). 학습 단위에 대한 증거가 아니다(측정이 없음). P̂ ≥ 0.5에 닿는 "
                 f"변경(기록 격자, AB를 바꾸지 않음): {_pareto_text(f['pareto'])}. 사용자 몫.", c, "0f")


def kc_repro_stop(odours: list, diffs: list, c: dict) -> dict:
    return _stop(STOP_KC_REPRO, diffs, f"V 냄새 4개({', '.join(odours)})의 KC 활성 중앙값을 다시 잰 값이 V `kc_input`과 "
                 f"{'; '.join(diffs)}에서 다르다 — KC 경로가 V와 같지 않으므로 2세대 KC 값으로 세트를 정하지 않고 멈춘다. "
                 "사용자 몫.", c, "3")


def budget_stop(stage_label: str, arith: str, state: str, c: dict) -> dict:
    return _stop(STOP_BUDGET, [arith], f"AB {stage_label}에서 {arith}가 핵심 상한 24 h를 넘는다 — {state}. 사용자 몫.",
                 c, stage_label)


def machine_stop(phase: str, items: str, c: dict, where: str) -> dict:
    return _stop(STOP_MACHINE, [items], f"AB {phase} 측정에서 기계 검사가 맞지 않았다({items}) — 같은 쌍으로 다시 "
                 "돌리지 않으며(W.9.8 H8), `INVALID_RUN` 여부는 사용자 몫이다.", c, where)


def few_pairs_oracle(n_rows: int, n_len: int, g: int, c: dict) -> dict:
    return _stop(STOP_FEW_PAIRS, [f"N_len {n_len}, X 라벨 묶음 {g}"],
                 f"AB 세트 {n_rows}행의 오라클에서 관대 사전 거름 통과가 {n_len}쌍(X 라벨 묶음 {g}개)이라 하한(N_len ≥ 8, "
                 "X 라벨 묶음 ≥ 5)에 못 미친다 — 판정 시드를 뽑지 않고 멈춘다. 사용자 몫.", c, "6")


def few_pairs_screen(n_len: int, k: int, gx: int, gt: int, rc: dict, c: dict) -> dict:
    return _stop(STOP_FEW_PAIRS, [f"k {k}, 묶음 {gx} · {gt}"],
                 f"AB 순진 거름(판정 시드 순진 pre, F 8 × K 8)에서 관대 통과 {n_len}쌍 가운데 Y 최종 거름 통과가 {k}쌍"
                 f"(X 라벨 묶음 {gx}개 · 상대 타입 집합 묶음 {gt}개; 탈락 사유별 균형 {rc['balance']} · MBON13(X) 바닥 "
                 f"{rc['a_floor']} · MBON05(X) 바닥 {rc['p_floor']})이라 하한(k ≥ 8, 두 묶음 정의 모두 g ≥ 5)에 못 미친다 — "
                 "학습 없이 멈춘다. 사용자 몫.", c, "7")


def coverage_stop(k: int, sx, st_, stat: str, phase: str, cell: int, cp: float, c: dict) -> dict:
    how = (f"격자 끝 α 0.0001에서도 칸 {cell}의 CP 97.5% 상한 {n3(cp)} > 0.025" if phase == "sel"
           else f"검증 흐름에서 칸 {cell}의 CP 상한 {n3(cp)} > 0.025")
    return _stop(STOP_COVERAGE, [f"S1 {stat} {phase} 칸 {cell}"],
                 f"S1 구조({k}쌍, X 라벨 묶음 크기 {tuple(sx)}, 상대 타입 집합 묶음 크기 {tuple(st_)})에서 {stat}의 PASS "
                 f"쪽 동시 놓침이 {how} — 막대에서의 거짓 PASS를 통제할 수준이 없으므로 학습 없이 멈춘다. 측정 전 "
                 "관문(0e)은 대표 구조에서 통과했고, AB 세트는 오라클(순서 6)부터 소비되었다. 사용자 몫.", c, "7b")


def protocol_stop(pair: str, medians: list, c: dict) -> dict:
    return _stop(STOP_PROTOCOL, [pair], f"AB S1 {pair}에서 기계 대조(두 부호 비율의 마리 중앙값 "
                 f"{n3(medians[0])} · {n3(medians[1])})가 0.75에 못 미친다 — 프로토콜 · 판독 결함이며 새 사전 등록이 "
                 "필요하다(F.7).", c, "9")


# ---------------------------------------------------------------- verdict sentences (AB.8 PASS / FAIL / UNDECIDED)
def head(k: int, gx: int, gt: int) -> str:
    return MODEL.format(k=k, gx=gx, gt=gt)


def pass_sentence(f: dict) -> str:
    """f: k, gx, gt, D_lim / Dfin_lim / R_lim (4 Hedges- or z-scaled unfavourable ends, GATES order), D_pt (4 points,
    Hedges), aD / aF / aR (α_P), k_fin, df, share (4), n10 (4), ninf (4), n_flies, phi_R, phi_P.
    The limits and the points read in AB.8's order (보상 수준 · 보상 연합, then 처벌 하락 · 처벌 연합 = index 0, 2, 1, 3);
    the per-gate shares and counts ("관문별") read in GATES order."""
    L, Fn, R, P = f["D_lim"], f["Dfin_lim"], f["R_lim"], f["D_pt"]
    return (f"{head(f['k'], f['gx'], f['gt'])} 네 관문 모두 효과 크기 기준을 넘었다 — **±10으로 잘라낸 마리 d′의 S1 쌍 "
            f"평균**(Hedges 척도, × J(8) 0.8889 — 막대 척도 관례)의 보정 신뢰 한계(두 방식 중 불리한 끝)가 보상 수준 "
            f"{n3(L[0])} · 보상 연합 {n3(L[2])} > +1, 처벌 하락 {n3(L[1])} · 처벌 연합 {n3(L[3])} < −1(점 {n3(P[0])} · "
            f"{n3(P[2])} · {n3(P[1])} · {n3(P[3])}; 한쪽 α_P^D {f['aD']}), ±∞ 마리를 뺀 유한 마리판도 {n3(Fn[0])} · "
            f"{n3(Fn[2])} · {n3(Fn[1])} · {n3(Fn[3])}로 막대를 넘었고(α_P^F {f['aF']}, 남은 쌍 {f['k_fin']}), 네 비표준화 "
            f"대조가 {n3(R[0])} · {n3(R[2])} · {n3(R[1])} · {n3(R[3])}로 ±0.25 z를 넘었다(α_P^R {f['aR']}). 두 방식 = X "
            f"라벨 묶음 두 단계 부트스트랩(B 10 000) · 두 방향 군집 강건 t(df {f['df']}). 관문별 S1 마리 가운데 기대 방향 "
            f"|d′| ≥ 1인 몫 {' · '.join(n3(x) for x in f['share'])}, ±10 마리 {' · '.join(str(x) for x in f['n10'])}"
            f"(그 가운데 ±∞ {' · '.join(str(x) for x in f['ninf'])}) / {f['n_flies']}마리. 기계 대조 · noplast · RN1 = R1 "
            "통과. → **M2 학습 단위 PASS — 2세대 넓힌 영역(상대 · 대체 상대가 2세대 기본 종), L_V, S1 쌍 평균 한정.** "
            "스펙 5의 '보상 20회 뒤 d′ ≥ 1, 처벌 20회 뒤 하락'을 부록 AB의 개정(쌍 평균 Hedges 척도의 보정 신뢰 한계, 연합 "
            "관문 · 유한 마리판 · 비표준화 최소 효과 포함)으로 판정했다 — 마리마다 d′ ≥ 1이라는 뜻이 아니다. 오류율 보장은 "
            f"AB.5의 모의 모형 범위 안에서다. 가르친 세포 바닥 몫 중앙 φ_R {n3(f['phi_R'])} · φ_P {n3(f['phi_P'])}(편향 "
            "부호 미정), Y 거름 통과(같은 프로브 실현값에서) 쌍 조건부. 기준은 AA의 여유를 알고 정했다(AB.0 1 · 15). POOL "
            "안 확인이 아니며 M3를 자동으로 열지 않는다(T.9.5, AB.9).")


def fail_sentence(f: dict) -> str:
    """f: k, gx, gt, gate (GATES name), sign (+1 / −1), vals (TS, CG ends, Hedges), aF, past_zero (bool)."""
    side = (f"상한 {n3(f['vals'][0])} · {n3(f['vals'][1])} < +1" if f["sign"] > 0
            else f"하한 {n3(f['vals'][0])} · {n3(f['vals'][1])} > −1")
    zero = "그 구간은 0의 잘못된 쪽까지 갔다" if f["past_zero"] else "0은 기대 방향으로 넘었다"
    return (f"{head(f['k'], f['gx'], f['gt'])} 관문 {GATE_KO[f['gate']]}에서 두 방식 모두의 {side}(FAIL 쪽 한쪽 α_F "
            f"{f['aF']} — 관문마다 거짓 FAIL ≤ 0.00625, 본페로니) — ±10으로 잘라낸 마리 d′의 S1 쌍 평균(Hedges 척도)이 "
            f"막대에 확실히 못 미친다; {zero}. → **M2 학습 단위 no-go — 2세대 넓힌 영역, L_V.** D.6 (c) 충족을 넓힌 영역 "
            "범위로 기록한다. 원인을 인코더 · STD · 지렛대에 귀속하지 않는다 — 대안 설명: 판독이 단일 타입, 바닥 절단(부호 "
            "미정), 쌍 조건부 대상, 2세대 냄새가 POOL 밖, (a) 묶음 7개.")


def undecided_sentence(f: dict) -> str:
    """f: k, gx, gt, causes (list of Korean cause strings from ab_estimate.verdict)."""
    return (f"{head(f['k'], f['gx'], f['gt'])} PASS도 FAIL도 아니다 — {'; '.join(f['causes'])}. 같은 세트로 다시 재지 "
            f"않는다. 학습이 없다는 증거가 아니다(AB.0 15). {CLOSURE} 사용자 판단.")


# ---------------------------------------------------------------- reasons
def reuse_reasons(f: dict, s) -> list:
    """AB.2 재사용 대조 (order 0b). `f` (built by the runner, read only): v_anc / aa_anc {commit: bool}, kc_sha,
    w_digest {digest_keys, n_b, n_a, last_turn}, aa_doc (summary), lv (ab_pairs.lv_sources), v_check (list of
    v_pairs.check_v_set reasons), keys {w_measure_key, u_measure_key}, want_keys, decl_sha / now_sha {file: sha},
    fut_src {n_units, n_pairs, missing, bad_sha} (the 0f source: AA S1's learn units, AB.2 / AB.9.3)."""
    why = []
    for b, c in s.v_commits:
        if not (f.get("v_anc") or {}).get(c):
            why.append(f"V 블록 {b} 커밋 {c}가 HEAD 이력에 없음")
    if f.get("kc_sha") != s.kc_input_sha256:
        why.append(f"results/v/kc_input.json sha256 {f.get('kc_sha')} ≠ {s.kc_input_sha256}")
    wd = f.get("w_digest") or {}
    want = dict(digest_keys=s.w_digest_keys, n_b=s.w_n_b, n_a=s.w_n_a, last_turn=s.w_last_turn)
    for k, v in want.items():
        if wd.get(k) != v:
            why.append(f"W 주 세트 {k} {wd.get(k)} ≠ {v}")
    doc = f.get("aa_doc") or {}
    for b, c in s.aa_blocks:
        if not (f.get("aa_anc") or {}).get(c):
            why.append(f"AA 블록 {b} 커밋 {c}가 HEAD 이력에 없음")
        if (doc.get(b) or {}).get("outcome") != PASS:
            why.append(f"AA 블록 {b}의 결과 {(doc.get(b) or {}).get('outcome')} ≠ PASS")
    why += list((f.get("lv") or {}).get("reasons") or [])
    src = f.get("fut_src") or {}                                              # AB.2: the 0f source (AB.9.3)
    if src.get("n_units") != s.fut_src_units or src.get("n_pairs") != s.fut_src_pairs:
        why.append(f"가망 관문 원천 단위 {src.get('n_units')}개 · 쌍 {src.get('n_pairs')}개 ≠ {s.fut_src_units} · "
                   f"{s.fut_src_pairs}")
    if src.get("missing") or src.get("bad_sha"):
        why.append(f"가망 관문 원천 캐시 파일 없음 {list(src.get('missing') or [])[:3]} · sha256 불일치 "
                   f"{list(src.get('bad_sha') or [])[:3]}")
    why += [f"V 세트: {x}" for x in f.get("v_check") or []]
    for k in ("w_measure_key", "u_measure_key"):
        if (f.get("keys") or {}).get(k) != (f.get("want_keys") or {}).get(k):
            why.append(f"{k} {(f.get('keys') or {}).get(k)} ≠ {(f.get('want_keys') or {}).get(k)}")
    decl, now = f.get("decl_sha") or {}, f.get("now_sha") or {}
    if not decl:
        why.append(f"{s.decl_commit} 시점 t_* · v_* · w_* · y_* · z_split · aa_* sha 목록이 비어 있음")
    diff = sorted(x for x in set(decl) | set(now) if decl.get(x) is None or decl.get(x) != now.get(x))
    if diff:
        why.append(f"t_* · v_* · w_* · y_* · z_split · aa_*가 {s.decl_commit} 이후 바뀜: {diff[:5]}")
    return why


def kc_repro_diffs(odours: list, v_kc: dict, got: dict) -> list:
    """Order 3: the four V odours re-measured against V's kc_input values, both engines, bit for bit."""
    out = []
    for e in ("none", "lever"):
        for o in odours:
            a, b = (v_kc.get(e) or {}).get(o), (got.get(e) or {}).get(o)
            if a is None or b is None or float(a) != float(b):
                out.append(f"{e} · {o}: {b!r} 대 {a!r}")
    return out


def oracle_machine_reasons(results: list, rows: list, seeds: dict, n_rep: int) -> list:
    """Order 6 (AB.7 6): unit count, key duplicates, seed lists, finite report values."""
    why = []
    if len(results) != len(rows):
        why.append(f"오라클 단위 수 {len(results)} ≠ {len(rows)}")
    keys = [g["key"] for g in results]
    dup = sorted({k for k in keys if keys.count(k) > 1})
    if dup:
        why.append(f"오라클 키 중복 {dup[:3]}")
    for g in results:
        res = g.get("result") or {}
        for side in ("pre", "R1", "R2"):
            rep = (res.get("report") or {}).get(side) or {}
            for cell in ("A", "P"):
                a = np.asarray(rep.get(cell, []), float)
                if a.shape[:1] != (n_rep,) or not np.isfinite(a).all():
                    why.append(f"{g['key']}: report {side}.{cell} 모양 · 유한성")
                    break
    del seeds
    return why


def reason_counts(rows: list, ys) -> dict:
    r = lambda x: round(float(x), ys.round_digits)      # noqa: E731
    return dict(balance=sum(not abs(r(x["d"])) < ys.naive_max for x in rows),
                a_floor=sum(not r(x["L_A"]) >= ys.c_a for x in rows),
                p_floor=sum(not r(x["L_P"]) >= ys.c_p for x in rows))


# ---------------------------------------------------------------- budget (AB.7 예산)
def core_ok(elapsed_h: float, remaining_h: float, s) -> bool:
    return round(elapsed_h + s.cost_margin * remaining_h - s.core_cap_h, 9) <= 0


def spent_ok(elapsed_h: float, s) -> bool:
    return round(elapsed_h - s.core_cap_h, 9) <= 0


def budget_text(e: float, r: float, s) -> str:
    return f"{e:.2f} h + {s.cost_margin} × {r:.2f} h = {e + s.cost_margin * r:.2f} h"


def spent_text(e: float) -> str:
    return f"실측 누적 {e:.2f} h"
