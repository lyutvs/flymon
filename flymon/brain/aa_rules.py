"""AA's labels and sentences (AA.8 6273–6278, verbatim), the candidate tail, the reuse / screen-machine reasons, the
budget arithmetic (AA.7 2 / 5) and the env comparison (Z.9.2 P2-11 form, reimplemented — z_rules is not imported)."""
from __future__ import annotations

import numpy as np

PASS, INVALID = "PASS", "INVALID"
STOP_REUSE, STOP_BUDGET, STOP_MACHINE, STOP_NO_PAIRS = "STOP_REUSE", "STOP_BUDGET", "STOP_MACHINE", "STOP_NO_PAIRS"
STOPS = (STOP_REUSE, STOP_BUDGET, STOP_MACHINE, STOP_NO_PAIRS)
STATES = ("untouched", "screened", "trained")
RECORDS_REASON, G6_REASON = "records 상한", "G.6 상한"
FORBIDDEN_RESULT_WORDS = ("PASS", "FAIL", "no-go", "학습이 섰다", "서지 않았다")


def tail(c: dict) -> str:
    return f"(후보 상태: untouched {c['untouched']} · screened {c['screened']} · trained {c['trained']})"


def candidate_counts(cands: list) -> dict:
    return {s: sum(1 for x in cands if x["state"] == s) for s in STATES}


def _stop(outcome, why, sentence, c) -> dict:
    return dict(outcome=outcome, reasons=list(why), sentence=f"{sentence} {tail(c)}", candidates_count=dict(c))


def reuse_stop(why: list, c: dict) -> dict:
    return _stop(STOP_REUSE, why, f"AA 재사용 조건(AA.7 1)이 깨졌다({'; '.join(why)}). AA는 Y · Z 블록, 주 세트 오라클 "
                 "상세(`oracle.json`), 주 세트 생성원, W 측정 경로를 다시 재거나 고치는 경로를 갖지 않으므로 주 세트 학습 측정 "
                 "없이 멈춘다 — 사용자 몫.", c)


def budget_stop(h: str, where: str, state: str, c: dict) -> dict:
    return _stop(STOP_BUDGET, [h], f"남은 추정 비용 {h}가 AA 핵심 상한 12 h를 넘는다({where}) — {state}.", c)


def machine_stop(phase: str, why: str, state: str, c: dict) -> dict:
    return _stop(STOP_MACHINE, [why], f"AA {phase} 측정에서 기계 검사가 맞지 않았다({why}) — 같은 쌍으로 다시 돌리지 "
                 f"않으며(W.5 · W.9.8 H8), `INVALID_RUN` 여부는 사용자 몫이다. {state}.", c)


def no_pairs_stop(r: dict, c: dict) -> dict:
    s = ("AA 순진 거름(판정 시드 순진 pre, F 8 × K 8)에서 주 세트 오라클 관대 통과 31쌍 가운데 Y 최종 거름(|d′| < 0.5 ∧ "
         "MBON13(X) ≥ 20 ∧ MBON05(X) ≥ 43)을 통과한 쌍이 0이다(관대 → 엄격 재현율 0/31; 탈락 사유별 개수 "
         f"균형 {r['balance']} · MBON13(X) 바닥 {r['a_floor']} · MBON05(X) 바닥 {r['p_floor']}) — 1차 집합이 비어 학습 "
         "측정이 1차 추정에 아무것도 보태지 않으므로 주 세트 학습 측정 없이 멈춘다. 31쌍 `screened`, `trained` 0, "
         "218쌍 `untouched` — 사용자 몫.")
    return _stop(STOP_NO_PAIRS, ["k = 0"], s, c)


def reason_counts(rows: list, ys) -> dict:
    r = lambda x: round(float(x), ys.round_digits)      # noqa: E731 — Y.3.2's rounding
    return dict(balance=sum(not abs(r(x["d"])) < ys.naive_max for x in rows),
                a_floor=sum(not r(x["L_A"]) >= ys.c_a for x in rows),
                p_floor=sum(not r(x["L_P"]) >= ys.c_p for x in rows))


def reuse_reasons(facts: dict, s, ys) -> list:
    """AA.7 1 (6252): one Korean reason per broken reuse item. `facts` (built by the runner, read-only):
    y_anc / z_anc {commit: bool}, y_doc / z_doc (summary blocks), keys {w_measure_key, u_measure_key},
    lenient (None, or the `LenientMismatch` message), axes {axis: n} of the lenient items, decl_sha / now_sha
    {file: sha256} over the y/x/w files and z_split.py, absent {path: True when absent}."""
    from .y_rules import digest_reasons
    why = []
    for who, blocks, outs, anc, doc in (("Y", s.y_blocks, s.y_outcomes, facts.get("y_anc") or {}, facts.get("y_doc") or {}),
                                        ("Z", s.z_blocks, s.z_outcomes, facts.get("z_anc") or {}, facts.get("z_doc") or {})):
        want = dict(outs)
        for b, c in blocks:
            if not anc.get(c):
                why.append(f"{who} 블록 {b} 커밋 {c}가 HEAD 이력에 없음")
            got = (doc.get(b) or {}).get("outcome")
            if got != want[b]:
                why.append(f"{who} 블록 {b}의 결과 {got} ≠ {want[b]}")
    keys = facts.get("keys") or {}
    for k in ("w_measure_key", "u_measure_key"):
        if keys.get(k) != getattr(ys, k):
            why.append(f"{k} {keys.get(k)} ≠ {getattr(ys, k)}")
    ydoc = facts.get("y_doc") or {}
    st = (ydoc.get("digest") or {}).get("set") or {}
    why += digest_reasons({k: st.get(k) for k in ("digest_keys", "n_b", "n_a", "last_turn")}, ys)
    orc = ydoc.get("oracle") or {}
    if orc.get("detail_sha256") != s.oracle_sha256:
        why.append(f"Y oracle 블록 detail_sha256 {orc.get('detail_sha256')} ≠ {s.oracle_sha256}")
    if ((orc.get("derived") or {}).get("yield_rule") or {}).get("n_len") != s.n_len:
        why.append(f"Y oracle 블록 N_len ≠ {s.n_len}")
    if facts.get("lenient") is not None:
        why.append(f"oracle.json: {facts['lenient']}")
    elif dict(facts.get("axes") or {}) != dict(s.n_len_axes):
        why.append(f"관대 통과 축별 개수 {dict(facts.get('axes') or {})} ≠ {dict(s.n_len_axes)}")
    decl, now = facts.get("decl_sha") or {}, facts.get("now_sha") or {}
    diff = sorted(f for f in set(decl) | set(now) if decl.get(f) is None or decl.get(f) != now.get(f))
    if diff:
        why.append(f"y_* · x_* · w_* · z_split 가 {s.decl_commit} 이후 바뀜: {diff[:5]}")
    for b in s.y_absent_blocks:
        if b in ydoc:
            why.append(f"Y {b} 블록이 있음")
    absent = facts.get("absent") or {}
    for p in s.y_absent_files:
        if not absent.get(p):
            why.append(f"{p}가 있음")
    return why


def screen_machine_reasons(units: list, counts: dict, want_seeds: dict, n_units: int, k: int) -> list:
    """AA.7 4 (6258): unit count, duplicate keys, seed list = AA.2 formula, finite counts, K probes per unit.
    `counts` / `want_seeds` are keyed (pair, fly); counts are [K, 2, 2]."""
    why = []
    if len(units) != n_units:
        why.append(f"단위 수 {len(units)} ≠ {n_units}")
    seen, uniq = set(), []
    for u in units:
        key = (u["pair"], u["fly"])
        if key in seen:
            why.append(f"단위 키 중복 {key}")
        else:
            seen.add(key)
            uniq.append(u)
    for u in uniq:
        key = (u["pair"], u["fly"])
        tag = f"{u['pair']} fly {u['fly']}"
        if [int(x) for x in u["probe_seeds"]] != [int(x) for x in (want_seeds.get(key) or [])]:
            why.append(f"{tag}: 시드 목록 ≠ 공식")
        c = counts.get(key)
        arr = np.asarray(c if c is not None else [], float)
        if not np.all(np.isfinite(arr)):
            why.append(f"{tag}: 카운트가 유한하지 않음")
        n = len(arr) if arr.ndim else 0
        if n != k:
            why.append(f"{tag}: 프로브 {n} ≠ K {k}")
    return why


def core_ok(elapsed_h: float, remaining_h: float, s) -> bool:
    """AA.7 2 / 5: elapsed + margin × remaining ≤ the core cap (Y.3.2 rounding)."""
    return round(elapsed_h + s.cost_margin * remaining_h - s.core_cap_h, 9) <= 0


def spent_ok(elapsed_h: float, s) -> bool:
    return round(elapsed_h - s.core_cap_h, 9) <= 0


def budget_text(e: float, r: float, s) -> str:
    return f"{e:.2f} h + {s.cost_margin} × {r:.2f} h = {e + s.cost_margin * r:.2f} h"


def env_diff(old: dict, new: dict) -> list:
    """Fields of two env records that differ; for the per-file maps only files present in OLD are compared (a file
    added later is not a difference), a file of OLD missing in NEW is (copied from z_rules.env_diff)."""
    out = []
    for k in sorted(old):
        a, b = old.get(k), new.get(k)
        if isinstance(a, dict):
            b = b or {}
            out += [f"{k}:{f}" for f in sorted(a) if a[f] != b.get(f)]
        elif a != b:
            out.append(k)
    return out


def env_reasons(blocks: list, now: dict) -> list:
    """Z.9.2 P2-11: `now` against the env of every committed block in order (stage0 first)."""
    out = []
    for name, env in blocks:
        out += [f"{name} 대비 {x}" for x in env_diff(env or {}, now)]
    return sorted(set(out))


def _n(x) -> str:
    return "null" if x is None else f"{round(float(x), 3):g}"


_MODEL = ("조합 지렛대 모델(C3, APL→MBON05 2간선 + MBON05→MBON09/MBON11/MBON01 11간선 제거, E-grid k2-norm s 1.0, "
          "z_V)에서 실제 학습 규칙(F.2 R · N + G.5 RN)의 ")
_FILTER = "판정 시드 순진 거름(|d′| < 0.5 ∧ MBON13(X) ≥ 20 ∧ MBON05(X) ≥ 43)을 통과한"
_FLOOR = ("가르친 세포 바닥 몫 중앙 φ_R {phi_r} · φ_P {phi_p} — 바닥 절단을 포함한 값이며(편향 부호 미정) Y 거름 통과"
          "(같은 프로브 실현값에서) 쌍 조건부다. 판정 문턱은 없고 M2 학습 단위의 PASS/FAIL이 아니다.")


def result_sentence(r: dict) -> str:
    """AA.8 (6278): the result sentence (not a verdict). k = 1: the pair-study opening, no pooled part."""
    assoc = (f"보상 연합 d′(ΔV_R1 − ΔV_N1)은 {_n(r['mu_ra'])} [95% CI {_n(r['lo_ra'])}, {_n(r['hi_ra'])}], "
             f"처벌 연합 d′((ΔV_R2 − ΔV_R1) − (ΔV_RN2 − ΔV_RN1))는 {_n(r['mu_pa'])} [{_n(r['lo_pa'])}, {_n(r['hi_pa'])}]"
             "로 추정됐다")
    floor = _FLOOR.format(phi_r=_n(r.get("phi_r")), phi_p=_n(r.get("phi_p")))
    if r["k"] == 1:
        return (f"쌍별 연구(k = 1) — 쌍 {r['key']}의 쌍별 추정: {_MODEL}{assoc} — 주 세트 오라클 관대 통과 31쌍 중 "
                f"{_FILTER} 1쌍, F 8 × K 8, 쌍 추정 = 마리 d′(±10 잘라냄)의 마리 평균. {floor}")
    few = ", '적은 묶음' 표식" if r.get("few") else ""
    under = ", '명목 미달'" if r.get("under") else ""
    if r["method"] == "two_stage":
        how = "묶음-마리 두 단계"
    else:
        how = f"마리 단계(묶음 < 5, 쌍별 값 범위 {_n(r['rng_min'])}–{_n(r['rng_max'])})"
    return (f"{_MODEL}{assoc} — 주 세트 오라클 관대 통과 31쌍 중 {_FILTER} {r['k']}쌍(X 냄새 묶음 {r['g']}개{few}), "
            f"F 8 × K 8, 쌍 추정 = 마리 d′(±10 잘라냄)의 마리 평균, 통합 = 쌍 평균 · {how} 부트스트랩 10 000회 백분위"
            f"(구조 맞춤 포함 확률 {_n(r.get('cov'))}{under}). 쌍 간 SD 보상 {_n(r.get('tau_ra'))} · 처벌 "
            f"{_n(r.get('tau_pa'))}(DL). {floor}")
