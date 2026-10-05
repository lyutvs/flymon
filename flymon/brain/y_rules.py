"""Y's 0p decisions and sentences (Y.3.1–Y.3.3, Y.7 0p-b / 0p-c, Y.8): key and digest reasons, STOP_REUSE, the oracle
levels, the filters as count-table columns, the count table (rows all / (b) / (a)) and the early STOP_FEW_PAIRS. Every
number comes from the YSpec passed in. Phase A adds its gates and sentences here and changes none of these.
- Every column requires testable. Balance is strict (<), floors include the bound (≥), all after rounding to
  round_digits. A pair with no oracle value (pair_stats None) counts in no_value and nowhere else; a lever mismatch
  counts in failures and makes the block INVALID (0p plan Reading 4).
- Y red-team P2-11 (adopted): the 0p bit-identity rule binds results/y/oracle.json only. The count table is a derived
  value: count_table is a pure function of the per-pair records oracle.json holds (no measurement, no state, input not
  mutated, order-free), so it is recomputed from oracle.json rather than re-measured. derive() bundles it with the
  other derived values Y.9.2 added: P1-7's yield rule (k range kept iff N_len ≥ k_lo × c; both dropped -> early
  STOP_FEW_PAIRS, with Y.8's "< 4" sentence when N_len < min k_lo, else P1-7's sentence) and P1-4's lenient-level
  quantiles.
- sentences(ys): Y.8's sentences as amended by Y.9.2 (STOP_PILOT_FEW, PASS, FAIL in place; P1-7's yield STOP), every
  number from ys; the templates for phase A's STOP_PILOT_FEW / PASS / FAIL live here so their text is fixed now."""
from __future__ import annotations

import math

import numpy as np

from .y_spec import SPEC

PASS, INVALID, FAIL = "PASS", "INVALID", "FAIL"
STOP_REUSE, STOP_FEW_PAIRS, STOP_PILOT_FEW = "STOP_REUSE", "STOP_FEW_PAIRS", "STOP_PILOT_FEW"
COLUMNS = ("testable", "balanced", "y_strict", "y_lenient", "f2_0", "f2_25", "user31", "user_onesided31")
ROWS = ("all", "b", "a")

def _d(x) -> str:
    """A d′ threshold as the spec prints it (0.5, 1.0): shortest form, at least one decimal."""
    s = f"{float(x):g}"
    return s if "." in s else f"{s}.0"


def _lv(x) -> str:
    """A level / factor as the spec prints it (20, 43, 16, 34.4, 1.5)."""
    return f"{float(x):g}"


def yield_need(k_lo: int, ys) -> int:
    """Y.9.2 P1-7: the smallest lenient-pass count N_len with N_len ≥ k_lo × c."""
    return int(math.ceil(round(k_lo * ys.yield_c, ys.round_digits)))


def sentences(ys) -> dict:
    """Y.8 / Y.9.2 sentence templates with every number from ys; {…} are the spec's 〈…〉 fills."""
    pre = (f"시험 가능 ∧ |d_pre| < {_d(ys.lenient_d)} ∧ 순진 MBON13(X) ≥ {_lv(ys.lenient_a)} ∧ "
           f"순진 MBON05(X) ≥ {_lv(ys.lenient_p)}")
    pre_short = (f"|d_pre| < {_d(ys.lenient_d)} ∧ MBON13(X) ≥ {_lv(ys.lenient_a)} ∧ "
                 f"MBON05(X) ≥ {_lv(ys.lenient_p)}")
    strict = f"|d′| < {_d(ys.naive_max)} ∧ 순진 MBON13(X) ≥ {_lv(ys.c_a)} ∧ 순진 MBON05(X) ≥ {_lv(ys.c_p)}"
    need = " · ".join(f"[{lo}, {hi}]의 {yield_need(lo, ys)}개" for lo, hi in ys.k_ranges)
    k_min = min(lo for lo, _ in ys.k_ranges)
    t0, t1 = ys.pool_turns
    return {
        STOP_REUSE: "Y 재사용 조건(Y.7 {where})이 깨졌다({why}). Y는 W 파일럿 · 경로, X 블록, V 블록, 주 세트 생성원을 "
                    "다시 재거나 고치는 경로를 갖지 않으므로 주 세트 학습 측정 없이 멈춘다 — 사용자 몫.",
        (STOP_FEW_PAIRS, "early"): f"Y 주 세트 {{n}}쌍에서 오라클 사전 거름({pre})을 통과한 쌍이 {{k}}개로 최소 관문 쌍 "
                                   f"수 {k_min}에 못 미쳤다 — 주 세트 학습 측정 없이 멈춘다.",
        (STOP_FEW_PAIRS, "yield"): f"Y 주 세트 {{n}}쌍에서 오라클 사전 거름({pre})을 통과한 쌍이 {{k}}개로, 수율 규칙"
                                   f"(Y.9.2 P1-7, c {_lv(ys.yield_c)})이 요구하는 k 범위 {need}에 모두 못 미쳤다 — "
                                   f"주 세트 학습 측정 없이 멈춘다.",
        STOP_PILOT_FEW: f"Y 파일럿 후보 {ys.pilot_w + ys.pilot_v}쌍(W 균형 {ys.pilot_w} · V 세트 후보 {ys.pilot_v}) 중 "
                        f"자기 파일럿 순진 측정에서 Y 거름({strict})을 통과한 쌍이 {{k}}개, X 냄새를 공유하는 쌍"
                        f"(Earthquake)을 묶은 뒤 {{n_sigma}}개로 최소 {ys.pilot_min_sigma}에 못 미쳤다({{per_pair}}) — "
                        f"다른 출처에서 채우지 않고, 주 세트 학습 측정 없이 멈춘다.",
        PASS: f"조합 지렛대 모델({ys.lever_text})에서 실제 학습 규칙으로 M2 학습 단위가 오라클 사전 거름 통과 쌍 가운데 "
              f"선언 순서로 고른 첫 {{k}} 관문 쌍 중 {{share}}에서 섰다 — 판정 가능 관문 쌍 {{n}} 중 PASS {{m}}"
              f"({{ratio}} ≥ p_set {{p}}) — 넓힌 풀(생성원 턴 {t0}–{t1}), 오라클 시험 가능 ∧ 오라클 관대 사전 거름"
              f"({pre_short}) ∧ 판정 시드 순진 균형(|d′| < {_d(ys.naive_max)}) ∧ 순진 수준 바닥(MBON13(X) ≥ "
              f"{_lv(ys.c_a)} · MBON05(X) ≥ {_lv(ys.c_p)}) 쌍 조건부(관문 쌍 {{k}}개, 설계 p_set {{p}} · q {{q}} · "
              f"K {{K}} · F {{F}} · k 범위 [{{k_lo}}, {{k_hi}}], 마리별 동시 충족; FAIL 쌍 {{fails}}). 거름 밖의 쌍에 "
              f"대한 주장이 아니다.",
        FAIL: f"F.7대로 M2 no-go를 기록한다 — 이 조합 지렛대 · 이 세트 · 오라클 시험 가능 ∧ 오라클 관대 사전 거름"
              f"({pre_short}) ∧ 순진 균형 ∧ 순진 수준 바닥 거름 조건부, 판정 가능 관문 쌍 {{n}} 중 PASS {{m}}"
              f"({{ratio}}), 걸린 조건 {{why}} ({{fails}}; 기계 대조 실패 {{n_mech}}쌍) (마리별 동시 충족 집계, "
              f"쌍 비율 판정).",
    }


SENTENCES = sentences(SPEC)
RECORDS_REASON = "OC 전 관문 STOP(X.9.1.3 P2-10)"


def reuse_stop(why: list, where: str) -> dict:
    return dict(outcome=STOP_REUSE, reasons=list(why),
                sentence=SENTENCES[STOP_REUSE].format(where=where, why="; ".join(why)),
                records_unavailable=True, records_reason=RECORDS_REASON)


def key_reasons(keys: dict, ys) -> list:
    why = []
    if keys.get("w_measure_key") != ys.w_measure_key:
        why.append(f"W 측정 키 {keys.get('w_measure_key')} ≠ {ys.w_measure_key}")
    if keys.get("u_measure_key") != ys.u_measure_key:
        why.append(f"U 측정 키 {keys.get('u_measure_key')} ≠ {ys.u_measure_key}")
    return why


def digest_reasons(js: dict, ys) -> list:
    want = dict(digest_keys=ys.main_digest_keys, n_b=ys.main_n_b, n_a=ys.main_n_a, last_turn=ys.main_last_turn)
    return [f"주 세트 {k}: 재생성 {js.get(k)!r} ≠ 선언 {v!r}" for k, v in want.items() if js.get(k) != v]


def levels(report: dict) -> tuple:
    """(L_A^or(X), L_P^or(X)): medians over the report seeds of odour X's cell-sum counts in report pre (Y.3.1)."""
    pre = report["pre"]
    return (float(np.median(np.asarray(pre["A"], float)[:, 0])), float(np.median(np.asarray(pre["P"], float)[:, 0])))


def passes(p: dict, ys) -> dict:
    if not p.get("value") or not p.get("testable"):
        return {c: False for c in COLUMNS}
    r = lambda x: round(float(x), ys.round_digits)  # noqa: E731
    d, a, q = r(p["d_pre"]), r(p["L_A"]), r(p["L_P"])
    bal = abs(d) < ys.naive_max
    return dict(testable=True, balanced=bal,
                y_strict=bal and a >= ys.c_a and q >= ys.c_p,
                y_lenient=abs(d) < ys.lenient_d and a >= ys.lenient_a and q >= ys.lenient_p,
                f2_0=bal and a >= ys.f2_0[0] and q >= ys.f2_0[1],
                f2_25=bal and a >= ys.f2_25[0] and q >= ys.f2_25[1],
                user31=bal and a >= ys.user_a,
                user_onesided31=d > ys.user_onesided_d and a >= ys.user_a)


def count_table(per: list, ys) -> dict:
    out = {}
    for row in ROWS:
        sel = [p for p in per if row == "all" or p["axis"] == row]
        s = [passes(p, ys) for p in sel]
        out[row] = dict({c: sum(x[c] for x in s) for c in COLUMNS}, n=len(sel),
                        no_value=sum(not p["value"] for p in sel), failures=sum(bool(p["failure"]) for p in sel))
    return out


def yield_rule(n_len: int, ys) -> dict:
    """Y.9.2 P1-7: a k range stays a design candidate only if N_len ≥ k_lo × c; record N_len, c, its basis, the kept
    ranges (c is a coarse value from 3 pairs, which the record says)."""
    need = [dict(k_range=[lo, hi], min_n_len=yield_need(lo, ys)) for lo, hi in ys.k_ranges]
    kept = [e["k_range"] for e in need if n_len >= e["min_n_len"]]
    strict = sum(ok for _, ok in ys.yield_basis_pairs)
    return dict(n_len=int(n_len), c=ys.yield_c, required=need, kept=kept,
                dropped=[e["k_range"] for e in need if e["k_range"] not in kept],
                c_basis=dict(w_pilot_pairs=ys.yield_basis_n, oracle_lenient=[p for p, _ in ys.yield_basis_pairs],
                             pilot_strict=[p for p, ok in ys.yield_basis_pairs if ok],
                             rate=f"{strict}/{len(ys.yield_basis_pairs)}"),
                note=f"c = 1 / ({strict}/{len(ys.yield_basis_pairs)})는 W 파일럿 {len(ys.yield_basis_pairs)}쌍에서 낸 "
                     f"거친 값이다(Y.9.2 P1-7).")


def lenient_levels(per: list, ys) -> dict:
    """Y.9.2 P1-4 (record only): the lenient-pass pairs' L_A^or(X) · L_P^or(X) summary quantiles (numpy's default
    linear interpolation) and their count; no per-pair value. None when no pair passes."""
    sel = [p for p in per if passes(p, ys)["y_lenient"]]
    q = list(ys.level_quantiles)

    def qs(k):
        return [float(v) for v in np.quantile(np.asarray([float(p[k]) for p in sel]), q)] if sel else None
    return dict(n=len(sel), quantiles=q, A=qs("L_A"), P=qs("L_P"))


def derive(per: list, ys) -> dict:
    """Everything the oracle block derives from oracle.json's per-pair records (Y.9.2 P2-11: recomputable, not
    bit-bound): the count table, the yield rule and the lenient-level quantiles."""
    table = count_table(per, ys)
    return dict(counts=table, yield_rule=yield_rule(table["all"]["y_lenient"], ys), lenient_levels=lenient_levels(per, ys))


def oracle_decision(table: dict, ys) -> dict:
    """INVALID on a lever mismatch; the early STOP_FEW_PAIRS when N_len < min k_lo (Y.8's sentence) or when the yield
    rule drops every k range (Y.9.2 P1-7's sentence); otherwise PASS with the k ranges that stay."""
    t = table["all"]
    if t["failures"]:
        return dict(outcome=INVALID, reasons=[f"지렛대 검사 실패 {t['failures']}쌍(edit / CSC / edges)"])
    y = yield_rule(t["y_lenient"], ys)
    k_min = min(lo for lo, _ in ys.k_ranges)
    if not y["kept"]:
        which = "early" if t["y_lenient"] < k_min else "yield"
        return dict(outcome=STOP_FEW_PAIRS, stage="early", rule=which, reasons=[],
                    sentence=sentences(ys)[(STOP_FEW_PAIRS, which)].format(n=t["n"], k=t["y_lenient"]),
                    records_unavailable=True, records_reason=RECORDS_REASON,
                    main_set="미사용(Y.0) — 다음 선언은 이 오라클 값을 공개해야 한다(Y.9)")
    return dict(outcome=PASS, reasons=[], k_ranges=y["kept"])
