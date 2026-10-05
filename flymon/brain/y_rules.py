"""Y's 0p decisions and sentences (Y.3.1–Y.3.3, Y.7 0p-b / 0p-c, Y.8): key and digest reasons, STOP_REUSE, the oracle
levels, the filters as count-table columns, the count table (rows all / (b) / (a)) and the early STOP_FEW_PAIRS. Every
number comes from the YSpec passed in. Phase A adds its gates and sentences here and changes none of these.
- Every column requires testable. Balance is strict (<), floors include the bound (≥), all after rounding to
  round_digits. A pair with no oracle value (pair_stats None) counts in no_value and nowhere else; a lever mismatch
  counts in failures and makes the block INVALID (0p plan Reading 4).
- Y red-team P2-11 (adopted): the 0p bit-identity rule binds results/y/oracle.json only. The count table is a derived
  value: count_table is a pure function of the per-pair records oracle.json holds (no measurement, no state, input not
  mutated, order-free), so it is recomputed from oracle.json rather than re-measured."""
from __future__ import annotations

import numpy as np

PASS, INVALID = "PASS", "INVALID"
STOP_REUSE, STOP_FEW_PAIRS = "STOP_REUSE", "STOP_FEW_PAIRS"
COLUMNS = ("testable", "balanced", "y_strict", "y_lenient", "f2_0", "f2_25", "user31", "user_onesided31")
ROWS = ("all", "b", "a")

SENTENCES = {
    STOP_REUSE: "Y 재사용 조건(Y.7 {where})이 깨졌다({why}). Y는 W 파일럿 · 경로, X 블록, V 블록, 주 세트 생성원을 "
                "다시 재거나 고치는 경로를 갖지 않으므로 주 세트 학습 측정 없이 멈춘다 — 사용자 몫.",
    (STOP_FEW_PAIRS, "early"): "Y 주 세트 {n}쌍에서 오라클 사전 거름(시험 가능 ∧ |d_pre| < 1.0 ∧ 순진 MBON13(X) ≥ 16 ∧ "
                               "순진 MBON05(X) ≥ 34.4)을 통과한 쌍이 {k}개로 최소 관문 쌍 수 4에 못 미쳤다 — 주 세트 "
                               "학습 측정 없이 멈춘다.",
}
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


def oracle_decision(table: dict, ys) -> dict:
    t = table["all"]
    if t["failures"]:
        return dict(outcome=INVALID, reasons=[f"지렛대 검사 실패 {t['failures']}쌍(edit / CSC / edges)"])
    k_min = min(lo for lo, _ in ys.k_ranges)
    if t["y_lenient"] < k_min:
        return dict(outcome=STOP_FEW_PAIRS, stage="early", reasons=[],
                    sentence=SENTENCES[(STOP_FEW_PAIRS, "early")].format(n=t["n"], k=t["y_lenient"]),
                    records_unavailable=True, records_reason=RECORDS_REASON,
                    main_set="미사용(Y.0) — 다음 선언은 이 오라클 값을 공개해야 한다(Y.9)")
    return dict(outcome=PASS, reasons=[])
