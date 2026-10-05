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
  number from ys; the templates for phase A's STOP_PILOT_FEW / PASS / FAIL live here so their text is fixed now.
Phase A (appended; nothing above changes): the final filter as ONE function (filter_values / filter_passes /
final_filter — the pilot admission, the main-set final filter of phase B and Y.9.2 P1-5's R-pre all call it; d′ is
w_verdict's naive d′, the levels are w_records.pilot_record's naive medians), Y.4's X-odour merge, admission and 4a
gates (INVALID → STOP_PILOT_FEW on n_Σ < 5 → STOP_PILOT_NO_EFFECT by W.9.8 H6), Y.7 1's X reuse facts, the precheck
decision with Y.8's sentence, and the record-only P1-4 level quantiles, P1-6 flips / P1-7 basis and fact flags."""
from __future__ import annotations

import math

import numpy as np

from . import w_verdict as WV
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


# ================================================================ phase A (Y.7 orders 0–5) — appended
STOP_PILOT_NO_EFFECT, STOP_OC_UNREACHABLE = "STOP_PILOT_NO_EFFECT", "STOP_OC_UNREACHABLE"


def sentences_a(ys) -> dict:
    """Y.8's phase-A sentences that sentences(ys) does not hold; every number from ys."""
    return {
        STOP_PILOT_NO_EFFECT: "Y 파일럿(거름 통과 {n}쌍)에서 조합 지렛대의 F.2 학습 효과가 {cond} — 주 세트 학습 측정 "
                              "없이 멈춘다.",
        (STOP_OC_UNREACHABLE, "precheck"): (
            f"점 θ 사전 점검(Y.6.1)에서 Y 파일럿(거름 통과 {{n}}쌍, Σ 추정 {{n_sigma}}쌍) 잡음의 점 추정으로도 "
            f"F ≤ {ys.f_max}로 G.6 작동 특성 목표를 맞출 수 없다 — p_set · q · K · F · k 범위 어느 설계도 그 k 범위의 "
            f"모든 k에서 점 검정력 ≥ {ys.p_power:.2f}과 점 거짓 통과 ≤ {ys.p_false:.2f}를 함께 만족하지 않는다"
            f"({{paren}}). 부트스트랩 작동 특성은 계산하지 않았고, 주 세트 학습 측정 없이 멈춘다."),
    }


def filter_values(pre, z: dict) -> tuple:
    """Y.3.1 on naive pre counts [..., F, K, 2, 2]: (d′, L_A(X), L_P(X)) over the F × K probes of each leading index.
    d′ = w_verdict.dprime of ΔV flattened (w_verdict.naive_dprime's computation); the levels are the medians of
    MBON13(X) / MBON05(X) (w_records.pilot_record's naive_median)."""
    a = np.asarray(pre, float)
    lead = a.shape[:-4]
    d = WV.dprime(WV.dv(a, z).reshape(*lead, -1))
    la = np.median(a[..., WV.A, WV.X].reshape(*lead, -1), -1)
    lp = np.median(a[..., WV.P, WV.X].reshape(*lead, -1), -1)
    return d, la, lp


def filter_passes(d, la, lp, ys):
    """Y.3.2 after rounding to round_digits: |d′| < naive_max (strict), L_A ≥ c_A, L_P ≥ c_P (bounds included); a
    non-finite d′ fails."""
    def r(x):
        return np.round(np.asarray(x, float), ys.round_digits)
    d = r(d)
    return np.isfinite(d) & (np.abs(d) < ys.naive_max) & (r(la) >= ys.c_a) & (r(lp) >= ys.c_p)


def final_filter(pre, z: dict, ys) -> dict:
    """One pair's final filter (Y.3.3 2, Y.4 admission, Y.9.2 P1-5): the shared function."""
    d, la, lp = filter_values(pre, z)
    return dict(d=float(d), L_A=float(la), L_P=float(lp), passed=bool(filter_passes(d, la, lp, ys)))


def x_odour(key: str) -> str:
    """The X odour of a row key axis|turn|x|y (plan Reading 5)."""
    return key.split("|")[2]


def merge_groups(keys: list) -> list:
    """Y.4: pairs sharing the X odour are one Σ_pair unit; groups of positions, in first-appearance order."""
    groups = {}
    for i, k in enumerate(keys):
        groups.setdefault(x_odour(k), []).append(i)
    return list(groups.values())


def admission(cands: dict, z: dict, ys) -> dict:
    """Y.4 입장 규칙: every candidate's own pilot pre {key: [F, K, 2, 2]} through final_filter (order kept)."""
    return {k: final_filter(pre, z, ys) for k, pre in cands.items()}


def _per_pair(adm: dict) -> str:
    return "; ".join(f"{k} d′ {v['d']:.3f} · MBON13(X) {v['L_A']:g} · MBON05(X) {v['L_P']:g}" for k, v in adm.items())


def pilot_gates(adm: dict, rec_admitted, machine: list, ys, w_spec) -> dict:
    """Y.4 4a in order: a machine reason → INVALID (W.3 3); n_Σ (admitted pairs after the X-odour merge) <
    pilot_min_sigma → STOP_PILOT_FEW (Y.9.2 P1-3); W.9.8 H6 on the admitted pairs' record (w_rules.pilot) →
    STOP_PILOT_NO_EFFECT; else PASS."""
    from . import w_rules
    keys = [k for k, v in adm.items() if v["passed"]]
    groups = merge_groups(keys)
    base = dict(admitted=keys, n_admitted=len(keys), n_sigma=len(groups),
                groups=[[keys[i] for i in g] for g in groups])
    if machine:
        return dict(base, outcome=INVALID, reasons=list(machine))
    if len(groups) < ys.pilot_min_sigma:
        return dict(base, outcome=STOP_PILOT_FEW, reasons=[f"n_Σ {len(groups)} < {ys.pilot_min_sigma}"],
                    sentence=sentences(ys)[STOP_PILOT_FEW].format(k=len(keys), n_sigma=len(groups),
                                                                   per_pair=_per_pair(adm)),
                    records_unavailable=True, records_reason=RECORDS_REASON)
    h6 = w_rules.pilot(rec_admitted, [], w_spec)
    if h6["outcome"] != w_rules.PASS:
        return dict(base, outcome=STOP_PILOT_NO_EFFECT, reasons=list(h6["reasons"]), values=h6.get("values"),
                    sentence=sentences_a(ys)[STOP_PILOT_NO_EFFECT].format(n=len(keys), cond="; ".join(h6["reasons"])),
                    records_unavailable=True, records_reason=RECORDS_REASON)
    return dict(base, outcome=PASS, reasons=[], values=h6.get("values"))


def x_reasons(f: dict, ys) -> list:
    """Y.7 1's X part. f = {x_doc, ancestors {commit: bool}, last (x_learning.json's last commit), git {tracked, dirty},
    diag_sha, x_files {path: sha256 now}}: X blocks stage0 / precheck present with their commits in HEAD's history,
    x_learning.json tracked, clean and last changed by the precheck commit, precheck_diag.json's sha256, and x_*
    unchanged (= X stage0 block's decision_files)."""
    why = []
    g = f.get("git") or {}
    if not g.get("tracked") or g.get("dirty"):
        why.append(f"{ys.x_summary} 미커밋")
    doc = f.get("x_doc") or {}
    for b, c in ys.x_commits:
        if not (f.get("ancestors") or {}).get(c):
            why.append(f"X 커밋 {c}({b})가 HEAD 이력에 없음")
        if b not in doc:
            why.append(f"X 블록 {b} 없음")
    last_c = ys.x_commits[-1][1]
    if not (f.get("last") or "").startswith(last_c):
        why.append(f"{ys.x_summary}의 마지막 커밋 {f.get('last')} ≠ {last_c}")
    if f.get("diag_sha") != ys.x_precheck_diag_sha256:
        why.append(f"{ys.x_precheck_diag} sha256 {f.get('diag_sha')} ≠ {ys.x_precheck_diag_sha256}")
    want = (doc.get("stage0") or {}).get("decision_files") or {}
    have = f.get("x_files") or {}
    if not want or have != want:
        bad = sorted(k for k in set(want) | set(have) if want.get(k) != have.get(k))
        why.append(f"x_* 파일 sha256 ≠ X stage0 decision_files: {bad}")
    return why


def w_block_reasons(f: dict, ys) -> list:
    """Y.7 1's W part beyond X's _reuse (whose phase A checked only pilot / oc): W blocks reuse · path · pilot · oc
    with their commits (744d1bc · 5620f95 · 5fbc4c8 · f30ae35) in HEAD's history and each on the W measurement key.
    f = {w_ancestors {commit: bool}, w_blocks {block: its w_measure_key}}."""
    why = []
    for b, c in ys.w_commits:
        if not (f.get("w_ancestors") or {}).get(c):
            why.append(f"W 커밋 {c}({b})가 HEAD 이력에 없음")
        got = (f.get("w_blocks") or {}).get(b)
        if got != ys.w_measure_key:
            why.append(f"W 블록 {b}의 W 측정 키 {got}")
    return why


def precheck_paren(pc: dict) -> str:
    """Y.8's parenthesis: "보정 불가 — …" when a θ̂ calibration failed, else the F = f_max rows per (p_set, q, K, k
    range) and the best-power design's k-wise point power."""
    cal = pc["calibration"]
    if not (cal["min"]["ok"] and cal["max"]["ok"]):
        def st(m):
            c = cal[m]
            b = "미시도" if not c.get("b") else "/".join(x["status"] for x in c["b"])
            return f"{m}: a {c['a']['status']}, b {b}"
        return "보정 불가 — " + "; ".join(st(m) for m in ("min", "max"))
    rows = "; ".join(
        f"p_set {r['p_set']}·q {r['q']}·K {r['K']}·F {r['F']}·k 범위 [{r['k_range'][0]}, {r['k_range'][1]}]: "
        + ", ".join(f"k {k} 점 검정력 {pw:.3f} / 점 거짓 통과 {fp:.3f}" for k, pw, fp in zip(r["k"], r["power"],
                                                                                         r["false"]))
        for r in pc["at_f32"])
    b = pc["best_power"]
    best = f"p_set {b['p_set']} · q {b['q']} · K {b['K']} · F {b['F']} · k 범위 [{b['k_range'][0]}, {b['k_range'][1]}]"
    bp = ", ".join(f"k {k} {v:.3f}" for k, v in zip(b["k"], b["power_by_k"]))
    return f"{rows}; 점 검정력 최대 설계 {best}의 k별 점 검정력 {bp}"


def precheck_decision(pc: dict, n: int, n_sigma: int, ys) -> dict:
    if pc["passed"]:
        return dict(outcome=PASS, reasons=[], n_passing=len(pc["passing"]))
    return dict(outcome=STOP_OC_UNREACHABLE, reasons=["점 θ 사전 점검 미달(Y.6.1)"],
                sentence=sentences_a(ys)[(STOP_OC_UNREACHABLE, "precheck")].format(
                    n=n, n_sigma=n_sigma, paren=precheck_paren(pc)))


def level_compare(oracle_levels: dict, adm: dict, ys) -> dict:
    """Y.9.2 P1-4 (record only): block oracle's lenient-pass level quantiles (0p code's summary, no per-pair value)
    beside the admitted Y pilot pairs' naive L_A · L_P at the same quantiles."""
    sel = [v for v in adm.values() if v["passed"]]
    q = list(ys.level_quantiles)

    def qs(k):
        return [float(x) for x in np.quantile(np.asarray([v[k] for v in sel], float), q)] if sel else None
    return dict(oracle_lenient=oracle_levels, pilot=dict(n=len(sel), quantiles=q, A=qs("L_A"), P=qs("L_P")),
                note="기록 전용(Y.9.2 P1-4) — 선택·STOP에 쓰지 않는다")


def flip_record(oracle: dict, pilot: dict, ys) -> dict:
    """Y.9.2 P1-6 (record only): oracle {key: oracle.json-shaped record (value, testable, d_pre, L_A, L_P)}, pilot
    {key: (naive d′, MBON13(X) median, MBON05(X) median)} over the same keys. Strict-vs-strict flips, lenient-vs-
    pilot-strict mismatches, the lenient pre-filter's recall [caught, pilot-strict], P1-7's basis (lenient count, how
    many of them pass pilot-strict, c), and whether these equal the spec's facts."""
    keys = [k for k in pilot if k in oracle]
    o = {k: passes(oracle[k], ys) for k in keys}
    s = {k: bool(filter_passes(*pilot[k], ys)) for k in keys}
    flips = [k for k in keys if o[k]["y_strict"] != s[k]]
    mism = [k for k in keys if o[k]["y_lenient"] != s[k]]
    lenient = [k for k in keys if o[k]["y_lenient"]]
    strict_len = [k for k in lenient if s[k]]
    c = len(lenient) / len(strict_len) if strict_len else None
    basis_ok = (len(lenient), len(strict_len)) == (len(ys.yield_basis_pairs), sum(ok for _, ok in ys.yield_basis_pairs))
    return dict(n=len(keys), strict_flips=flips, lenient_mismatch=mism,
                recall=[sum(o[k]["y_lenient"] and s[k] for k in keys), sum(s.values())],
                matches_spec=bool(sorted(flips) == sorted(ys.flip_strict) and sorted(mism) == sorted(ys.flip_lenient)),
                yield_basis=dict(lenient=len(lenient), strict_of_lenient=len(strict_len), c=c,
                                 matches_spec=bool(basis_ok and c == ys.yield_c)),
                note="기록 전용(Y.9.2 P1-6 · P1-7) — 문턱을 바꾸지 않는다(P0-1)")


def fact_flags(values: dict, facts: dict, ys) -> dict:
    """Record-only: does a measured (d, L_A, L_P) equal the spec's printed fact (d to fact_digits)? None when the key
    was not measured."""
    out = {}
    for k, (d, a, p) in facts.items():
        v = values.get(k)
        out[k] = None if v is None else bool(round(float(v[0]), ys.fact_digits) == d and float(v[1]) == a
                                             and float(v[2]) == p)
    return out
