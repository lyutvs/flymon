"""X's gate decisions and sentences for phase A (X.6, X.9.1.1, X.9.1.3 P2-10; Reading 1): the W reuse facts the
precheck rests on, the W diagnosis check, STOP_REUSE and the point-θ precheck. Phase B adds its gates and the X.3
sentences here. Every number comes from the XSpec passed in."""
from __future__ import annotations

import numpy as np

PASS, INVALID = "PASS", "INVALID"
STOP_REUSE, STOP_OC_UNREACHABLE = "STOP_REUSE", "STOP_OC_UNREACHABLE"

SENTENCES = {
    STOP_REUSE: "X 재사용 조건(X.5 1 · X.4.3 진단)이 깨졌다({why}). X는 W 파일럿·경로와 V 블록을 다시 재는 경로를 갖지 "
                "않으므로 주 세트를 쓰지 않고 멈춘다 — 사용자 몫.",
    STOP_OC_UNREACHABLE: "점 θ 사전 점검(X.9.1.1)에서 파일럿 잡음의 점 추정으로도 F ≤ 32로 G.6 작동 특성 목표를 맞출 수 "
                         "없다 — p_set · q · K · F 어느 설계도 k = 4–8 모두에서 점 검정력 ≥ 0.80과 점 거짓 통과 ≤ 0.05를 "
                         "함께 만족하지 않는다({paren}). 부트스트랩 작동 특성은 계산하지 않았고, 주 세트를 쓰지 않고 "
                         "멈춘다.",
}


def sentence(outcome: str, fields: dict) -> str:
    return SENTENCES[outcome].format(**fields)


def reuse_stop(why: list) -> dict:
    return dict(outcome=STOP_REUSE, reasons=list(why), sentence=sentence(STOP_REUSE, dict(why="; ".join(why))),
                records_unavailable=True, records_reason="OC 전 관문 STOP(X.9.1.3 P2-10)")


def w_reuse_reasons(w_doc: dict, keys: dict, facts: dict, oc_sha, xs) -> list:
    """The W facts the precheck rests on (Reading 2)."""
    why = []
    if keys.get("w_measure_key") != xs.w_measure_key:
        why.append(f"W 측정 키 {keys.get('w_measure_key')} ≠ {xs.w_measure_key}")
    if keys.get("pipeline_key") != xs.w_pipeline_key:
        why.append(f"W 파이프라인 키 {keys.get('pipeline_key')} ≠ {xs.w_pipeline_key}")
    st0 = ((w_doc.get("stage0") or {}).get("decision_files") or {}).get("flymon/brain/w_measure.py")
    if keys.get("w_measure_sha") != xs.w_measure_file_sha or st0 != xs.w_measure_file_sha:
        why.append(f"w_measure.py sha256 {keys.get('w_measure_sha')} / W stage0 {st0} ≠ {xs.w_measure_file_sha}")
    g = facts.get("git") or {}
    if not g.get("tracked") or g.get("dirty"):
        why.append(f"{xs.w_summary} 미커밋")
    if not (facts.get("last") or "").startswith(xs.w_summary_commit):
        why.append(f"{xs.w_summary}의 마지막 커밋 {facts.get('last')} ≠ {xs.w_summary_commit}")
    for b, c in xs.w_commits:
        if not (facts.get("ancestors") or {}).get(c):
            why.append(f"W 커밋 {c}({b})가 HEAD 이력에 없음")
        blk = w_doc.get(b)
        if not isinstance(blk, dict):
            why.append(f"W 블록 {b} 없음")
        elif blk.get("w_measure_key") != xs.w_measure_key:
            why.append(f"W 블록 {b}의 W 측정 키 {blk.get('w_measure_key')}")
    p = w_doc.get("pilot") or {}
    if p.get("outcome") != PASS:
        why.append(f"W 파일럿 블록 {p.get('outcome')}")
    if len(p.get("manifest") or []) != xs.w_pilot_units or len(p.get("pairs") or []) != xs.w_pilot_pairs:
        why.append(f"W 파일럿 매니페스트 {len(p.get('manifest') or [])} / 쌍 {len(p.get('pairs') or [])}")
    if (w_doc.get("oc") or {}).get("detail_sha256") != xs.w_oc_detail_sha or oc_sha != xs.w_oc_detail_sha:
        why.append(f"{xs.w_oc_detail} sha256 {oc_sha} ≠ {xs.w_oc_detail_sha}")
    return why


def _naive_d(w_doc: dict, keys) -> dict:
    return {k: float(w_doc["pilot"]["record"]["pairs"][k]["naive_d"]) for k in keys}


def abs_naive_d(w_doc: dict, keys) -> np.ndarray:
    nd = _naive_d(w_doc, keys)
    return np.array([abs(nd[k]) for k in keys])


def pair_fact_reasons(w_doc: dict, keys, xs) -> list:
    """X.9.1.3: the balanced pilot pairs (|naive_d| < naive_max) are the declared three; V1 keeps v1_pairs."""
    nd = _naive_d(w_doc, keys)
    why = []
    bal = sorted(k for k in keys if abs(nd[k]) < xs.naive_max)
    if bal != sorted(xs.balanced_pairs):
        why.append(f"순진 균형 파일럿 쌍 {bal} ≠ {sorted(xs.balanced_pairs)}")
    n1 = sum(abs(v) < xs.v1_exclude_from for v in nd.values())
    if n1 != xs.v1_pairs:
        why.append(f"V1 잔존 쌍 {n1} ≠ {xs.v1_pairs}")
    return why


def diagnosis_reasons(diag: dict, xs) -> list:
    """X.4.3: the reproduction must match W draw for draw, and over all draws give W's counts."""
    why = []
    if diag["diffs"]:
        d0 = diag["diffs"][0]
        why.append(f"W 보정 진단 재현 불일치 {len(diag['diffs'])}건(첫: 추출 {d0['draw']} {d0['side']})")
    c = diag["counts"]
    want = (xs.w_boot_fail_total, xs.w_boot_fail_power, xs.w_boot_fail_false, xs.w_boot_fail_both)
    got = (c["total"], c["power"], c["false"], c["both"])
    if diag["n_draws"] == xs.boot_draws and got != want:
        why.append(f"W 보정 실패 수 {got} ≠ {want}")
    return why


def precheck_paren(pc: dict) -> str:
    cal = pc["calibration"]
    if not (cal["min"]["ok"] and cal["max"]["ok"]):
        def st(m, h):
            if cal[m].get(h) is None:                        # a failed → b never attempted (x_oc.calibrate)
                return "미시도"
            return cal[m][h].get("status")
        return "보정 불가 — " + "; ".join(f"{m}: a {st(m, 'a')}, b {st(m, 'b')}" for m in ("min", "max"))
    rows = "; ".join(f"p_set {r['p_set']}·q {r['q']}·K {r['K']}·F {r['F']}: " + ", ".join(
        f"k {k} 점 검정력 {pw:.3f} / 점 거짓 통과 {fp:.3f}" for k, pw, fp in zip(r["k"], r["power"], r["false"]))
        for r in pc["at_f32"])
    b = pc["best_power"]
    best = f"p_set {b['p_set']} · q {b['q']} · K {b['K']} · F {b['F']}"
    bp = ", ".join(f"k {k} {v:.3f}" for k, v in zip(pc["axes"]["k"], b["power_by_k"]))
    return f"{rows}; 점 검정력 최대 설계 {best}의 k = 4–8 점 검정력 {bp}"


def precheck_decision(pc: dict) -> dict:
    if pc["passed"]:
        return dict(outcome=PASS, reasons=[], n_passing=len(pc["passing"]))
    return dict(outcome=STOP_OC_UNREACHABLE, reasons=["점 θ 사전 점검 미달(X.9.1.1)"],
                sentence=sentence(STOP_OC_UNREACHABLE, dict(paren=precheck_paren(pc))))
