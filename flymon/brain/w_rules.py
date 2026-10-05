"""W's gate decisions and sentences (W.3, W.7 as amended by W.9.3 / W.9.4 / W.9.6 / W.9.7 / W.9.8 / W.9.9). The pair
and overall verdict live in w_verdict (the authoritative judgement code); this module decides the gates around it and
fills the closing sentences (〈…〉 → {field}). Every number comes from the WSpec passed in. W.9.10 (the user's answers
to the plan's OPEN 1-7) fixes STOP_OC_UNREACHABLE's text (3), STOP_FEW_PAIRS (4), UNDECIDED's cause (5), STOP_MACHINE
(6) and FAIL's parenthesis (7)."""
from __future__ import annotations

PASS, INVALID, NOT_READ, SEALED, READ, INVALID_RUN = "PASS", "INVALID", "NOT_READ", "SEALED", "READ", "INVALID_RUN"
STOP_REUSE, STOP_W_PATH_REPRO = "STOP_REUSE", "STOP_W_PATH_REPRO"
STOP_PILOT_NO_EFFECT, STOP_OC_UNREACHABLE = "STOP_PILOT_NO_EFFECT", "STOP_OC_UNREACHABLE"
STOP_BUDGET, STOP_FEW_PAIRS = "STOP_BUDGET", "STOP_FEW_PAIRS"
V_PASS, V_FAIL, V_UNDECIDED, V_STOP_MACHINE = "PASS", "FAIL", "UNDECIDED", "STOP_MACHINE"
LEVER_TXT = "C3, APL→MBON05 2간선 + MBON05→MBON09/MBON11/MBON01 11간선 제거, E-grid k2-norm s 1.0, 엔진별 z"

SENTENCES = {
    # STOP_REUSE's sentence is W's own wording: the spec names the stop but quotes no sentence (plan Reading 19);
    # disclosed in W.10.
    STOP_REUSE: "W 재사용 조건(W.3 1)이 깨졌다({why}). W는 V의 블록을 다시 재는 경로를 갖지 않으므로 주 세트를 쓰지 "
                "않고 멈춘다 — 사용자 몫.",
    STOP_W_PATH_REPRO: "W 학습 경로가 {ref}을 재현하지 못했다({diff}).",
    STOP_PILOT_NO_EFFECT: "파일럿에서 조합 지렛대의 F.2 학습 효과가 {cond} — 주 세트를 쓰지 않고 멈춘다.",
    STOP_OC_UNREACHABLE: "파일럿 잡음에서 F ≤ 32로 G.6 작동 특성 목표를 맞출 수 없다({p_by_k}).",
    STOP_BUDGET: "남은 추정 비용 {h}가 W 상한 24 h를 넘는다.",
    STOP_FEW_PAIRS: "W 주 세트 {n}쌍에서 순진 프로브 균형·오라클 시험 가능 쌍이 {k}개로 최소 4에 못 미쳤다.",
    V_STOP_MACHINE: "W 학습 측정에서 기계 검사가 맞지 않았다({why}) — 같은 관문 쌍으로 다시 돌리지 않으며(W.5·W.9.8 "
                    "H8), INVALID_RUN 여부는 사용자 몫이다.",
    V_UNDECIDED: "원인({cause})을 기록하고 사용자 판단.",
    V_PASS: "조합 지렛대 모델(" + LEVER_TXT + ")에서 실제 학습 규칙으로 M2 학습 단위가 섰다 — 넓힌 풀(생성원 턴 "
            "306–1985), 오라클 순진·시험 가능 쌍 조건부(관문 쌍 {k}개, 판정 가능 {m}개, 설계 q {q} · K {K} · F {F} · "
            "k 상한 {k_max}, 마리별 동시 충족).",
    V_FAIL: "F.7대로 M2 no-go를 기록한다 — 이 조합 지렛대·이 세트·오라클 거름 조건부({fails}; 기계 대조 실패 "
            "{n_mech}쌍) (마리별 동시 충족 집계).",
}
CONSEQUENCE = {
    V_PASS: "POOL 배틀 과제(M3)는 별도 선언이 필요하다(T.9.5 좁힘 유지).",
    V_FAIL: "STD 재설계 여부는 사용자 몫(F.7은 D.6 (c) 충족을 말하지만, 이 판정이 지렛대 엔진 조건부임을 함께 적는다).",
    V_UNDECIDED: "사용자 판단(F.7).",
    V_STOP_MACHINE: "사용자 몫(W.7, W.9.8 H8).",
}


def sentence(outcome: str, fields: dict) -> str:
    return SENTENCES[outcome].format(**fields)


# ================================================================ W.3 1: reuse
def reuse(v_dec: dict, v_doc: dict, v_git: dict, ancestors: dict, last_v_commit: str | None, u_key: str,
          code_key: str, t_key: str, spec) -> dict:
    """V's own reuse condition (R, T, U through v_rules.reuse on V's spec) plus V's blocks: V's summary tracked and
    clean and last changed by V's judge commit, the four V commits in HEAD's history, V's blocks z / kc_input / set /
    judge on V's keys (U's measurement key literal, R's shared key, T's key), V's judgement SELECTED, z_V as
    declared."""
    why = list(v_dec.get("reasons") or [])
    if u_key != spec.u_measure_key_u:
        why.append(f"U 측정 키 {u_key} ≠ {spec.u_measure_key_u}")
    if code_key != spec.r_shared_key or t_key != spec.t_measure_key_t:
        why.append(f"공유 키 {code_key} / T 키 {t_key}")
    if not v_git.get("tracked") or v_git.get("dirty"):
        why.append(f"{spec.v_summary} 미커밋")
    judge_commit = dict(spec.v_commits)["judge"]
    if not (last_v_commit or "").startswith(judge_commit):
        why.append(f"{spec.v_summary}의 마지막 커밋 {last_v_commit} ≠ V 판정 {judge_commit}")
    for b, c in spec.v_commits:
        if not ancestors.get(c):
            why.append(f"V 커밋 {c}({b})가 HEAD 이력에 없음")
        blk = v_doc.get(b)
        if not isinstance(blk, dict):
            why.append(f"V 블록 {b} 없음")
            continue
        if (blk.get("u_measure_key"), blk.get("code_key"), blk.get("t_measure_key")) != (
                spec.u_measure_key_u, spec.r_shared_key, spec.t_measure_key_t):
            why.append(f"V 블록 {b}의 키")
    if (v_doc.get("judge") or {}).get("band") != spec.v_band:
        why.append(f"V 판정 {(v_doc.get('judge') or {}).get('band')}")
    zv = (v_doc.get("z") or {}).get("z_V") or {}
    if {k: tuple(round(float(x), spec.z_v_digits) for x in v) for k, v in zv.items()} != spec.z_v():
        why.append(f"V z_V {zv}")
    if why:
        return dict(outcome=STOP_REUSE, reasons=why, sentence=sentence(STOP_REUSE, dict(why="; ".join(why))))
    return dict(outcome=PASS, reasons=[])


# ================================================================ W.3 2: the path gate
def path(checks: list) -> dict:
    """checks = [dict(ref, diffs, invalid)] in order; a defect is INVALID, a difference STOP_W_PATH_REPRO (the first
    failing check in the sentence, every one listed)."""
    bad = [m for c in checks for m in c.get("invalid", [])]
    if bad:
        return dict(outcome=INVALID, reasons=bad, failed=[])
    failed = [c for c in checks if c["diffs"]]
    if failed:
        c = failed[0]
        return dict(outcome=STOP_W_PATH_REPRO, reasons=[],
                    failed=[dict(ref=x["ref"], diffs=x["diffs"]) for x in failed],
                    sentence=sentence(STOP_W_PATH_REPRO, dict(ref=c["ref"], diff="; ".join(c["diffs"][:6]))))
    return dict(outcome=PASS, reasons=[], failed=[])


# ================================================================ W.9.8 H6: the pilot's stop
def pilot(rec: dict, machine: list, spec) -> dict:
    """INVALID on a machine reason; then H6 (i)-(iii) on the pilot pairs' fly medians; else PASS."""
    if machine:
        return dict(outcome=INVALID, reasons=machine)
    meds = [p["median"] for p in rec["pairs"].values()]
    n = len(meds)
    r_ok = sum(m["reward_assoc"] >= spec.no_effect_d for m in meds) / n
    p_ok = sum(m["punish_assoc"] <= -spec.no_effect_d for m in meds) / n
    rev = sum(m["reward_assoc"] < 0 or m["punish_assoc"] > 0 for m in meds) / n
    fl = rec["floor_both_share"]
    hit = []
    if not (r_ok >= spec.no_effect_share and p_ok >= spec.no_effect_share):
        hit.append(f"(i) 보상 연합 d′ ≥ 0.5 쌍 비율 {r_ok:.3f}·처벌 연합 d′ ≤ −0.5 쌍 비율 {p_ok:.3f}(둘 다 ≥ 0.5 아님)")
    if rev > 0.5:
        hit.append(f"(ii) 보상 또는 처벌 연합 부호가 반대인 쌍 비율 {rev:.3f}(> 0.5)")
    if fl > spec.floor_share_max:
        hit.append(f"(iii) X·Y 함께 바닥인 프로브 비율 {fl:.3f}(> 0.5)")
    vals = dict(reward_share=r_ok, punish_share=p_ok, reversed_share=rev, floor_both_share=fl)
    if hit:
        return dict(outcome=STOP_PILOT_NO_EFFECT, reasons=hit, values=vals,
                    sentence=sentence(STOP_PILOT_NO_EFFECT, dict(cond="; ".join(hit))))
    return dict(outcome=PASS, reasons=[], values=vals)


# ================================================================ W.9.3 / H1: the OC's outcome
def oc(doc: dict, spec) -> dict:
    if doc.get("selected"):
        return dict(outcome=PASS, reasons=[], design=doc["selected"])
    p_by_k = oc_unreachable_text(doc, spec)
    return dict(outcome=STOP_OC_UNREACHABLE, reasons=["선택 규칙을 만족하는 설계 없음"],
                sentence=sentence(STOP_OC_UNREACHABLE, dict(p_by_k=p_by_k)))


def oc_unreachable_text(doc: dict, spec) -> str:
    """W.9.10 3: for every (q, K) the bootstrap limits at F = 32 for each k = 4-8 (w_oc.limits_at_f on the per-k
    arrays power_lo_by_k / false_hi_by_k: 5th / 95th percentile over draws, worst cluster level), as "검정력 하한 /
    거짓 통과 상한" per k; or the calibration status when the targets were unreachable."""
    if not doc.get("reachable"):
        cal = doc.get("calibration") or {}

        def st(m, ab):
            return ((cal.get(m) or {}).get(ab) or {}).get("status")
        return "보정 불가 — " + "; ".join(f"{m}: a {st(m, 'a')}, b {st(m, 'b')}" for m in ("min", "max"))
    from . import w_oc
    out = []
    for r in w_oc.limits_at_f(doc, 32):
        per_k = ", ".join(f"k {k} 검정력 하한 {lo:.3f} / 거짓 통과 상한 {hi:.3f}"
                          for k, lo, hi in zip(r["k"], r["power_lo"], r["false_hi"]))
        out.append(f"q {r['q']}·K {r['K']}·F {r['F']}: {per_k}")
    return "; ".join(out)


# ================================================================ W.9.6 F / W.9.9 P1-4 / P1-5: the budget
BUDGET_STAGES = ("5a", "8")


def budget_h_text(elapsed_h: float, remaining_h: float) -> str:
    """STOP_BUDGET's 〈h〉: "누적 X h + 남은 Y h = Z h" (block budget / estimate and the in-stage ledger alike)."""
    return f"누적 {elapsed_h:.2f} h + 남은 {remaining_h:.2f} h = {elapsed_h + remaining_h:.2f} h"


def budget(elapsed_h: float, options: list, spec, stage: str) -> dict:
    """options = [dict(design, with_c, total_h)] in the order to try. Stage "5a" (before the screen): selected with
    C, selected without C, then the alternative designs without C. Stage "8" (after the screen, W.9.9 P1-5): only
    [selected with C, selected without C] — C dropped, then stop; no alternative designs (anything else is refused).
    The first whose elapsed + total ≤ 24 h is the plan; none → STOP_BUDGET with the cheapest option's estimate."""
    if stage not in BUDGET_STAGES:
        raise ValueError(f"budget stage {stage!r} not in {BUDGET_STAGES}")
    if not options:
        raise ValueError("budget: no options")
    if stage == "8":
        if (len(options) > 2 or options[0].get("with_c") is not True
                or (len(options) == 2 and (options[1].get("with_c") is not False
                                           or options[1].get("design") != options[0].get("design")))):
            raise ValueError("budget stage 8 (W.9.9 P1-5): only [selected with C, selected without C]")
    for o in options:
        if elapsed_h + o["total_h"] <= spec.budget_h:
            return dict(outcome=PASS, reasons=[], plan=o, elapsed_h=elapsed_h, stage=stage)
    best = min(options, key=lambda o: o["total_h"])
    h = budget_h_text(elapsed_h, best["total_h"])
    return dict(outcome=STOP_BUDGET, reasons=[h], plan=None, elapsed_h=elapsed_h, stage=stage,
                sentence=sentence(STOP_BUDGET, dict(h=h)))


# ================================================================ W.2 / W.9.5 / H7: the gate pairs
def gate_pairs(n_set: int, k: int, spec) -> dict:
    if k < spec.min_gate_pairs:
        return dict(outcome=STOP_FEW_PAIRS, reasons=[f"관문 쌍 {k} < {spec.min_gate_pairs}"],
                    sentence=sentence(STOP_FEW_PAIRS, dict(n=n_set, k=k)))
    return dict(outcome=PASS, reasons=[])


# ================================================================ W.7 / W.9.7 / W.9.8: the verdict's sentence
def verdict_sentence(v: dict, design: dict, spec) -> dict:
    """v = w_verdict.judge's output; design = (q, K, F) of the plan."""
    out = v["verdict"]
    if out == V_PASS:
        f = dict(k=v["n_pairs"], m=v["n_judgeable"], q=design["q"], K=design["K"], F=design["F"], k_max=spec.k_cap)
    elif out == V_FAIL:
        f = dict(fails="; ".join(f"{k}: " + "·".join(v["pairs"][k]["reasons"]) + _gates(v["pairs"][k])
                                 for k in v["failing"]), n_mech=len(v["mech_fail"]))
    elif out == V_UNDECIDED:
        f = dict(cause=v["undecided_cause"])
    else:
        f = dict(why="; ".join([f"INVALID 쌍 {k}" + _why(v["pairs"].get(k) or {}) for k in v["invalid"]]
                               + list(v["machine"])))      # W.9.10 6: every INVALID pair first, nothing cut
    return dict(sentence=sentence(out, f), consequence=CONSEQUENCE[out])


def _gates(p: dict) -> str:
    g = p.get("failing_gates") or {}
    hit = [f"{k} {n}마리" for k, n in g.items() if n]
    return f"({', '.join(hit)})" if hit else ""


def _why(p: dict) -> str:
    """W.9.10 6: an INVALID pair's reasons in the STOP_MACHINE sentence."""
    r = p.get("reasons") or []
    return f"({'·'.join(r)})" if r else ""
