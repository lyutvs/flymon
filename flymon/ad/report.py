"""AD.3's sentence frames (AC.8 with AD's arm names) and the assembled AD result. Every sentence ends with AC.5's LABEL;
check() refuses AC.8's forbidden words in every sentence. Notes: C-off rows carry C_OFF_NOTE, every non-primary row
MULTI_NOTE; under STOP_BUDGET the primaries carry "예산 소진 — 최종 판독 시점 <t>"; at or after the first point whose
floor contact exceeds 0.5 every reading carries "회복 r 보정 범위(1 000펄스) 밖 — 바닥 접촉 <x>" and the result adds the
primary contrast at the last point at or below 0.5."""
from __future__ import annotations

from . import analysis, learncheck as lcm, validity
from .spec import ARM_TEXT, FORBIDDEN, LABEL

RESULT = "results/summary/ad_m4.json"
C_OFF_NOTE = "(C-off는 1단계 0 시점 기록; 6마리 팔의 CI는 불안정할 수 있다)"
MULTI_NOTE = "(기술 통계, 다중 비교 미보정)"
REASON = {"fallback": "대체 80", "STOP_FUTILE": "STOP_FUTILE", "STOP_BUDGET": "예산 소진의 마지막 공통 시점"}


def f3(x) -> str:
    return "기록 없음" if x is None else f"{x:.3f}"


def _ci(c) -> str:
    return f"{c['diff']:.3f} [95% CI {c['lo']:.3f}, {c['hi']:.3f}]"


def primary_sentence(c, ref, null_os, n_later_drop, kc_flyl, notes=()) -> str:
    t = "—" if c["point"] is None else c["point"]
    head = f"FLY-L − FLY-OS(우연 전환 대조, {t}배틀 시점)에서 4.3 기준 1: "
    if c["status"] == "측정 불성립":
        body = head + f"유효 마리 FLY-L {c['n_a']}/{c['min_a']} · FLY-OS {c['n_b']}/{c['min_b']} — 탐색 기준 측정 불성립."
    else:
        body = (head + f"{_ci(c)} — 탐색 기준 {c['status']}. FLY-OS 쪽은 우연 전환이다(자기 잡음 기대치 Σq·q {f3(null_os)}). "
                f"확인 세트 20쌍에 한함(교차 부트스트랩 95% CI {c['crossed']['lo']:.3f}, {c['crossed']['hi']:.3f}, 기술).")
    body += " 기준 행 FLY-L(0) − FLY-OS(0): " + (f"{_ci(ref)}." if "diff" in ref else "측정 불성립.")
    body += f" 유효 마리 FLY-L {c['n_a']} · FLY-OS {c['n_b']}({t}배틀 시점, 이후 탈락 {n_later_drop})."
    for n in notes:
        body += f" {n}"
    return f"{body} {LABEL} 같은 턴 후보 KC 비율 > 2 턴 비율(FLY-L): {f3(kc_flyl)}."


def row_sentence(c, notes=()) -> str:
    if c["status"] == "측정 불성립":
        why = f" — {c['reason']}" if c.get("reason") else ""
        body = (f"{c['name']}: 유효 마리 {ARM_TEXT[c['a']]} {c.get('n_a', 0)}/{c.get('min_a', '—')} · "
                f"{ARM_TEXT[c['b']]} {c.get('n_b', 0)}/{c.get('min_b', '—')}{why} — 측정 불성립.")
    else:
        body = f"{c['name']}: {_ci(c)}"
        if "crossed" in c:
            body += f" (교차 부트스트랩 95% CI {c['crossed']['lo']:.3f}, {c['crossed']['hi']:.3f})"
        body += f" — {c['status']}."
    if "COFF" in (c["a"], c["b"]):
        body += f" {C_OFF_NOTE}"
    body += f" {MULTI_NOTE}"
    for n in notes:
        body += f" {n}"
    return f"{body} {LABEL}"


def unmeasured_sentence(n: int) -> str:
    return f"4.3 기준 {n}: — 탐색 기준 미측정(AC.2 2단계 팔 보류, 부록 AD). {LABEL}"


def summary_sentence(p1, p2, final, reason) -> str:
    if final is None:                       # STOP_BUDGET with no common point (every arm stopped or no point reached)
        s = "AD 1차 대조는 측정 불성립이다(최종 판독 시점 없음 — 유효한 마리가 함께 도달한 시점이 없다)."
    elif p1["status"] == "충족":
        s = (f"상대 타입을 실제 상대와 무관하게 섞고 총 세기를 같게 둔 우연 전환 대조(FLY-OS)와 비교해, 배틀로 {final}배틀 "
             f"학습한 마리에서 상성 조건부 선택이 더 많이 관찰됐다(탐색, 확인 세트 20쌍에 한함, 학습 중 같은 냄새를 봤을 수 있음).")
    elif p1["status"] == "미충족":
        s = (f"배틀로 {final}배틀 학습한 마리의 상성 조건부 선택이 상대 타입을 섞은 우연 전환 대조(FLY-OS)와 비교해 탐색 기준에 "
             f"못 미쳤다(확인 세트 20쌍에 한함).")
    else:
        s = "AD 1차 대조는 측정 불성립이다(최소 유효 마리 미달 또는 팔 STOP_INFRA)."
    s += " 확인 세트 20쌍은 1단계에서 이미 쓰였고, 그 상황은 학습 중에도 나왔을 수 있다(AC.4 · AD.0 3)."
    if p2["status"] != p1["status"]:
        s += f" 40배틀 시점 대조는 {p2['status']}로 최종 시점과 다르다 — 둘을 합쳐 읽지 않는다."
    if reason != "complete" and final is not None:
        s += f" 최종 판독 시점은 {final}배틀이다({REASON[reason]})."
    return f"{s} {LABEL}"


def futility_sentence(fut) -> str:
    return (f"STOP_FUTILE: {fut['point']}배틀 시점 FLY-L 맞는 방향 쌍 비율 {f3(fut['share_at_point'])}가 0 시점 "
            f"{f3(fut['share_at_0'])}보다 크지 않아 학습을 {fut['point']}배틀에서 멈췄다(판정 아님). 판독은 "
            f"{fut['point']}배틀 시점이다. {LABEL}")


def learncheck_sentence(arm, lc, points, final) -> str:
    rs = " → ".join(f"{p} {f3(lc['right_share'][arm].get(p, {}).get('share'))}" for p in points)
    sc = " → ".join(f"{p} {f3(lc['switch_curve'][arm][p]['mean'])}" for p in sorted(lc["switch_curve"][arm])
                    if p in (points[0], final))
    nul = (lc["null"].get("FLYOS") or {}).get(final)
    return (f"{ARM_TEXT[arm]} 학습 확인(기록, 판정 아님): 맞는 방향 쌍 비율 {rs}; 전환율 {sc}; FLY-OS 잡음 기대치 Σq·q "
            f"{final} {f3(nul)}. {LABEL}")


def check(sentences) -> None:
    for s in sentences:
        bad = [w for w in FORBIDDEN if w in s]
        if bad:
            raise ValueError(f"forbidden words {bad} in a result sentence (AC.8): {s}")


def dropouts_with_stopped(invalid: dict, stopped: dict, lay, sits) -> list:
    """AD.7's later drop-outs: the learning book's marked flies (validity.dropouts) plus every fly of a stopped arm
    (book `stopped`) that carries no invalid mark - hooks mark such a fly "arm_stopped" only when its skipped battle
    comes back, which a STOP_BUDGET session may never reach (Task 8 carry)."""
    out = validity.dropouts(invalid, lay, sits)
    rp = validity.reached(sits)
    bad = {int(k) for k in invalid}
    for i, (arm, _, g) in enumerate(lay):
        if arm in (stopped or {}) and i not in bad:
            out.append(dict(fly=g, arm=arm, reason=f"arm_stopped(장부 stopped, 무효 표시 없음): {stopped[arm]}",
                            last_point=rp.get(g), marked=False))
    return sorted(out, key=lambda d: d["fly"])


def futility_recheck(sits, pairs, spec, lay, invalid: dict, stopped: dict, fut: dict):
    """Recompute AD.2's futility check from the records, only when the run took the decision (result.futility set)
    and every fly still in the reading (not invalid, arm not stopped) has its point-40 record (Task 8 carry: never call
    futility before that). A different result is STOP_INFRA, as in validity.persist_futility."""
    if not fut or fut.get("status") is None:
        return None
    rp, bad, halted = validity.reached(sits), {int(k) for k in invalid}, set(stopped or ())
    alive = [g for i, (arm, _, g) in enumerate(lay) if i not in bad and arm not in halted]
    if any(rp.get(g, -1) < spec.futility_point for g in alive):
        return dict(checked=False, reason="not every valid fly has its point-40 record")
    now = validity.futility(sits, pairs, spec)
    keys = ("status", "share_at_point", "share_at_0", "n_at_point", "n_at_0")
    if any(now.get(k) != fut.get(k) for k in keys if k in fut):
        raise SystemExit(f"STOP_INFRA: the futility check recomputed from the records differs ({fut} -> {now})")
    return dict(checked=True, same=True)


def analyse(*, spec, T, result, sits, pairs, coff, fly1, battles_L, eval_records, learn_logs, nobrain_rows, move_info,
            kc_by_arm, load_summary) -> dict:
    lay = spec.layout()
    book = result["books"]["L"]
    invalid_L = {int(k): v for k, v in book["invalid"].items()}
    stopped = dict(book.get("stopped") or {})
    fut = result.get("futility") or {}
    fut_check = futility_recheck(sits, pairs, spec, lay, invalid_L, stopped, fut)
    fp = validity.final_point(spec, T, fut.get("status"), result["status"], sits, lay, invalid_L, stopped=stopped)
    final, reason = fp["point"], fp["reason"]          # final is None when no common point exists (STOP_BUDGET)
    points = [p for p in spec.points(T) if final is not None and p <= final]
    retried = sorted({lay[int(r["fly"])][2] for r in battles_L if int(r.get("retries", 0)) > 0})
    new_pts = [p for p in dict.fromkeys((0, spec.futility_point, final))
               if p is not None and any(r.get("set") == "new" and r.get("point") == p for r in sits)]
    rd = analysis.situation_readings(sits, spec, final=final, new_points=new_pts, coff=coff, fly1=fly1,
                                     retried=retried)
    ev = analysis.eval_readings(result, eval_records, nobrain_rows, spec)
    lc = lcm.learncheck(sits, pairs, move_info, learn_logs, lay, points, spec.floor_read)
    fr = lc["floor_reading"]
    floor_row = None
    if fr["first_over"] is not None and fr["last_ok"] is not None:     # AD.3: the primary at the last point <= 0.5
        t = fr["last_ok"]
        floor_row = analysis.contrast("1차 1(바닥 접촉 ≤ 0.5 마지막 시점)", "FLYL", "FLYOS",
                                      analysis.switch_units(sits, "FLYL", t), analysis.switch_units(sits, "FLYOS", t),
                                      spec, point=t, rule=True)
    drops = dropouts_with_stopped(invalid_L, stopped, lay, sits)

    def later(t) -> int:                    # AD.7: flies in the reading at t that dropped out afterwards
        return sum(1 for d in drops if t is not None and d["last_point"] is not None and d["last_point"] >= t)

    def floor_note(t) -> list:
        if fr["first_over"] is not None and t is not None and t >= fr["first_over"]:
            return [f"회복 r 보정 범위(1 000펄스) 밖 — 바닥 접촉 {f3(fr['values'].get(t))}"]
        return []

    def notes(t) -> list:
        out = []
        if reason == "STOP_BUDGET" and t is not None and t == final:
            out.append(f"예산 소진 — 최종 판독 시점 {final}")
        return out + floor_note(t)

    nul = lc["null"]["FLYOS"]
    p1, p2 = rd["primary"]
    sentences = [primary_sentence(c, rd["reference"], nul.get(c["point"]), later(c["point"]), kc_by_arm.get("FLYL"),
                                  notes(c["point"])) for c in (p1, p2)]
    if floor_row is not None:
        t = floor_row["point"]
        sentences.append(primary_sentence(floor_row, rd["reference"], nul.get(t), later(t), kc_by_arm.get("FLYL"),
                                          [f"바닥 접촉 0.5 이하 마지막 시점의 병기 행(바닥 접촉 {f3(fr['values'].get(t))}; "
                                           f"0.5 초과 첫 시점 {fr['first_over']})"]))
    for key in ("did", "aux", "new_set", "paired_os", "sensitivity"):
        sentences += [row_sentence(c, notes(c.get("point"))) for c in rd[key]]
    # Ruling S2: the evaluation battles run after learning ends (the final point), so criteria 2-3 carry the floor
    # note whenever the first point above 0.5 is at or before the final point.
    sentences += [row_sentence(c, floor_note(final)) for c in ev] + [unmeasured_sentence(4), unmeasured_sentence(5)]
    summary = summary_sentence(p1, p2, final, reason)
    extra = [futility_sentence(fut)] if fut.get("status") == "STOP_FUTILE" else []
    extra += [learncheck_sentence(arm, lc, points, final) for arm in ("FLYL", "FLYOS")] if points else []
    check(sentences + [summary] + extra)
    return dict(appendix="AD", label=LABEL, T=T, final_point=fp, futility=fut or None, futility_recheck=fut_check,
                run_status=result["status"], stopped_arms=stopped,
                primary=rd["primary"], reference=rd["reference"], did=rd["did"], aux=rd["aux"], new_set=rd["new_set"],
                paired_os=rd["paired_os"], sensitivity=rd["sensitivity"], floor_row=floor_row, criteria_2_3=ev,
                criteria_4_5=[dict(criterion=n, status="미측정") for n in (4, 5)], learncheck=lc,
                kc_ratio_gt2=kc_by_arm, n_valid_by_point=validity.n_valid_by_point(sits, lay, points),
                dropouts=drops, retried_flies=retried, load=load_summary, sentences=sentences, summary=summary,
                futility_and_learncheck_sentences=extra, boot=dict(draws=spec.boot_draws, seed=spec.boot_seed),
                notes=["다중 비교 보정 없음(AD.0 8): 1차는 최종 시점과 40배틀 시점 두 대조, 나머지는 기술 · 기록이다.",
                       "1차의 추론은 확인 세트 20쌍에 한한다; 교차 부트스트랩 CI는 기술이다(AD.0 3).",
                       "FLY-L − FLY-OS에는 실현된 보상 노출의 차이가 섞인다(AD.0 6).",
                       "기준 4 · 5와 AC.2 2단계 팔은 미측정(보류)이다(AD.0 10).",
                       "AD 결과는 AC.10 문장을 바꾸지 않으며 두 단계의 기준 1을 합치지 않는다(AD.7)."])
