"""U's decisions (U.3, U.4, U.7 as amended by U.9). The judgement code is the authoritative source: the bands are
r_rules.read_band itself (U.4 = T.4 = S.4 = R.3's order, n_a 43), G_fail_S is S's (net drop ≥ 3 per axis) and gate ②'s
order is S's (read on block h4's z, T.9.3). New here: the reuse of R's repro and T's unedited z (U.3 1), the endpoint
reproduction (U.9.1), the guard scan and its candidate cap (U.3 3 (a)(b)), the KC band per candidate (U.3 3 (c)), the
even reproduction and validity per qualified f (U.3 4), the f choice (U.3 5 with U.9.2's filter), the mechanism
contrast readings (U.9.3, records), the sentences (U.7, U.9.2, U.9.4; 〈…〉 → {field}) and the OC notes (U.6, U.9.4
P3-10). Every number is a field of the USpec passed in; no OC value and no contrast reading ever changes a band."""
from __future__ import annotations

import hashlib

from . import r_rules, s_rules, t_rules
from .h3_store import canonical

PASS, INVALID, NOT_READ, SEALED, READ, INVALID_RUN = (t_rules.PASS, t_rules.INVALID, t_rules.NOT_READ,
                                                       t_rules.SEALED, t_rules.READ, t_rules.INVALID_RUN)
STOP_REUSE, STOP_SET_SHORT = t_rules.STOP_REUSE, t_rules.STOP_SET_SHORT
STOP_U_PATH_REPRO, STOP_NO_QUALIFIED_F, STOP_EVEN_PUNISH = "STOP_U_PATH_REPRO", "STOP_NO_QUALIFIED_F", "STOP_EVEN_PUNISH"
STOP_EVEN_REPRO, STOP_EVEN_LOW_LEVER = t_rules.STOP_EVEN_REPRO, t_rules.STOP_EVEN_LOW_LEVER
STOP_PUNISH_BROKEN, STOP_P_REFERENCE, STOP_PUNISH_WEAKENED = (t_rules.STOP_PUNISH_BROKEN, t_rules.STOP_P_REFERENCE,
                                                              t_rules.STOP_PUNISH_WEAKENED)
SELECTED, B_TB, B_FA, B_NC, B_PG = t_rules.SELECTED, t_rules.B_TB, t_rules.B_FA, t_rules.B_NC, t_rules.B_PG
BANDS = t_rules.BANDS
C_ABOVE_BAR, BELOW_BAR, MARGIN = t_rules.C_ABOVE_BAR, t_rules.BELOW_BAR, t_rules.MARGIN
NQ_GUARD, NQ_KC = "GUARD", "KC"                      # the two STOP_NO_QUALIFIED_F sentences (U.9.4 P1-4)
CHAIN_FOR, CHAIN_AGAINST, NO_CONCLUSION, KC_SIDE = "사슬 지지", "사슬 비지지", "결론 없음", "KC 경로 쪽"
read_band = r_rules.read_band                        # U.4 = T.4: R.3's order
g_fail_s = s_rules.g_fail_s                          # U.4: S's guard, each condition on its own z
gate2 = s_rules.gate2                                # U.3 8 = T.9.3: S's order, read on block h4's z
set_outcome = t_rules.set_outcome                    # U.2: T's set, STOP_SET_SHORT when the turns ran out


def _ff(f) -> str:
    return f"{float(f):.1f}"


def _fl(fs) -> str:
    return "{" + ", ".join(_ff(f) for f in fs) + "}"


# ================================================================ gates before the set is used
def reuse(r_doc: dict, r_git: dict, t_doc: dict, t_git: dict, shared_key: str, t_key: str, spec) -> dict:
    """U.3 1: R's repro is reused iff the shared measurement key equals R's and R's summary holds the passed repro block
    (t_rules.reuse with r_reused = ("repro",)); T's unedited z iff the current T measurement key equals T's z block's
    and the declared one, T's summary is tracked and clean, its z block carries R's shared key and its unedited side
    reproduced block h4's z with both readout types passing the guard on the unedited CSC. Else STOP_REUSE."""
    r = t_rules.reuse(r_doc, r_git, shared_key, spec)
    why = list(r.get("reasons") or [])
    z = t_doc.get("z")
    if t_key != spec.t_measure_key_t:
        why.append(f"T 측정 키 {t_key} ≠ 선언 {spec.t_measure_key_t}")
    if not t_git.get("tracked") or t_git.get("dirty"):
        why.append(f"{spec.t_summary} 미커밋")
    if not isinstance(z, dict):
        why.append("T 블록 z 없음")
    else:
        if z.get("t_measure_key") != t_key:
            why.append(f"T 블록 z의 T 측정 키 {z.get('t_measure_key')}")
        if z.get("code_key") != spec.r_shared_key:
            why.append(f"T 블록 z의 코드 키 {z.get('code_key')}")
        if z.get("detail_sha256") != spec.t_z_detail_sha256:
            why.append(f"T 블록 z의 원자료 sha {z.get('detail_sha256')}")
        none = z.get("none") or {}
        zn = {k: (float(v[0]), float(v[1])) for k, v in (none.get("z") or {}).items()}
        guard = none.get("guard") or {}
        if (zn != spec.z_h4_dict() or none.get("csc_sha256") != [spec.sha_none] or none.get("edit_edges") != [0]
                or set(guard) != {spec.a_type, spec.p_type} or not all(g.get("passes") for g in guard.values())):
            why.append("T 블록 z의 편집 없는 엔진 재현 통과 아님")
    if why:
        return dict(outcome=STOP_REUSE, reasons=why, sentence=sentence(STOP_REUSE, dict(why="; ".join(why))))
    return dict(outcome=PASS, reasons=[])


def path(checks: list, spec) -> dict:
    """U.9.1: checks = [dict(f, ref, diffs, invalid)] in measurement order. A defect (presentation counts, edge
    counts) is INVALID; any bit difference is STOP_U_PATH_REPRO, its sentence naming the first failing endpoint and
    every failing check listed."""
    bad = [m for c in checks for m in c.get("invalid", [])]
    if bad:
        return dict(outcome=INVALID, reasons=bad, failed=[])
    failed = [c for c in checks if c["diffs"]]
    if failed:
        c = failed[0]
        return dict(outcome=STOP_U_PATH_REPRO, reasons=[], failed=[dict(f=x["f"], ref=x["ref"], diffs=x["diffs"])
                                                                   for x in failed],
                    sentence=sentence(STOP_U_PATH_REPRO, dict(f=_ff(c["f"]), ref=c["ref"], diff="; ".join(c["diffs"]))))
    return dict(outcome=PASS, reasons=[], failed=[])


def scan(points: dict, spec) -> dict:
    """U.3 3 (a)(b): points = {f: t_rules.z_lever's decision on L_f's side} for the 9 grid points. An INVALID point is
    INVALID; the passing f are the candidates (no monotonicity assumed); the smallest n_candidates_max of them, in
    ascending order, are checked next (no substitutes). No candidate → STOP_NO_QUALIFIED_F (guard sentence)."""
    if set(points) != set(spec.f_grid):
        return dict(outcome=INVALID, reasons=[f"scan points {sorted(points)} are not {list(spec.f_grid)}"])
    bad = [f"f {_ff(f)}: {m}" for f in spec.f_grid if points[f]["outcome"] == INVALID for m in points[f]["reasons"]]
    if bad:
        return dict(outcome=INVALID, reasons=bad)
    cands = sorted(f for f in spec.f_grid if points[f]["outcome"] == PASS)
    checked = cands[:spec.n_candidates_max]
    base = dict(reasons=[], candidates=cands, checked=checked, unchecked=cands[spec.n_candidates_max:])
    if not cands:
        detail = "; ".join(f"f {_ff(f)}: " + ", ".join(
            f"{t} Δ {g['median_delta']:.1f}·0 비율 {g['zero_share']:.3f}" for t, g in sorted(points[f]["guard"].items()))
            for f in spec.f_grid)
        return dict(base, outcome=STOP_NO_QUALIFIED_F, kind=NQ_GUARD,
                    sentence=sentence(f"{STOP_NO_QUALIFIED_F}:{NQ_GUARD}", dict(detail=detail)))
    return dict(base, outcome=PASS)


def kc_point(rec112: dict, cap_ok: bool, rec55: dict, spec_f) -> dict:
    """U.3 3 (c) for one checked f: R's gate ① rule on the 112 calibration odours (r_rules.gate1: their median of
    per-odour medians in valid_band, the ORN cap) and T.9.1's on the T set's 55 odours (t_rules.gate1s: every odour's
    median in valid_band); INVALID when either is; PASS iff both pass."""
    g1 = r_rules.gate1(rec112, cap_ok, spec_f)
    g55 = t_rules.gate1s(rec55, spec_f)
    out = dict(calib=g1, set=g55)
    if INVALID in (g1["outcome"], g55["outcome"]):
        return dict(out, outcome=INVALID, reasons=g1.get("reasons", []) + g55.get("reasons", []))
    ok = g1["outcome"] == PASS and g55["outcome"] == PASS
    return dict(out, outcome=PASS if ok else "FAIL", reasons=[],
                failed=list(g1.get("failed", [])) + [f"T 세트 {o}" for o in g55.get("failed", [])])


def kc(points: dict, scan_block: dict, spec) -> dict:
    """The qualified f = the checked f whose KC band passed; none → STOP_NO_QUALIFIED_F (KC sentence)."""
    checked = list(scan_block["checked"])
    if sorted(points) != sorted(checked):
        return dict(outcome=INVALID, reasons=[f"KC points {sorted(points)} are not the checked f {checked}"])
    bad = [f"f {_ff(f)}: {m}" for f in checked if points[f]["outcome"] == INVALID for m in points[f]["reasons"]]
    if bad:
        return dict(outcome=INVALID, reasons=bad)
    q = [f for f in checked if points[f]["outcome"] == PASS]
    base = dict(reasons=[], qualified=q, checked=checked)
    if not q:
        return dict(base, outcome=STOP_NO_QUALIFIED_F, kind=NQ_KC, sentence=sentence(
            f"{STOP_NO_QUALIFIED_F}:{NQ_KC}", dict(cands=_fl(scan_block["candidates"]), checked=_fl(checked),
                                                   unchecked=_fl(scan_block["unchecked"]))))
    return dict(base, outcome=PASS)


def even_repro(C: dict, spec, l_h4) -> dict | None:
    """U.3 4: C (R's even raw, h4 z) must reproduce c_even_expected first, else STOP_EVEN_REPRO (T's sentence) and no
    L_f is measured; None when it reproduced."""
    c = (C.get("aggregate") or {}).get("testable_b")
    if C["reasons"] or c != spec.c_even_expected:
        return dict(outcome=STOP_EVEN_REPRO, reasons=[f"C: {m}" for m in C["reasons"]], c_even=c,
                    sentence=sentence(STOP_EVEN_REPRO, dict(l=l_h4, c=c)))
    return None


def even_validity(L: dict, C: dict, spec_f, repro_sha: str) -> list:
    """r_rules.gate3's validity list for one L_f: reasons, exactly lever_edges edges, C none on the repro's CSC, L ≠ C."""
    reasons = [f"L: {m}" for m in L["reasons"]] + [f"C: {m}" for m in C["reasons"]]
    if L["edit_edges"] != [spec_f.lever_edges]:
        reasons.append(f"L changed {L['edit_edges']} CSC edges, declared {spec_f.lever_edges}")
    if C["edit_edges"] != [0]:
        reasons.append(f"C changed {C['edit_edges']} CSC edges, declared none")
    if C["csc_sha256"] != repro_sha:
        reasons.append(f"C ran on CSC {C['csc_sha256']}, not the reproduction gate's {repro_sha}")
    if L["csc_sha256"] == C["csc_sha256"]:
        reasons.append("L and C ran on the same CSC weights")
    if L.get("aggregate") is None or C.get("aggregate") is None:
        reasons.append("no (b) aggregate")
    return reasons


def choose(records: dict, c_even: int, spec) -> dict:
    """U.3 5 with U.9.2, among the qualified f (records = {f: u_records.even_record}):
    ① keep f with (b) net drop < even_drop_b_lt and (a) net drop ≤ even_drop_a_le (pun = #(−p ≥ 2), C on h4 z, L_f on
    z_f — S's G_fail_S formula); none → STOP_EVEN_PUNISH;
    ② f* = the kept f with the largest even testable_b, ties → the larger f;
    ③ f*'s testable_b < bar_b → STOP_EVEN_LOW_LEVER (R.9.3's sentence, f* and the checked f named); else PASS."""
    fs = sorted(records)
    keep = [f for f in fs if records[f]["drops"]["b"]["net_drop"] < spec.even_drop_b_lt
            and records[f]["drops"]["a"]["net_drop"] <= spec.even_drop_a_le]
    drops = "; ".join(f"f {_ff(f)}: (b) {records[f]['drops']['b']['net_drop']}·(a) {records[f]['drops']['a']['net_drop']}"
                      for f in fs)
    base = dict(reasons=[], checked=fs, kept=keep)
    if not keep:
        return dict(base, outcome=STOP_EVEN_PUNISH, sentence=sentence(STOP_EVEN_PUNISH, dict(detail=drops)))
    f_star = max(keep, key=lambda f: (records[f]["testable_b"], f))
    r = records[f_star]
    base = dict(base, f_star=f_star, testable_b=r["testable_b"])
    if r["testable_b"] < spec.bar_b:
        return dict(base, outcome=STOP_EVEN_LOW_LEVER, sentence=sentence(STOP_EVEN_LOW_LEVER, dict(
            tb=r["testable_b"], c_even=c_even, k=r["F_a"], f=_ff(f_star), checked=_fl(fs))))
    return dict(base, outcome=PASS)


# ================================================================ the mechanism contrast readings (U.9.3, records)
def contrast_block(d_block: float, d_none: float, spec) -> str:
    """D_block ≤ 0.5 × D_none → 사슬 지지; D_block ≥ 0.8 × D_none → 사슬 비지지; else 결론 없음 (fixed before results).
    D_none ≤ 0 → 결론 없음: U.9.3 leaves the reading undefined there (record only)."""
    if d_none <= 0:
        return NO_CONCLUSION
    if d_block <= spec.chain_support_max * d_none:
        return CHAIN_FOR
    if d_block >= spec.chain_against_min * d_none:
        return CHAIN_AGAINST
    return NO_CONCLUSION


def contrast_entry(cut: dict, delta_f0: float, spec) -> str:
    """MBON13 under the chain-entry cut passes the scan's guard → 사슬 지지; |Δ(cut) − Δ(f = 0 alone)| < 1 → KC 경로 쪽;
    else 결론 없음. No SD > 0 check here: U.9.3 lists only the Δ median and the zero share for this reading."""
    if cut["median_delta"] >= spec.z_guard_med_min and cut["zero_share"] <= spec.z_guard_zero_max:
        return CHAIN_FOR
    if abs(cut["median_delta"] - delta_f0) < spec.kc_side_delta_max:
        return KC_SIDE
    return NO_CONCLUSION


# ================================================================ the closing sentences (U.7, U.9.2, U.9.4)
_NUMS = " ({n}/21 대 {c}/21, F_a {f_a}/43, f = {f})"
SENTENCES = {
    STOP_REUSE: "R·T 재사용 조건(U.3 1)이 깨졌다({why}). U는 R·T 관문을 다시 재는 경로를 갖지 않으므로 판정 세트를 쓰지 않고 "
                "멈춘다 — 사용자 몫.",
    STOP_U_PATH_REPRO: "U 부분 편집 경로가 끝점 {f}에서 {ref}을 재현하지 못했다({diff}).",
    STOP_SET_SHORT: t_rules.SENTENCES[STOP_SET_SHORT],
    f"{STOP_NO_QUALIFIED_F}:{NQ_GUARD}": "f ∈ {{0.1, …, 0.9}} 9점 모두 판독 반응성 가드에서 떨어졌다({detail}).",
    f"{STOP_NO_QUALIFIED_F}:{NQ_KC}": "가드 스캔 통과 후보 {cands} 가운데 작은 f부터 검사한 {checked}이 모두 KC 대역에서 "
                                     "떨어졌다(검사하지 않은 후보 {unchecked}).",
    STOP_EVEN_REPRO: t_rules.SENTENCES[STOP_EVEN_REPRO],
    STOP_EVEN_PUNISH: "자격 f 모두 짝수 쌍에서 처벌 순감소가 거름 기준((b) < 3, (a) ≤ 1)을 넘었다({detail}).",
    STOP_EVEN_LOW_LEVER: r_rules.SENTENCES[STOP_EVEN_LOW_LEVER] + " (f* = {f}, 검사한 f {checked})",
    STOP_PUNISH_BROKEN: s_rules.SENTENCES[STOP_PUNISH_BROKEN],
    STOP_P_REFERENCE: s_rules.SENTENCES[STOP_P_REFERENCE],
    STOP_PUNISH_WEAKENED: s_rules.SENTENCES[STOP_PUNISH_WEAKENED],
    B_TB: "APL→MBON05 부분 제거(f = {f})가 넓힌 상대 풀 판정 세트(생성원 턴 0–{T})에서 {n}/21로 지렛대 없는 같은 세트 "
          "{c}/21보다 오르지 않았다(엔진마다 자기 기준 집합 z). → 넓힌 풀·엔진별 z에서의 이 부분 제거 주장을 닫는다.",
    f"{B_NC}:{C_ABOVE_BAR}": "지렛대 없는 E-grid가 이 세트에서 이미 기준을 넘어 지렛대 효과로 말할 수 없다." + _NUMS,
    f"{B_NC}:{BELOW_BAR}": "지렛대 없는 쪽보다 올랐지만 M2 기준에 못 미쳤다." + _NUMS,
    f"{B_NC}:{MARGIN}": "기준은 넘었으나 지렛대 없는 쪽 대비 여유가 2 미만이다" + _NUMS,
    B_PG: s_rules.SENTENCES[B_PG] + " (f = {f})",
    B_FA: s_rules.SENTENCES[B_FA] + " (f = {f})",
    SELECTED: "C3에서 APL→MBON05 2간선의 가중치를 f = {f}배로 줄인 모델 변형과 E-grid k2-norm(s 1.0)에서, 엔진 변형마다 자기 "
              "기준 집합 z로 읽었을 때, 판정 세트(상대를 1세대 기본 폼으로 넓힌 풀의 생성원 턴 0–{T}, 기존 키·사구체 중복 제외, "
              "모든 행이 POOL 밖 상대 타입 조합 포함) (b) 21쌍의 H.4 오라클 시험 가능성이 M2 기준을 만족했고 지렛대 없는 같은 "
              "세트보다 높았다({n}/21 대 {c}/21, 여유 ≥ 2, F_a {f_a}/43, 처벌 순감소 가드(축마다 ≥ 3) 통과, 같은 시드 P 비 "
              "ℓ_r1 {rho1}·ℓ_r2 {rho2} ≥ 0.5(약화 정도 기록), 짝수 {k_even}/21, 판정 시드 24_500_xxx). f는 {{0.1, …, 0.9}} "
              "가드 스캔을 통과한 후보 가운데 작은 f부터 최대 3개를 KC 대역·짝수 처벌 사전 거름에 넣은 뒤 짝수 testable_b로 "
              "골랐다. 판정 세트의 상대는 1세대 기본 폼으로 넓힌 풀이며, 새 키는 POOL에 없는 상대 타입 조합(12개 중)에서 나온다 "
              "— 과제 풀 안의 시험 가능성은 R·S가 마지막이다. (b) 21쌍은 9개 상대 타입 조합 쌍에 몰려 있어 쌍끼리 독립이 아니다. "
              "지렛대 계열은 Q 결과를 보고 골랐고, 이것은 R(뒤에 가드 변경)·S(뒤에 z 규칙)·T(z 단계 STOP 뒤 부분 제거)에 이은 네 "
              "번째 시도다(U.0). 작동 특성은 U 세트 조건부 값이며 Q → R → S → T → U 전체 절차의 오선택률이 아니다. 실제 커넥톰 "
              "간선을 줄인 모델이며 오라클 시험 가능성이지 학습 시험이 아니다. 귀결은 넓힌 풀에서 엔진별 z로 M2 시험 가능성이 "
              "섰다는 것까지이며, POOL 배틀 과제의 F v4 학습 시험을 이것만으로 정당화하지 않는다.",
}


def sentence(outcome: str, fields: dict) -> str:
    """The closing sentence with fields filled; a missing field or an unknown outcome raises KeyError. B_결론없음
    takes its line from fields["reason"]; STOP_NO_QUALIFIED_F is keyed with its kind."""
    key = f"{B_NC}:{fields['reason']}" if outcome == B_NC else outcome
    return SENTENCES[key].format(**fields)


# ================================================================ the operating characteristics (U.6, U.9.4 P3-10)
OC_NOTES = (s_rules.OC_NOTES[0],
            "U.6: 독립 가정은 U 세트(T 세트 그대로)에서 낙관적이다 — (b) 21쌍은 9개 상대 타입 조합 쌍, (a) 43쌍은 8개 상대 타입 "
            "조합에 몰려 있다. 군집 상관 행(T.9.6·T.9.7 fixture, ICC 0.3)을 옆에 기록한다.",
            "U.9.4 P3-10: f 선택(자격 f 최대 3개 가운데 짝수 testable_b 최대)으로 짝수 관문 ③은 낙관적이다.",
            "작동 특성은 U 세트 조건부 값이며 Q → R → S → T → U 전체 절차의 오선택률이 아니다.")


def oc(spec) -> dict:
    """U.6: S.6's exact independent model (s_rules.oc) on n_b 21 · n_a 43 with U's notes."""
    table = {k: v for k, v in s_rules.oc(spec).items() if k != "sha256"}
    table["notes"] = list(OC_NOTES)
    return dict(table, sha256=hashlib.sha256(canonical(table).encode()).hexdigest())


oc_cluster = t_rules.oc_cluster                      # U.6: T's cluster model unchanged — its fixture, same set and n
