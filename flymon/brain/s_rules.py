"""S's decisions (S.2-S.4, S.7, S.9). The judgement code is the authoritative source: the bands are r_rules.read_band
itself (R.3's order and R.9.4, S.4 "R.3 구간 순서 그대로") read with S's numbers (n_a 43); only G_fail is S's (S.4,
S.9.1: net drop pun_C − pun_L ≥ g_fail_drop on either axis, no pass->fail branch). Every number is a field of the
LastSetSpec passed in. SENTENCES are S.7 as amended by S.9.1 / S.9.7, 〈…〉 replaced by {field}, fixed before any
measurement (plan Reading 8). The operating characteristic is exact and never changes a band (S.6)."""
from __future__ import annotations

import dataclasses
import hashlib
from math import comb

from ..agent import e_rules
from . import r_rules
from .h3_store import canonical
from .n_rules import INVALID as P_INVALID
from .p_rules import LEARNS_CONFIRMATORY
from .r_spec import SPEC as R_SPEC

PASS, INVALID, NOT_READ, SEALED, READ, INVALID_RUN = (r_rules.PASS, r_rules.INVALID, r_rules.NOT_READ, r_rules.SEALED,
                                                       r_rules.READ, r_rules.INVALID_RUN)
STOP_REUSE, STOP_SET_SHORT = "STOP_REUSE", e_rules.STOP_SET_SHORT
STOP_PUNISH_BROKEN, STOP_P_REFERENCE, STOP_PUNISH_WEAKENED = (r_rules.STOP_PUNISH_BROKEN, "STOP_P_REFERENCE",
                                                              "STOP_PUNISH_WEAKENED")
SELECTED, B_TB, B_FA, B_NC, B_PG = r_rules.SELECTED, r_rules.B_TB, r_rules.B_FA, r_rules.B_NC, r_rules.B_PG
BANDS = r_rules.BANDS
C_ABOVE_BAR, BELOW_BAR, MARGIN = r_rules.C_ABOVE_BAR, r_rules.BELOW_BAR, r_rules.MARGIN
read_band = r_rules.read_band                      # the same function: S.4 keeps R.3's order (Reading 7)


# ================================================================ gates (S.2, S.3 ①, S.3 ④, S.9.5)
def reuse(r_doc: dict, r_git: dict, shared_key: str, spec) -> dict:
    """S.3 ① / S.9.5: R's gates are reused iff the shared measurement key equals R's, R's summary is tracked and clean,
    and its reused blocks (repro, gate ①, gate ③) exist with R's key and passed. Else STOP_REUSE (Reading 4: the
    re-measurement path is not built; S stops for the user)."""
    why = []
    if shared_key != spec.r_shared_key:
        why.append(f"공유 측정 키 {shared_key} ≠ R {spec.r_shared_key}")
    if not r_git.get("tracked") or r_git.get("dirty"):
        why.append(f"{spec.r_summary} 미커밋")
    for b in spec.r_reused:
        blk = r_doc.get(b)
        if not isinstance(blk, dict):
            why.append(f"R 블록 {b} 없음")
            continue
        if blk.get("code_key") != spec.r_shared_key:
            why.append(f"R 블록 {b}의 코드 키 {blk.get('code_key')}")
        ok = blk.get("passed") if b == "repro" else blk.get("outcome") == PASS
        if not ok:
            why.append(f"R 블록 {b} 통과 아님")
    if why:
        return dict(outcome=STOP_REUSE, reasons=why, sentence=sentence(STOP_REUSE, dict(why="; ".join(why))))
    return dict(outcome=PASS, reasons=[])


def set_outcome(js: dict) -> dict:
    """S.2: STOP_SET_SHORT when (b) ran out before the last turn (the declared-value check is the runner's refusal)."""
    if js["status"] == STOP_SET_SHORT:
        return dict(outcome=STOP_SET_SHORT, sentence=sentence(STOP_SET_SHORT, dict(m=js["n_b"])))
    return dict(outcome=PASS)


def gate2(res_l: dict, res_c: dict, ratio: dict | None, spec) -> dict:
    """S.3 ④ in its order: an INVALID P judgement (either condition) stays INVALID (one rerun, S.3 ④ / R.5); L not
    LEARNS_CONFIRMATORY -> STOP_PUNISH_BROKEN; C not -> STOP_P_REFERENCE; ℓ_L / ℓ_C < p_ratio_min in either direction
    -> STOP_PUNISH_WEAKENED; else PASS."""
    bad = [f"{n}: {m}" for n, res in (("L", res_l), ("C", res_c)) if res.get("outcome") == P_INVALID
           for m in (res.get("reasons") or ["INVALID"])]
    if bad:
        return dict(outcome=INVALID, reasons=bad, label_L=res_l.get("label"), label_C=res_c.get("label"))
    names = list(spec.p.directions)
    dl, dc = res_l["directions"], res_c["directions"]
    base = dict(reasons=[], label_L=res_l["label"], label_C=res_c["label"])
    if res_l["label"] != LEARNS_CONFIRMATORY:
        return dict(base, outcome=STOP_PUNISH_BROKEN, sentence=sentence(STOP_PUNISH_BROKEN, dict(
            label=res_l["label"], l1=f"{dl[names[0]]['ell']:.3f}", l2=f"{dl[names[-1]]['ell']:.3f}")))
    if res_c["label"] != LEARNS_CONFIRMATORY:
        return dict(base, outcome=STOP_P_REFERENCE, sentence=sentence(STOP_P_REFERENCE, dict(label=res_c["label"])))
    low = [d for d in names if ratio[d]["ratio"] < spec.p_ratio_min]
    fields = dict(l1L=f"{dl[names[0]]['ell']:.3f}", l1C=f"{dc[names[0]]['ell']:.3f}",
                  l2L=f"{dl[names[-1]]['ell']:.3f}", l2C=f"{dc[names[-1]]['ell']:.3f}")
    if low:
        return dict(base, outcome=STOP_PUNISH_WEAKENED, low=low, sentence=sentence(STOP_PUNISH_WEAKENED, fields))
    return dict(base, outcome=PASS)


# ================================================================ the punishment guard (S.4, S.9.1)
def g_fail_s(lp: list, cp: list, spec) -> dict:
    """Per axis over the pairs in both: pun = #(−p ≥ 2); net_drop = pun_C − pun_L; fail ⇔ net_drop ≥ g_fail_drop.
    pass->fail / fail->pass are records (S.4). G_fail_S ⇔ fail on (b) or (a)."""
    lm, cm = {p["key"]: p for p in lp}, {p["key"]: p for p in cp}
    axes = {}
    for ax in ("b", "a"):
        ks = sorted(k for k, p in cm.items() if p["axis"] == ax and k in lm)
        pun_c = sum(bool(cm[k]["punish_pass"]) for k in ks)
        pun_l = sum(bool(lm[k]["punish_pass"]) for k in ks)
        pf = sum(bool(cm[k]["punish_pass"]) and not lm[k]["punish_pass"] for k in ks)
        fp = sum(bool(lm[k]["punish_pass"]) and not cm[k]["punish_pass"] for k in ks)
        axes[ax] = dict(n=len(ks), pun_L=pun_l, pun_C=pun_c, net_drop=pun_c - pun_l, pass_to_fail=pf,
                        fail_to_pass=fp, fail=bool(pun_c - pun_l >= spec.g_fail_drop))
    return dict(g_fail=any(a["fail"] for a in axes.values()), axes=axes)


# ================================================================ the closing sentences (S.7, S.9.1, S.9.7; Reading 8)
_NUMS = " ({n}/21 대 {c}/21, F_a {f_a}/43)"
SENTENCES = {
    STOP_REUSE: "R 관문 재사용 조건(S.9.5)이 깨졌다({why}). S는 R 관문을 다시 재는 경로를 갖지 않으므로 판정 세트를 쓰지 않고 "
                "멈춘다 — 사용자 몫.",
    STOP_SET_SHORT: "L 생성기 턴 104–209에서 E-grid 키 중복을 뺀 (b) 쌍이 21개에 못 미쳤다({m}쌍).",
    STOP_PUNISH_BROKEN: r_rules.SENTENCES[STOP_PUNISH_BROKEN],
    STOP_P_REFERENCE: "지렛대 없는 P가 새 시드 블록에서 처벌 학습 확인을 재현하지 못했다({label}) — 비교 기준이 없다.",
    STOP_PUNISH_WEAKENED: "APL→MBON05 제거 아래 같은 시드의 처벌 학습량이 지렛대 없는 쪽의 절반에 못 미쳤다"
                          "(ℓ_r1 {l1L} 대 {l1C}, ℓ_r2 {l2L} 대 {l2C}).",
    B_TB: "APL→MBON05 제거가 마지막 판정 세트(L 생성기 턴 104–{T})에서 {n}/21로 지렛대 없는 같은 세트 {c}/21보다 오르지 "
          "않았다. → 이 지렛대를 닫는다.",
    f"{B_NC}:{C_ABOVE_BAR}": "지렛대 없는 E-grid가 이 세트에서 이미 기준을 넘어 지렛대 효과로 말할 수 없다." + _NUMS,
    f"{B_NC}:{BELOW_BAR}": "지렛대 없는 쪽보다 올랐지만 M2 기준에 못 미쳤다." + _NUMS,
    f"{B_NC}:{MARGIN}": "기준은 넘었으나 지렛대 없는 쪽 대비 여유가 2 미만이다" + _NUMS,
    B_PG: "M2 (b) 기준과 여유는 넘었지만 지렛대 아래 처벌 통과가 순감소했다(처벌 통과 (b) {pb_L} 대 {pb_C}, (a) {pa_L} 대 "
          "{pa_C}; 순감소 (b) {d_b}, (a) {d_a}).",
    B_FA: "M2 (b) 기준·여유·처벌 가드는 넘었지만 F_a가 기준에 못 미쳤다({n}/21 대 {c}/21, F_a {f_a}/43, naive_a {naive_a})."
          " 다음 병목은 F_a(순진 균형 (a) 쌍)다.",
    SELECTED: "C3에서 APL→MBON05 2간선을 지운 모델 변형과 E-grid k2-norm(s 1.0)에서, 마지막 판정 세트(L 생성기 턴 104–{T}, "
              "E-grid 키 중복 제외) (b) 21쌍의 H.4 오라클 시험 가능성이 M2 기준을 만족했고 지렛대 없는 같은 세트보다 높았다"
              "({n}/21 대 {c}/21, 여유 ≥ 2, F_a {f_a}/43, 처벌 순감소 가드(축마다 ≥ 3) 통과, 같은 시드 P 비 ℓ_r1 {rho1}·ℓ_r2 "
              "{rho2} ≥ 0.5(약화 정도 기록), 짝수 {k_even}/21, 판정 시드 24_300_xxx). 지렛대는 Q 결과를 보고 골랐고, 이 판정은 "
              "R 판정 결과를 본 뒤 가드를 바꾼 두 번째 판정이다(S.0). 작동 특성은 S 세트 조건부 값이며 Q → R → S 전체 절차의 "
              "오선택률이 아니다. 실제 커넥톰 간선을 지운 모델이며 오라클 시험 가능성이지 학습 시험이 아니다.",
}


def sentence(outcome: str, fields: dict) -> str:
    """The closing sentence with fields filled; a missing field or an unknown outcome raises KeyError. B_결론없음
    takes its line from fields["reason"]."""
    key = f"{B_NC}:{fields['reason']}" if outcome == B_NC else outcome
    return SENTENCES[key].format(**fields)


# ================================================================ the operating characteristic (S.6, S.9.1, S.9.7)
def g_axis(N: int, a_pf: float, a_fp: float, spec) -> float:
    """P(k_pf − k_fp ≥ g_fail_drop) on an axis of N independent pairs (multinomial: C pass ∧ L fail with a_pf, C fail ∧
    L pass with a_fp)."""
    rest = 1 - a_pf - a_fp
    tot = 0.0
    for k in range(N + 1):
        for j in range(N - k + 1):
            if k - j >= spec.g_fail_drop:
                tot += comb(N, k) * comb(N - k, j) * a_pf ** k * a_fp ** j * rest ** (N - k - j)
    return tot


def g_prob(a_pf: float, a_fp: float, spec) -> float:
    """G_fail_S on (b) n_b or (a) n_a pairs, the axes independent (OR)."""
    return 1 - (1 - g_axis(spec.n_b, a_pf, a_fp, spec)) * (1 - g_axis(spec.n_a, a_pf, a_fp, spec))


OC_NOTES = ("S.9.7: 지렛대 아래 보상 (b)가 포화하면(R 21/21) testable_b = pun_L(b)이므로 (b) 가드는 pun_C(b) − n ≥ 3으로 n과 "
            "결합한다. 'G_fail은 n과 독립' 가정이 (b) 축에서 깨지고, 실질 처벌 가드는 (a) 축이 맡는다.",
            "작동 특성은 S 세트 조건부 값이며 Q → R → S 전체 절차의 오선택률이 아니다.")


def oc(spec) -> dict:
    """Exact conditional OC with R's model (r_rules._cell: n ~ Bin(n_b, q), F_a ~ Bin(naive_a, q), G_fail ~
    Bernoulli(g), c fixed, each cell read through read_band) on S's n_b 21 · n_a 43, g from G_fail_S; beside each g row
    R's guard on the same pair counts (r_guard: drop ≥ 2 ∨ pass->fail ≥ 3, S.6)."""
    rg = dataclasses.replace(R_SPEC, n_b=spec.n_b, n_a=spec.n_a)

    def grow(kind, x, y):
        return dict(kind=kind, a_pf=x, a_fp=y, p=g_prob(x, y, spec), r_guard=r_rules.g_prob(x, y, rg))

    g_rows = ([grow("null", a, a) for a in spec.oc_discord] + [grow("harm", x, y) for x, y in spec.oc_harm])
    rows = []
    for q in spec.oc_q:
        for c in spec.oc_c:
            for na in range(min(spec.n_a, spec.oc_naive_max) + 1):
                for gr in [dict(kind="none", a_pf=0.0, a_fp=0.0, p=0.0)] + g_rows:
                    rows.append(dict(q=q, c=c, naive_a=na, g_kind=gr["kind"], a_pf=gr["a_pf"], a_fp=gr["a_fp"],
                                     g=gr["p"], P=r_rules._cell(q, c, na, gr["p"], spec)))
    table = dict(assumption="pairs independent; q_b = q_a; c fixed; G_fail_S independent of n and F_a with probability "
                            "g from a per-pair discordance model per axis (net drop ≥ g_fail_drop), OR over (b) and (a)",
                 notes=list(OC_NOTES), g_fail=g_rows, rows=rows)
    return dict(table, sha256=hashlib.sha256(canonical(table).encode()).hexdigest())
