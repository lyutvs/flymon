# flymon/brain/r_rules.py
"""R's decisions (R.2, R.3, R.7, R.9.2-R.9.5). The judgement code is the authoritative source (R.3): read_band is R.3's
ordered list with R.9.4 / R.9.5; every number is a field of the LeverSpec passed in. SENTENCES are R.7 as amended by
R.9.3, verbatim, 〈…〉 replaced by {field}, fixed before any measurement (plan Readings 6, 8, 9). The operating
characteristic is exact (Reading 13) and never changes a band."""
from __future__ import annotations

import hashlib
from math import comb

from ..agent import e_rules
from .h3_store import canonical
from .n_rules import INVALID as P_INVALID
from .p_rules import LEARNS_CONFIRMATORY

PASS, INVALID, NOT_READ, SEALED, READ, INVALID_RUN = "PASS", "INVALID", "NOT_READ", "SEALED", "READ", "INVALID_RUN"
STOP_STRENGTH_LEVER, STOP_PUNISH_BROKEN = "STOP_STRENGTH_LEVER", "STOP_PUNISH_BROKEN"
STOP_EVEN_LOW_LEVER, STOP_C_EVEN_MISMATCH = "STOP_EVEN_LOW_LEVER", "STOP_C_EVEN_MISMATCH"
SELECTED, B_TB, B_FA, B_NC = e_rules.SELECTED, e_rules.B_TB, e_rules.B_FA, e_rules.B_NC
B_PG = "B_처벌가드"
BANDS = (SELECTED, B_TB, B_FA, B_NC, B_PG)
C_ABOVE_BAR, BELOW_BAR, MARGIN = "C_ABOVE_BAR", "BELOW_BAR", "MARGIN"


# ================================================================ gates (R.2, R.9.2, R.9.3, R.9.6)
def gate1(rec: dict, cap_ok: bool, spec) -> dict:
    """R.9.2: INVALID when the lever did not change exactly lever_edges edges on one CSC or the odours / seeds are not
    the declared ones; else STOP_STRENGTH_LEVER when the 112-odour median is outside valid_band or the ORN cap fails."""
    bad = []
    if rec["edit_edges"] != [spec.lever_edges] or len(rec["csc_sha256"]) != 1:
        bad.append(f"lever edges {rec['edit_edges']} on CSC {rec['csc_sha256']}, declared {spec.lever_edges} on one")
    if rec["n_odours"] != spec.n_calib or rec["n_seeds"] != [len(spec.kc_seeds())]:
        bad.append(f"{rec['n_odours']} odours × {rec['n_seeds']} seeds, declared {spec.n_calib} × "
                   f"{len(spec.kc_seeds())}")
    if bad:
        return dict(outcome=INVALID, reasons=bad, failed=[])
    lo, hi = spec.valid_band
    failed = []
    if not lo <= rec["median"] <= hi:
        failed.append(f"냄새별 KC 활성 중앙값 {rec['median']:.4f} ∉ [{lo}, {hi}]")
    if not cap_ok:
        failed.append(f"ORN 상한(s {spec.strength})")
    if failed:
        return dict(outcome=STOP_STRENGTH_LEVER, reasons=[], failed=failed,
                    sentence=sentence(STOP_STRENGTH_LEVER, dict(cond="; ".join(failed))))
    return dict(outcome=PASS, reasons=[], failed=[])


def gate2(res: dict, spec) -> dict:
    """R.2: p_rules.p_judge's result on R's P spec. INVALID stays INVALID (one rerun, R.5); LEARNS_CONFIRMATORY passes;
    any other label is STOP_PUNISH_BROKEN (no looser rule)."""
    if res.get("outcome") == P_INVALID:
        return dict(outcome=INVALID, reasons=list(res.get("reasons", [])), label=P_INVALID)
    if res["label"] == LEARNS_CONFIRMATORY:
        return dict(outcome=PASS, reasons=[], label=res["label"])
    d, names = res["directions"], list(spec.p.directions)
    return dict(outcome=STOP_PUNISH_BROKEN, reasons=[], label=res["label"],
                sentence=sentence(STOP_PUNISH_BROKEN, dict(label=res["label"], l1=f"{d[names[0]]['ell']:.3f}",
                                                           l2=f"{d[names[-1]]['ell']:.3f}")))


def gate3(L: dict, C: dict, spec, repro_sha: str) -> dict:
    """R.9.3 / R.9.6 on the even block's L and C summaries (Reading 6): validity first (reasons, L exactly
    lever_edges edges, C none, C on the unedited engine of the reproduction gate, L ≠ C); then C's testable_b must be
    c_even_expected (else STOP_C_EVEN_MISMATCH); then L's testable_b ≥ bar_b (else STOP_EVEN_LOW_LEVER)."""
    reasons = [f"L: {m}" for m in L["reasons"]] + [f"C: {m}" for m in C["reasons"]]
    if L["edit_edges"] != [spec.lever_edges]:
        reasons.append(f"L changed {L['edit_edges']} CSC edges, declared {spec.lever_edges}")
    if C["edit_edges"] != [0]:
        reasons.append(f"C changed {C['edit_edges']} CSC edges, declared none")
    if C["csc_sha256"] != repro_sha:
        reasons.append(f"C ran on CSC {C['csc_sha256']}, not the reproduction gate's {repro_sha}")
    if L["csc_sha256"] == C["csc_sha256"]:
        reasons.append("L and C ran on the same CSC weights")
    if L.get("aggregate") is None or C.get("aggregate") is None:
        reasons.append("no (b) aggregate")
    if reasons:
        return dict(outcome=INVALID, reasons=reasons)
    aL, aC = L["aggregate"], C["aggregate"]
    base = dict(reasons=[], testable_b=aL["testable_b"], c_even=aC["testable_b"], F_a=aL["F_a"],
                naive_a=aL["naive_a"], C_F_a=aC["F_a"], C_naive_a=aC["naive_a"])
    fields = dict(tb=aL["testable_b"], c_even=aC["testable_b"], k=aL["F_a"])
    if aC["testable_b"] != spec.c_even_expected:
        return dict(base, outcome=STOP_C_EVEN_MISMATCH, sentence=sentence(STOP_C_EVEN_MISMATCH, fields))
    if aL["testable_b"] < spec.bar_b:
        return dict(base, outcome=STOP_EVEN_LOW_LEVER, sentence=sentence(STOP_EVEN_LOW_LEVER, fields))
    return dict(base, outcome=PASS)


# ================================================================ the punishment guard (R.4, R.9.5, Reading 7)
def g_fail(lp: list, cp: list, spec) -> dict:
    """Per axis over the pairs in both: pun = #(−p ≥ 2); drop ⇔ pun_C − pun_L ≥ g_fail_drop; flips ⇔ #(C pass ∧ L
    fail) ≥ g_fail_flips. G_fail ⇔ drop or flips on (b) or (a)."""
    lm, cm = {p["key"]: p for p in lp}, {p["key"]: p for p in cp}
    axes = {}
    for ax in ("b", "a"):
        ks = sorted(k for k, p in cm.items() if p["axis"] == ax and k in lm)
        pun_c = sum(bool(cm[k]["punish_pass"]) for k in ks)
        pun_l = sum(bool(lm[k]["punish_pass"]) for k in ks)
        pf = sum(bool(cm[k]["punish_pass"]) and not lm[k]["punish_pass"] for k in ks)
        fp = sum(bool(lm[k]["punish_pass"]) and not cm[k]["punish_pass"] for k in ks)
        drop, flips = pun_c - pun_l >= spec.g_fail_drop, pf >= spec.g_fail_flips
        axes[ax] = dict(n=len(ks), pun_L=pun_l, pun_C=pun_c, pass_to_fail=pf, fail_to_pass=fp, drop=bool(drop),
                        flips=bool(flips), fail=bool(drop or flips))
    return dict(g_fail=any(a["fail"] for a in axes.values()), axes=axes)


# ================================================================ the bands (R.3, R.9.4, R.9.5)
def read_band(n, c, f_a, g, n_b, n_a, naive_a, spec) -> dict:
    """The first matching line of R.3, in its fixed order."""
    if n_b != spec.n_b or n_a != spec.n_a:
        return dict(band=NOT_READ, reason="COUNTS")
    if c >= spec.bar_b:
        return dict(band=B_NC, reason=C_ABOVE_BAR)
    if n <= c:
        return dict(band=B_TB, reason="NO_GAIN")
    if n < spec.bar_b:
        return dict(band=B_NC, reason=BELOW_BAR)
    if n - c < spec.margin:
        return dict(band=B_NC, reason=MARGIN)
    if g:
        return dict(band=B_PG, reason="PUNISH_GUARD")
    if f_a < spec.f_a_min:
        return dict(band=B_FA, reason="F_A", naive_a=naive_a, f_a_possible=bool(naive_a >= spec.f_a_min))
    return dict(band=SELECTED, reason="PASS")


# ================================================================ the closing sentences (R.7, R.9.3; Readings 6, 8, 9)
_NUMS = " ({n}/21 대 {c}/21, F_a {f_a}/32)"
SENTENCES = {
    STOP_STRENGTH_LEVER: "APL→MBON05 제거 아래에서 E-grid k2-norm s 1.0이 KC 활성 자격({cond})을 잃었다.",
    STOP_PUNISH_BROKEN: "APL→MBON05 제거 아래에서 P의 처벌 학습 확인이 재현되지 않았다({label}, ℓ_r1 {l1}, ℓ_r2 {l2}).",
    STOP_EVEN_LOW_LEVER: "APL→MBON05 제거 아래 짝수 (b) 21쌍에서 testable_b가 M2 기준(11)에 못 미쳤다"
                         "({tb}/21, 지렛대 없는 같은 실행 {c_even}/21, F_a 기록 {k}/18).",
    STOP_C_EVEN_MISMATCH: "지렛대 없는 같은 실행의 짝수 (b) testable_b가 인코더 13절의 7/21과 달랐다({c_even}/21) — "
                          "측정 경로 결함 신호로 기록하고 멈춘다(R.9.6).",
    B_TB: "APL→MBON05 제거가 판정 세트(L 생성기 턴 64–{T})에서 {n}/21로 지렛대 없는 같은 세트 {c}/21보다 오르지 않았다."
          " → 이 지렛대를 닫는다.",
    f"{B_NC}:{C_ABOVE_BAR}": "지렛대 없는 E-grid가 이 세트에서 이미 기준을 넘어 지렛대 효과로 말할 수 없다." + _NUMS,
    f"{B_NC}:{BELOW_BAR}": "지렛대 없는 쪽보다 올랐지만 M2 기준에 못 미쳤다." + _NUMS,
    f"{B_NC}:{MARGIN}": "기준은 넘었으나 지렛대 없는 쪽 대비 여유가 2 미만이다" + _NUMS,
    B_PG: "M2 (b) 기준과 여유는 넘었지만 지렛대가 처벌 통과를 줄이거나 바꿨다(처벌 통과 (b) {pb_L} 대 {pb_C}, (a) {pa_L} 대 "
          "{pa_C}; pass→fail (b) {k_b}, (a) {k_a}).",                       # R.9.5 replaces R.7
    B_FA: "M2 (b) 기준·여유·처벌 가드는 넘었지만 F_a가 기준에 못 미쳤다({n}/21 대 {c}/21, F_a {f_a}/32, naive_a {naive_a})."
          " 다음 병목은 F_a(순진 균형 (a) 쌍)다.",
    SELECTED: "C3에서 APL→MBON05 2간선을 지운 모델 변형과 E-grid k2-norm(s 1.0)에서, 판정 세트(L 생성기 턴 64–{T}, E-grid 키 "
              "중복 제외) (b) 21쌍의 H.4 오라클 시험 가능성이 M2 기준을 만족했고 지렛대 없는 같은 세트보다 높았다({n}/21 대 "
              "{c}/21, 여유 ≥ 2, F_a {f_a}/32, 처벌 가드 통과, 짝수 {k_even}/21, P 재현 `LEARNS_CONFIRMATORY`, 판정 시드 "
              "24_100_xxx). 지렛대는 Q 결과를 보고 골랐고 Q의 사전 ② 일치 조건은 넘지 못했다(R.0). 실제 커넥톰 간선을 지운 "
              "모델이며 오라클 시험 가능성이지 학습 시험이 아니다.",
}


def sentence(outcome: str, fields: dict) -> str:
    """The closing sentence with fields filled; a missing field or an unknown outcome raises KeyError. B_결론없음
    takes its line from fields["reason"]."""
    key = f"{B_NC}:{fields['reason']}" if outcome == B_NC else outcome
    return SENTENCES[key].format(**fields)


# ================================================================ the operating characteristic (R.6, R.9.5, Reading 13)
def g_axis(N: int, a_pf: float, a_fp: float, spec) -> float:
    """P(drop ∨ flips) on an axis of N independent pairs, each C pass ∧ L fail with probability a_pf and C fail ∧ L
    pass with a_fp (multinomial; drop ⇔ k_pf − k_fp ≥ g_fail_drop, flips ⇔ k_pf ≥ g_fail_flips)."""
    rest = 1 - a_pf - a_fp
    tot = 0.0
    for k in range(N + 1):
        for j in range(N - k + 1):
            if k - j >= spec.g_fail_drop or k >= spec.g_fail_flips:
                tot += comb(N, k) * comb(N - k, j) * a_pf ** k * a_fp ** j * rest ** (N - k - j)
    return tot


def g_prob(a_pf: float, a_fp: float, spec) -> float:
    """G_fail on (b) n_b or (a) n_a pairs, the axes independent."""
    return 1 - (1 - g_axis(spec.n_b, a_pf, a_fp, spec)) * (1 - g_axis(spec.n_a, a_pf, a_fp, spec))


def _cell(q: float, c: int, naive_a: int, g: float, spec) -> dict:
    pn, pf = e_rules._binom(spec.n_b, q), e_rules._binom(naive_a, q)
    P = {b: 0.0 for b in BANDS}
    for i, wi in enumerate(pn):
        for j, wj in enumerate(pf):
            for gv, wg in ((False, 1 - g), (True, g)):
                if wg:
                    P[read_band(i, c, j, gv, spec.n_b, spec.n_a, naive_a, spec)["band"]] += wi * wj * wg
    return P


def oc(spec) -> dict:
    """Exact conditional OC: n ~ Bin(n_b, q), F_a ~ Bin(naive_a, q), G_fail ~ Bernoulli(g), c fixed, each cell weighted
    through read_band. g rows: none, the null false alarm (a_pf = a_fp ∈ oc_discord) and harm rows (oc_harm)."""
    g_rows = ([dict(kind="null", a_pf=a, a_fp=a, p=g_prob(a, a, spec)) for a in spec.oc_discord]
              + [dict(kind="harm", a_pf=x, a_fp=y, p=g_prob(x, y, spec)) for x, y in spec.oc_harm])
    rows = []
    for q in spec.oc_q:
        for c in spec.oc_c:
            for na in range(min(spec.n_a, spec.oc_naive_max) + 1):
                for gr in [dict(kind="none", a_pf=0.0, a_fp=0.0, p=0.0)] + g_rows:
                    rows.append(dict(q=q, c=c, naive_a=na, g_kind=gr["kind"], a_pf=gr["a_pf"], a_fp=gr["a_fp"],
                                     g=gr["p"], P=_cell(q, c, na, gr["p"], spec)))
    table = dict(assumption="pairs independent; q_b = q_a; c fixed; G_fail independent of n and F_a with probability "
                            "g from a per-pair discordance model per axis, OR over (b) and (a)",
                 g_fail=g_rows, rows=rows)
    return dict(table, sha256=hashlib.sha256(canonical(table).encode()).hexdigest())
