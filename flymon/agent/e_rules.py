"""Spec 4.3, 4.4, 5.3-5.5: the encoder track's decisions. The judgement code is the authoritative source.
Every number comes from e_spec.SPEC; the sentences are spec 5.4 verbatim with 〈…〉 replaced by {field}."""
from __future__ import annotations

from math import comb, floor
from statistics import median

SELECTED, B_TB, B_FA, B_NC, NOT_READ = "SELECTED", "B_Tb", "B_Fa", "B_결론없음", "NOT_READ"
STOP_NO_ELIGIBLE, STOP_EVEN_LOW, STOP_SET_SHORT = "STOP_NO_ELIGIBLE", "STOP_EVEN_LOW", "STOP_SET_SHORT"
UNDECIDED = "UNDECIDED"
E0_ABOVE_BAR, BELOW_BAR, MARGIN = "E0_ABOVE_BAR", "BELOW_BAR", "MARGIN"
BANDS = (SELECTED, B_TB, B_FA, B_NC)


# ---- 4.3 strength --------------------------------------------------------------------------------------------

def odour_activity(per_seed: dict) -> dict:
    """Per-odour KC activity = median over the seeds of that odour (E3)."""
    return {k: float(median(v)) for k, v in per_seed.items()}


def strength_ok(single, dual, spec) -> dict:
    """Conditions 1-3 of 4.3 on per-odour activities. The tail limits apply to the single-type and dual-type
    odours separately (four shares), never to all of them pooled."""
    single, dual = list(single), list(dual)
    if not single or not dual:
        raise ValueError("strength_ok needs both single-type and dual-type odours")

    def share(xs, f):
        return sum(1 for x in xs if f(x)) / len(xs)

    lo = lambda x: x < spec.tail_lo
    hi = lambda x: x > spec.tail_hi
    med = float(median(single + dual))
    d = dict(median=med, lo_single=share(single, lo), lo_dual=share(dual, lo),
             hi_single=share(single, hi), hi_dual=share(dual, hi))
    d["ok"] = bool(spec.kc_band[0] <= med <= spec.kc_band[1]
                   and all(d[k] <= spec.tail_share_max for k in ("lo_single", "lo_dual", "hi_single", "hi_dual")))
    return d


def choose_strength(table: dict, spec) -> dict:
    """Eligible-first: cap_ok=False (condition 4) -> not eligible; then strength_ok; among eligible s the median
    closest to kc_target, ties to the smaller s. s=None only when no s is eligible."""
    reasons, ok = {}, {}
    for s, row in sorted(table.items()):
        if not row["cap_ok"]:
            reasons[s] = "ORN_CAP"
            continue
        st = strength_ok(row["single"], row["dual"], spec)
        reasons[s] = "OK" if st["ok"] else "KC"
        if st["ok"]:
            ok[s] = st["median"]
    if not ok:
        return dict(s=None, reasons=reasons)
    s = min(ok, key=lambda v: (abs(ok[v] - spec.kc_target), v))
    return dict(s=s, reasons=reasons)


# ---- 4.4 selection -------------------------------------------------------------------------------------------

def select_config(results: dict, spec) -> dict:
    """Pass-first: passing = eligible ∧ testable_b ≥ bar_b ∧ F_a ≥ f_a_min; among passing configs within tie_pairs
    of the top testable_b, the earliest in spec.configs order."""
    unknown = set(results) - set(spec.configs)
    if unknown:
        raise ValueError(f"undeclared configs: {sorted(unknown)}")
    elig = {c: r for c, r in results.items() if r["eligible"]}
    if not elig:
        return dict(outcome=STOP_NO_ELIGIBLE, winner=None, passing=[], near=[])
    passing = [c for c in spec.configs
               if c in elig and elig[c]["testable_b"] >= spec.bar_b and elig[c]["F_a"] >= spec.f_a_min]
    if not passing:
        return dict(outcome=STOP_EVEN_LOW, winner=None, passing=[], near=[])
    top = max(elig[c]["testable_b"] for c in passing)
    near = [c for c in passing if top - elig[c]["testable_b"] <= spec.tie_pairs]
    return dict(outcome=SELECTED, winner=near[0], passing=passing, near=near)


# ---- 5.3 bands -----------------------------------------------------------------------------------------------

def read_band(n, c, f_a, n_b, n_a, n_a_declared, naive_a, spec) -> dict:
    """The first matching line of 5.3, in the fixed order."""
    if n_b != spec.n_b or n_a != n_a_declared:
        return dict(band=NOT_READ, reason="COUNTS")
    if c >= spec.bar_b:
        return dict(band=B_NC, reason=E0_ABOVE_BAR)
    if n <= c:
        return dict(band=B_TB, reason="NO_GAIN")
    if n < spec.bar_b:
        return dict(band=B_NC, reason=BELOW_BAR)
    if n - c < spec.margin:
        return dict(band=B_NC, reason=MARGIN)
    if f_a < spec.f_a_min:
        return dict(band=B_FA, reason="F_A", naive_a=naive_a, f_a_possible=naive_a >= spec.f_a_min)
    return dict(band=SELECTED, reason="PASS")


# ---- 5.4 sentences -------------------------------------------------------------------------------------------

_NUMS = " ({n}/21 대 {c}/21, F_a {f_a}/{n_a})"
SENTENCES = {
    STOP_NO_ELIGIBLE: "결합 부호 E-grid의 선언한 4설정 가운데 하드 제약 위반 0과 KC 활성 자격(중앙값 5–9%, 3% 미만 ≤ 5%)을 "
                      "함께 만족한 설정이 없었다({reasons}).",
    STOP_EVEN_LOW: "E-grid 자격 설정 가운데 짝수 (b) 21쌍에서 M2 기준(testable_b ≥ 11 ∧ F_a ≥ 2)을 넘은 설정이 없었다"
                   "(설정별 {table}).",
    STOP_SET_SHORT: "L 생성기 턴 64–209에서 E-grid 키 중복을 뺀 (b) 쌍이 21개에 못 미쳤다({m}쌍).",
    UNDECIDED: "k = {k}의 하드 제약 해를 구성·정확 탐색으로 찾지 못했고 불가능도 증명하지 못했다.",
    B_TB: "선택된 설정 {config}(4설정에서 짝수 턴으로 선택, 짝수 {k_even}/21)이 판정 세트(L 생성기 턴 64–{T})에서 "
          "{n}/21로 같은 세트 E0의 {c}/21보다 오르지 않았다.",
    f"{B_NC}:{E0_ABOVE_BAR}": "E0 자체가 이 세트에서 기준을 넘어 인코더 효과로 말할 수 없다." + _NUMS,
    f"{B_NC}:{BELOW_BAR}": "E0보다 올랐지만 M2 기준에 못 미쳤다." + _NUMS,
    f"{B_NC}:{MARGIN}": "기준은 넘었으나 E0 대비 여유가 2 미만이다" + _NUMS,
    B_FA: "M2 (b) 기준과 여유는 넘었지만 순진 균형 (a) 쌍의 F_a가 기준에 못 미쳤다"
          "({n}/21 대 {c}/21, F_a {f_a}/{n_a}, naive_a {naive_a}).",
    SELECTED: "C3 엔진·H.4 판독(MBON13/05)에서 결합 부호 E-grid {config}이 판정 세트(L 생성기 턴 64–{T}, E-grid 키 중복 제외) "
              "(b) 21쌍에서 오라클 편집으로 M2 기준을 만족했고 같은 세트 E0보다 높았다 "
              "({n}/21 대 {c}/21, 여유 ≥ 2, F_a {f_a}/{n_a}, 짝수 선택 {k_even}/21, 판정 시드 24_100_xxx). "
              "오라클 시험 가능성이며 학습 시험이 아니다. 4설정에서 짝수 턴으로 골랐고, 짝수 21쌍은 G·H·J·K·L·M에서 선택에 쓰였다. "
              "축 쌍의 입력 사구체 겹침은 0이며 결합 간 일반화는 측정하지 않았다.",
}


def sentence(outcome: str, fields: dict) -> str:
    """The 5.4 closing sentence with fields filled; a missing field (or unknown outcome) raises KeyError.
    B_결론없음 takes its reason from fields["reason"] unless the key already carries it ("B_결론없음:MARGIN")."""
    key = f"{B_NC}:{fields['reason']}" if outcome == B_NC else outcome
    return SENTENCES[key].format(**fields)


# ---- 5.5 operating characteristics ---------------------------------------------------------------------------

def _binom(n: int, q: float) -> list:
    return [comb(n, i) * q ** i * (1 - q) ** (n - i) for i in range(n + 1)]


def _naive_from_even_share(n_a_row: int) -> int:
    return int(floor(n_a_row * 4 / 18 + 0.5))


NAIVE_A_RULE = "naive_a = round(n_a_row * 4 / 18) (C3 even-set naive share 4/18)"


def _cell(q, c, naive_a, n_a, spec) -> dict:
    pn, pf = _binom(spec.n_b, q), _binom(naive_a, q)
    P = {b: 0.0 for b in BANDS}
    for i, wi in enumerate(pn):
        for j, wj in enumerate(pf):
            P[read_band(i, c, j, spec.n_b, n_a, n_a, naive_a, spec)["band"]] += wi * wj
    return P


def operating_characteristics(n_a: int, spec) -> dict:
    """Exact conditional OC: n ~ Bin(n_b, q), F_a ~ Bin(naive_a, q), c fixed, each cell weighted through read_band.
    Rows: base q × c × naive_a ∈ 0..min(n_a, oc_naive_max); c ± oc_c_delta (clipped to 0..n_b) over the same naive_a;
    n_a rows oc_n_a_rows with naive_a from the even-set share 4/18."""
    rows = []
    naive_range = range(min(n_a, spec.oc_naive_max) + 1)
    for q in spec.oc_q:
        for c in spec.oc_c:
            for na in naive_range:
                rows.append(dict(q=q, c=c, naive_a=na, n_a=n_a, tag="base", P=_cell(q, c, na, n_a, spec)))
            for cd in (c - spec.oc_c_delta, c + spec.oc_c_delta):
                cd = min(max(cd, 0), spec.n_b)
                for na in naive_range:
                    rows.append(dict(q=q, c=cd, c_base=c, naive_a=na, n_a=n_a, tag="c_delta",
                                     P=_cell(q, cd, na, n_a, spec)))
            for nar in spec.oc_n_a_rows:
                na = _naive_from_even_share(nar)
                rows.append(dict(q=q, c=c, naive_a=na, n_a=nar, tag="n_a_row", naive_a_rule=NAIVE_A_RULE,
                                 P=_cell(q, c, na, nar, spec)))
    return {"assumption": "pairs independent; q_b = q_a; conditional on fixed c; n_a rows use " + NAIVE_A_RULE,
            "rows": rows}
