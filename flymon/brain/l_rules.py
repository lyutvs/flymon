"""Spec L.5 / L.11.3 as pure functions: the oracle rows' checks (H.4's G.14.4 row 1 without its even-turn rule — L
labels odd and new turns), the screening stop and coverage (reading 10: n* = the order position of the n-th pass, pairs
after it are neither screened nor counted), the lift draw (decision 12), the reading (J.12.9's bands through
j_rules.stage2_reading on 21 pairs), the diversity record and the closing sentences. Nothing here runs the engine; pair
lists reach these functions as keys or feature rows, and the stage CLIs run l_pairs.check_digests before building them."""
from __future__ import annotations

import numpy as np

from .h4_formula import pair_stats
from .j_rules import B_FA, B_NO_CONCLUSION, B_TB, SELECTED, stage2_reading
from .l_screen import SCREEN_FEW, SCREEN_IMPRECISE, SCREEN_NO_RULE, passes
from .l_spec import SPEC

SCREENED = "SCREENED"                     # n_pass passes by order position max_screened: stage 3 runs
COVERAGE_SHORT = "COVERAGE_SHORT"         # fewer than n_pass by min(max_screened, len): no oracle, the claim is too narrow
_key = lambda r: (r["axis"], int(r["turn"]), r["x"], r["y"])


# ================================================================ the oracle rows (G.14.4 row 1, any turn)
def pair_rows_stats(rows: list, z: dict, expected: list, spec4) -> dict:
    """h4_rules.combo_stats' checks minus "odd-turn rows present" and minus the aggregate: duplicate / missing / extra
    keys, probes not on spec4.report_seeds, undefined d', NaN. stats = {key: pair_stats} for every well-formed row."""
    reasons = []
    keys = [_key(r) for r in rows]
    if len(set(keys)) != len(keys):
        reasons.append("duplicate pair rows")
    exp = {tuple(k) for k in expected}
    if set(keys) != exp:
        reasons.append(f"pairs differ from the declared list (missing {len(exp - set(keys))}, "
                       f"extra {len(set(keys) - exp)})")
    n = len(spec4.report_seeds)
    short = {_key(r) for r in rows if any(len(r["report"][ph][k]) != n for ph in ("pre", "R1", "R2") for k in "AP")}
    if short:
        reasons.append(f"{len(short)} pairs whose report probes are not the {n} report seeds")
    stats = {_key(r): pair_stats(r["report"], z, spec4.testable_min) for r in rows if _key(r) not in short}
    undefined = [k for k, s in stats.items() if s is None]
    if undefined:
        reasons.append(f"{len(undefined)} pairs with an undefined d'")
    nan = [k for k, s in stats.items() if s is not None and any(np.isnan(s[f]) for f in ("d_pre", "r", "p", "m"))]
    if nan:
        reasons.append(f"{len(nan)} pairs with a NaN statistic")
    return dict(reasons=reasons, stats=stats,
                pairs=[dict(axis=k[0], turn=k[1], x=k[2], y=k[3], **(s or {})) for k, s in stats.items()])


# ================================================================ stage 2: the screen over the declared order
def screen_order(feats: list, rule, n_pass: int, max_screened: int) -> dict:
    """feats in declared (b) order (whatever was measured, batches included). Stops at the n_pass-th pass: n* is its
    1-based position, coverage = n_pass / n*, and later rows (the rest of the last batch) are not screened. Fewer than
    n_pass passes by position min(max_screened, len(feats)) is COVERAGE_SHORT (coverage over the screened rows)."""
    passed, screened = [], []
    for i, f in enumerate(feats[:max_screened]):
        screened.append(i)
        if passes(rule, f):
            passed.append(i)
            if len(passed) == n_pass:
                return dict(outcome=SCREENED, passed=passed, n_star=i + 1, screened=screened,
                            coverage=n_pass / (i + 1))
    return dict(outcome=COVERAGE_SHORT, passed=passed, n_star=None, screened=screened,
                coverage=len(passed) / len(screened) if screened else 0.0)


def lift_draw(failed_idx: list, n: int, seed: int) -> list:
    """Decision 12: n of the screened failing pairs by default_rng(seed) without replacement (all of them if fewer)."""
    if len(failed_idx) <= n:
        return list(failed_idx)
    pick = np.random.default_rng(seed).choice(len(failed_idx), size=n, replace=False)
    return [failed_idx[int(i)] for i in pick]


# ================================================================ stage 3
def stage3_reading(agg: dict, spec) -> dict:
    """J.12.9's bands, unchanged: j_rules.stage2_reading on the L spec's J spec (band None off 21 pairs)."""
    return stage2_reading(agg, spec.j)


def diversity(pairs: list) -> dict:
    """The passing (b) pairs' distinct my-species / moves / original-opponent types and the top move's share. Each
    pair carries `me` and `opp_types` (the CLI adds them from its turn); the move is x's part before " vs "."""
    moves = [p["x"].split(" vs ")[0] for p in pairs]
    top = max(moves.count(m) for m in set(moves)) if moves else 0
    return dict(n_me=len({p["me"] for p in pairs}), n_moves=len(set(moves)),
                n_opp_types=len({t for p in pairs for t in p["opp_types"]}),
                top_move_share=top / len(moves) if moves else 0.0)


# ================================================================ closing sentences (L.5 as replaced by L.11.2 / L.11.3)
def _lift(c: dict) -> str:
    return f"통과 {c['n']}/{c['n_b']} 대 불통과 표본 {c['m']}/{c['n_lift']}"


def _fa(c: dict, spec) -> str:
    """Decision 9: F_a is read from a fixed sample and is an engine-wide requirement."""
    return (f"F_a {c['k']}는 새 세트 앞 {spec.a_turns}턴의 고정 (a) 표본에서 나온다 — F_a는 엔진 전체의 요건이며, "
            f"선별된 상황의 요건이 아니다")


def sentence(state: str, c: dict) -> str:
    """ctx keys by state — SCREEN_FEW: n_pass, n; SCREEN_IMPRECISE: precision, k, n_pass; SCREEN_NO_RULE: cov;
    COVERAGE_SHORT: rule, n_pass, n_screened, floor; stage 3 (B_Tb / B_NO_CONCLUSION / B_Fa / SELECTED): rule, coverage,
    n (testable_b), n_b, k (F_a), m (testable lift pairs), n_lift (lift sample size), plus naive_a / f_a_possible on
    B_Fa and p / q / n_opp_types / top_move_share (diversity) on SELECTED. Optional `spec` (default SPEC)."""
    spec = c.get("spec", SPEC)
    if state == SCREEN_FEW:
        return f"선별 규칙이 보정 쌍에서 너무 적게 통과해 정밀도를 판단할 수 없었다({c['n_pass']}/{c['n']})."
    if state == SCREEN_IMPRECISE:
        return (f"순진 판독(G와 S·f2·f3 중 하나의 문턱)으로 고른 (b) 쌍의 시험 가능 비율이 LOTO {c['precision']:.3g}"
                f"({c['k']}/{c['n_pass']})였다.")
    if state == SCREEN_NO_RULE:
        return f"41쌍 전체에서 커버리지 {c['cov']}를 만족하는 선별 규칙이 없었다."
    if state == COVERAGE_SHORT:
        return (f"선별 규칙 {c['rule']}은 새 세트에서 커버리지 {c['floor']:.3g}에 못 미쳤다"
                f"({c['n_pass']}/{c['n_screened']}).")
    base = spec.j.stage2_close_max_testable_b
    if state == B_TB:
        return (f"C3에서 순진 판독 선별 {c['rule']}을 통과한 새 (b) {c['n_b']}쌍의 시험 가능 쌍이 {c['n']}/{c['n_b']}로 "
                f"선별하지 않은 C3({base}/{spec.j.stage2_n_b})보다 오르지 않았다(커버리지 {c['coverage']:.3g}, 독립 세트; "
                f"{_lift(c)}).")
    if state == B_NO_CONCLUSION:
        return (f"{B_NO_CONCLUSION}: C3에서 순진 판독 선별 {c['rule']}(커버리지 {c['coverage']:.3g})을 통과한 새 (b) "
                f"{c['n_b']}쌍의 시험 가능 쌍이 {c['n']}/{c['n_b']}로 선별하지 않은 C3({base}/{spec.j.stage2_n_b})보다 "
                f"올랐으나 기준에 못 미쳤다({_lift(c)}; {_fa(c, spec)}). 기록하고 사용자가 판단한다.")
    if state == B_FA:
        return (f"{B_FA}: C3에서 순진 판독 선별 {c['rule']}(커버리지 {c['coverage']:.3g})을 통과한 새 (b) {c['n_b']}쌍의 "
                f"시험 가능 쌍은 {c['n']}/{c['n_b']}였으나 F_a가 기준에 못 미쳤다({_fa(c, spec)}; naive_a {c['naive_a']}, "
                f"f_a_possible {c['f_a_possible']}; {_lift(c)}). 기록하고 사용자가 판단한다.")
    if state == SELECTED:
        return (f"C3에서 순진 판독 선별 {c['rule']}(커버리지 {c['coverage']:.3g})을 통과한 새 (b) {c['n_b']}쌍은 오라클 "
                f"편집에서 M2 기준을 만족했다({c['n']}/{c['n_b']}; {_fa(c, spec)}; {_lift(c)}; 내 포켓몬 {c['p']}종·"
                f"기술 {c['q']}개·원래 상대 타입 {c['n_opp_types']}개, 가장 많은 기술의 몫 {c['top_move_share']:.2g}). "
                f"이것은 오라클 시험 가능성이며 학습 시험이 아니다. 이 주장은 I(M2 no-go) 뒤 J·K·L의 네 번째 선언에서 나왔다.")
    raise ValueError(f"no sentence for {state!r}")
