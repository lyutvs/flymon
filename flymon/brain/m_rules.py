"""The pure rules of spec appendix M (M.10.3-M.10.5 with M.10.7's reading fixes): per-arm counts (n_r = pairs with
r >= testable_min, n_p = pairs with -p >= testable_min; an undefined pair counts as not passing), ranking (n, then the
median of r or -p over the defined pairs, then the core's KC input), top-k, the predicted joint counts (record),
overlap-free combinations (an arm with no candidate arrives here as [incumbent]), the bar-first choice (testable_b >= 11
and F_a >= 2 first), the judgement list (L set turns >= 4, first 21 (b) + those turns' (a), digests pinned), the reading
against the same-set C3 and the closing sentences with their scope. Nothing here runs the engine."""
from __future__ import annotations

import numpy as np

from .h4_formula import pair_stats
from .h4_pairs import pairs_digest
from .j_rules import B_FA, B_NO_CONCLUSION, B_TB, SELECTED   # same strings; M.10.5 gives them M's conditions below
from .m_cands import STOP_NO_SPECIFICITY, overlap
from .m_spec import SPEC as DECLARED

STAGE2_GO = "STAGE2_GO"                   # the stage-2 winner meets the M2 bar on the even pairs (M.10.3)
STOP_NO_GAIN = "STOP_NO_GAIN"             # no top-2 x 2 combination meets it
STOP_NO_CANDIDATE = "STOP_NO_CANDIDATE"   # no guard-passing candidate in either arm (M.10.7)
STOP_INCUMBENT = "STOP_INCUMBENT"         # an incumbent population fails the guard or z (plan reading 7, the user decides)
C3_PASSES = "C3 자체가 기준을 넘어 판독 확장의 효과로 말할 수 없다"
SCOPE = "팔별 상위 2 × 2 조합"


# ================================================================ stage 1 (M.10.3)
def arm_rows(reports: list, z: dict, arm: str, h4spec) -> dict:
    """reports: the rows' report dicts ({pre, R1, R2} of {A, P}) in pair order; None = a missing row (not passing)."""
    if arm not in ("reward", "punish"):
        raise ValueError(f"arm must be reward or punish, not {arm!r}")
    st = [pair_stats(rep, z, h4spec.testable_min) if rep is not None else None for rep in reports]
    r = [s["r"] if s else None for s in st]
    p = [s["p"] if s else None for s in st]
    val = r if arm == "reward" else [(-x if x is not None else None) for x in p]
    ok = [v for v in val if v is not None]
    return dict(n=int(sum(1 for v in ok if v >= h4spec.testable_min)), med=float(np.median(ok)) if ok else None,
                r=r, p=p, defined=len(ok))                                  # M.10.8: no defined pair -> None (null)


def low_none(x) -> tuple:
    """A sort key for "higher first" with None (an undefined median, M.10.8) lowest."""
    return (x is None, -x if x is not None else 0.0)


def rank(entries: list, arm: str) -> list:
    """Best first: n, then the median of r (reward) or -p (punish; None = undefined, lowest), then the core's KC input
    (Σ w_c)."""
    return sorted(entries, key=lambda e: (-e["n"], low_none(e["med"]), -e["kc_input"]))


def top(ranked: list, k: int) -> list:
    return ranked[:k]                                                       # all of them when fewer than k


def predicted_joint(reward_r: dict, punish_p: dict, tmin: float) -> dict:
    """{"i|j": #pairs with r_i >= tmin and -p_j >= tmin on the same pair}; a record, not a judgement."""
    out = {}
    for i, r in reward_r.items():
        for j, p in punish_p.items():
            out[f"{i}|{j}"] = int(sum(1 for a, b in zip(r, p) if a is not None and b is not None and a >= tmin
                                      and -b >= tmin))
    return out


# ================================================================ stage 2 (M.10.3)
def combos(top_r: list, top_p: list) -> tuple:
    """(top_r x top_p in top_r-major order minus overlapping groups, [[reward, punish] excluded])."""
    ok, bad = [], []
    for a in top_r:
        for b in top_p:
            if overlap(a, b):
                bad.append([a["name"], b["name"]])
            else:
                ok.append((a, b))
    return ok, bad


def meets_bar(agg: dict, jspec) -> bool:
    return bool(agg["testable_b"] >= jspec.stage2_select_testable_b and agg["F_a"] >= jspec.h4.f_a_min)


def choose(aggs: dict, jspec) -> list:
    """[(name, agg)] best first: bar met, then testable_b, F_a, median m (None = undefined, lowest; stable on full
    ties)."""
    return sorted(aggs.items(), key=lambda kv: (not meets_bar(kv[1], jspec), -kv[1]["testable_b"], -kv[1]["F_a"],
                                                 low_none(kv[1].get("m_median", 0.0))))


# ================================================================ the judgement list (M.10.4)
def judgement_set(new_all: dict, spec) -> dict:
    """new_all: l_pairs.new_pairs with a_turns = n_turns. The first judge_n_b (b) pairs with turn >= judge_from_turn and
    every (a) pair of their turns; ValueError unless both digests are the pinned ones (an empty pin = smoke)."""
    b = [p for p in new_all["b"] if int(p["turn"]) >= spec.judge_from_turn][:spec.judge_n_b]
    turns = {int(p["turn"]) for p in b}
    a = [p for p in new_all["a"] if int(p["turn"]) in turns]
    bd, ad = pairs_digest(b), pairs_digest(a)
    if (spec.judge_b_digest or spec.judge_a_digest) and (bd != spec.judge_b_digest or ad != spec.judge_a_digest):
        raise ValueError(f"judgement list digest (b {bd[:12]}, a {ad[:12]}) is not the pinned "
                         f"(b {spec.judge_b_digest[:12]}, a {spec.judge_a_digest[:12]}) — M.10.4")
    return dict(b=b, a=a, b_digest=bd, a_digest=ad)


# ================================================================ stage 3 (M.10.5, M.10.7)
def stage3_reading(agg: dict, agg_c3: dict, spec) -> dict:
    """SELECTED iff n >= 11, F_a >= 2, n > c; B_Fa iff n >= 11, F_a < 2, n > c; B_NO_CONCLUSION iff n >= 11 and c >= n
    (M.10.7: the specific clause wins) or c < n < 11; B_Tb iff n <= c and n < 11. The note C3_PASSES whenever c >= 11.
    No reading unless both lists have the declared 21 (b) pairs (so never under smoke)."""
    j = spec.j
    n, c, fa, n_b = int(agg["testable_b"]), int(agg_c3["testable_b"]), int(agg["F_a"]), int(agg["n_b"])
    base = dict(n=n, c=c, F_a=fa, n_b=n_b, naive_a=int(agg["naive_a"]),
                f_a_possible=bool(agg["naive_a"] >= j.h4.f_a_min))
    if not (n_b == int(agg_c3["n_b"]) == spec.judge_n_b == DECLARED.judge_n_b):
        return dict(base, outcome=None, note=f"pair list is not the declared {DECLARED.judge_n_b}: not a judgement")
    bar_n = j.stage2_select_testable_b
    if n >= bar_n:
        out = (SELECTED if fa >= j.h4.f_a_min else B_FA) if n > c else B_NO_CONCLUSION
    else:
        out = B_TB if n <= c else B_NO_CONCLUSION
    return dict(base, outcome=out, note=C3_PASSES if c >= bar_n else None)


def sentence(outcome: str, ctx: dict) -> str:
    """M.10.5's sentences, the scope written into them. ctx keys by outcome: STOP_NO_GAIN {combo, n, fa}, or
    {excluded} when no combination was left (M.10.8); B_Tb / B_Fa /
    B_NO_CONCLUSION {combo, n, c, k (even selection), h (odd record), n_b} + B_Fa {fa, naive_a, f_a_possible} +
    B_NO_CONCLUSION {fa, note}; SELECTED {combo, n, c, fa, k, h, n_b, moves, top_move, n_cands, m, w, dan = (PPL, PAM)}.
    Every listed key is looked up with ctx[...] (a missing one raises KeyError)."""
    if outcome == STOP_NO_SPECIFICITY:
        return ("C3에서 어느 PPL1·PAM 구획의 core KC 입력도 짝수 (b) 21쌍의 X 전용 가중 몫이 현 조합(PPL105·PAM08)보다 "
                "크지 않았다. → 이 판독 확장을 닫는다.")
    if outcome == STOP_NO_CANDIDATE:
        return "특이성 사전 검사를 통과한 후보 가운데 반응 가드를 통과한 core 집단이 없었다. → 이 판독 확장을 닫는다."
    if outcome == STOP_INCUMBENT:
        return "현 조합(PPL105·PAM08)의 core 집단 판독이 반응 가드 또는 z 검사를 통과하지 못했다. → 기록하고 사용자가 판단한다."
    if outcome == STOP_NO_GAIN and "combo" not in ctx:                     # M.10.8: every combination overlapped
        return (f"특이성 검사를 통과한 후보의 {SCOPE}(집단 판독) 가운데 겹치지 않는 조합이 없었다(core 겹침으로 제외 "
                f"{ctx['excluded']}). → 이 범위의 판독 확장을 닫는다.")
    if outcome == STOP_NO_GAIN:
        return (f"특이성 검사를 통과한 후보의 {SCOPE}(집단 판독) 가운데 짝수 (b) 쌍에서 M2 기준을 넘은 조합이 "
                f"없었다(최대 {ctx['n']}/21·F_a {ctx['fa']}, {ctx['combo']}). → 이 범위의 판독 확장을 닫는다.")
    if outcome not in (B_TB, B_FA, B_NO_CONCLUSION, SELECTED):
        return f"{outcome}: 기록하고 사용자가 판단한다 ({ctx})."
    nb = ctx["n_b"]                                           # a missing key fails loudly (never a silent None)
    tail = f"(짝수 선택 {ctx['k']}/21, 홀수 기록 {ctx['h']}/20)"
    if outcome == B_TB:
        return (f"선택된 구획 쌍 {ctx['combo']}({SCOPE}에서 선택)의 집단 판독이 새 판정 세트에서 "
                f"{ctx['n']}/{nb}로 같은 세트의 C3 {ctx['c']}/{nb}보다 오르지 않았다{tail}. → 이 범위의 판독 확장을 닫는다.")
    if outcome == B_FA:
        return (f"선택된 구획 쌍 {ctx['combo']}({SCOPE}에서 선택)의 집단 판독이 새 판정 세트 (b)에서 {ctx['n']}/{nb}로 "
                f"같은 세트의 C3 {ctx['c']}/{nb}보다 높았으나 (a) F_a {ctx['fa']} < 2였다(naive_a {ctx['naive_a']}, "
                f"f_a_possible {ctx['f_a_possible']}){tail}. → 기록하고 사용자가 판단한다.")
    if outcome == B_NO_CONCLUSION:
        note = f" {ctx['note']}." if ctx["note"] else ""
        return (f"선택된 구획 쌍 {ctx['combo']}({SCOPE}에서 선택)의 집단 판독이 새 판정 세트에서 {ctx['n']}/{nb}"
                f"(F_a {ctx['fa']}), 같은 세트의 C3 {ctx['c']}/{nb}였다{tail}.{note} → 기록하고 사용자가 판단한다.")
    if outcome == SELECTED:
        ppl, pam = ctx["dan"]
        return (f"C3 엔진에서 구획 {ctx['combo']}({SCOPE}에서 선택)·core 집단 판독이 새 판정 세트 (b) {nb}쌍에서 오라클 "
                f"편집으로 M2 기준을 만족했고 같은 세트 C3보다 높았다({ctx['n']}/{nb} 대 {ctx['c']}/{nb}, F_a "
                f"{ctx['fa']}; 홀수 기록 {ctx['h']}/20; 기술 {ctx['moves']}종·{ctx['top_move']}). 오라클 시험 가능성이며 "
                f"학습 시험이 아니다. {ctx['n_cands']}후보(특이성 통과 {ctx['m']}개)에서 골랐고 짝수 21쌍은 G·H·J·K·L·M에 "
                f"걸쳐 선택에 쓰였다. I 뒤 다섯 번째 선언(J·K·L·M)이다. core w_mbon {ctx['w']}, DAN 세포 수 "
                f"PPL {ppl}·PAM {pam}. → F v4 학습 시험 선언(3.4 펄스 표 개정, J.12.8 억압 진단을 선행 과제로).")
