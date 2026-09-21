"""The M0d H.4 decision rules (spec appendix H.4, amended by H.3a.1, H.3a.9 and H.4a) as pure functions over measured rows.

Nothing here runs the engine: readout reselection (reactivity, teachability, the readout, z constants), the row checks
of G.14.4's first row, the records kept next to the verdict and the selection. The per-pair arithmetic is h4_formula's (G.14's code).
Editing this module never invalidates a cached measurement (it is outside h4_measure.MEASURE_FILES).
"""
from __future__ import annotations

import numpy as np

from .h3_rules import mbon_type_stats
from .h4_formula import arm_aggregate, pair_stats

# ---- combination statuses at reselection ----------------------------------------------------------------------
READOUT_SELECTED = "readout_selected"
DROPPED_NO_READOUT = "dropped_no_readout"       # A or P empty after reselection (H.4 step 1: 탈락)
STOP_MULTI_TYPE = "stop_multiple_types"         # two types of one pool pass: H.4 does not say how to combine them
# ---- run outcomes ------------------------------------------------------------------------------------------------
INVALID = "INVALID"                             # G.14.4 row 1: missing / extra / duplicate / odd-turn rows, undefined d', NaN
SELECTED = "SELECTED"
STOP_NO_ELIGIBLE = "STOP_NO_ELIGIBLE"           # no combination with F_a >= 2 (H.4 step 3: 멈추고 사용자 판단)
STOP_LOW_T_B = "STOP_LOW_T_B"                   # the best T_b < 0.5 (멈추고 사용자 판단)


# ================================================================ readout reselection (H.4 step 1)
def reactivity(ref_rows: list, rest_by_seed: dict, types: list, spec) -> dict:
    """{type: h3_rules.mbon_type_stats}: the H.3 guard statistic — median over the reference presentations of
    (read - same-seed rest) >= react_med_delta_min and a zero-read share <= react_zero_share_max."""
    return {t: mbon_type_stats(ref_rows, rest_by_seed, t, spec.react_med_delta_min, spec.react_zero_share_max)
            for t in types}


def teach_type(rows: list, t: str, slot: str, spec) -> dict:
    """One type on one arm-and-order's rows: decreased <=> post < pre on the odour in `slot` ("plus" / "minus");
    teachable <=> decreased on at least teach_min_decreased of the declared seeds."""
    by_seed = {int(r["seed"]): r for r in rows}
    if sorted(by_seed) != sorted(spec.teach_seeds) or len(by_seed) != len(rows):
        raise ValueError(f"teach rows must cover the seeds {list(spec.teach_seeds)} once each")
    pre = [int(by_seed[s]["pre"][slot][t]) for s in spec.teach_seeds]
    post = [int(by_seed[s]["post"][slot][t]) for s in spec.teach_seeds]
    n = sum(b < a for a, b in zip(pre, post))
    return dict(slot=slot, pre=pre, post=post, n_decreased=n, teachable=bool(n >= spec.teach_min_decreased))


def teach_choice(rows: list, t: str, arm: str, spec) -> dict:
    """H.4a: a type is judged on the design odour it answers more — the larger median naive (pre) count over the teach
    seeds; a tie keeps M0c's assignment — taught in the order that pairs that odour with the arm's DAN. arms():
    punish_only drives the punishment DAN on the CS+ slot, reward_only the reward DAN on the CS- slot; order "ab" has
    CS+ = odour a (M0c), "ba" exchanges the odours. Naive counts do not depend on the arm or the order."""
    slot = "plus" if arm == "punish_only" else "minus"
    ab = [r for r in rows if r["arm"] == arm and r["order"] == "ab"]
    if sorted(int(r["seed"]) for r in ab) != sorted(spec.teach_seeds):      # else median([]) -> NaN -> odour a, silently
        raise ValueError(f"{arm} naive (order ab) rows must cover the seeds {list(spec.teach_seeds)} once each")
    naive = {"a": float(np.median([r["pre"]["plus"][t] for r in ab])),
             "b": float(np.median([r["pre"]["minus"][t] for r in ab]))}
    m0c = "a" if slot == "plus" else "b"
    odour = m0c if naive["a"] == naive["b"] else max(naive, key=naive.get)
    order = "ab" if odour == m0c else "ba"
    mine = [r for r in rows if r["arm"] == arm and r["order"] == order]
    res = teach_type(mine, t, slot, spec)
    other = teach_type(mine, t, "minus" if slot == "plus" else "plus", spec)      # recorded, not judged
    return dict(res, arm=arm, odour=odour, order=order, naive_median=naive, untaught_n_decreased=other["n_decreased"])


def pick_readout(react: dict, teach: dict, pools: dict) -> dict:
    """A (P) = the pool's types that are reactive and teachable. Either empty -> dropped; two in one pool -> stop."""
    chosen = {k: [t for t in pools[k] if react[t]["passes"] and teach[t]["teachable"]] for k in ("A", "P")}
    if not chosen["A"] or not chosen["P"]:
        return dict(status=DROPPED_NO_READOUT, readout=None, passing=chosen)
    if len(chosen["A"]) > 1 or len(chosen["P"]) > 1:
        return dict(status=STOP_MULTI_TYPE, readout=None, passing=chosen)
    return dict(status=READOUT_SELECTED, readout={"A": chosen["A"][0], "P": chosen["P"][0]}, passing=chosen)


def z_constants(ref_rows: list, readout: dict, ddof: int) -> dict:
    """{"A": (mean, sd), "P": (mean, sd)} of the readout type's count (all its cells) over the reference presentations
    (H.4: 기준 집합의 순진 응답, 타입별 좌우 세포 합). F.3's frozen constants are population SDs (ddof 0)."""
    out = {}
    for k, t in readout.items():
        x = np.array([r["types"][t] for r in ref_rows], float)
        out[k] = (float(x.mean()), float(x.std(ddof=ddof)))
        if not out[k][1] > 0:
            raise ValueError(f"readout {t}: zero SD over the reference set")
    return out


# ================================================================ the oracle rows (G.14.4)
def check_rows(rows: list, expected: list) -> list:
    """G.14.4 row 1 for one combination and variant: the reasons the rows cannot be judged (empty = fine)."""
    reasons = []
    keys = [(r["axis"], int(r["turn"]), r["x"], r["y"]) for r in rows]
    if len(set(keys)) != len(keys):
        reasons.append("duplicate pair rows")
    if any(k[1] % 2 for k in keys):
        reasons.append("odd-turn rows present")
    exp = {tuple(k) for k in expected}
    if set(keys) != exp:
        reasons.append(f"pairs differ from the declared list (missing {len(exp - set(keys))}, "
                       f"extra {len(set(keys) - exp)})")
    return reasons


def combo_stats(rows: list, z: dict, expected: list, spec) -> dict:
    """pair_stats for every well-formed row, the INVALID reasons and, when there are none, the aggregate. A row whose
    probes are not the report seeds is a reason, not a pair_stats call (unequal A / P lengths would raise in dv)."""
    reasons = check_rows(rows, expected)
    n = len(spec.report_seeds)
    key = lambda r: (r["axis"], int(r["turn"]), r["x"], r["y"])
    short = {key(r) for r in rows if any(len(r["report"][ph][k]) != n for ph in ("pre", "R1", "R2") for k in ("A", "P"))}
    if short:
        reasons.append(f"{len(short)} pairs whose report probes are not the {n} report seeds")
    stats = {key(r): pair_stats(r["report"], z, spec.testable_min) for r in rows if key(r) not in short}
    undefined = [k for k, s in stats.items() if s is None]
    if undefined:
        reasons.append(f"{len(undefined)} pairs with an undefined d'")
    nan = [k for k, s in stats.items() if s is not None and any(np.isnan(s[f]) for f in ("d_pre", "r", "p", "m"))]
    if nan:
        reasons.append(f"{len(nan)} pairs with a NaN statistic")
    agg = None if reasons else arm_aggregate(stats, spec.naive_max, spec.t_b_min, spec.f_a_min)
    return dict(reasons=reasons, aggregate=agg,
                pairs=[dict(axis=k[0], turn=k[1], x=k[2], y=k[3], **(s or {})) for k, s in stats.items()])


# ================================================================ records (not judged)
def _only(report: dict, keep: str) -> dict:
    """The report with the other readout type's counts set to 0: its term is then constant and cancels in V(X) - V(Y),
    so pair_stats measures the kept type alone (G.14.3: MBON13만·MBON05만의 d′)."""
    drop = "P" if keep == "A" else "A"
    return {ph: {keep: pr[keep], drop: [[0, 0] for _ in pr[drop]]} for ph, pr in report.items()}


def combo_records(rows: list, z: dict, spec) -> dict:
    """Per combination: the naive readout floor on the pairs' odours (mean count and zero share of each readout type over
    the report seeds and both candidates, G.14.6), the per-type statistics (G.14.3) and the testable (b) count on each
    half of the report seeds (the selection's seed noise). None of these enters the selection."""
    floor = {}
    for k in ("A", "P"):
        x = np.array([c for r in rows for pair in r["report"]["pre"][k] for c in pair], float)
        floor[k] = dict(mean=float(x.mean()), zero_share=float((x == 0).mean()), n=int(x.size))
    single = {}
    for k in ("A", "P"):
        st = {(r["axis"], int(r["turn"]), r["x"], r["y"]): pair_stats(_only(r["report"], k), z, spec.testable_min)
              for r in rows}
        ok = {key: s for key, s in st.items() if s is not None}
        single[k] = dict(aggregate=arm_aggregate(ok, spec.naive_max, spec.t_b_min, spec.f_a_min) if ok else None,
                         pairs=[dict(axis=key[0], turn=key[1], x=key[2], y=key[3], **s) for key, s in ok.items()])
    h = len(spec.report_seeds) // 2
    halves = []
    for part in (slice(0, h), slice(h, None)):
        st = [pair_stats({ph: {k: pr[k][part] for k in ("A", "P")} for ph, pr in r["report"].items()}, z,
                         spec.testable_min) for r in rows if r["axis"] == "b"]
        halves.append(dict(seeds=list(spec.report_seeds[part]), testable_b=sum(bool(s and s["testable"]) for s in st),
                           undefined=sum(s is None for s in st)))
    return dict(naive_floor=floor, single_type=single, report_halves=halves)


# ================================================================ selection (H.4 step 3)
def select(aggs: dict, spec) -> dict:
    """aggs: {combo: aggregate, or None for a combination without an oracle}, in spec.combos order. Eligible = F_a >= 2;
    the best T_b; every eligible combination within tie_pairs testable (b) pairs of the best is 'near', and the first
    of them in spec.combos (C0 > C1 > C3) is selected; no eligible combination or a best T_b < 0.5 -> stop. The stop looks
    at the best only, so the selected combination can itself be below 0.5 (H.4a.3-5, user decision): it is flagged."""
    elig = {c: a for c, a in aggs.items() if a is not None and a["F_a"] >= spec.f_a_min}
    if not elig:
        return dict(outcome=STOP_NO_ELIGIBLE, winner=None, eligible=[], near=[])
    top = max(a["testable_b"] for a in elig.values())
    top_t_b = max(a["T_b"] for a in elig.values())
    eligible = [c for c in spec.combos if c in elig]
    if top_t_b < spec.t_b_min:
        return dict(outcome=STOP_LOW_T_B, winner=None, eligible=eligible, near=[], top_T_b=top_t_b)
    near = [c for c in eligible if top - elig[c]["testable_b"] <= spec.tie_pairs]
    return dict(outcome=SELECTED, winner=near[0], eligible=eligible, near=near, top_T_b=top_t_b,
                engine_unchanged=bool(near[0] == "C0"), winner_below_bar=bool(elig[near[0]]["T_b"] < spec.t_b_min))
