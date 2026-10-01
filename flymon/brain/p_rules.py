"""The pure rules of spec appendix P as amended by P.6 (P.6 wins over P.1-P.4). No engine, no pool.
- The gate (P.3, P.6.5): o_rules.validity (O.7.5: complete, unique keys, finite, declared seeds, one CSC sha) plus every
  row's declared expectations (direction pair, operating point, arm flags) and a positive finite c1; then arm ②'s
  bit-identity (sha256 of the plastic weights == w0's). Any failure: the whole judgement INVALID.
- The judgement (P.6.2): per direction r, per seed, over the punish (③) and plastic (①) arms
    dl = -(Δ(dV)[③] - Δ(dV)[①]),  ds = ΔV_X[③] - ΔV_X[①],  dt = ΔV_Y[③] - ΔV_Y[①]   (dl = dt - ds),
  one seed bootstrap shared by both directions (the same seed numbers resampled together, P.3), the per-direction
  outcome (confirmed / excluded / undecided) and the label in P.6.2's order.
- The records (P.3, P.6.2, P.6.3): the averaged ℓ, raw-unit changes, the non-associative change, the flip re-analysis
  (no suffix), the pre-state strata, the state-conditional numbers, dopamine integrals and core weight fractions.
- The operating characteristic (P.6.4) from O2's per-seed values.
The presentation state is n_rules.state, Δ(dV) is n_rules.delta; every threshold is a field of the PSpec passed in."""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from .n_rules import INVALID, NO_LEARNING, _ci, delta, state
from .o_rules import INCONCLUSIVE, JUDGED, boot_weights, finite, validity

LEARNS_CONFIRMATORY = "LEARNS_CONFIRMATORY"
DIRECTION_DEPENDENT = "DIRECTION_DEPENDENT"
LABELS = (INVALID, LEARNS_CONFIRMATORY, DIRECTION_DEPENDENT, NO_LEARNING, INCONCLUSIVE)   # P.6.2's evaluation order
CONFIRMED = "LEARNING_CONFIRMED"                  # per direction: (i)-(iii) all hold
EXCLUDED = "LEARNING_EXCLUDED"                    # per direction: CI upper of ℓ_r < c1
UNDECIDED = "UNDECIDED"
OC_RECORDED = "RECORDED"
FLIP_NOTE = ("P.6.3: 판정은 무조건부 분석만으로 한다 — 뒤집힌 시드를 뺀 재분석은 접미사 없는 기록이다(①·③은 사전 프로브가 같아 "
             "뒤집힘은 처치 뒤 변수다)")


# ================================================================ the gate (P.3, P.6.5)
def p_key(r) -> tuple:
    return (r["direction"], r["arm"], int(r["seed"]))


def p_declared(spec) -> set:
    return {(d, a, int(s)) for d in spec.directions for a in spec.arms for s in spec.seeds}


def expectations(rows, spec) -> list:
    """P.6.5: every row ran its direction's (X, Y), its arm's (punish, plastic, da_zero) flags and the declared point.
    The reasons, [] when every row matches."""
    pairs, flags, point = spec.pairs(), spec.flags(), [float(v) for v in spec.o.o2_point]
    wrong = defaultdict(list)
    for r in rows:
        k = p_key(r)
        if r["direction"] in pairs and (r.get("x"), r.get("y")) != pairs[r["direction"]]:
            wrong["ran another (X, Y) than its direction declares"].append(k)
        got = tuple(bool(r.get(f)) for f in ("punish", "plastic", "da_zero"))
        if r["arm"] in flags and got != tuple(flags[r["arm"]]):
            wrong["ran other arm flags than its arm declares"].append(k)
        if [float(v) for v in r.get("point", ())] != point:
            wrong[f"ran another operating point than {point}"].append(k)
    return [f"{len(ks)} row(s) {why}, e.g. {sorted(ks, key=str)[:3]}" for why, ks in sorted(wrong.items())]


def group(rows, spec) -> dict:
    """{direction: {arm: rows in seed order}}."""
    by = {d: {a: [] for a in spec.arms} for d in spec.directions}
    for r in rows:
        by[r["direction"]][r["arm"]].append(r)
    for d in by:
        for a in by[d]:
            by[d][a].sort(key=lambda r: int(r["seed"]))
    return by


def plumbing(by: dict, spec) -> dict:
    """P.3: arm ② (plasticity off) ends with the plastic weights bit-identical to w0 (sha256), every seed. {direction:
    reasons}, only directions with a reason."""
    two = spec.arms[1]
    out = {}
    for d in spec.directions:
        bad = [f"plumbing: direction {d} arm {two} seed {r['seed']} moved its weights (sha256 "
               f"{r['w_post_sha256'][:12]} != w0 {r['w0_sha256'][:12]})" for r in by[d][two]
               if r["w_post_sha256"] != r["w0_sha256"]]
        if bad:
            out[d] = bad
    return out


# ================================================================ the judgement (P.6.2)
def v_of(probe: dict, z: dict) -> float:
    """V = z_A - z_P of one probe, C3's frozen z (h4_formula.dv is V_X - V_Y of the same: test)."""
    return float((probe["A"] - z["A"][0]) / z["A"][1] - (probe["P"] - z["P"][0]) / z["P"][1])


def _dv(rows, side: str, z: dict) -> np.ndarray:
    return np.array([v_of(r["post"][side], z) - v_of(r["pre"][side], z) for r in rows], float)


def arm_vectors(by: dict, z: dict, spec) -> dict:
    """One direction's per-seed vectors over arms ① and ③ (module docstring). Both arms must hold the same seeds in
    the same order (ValueError otherwise: a pairing defect is never judged)."""
    one, _, three = spec.arms
    seeds = [int(r["seed"]) for r in by[one]]
    if seeds != [int(r["seed"]) for r in by[three]]:
        raise ValueError("arms 1 and 3 must hold the same seeds in the same order")
    ds = _dv(by[three], "x", z) - _dv(by[one], "x", z)
    dt = _dv(by[three], "y", z) - _dv(by[one], "y", z)
    dl = -(np.array([delta(r, z) for r in by[three]], float) - np.array([delta(r, z) for r in by[one]], float))
    return dict(seeds=seeds, dl=dl, ds=ds, dt=dt)


def direction_stats(dl, ds, dt, W, c1: float, spec) -> dict:
    """P.6.2 for one direction: (i) CI lower of ℓ_r > 0 and ℓ_r >= c1; (ii) CI upper of s_r < 0; (iii) |t_r| <=
    concentration_frac |s_r| (point estimates). Confirmed iff (i)-(iii), else excluded iff CI upper of ℓ_r < c1, else
    undecided (reading 7: confirmed is tested first)."""
    lv = spec.o.ci_level
    dl, ds, dt = (np.asarray(v, float) for v in (dl, ds, dt))
    ell, s, t = float(dl.mean()), float(ds.mean()), float(dt.mean())
    ell_ci, s_ci, t_ci = _ci(W @ dl, lv), _ci(W @ ds, lv), _ci(W @ dt, lv)
    i = bool(ell_ci[0] > 0 and ell >= c1)
    ii = bool(s_ci[1] < 0)
    iii = bool(abs(t) <= spec.concentration_frac * abs(s))
    if i and ii and iii:
        out = CONFIRMED
    elif ell_ci[1] < c1:
        out = EXCLUDED
    else:
        out = UNDECIDED
    return dict(n=int(dl.size), ell=ell, ell_ci=ell_ci, s=s, s_ci=s_ci, t=t, t_ci=t_ci, i=i, ii=ii, iii=iii,
                outcome=out)


def label(outcomes: dict) -> str:
    """P.6.2's order after INVALID: every direction confirmed; one confirmed and the rest excluded; every direction
    excluded; anything else."""
    v = set(outcomes.values())
    if v == {CONFIRMED}:
        return LEARNS_CONFIRMATORY
    if v == {CONFIRMED, EXCLUDED}:
        return DIRECTION_DEPENDENT
    if v == {EXCLUDED}:
        return NO_LEARNING
    return INCONCLUSIVE


def judge_vectors(vec: dict, c1: float, spec, W=None) -> dict:
    """The label from per-direction vectors {direction: {seeds, dl, ds, dt}}. One resampling matrix W for every
    direction (P.3: the same seed numbers resampled together), so the directions must hold the same seeds (ValueError).
    The averaged ℓ and its CI are a record (P.6.2)."""
    if len({tuple(v["seeds"]) for v in vec.values()}) != 1:
        raise ValueError("the directions must hold the same seed numbers in the same order")
    n = len(next(iter(vec.values()))["seeds"])
    if W is None:
        W = boot_weights(n, spec.o.boot_draws, spec.o.boot_seed)
    dirs = {d: direction_stats(v["dl"], v["ds"], v["dt"], W, c1, spec) for d, v in vec.items()}
    mean_dl = np.mean([np.asarray(vec[d]["dl"], float) for d in vec], axis=0)
    return dict(label=label({d: s["outcome"] for d, s in dirs.items()}), directions=dirs,
                mean_ell=dict(ell=float(mean_dl.mean()), ci=_ci(W @ mean_dl, spec.o.ci_level), record=True))


# ================================================================ the stage (P.6.2, P.6.5)
def p_judge(rows: list, z: dict, c1, spec) -> dict:
    """The gate (INVALID), arm ②'s plumbing (any direction: INVALID), then the judgement (Task 3 adds the records)."""
    bad = validity(rows, p_declared(spec), p_key, spec.seeds, lambda r: spec.o.on_edit) + expectations(rows, spec)
    if not (finite(c1) and float(c1) > 0):
        bad.append(f"c1 {c1!r} is not a positive finite number")
    if bad:
        return dict(outcome=INVALID, label=INVALID, reasons=bad, n_rows=len(rows))
    by = group(rows, spec)
    pl = plumbing(by, spec)
    if pl:
        return dict(outcome=INVALID, label=INVALID, reasons=[m for d in pl for m in pl[d]],
                    invalid_directions=sorted(pl), n_rows=len(rows))
    vec = {d: arm_vectors(by[d], z, spec) for d in spec.directions}
    main = judge_vectors(vec, float(c1), spec)
    return dict(outcome=JUDGED, label=main["label"], reasons=[], c1=float(c1), pairs=spec.pairs(),
                directions=main["directions"], mean_ell=main["mean_ell"], seeds={d: v["seeds"] for d, v in vec.items()})
