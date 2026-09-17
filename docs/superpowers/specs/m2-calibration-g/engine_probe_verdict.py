"""Engine-premise probe verdict (spec G.14) — the authoritative rule is this file, not the prose.

Pure functions only: raw oracle rows in, a verdict dict out. No I/O, no simulation.

Per pair and arm (report seeds 608-615, alphas picked on 600-607):
  d_pre = d'(dV_pre)               naive separation
  r     = d'(dV_R1 - dV_pre)       reward change      (G.12 used the R1 level; P0-1)
  p     = d'(dV_R2 - dV_R1)        punishment change
  m     = min(r, -p)               testable  <=>  m >= 2   (unclipped)
Per arm:
  T_b = testable fraction on axis (b);  F_a = axis-(a) pairs testable with |d_pre| < 0.5
Paired contrast on axis (b): delta_i = clip(m_G) - clip(m_S-match), clustered by turn.
"""
from __future__ import annotations

import numpy as np

Z = {"A": (21.8293, 18.1032), "P": (40.6433, 24.0891)}   # F.3 frozen z constants (MBON13 / MBON05)
ARMS = ("G", "S-match", "S-ref")
TEST_ARM, CONTRAST_ARM = "G", "S-match"
TESTABLE = 2.0
NAIVE = 0.5
BAR_T_B = 0.5
BAR_F_A = 2
CLIP = 10.0
MATCH_TOL_PP = 0.5
SIGNFLIP_ALPHA = 0.025
N_BOOT = 10_000
BOOT_SEED = 20260917
TESTS = ("bootstrap", "signflip")
OUTCOMES = ("INVALID", "GO_M0D", "PROMISING", "SPARSITY", "NO_EFFECT")


# ---- per pair ------------------------------------------------------------------------------------------------
def dprime(x) -> float | None:
    """mean / sd (ddof 1). sd = 0 is a limit: 0 if the mean is 0, else +-inf. Fewer than 2 seeds: undefined."""
    x = np.asarray(x, float)
    if x.size < 2:
        return None
    sd, m = float(x.std(ddof=1)), float(x.mean())
    if sd == 0:
        return 0.0 if m == 0 else float(np.copysign(np.inf, m))
    return m / sd


def dv(probe: dict) -> np.ndarray:
    """V(X) - V(Y) per probe seed, V = z(MBON13) - z(MBON05)."""
    A = np.asarray(probe["A"], float); P = np.asarray(probe["P"], float)
    if A.ndim != 2 or A.shape[1] != 2 or A.shape != P.shape:
        raise ValueError(f"probe must be [seeds x 2] for A and P, got {A.shape} / {P.shape}")
    V = (A - Z["A"][0]) / Z["A"][1] - (P - Z["P"][0]) / Z["P"][1]
    return V[:, 0] - V[:, 1]


def reward_stat(pre: np.ndarray, r1: np.ndarray) -> float | None:
    return dprime(r1 - pre)


def pair_stats(report: dict) -> dict | None:
    """report = {"pre": probe, "R1": probe, "R2": probe} on the same seeds. None if any d' is undefined."""
    pre, r1, r2 = dv(report["pre"]), dv(report["R1"]), dv(report["R2"])
    if not (len(pre) == len(r1) == len(r2)):
        raise ValueError("pre / R1 / R2 must share the probe seeds")
    d_pre, r, p = dprime(pre), reward_stat(pre, r1), dprime(r2 - r1)
    if d_pre is None or r is None or p is None:
        return None
    m = min(r, -p)
    return {"d_pre": d_pre, "r": r, "p": p, "m": m, "testable": bool(m >= TESTABLE)}


# ---- aggregation ---------------------------------------------------------------------------------------------
def clip(x: float) -> float:
    return float(np.clip(x, -CLIP, CLIP))


def naive_ok(d_pre: float) -> bool:
    return abs(d_pre) < NAIVE


def turn_ok(turn: int) -> bool:
    return turn % 2 == 0          # G.2: odd turns are the F v4 gate pool and never enter a calibration verdict


def arm_aggregate(stats: dict) -> dict:
    """stats: {(axis, turn, x, y): pair_stats}."""
    b = [s for (ax, *_), s in stats.items() if ax == "b"]
    a = [s for (ax, *_), s in stats.items() if ax == "a"]
    t_b = sum(s["testable"] for s in b) / len(b)
    f_a = sum(s["testable"] and naive_ok(s["d_pre"]) for s in a)
    return {"n_a": len(a), "n_b": len(b), "T_b": t_b, "F_a": f_a,
            "testable_a": sum(s["testable"] for s in a), "testable_b": sum(s["testable"] for s in b),
            "naive_a": sum(naive_ok(s["d_pre"]) for s in a), "naive_b": sum(naive_ok(s["d_pre"]) for s in b),
            "bar": bool(t_b >= BAR_T_B and f_a >= BAR_F_A)}


def paired_deltas(stats_test: dict, stats_contrast: dict) -> tuple[list[int], list[float]]:
    keys = sorted(k for k in stats_test if k[0] == "b")
    return [k[1] for k in keys], [clip(stats_test[k]["m"]) - clip(stats_contrast[k]["m"]) for k in keys]


def _turn_sums(turns, deltas):
    ut = sorted(set(turns))
    sums = np.array([sum(d for t, d in zip(turns, deltas) if t == u) for u in ut], float)
    counts = np.array([sum(1 for t in turns if t == u) for u in ut], float)
    return sums, counts


def turn_bootstrap(turns, deltas, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """Resample turns with replacement; pair-weighted mean of the resampled pairs; 95% percentile CI."""
    sums, counts = _turn_sums(turns, deltas)
    idx = np.random.default_rng(seed).integers(0, len(sums), size=(n_boot, len(sums)))
    means = sums[idx].sum(1) / counts[idx].sum(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(sums.sum() / counts.sum()), "lo": float(lo), "hi": float(hi), "improve": bool(lo > 0)}


def turn_signflip(turns, deltas) -> dict:
    """Exact one-sided test: flip the sign of every turn's deltas jointly (2^T patterns, identity included)."""
    sums, counts = _turn_sums(turns, deltas)
    T = len(sums)
    signs = ((np.arange(2 ** T)[:, None] >> np.arange(T)) & 1) * 2 - 1
    stats = (signs * sums).sum(1) / counts.sum()
    obs = float(sums.sum() / counts.sum())
    p = float((stats >= obs - 1e-12 * max(1.0, abs(obs))).mean())
    return {"mean": obs, "p": p, "improve": bool(p < SIGNFLIP_ALPHA)}


# ---- verdict -------------------------------------------------------------------------------------------------
def _invalid(reasons: list[str], **extra) -> dict:
    return {"outcome": "INVALID", "reasons": reasons, **extra}


def verdict_from_stats(stats_by_arm: dict, expected: list, match: dict, test: str) -> dict:
    """stats_by_arm: {arm: {(axis, turn, x, y): pair_stats or None}}; expected: declared pair keys."""
    if test not in TESTS:
        raise ValueError(f"test must be one of {TESTS}")
    reasons = []
    expected = sorted(tuple(k) for k in expected)
    if not expected or not any(k[0] == "a" for k in expected) or not any(k[0] == "b" for k in expected):
        reasons.append("expected pair list must hold both axes")
    bad = [k for k in expected if not turn_ok(k[1])]
    if bad:
        reasons.append(f"odd turn in the expected pairs: {bad[:3]}")
    for arm in ARMS:
        got = stats_by_arm.get(arm)
        if got is None:
            reasons.append(f"arm {arm} missing"); continue
        keys = sorted(tuple(k) for k in got)
        if [k for k in keys if not turn_ok(k[1])]:
            reasons.append(f"arm {arm}: odd-turn rows present")
        if set(keys) != set(expected):
            reasons.append(f"arm {arm}: pairs differ from the declared list "
                           f"(missing {len(set(expected) - set(keys))}, extra {len(set(keys) - set(expected))})")
        undefined = [k for k, s in got.items() if s is None]
        if undefined:
            reasons.append(f"arm {arm}: {len(undefined)} pairs with an undefined d'")
        nan = [k for k, s in got.items() if s is not None and any(np.isnan(s[f]) for f in ("d_pre", "r", "p", "m"))]
        if nan:
            reasons.append(f"arm {arm}: {len(nan)} pairs with a NaN statistic")
    diff = match.get("diff_pp")
    if diff is None or not np.isfinite(diff) or abs(diff) > MATCH_TOL_PP:
        reasons.append(f"activity match failed: diff {diff} pp (tolerance {MATCH_TOL_PP})")
    if reasons:
        return _invalid(reasons)

    agg = {arm: arm_aggregate({tuple(k): s for k, s in stats_by_arm[arm].items()}) for arm in ARMS}
    turns, deltas = paired_deltas({tuple(k): s for k, s in stats_by_arm[TEST_ARM].items()},
                                  {tuple(k): s for k, s in stats_by_arm[CONTRAST_ARM].items()})
    contrast = turn_bootstrap(turns, deltas) if test == "bootstrap" else turn_signflip(turns, deltas)
    if not all(np.isfinite(v) for f, v in contrast.items() if f != "improve"):
        return _invalid([f"non-finite contrast statistic: {contrast}"], arms=agg)   # never read NaN as "no effect"
    improve = contrast["improve"]
    if improve and agg[TEST_ARM]["bar"]:
        outcome = "GO_M0D"
    elif improve:
        outcome = "PROMISING"
    elif agg[TEST_ARM]["bar"] or agg[CONTRAST_ARM]["bar"]:
        outcome = "SPARSITY"
    else:
        outcome = "NO_EFFECT"
    return {"outcome": outcome, "reasons": [], "test": test, "contrast": contrast, "arms": agg}


def verdict(raw_by_arm: dict, expected: list, match: dict, test: str) -> dict:
    """raw_by_arm: {arm: [row]}, row = {"axis", "turn", "x", "y", "report": {"pre", "R1", "R2"}}."""
    dup = [arm for arm, rows in raw_by_arm.items()
           if len({(r["axis"], r["turn"], r["x"], r["y"]) for r in rows}) != len(rows)]
    if dup:
        return _invalid([f"arm {arm}: duplicate pair rows" for arm in dup])
    stats = {arm: {(r["axis"], r["turn"], r["x"], r["y"]): pair_stats(r["report"]) for r in rows}
             for arm, rows in raw_by_arm.items()}
    return verdict_from_stats(stats, expected, match, test)
