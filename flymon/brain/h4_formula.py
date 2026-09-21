"""G.14's oracle arithmetic for M0d H.4, with the readout's z constants passed in (spec H.4 step 2: "판정 산식
pair_stats·arm_aggregate는 바꾸지 않는다").

This is `docs/superpowers/specs/m2-calibration-g/engine_probe_verdict.py` (sha256 c9c81ab9…) — `dprime`, `dv`,
`pair_stats`, `arm_aggregate` — except that `dv` takes z = {"A": (mean, sd), "P": (mean, sd)} instead of reading F.3's
frozen constants, and the thresholds come from the configuration object. tests/brain/test_h4_formula.py loads the frozen
module and checks every function against it with F.3's constants. The oracle job imports `dprime` and `dv` (its alpha
choice), so this module is part of the measurement key; the decision rules live in h4_rules, outside it.
"""
from __future__ import annotations

import numpy as np


def dprime(x) -> float | None:
    """mean / sd (ddof 1). sd = 0 is a limit: 0 if the mean is 0, else +-inf. Fewer than 2 seeds: undefined."""
    x = np.asarray(x, float)
    if x.size < 2:
        return None
    sd, m = float(x.std(ddof=1)), float(x.mean())
    if sd == 0:
        return 0.0 if m == 0 else float(np.copysign(np.inf, m))
    return m / sd


def dv(probe: dict, z: dict) -> np.ndarray:
    """V(X) - V(Y) per probe seed, V = z_A - z_P."""
    A = np.asarray(probe["A"], float); P = np.asarray(probe["P"], float)
    if A.ndim != 2 or A.shape[1] != 2 or A.shape != P.shape:
        raise ValueError(f"probe must be [seeds x 2] for A and P, got {A.shape} / {P.shape}")
    V = (A - z["A"][0]) / z["A"][1] - (P - z["P"][0]) / z["P"][1]
    return V[:, 0] - V[:, 1]


def pair_stats(report: dict, z: dict, testable_min: float) -> dict | None:
    """report = {"pre", "R1", "R2"} on the same seeds: d_pre = d'(dV_pre), r = d'(dV_R1 - dV_pre),
    p = d'(dV_R2 - dV_R1), m = min(r, -p), testable <=> m >= testable_min. None if any d' is undefined."""
    pre, r1, r2 = dv(report["pre"], z), dv(report["R1"], z), dv(report["R2"], z)
    if not (len(pre) == len(r1) == len(r2)):
        raise ValueError("pre / R1 / R2 must share the probe seeds")
    d_pre, r, p = dprime(pre), dprime(r1 - pre), dprime(r2 - r1)
    if d_pre is None or r is None or p is None:
        return None
    m = min(r, -p)
    return {"d_pre": d_pre, "r": r, "p": p, "m": m, "testable": bool(m >= testable_min)}


def arm_aggregate(stats: dict, naive_max: float, t_b_min: float, f_a_min: int) -> dict:
    """stats: {(axis, turn, x, y): pair_stats}. T_b = testable share on axis (b); F_a = axis-(a) pairs testable with
    |d_pre| < naive_max; bar = T_b >= t_b_min and F_a >= f_a_min."""
    b = [s for (ax, *_), s in stats.items() if ax == "b"]
    a = [s for (ax, *_), s in stats.items() if ax == "a"]
    naive_ok = lambda s: abs(s["d_pre"]) < naive_max
    t_b = sum(s["testable"] for s in b) / len(b)
    f_a = sum(s["testable"] and naive_ok(s) for s in a)
    return {"n_a": len(a), "n_b": len(b), "T_b": t_b, "F_a": f_a,
            "testable_a": sum(s["testable"] for s in a), "testable_b": sum(s["testable"] for s in b),
            "naive_a": sum(naive_ok(s) for s in a), "naive_b": sum(naive_ok(s) for s in b),
            "bar": bool(t_b >= t_b_min and f_a >= f_a_min)}
