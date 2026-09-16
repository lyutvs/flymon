"""Verdict for the M2 learning unit test, appendix F v3. Pure functions over raw counts; no engine.

Written and exercised BEFORE the prose is frozen: v1 and v2 each failed on contact with data that
already existed (v2's protocol check would have stopped a working protocol), so the rules live here
first and the appendix quotes them.

Spec 5, read sequentially on one fly: naive d' ~ 0 -> 20 rewards on X -> d' >= 1 -> 20 punishments on
X -> d' drops. Every fly also runs a paired no-DAN brain with the SAME training seeds, so the
non-associative part (M0c: an odour that got no DAN still lost ~25 spikes of MBON05) is subtracted per
probe seed rather than tested against an absolute threshold.

Data per fly, per probe seed, per candidate (X taught, Y untaught), cells A = MBON13, P = MBON05:
  pre  R1 (after 20 reward)  R2 (after 20 reward + 20 punish)  N1 (20 no-DAN)  N2 (40 no-DAN)
Counts are dicts {"A": [[x, y] per seed], "P": [[x, y] per seed]}.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Z = {"A": (21.8293, 18.1032), "P": (40.6433, 24.0891)}    # frozen exploration-set constants (F.3)
BAR = 1.0            # spec 5: d' >= 1
NAIVE_MAX = 0.5      # "~ 0" = below half the learning bar, on the POOLED naive estimate
BAND = 0.2           # short of the bar by <= BAND (inclusive) -> undecided, extend probe seeds once
MIN_VALID = 0.75     # fraction of flies whose statistic is defined (>= 2 probe seeds)


def dv(counts: dict) -> np.ndarray:
    """dV = V(X) - V(Y) per probe seed, V = z_A - z_P."""
    A = np.asarray(counts["A"], float); P = np.asarray(counts["P"], float)
    V = (A - Z["A"][0]) / Z["A"][1] - (P - Z["P"][0]) / Z["P"][1]
    return V[:, 0] - V[:, 1]


def dprime(x: np.ndarray) -> float | None:
    """Spec 3.1 d': mean / sample sd (ddof = 1).

    sd = 0 is a limit, not a hole: an effect that is exactly 0 on every probe seed is the strongest
    evidence of NO effect, so it is 0.0 (an earlier draft returned None here and turned "no learning"
    into UNDEFINED instead of FAIL). A non-zero effect with sd = 0 is +/-inf. Only fewer than two
    probe seeds is undefined."""
    x = np.asarray(x, float)
    if x.size < 2:
        return None
    sd = float(x.std(ddof=1)); m = float(x.mean())
    if sd == 0.0:
        return 0.0 if m == 0.0 else float(np.copysign(np.inf, m))
    return m / sd


@dataclass
class FlyStats:
    reward_level: float | None       # C2  d'(dV_R1)                          >= +1
    punish_drop: float | None        # C3  d'(dV_R2 - dV_R1)                  <= -1
    reward_assoc: float | None       # C4  d'(dV_R1 - dV_N1)                  >= +1
    punish_assoc: float | None       # C5  d'((dV_R2-dV_R1) - (dV_N2-dV_N1))  <= -1
    nodan_1: float | None            # descriptive: d'(dV_N1 - dV_pre)
    nodan_2: float | None            # descriptive: d'(dV_N2 - dV_N1)
    floor: dict = field(default_factory=dict)   # descriptive: fraction of probe seeds at <= 1 spike


def fly_stats(pre: dict, R1: dict, R2: dict, N1: dict, N2: dict) -> FlyStats:
    p, r1, r2, n1, n2 = dv(pre), dv(R1), dv(R2), dv(N1), dv(N2)

    def at_floor(c, cell, cand):
        return float((np.asarray(c[cell], float)[:, cand] <= 1).mean())

    return FlyStats(
        reward_level=dprime(r1), punish_drop=dprime(r2 - r1),
        reward_assoc=dprime(r1 - n1), punish_assoc=dprime((r2 - r1) - (n2 - n1)),
        nodan_1=dprime(n1 - p), nodan_2=dprime(n2 - n1),
        floor={"R1_P_X": at_floor(R1, "P", 0), "R1_P_Y": at_floor(R1, "P", 1),
               "R2_A_X": at_floor(R2, "A", 0), "R2_A_Y": at_floor(R2, "A", 1)},
    )


GATES = (("reward_level", +1), ("punish_drop", -1), ("reward_assoc", +1), ("punish_assoc", -1))


def _median_defined(vals, n_flies):
    ok = [v for v in vals if v is not None]
    if len(ok) < MIN_VALID * n_flies:
        return None, len(ok)
    return float(np.median(ok)), len(ok)


def pair_verdict(pre_pooled: list, flies: list) -> dict:
    """pre_pooled: every fly's pre counts (the naive brain is shared, so they pool). flies: [FlyStats].

    Returns {"state": PASS | FAIL | NAIVE_REGRESSED | UNDEFINED | BAND, ...}."""
    out = {"n_flies": len(flies)}
    d0 = dprime(np.concatenate([dv(c) for c in pre_pooled]))
    out["naive_pooled"] = d0
    if d0 is None or abs(d0) >= NAIVE_MAX:
        out["state"] = "NAIVE_REGRESSED"
        return out
    gates = GATES
    med, fail, band = {}, [], []
    for name, sign in gates:
        m, n_ok = _median_defined([getattr(f, name) for f in flies], len(flies))
        med[name] = m; out[f"n_valid_{name}"] = n_ok
        if m is None:
            out.update(medians=med, state="UNDEFINED", undefined=name)
            return out
        margin = round(sign * m - BAR, 9)   # >= 0 passes; rounded so 0.8 - 1.0 is exactly -0.2
        if margin < 0:
            (band if margin >= -BAND else fail).append(name)   # short by <= 0.2 -> band, by more -> fail
    out["medians"] = med
    out["state"] = "FAIL" if fail else ("BAND" if band else "PASS")
    out["failing"], out["in_band"] = fail, band
    return out


SIGN_MIN = 0.75      # protocol check: taught cell lower than the paired no-DAN brain on >= 6/8 probe seeds


def lower_fraction(dan: np.ndarray, nodan: np.ndarray) -> float:
    """Fraction of probe seeds where the DAN brain is lower than the paired no-DAN brain; ties count 1/2."""
    dan, nodan = np.asarray(dan, float), np.asarray(nodan, float)
    return float(((dan < nodan).sum() + 0.5 * (dan == nodan).sum()) / dan.size)


def control_fly(pre: dict, R1: dict, R2: dict, N1: dict, N2: dict) -> dict:
    """Protocol check (qualitative: does the pulse act on the taught cell of the taught odour?).
    reward: MBON05 of X after 20 rewards is below the paired no-DAN brain.
    punish: the change in MBON13 of X over the punish stage is below the no-DAN brain's change.
    The pair's naive state is bimodal by construction (A.4: MBON13 answers odour b only; MBON05 of b is
    near-silent on some seeds), so an effect-size bar on z-scored V is the wrong instrument here - on
    M0c raw data it reads 0.85 / 0.97 while MBON05 of b falls to 0 in 16 of 16 seeds."""
    PX = lambda c: np.asarray(c["P"], float)[:, 0]
    AX = lambda c: np.asarray(c["A"], float)[:, 0]
    return {"reward": lower_fraction(PX(R1), PX(N1)),
            "punish": lower_fraction(AX(R2) - AX(R1), AX(N2) - AX(N1))}


def control_verdict(flies: list) -> dict:
    med = {k: float(np.median([f[k] for f in flies])) for k in ("reward", "punish")}
    return {"state": "PASS" if all(v >= SIGN_MIN for v in med.values()) else "FAIL", "medians": med}


def noplast_ok(pre: dict, post: dict) -> bool:
    """Plasticity off: every raw count identical. A bit-level check, kept outside the d' state machine:
    the question is whether the machine is deterministic, not how large an effect is."""
    return all(np.array_equal(np.asarray(pre[c]), np.asarray(post[c])) for c in ("A", "P"))


def overall(noplast: dict, control: dict, gates: dict) -> dict:
    """noplast: {pair: bool}; control: control_verdict of the protocol-check pair; gates: {pair: pair_verdict}."""
    if not all(noplast.values()):
        return {"verdict": "STOP_MACHINE", "why": [p for p, ok in noplast.items() if not ok]}
    if control["state"] != "PASS":
        return {"verdict": "STOP_PROTOCOL", "why": control["state"]}
    evaluable = {p: v for p, v in gates.items() if v["state"] in ("PASS", "FAIL", "BAND")}
    if any(v["state"] == "FAIL" for v in evaluable.values()):
        return {"verdict": "FAIL", "failing_pairs": [p for p, v in evaluable.items() if v["state"] == "FAIL"]}
    if any(v["state"] == "BAND" for v in evaluable.values()):
        return {"verdict": "UNDECIDED", "why": "band", "pairs": [p for p, v in evaluable.items() if v["state"] == "BAND"]}
    if len(evaluable) >= 2:
        return {"verdict": "PASS", "pairs": list(evaluable)}
    return {"verdict": "UNDECIDED", "why": "fewer than 2 evaluable gate pairs",
            "states": {p: v["state"] for p, v in gates.items()}}
