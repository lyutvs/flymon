"""Spec AC.5's unpaired two-stage percentile bootstrap for a difference between two arms. Each draw resamples each
arm's flies with replacement, independently of the other arm, then each drawn fly's units with replacement; the
statistic is the mean over the drawn flies of their unit means, A minus B. Arms may differ in size (12 vs 6, 12 vs
16). Flies are put in a canonical order by content (units sorted inside a fly, flies sorted), so relabelling fly ids
changes nothing. rescope/stats.py's paired_boot / m4_verdict assume paired flies of equal number and are not used."""
from __future__ import annotations

import numpy as np


def fly_units(arm) -> list:
    flies = list(arm.values()) if isinstance(arm, dict) else list(arm)
    out = []
    for u in flies:
        a = np.sort(np.asarray(u, float))
        if a.size == 0:
            raise ValueError("a fly with no units cannot enter the bootstrap (exclude it before)")
        out.append(a)
    return sorted(out, key=lambda a: (a.size, tuple(a)))


def arm_mean(arm) -> float:
    return float(np.mean([u.mean() for u in fly_units(arm)]))


def _draw(rng, flies) -> float:
    pick = rng.integers(0, len(flies), len(flies))
    return float(np.mean([flies[i][rng.integers(0, flies[i].size, flies[i].size)].mean() for i in pick]))


def two_stage_diff(arm_a, arm_b, draws: int, seed: int) -> dict:
    fa, fb = fly_units(arm_a), fly_units(arm_b)
    rng = np.random.default_rng(int(seed))
    d = np.empty(int(draws))
    for b in range(int(draws)):
        d[b] = _draw(rng, fa) - _draw(rng, fb)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(diff=arm_mean(fa) - arm_mean(fb), lo=float(lo), hi=float(hi), n_a=len(fa), n_b=len(fb),
                draws=int(draws), seed=int(seed))


def meets(res: dict, min_effect: float) -> bool:
    return res["diff"] >= min_effect and res["lo"] > 0
