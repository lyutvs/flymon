"""A one-arm version of AC.5's two-stage percentile bootstrap (flies with replacement, then each drawn fly's units with
replacement): the paired OS evaluation's within-fly differences (AD.3). Same canonical fly order as boot.two_stage_diff."""
from __future__ import annotations

import numpy as np

from ..ac import boot


def two_stage_mean(units: dict, draws: int, seed: int) -> dict:
    flies = boot.fly_units(units)
    rng = np.random.default_rng(int(seed))
    d = np.array([boot._draw(rng, flies) for _ in range(int(draws))])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(mean=boot.arm_mean(flies), lo=float(lo), hi=float(hi), n=len(flies), draws=int(draws), seed=int(seed))
