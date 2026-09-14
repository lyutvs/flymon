"""Odours are defined over olfactory receptor TYPES (glomeruli), never over individual receptors."""
from __future__ import annotations

import numpy as np

from .circuits import Populations
from .engine_cpu import Engine

Odor = dict


def channel_strengths(pops: Populations, types, equalize: bool = True) -> Odor:
    types = list(types)
    if not equalize:
        return {t: 1.0 for t in types}
    inv = np.array([1.0 / len(pops.receptor_types[t]) for t in types])
    inv = inv / inv.mean()
    return {t: float(s) for t, s in zip(types, inv)}


def total_drive(pops: Populations, odor: Odor) -> float:
    return float(sum(s * len(pops.receptor_types[t]) for t, s in odor.items()))


def present(engine: Engine, pops: Populations, odor: Odor, strength: float) -> None:
    for t in pops.receptor_types:
        engine.drive_hz[pops.receptor_types[t]] = 0.0
    for t, s in odor.items():
        engine.drive_hz[pops.receptor_types[t]] = np.float32(engine.p.max_rate_hz * strength * s)


def design_odor_pair(pops: Populations, k: int = 8, exclude=("ORN_DA1", "ORN_V"), seed: int = 0):
    """Two disjoint k-glomerulus odours with matched total receptor drive.

    Sort candidate types by receptor count and take a 2k-wide band, then alternate A/B so the
    two odours interleave neighbouring receptor counts. Seed 0 takes the middle band; other
    seeds draw a random band start so repeated designs vary.
    """
    cand = [t for t in pops.receptor_types if t not in exclude]
    cand.sort(key=lambda t: len(pops.receptor_types[t]))
    n = len(cand)
    if n < 2 * k:
        raise ValueError(f"need {2 * k} receptor types, have {n}")
    if seed == 0:
        offset = (n - 2 * k) // 2
    else:
        offset = int(np.random.default_rng(seed).integers(0, n - 2 * k + 1))
    band = cand[offset:offset + 2 * k]
    a_types, b_types = band[0::2], band[1::2]
    return channel_strengths(pops, a_types), channel_strengths(pops, b_types)
