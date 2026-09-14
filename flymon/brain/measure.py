"""Measurements used by the M0 gate: Kenyon-cell sparsity/overlap and MBON baseline."""
from __future__ import annotations

import numpy as np

from .circuits import Populations
from .engine_cpu import Engine
from .stimuli import present


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def chance_jaccard(pa: float, pb: float) -> float:
    return float(pa * pb / (pa + pb - pa * pb)) if (pa + pb - pa * pb) > 0 else 0.0


def kc_sparsity(engine: Engine, pops: Populations, odor, strength: float, seed: int,
                settle_ms: float = 200.0, read_ms: float = 600.0) -> dict:
    engine.reset(seed)
    engine.clear_drive()
    present(engine, pops, odor, strength)
    engine.run(settle_ms)
    counts = engine.run(read_ms)
    active = counts[pops.kc] > 0
    sec = read_ms / 1000.0
    return {
        "frac_active": float(active.mean()),
        "active": active,
        "kc_hz": float(counts[pops.kc].mean() / sec),
        "mbon_hz": float(counts[pops.mbon].mean() / sec),
    }


def mbon_baseline(engine: Engine, pops: Populations, seed: int, ms: float = 1000.0) -> dict:
    engine.reset(seed)
    engine.clear_drive()
    counts = engine.run(ms)
    hz = counts[pops.mbon] / (ms / 1000.0)
    types = engine.conn.type[pops.mbon]
    return {"mbon_hz": float(hz.mean()), "n_types_active": int(len(set(types[hz > 0].tolist())))}
