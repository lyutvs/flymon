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


def mbon_baseline(engine: Engine, pops: Populations, seed: int, ms: float = 1000.0,
                  sat_hz: float = 100.0, counts: np.ndarray | None = None) -> dict:
    """Resting MBON rates.

    A self-sustaining cholinergic clique (FR1) saturates a few MBONs on the real
    connectome, so the raw mean is unstable; the gate uses the trimmed mean over
    cells at or below `sat_hz`.

    `counts` lets a caller that already ran the resting window (and wants the raw spike counts for
    its own statistics, e.g. the M0b runaway set) reuse this one definition of the convention
    instead of copying it: pass the per-neuron counts of a `ms`-long rest run and the engine is
    left untouched. When it is None the engine is reset to `seed` and run here, as usual.
    """
    if counts is None:
        engine.reset(seed)
        engine.clear_drive()
        counts = engine.run(ms)
    hz = counts[pops.mbon] / (ms / 1000.0)
    types = engine.conn.type[pops.mbon]
    keep = hz <= sat_hz
    trimmed = float(hz[keep].mean()) if keep.any() else 0.0
    return {
        "mbon_hz": float(hz.mean()),
        "mbon_hz_trimmed": trimmed,
        "n_saturated": int((~keep).sum()),
        "n_types_active": int(len(set(types[hz > 0].tolist()))),
    }


def mbon_baseline_multi(engine: Engine, pops: Populations, seeds, ms: float = 3000.0,
                        sat_hz: float = 100.0) -> dict:
    """`mbon_baseline` averaged over seeds; the gate statistic is `mbon_hz_rest_trimmed`."""
    seeds = [int(s) for s in seeds]
    runs = [mbon_baseline(engine, pops, seed=s, ms=ms, sat_hz=sat_hz) for s in seeds]
    trimmed = np.array([r["mbon_hz_trimmed"] for r in runs], float)
    return {
        "mbon_hz_rest": float(np.mean([r["mbon_hz"] for r in runs])),
        "mbon_hz_rest_trimmed": float(trimmed.mean()),
        "mbon_hz_rest_trimmed_sd": float(trimmed.std()),
        "mbon_hz_rest_trimmed_per_seed": [float(x) for x in trimmed],
        "mbon_n_saturated": float(np.mean([r["n_saturated"] for r in runs])),
        "mbon_types_active_rest": float(np.mean([r["n_types_active"] for r in runs])),
        "rest_seeds": seeds,
        "rest_ms": float(ms),
    }
