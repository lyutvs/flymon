"""Decision and reinforcement presentations for one fly on one CPU engine (spec 2, steps 3 and 5).

decide: every candidate odour is presented from the same reset seed, so the candidates see identical
membrane noise and receptor spike trains (paired noise) and differ only through the odour; plasticity
is off and the weights are untouched. reinforce: one presentation of conditioning.train_block — settle
with the weights frozen so the dopamine baseline adapts to the odour-evoked DAN level, then the DAN
pulse, the gap, and one recovery step (spec 3.1). Whether a DAN is driven at all is the caller's
decision (spec 3.4 table; C-PAM / C-PPL arms simply never pass that channel).

Neither function restores a quiescent engine on exit: `decide` returns with the last candidate's drive
still set and `reinforce` leaves `pl.enabled` at the flag it was given. Every entry point starts with
`_fresh`, so callers must not assume the engine is idle between calls.
"""
from __future__ import annotations

import numpy as np

from .circuits import Populations
from .engine_cpu import Engine
from .plasticity import Plasticity
from .stimuli import present


def _fresh(engine: Engine, pl: Plasticity, pops: Populations, odor, strength: float, seed: int) -> None:
    engine.reset(seed)
    pl.reset_traces()
    engine.clear_drive()
    pl.quiet_dan()
    present(engine, pops, odor, strength)


def decide(engine: Engine, pl: Plasticity, pops: Populations, candidates, strength: float, seed: int,
           settle_ms: float = 800.0, read_ms: float = 600.0, idx=None) -> np.ndarray:
    """Spike counts of the read window for each candidate: [n_candidates, N] or [n_candidates, len(idx)]."""
    candidates = list(candidates)
    if not candidates:
        raise ValueError("decide needs at least one candidate odour")
    sel = None if idx is None else np.asarray(idx, np.int64)
    was = pl.enabled
    pl.set_enabled(False)
    try:
        out = []
        for odor in candidates:
            _fresh(engine, pl, pops, odor, strength, seed)
            engine.run(settle_ms)
            c = engine.run(read_ms)
            out.append(c if sel is None else c[sel])
    finally:
        pl.set_enabled(was)
    return np.stack(out)


def reinforce(engine: Engine, pl: Plasticity, pops: Populations, odor, strength: float, dan: str | None,
              pulse_ms: float, seed: int, settle_ms: float = 800.0, gap_ms: float = 200.0, enabled: bool = True) -> None:
    """One reinforcement presentation. `dan` is the DAN type to drive for `pulse_ms` (None = no signal;
    the odour is still presented so the call is uniform). Learning happens only if `enabled`."""
    if dan is not None and dan not in pl.types:
        raise ValueError(f"unknown DAN type {dan!r}; known: {sorted(pl.types)}")
    _fresh(engine, pl, pops, odor, strength, seed)
    pl.set_enabled(False)                       # settle: baseline adapts, weights frozen
    engine.run(settle_ms)
    try:
        pl.set_enabled(bool(enabled))
        if dan is not None:
            pl.drive_dan(dan, engine.p.dan_drive_mv)
        engine.run(pulse_ms)
    finally:                                    # a failing pulse never leaves a DAN driven or an odour on
        pl.quiet_dan()
        engine.clear_drive()
    engine.run(gap_ms)
    if dan is not None:
        pl.recover_pulse()
