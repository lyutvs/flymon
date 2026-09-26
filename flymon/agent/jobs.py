"""FlyPool worker jobs for M3 (signature fn(eng, pl, pops, comps, ro, **kw), module-level for the spawn pool).
Each leaves the worker's weights reset."""
from __future__ import annotations

import numpy as np

from ..brain.presentation import reinforce


def nonplastic_invariance_job(eng, pl, pops, comps, ro, odor, dan, pulse_ms, seed, strength) -> dict:
    """Spec 4.3 safety constraint 1: run one reinforcement and count changed CSC weights inside and outside the
    plastic KC->MBON edges."""
    pl.reset_weights()
    w_before = eng.csc.w.copy()
    reinforce(eng, pl, pops, odor, strength, dan, pulse_ms, seed, enabled=True)
    changed = eng.csc.w != w_before
    plastic = np.zeros(changed.shape, bool); plastic[pl.edges] = True
    out = {"plastic_changed": int(changed[plastic].sum()), "nonplastic_changed": int(changed[~plastic].sum())}
    pl.reset_weights(); pl.set_enabled(True)
    return out


def edge_compartments_job(eng, pl, pops, comps, ro, types) -> dict:
    """{DAN type: bool mask over pl.edges} — the plastic edges whose MBON is in that type's core compartment."""
    tgt = eng.csc.tgt[pl.edges]
    return {t: np.isin(tgt, comps[t].core) for t in types}
