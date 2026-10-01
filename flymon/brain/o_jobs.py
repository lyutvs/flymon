"""FlyPool worker jobs of spec appendix O (O.7.1, O.7.3). Signature fn(engine, plasticity, pops, comps, readout,
**kwargs), module-level so the spawn pool can pickle them; the worker's default engine only lends its connectome.

O's rig is cached per worker under (Params, edit), edit in EDITS = n_jobs.EDITS + ("apl_to_nonkc_zero",):
- N's three edits are made by n_jobs.apply_edit, unchanged;
- "apl_to_nonkc_zero" (O.7.1): every APL out-edge whose target is not a KC set to 0 (APL -> MBON05 included). With
  "apl_to_kc_zero" it partitions "apl_all_zero"'s edge set (test).
The edit is made in place on a freshly built engine's CSC (the graded-APL out-edge views follow). The cache is O's own:
n_jobs._RIG is never touched, and no N module changes (O.5).

presentation_job: O1's presentations, N0f's procedure (n_jobs._present: reset, clear_drive, present, settle, read,
plasticity off), scalars only."""
from __future__ import annotations

import hashlib

import numpy as np

from . import n_jobs
from .circuits import compartments
from .engine_cpu import Engine
from .h3_jobs import edge_sources
from .plasticity import Plasticity

NONKC = "apl_to_nonkc_zero"
EDITS = n_jobs.EDITS + (NONKC,)
_RIG: dict = {}          # (Params, edit) -> (Engine, Plasticity, comps, csc sha256); one entry per worker


def apply_edit(eng, pops, edit: str) -> str:
    """N's edits through n_jobs.apply_edit; NONKC zeroes every APL -> non-KC edge in place. The sha256 of csc.w.
    Adapted from n_jobs:apply_edit (copied pattern, O.5 forbids editing N)."""
    if edit != NONKC:
        return n_jobs.apply_edit(eng, pops, edit)
    is_apl = np.zeros(eng.N, bool); is_apl[np.asarray(pops.apl, np.int64)] = True
    is_kc = np.zeros(eng.N, bool); is_kc[np.asarray(pops.kc, np.int64)] = True
    m = is_apl[edge_sources(eng.csc)] & ~is_kc[eng.csc.tgt.astype(np.int64)]
    eng.csc.w[m] = np.float32(0.0)
    return hashlib.sha256(eng.csc.w.tobytes()).hexdigest()


def rig(conn, pops, params, edit: str):
    """The worker's rig for (params, edit), built once and reused while jobs keep asking for it (one entry: a switch
    rebuilds, it never edits a cached engine).
    Copied from n_jobs:rig (O's own cache; O.5 forbids editing N)."""
    key = (params, edit)
    if key not in _RIG:
        _RIG.clear()
        eng = Engine(conn, pops, params, seed=0)
        sha = apply_edit(eng, pops, edit)
        comps = compartments(conn, pops, params.core_frac)
        _RIG[key] = (eng, Plasticity(eng, pops, comps), comps, sha)
    return _RIG[key]


def presentation_job(eng, pl, pops, comps, ro, params, edit: str, odor: dict, seeds, readout: dict, strength: float,
                     settle_ms: float, read_ms: float, window_ms: int) -> list:
    """O1's presentations on O's rig: n_jobs._present per seed, plasticity off, weights at w0; scalars only.
    Copied from n_jobs:presentation_job (O.5 forbids editing N)."""
    e, p, _, sha = rig(eng.conn, pops, params, edit)
    a_cells, p_cells = n_jobs.readout_cells(e.conn, readout)
    p.reset_weights()
    was = p.enabled
    p.set_enabled(False)
    try:
        return [dict(n_jobs._present(e, p, pops, odor, s, strength, settle_ms, read_ms, window_ms, a_cells,
                                     p_cells)[0], edit=edit, csc_sha256=sha) for s in seeds]
    finally:
        p.set_enabled(was)
