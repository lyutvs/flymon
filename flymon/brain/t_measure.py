"""T's only measurement file (T.1, T.8, T.9.4, T.9.6): the H.3 reference set and its same-seed rest on an edited
engine, for each engine variant's own z and the readout guard. The shared measurement files (r_measure.R_MEASURE_FILES)
are not touched; this file and h3_spec (the reference set's generator) form the T measurement key with them.
- t_ref_job: h3_jobs.reference_job's call sequence (reset / clear_drive / present / run(settle) / step x read) on an
  engine built as h3_jobs.engine_for builds it (Engine(conn, pops, params, seed=0), no plasticity) and then edited in
  place by q_jobs.apply_q_edit (the lever's 2 edges; "none" changes nothing), returning per presentation the summed
  read-window count of every listed MBON type (both hemispheres, h3_jobs.mbon_type_index), the KC activity, the CSC
  sha and the edited edge count. With edit "none" its counts are reference_job's (test), so the unedited z reproduces
  block h4's z bit for bit (T.1).
- t_rest_job: h3_jobs.rest_job's sequence on the same engine (no odour): the guard's same-seed resting counts.
- ZMeasurer: both jobs over a pool in h3_measure.chunks, no cache — the unedited reproduction runs without a cache
  (T.9.6) and the lever's z is minutes of work; the rows are written once under results/t/ by the runner."""
from __future__ import annotations

import numpy as np

from .engine_cpu import Engine
from .h3_jobs import mbon_type_index
from .h3_measure import chunks
from .h3_store import code_key
from .q_jobs import apply_q_edit
from .r_measure import R_MEASURE_FILES
from .stimuli import present

T_MEASURE_FILES = ("flymon/brain/t_measure.py", "flymon/brain/h3_spec.py")
_RIG: dict = {}          # (Params, edit, p_type) -> (Engine, type index, csc sha256, edges changed); one per worker


def t_measure_key(npz: str) -> dict:
    """T.8 / T.9.6: the shared measurement key's files plus T's measurement files."""
    return code_key(npz, files=tuple(dict.fromkeys(R_MEASURE_FILES + T_MEASURE_FILES)))


def z_engine(conn, pops, params, edit: str, p_type: str):
    key = (params, edit, p_type)
    if key not in _RIG:
        _RIG.clear()
        eng = Engine(conn, pops, params, seed=0)
        sha, n = apply_q_edit(eng, pops, edit, p_type)
        _RIG[key] = (eng, mbon_type_index(conn, pops), sha, n)
    return _RIG[key]


def _read(e, steps: int) -> np.ndarray:
    counts = np.zeros(e.N, np.int32)
    for _ in range(int(steps)):
        counts[e.step()] += 1
    return counts


def t_ref_job(eng, pl, pops, comps, ro, params, edit: str, p_type: str, types, odors, strength: float,
              settle_ms: float, read_steps: int) -> list:
    e, idx, sha, n = z_engine(eng.conn, pops, params, edit, p_type)
    out = []
    for o in odors:
        for seed in o["seeds"]:
            e.reset(int(seed)); e.clear_drive(); present(e, pops, o["strengths"], strength); e.run(settle_ms)
            counts = _read(e, read_steps)
            kc = counts[pops.kc]
            out.append(dict(odor=o["name"], seed=int(seed), types={t: int(counts[idx[t]].sum()) for t in types},
                            kc_active_frac=float((kc > 0).mean()), csc_sha256=sha, edit_edges=int(n)))
    return out


def t_rest_job(eng, pl, pops, comps, ro, params, edit: str, p_type: str, types, seeds, settle_ms: float,
               read_steps: int) -> list:
    e, idx, sha, n = z_engine(eng.conn, pops, params, edit, p_type)
    out = []
    for seed in seeds:
        e.reset(int(seed)); e.clear_drive(); e.run(settle_ms)
        counts = _read(e, read_steps)
        out.append(dict(seed=int(seed), types={t: int(counts[idx[t]].sum()) for t in types}, csc_sha256=sha,
                        edit_edges=int(n)))
    return out


class ZMeasurer:
    def __init__(self, pool, params, p_type: str, types):
        self.pool, self.params, self.p_type, self.types = pool, params, p_type, [str(t) for t in types]

    def _common(self, edit: str, settle_ms: float, read_steps: int) -> dict:
        return dict(params=self.params, edit=edit, p_type=self.p_type, types=list(self.types),
                    settle_ms=float(settle_ms), read_steps=int(read_steps))

    def reference(self, edit: str, odors: list, strength: float, settle_ms: float, read_steps: int) -> list:
        common = dict(self._common(edit, settle_ms, read_steps), strength=float(strength))
        parts = self.pool.run_jobs(t_ref_job, [dict(common, odors=c) for c in chunks(odors, self.pool.n_workers)])
        return [r for part in parts for r in part]

    def rest(self, edit: str, seeds: list, settle_ms: float, read_steps: int) -> list:
        common = self._common(edit, settle_ms, read_steps)
        seeds = [int(s) for s in seeds]
        parts = self.pool.run_jobs(t_rest_job, [dict(common, seeds=c) for c in chunks(seeds, self.pool.n_workers)])
        return [r for part in parts for r in part]
