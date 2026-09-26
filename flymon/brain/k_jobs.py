"""FlyPool worker jobs of spec appendix K (K.8.2): per-KC read-window counts for a batch of (odour, seed) items on the
naive brain (plasticity off, weights reset — H.4a.7's ceiling_job presentation through h4_jobs._present_kc), and one
raster with the mean membrane of probe KCs.

Signature fn(engine, plasticity, pops, comps, readout, **kwargs), module-level so the spawn pool can pickle them. The
rig is built from the Params given (h4_jobs.rig_for), so a job carries its g; the worker's engine only lends its
connectome."""
from __future__ import annotations

import numpy as np

from .h4_jobs import _present_kc, rig_for
from .stimuli import present


def activity_job(eng, pl, pops, comps, ro, params, items, strength: float, settle_ms: float, read_ms: float,
                 window_ms: int) -> list:
    e, p, _ = rig_for(eng.conn, pops, params)
    out = []
    try:
        p.reset_weights()
        p.set_enabled(False)
        for i, odor, seed in items:
            o = _present_kc(e, p, pops, odor, int(seed), strength, settle_ms, read_ms, int(window_ms))
            nz = np.flatnonzero(o["read"])
            out.append(dict(i=int(i), seed=int(seed), kc=nz.tolist(), n=o["read"][nz].astype(int).tolist(),
                            max_win=int(o["max_win"])))
    finally:
        p.reset_weights()
        p.set_enabled(True)
    return out


def raster_job(eng, pl, pops, comps, ro, params, odor, seed: int, strength: float, settle_ms: float, read_ms: float,
               probe) -> dict:
    """KC spikes [(step, KC position)] over settle + read and the mean membrane of `probe` (global indices) per step."""
    e, p, _ = rig_for(eng.conn, pops, params)
    kc = np.asarray(pops.kc, np.int64)
    pos = np.full(e.N, -1, np.int64); pos[kc] = np.arange(len(kc))
    probe = np.asarray(probe, np.int64)
    spikes, v_mean = [], []
    try:
        p.reset_weights(); p.set_enabled(False)
        e.reset(int(seed)); p.reset_traces(); e.clear_drive(); p.quiet_dan()
        present(e, pops, odor, strength)
        for step in range(int(round((settle_ms + read_ms) / e.p.dt))):
            f = pos[e.step()]; f = f[f >= 0]
            spikes += [(step, int(k)) for k in f]
            v_mean.append(float(e.v[probe].mean()) if probe.size else 0.0)
    finally:
        p.reset_weights(); p.set_enabled(True)
    return dict(spikes=spikes, v_mean=v_mean)
