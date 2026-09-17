"""Scratch prototype: a graded (non-spiking) APL, converted onto a live Engine inside a pool worker.

Amin et al. 2020 (eLife 56954): "Like its locust homolog, the giant GABAergic neuron (GGN), APL is
non-spiking" / "APL does not fire action potentials"; its output is graded GABA release. Our engine
runs APL as an ordinary LIF cell, so the 2.2 ms refractory period caps it at 333 Hz and it sits at
88-100% of that cap during a candidate presentation - a ceilinged inhibitor cannot do gain control.

Graded APL here:
  - APL is removed from the spiking path: no threshold, no reset, no refractory, so v integrates
    freely and rises above v_thresh under strong drive.
  - Each step APL emits r = gain * max(0, v)  (spikes-per-step equivalent; r = 0.3 matches the
    300 Hz the spiking APL reached), delivered through the same out-edges and the same synaptic
    delay as a spike would be.
This is an instance-level conversion, not a repo edit: nothing here touches flymon/.
"""
from __future__ import annotations

from collections import deque

import numpy as np


def _apl_slices(eng, apl_idx):
    ptr, tgt, w = eng.csc.ptr, eng.csc.tgt, eng.csc.w
    return [(tgt[ptr[i]:ptr[i + 1]].copy(), w[ptr[i]:ptr[i + 1]].astype(np.float32).copy()) for i in apl_idx]


def make_graded(eng, pops, gain: float, v_half: float = 0.0) -> None:
    """Idempotent: converts `eng` to a graded APL with the given gain (spikes-per-step per mV).

    v_half > 0 makes the release saturate:  r = gain * v_half * v / (v + v_half)  for v > 0,
    which is the finite-GABA-pool version; v_half = 0 is plain rectified-linear.
    """
    apl = np.asarray(pops.apl, np.int64)
    eng._apl_idx = apl
    eng._apl_gain = float(gain)
    eng._apl_vhalf = float(v_half)
    eng._apl_edges = _apl_slices(eng, apl)
    eng._apl_r_log = []
    if getattr(eng, "_graded", False):
        return
    eng._graded = True
    base_reset = eng.reset

    def reset(seed=None, _base=base_reset):
        _base(seed)
        eng._apl_delay = deque([np.zeros(len(eng._apl_idx), np.float32) for _ in range(eng.p.dly_steps())],
                               maxlen=eng.p.dly_steps())
        eng._apl_r_log = []
    eng.reset = reset

    def step():
        p = eng.p
        arrived = eng.delay.popleft()
        if arrived.size:
            eng.g += eng.propagate(arrived)
        r_in = eng._apl_delay.popleft()                      # graded APL release from dly_steps ago
        if r_in.any():
            for r, (tgt, w) in zip(r_in, eng._apl_edges):
                if r:
                    eng.g[tgt] += (w * r).astype(np.float32)
        dt_m = p.dt / p.tau_m
        dv = (-eng.v + eng.g + eng.ext) * dt_m
        if p.noise_mv:
            dv += eng.rng.standard_normal(eng.N, dtype=np.float32) * p.noise_mv
        free = eng.refrac <= 0
        eng.v = np.where(free, eng.v + dv, eng.v)
        eng.refrac = np.where(free, eng.refrac, eng.refrac - 1)
        np.maximum(eng.v, -p.v_thresh, out=eng.v)
        eng.g *= (1.0 - p.dt / p.tau_syn)
        spk = (eng.v >= eng.v_th) & free
        ri = eng.receptor_idx
        hz = eng.drive_hz[ri]
        spk[ri] = (eng.rng.random(ri.size) < hz * (p.dt / 1000.0)) & free[ri]
        spk[eng._apl_idx] = False                            # APL never spikes: no reset, no refractory
        fired = np.flatnonzero(spk)
        eng.v[fired] = p.v_reset
        eng.refrac[fired] = p.refrac_steps()
        eng.last = fired
        eng.delay.append(fired)
        v_apl = np.maximum(eng.v[eng._apl_idx], 0.0)
        r = eng._apl_gain * (v_apl if eng._apl_vhalf <= 0
                             else eng._apl_vhalf * v_apl / (v_apl + eng._apl_vhalf))
        eng._apl_delay.append(r.astype(np.float32))
        eng._apl_r_log.append(float(r.mean()))
        eng.t_ms += p.dt
        if eng.on_step is not None:
            eng.on_step(eng, fired)
        return fired
    eng.step = step
    eng.reset(eng._seed)
