"""Dopamine-gated depression of KC->MBON synapses (three-factor rule), compartment by DAN type."""
from __future__ import annotations

import numpy as np

from .circuits import Populations
from .engine_cpu import Engine


class Plasticity:
    def __init__(self, engine: Engine, pops: Populations, comps: dict):
        self.eng, self.pops, self.p = engine, pops, engine.p
        csc = engine.csc
        pre = csc.pre_of_edge()
        is_kc = np.zeros(engine.N, bool); is_kc[pops.kc] = True
        is_mbon = np.zeros(engine.N, bool); is_mbon[pops.mbon] = True
        self.edges = np.flatnonzero(is_kc[pre] & is_mbon[csc.tgt])
        self.w0 = csc.w[self.edges].copy()
        kc_local = np.full(engine.N, -1, np.int64); kc_local[pops.kc] = np.arange(len(pops.kc))
        mb_local = np.full(engine.N, -1, np.int64); mb_local[pops.mbon] = np.arange(len(pops.mbon))
        if self.edges.size == 0:
            raise ValueError("no plastic KC->MBON edges: check min_weight and the KC/MBON populations")
        if (self.w0 <= 0).any():
            raise ValueError("non-positive KC->MBON weight: Kenyon cells are cholinergic, so a "
                             "w0 <= 0 means a sign/transmitter problem in the connectome")
        self.pre_kc = kc_local[pre[self.edges]]
        self.post_mb = mb_local[csc.tgt[self.edges]]
        self.mb_local = mb_local
        # per DAN type: cells and MBON weight vector restricted to the core compartment
        self.types = {}
        for name, cp in comps.items():
            w = np.zeros(len(pops.mbon), np.float32)
            w[mb_local[cp.core]] = cp.w_mbon[cp.core]
            self.types[name] = (cp.cells.astype(np.int64), w)
        self.kc_trace = np.zeros(len(pops.kc), np.float32)
        self.da = np.zeros(len(pops.mbon), np.float32)
        self.da_base = np.zeros(len(pops.mbon), np.float32)
        self.enabled = True
        engine.on_step = self.on_step

    # ---- dopamine drive ------------------------------------------------------------------
    def drive_dan(self, type_name: str, mv: float) -> None:
        cells, _ = self.types[type_name]
        self.eng.set_ext(cells, mv)

    def quiet_dan(self) -> None:
        for cells, _ in self.types.values():
            self.eng.ext[cells] = self.eng.ext0[cells]

    # ---- rule ----------------------------------------------------------------------------
    def on_step(self, engine: Engine, fired: np.ndarray) -> None:
        p, dt = self.p, self.p.dt
        self.kc_trace *= (1.0 - dt / p.kc_trace_ms)
        kf = self.pops.kc
        fired_kc = fired[np.isin(fired, kf)]
        if fired_kc.size:
            self.kc_trace[np.searchsorted(kf, fired_kc)] += dt / p.kc_trace_ms
        self.da *= (1.0 - dt / p.da_trace_ms)
        for cells, wvec in self.types.values():
            n = int(np.isin(fired, cells).sum())
            if n:
                self.da += wvec * (n / len(cells)) * (dt / p.da_trace_ms)
        self.da_base += (self.da - self.da_base) * (dt / p.da_baseline_ms)
        if not self.enabled:
            return
        phasic = np.maximum(self.da - self.da_base, 0.0)
        coincide = (self.kc_trace[self.pre_kc] * p.kc_trace_scale) * (phasic[self.post_mb] * p.da_trace_scale)
        if coincide.any():
            w = self.eng.csc.w
            # w[self.edges] is a fancy-index copy: depress and floor it, then write the block back
            we = w[self.edges] * (1.0 - p.learn_rate * np.tanh(coincide)).astype(np.float32)
            np.maximum(we, self.w0 * p.min_weight_frac, out=we)
            w[self.edges] = we

    # ---- bookkeeping ---------------------------------------------------------------------
    def weights_frac(self) -> float:
        if self.edges.size == 0:
            raise ValueError("no plastic edges selected")
        return float(np.mean(self.eng.csc.w[self.edges] / self.w0))

    def weights_frac_by_mbon_set(self, mbon_idx) -> float:
        sel = np.isin(self.post_mb, self.mb_local[np.asarray(mbon_idx)])
        if not sel.any():
            raise ValueError("no plastic edges selected")
        return float(np.mean(self.eng.csc.w[self.edges][sel] / self.w0[sel]))

    def recover_pulse(self) -> None:
        r = self.p.recovery_per_pulse
        if r <= 0:
            return
        w = self.eng.csc.w
        w[self.edges] += (self.w0 - w[self.edges]) * np.float32(r)

    def reset_traces(self) -> None:
        """Zero the eligibility/dopamine traces (state carries no meaning across presentations)."""
        self.kc_trace[:] = 0; self.da[:] = 0; self.da_base[:] = 0

    def reset_weights(self) -> None:
        self.eng.csc.w[self.edges] = self.w0
        self.reset_traces()

    def set_enabled(self, on: bool) -> None:
        self.enabled = bool(on)
