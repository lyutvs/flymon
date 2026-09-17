"""Event-driven leaky integrate-and-fire engine over the MaleCNS connectome (reference CPU implementation).

Membrane (mV above rest):  v <- v + (-v + g + ext) * dt/tau_m + noise, while not refractory
Alpha synapse:             g <- g + W on arrival of a spike emitted syn_delay ago;  g <- g * (1 - dt/tau_syn)
Threshold:                 spike when v >= v_th; v <- v_reset; refractory for refrac_steps
Refractory:                a spike sets refrac = refrac_steps() and the cell is eligible again after
                           that many further steps; at dt = 1 ms that is 3 ms of wall time, one step
                           longer than the published 2.2 ms.
Membrane floor:            v is clamped at >= -v_thresh, so inhibition cannot drive a cell
                           arbitrarily far below rest (our design decision).
Receptors (sensory classes) ignore the membrane and fire as Poisson sources at drive_hz.
M0d modes (spec appendix H.2, all off by default):
  apl_mode "graded"          APL never spikes; each step it queues r = apl_r_max / (1 + exp(-(v - apl_v_mid) / apl_slope))
                             from its updated membrane, delivered through its out-edges after the synaptic delay.
  orn_std                    each receptor carries a resource R (1 at reset); a receptor spike delivers its out-edges
                             scaled by R, then R <- orn_std_f * R; every step R <- R + (1 - R) * dt / orn_std_tau_ms.
  kc_thresh_mode "homeostatic"  KC thresholds come from a validated file (flymon.brain.thresholds).
Reproduction target for the design decisions (MBON hold, KC threshold normalisation, APL scale):
flybrain FINDINGS.md; constants: Shiu et al. 2024.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .circuits import Populations
from .config import Params
from .connectome import Connectome, build_csc
from .thresholds import load_kc_thresholds

APL_MODES = ("spiking", "graded")
KC_THRESH_MODES = ("pn_norm", "homeostatic")


class Engine:
    def __init__(self, conn: Connectome, pops: Populations, params: Params, seed: int = 0):
        _validate_modes(params)
        self.p = params
        self.conn, self.pops = conn, pops
        self.N = conn.N
        self.csc = build_csc(conn, params, pops.apl, pops.kc)
        self.on_step = None

        # thresholds: KC thresholds normalised by their PN input (our design decision)
        self.v_th = np.full(self.N, params.v_thresh, np.float32)
        pn_in = np.zeros(self.N, np.float64)
        is_alpn = np.zeros(self.N, bool); is_alpn[pops.alpn] = True
        is_kc = np.zeros(self.N, bool); is_kc[pops.kc] = True
        m = is_alpn[conn.pre] & is_kc[conn.post]   # one boolean mask, not two np.isin scans
        np.add.at(pn_in, conn.post[m], conn.w[m])
        med = float(np.median(pn_in[pops.kc])) if len(pops.kc) else 1.0
        if med > 0:
            lo, hi = params.kc_norm_clip
            self.v_th[pops.kc] = params.v_thresh * params.kc_thresh * np.clip(pn_in[pops.kc] / med, lo, hi)
        self.kc_pn_input = pn_in[pops.kc]          # total PN->KC synapses per KC, in KC order
        if params.kc_thresh_mode == "homeostatic":
            self.v_th[pops.kc] = load_kc_thresholds(params.kc_thresh_file, params.kc_thresh_sha256,
                                                    conn.bodyId[pops.kc], self.v_th[pops.kc], self.kc_pn_input > 0)

        # tonic drive: MBON hold (our design decision); everything else 0 until set_ext
        self.ext0 = np.zeros(self.N, np.float32)
        self.ext0[pops.mbon] = params.mbon_hold_frac * params.v_thresh
        self.ext = self.ext0.copy()

        self.is_receptor = np.zeros(self.N, bool)
        self.is_receptor[pops.sensory] = True
        self.receptor_idx = pops.sensory.astype(np.int64)
        self.drive_hz = np.zeros(self.N, np.float32)

        self._graded_apl = params.apl_mode == "graded"
        self.apl_idx = np.asarray(pops.apl, np.int64)
        if self._graded_apl:                        # views into the CSC: APL out-edges are never plastic
            ptr, tgt, w = self.csc.ptr, self.csc.tgt, self.csc.w
            self._apl_edges = [(tgt[ptr[i]:ptr[i + 1]], w[ptr[i]:ptr[i + 1]]) for i in self.apl_idx]
            if any(np.unique(t).size != t.size for t, _ in self._apl_edges):   # g[tgt] += adds a repeated target once
                raise ValueError("graded APL needs unique out-edge targets per APL cell")
        self._orn_std = bool(params.orn_std)

        self._seed = seed
        self.reset(seed)

    # ---- state -------------------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._seed = seed
        self.rng = np.random.default_rng(self._seed)
        self.v = np.zeros(self.N, np.float32)
        self.g = np.zeros(self.N, np.float32)
        self.refrac = np.zeros(self.N, np.int32)
        # delay line: spikes emitted at step t are popped (delivered) at step t + dly_steps.
        # The deque holds dly_steps entries; step() pops one at the start and appends this step's
        # spikes at the end, so a 2-step delay means: fire on step 1 -> arrive on step 3.
        self.delay = deque([np.zeros(0, np.int64) for _ in range(self.p.dly_steps())], maxlen=self.p.dly_steps())
        self.last = np.zeros(0, np.int64)
        self.t_ms = 0.0
        if self._graded_apl:
            self._apl_release = deque([np.zeros(self.apl_idx.size, np.float32) for _ in range(self.p.dly_steps())],
                                      maxlen=self.p.dly_steps())
        if self._orn_std:
            self._std_r = np.ones(self.N, np.float32)          # stays 1 for every non-receptor
            self._std_delay = deque([np.zeros(0, np.float32) for _ in range(self.p.dly_steps())], maxlen=self.p.dly_steps())

    def set_ext(self, idx, mv: float) -> None:
        self.ext[np.asarray(idx, dtype=np.int64)] = np.float32(mv)

    def clear_ext(self) -> None:
        self.ext = self.ext0.copy()

    def set_drive_hz(self, idx, hz: float) -> None:
        self.drive_hz[np.asarray(idx, dtype=np.int64)] = np.float32(hz)

    def clear_drive(self) -> None:
        self.drive_hz[:] = 0.0

    # ---- dynamics ----------------------------------------------------------------------------
    def propagate(self, src: np.ndarray, gain: np.ndarray | None = None) -> np.ndarray:
        """Sum signed mV of every out-edge of the spiking sources into a dense [N] vector; `gain` scales each
        source's out-edges (ORN depression)."""
        out = np.zeros(self.N, np.float32)
        if src.size == 0:
            return out
        ptr, tgt, w = self.csc.ptr, self.csc.tgt, self.csc.w
        starts, ends = ptr[src], ptr[src + 1]
        idx = np.concatenate([tgt[a:b] for a, b in zip(starts, ends)])
        val = np.concatenate([w[a:b] for a, b in zip(starts, ends)])
        if gain is not None:
            val = val * np.repeat(np.asarray(gain, np.float32), ends - starts)
        # bincount, not np.add.at: the unbuffered ufunc path dominates per-step cost at 162k neurons
        return np.bincount(idx, weights=val, minlength=self.N).astype(np.float32)

    def step(self) -> np.ndarray:
        p = self.p
        arrived = self.delay.popleft()
        arrived_gain = self._std_delay.popleft() if self._orn_std else None
        if arrived.size:
            self.g += self.propagate(arrived, arrived_gain)
        if self._graded_apl:
            r_in = self._apl_release.popleft()
            for k in np.flatnonzero(r_in):
                tgt, w = self._apl_edges[k]
                self.g[tgt] += w * r_in[k]
        dt_m = p.dt / p.tau_m
        dv = (-self.v + self.g + self.ext) * dt_m
        if p.noise_mv:
            dv += self.rng.standard_normal(self.N, dtype=np.float32) * p.noise_mv
        free = self.refrac <= 0
        self.v = np.where(free, self.v + dv, self.v)
        self.refrac = np.where(free, self.refrac, self.refrac - 1)
        np.maximum(self.v, -p.v_thresh, out=self.v)
        self.g *= (1.0 - p.dt / p.tau_syn)
        spk = (self.v >= self.v_th) & free
        # receptors: Poisson at commanded rate, membrane ignored
        ri = self.receptor_idx
        hz = self.drive_hz[ri]
        pois = (self.rng.random(ri.size) < hz * (p.dt / 1000.0)) & free[ri]
        spk[ri] = pois
        if self._graded_apl:
            spk[self.apl_idx] = False     # non-spiking: no threshold, reset or refractory
        fired = np.flatnonzero(spk)
        self.v[fired] = p.v_reset
        self.refrac[fired] = p.refrac_steps()
        self.last = fired
        self.delay.append(fired)          # delivered dly_steps steps from now
        if self._orn_std:
            self._std_delay.append(self._std_r[fired].copy())
            rf = fired[self.is_receptor[fired]]
            self._std_r[rf] *= np.float32(p.orn_std_f)
            self._std_r[ri] += (1.0 - self._std_r[ri]) * np.float32(p.dt / p.orn_std_tau_ms)
        if self._graded_apl:
            self._apl_release.append(self.apl_release(self.v[self.apl_idx]))
        self.t_ms += p.dt
        if self.on_step is not None:
            self.on_step(self, fired)
        return fired

    def apl_release(self, v: np.ndarray) -> np.ndarray:
        """Graded APL release per step (spike equivalents) at membrane v, mV above rest."""
        p = self.p
        return (p.apl_r_max / (1.0 + np.exp(-(np.asarray(v, np.float32) - p.apl_v_mid) / p.apl_slope))).astype(np.float32)

    def run(self, ms: float, count_idx=None) -> np.ndarray:
        counts = np.zeros(self.N, np.int32)
        for _ in range(int(round(ms / self.p.dt))):
            counts[self.step()] += 1
        return counts


def _validate_modes(p: Params) -> None:
    if p.apl_mode not in APL_MODES:
        raise ValueError(f"apl_mode must be one of {APL_MODES}, got {p.apl_mode!r}")
    if p.kc_thresh_mode not in KC_THRESH_MODES:
        raise ValueError(f"kc_thresh_mode must be one of {KC_THRESH_MODES}, got {p.kc_thresh_mode!r}")
    if p.apl_mode == "graded" and not (p.apl_r_max > 0 and p.apl_slope > 0):
        raise ValueError(f"graded APL needs apl_r_max > 0 and apl_slope > 0, got {p.apl_r_max}, {p.apl_slope}")
    if p.orn_std and not (0 < p.orn_std_f <= 1 and p.orn_std_tau_ms > 0):
        raise ValueError(f"orn_std needs 0 < orn_std_f <= 1 and orn_std_tau_ms > 0, got {p.orn_std_f}, {p.orn_std_tau_ms}")
