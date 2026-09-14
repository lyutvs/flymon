"""Model constants.

Shiu et al. 2024 (Nature 634:210) LIF constants are used at their published
values. Everything under "our design decisions" is a modelling choice made in
this project; reproduction target for those choices: flybrain FINDINGS.md.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    # --- Shiu et al. 2024, membrane and synapse (mV above rest, ms) ---
    v_thresh: float = 7.0          # -52 -> -45 mV
    v_reset: float = 0.0
    tau_m: float = 20.0
    tau_syn: float = 5.0
    syn_delay_ms: float = 1.8
    refractory_ms: float = 2.2
    mv_per_synapse: float = 0.275
    dt: float = 1.0
    noise_mv: float = 0.15         # per-step Gaussian membrane jitter
    max_rate_hz: float = 200.0     # receptor Poisson rate at strength 1.0

    # --- our design decisions (spec 3.1) ---
    min_weight: int = 5            # synapse-count threshold when building CSC
    sign_override: tuple = (("lLN1", -1), ("lLN2", -1))
    apl_scale: float = 0.1         # multiplier on APL out-edge weights
    mbon_hold_frac: float = 0.85   # tonic drive on MBONs as fraction of v_thresh
    kc_thresh: float = 1.0         # multiplier on KC threshold after PN-input normalisation
    kc_norm_clip: tuple = (0.5, 3.0)
    balance_hemispheres: bool = True

    # --- plasticity (spec 3.1, tuned in Task 9 within these defaults) ---
    learn_rate: float = 3e-4
    kc_trace_ms: float = 200.0
    da_trace_ms: float = 100.0
    da_baseline_ms: float = 1000.0
    kc_trace_scale: float = 40.0   # normalise a strongly driven KC trace to ~1
    da_trace_scale: float = 20.0   # normalise phasic dopamine at a core MBON to ~1
    min_weight_frac: float = 0.2
    core_frac: float = 0.2         # DAN->MBON weight >= this fraction of type peak = core
    dan_drive_mv: float = 70.0
    recovery_per_pulse: float = 0.0  # fraction of (w0 - w) restored after each pulse; 0 = off

    def dly_steps(self) -> int:
        return max(1, round(self.syn_delay_ms / self.dt))

    def refrac_steps(self) -> int:
        return max(1, round(self.refractory_ms / self.dt))
