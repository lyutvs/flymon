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
    kc_thresh: float = 1.5         # multiplier on KC threshold after PN-input normalisation
                                   # (frozen at 1.5: 1.0 leaves KC overlap above chance)
    kc_norm_clip: tuple = (0.5, 3.0)
    balance_hemispheres: bool = True
    kc_kc_scale: float = 0.0       # multiplier on KC->KC edge weights; 0 removes them from the CSC.
                                   # KC axo-axonic contacts act through mAChR-B and suppress neighbouring
                                   # KCs (Manoim et al. 2022), so fast excitation is the wrong sign; 0 is
                                   # the first-order stand-in. 1.0 reproduces the M0/M0b engine (spec D).

    # --- M0d engine modes (spec appendix H.2); every default keeps the M0c engine bit-identical ---
    apl_mode: str = "spiking"      # "graded": APL never spikes and releases apl_release(v) per step (Amin et al. 2020)
    apl_r_max: float = 0.333       # graded release at saturation, spike equivalents per step (spiking cap 333 Hz)
    apl_v_mid: float = 11.0        # sigmoid midpoint, mV above rest (GGN model, Ray et al. 2020: -40 mV mid, -51 mV rest)
    apl_slope: float = 5.0         # sigmoid slope, mV
    orn_std: bool = False          # presynaptic depression of receptor out-edges (Nagel et al. 2015)
    orn_std_f: float = 0.78        # resource kept per spike
    orn_std_tau_ms: float = 893.0  # recovery time constant
    kc_thresh_mode: str = "pn_norm"   # "homeostatic": per-KC thresholds from kc_thresh_file (spec H.3)
    kc_thresh_file: str = ""
    kc_thresh_sha256: str = ""

    # --- plasticity (spec 3.1, tuned in Task 9 within these defaults) ---
    learn_rate: float = 3e-4
    kc_trace_ms: float = 200.0
    da_trace_ms: float = 100.0
    da_baseline_ms: float = 200.0   # short: the baseline must track the odour-evoked DAN level
                                   # within a presentation so only phasic dopamine teaches
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
