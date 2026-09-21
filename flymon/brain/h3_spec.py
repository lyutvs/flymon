"""The M0d H.3 runner's one configuration object (spec appendix H.3a.3-H.3a.8, with D.4 for the runaway clause).

Spec H.3a.11: the runner reads the H.3a.3 measurement table as a single configuration object, and no other code
restates its values. Every number the runner, its rules and its records use is a field of `SPEC` (or of a
`dataclasses.replace` of it, for smoke runs and tests). The odour generators are ported from the committed
diagnostics (`apl_input_scale_sweep.reference_odors`, 8b51594; `clamp_control_power.extended_odors`, c2a0f32) with
the same random-number calls in the same order; tests/brain/test_h3_spec.py pins their output by digest.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from .circuits import Populations
from .stimuli import channel_strengths


@dataclass(frozen=True)
class OdorSet:
    """`n` random mixtures of k in [k_min, k_max] glomeruli drawn from the receptor types minus `exclude`,
    strengths proportional to 1/receptor count (`channel_strengths`); odour j is probed at seeds
    (probe0 + 2j, probe0 + 1 + 2j). `n_candidates`, when set, is asserted (the 51 assignable glomeruli)."""
    name_fmt: str
    n: int
    rng_seed: int
    probe0: int
    k_min: int = 6
    k_max: int = 9
    exclude: tuple = ("ORN_DA1", "ORN_V")
    n_candidates: int | None = 51


@dataclass(frozen=True)
class Window:
    settle_ms: float
    read_ms: float


@dataclass(frozen=True)
class H3Spec:
    # ---- the data the table is defined on (a summary is written only for this connectome) --------------------
    connectome_sha256: str = "ff5ffa885d9e0d2e31258fab6b5a613e602f644f6f9e1257c8ba1b49e4247d88"   # data/malecns.npz
    # ---- H.3a.3: the measurement table -------------------------------------------------------------------
    strength: float = 0.35
    reference: OdorSet = OdorSet("R{j:02d}", 48, 800_000, 800_100)       # 48 odours x 2 seeds = 96 presentations
    reference_window: Window = Window(800.0, 600.0)                       # read 600 steps at dt = 1 ms
    design_k: int = 8                                                     # design_odor_pair(k=8, seed=0)
    design_odor_seed: int = 0
    design_seeds: tuple = tuple(range(100, 108))
    design_extra_seeds: tuple = tuple(range(108, 116))                   # only when the overlap CI holds 0
    design_window: Window = Window(200.0, 600.0)                          # kc_sparsity's defaults
    boot_draws: int = 10_000
    boot_seed: int = 20260918
    baseline_ms: float = 3000.0
    baseline_sat_hz: float = 100.0
    baseline_cal_seeds: tuple = tuple(range(100, 116))
    baseline_gate_seeds: tuple = tuple(range(116, 148))
    extended: OdorSet = OdorSet("X{j:03d}", 384, 800_001, 810_100)       # gain-control record (768 presentations)
    all51_seeds: tuple = tuple(range(200, 208))
    all51_window: Window = Window(800.0, 600.0)
    # ---- D.4: the runaway clause (H.3a.5 step 4 keeps it as defined there) --------------------------------
    runaway_rest_seeds: tuple = tuple(range(100, 108))
    runaway_rest_sat_hz: float = 100.0
    runaway_odor_seeds: tuple = tuple(range(100, 164))
    runaway_odor_window: Window = Window(800.0, 600.0)
    runaway_odor_sat_hz: float = 150.0
    runaway_odor_record_hz: float = 100.0         # D.4: KCs above 100 Hz are counted and identified, not judged
    # ---- H.3a.4: the three qualifications and the membrane recheck ----------------------------------------
    d4_band: tuple = (0.03, 0.07)
    overlap_sd: float = 0.00349                   # pooled seed SD of (chance - J); the ranking's overlap unit
    punish_type: str = "PPL105"                   # pool A = PPL105 core MBON types
    reward_type: str = "PAM08"                    # pool P = PAM08 core MBON types
    guard_med_delta_min: float = 5.0
    guard_zero_share_max: float = 0.25
    guard_half: int = 24                          # the single-type reproduction splits odours 0-23 / 24-47
    membrane_target_mv: float = 11.0
    membrane_tol_mv: float = 1.0
    # ---- H.3a.5: search and baseline -----------------------------------------------------------------------
    apl_r_max: float = 0.333
    kc_grid: tuple = (1.55, 1.60, 1.65, 1.70)
    scale_bracket: tuple = (0.005, 1.0)
    scale_bisect_steps: int = 8
    hold_bracket: tuple = (0.5, 1.0)
    hold_bisect_steps: int = 8
    baseline_target_hz: float = 3.5
    baseline_band_hz: tuple = (3.0, 4.0)
    baseline_ci_z: float = 1.96
    max_candidates: int = 4
    # ---- H.3a.6: C3 ----------------------------------------------------------------------------------------
    homeo_eta: float = 0.1
    homeo_target: float = 0.062                   # A0 when C1 adopts nothing; otherwise A0 follows C1 (H.3a.6 as amended)
    homeo_target_digits: int = 3                  # A0 = C1's adopted reference KC median (fraction) rounded to 3 digits
    homeo_clip: tuple = (0.25, 4.0)               # multiples of the rule threshold
    homeo_boundary_share: float = 0.10
    homeo_median_tol: float = 0.0104
    homeo_dtheta_q: float = 95.0
    homeo_dtheta_max: float = 0.05
    homeo_max_iter: int = 40
    c3_max_cycles: int = 3
    c3_stall_mv: float = 0.2
    c3_stall_pp: float = 0.2
    # ---- H.3a.8: records -----------------------------------------------------------------------------------
    release_quasi_linear: tuple = (0.269, 0.731)  # fraction of apl_r_max
    release_match_tol: float = 0.02
    clamp_slope: float = 1e9
    half_split_mv: float = 4.70
    half_split_pp: float = 2.03
    apl_v_quantiles: tuple = (5.0, 25.0, 50.0, 75.0, 95.0)
    callout_types: tuple = ("MBON05", "MBON13")
    ai_clip_mark: float = 0.49


SPEC = H3Spec()


def make_odors(pops: Populations, oset: OdorSet) -> list[dict]:
    """The generator rule of the reference set (and, with another OdorSet, of the extended set)."""
    cand = [t for t in pops.receptor_types if t not in oset.exclude]            # dict order = sorted names
    if oset.n_candidates is not None and len(cand) != oset.n_candidates:
        raise ValueError(f"{len(cand)} candidate glomeruli, the odour set declares {oset.n_candidates}")
    rng = np.random.default_rng(oset.rng_seed)
    odors = []
    for j in range(oset.n):
        k = int(rng.integers(oset.k_min, oset.k_max + 1))
        types = sorted(str(t) for t in rng.choice(cand, size=k, replace=False))
        odors.append(dict(name=oset.name_fmt.format(j=j), types=types, strengths=channel_strengths(pops, types),
                          seeds=[oset.probe0 + 2 * j, oset.probe0 + 1 + 2 * j]))
    if len({tuple(o["types"]) for o in odors}) != oset.n:
        raise ValueError("the odour type-sets are not distinct")
    return odors


def odor_digest(odors: list[dict]) -> str:
    rows = [dict(name=o["name"], types=list(o["types"]), strengths={k: float(v) for k, v in o["strengths"].items()},
                 seeds=[int(s) for s in o["seeds"]]) for o in odors]
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def all51_glomeruli(pops: Populations, exclude=("ORN_DA1", "ORN_V")) -> tuple[list[str], float]:
    """The all51 stimuli: every assignable glomerulus alone at strength c_norm / receptor count, c_norm the median
    receptor count (apl_cap_disparity.py)."""
    rts = sorted(t for t in pops.receptor_types if t not in exclude)
    return rts, float(np.median([len(pops.receptor_types[t]) for t in rts]))
