"""The configuration of spec appendix K (K.8): fast KC->KC inhibition, g = kc_kc_scale < 0, scanned over a declared
grid; every setting re-converged by C3's rule (J.10.4 through `j`), ranked by the X-only MBON13 drive N on the odd-turn
(b) pairs, gated on N / N_C3 > 1.0, the top one judged by J's stage-2 path and J.12.9's bands.

`j` carries every J/H constant K reuses (C3's rule, first candidate, homeostasis target, the H.4 oracle, the bands);
nothing here repeats one of them.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from .j_spec import SPEC as J_SPEC, JSpec


@dataclass(frozen=True)
class KSpec:
    j: JSpec = J_SPEC
    grid: tuple = (-0.05, -0.1, -0.2, -0.4, -0.8)      # K.2 / K.8.2
    target_type: str = "MBON13"                         # the punishment readout N is measured on (K.8.3)
    reward_type: str = "MBON05"                         # the reward-side N05, recorded only
    min_weight: int = 5                                 # synapse floor of w13 / w05 / KC->KC edges (Params.min_weight)
    tie_tol: float = 0.01                               # relative tie on N (K.8.3)
    gate_ratio: float = 1.0                             # stage 2 only if N / N_C3 > this (K.8.4)
    raster_g: float = -0.8                              # the setting whose raster and membrane are recorded (K.8.2)
    odd_pairs_digest: str = ""                          # pinned in Task 2 (K.8.7 self-check iii)
    even_pair_uses: tuple = ("H.4 C0-C3", "H.4a.8 ceiling freq", "H.4a.8 ceiling all", "J.13")   # K.8.4 record


SPEC = KSpec()


def smoke(spec: KSpec) -> KSpec:
    """J's smoke configuration (scripts/run_j_std_judge.smoke_spec, copied: scripts are not importable) on one g."""
    r = dataclasses.replace
    h3 = r(spec.j.h4.h3, reference=r(spec.j.h4.h3.reference, n=4), all51_seeds=(200,), kc_grid=(1.65,),
           scale_bisect_steps=2, hold_bisect_steps=2, membrane_tol_mv=5.0, design_seeds=(100, 101),
           design_extra_seeds=(102, 103), baseline_cal_seeds=(100, 101, 102, 103), baseline_gate_seeds=(116, 117),
           runaway_rest_seeds=(100, 101), runaway_odor_seeds=(100, 101), boot_draws=200, homeo_max_iter=2,
           c3_max_cycles=1)
    h4 = r(spec.j.h4, h3=h3, teach_seeds=(8, 9), teach_min_decreased=1, act_seeds=(500, 501), select_seeds=(600, 601),
           report_seeds=(608, 609))
    return r(spec, j=r(spec.j, h4=h4, max_settings=1), grid=(-0.8,))
