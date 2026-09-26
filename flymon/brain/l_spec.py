"""The configuration of spec appendix L as amended by L.11: C3 unchanged, a naive-readout screen fitted on 41 labelled
(b) pairs (even 21 from H.4, odd 20 labelled in stage 0) behind a leave-one-turn-out gate, a new pair set screened to 21
passes, and the oracle reading in J.12.9's bands. `j` carries every J/H constant (H.4's oracle, seeds, formula, bands);
`k` carries K's readout weights. Nothing here repeats one of them."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from .j_spec import SPEC as J_SPEC, JSpec
from .k_spec import SPEC as K_SPEC, KSpec


@dataclass(frozen=True)
class LSpec:
    j: JSpec = J_SPEC
    k: KSpec = K_SPEC
    # ---- inputs pinned by L.11.1 ---------------------------------------------------------------------------------
    ceiling_path: str = "results/m0d/diag/h4_specificity_ceiling.json"
    ceiling_sha256: str = "81bfab20df8bb9ada380a14106bead35039bc7a8d7910d1975cd5245a78585e3"
    k_act_path: str = "results/m0d/k/run/cache/k_act/f16f19e4602bf075d866cecf.json"
    k_act_sha256: str = "93d86ee0685c06837e93a539ec8483e275c67ba0bb5baa67298df39f7d835fcf"
    m0d_path: str = "results/summary/m0d.json"
    readout: tuple = (("A", "MBON13"), ("P", "MBON05"))   # H.4's C3 readout (reading 1); dict(spec.readout) to use
    # ---- stage 1 (L.11.2) -----------------------------------------------------------------------------------------
    guard_types: tuple = ("MBON13", "MBON05")
    guard_min: int = 5
    family_order: tuple = ("G", "S", "f2", "f3")
    directions: tuple = (("S", ">"), ("f2", ">"), ("f3", "<"))   # dict(spec.directions) to use (frozen, hashable)
    cov_min_stage1: float = 0.35
    gate_min_passes: int = 10
    gate_min_precision: float = 0.6
    # ---- stage 2 (L.4 / L.11.3) ------------------------------------------------------------------------------------
    rng_seed: int = 20260927
    n_turns: int = 64
    a_turns: int = 8                      # F_a's fixed sample: the (a) pairs of the first 8 turns (decision 9)
    n_pass: int = 21
    max_screened: int = 84                # 21 / 84 = the 0.25 coverage floor
    lift_seed: int = 20260928             # 20260927 + 1
    n_lift: int = 10
    b_digest: str = "8017c9703848f86ca5c425de805550d4c2d076453dc4e472be5d554336af3dff"  # 160 (b) pairs, skipped 6 (old_set)
    a_digest: str = "47edaced4980982034f8422c36ff1f41571a46db27e403f1877155e992836c1f"  # 20 (a) pairs (turns 0-7)
    # ---- operating characteristics (L.11.4) --------------------------------------------------------------------------
    oc_q: tuple = (0.4, 0.5, 0.6, 0.7)
    oc_c: tuple = (0.25, 0.3, 0.35, 0.45)
    oc_q_fail: tuple = (0.0, 0.1)         # label probability of a G-failing pair in the gate simulation (reading 14)
    oc_naive_a: tuple = tuple(range(2, 9))
    oc_ratio: tuple = (0.5, 1.0)
    oc_draws: int = 2000
    oc_seed: int = 20260929
    attempts: tuple = (("I", "M2 no-go", "dc9d8b2"), ("J", "STD B_Tb 4/21", "0917b1a"),
                       ("K", "fast KC-KC inhibition B_Tb 4/21", "73bfc2a"), ("L", "this declaration", "f89d4a3"))


SPEC = LSpec()


def smoke(spec: LSpec) -> LSpec:
    """Fewer turns, passes and seeds; the rule constants stay the declared ones."""
    r = dataclasses.replace
    h4 = r(spec.j.h4, act_seeds=(500, 501), select_seeds=(600, 601), report_seeds=(608, 609))
    return r(spec, j=r(spec.j, h4=h4), n_turns=4, a_turns=1, n_pass=2, max_screened=8, n_lift=1, oc_draws=20)
