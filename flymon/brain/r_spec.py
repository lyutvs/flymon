# flymon/brain/r_spec.py
"""Every number of spec appendix R as amended by R.9 (R.9 wins over R.0-R.8): the lever APL->MBON05 removal and its
one M2 judgement on the encoder track's unused judgement set (L generator turns 64-103).
- Numbers the encoder track, Q and P hold are read (E = e_spec.SPEC, Q = q_spec.SPEC, P = p_spec.SPEC), never restated.
- The reused seed blocks — gate ① = encoder ③'s strength seeds (on purpose, R.8), gate ③ = H.4's 500-615, the
  judgement oracle block (encoder 4.5) — are returned by methods and never stored in a field: the encoder track's
  collision test reads every "seed" field of every flymon/brain/*_spec.py and would flag them (plan Reading 2).
- The class is LeverSpec, not RSpec: tests/brain/test_p_spec.py's collector expands a class named RSpec as
  flymon.rescope's (plan Reading 2).
- Gate ②'s P spec R_P is p_spec.SPEC with only the seed blocks (25_000_000+i, smoke 25_009_xxx) and O's first
  condition — the edit p_rules.p_judge's gate expects through spec.o.on_edit — changed (plan Reading 3)."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from ..agent.e_measure import PUNISH_TYPE, REWARD_TYPE
from ..agent.e_spec import SPEC as E
from .p_spec import SPEC as P_SPEC
from .p_spec import PSpec
from .p_spec import smoke as p_smoke
from .q_jobs import MBON05 as LEVER_EDIT
from .q_jobs import NONE as NO_EDIT
from .q_spec import SPEC as Q

LEVER = "lever"
R_P = dataclasses.replace(P_SPEC, o=dataclasses.replace(
    P_SPEC.o, o2_seed0=25_000_000, smoke_seed0=25_009_000,
    o1_conditions=((LEVER, LEVER_EDIT),) + P_SPEC.o.o1_conditions[1:]))


@dataclass(frozen=True)
class Cond:
    name: str
    edit: str               # q_jobs edit: the lever (apl_to_mbon05_zero) or none
    odour: str              # "egrid": the row's E-grid k2-norm odor_x / odor_y; "e0": its odor_x_e0 / odor_y_e0
    strength: float


@dataclass(frozen=True)
class LeverSpec:
    # ---- the model variant and the encoder (R.1) ---------------------------------------------------------------
    config: str = "k2-norm"
    strength: float = 1.0
    lever_edit: str = LEVER_EDIT
    no_edit: str = NO_EDIT
    lever_edges: int = 2                        # R.6: exactly 2 CSC edges, else INVALID
    a_type: str = "MBON13"
    p_type: str = "MBON05"
    punish_type: str = PUNISH_TYPE
    reward_type: str = REWARD_TYPE
    e0_strength: float = E.e0_strength
    cond_names: tuple = ("L", "C", "E0")
    # ---- the oracle (H.4 / encoder 4.4 as is) ---------------------------------------------------------------------
    settle_ms: float = E.settle_ms
    read_ms: float = E.read_ms
    window_ms: int = E.window_ms
    alphas: tuple = E.alphas
    fixed_alphas: tuple = ()                    # q_oracle_job's fixed reward-only arms are not run (Reading 5)
    active_fx: float = Q.active_fx              # q_oracle_job's reach record threshold, unchanged
    testable_min: float = E.testable_min
    naive_max: float = E.naive_max
    bar_b: int = E.bar_b
    f_a_min: int = E.f_a_min
    margin: int = E.margin
    n_b: int = E.n_b
    # ---- the judgement set (encoder 5.1; ⓪ 2c6a292) -----------------------------------------------------------------
    n_a: int = 32
    n_a_even: int = 18
    last_turn: int = 103
    digest_e0_b: str = "4b9bccc9a640d8affb36754f862f8ed67ea04c0b07e64f3d1529c57c6fe1ea3f"
    digest_e0_a: str = "78509547f571ec8ddbd0b2b575d989f761a547b2b52a48cc89bd0e8fe7c3b5c1"
    digest_keys: str = "33be39a1c5b086a7034e941ad20f4fbe055ce0708bc84ea6e138ba8fe44e028e"
    # ---- gate ① (R.9.2) -----------------------------------------------------------------------------------------------
    valid_band: tuple = Q.kc_band               # [0.03, 0.15], "Q와 같은 대역"
    n_calib: int = 112
    # ---- gate ② (R.2, Reading 3) ----------------------------------------------------------------------------------------
    p: PSpec = R_P
    # ---- gate ③ (R.9.3, R.9.6) ------------------------------------------------------------------------------------------
    c_even_expected: int = 7                    # encoder 13: k2-norm 7/21 on the same even pairs and seeds
    # ---- bands (R.3, R.9.4, R.9.5) -----------------------------------------------------------------------------------
    g_fail_drop: int = 2
    g_fail_flips: int = 3
    # ---- records (R.6, R.9.7) -------------------------------------------------------------------------------------------
    sat_fracs: tuple = (0.5, 0.8)
    sat_quantile: float = 95.0
    ms_per_s: float = 1000.0
    # ---- the operating characteristic (R.6, R.9.5, Reading 13) ------------------------------------------------------
    oc_q: tuple = E.oc_q
    oc_c: tuple = E.oc_c
    oc_naive_max: int = E.oc_naive_max
    oc_discord: tuple = (0.02, 0.05, 0.1, 0.15)
    oc_harm: tuple = ((0.1, 0.02), (0.15, 0.02), (0.2, 0.05))
    # ---- the no-edit reproduction gate (R.9.6, Reading 4) -----------------------------------------------------------
    kc_repro_odours: tuple = ("ELECTRIC|FIGHTING", "GRASS|FIRE+FLYING", "WATER|PSYCHIC+WATER")
    p_repro_idx: tuple = (0, 15, 31)
    oracle_repro_idx: tuple = (0,)
    # ---- smoke (R.8) ----------------------------------------------------------------------------------------------------
    smoke_seeds: tuple = tuple(range(25_008_000, 25_008_100))
    smoke_pairs: tuple = (0, 20)                # even (b) indices; the judgement set is never in smoke
    smoke_odours: int = 4
    smoke: bool = False
    # ---- paths and the pool ---------------------------------------------------------------------------------------------
    encoder_summary: str = Q.encoder_summary
    m0d_summary: str = E.m0d_summary
    p_summary: str = "results/summary/p_learning.json"
    p_cache_dir: str = "results/p/run/cache/p_arm"
    ref_dir: str = "results/r/ref/activity"
    cache_dir: str = "results/r/cache"
    smoke_cache_dir: str = "results/r/smoke/cache"
    smoke_detail: str = "results/r/smoke.json"
    summary: str = "results/summary/r_lever.json"
    archive_root: str = "~/flymon-archive/r"
    workers: int = 16
    pool_timeout_s: float = 3600.0

    def conditions(self) -> tuple:
        names = self.cond_names
        return (Cond(names[0], self.lever_edit, "egrid", self.strength),
                Cond(names[1], self.no_edit, "egrid", self.strength),
                Cond(names[2], self.no_edit, "e0", self.e0_strength))

    def cond(self, name: str) -> Cond:
        return {c.name: c for c in self.conditions()}[name]

    def kc_seeds(self) -> tuple:
        """Gate ① (and the KC repro): encoder ③'s strength seeds; smoke: two R smoke seeds."""
        return tuple(self.smoke_seeds[0:2]) if self.smoke else tuple(E.strength_seeds)

    def h4_seeds(self) -> dict:
        return dict(act=list(E.even_act_seeds), select=list(E.even_select_seeds), report=list(E.even_report_seeds))

    def even_seeds(self) -> dict:
        if self.smoke:
            s = self.smoke_seeds
            return dict(act=list(s[2:4]), select=list(s[4:6]), report=list(s[6:9]))
        return self.h4_seeds()

    def judge_seeds(self) -> dict:
        if self.smoke:
            raise ValueError("smoke never measures the judgement set (R.8)")
        return dict(act=list(E.judge_act_seeds), select=list(E.judge_select_seeds), report=list(E.judge_report_seeds))

    def p_repro_seeds(self) -> tuple:
        return tuple(P_SPEC.seeds[i] for i in self.p_repro_idx)


SPEC = LeverSpec()


def smoke(spec: LeverSpec = SPEC) -> LeverSpec:
    """Scale only: P's smoke derivation on R's P spec (4 seeds 25_009_100+), R's smoke seeds through the methods, 4
    workers. Every threshold stays."""
    return dataclasses.replace(spec, p=p_smoke(spec.p), smoke=True, workers=4)
