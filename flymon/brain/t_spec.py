"""Every number of spec appendix T as amended by T.9 (T.9 wins over T.0-T.8, T.9.7 lists the replacements): the same
lever (APL->MBON05 removal), each engine variant's own reference-set z for the oracle, one M2 judgement on a new set
drawn from the widened Gen-1 opponent pool.
- TSpec subclasses S's LastSetSpec (itself R's LeverSpec), so R's records and bands (r_records, r_rules.read_band) and
  S's guard, gate ② and OC code (s_rules, s_records) read T's numbers through the same field names; only what T
  changes is restated here (plan Reading 2). Every seed-bearing field S declared is overridden, so no S block is
  reused by name.
- T's new seed blocks are FIELDS (judgement 24_400_xxx, smoke 24_409_xxx, gate ② 25_200_000+i and smoke 25_209_xxx in
  the nested P specs, the set generator's 20261004 and the cluster OC's 20261005), so every collision collector sees
  them (T.9.6). Gate ① supplement's strength seeds stay R's kc_seeds() method (encoder ③'s block, T.9.1).
- z (T.1, T.9.2-T.9.4): z_h4 is block h4's C3 z, which the unedited engine must reproduce bit for bit; the readout
  guard's thresholds and z's ddof are H.4's (read from h4_spec, never restated); the reference set is H.3's."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from .p_spec import SPEC as P_SPEC
from .p_spec import PSpec
from .p_spec import smoke as p_smoke
from .h4_spec import SPEC as H4
from .q_jobs import MBON05 as LEVER_EDIT
from .r_spec import LEVER
from .r_spec import SPEC as R_SPEC
from .s_spec import SPEC as S_SPEC
from .s_spec import LastSetSpec

_T_O = dict(o2_seed0=25_200_000, smoke_seed0=25_209_000)
P_C = dataclasses.replace(P_SPEC, o=dataclasses.replace(P_SPEC.o, **_T_O))
P_L = dataclasses.replace(P_SPEC, o=dataclasses.replace(
    P_SPEC.o, **_T_O, o1_conditions=((LEVER, LEVER_EDIT),) + P_SPEC.o.o1_conditions[1:]))


@dataclass(frozen=True)
class TSpec(LastSetSpec):
    # ---- the set (T.2 as replaced by T.9.1) -------------------------------------------------------------------------
    opp_num_max: int = 151                          # Gen-1 dex numbers 1..151, base forms (no `forme`)
    n_opp: int = 127                                # opponents whose types all lie in the codebook's 12 types
    n_combos: int = 1986
    set_rng_seed: int = 20261004
    first_turn: int = 0
    set_last_turn: int = 1985                       # the permutation's end (n_combos - 1)
    n_a: int = 43
    last_turn_b: int = 17                           # the 21st (b) row's turn
    last_turn: int = 103                            # T: the 43rd (a) row's turn (T_a in the sentences)
    digest_e0_b: str = "8c9729bfba529a7bea8d783a629c1c1f54380af79ffc8032e46f657d669a0c71"
    digest_e0_a: str = "e31b552636bc237f1d9fc85034432849b8a922e0d8b5fd17b1f87274b725e989"
    digest_keys: str = "37dde1ca0c9d0bf33772b42359e45257726b46d552837ffb4cde75e8beb03c97"
    skipped_declared: tuple = (("cap", 9), ("collision", 0), ("used", 164), ("glom_dup", 0), ("in_set", 48),
                               ("pool_only", 57))
    n_set_odours: int = 55
    clusters_b: tuple = (("FIRE*", "WATER", 1), ("GRASS+POISON", "FIRE*", 3), ("GRASS+POISON", "GROUND*", 3),
                         ("GROUND*", "NORMAL", 3), ("GROUND*", "POISON*", 3), ("NORMAL", "ROCK+WATER*", 1),
                         ("POISON*", "NORMAL", 1), ("ROCK+WATER*", "NORMAL", 2), ("WATER", "FLYING+NORMAL*", 4))
    clusters_a: tuple = (("FIRE*", 7), ("FLYING+NORMAL*", 8), ("FLYING+POISON*", 2), ("FLYING+WATER*", 1),
                         ("GRASS*", 5), ("GROUND*", 9), ("POISON*", 9), ("ROCK+WATER*", 2))
    # ---- z (T.1, T.9.2-T.9.4) ----------------------------------------------------------------------------------------
    z_h4: tuple = (("A", (10.78125, 9.412096743243064)), ("P", (26.25, 19.30889259728101)))
    z_guard_med_min: float = H4.react_med_delta_min        # 5: median(read − same-seed rest) per readout type
    z_guard_zero_max: float = H4.react_zero_share_max      # 0.25: share of zero read counts
    z_ddof: int = H4.z_ddof                                # 0: population SD
    # ---- the judgement seeds (T.3 8) and T's smoke (T.3 4) -----------------------------------------------------------
    judge_act_seeds: tuple = tuple(range(24_400_000, 24_400_008))
    judge_select_seeds: tuple = tuple(range(24_400_100, 24_400_108))
    judge_report_seeds: tuple = tuple(range(24_400_200, 24_400_208))
    smoke_seeds: tuple = tuple(range(24_409_000, 24_409_100))
    # ---- gate ② (T.3 6, T.9.3) ---------------------------------------------------------------------------------------
    p: PSpec = P_L
    p_c: PSpec = P_C
    # ---- gate ③ (T.9.2) ----------------------------------------------------------------------------------------------
    r_even_L_h4: int = 16                         # R's gate ③ L testable_b on h4 z (a record beside STOP_EVEN_REPRO)
    # ---- the cluster-correlated OC (T.6, T.9.6, T.9.7) ---------------------------------------------------------------
    oc_cluster_seed: int = 20261005
    oc_cluster_icc: float = 0.3
    oc_cluster_draws: int = 100_000
    oc_cluster_g: tuple = (("null", 0.1, 0.1), ("harm", 0.1, 0.02), ("harm", 0.1, 0.05))
    oc_cluster_fixture: str = "tests/brain/fixtures/t_oc_cluster.json"
    # ---- R's reused gates (T.3 1) ------------------------------------------------------------------------------------
    r_reused: tuple = ("repro", "gate1")
    r_commits: tuple = (("repro", "22c934b"), ("gate1", "6ad2201"), ("gate3", "a83977f"))
    r_cache_dir: str = R_SPEC.cache_dir           # R's even raw (gate ③'s C), read only, by content key
    s_summary: str = S_SPEC.summary               # S's judgement raw, read only after judge (T.9.6)
    # ---- paths -------------------------------------------------------------------------------------------------------
    cache_dir: str = "results/t/cache"
    smoke_cache_dir: str = "results/t/smoke/cache"
    smoke_detail: str = "results/t/smoke.json"
    z_detail: str = "results/t/z.json"
    summary: str = "results/summary/t_lever.json"
    archive_root: str = "~/flymon-archive/t"

    def z_h4_dict(self) -> dict:
        return {k: (float(v[0]), float(v[1])) for k, v in self.z_h4}

    def judge_seeds(self) -> dict:
        if self.smoke:
            raise ValueError("smoke never measures the judgement set (T.3 4)")
        return dict(act=list(self.judge_act_seeds), select=list(self.judge_select_seeds),
                    report=list(self.judge_report_seeds))


SPEC = TSpec()


def smoke(spec: TSpec = SPEC) -> TSpec:
    """Scale only: P's smoke derivation on both gate-② P specs (4 seeds 25_209_100+), T's smoke seeds through the
    inherited methods, 4 workers. Every threshold stays."""
    return dataclasses.replace(spec, p=p_smoke(spec.p), p_c=p_smoke(spec.p_c), smoke=True, workers=4)
