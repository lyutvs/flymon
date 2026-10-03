"""Every number of spec appendix U as amended by U.9 (U.9 wins over U.0-U.8): the partial lever L_f (the 2 APL->MBON05
edges × f), each engine variant's own reference-set z for the oracle (T.1), and one M2 judgement on T's unused set.
- USpec subclasses T's TSpec, so R's records and bands, S's guard / gate ② / OC code and T's set, z and OC code read
  U's numbers through the same field names; only what U changes is restated here. Every seed-bearing field T declared
  is overridden (judgement 24_500_xxx, smoke 24_509_xxx, gate ② 25_300_000+i and its smoke 25_309_xxx in the nested P
  specs); T's set's declared values and T's cluster-OC numbers are inherited unchanged (U.2, U.6).
- T's set (U.2) and T's cluster OC (U.6) are generated through T's own spec (t_spec.SPEC): U's spec sets their two
  generator seeds to () (no int, no None — the encoder's
  collector reads a None default as a missing one), so no second spec module declares T's 20261004 / 20261005 (T's global collision test).
- The lever is f-dependent: SPEC leaves it unresolved (lever_edit "" — an R / U job refuses that edit), and
  `at(spec, f)` returns the spec of L_f: lever_edit = u_measure.u_edit(f), the lever P spec's first O condition = that
  edit, f recorded. The runner resolves f* from block `choose` (and each scan / even f from its own block).
- The f grid and the contrast edits' type pairs are the measurement file's (u_measure.F_GRID / BLOCKS); the declared
  contrast edge counts (U.9.3 4) are here and must equal what apply_u_edit reports."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from .p_spec import SPEC as P_SPEC
from .p_spec import PSpec
from .p_spec import smoke as p_smoke
from .q_jobs import MBON05 as T_LEVER_EDIT
from .r_spec import LEVER
from .t_spec import SPEC as T_SPEC
from .t_spec import TSpec
from .u_measure import F_GRID, u_edit

_U_O = dict(o2_seed0=25_300_000, smoke_seed0=25_309_000)
P_C = dataclasses.replace(P_SPEC, o=dataclasses.replace(P_SPEC.o, **_U_O))


def p_lever(p: PSpec, edit: str) -> PSpec:
    """p with O's first condition (the edit p_rules.p_judge expects through spec.o.on_edit) set to `edit`."""
    return dataclasses.replace(p, o=dataclasses.replace(p.o, o1_conditions=((LEVER, edit),) + p.o.o1_conditions[1:]))


UNRESOLVED = ""
P_L = p_lever(P_C, UNRESOLVED)


@dataclass(frozen=True)
class USpec(TSpec):
    # ---- the lever (U.1) --------------------------------------------------------------------------------------------
    lever_edit: str = UNRESOLVED                    # at(spec, f) sets u_edit(f)
    f: object = None                                # the resolved f (None: unresolved)
    f_grid: tuple = F_GRID                          # the guard scan's 9 points, fixed before results
    f_ends: tuple = (0.0, 1.0)                      # U.9.1's endpoints (f = 0: R / S / T's L, f = 1: no lever)
    n_candidates_max: int = 3                       # U.3 3 (b): the smallest passing f first, at most 3, no substitutes
    # ---- the endpoint reproduction (U.9.1) ----------------------------------------------------------------------------
    sha_none: str = "1aee839811b8aa662588fbfab3050cee00962ddc8ecd4d2fd1361fd0c72692d5"
    sha_zero: str = "860cba4f9eead9d7a85b632f2c93c51790c5082fe80460975cb1300239c73e46"
    t_none_edit: str = "none"                       # T's z rows: rows["none"] (f = 1) and rows[T's lever] (f = 0)
    t_lever_edit: str = T_LEVER_EDIT
    path_even_n: int = 3                            # the first 3 even pairs in their declared order
    # ---- T's reused z block (U.3 1) -----------------------------------------------------------------------------------
    t_summary: str = T_SPEC.summary
    t_z_detail: str = T_SPEC.z_detail
    t_z_detail_sha256: str = "fc8bb380eb4050c286624bf697422d6b57cb109ba55b13f91bf7d5f6c127762c"
    t_measure_key_t: str = "7255f872802602bbe80244af8a6a607a44acc415f7434cbdac9ff29e6cb2d374"
    t_commits: tuple = (("z", "0eb642e"),)
    # ---- the mechanism records (U.6, U.9.3, U.9.4 P2-6 / P3-11) ---------------------------------------------------------
    chain_types: tuple = ("MBON09", "MBON11", "MBON01", "MBON03", "CRE055", "MBON30", "LHMB1")
    kc_types: tuple = ("KCa'b'-ap1", "KCa'b'-ap2", "KCa'b'-m")
    contrast_edges: tuple = (("out_block", (("MBON03->MBON13", 4), ("CRE055->MBON13", 17))),
                             ("chain_entry", (("MBON05->MBON09", 7), ("MBON05->MBON11", 2), ("MBON05->MBON01", 2))))
    contrast_points: tuple = (("block_f1", 1.0, "out_block"), ("block_f0", 0.0, "out_block"),
                              ("entry_f0", 0.0, "chain_entry"))
    chain_support_max: float = 0.5                  # D_block ≤ 0.5 × D_none → "사슬 지지"
    chain_against_min: float = 0.8                  # D_block ≥ 0.8 × D_none → "사슬 비지지"
    kc_side_delta_max: float = 1.0                  # |Δ(entry cut) − Δ(f = 0 alone)| < 1 → "KC 경로 쪽"
    # ---- the f choice on the even pairs (U.3 5, U.9.2) ------------------------------------------------------------
    even_drop_b_lt: int = 3                         # (b): net drop pun_C − pun_L < 3
    even_drop_a_le: int = 1                         # (a): net drop ≤ 1
    # ---- the judgement seeds (U.3 9) and U's smoke (U.3 6) ----------------------------------------------------------
    judge_act_seeds: tuple = tuple(range(24_500_000, 24_500_008))
    judge_select_seeds: tuple = tuple(range(24_500_100, 24_500_108))
    judge_report_seeds: tuple = tuple(range(24_500_200, 24_500_208))
    smoke_seeds: tuple = tuple(range(24_509_000, 24_509_100))
    # ---- gate ② (U.3 8) ------------------------------------------------------------------------------------------------
    p: PSpec = P_L
    p_c: PSpec = P_C
    # ---- T's set (U.2) and T's cluster OC (U.6) are read through T's own spec ------------------------------------------
    set_rng_seed: tuple = ()                        # T's 20261004 stays T's: U generates the set with t_spec.SPEC
    oc_cluster_seed: tuple = ()                     # T's 20261005 stays T's: U recomputes the cluster OC with t_spec.SPEC
    oc_cluster_sha256: str = "83a4d0bdce784536908e3b0e2e00efe8cb6610dec43ee6fca2be4060c7ecffb0"
    # ---- R's reused gate (U.3 1) --------------------------------------------------------------------------------------
    r_reused: tuple = ("repro",)
    r_commits: tuple = (("repro", "22c934b"), ("gate3", "a83977f"))
    # ---- paths ----------------------------------------------------------------------------------------------------------
    cache_dir: str = "results/u/cache"
    smoke_cache_dir: str = "results/u/smoke/cache"
    smoke_detail: str = "results/u/smoke.json"
    path_detail: str = "results/u/path.json"
    scan_detail: str = "results/u/scan.json"
    summary: str = "results/summary/u_lever.json"
    archive_root: str = "~/flymon-archive/u"

    def judge_seeds(self) -> dict:
        if self.smoke:
            raise ValueError("smoke never measures the judgement set (U.3 6)")
        return dict(act=list(self.judge_act_seeds), select=list(self.judge_select_seeds),
                    report=list(self.judge_report_seeds))

    def contrast_declared(self) -> dict:
        return {b: dict(pairs) for b, pairs in self.contrast_edges}


SPEC = USpec()


def at(spec: USpec, f: float) -> USpec:
    """The spec of L_f: lever_edit u_edit(f) (ValueError outside the allowed f), the lever P spec on that edit."""
    e = u_edit(f)
    return dataclasses.replace(spec, f=float(f), lever_edit=e, p=p_lever(spec.p, e))


def smoke(spec: USpec = SPEC) -> USpec:
    """Scale only: P's smoke derivation on both gate-② P specs (4 seeds 25_309_100+), U's smoke seeds through the
    inherited methods, 4 workers. Every threshold stays."""
    return dataclasses.replace(spec, p=p_smoke(spec.p), p_c=p_smoke(spec.p_c), smoke=True, workers=4)
