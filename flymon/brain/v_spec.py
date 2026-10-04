"""Every number of spec appendix V as amended by V.9 (V.9 wins over V.0-V.8): the combined lever L_V (APL->MBON05 removed
and the MBON05->MBON09 / MBON11 / MBON01 chain entry cut — u_measure.u_edit(0.0, "chain_entry"), 13 CSC edges), each
engine variant's own reference-set z for the oracle (z_V for L_V), a new set from T's widened-pool generator filtered by
the per-odour KC input of both engines, and one M2 judgement on it.
- VSpec subclasses U's USpec (and through it T's TSpec, S's and R's), so R's records and bands, S's guard / gate ② / OC
  code, T's z / OC code and U's sides and records read V's numbers through the same field names; only what V changes is
  restated. Every seed-bearing field U declared is overridden (judgement 24_600_xxx, smoke 24_609_xxx, gate ②
  25_400_000+i and its smoke 25_409_xxx in the nested P specs).
- V measures with U's measurement file unchanged (V.9.5 P1-3): the lever is the fixed string u_edit(0.0, "chain_entry");
  its APL->MBON05 mask count is 2 (the jobs' edit_edges) and the chain entry is 7 / 2 / 2 (the reference rows'
  block_edges) — 13 edges, CSC 2d359b8b…. The V measurement key is U's, declared as a literal.
- V's set has no declared digests (V.2: the set block fixes them before any oracle on the set); T's set's declared
  values, which USpec inherits, are cleared here so nothing reads them as V's. The generator's permutation seed and T's
  cluster-OC seed stay T's (t_spec.SPEC), as in U.
- Paths: results/v/ (git-ignored), results/summary/v_lever.json (tracked), ~/flymon-archive/v."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from .p_spec import SPEC as P_SPEC
from .p_spec import PSpec
from .p_spec import smoke as p_smoke
from .u_measure import u_edit
from .u_spec import SPEC as U_SPEC
from .u_spec import USpec, p_lever

LEVER_V = u_edit(0.0, "chain_entry")                # "u_apl_mbon05_x0.0+chain_entry" (= U's entry_f0)
_V_O = dict(o2_seed0=25_400_000, smoke_seed0=25_409_000)
P_C = dataclasses.replace(P_SPEC, o=dataclasses.replace(P_SPEC.o, **_V_O))
P_L = p_lever(P_C, LEVER_V)


@dataclass(frozen=True)
class VSpec(USpec):
    # ---- the lever (V.1, V.9.5) ---------------------------------------------------------------------------------------
    lever_edit: str = LEVER_V
    f: object = 0.0                                  # U's f of the APL->MBON05 part (removed)
    sha_combined: str = "2d359b8b6947d542348b658db977e1edbe0db5919e7f4b1187598cb10ef1253a"
    n_lever_edges_total: int = 13                    # 2 APL->MBON05 + 7 / 2 / 2 chain entry
    mbon05_apl_edges: int = 2                        # V.1: untouched, count and weights
    mbon05_apl_weights: tuple = (-7.865, -12.929)    # as V.0 prints them (float32 −7.8651862, −12.928195)
    mbon05_apl_weight_tol: float = 0.001             # V.0's printed precision; the rule itself is "bytes unchanged"
    # ---- the reuse (V.3 1, V.9.5 P3-12) -------------------------------------------------------------------------------
    u_measure_key_u: str = "8a4e09302f7f2f7e7d69f3540d8fc392da582cb1246bfebeb150348776e26523"
    u_summary: str = U_SPEC.summary
    u_commits: tuple = (("reuse", "d00fdf2"), ("path", "3b83824"), ("scan", "d732b20"), ("kc", "c0d09a7"))
    u_path_detail: str = U_SPEC.path_detail
    u_scan_detail: str = U_SPEC.scan_detail
    u_path_detail_sha256: str = "57ae0c4ed11396421e21a91ae0bd5f88a20c2e8215e612a1225d6599b94be329"
    u_scan_detail_sha256: str = "a4cddfd7fbdec1dd8a95e93088aeecff2c1e9eca389454bcd8d81d4f45c0ab7e"
    u_entry_point: str = "entry_f0"                  # U's scan row of the same edit (V.3 2's base for L_V)
    u_kc_points: tuple = ("0.6", "0.7", "0.8")       # every U KC point holds the same per_odour_none
    # ---- the set (V.2, V.9.1, V.9.5 P2-10 / P3-12) --------------------------------------------------------------------
    first_turn: int = 0
    n_a: int = 43
    kc_reasons_last: str = "kc_input"
    digest_e0_b: str = ""                            # T's declared set values are not V's: V's are in block set
    digest_e0_a: str = ""
    digest_keys: str = ""
    last_turn: object = None
    last_turn_b: object = None
    skipped_declared: tuple = ()
    n_set_odours: object = None
    clusters_b: tuple = ()
    clusters_a: tuple = ()
    # ---- the even gate (V.3 7, V.9.3) ---------------------------------------------------------------------------------
    even_drop_b_lt: int = 3
    even_drop_a_le: int = 1
    # ---- the judgement seeds (V.3 11) and V's smoke (V.3 8) -----------------------------------------------------------
    judge_act_seeds: tuple = tuple(range(24_600_000, 24_600_008))
    judge_select_seeds: tuple = tuple(range(24_600_100, 24_600_108))
    judge_report_seeds: tuple = tuple(range(24_600_200, 24_600_208))
    smoke_seeds: tuple = tuple(range(24_609_000, 24_609_100))
    # ---- gate ② (V.3 10, V.9.2) ---------------------------------------------------------------------------------------
    p: PSpec = P_L
    p_c: PSpec = P_C
    # ---- the cluster OC (V.6, V.9.5 P2-9): T's method on V's clusters, a runtime output -------------------------------
    oc_cluster_out: str = "results/v/oc_cluster.json"
    oc_cluster_sha256: str = ""                      # V's table is computed after block set; no declared sha
    # ---- paths ----------------------------------------------------------------------------------------------------------
    cache_dir: str = "results/v/cache"
    smoke_cache_dir: str = "results/v/smoke/cache"
    smoke_detail: str = "results/v/smoke.json"
    path_detail: str = "results/v/path.json"
    kc_input_detail: str = "results/v/kc_input.json"
    summary: str = "results/summary/v_lever.json"
    archive_root: str = "~/flymon-archive/v"

    def u_kc_none(self, u_doc: dict) -> dict:
        """U's per_odour_none (the unedited engine's 55 T-set odours, U block kc), equal at every U KC point."""
        pts = [u_doc["kc"]["points"][k]["record_set"]["per_odour_none"] for k in self.u_kc_points]
        if any(p != pts[0] for p in pts):
            raise ValueError("U's KC points disagree on per_odour_none")
        return dict(pts[0])


SPEC = VSpec()


def smoke(spec: VSpec = SPEC) -> VSpec:
    """Scale only: P's smoke derivation on both gate-② P specs (4 seeds 25_409_100+), V's smoke seeds through the
    inherited methods, 4 workers. Every threshold stays."""
    return dataclasses.replace(spec, p=p_smoke(spec.p), p_c=p_smoke(spec.p_c), smoke=True, workers=4)
