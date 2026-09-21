"""The M0d H.4 runner's one configuration object (spec appendix H.4, amended by H.3a.1, H.3a.9 and H.4a).

H.3a.11's rule carries over: every number the H.4 runner, its rules and its records use is a field of `SPEC` (or of a
`dataclasses.replace` of it, for smoke runs and tests), and no other module restates one. The reference set, its
windows, the design pair and the connectome are H.3's (`h3`): H.4's reactivity and z constants are defined on H.3's
reference set (H.3: "활성 조정과 H.4의 반응성·z 상수는 이 집합에서만 한다"), and the reactivity is the H.3 guard's
measurement, served from H.3's measurement cache.
"""
from __future__ import annotations

from dataclasses import dataclass

from .h3_spec import SPEC as H3_SPEC, H3Spec, Window


@dataclass(frozen=True)
class H4Spec:
    h3: H3Spec = H3_SPEC
    # ---- the combinations (H.3a.1: C2 dropped); the order is also H.4's tie order C0 > C1 > C3 -----------------
    combos: tuple = ("C0", "C1", "C3")
    h3_block: str = "h3"                        # results/summary/m0d.json block holding the adopted Params
    # ---- H.4 step 1: readout reselection ---------------------------------------------------------------------
    react_med_delta_min: float = 5.0            # median of (read - same-seed rest) over the reference set
    react_zero_share_max: float = 0.25          # share of reference presentations with a zero read count
    teach_seeds: tuple = tuple(range(8, 16))    # M0c's judged seeds
    teach_min_decreased: int = 6                # H.3a.9 (1): 6 of 8 (was 7)
    teach_orders: tuple = ("ab", "ba")          # H.4a: both odour orders run; each type is judged on its odour
    teach_trials: int = 12                      # the M0c arm (pool_jobs.conditioning_arm_job)
    teach_present_ms: float = 800.0
    teach_gap_ms: float = 200.0
    teach_window: Window = Window(800.0, 600.0)
    z_ddof: int = 0                             # F.3's frozen constants are population SDs
    # ---- H.4 step 2: the oracle (G.14.3) -------------------------------------------------------------------------
    oracle_alphas: tuple = (0.2, 0.5, 0.8)
    act_seeds: tuple = tuple(range(500, 508))
    select_seeds: tuple = tuple(range(600, 608))
    report_seeds: tuple = tuple(range(608, 616))
    oracle_window: Window = Window(800.0, 600.0)
    kc_window_ms: int = 200                     # G.8's sliding window, recorded
    n_pairs_a: int = 18
    n_pairs_b: int = 21
    pairs_digest: str = "4e7298340f2225b839059e0b9508db1351c9e5630998d9deae2b4d0712946338"
    # ---- G.14.4: the per-pair formula and the per-combination aggregate ------------------------------------------
    testable_min: float = 2.0                   # m = min(r, -p) >= 2
    naive_max: float = 0.5                      # |d_pre| < 0.5
    # ---- H.4 step 3: selection -------------------------------------------------------------------------------------
    f_a_min: int = 2
    t_b_min: float = 0.5
    tie_pairs: int = 2                          # within 2 testable (b) pairs of the top -> the earlier combination
    # ---- interpretation notes carried with the result (H.3a.10, H.4a.2, H.4a.4) ------------------------------------
    notes: tuple = (
        "C0 대 C1·C3 비교는 기저에서 교락돼 있다(H.3a.10): C1·C3는 mbon_hold_frac을 3.5 Hz로 재보정했고 C0은 동결 0.85로 간다. "
        "기저는 hold에 약 27.9 Hz/unit로 반응하고 H.4의 판독이 MBON 발화율이다.",
        "V는 APL→MBON05 억제에 직접 반응할 수 있다(H.3a.9 ②, H.4a.2): A·P 풀이 모두 APL 표적이고 MBON05는 APL의 MBON 표적 1위다. "
        "APL→MBON05를 끊으면 MBON05가 풀려 MBON13이 바닥으로 내려가므로 절제판 V로는 이 교락을 가를 수 없다.",
        "판독 바닥이 조합마다 다르다(H.4a.4): H.3 가드의 MBON13 중앙값 Δ가 C0 17.5 · C1 9.5 · C3 6.0(문턱 5)이라 T_b 차이의 일부는 "
        "엔진이 아니라 판독 바닥의 차이일 수 있다(G.14.8과 같은 종류). 쌍 냄새의 순진 판독 수준·0 비율과 타입별 d′를 함께 싣는다.")


SPEC = H4Spec()
