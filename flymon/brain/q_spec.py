# flymon/brain/q_spec.py
"""Every number of spec appendix Q as amended by Q.6 (Q.6 and Q.6.9 win over Q.2-Q.5): the reward-readout (MBON05) bottleneck
diagnosis on the E-grid k2-norm even (b) pairs — a characterisation: per candidate 일치 / 불일치 / 판단 불가, INVALID
for defective data, no verdict label and no STOP (Q.1).
The encoder track's numbers (windows, alphas, the H.4 oracle seeds, n_b, the r split, the M0d path) and the DAN types
are read from e_spec / e_measure, never restated. Smoke runs and tests use smoke(SPEC) or dataclasses.replace."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from ..agent.e_measure import PUNISH_TYPE, REWARD_TYPE
from ..agent.e_spec import SPEC as E


@dataclass(frozen=True)
class Condition:
    name: str
    edit: str               # "none" | "apl_to_mbon05_zero" | "apl_to_nonkc_zero"
    mv_scale: float         # x mv_per_synapse (Q.6.5 mv_scale)
    strength: float         # presentation strength s
    role: str               # "baseline" | "primary" | "record"
    kc_gate: bool           # Q.6.4: the KC band gates (b), (c), (d)


@dataclass(frozen=True)
class QSpec:
    # ---- pairs and the split (Q.3, Q.6.1, Q.6.7) -----------------------------------------------------------------
    config: str = "k2-norm"
    strength: float = 1.0                       # E-grid k2-norm s 1.0 (encoder block strength)
    n_b: int = E.n_b
    n_f: int = 12                               # Q.3: r < 2 on Q0
    n_s: int = 9
    split_r: float = E.testable_min             # F: r < 2, S: r >= 2
    borderline: tuple = (1.89, 1.97)            # Q.6.7: F pairs with r in [1.89, 1.97]
    even_turns: int = 16                        # h4_pairs.build_turns(..., 16) as e_pairs.even_situations builds them
    pairs_subset: tuple = ()                    # () = all 21 (b) pairs; smoke: smoke_pairs
    smoke_pairs: tuple = (0, 20)
    # ---- the oracle (H.4 as is) and Q's records (Q.6.2, Q.3) --------------------------------------------------------
    alphas: tuple = E.alphas
    fixed_alphas: tuple = (0.8, 1.0)            # Q.6.2: fixed reward-only arms on the report seeds
    active_fx: float = 0.5                      # Q.3: f_X counts KCs with naive firing probability >= 0.5
    settle_ms: float = E.settle_ms
    read_ms: float = E.read_ms
    window_ms: int = E.window_ms
    punish_type: str = PUNISH_TYPE
    reward_type: str = REWARD_TYPE
    p_type: str = "MBON05"                      # the P readout; apl_to_mbon05_zero's target type
    # ---- the reproduction gate (Q.6.8): H.4 seeds, 3 of the 21 (b) pairs ---------------------------------------
    repro_act_seeds: tuple = E.even_act_seeds
    repro_select_seeds: tuple = E.even_select_seeds
    repro_report_seeds: tuple = E.even_report_seeds
    repro_pairs: tuple = (0, 10, 20)
    # ---- Q1 seeds (Q.3) and smoke (Q.6.8) ----------------------------------------------------------------------
    act_seeds: tuple = tuple(range(24_000_000, 24_000_008))
    select_seeds: tuple = tuple(range(24_000_100, 24_000_108))
    report_seeds: tuple = tuple(range(24_000_200, 24_000_208))
    smoke_seeds: tuple = tuple(range(24_008_000, 24_008_100))
    kc_probe_seeds: tuple = ()                  # smoke only: the mv_scale KC-band probe
    # ---- conditions (Q.3, Q.6.4-Q.6.6) ------------------------------------------------------------------------
    cond_names: tuple = ("base", "apl_mbon05", "apl_nonkc", "s_up", "mv_lo", "mv_hi")
    edit_none: str = "none"
    edit_mbon05: str = "apl_to_mbon05_zero"
    edit_nonkc: str = "apl_to_nonkc_zero"
    apl_mbon05_edges: int = 2                   # Q.6.6: APL -> MBON05 on the real connectome
    mv_scales: tuple = (0.8, 1.25)
    mv_fallback: tuple = (0.9, 1.1)
    s_down: tuple = (0.7, 0.8)                  # Q.6.9: (c) lowers s: 0.7, fallback 0.8 by the KC-band gate at smoke
    s_step: float = 0.05                        # Q.6.9: only the RECORD of the cap-limited max s (no longer sets (c))
    s_weak: float = 0.15                        # Q.6.9: |s - 1.0| < 0.15 -> "조작 약함", ① 판단 불가
    kc_band: tuple = (0.03, 0.15)
    stop_p: float = 5.0                         # P < 5 = silent (정지)
    # ---- rules (Q.4, Q.6.1-Q.6.6) ------------------------------------------------------------------------------
    auc_match: float = 0.75
    auc_mismatch: float = 0.6
    rho_min: float = 0.4                        # Q.6.9: ① rho(naive P_X, dr_P_c) >= +rho_min, pairs with naive P_X > 0 only
    rho_positive_only: bool = True              # Q.6.9: P_X = 0 pairs are "already at floor", recorded apart
    change_two_sided: bool = True               # Q.6.9: ④ |median dr_P| and two-sided Wilcoxon; ② increase only
    wilcoxon_p: float = 0.05
    delta_min: float = 0.5
    noise_mult: float = 2.0
    var_mean_match: float = 0.75
    var_sd_match: float = 1.5
    var_mean_mismatch: float = 0.5
    floor_guard_pairs: int = 2
    fs_agree_min: float = 0.7
    fs_disagree_max: int = 5
    mv_limitation: str = "mv_scale은 입력 구동도 바꾸므로 ④의 일치는 ①과 완전히 갈린 것이 아니다"
    weak_note: str = "조작 약함"
    cancel_note: str = "A·Y항 상쇄"
    # ---- paths and the pool -------------------------------------------------------------------------------------
    encoder_summary: str = "results/summary/encoder_grid.json"
    m0d_summary: str = E.m0d_summary
    q0_cache_dir: str = "results/q/q0_cache"
    cache_dir: str = "results/q/cache"
    smoke_detail: str = "results/q/smoke.json"
    summary: str = "results/summary/q_reward.json"
    smoke: bool = False
    workers: int = 16
    pool_timeout_s: float = 3600.0

    def q1_seeds(self) -> dict:
        return dict(act=list(self.act_seeds), select=list(self.select_seeds), report=list(self.report_seeds))

    def repro_seeds(self) -> dict:
        return dict(act=list(self.repro_act_seeds), select=list(self.repro_select_seeds),
                    report=list(self.repro_report_seeds))

    def conditions(self, s_c: float, mv_lo: float, mv_hi: float) -> tuple:
        s = self.strength
        return (Condition(self.cond_names[0], self.edit_none, 1.0, s, "baseline", False),
                Condition(self.cond_names[1], self.edit_mbon05, 1.0, s, "primary", True),
                Condition(self.cond_names[2], self.edit_nonkc, 1.0, s, "record", True),
                Condition(self.cond_names[3], self.edit_none, 1.0, float(s_c), "primary", True),
                Condition(self.cond_names[4], self.edit_none, float(mv_lo), s, "primary", True),
                Condition(self.cond_names[5], self.edit_none, float(mv_hi), s, "primary", True))


SPEC = QSpec()


def smoke(spec: QSpec = SPEC) -> QSpec:
    """Scale only: 2 activity, 2 selection, 3 report and 4 KC-probe seeds from the smoke block, the two smoke pairs,
    4 workers. Every threshold stays."""
    s = spec.smoke_seeds
    return dataclasses.replace(spec, act_seeds=s[0:2], select_seeds=s[2:4], report_seeds=s[4:7],
                               kc_probe_seeds=s[7:11], pairs_subset=spec.smoke_pairs, smoke=True, workers=4)
