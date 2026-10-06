"""Every number of spec appendix Z (Z.0–Z.9.1 and Z.9.2, which takes precedence) that Z's code reads. Order 0 (0a–0d)
reads the fields above the marker; the fields below it are declared now (seeds, pilot and design numbers) so that the
seed-collision test sees every Z block and the 0a environment hash binds them, and the later-orders code only reads
them. Y's numbers that Z's imported Y functions read are passed through dataclasses.replace(y_spec.SPEC, …) and never
restated here, except the facts Z checks against Y's committed blocks.
- Seeds (Z.2, Z.9.2): Z's own roots carry "seed" in their field names (the collectors see them); the roots Z reuses
  only to bit-reproduce Y's diagnosis (Z.2 "새 흐름이 아니다": 77_000_000's Y tags, the pilot-size emulation's
  77_000_007, the counterfactual scripts' temporary roots 1–6) are named *_root so that the collectors, which bind Y's
  block 60_000_000–78_299_999 to Y alone, do not count them as Z declarations.
- Printed diagnosis values (Z.0, record only except the three gate lines): (name, printed value, digits).
- Paths: results/z/ (git-ignored), results/summary/z_learning.json and results/summary/z_oc_draws.json (tracked),
  ~/flymon-archive/z."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ZSpec:
    # ---- the stage chain (Z.7 as amended by Z.9.2's 순서; plan Reading 1) -----------------------------------------
    stages: tuple = ("stage0", "ydiag", "reuse", "sens", "split", "pilot", "precheck", "oc", "smoke", "gates", "learn",
                     "band", "records", "seal", "judge")
    order0: tuple = ("stage0", "ydiag", "reuse", "sens", "split")
    stage_labels: tuple = (("stage0", "0a"), ("ydiag", "0b"), ("reuse", "0c"), ("sens", "0c′"), ("split", "0d"))
    later_modules: tuple = ("flymon.brain.z_runner_b",)          # the later-orders plan's stages (Reading 1)
    pool_stages: tuple = ("pilot", "smoke", "gates", "learn", "band", "records")
    # ---- reuse facts (Z.2, Z.7 0c) -------------------------------------------------------------------------------
    decl_commit: str = "7c4bde5"                                 # Z.9.2 commit: y_* / x_* / w_* as of Z's declaration
    y_blocks: tuple = (("digest", "36df057"), ("oracle", "2a9425a"), ("stage0", "b1279d2"), ("reuse", "2b82a8f"),
                       ("pilot", "eaffc5a"), ("precheck", "279e916"), ("oc", "547c3b0"))
    y_outcomes: tuple = (("digest", "PASS"), ("oracle", "PASS"), ("stage0", "PASS"), ("reuse", "PASS"),
                         ("pilot", "PASS"), ("precheck", "PASS"), ("oc", "STOP_OC_UNREACHABLE"))
    oracle_sha256: str = "fa4750ac80883bcdc571e8e5b6ba7cc3d25f320549988dc0c3ab8823f75768bb"
    n_len: int = 31
    n_len_axes: tuple = (("b", 11), ("a", 20))
    y_admitted: tuple = ("a|4|Rock Slide|Strength", "b|2|Mega Drain vs Machamp|Mega Drain vs Tentacruel",
                         "b|10|Surf vs Slowbro|Surf vs Venusaur", "b|17|Thunderbolt vs Nidoran-M|Thunderbolt vs Clefairy",
                         "a|218|Earthquake|Strength", "a|305|Earthquake|Rock Slide")
    y_n_sigma: int = 5
    # ---- 0b: the Y order-6 cells and the gate lines (Z.7 0b (i)–(iii)) --------------------------------------------
    y_cells_dir: str = "results/y/progress/oc"
    y_cells_n: int = 200
    y_oc_detail: str = "results/y/oc.json"
    redraw_n: int = 50                                           # (ii): draws 0–49 re-simulated
    diag_design: tuple = (0.5, 0.5, 16, 8, (4, 8))               # p_set, q, K, F, k range (the records target)
    gate_by_k: tuple = (0.047, 0.087, 0.159, 0.050, 0.082)
    gate_sim: float = 0.047
    gate_false: float = 0.0
    gate_digits: int = 3
    # ---- 0b record-only reproduction (Z.0 [사후 진단]; the diagnosis's own streams, never Z's) -------------------
    diag_root: int = 77_000_000                                  # Y's oc root (Y tags)
    psize_root: int = 77_000_007                                 # the pilot-size emulation (oc root + 7)
    cf_roots: tuple = (1, 2, 3, 4, 5, 6)                         # the counterfactual scripts' temporary roots
    draw_qs: tuple = (0.0, 1.0, 5.0, 10.0, 20.0, 25.0, 50.0, 75.0)
    below: tuple = (0.8, 0.5, 0.1)
    scen_qs: tuple = (2.5, 5.0)
    low_power: float = 0.8                                       # "낮은 추출" = per-draw worst < 0.80
    near_qs: tuple = (5.0, 10.0, 25.0, 50.0)
    near_split: float = 1.2
    reach_grid: tuple = (0.0, 100.0, 2.0)                        # a 0–100 step 2
    reach_resid: int = 500
    point_grid: tuple = (0.0, 120.0, 0.5)
    selfbase_ranks: tuple = (0, 1, 2, 3, 16, 25, 100)
    cf_p: tuple = (43.0, 55.0, 70.0)                             # cf: c_P only, worst 4 + rank 100
    cf_p_ranks: tuple = (0, 1, 2, 3, 100)
    cf_a: tuple = (20.0, 28.0, 36.0)                             # cfA: c_A only, worst 4 + ranks 16 · 25
    cf_ab: tuple = ((28.0, 55.0), (36.0, 70.0))                  # cfB: both, the same 6
    cf_ranks: tuple = (0, 1, 2, 3, 16, 25)
    cf_ab_bars: tuple = (0.91, 0.94)                             # Z.0: (28, 55) 5 of 6 ≥ 0.91, (36, 70) 6 of 6 ≥ 0.94
    sub_thr: tuple = ((20.0, 43.0), (24.0, 50.0), (28.0, 55.0))  # sub50: draws 0–49
    thr_qs: tuple = (5.0, 50.0, 90.0, 95.0, 99.0, 100.0)
    psize_ns: tuple = (6, 12, 24)
    psize_draws: int = 40
    sigma_share: tuple = (5, 6)                                  # n_Σ = round(n · 5 / 6)
    psize_qs: tuple = (5.0, 10.0, 50.0)
    rho_cols: tuple = ("a_pow", "drift_N1pre_AX", "drift_N1pre_PX", "sdpair_AY", "b_pow", "drift_RN2RN1_AX")
    printed: tuple = (
        # gate (i) lines are gate_*; everything below is record only (Z.7 0b 기록 전용)
        ("draw_q0", 0.000, 3), ("draw_q1", 0.000, 3), ("draw_q5", 0.040, 3), ("draw_q10", 0.349, 3),
        ("draw_q20", 0.852, 3), ("draw_q25", 0.889, 3), ("draw_q50", 0.962, 3), ("draw_q75", 0.986, 3),
        ("n_below_0.8", 37, 0), ("n_below_0.5", 25, 0), ("n_below_0.1", 15, 0),
        ("base_q5", 0.885, 3), ("base_q2.5", 0.872, 3), ("near_q5", 0.040, 3),
        ("base_only_sim", 0.885, 3), ("base_only_k4", 0.885, 3), ("base_only_k5", 0.940, 3),
        ("base_only_k6", 0.962, 3), ("base_only_k7", 0.948, 3), ("base_only_k8", 0.965, 3),
        ("base_only_n_4_8", 114, 0), ("base_only_n_6_10", 205, 0), ("base_only_max_4_8", 0.892, 3),
        ("base_only_max_6_10", 0.950, 3), ("near_only_n", 0, 0), ("near_only_max_4_8", 0.047, 3),
        ("near_only_max_6_10", 0.034, 3),
        ("low_near_g0", 0.31, 2), ("low_near_g0.5", 0.32, 2), ("low_near_g1", 0.37, 2),
        ("low_base_g0", 0.995, 3), ("low_base_g0.5", 0.99, 2), ("low_base_g1", 0.978, 3),
        ("rho_drift_N1pre_AX", 0.610, 3), ("rho_a_pow", -0.548, 3), ("rho_sdpair_AY", -0.487, 3),
        ("rho_drift_N1pre_PX", 0.425, 3), ("rho_drift_RN2RN1_AX", 0.407, 3), ("rho_a_drift", -0.879, 3),
        ("rho_sdpair_AY_partial_a", -0.24, 2),
        ("r2_drift", 0.53, 2), ("r2_a", 0.51, 2), ("r2_six", 0.59, 2),
        ("n_low", 37, 0), ("cell_hi_lo_n", 85, 0), ("cell_hi_lo_low_pct", 42, 0),
        ("near_td_q5", 0.73, 2), ("near_td_q10", 0.97, 2), ("near_td_q25", 1.34, 2), ("near_td_q50", 1.47, 2),
        ("near_td_lt_n", 36, 0), ("near_td_lt_low", 36, 0),
        ("low_gate_reward_level", 0.94, 2), ("low_gate_punish_drop", 1.13, 2), ("low_gate_punish_assoc", 1.20, 2),
        ("unreachable", 58, 0), ("unreachable_low", 36, 0), ("selfbase_floor", 7, 0),
        ("point_near_reward_level", 1.457, 3), ("point_near_punish_drop", 1.469, 3),
        ("point_near_reward_assoc", 3.098, 3), ("point_near_punish_assoc", 1.540, 3),
        ("point_a", 32.27, 2), ("point_b", 6.11, 2), ("point_reach_max", 1.81, 2),
        ("cA_q5", 16.2, 1), ("cA_q50", 20.0, 1), ("cA_q90", 24.0, 1), ("cA_q95", 24.5, 1), ("cA_q99", 26.8, 1),
        ("cA_q100", 30.5, 1), ("cP_q5", 33.2, 1), ("cP_q50", 43.0, 1), ("cP_q90", 52.0, 1), ("cP_q95", 55.0, 1),
        ("cP_q99", 59.8, 1), ("cP_q100", 67.8, 1), ("thr_both_n", 81, 0), ("thr_both_low", 36, 0),
        ("cfB_28_55_n_ge", 5, 0), ("cfB_36_70_n_ge", 6, 0), ("sub_20_43", 0.154, 3), ("sub_24_50", 0.880, 3), ("sub_28_55", 0.874, 3),
        ("psize_6", 0.346, 3), ("psize_12", 0.837, 3), ("psize_24", 0.944, 3),
        ("psize_6_base", 0.897, 3), ("psize_12_base", 0.932, 3), ("psize_24_base", 0.944, 3),
        ("psize_6_below", 9, 0), ("psize_12_below", 1, 0), ("psize_24_below", 0, 0),
        ("w_drift_corr", -0.44, 2), ("w_drift_mean", -11.16, 2), ("w_drift_sd", 7.26, 2),
        ("w_p_drift_mean", -2.07, 2), ("w_p_drift_sd", 7.06, 2),
        ("y_drift_sd", 5.89, 2), ("y_drift_se", 2.41, 2), ("y_drift_boot_sd", 2.48, 2))
    # ---- 0c′ sensitivity (Z.9.2 P1-3) ------------------------------------------------------------------------------
    sens_seed: int = 86_000_000
    sens_deltas: tuple = (0.0, -2.41, -4.82)
    sens_ns: tuple = (12, 24)
    sens_draws: int = 100
    sens_reps: int = 400
    sens_cont_delta: int = 1                                     # index of −2.41 (−1 SE) in sens_deltas
    sens_qs: tuple = (5.0, 10.0, 50.0)
    plan_bar: float = 0.80                                       # the continue rules' bar (= p_power)
    # ---- 0d split (Z.3 + Z.9.2 P0-1) and the key pre-check -----------------------------------------------------------
    split_seed: int = 79_000_000
    strata: tuple = ("b", "a")
    halves: tuple = ("C", "P")
    j_min: int = 12
    n_star: tuple = (12, 24)                                     # n* = 24 if J_max ≥ 24 else 12
    # ---- budget (Z.7 예산, Z.9.2 P2-10) ------------------------------------------------------------------------------
    budget_h: float = 24.0
    records_budget_h: float = 6.0
    cost_margin: float = 1.3
    s_per_h: float = 3600.0
    # ---- 0a: tests, timing, environment ------------------------------------------------------------------------------
    tests_log: str = "results/z/tests_0a.log"
    tests_ok_line: str = "exit 0"
    timing_reps: int = 50
    timing_pairs: int = 16
    thread_env: tuple = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    workers: int = 16
    pool_timeout_s: float = 3600.0
    # ---- paths ---------------------------------------------------------------------------------------------------------
    summary: str = "results/summary/z_learning.json"
    draws_summary: str = "results/summary/z_oc_draws.json"
    raw_dir: str = "results/z"
    progress_dir: str = "results/z/progress"
    stage0_detail: str = "results/z/stage0.json"
    ydiag_detail: str = "results/z/ydiag.json"
    sens_detail: str = "results/z/sens.json"
    split_detail: str = "results/z/split.json"
    archive_root: str = "~/flymon-archive/z"
    y_cells_archive: str = "y_cells"
    cli_print_chars: int = 2000
    # ================================================================ declared now, read by the later-orders plan
    pilot_probe_seed0: int = 80_000_000                          # + j × 4_000 + f × 100 + k (j < 32, Z.9.2 P3-13)
    pilot_train_seed0: int = 81_000_000                          # + j × 40_000 + f × 1_000 + t
    pilot_j_max: int = 32
    oc_seed: int = 85_000_000
    smoke_probe_seed0: int = 85_100_000
    smoke_train_seed0: int = 85_110_000
    smoke_oracle_seed0: int = 85_150_000
    precheck_seed: int = 85_200_000
    compare_seed: int = 85_300_000
    reconfirm_seed: int = 85_400_000
    small_boot_seed: int = 85_500_000
    records_seed: int = 85_600_000
    se_max: float = 1.62
    yield_pseudo: tuple = (2, 3)                                 # ρ_Z = (a_h + 2) / (h + 3) (Z.9.2 P1-6)
    rpre_reps: int = 400
    reconfirm_max: int = 5
    seed_blocks: tuple = ((79_000_000, 79_100_000), (80_000_000, 81_000_000), (81_000_000, 83_000_000),
                          (85_000_000, 85_100_000), (85_100_000, 85_200_000), (85_200_000, 85_300_000),
                          (85_300_000, 85_400_000), (85_400_000, 85_500_000), (85_500_000, 85_600_000),
                          (85_600_000, 85_700_000), (86_000_000, 86_100_000))   # [lo, hi) per Z.2 / Z.9.2 block


SPEC = ZSpec()
