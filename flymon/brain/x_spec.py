"""Every number of spec appendix X as amended by X.9.1 (X.9.1 > X.0–X.9; for W's rules X keeps W.9.10 > … > W.0) —
phase A only (X.9.1.1: orders 0 · 0a). Phase B adds its numbers here and changes none of these.
- XSpec is a plain frozen dataclass. W_SHARED lists the fields W's functions read (w_oc.calibrate / simulate /
  simple_normal, w_verdict): they carry W's values unchanged (X.2; test_x_spec pins them to w_spec.SPEC), so an
  XSpec can be passed where w_oc expects a spec. X's generator roots are its own.
- Seeds (X.2, X.9.1.1): X OC root 43_000_000 (phase A: X.4.6's synthetic validation, P2-6), precheck root 43_200_000
  (block 43_200_000–43_299_999), record-only diagnostics root 43_300_000 (block 43_300_000–43_399_999); every stream is
  SeedSequence([root, tag, …]). Phase B adds the probe 44_000_000 · training 46_000_000 · smoke 43_1xx_xxx blocks. The
  oracle seeds 24_700_xxx and W's root 42_000_000 (the W calibration diagnosis) are read from w_spec where used, never
  restated here (the seed collectors would see a collision with W).
- Paths: results/x/ (git-ignored), results/summary/x_learning.json (tracked), ~/flymon-archive/x."""
from __future__ import annotations

from dataclasses import dataclass

W_SHARED = ("bar", "band_width", "naive_max", "mech_min", "round_digits", "min_gate_pairs", "q_grid", "k_grid",
            "f_min", "f_max", "k_min", "k_cap", "envelope", "envelope_solo_from", "d_power", "p_power", "d_false",
            "p_false", "oc_reps", "cal_reps", "cal_tol", "cal_iter", "boot_draws", "boot_reps", "boot_level",
            "cluster_grid", "record_dprimes", "oc_chunk", "cal_floor_rule", "synth_reps", "synth_null_max",
            "synth_big_min", "synth_big_dprime", "synth_drift_dprime_min", "simple_normal_fs", "simple_normal_reps",
            "budget_h")


@dataclass(frozen=True)
class XSpec:
    # ---- W's numbers X keeps (X.2; equal to w_spec.SPEC, test_x_spec) ----------------------------------------------
    bar: float = 1.0
    band_width: float = 0.2
    naive_max: float = 0.5
    mech_min: float = 0.75
    round_digits: int = 9
    min_gate_pairs: int = 4
    q_grid: tuple = (0.5, 0.625, 0.75)
    k_grid: tuple = (8, 16)
    f_min: int = 8
    f_max: int = 32
    k_min: int = 4
    k_cap: int = 8
    envelope: int = 3
    envelope_solo_from: int = 29
    d_power: float = 1.5
    p_power: float = 0.80
    d_false: float = 0.5
    p_false: float = 0.05
    oc_reps: int = 4000
    cal_reps: int = 2000
    cal_tol: float = 0.02
    cal_iter: int = 40
    boot_draws: int = 200
    boot_reps: int = 400
    boot_level: float = 0.95
    cluster_grid: tuple = (0.0, 0.5, 1.0)
    record_dprimes: tuple = (0.5, 1.0, 1.5, 2.0)
    oc_chunk: int = 50
    cal_floor_rule: str = "zero"
    synth_reps: int = 1000
    synth_null_max: float = 0.02
    synth_big_min: float = 0.98
    synth_big_dprime: float = 4.0
    synth_drift_dprime_min: float = 1.5
    simple_normal_fs: tuple = (8, 16, 24, 32)
    simple_normal_reps: int = 2000
    budget_h: float = 24.0                          # X's own 24 h (X.8); W's spend not included
    # ---- the set rule (X.3, Q1 + Q2) ---------------------------------------------------------------------------------
    p_set_grid: tuple = (0.5, 0.625, 0.75, 0.875, 1.0)
    min_pass_pairs: int = 3
    # ---- calibration failure (X.9.1.2) -------------------------------------------------------------------------------
    bracket_mult: float = 4.0                       # W's hi_a / hi_b × 4 for every X calibration
    w_bracket_mult: float = 1.0                     # the W reproduction (diagnosis) only
    cal_retry_iter: int = 160                       # no_convergence → once more with 160 steps
    chol_jitter: float = 1e-9                       # w_oc's Cholesky jitter (x_oc's simulator copy)
    w_rejection_tries: int = 200                    # w_oc._pair_bases's rounds (x_oc.pair_bases = W at this value)
    # ---- synthetic fixtures (W.9.9 P2-11's structure, as w_oc.synthetic_validation) ---------------------------------
    synth_base: tuple = (40.0, 90.0)
    synth_sd: float = 4.0
    synth_corr: float = 0.8
    synth_learn: tuple = (20.0, 10.0)
    synth_drift: float = 15.0
    p26_reps: int = 200                             # stage0's P2-6 check on θ̂ (experiments per g)
    # ---- seeds -------------------------------------------------------------------------------------------------------
    oc_seed: int = 43_000_000
    precheck_seed: int = 43_200_000
    diag_seed: int = 43_300_000
    # ---- the precheck (X.9.1.1) and its records (X.4.5 at θ̂) --------------------------------------------------------
    precheck_reps: int = 4000
    het_scales: tuple = (0.5, 1.0, 2.0)             # Σ_v × (X.4.5)
    het_low_dprime: float = 0.5                     # W's heterogeneous scenarios: one / half pairs at 0.5, rest 1.5
    het_all_dprime: float = 1.0                     # W's "all 1.0"
    # ---- variants and pilot facts (X.4.4, X.0, X.9.1.3) -------------------------------------------------------------
    variants: tuple = ("V0", "V1", "V2")
    v1_exclude_from: float = 5.0                    # V1 drops |d′| ≥ 5
    min_variant_pairs: int = 6                      # fewer → diagonal only, V0's correlations
    v1_pairs: int = 7
    balanced_pairs: tuple = ("a|4|Rock Slide|Strength", "b|2|Mega Drain vs Machamp|Mega Drain vs Tentacruel",
                             "b|10|Surf vs Slowbro|Surf vs Venusaur")
    resid_quantiles: tuple = (5.0, 25.0, 50.0, 75.0, 95.0)
    # ---- record-only diagnostics (X.9.1.4) ---------------------------------------------------------------------------
    diag_reps: int = 4000
    diag_pairs: int = 16
    diag_f_max: int = 64
    diag_fs: tuple = (8, 16, 32, 64)                # (i)
    diag_k_max: int = 16                            # (iii)
    diag_k_los: tuple = (4, 6, 8, 10, 12)
    diag_k_width: int = 4                           # k range [k_lo, k_lo + 4]
    diag_floor_share: float = 0.10                  # floor "caught" when ≥ 10 % of draws truncate
    diag_tries: int = 2000
    diag_fill_flag: float = 0.01
    diag_levels: tuple = (0.0, 25.0, 50.0, 75.0)    # F2(ℓ) percentiles
    diag_chunk: int = 20
    diag_pair_chunk: int = 100
    # ---- W facts re-checked (X.2, X.4.3, X.5 1 in part) -------------------------------------------------------------
    w_measure_key: str = "761274e0ed09f9ec7043079e477588b4a2eef76c8076614782542573b2138df1"
    w_pipeline_key: str = "9636cf4c7f868338f512cd7cf1a722027bd842c37528188859c14bebb04a2853"
    w_measure_file_sha: str = "e23009146ee04e3f6f328445a51685f8f045f6bb194749bf4ebd8f80c6e2be63"
    w_oc_detail_sha: str = "7a0a65f3f6b969d47b29f9a41ae45f2d1cfc4bf9bb76d028bf7172a9a6fbeb77"
    w_commits: tuple = (("pilot", "5fbc4c8"), ("oc", "f30ae35"))
    w_summary_commit: str = "f30ae35"
    w_boot_fail_total: int = 29
    w_boot_fail_power: int = 14
    w_boot_fail_false: int = 18
    w_boot_fail_both: int = 3
    w_pilot_pairs: int = 16
    w_pilot_units: int = 384
    # ---- paths and CLI -----------------------------------------------------------------------------------------------
    summary: str = "results/summary/x_learning.json"
    raw_dir: str = "results/x"
    wcal_detail: str = "results/x/w_cal_diag.json"
    precheck_detail: str = "results/x/precheck.json"
    diag_detail: str = "results/x/precheck_diag.json"
    progress_dir: str = "results/x/progress"
    archive_root: str = "~/flymon-archive/x"
    w_summary: str = "results/summary/w_learning.json"
    w_oc_detail: str = "results/w/oc.json"
    cli_print_chars: int = 2000


SPEC = XSpec()
