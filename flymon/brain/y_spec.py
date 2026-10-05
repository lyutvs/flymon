"""Every number of spec appendix Y used so far — step 0p only (Y.7 0p-a). Phase A appends its numbers here and changes
none of these (a change means recomputing the 0p blocks with the same seeds, bit for bit).
- Thresholds: Y.3.2 (balance < 0.5, c_A 20, c_P 43, 9-digit rounding), Y.3.3 1 (lenient 1.0 · 16 · 34.4), the record-
  only columns of Y.7 0p-c: F2(0) / F2(25) exactly as X's results/x/precheck_diag.json filters (spec prints 8.9 · 51.6 /
  30.7 · 94.4), the user's 31 and one-sided −1. Testable = V_SPEC.testable_min (2.0), not restated.
- Main set (Y.2): w_pairs.w_set's digest_keys (Reading 1 of the 0p plan), counts and last turn.
- Seeds (Y.2, author's reading 2): Y's roots only; the oracle seeds 24_700_xxx are W's (w_spec.SPEC.oracle_seeds()) and
  never restated here (the seed collectors would see a collision with W).
- Paths: results/y/ (git-ignored), results/summary/y_learning.json (tracked), ~/flymon-archive/y.
- Y red-team Y.9.2 (appended under P2-11, which allows y_spec / y_rules additions for the rules it changed): P1-7's
  yield factor c = 1 / (2/3) = 1.5 with its basis (W pilot 16 pairs -> 3 oracle-lenient passes -> 2 pilot-strict
  passes), P1-4's summary quantiles of the lenient-pass oracle levels, and the facts Y.8's sentences print (pilot
  candidates W 3 · V 4, n_Σ minimum 5, the widened pool's generator turns, the lever's description) so that every number
  in a sentence comes from here. Y.9.2's new seed roots (78_000_000 · 78_100_000 · 78_200_000) are phase A's to append.
Phase A (Y.7 orders 0–5; appended, the 0p fields above unchanged): W / X numbers the imported functions read (w_oc,
w_verdict, x_oc, x_verdict) restated under their own names and pinned to x_spec.SPEC by test_y_spec (k_cap = 10, the
largest k_hi, is Y's); Y.6.2 + Y.9.2 P1-2's calibration numbers; Y.6.3's 2000 rounds and P1-5's 1 % fill; the
precheck, P2-12, P2-10 / Y.0 threshold numbers (0.25 grid: plan Reading 3); phase B's ×1.3 / reconfirmation numbers
(fixed now, Y.9.2 P2-8 / P2-9); Y.3.4's reps; the pilot's candidates and printed facts (Y.4, record-only checks);
Y.9.2's seed roots 78_000_000 (phase B reconfirmation) · 78_100_000 (P2-12) · 78_200_000 (R-V · R-pre · thresholds);
the calibration fixtures' synthetic numbers (Y.6.6, Y.9.2 P1-2 (d)); X's facts and the phase-A detail paths.
Phase B (Y.7 orders 6–12; appended after records_seed, nothing above changed): detail / cache / marker paths, the
bootstrap pool's thread variables, the percentile scale, the smoke's pair / flies / seed block, seconds per hour and
the unit-cost fields 7a / 8a take the maximum of (phase-B plan Readings 2 · 7)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class YSpec:
    # ---- the filter (Y.3.2) and the oracle lenient pre-filter (Y.3.3 1) ---------------------------------------------
    naive_max: float = 0.5
    c_a: float = 20.0
    c_p: float = 43.0
    lenient_d: float = 1.0
    lenient_a: float = 16.0                         # 0.8 c_A
    lenient_p: float = 34.4                         # 0.8 c_P
    round_digits: int = 9
    # ---- record-only count-table columns (Y.7 0p-c) ------------------------------------------------------------------
    f2_0: tuple = (8.875, 51.640625)
    f2_25: tuple = (30.7421875, 94.421875)
    user_a: float = 31.0
    user_onesided_d: float = -1.0
    # ---- the set rule's k ranges (Y.5); 0p uses min k_lo for the early STOP_FEW_PAIRS -------------------------------
    k_ranges: tuple = ((4, 8), (6, 10))
    # ---- Y.9.2 P1-7: a k range stays only if the lenient-pass count N_len ≥ k_lo × c ([4, 8] ≥ 6, [6, 10] ≥ 9) -------
    yield_c: float = 1.5
    yield_basis_n: int = 16                         # W pilot pairs, all oracle-testable (V oracle cache)
    yield_basis_pairs: tuple = (                    # (pair, oracle-lenient pass -> pilot-strict pass?)
        ("a|4 Rock Slide|Strength", True), ("b|10 Surf vs Slowbro", True), ("b|4 Rock Slide vs Venusaur", False))
    # ---- Y.9.2 P1-4: summary quantiles of the lenient-pass pairs' L_A^or(X) · L_P^or(X) (record only) ---------------
    level_quantiles: tuple = (0.0, 0.25, 0.5, 0.75, 1.0)
    # ---- facts Y.8's sentences print (Y.1, Y.4, W.2) ------------------------------------------------------------------
    pilot_w: int = 3                                # W pilot balanced pairs among the Y pilot candidates
    pilot_v: int = 4                                # V set candidates among them
    pilot_min_sigma: int = 5                        # n_Σ (after the Earthquake merge) minimum, Y.9.2 P1-3
    pool_turns: tuple = (306, 1985)                 # the widened pool's generator turns (W_SPEC first / last turn)
    lever_text: str = ("C3, APL→MBON05 2간선 + MBON05→MBON09/MBON11/MBON01 11간선 제거, E-grid k2-norm s 1.0, "
                       "엔진별 z")
    # ---- X detail the F2 columns are read from (Y.2 X 블록) -----------------------------------------------------------
    x_precheck_diag: str = "results/x/precheck_diag.json"
    x_precheck_diag_sha256: str = "33f83895f432ad6400166540ee29b6d081a3122d67e1d06715d0822b5f8839ae"
    # ---- reused keys and the main set (Y.2, Y.7 0p-b) -----------------------------------------------------------------
    u_measure_key: str = "8a4e09302f7f2f7e7d69f3540d8fc392da582cb1246bfebeb150348776e26523"
    w_measure_key: str = "761274e0ed09f9ec7043079e477588b4a2eef76c8076614782542573b2138df1"
    main_digest_keys: str = "65dbf001a61ea7f484cf4e61a212a3973fde8709c5225e15e2221b0777a5712a"
    main_n_b: int = 167
    main_n_a: int = 82
    main_last_turn: int = 1967
    # ---- seeds (Y.2) --------------------------------------------------------------------------------------------------
    pilot_probe_seed0: int = 60_000_000             # + j × 4_000 + f × 100 + k
    pilot_train_seed0: int = 61_000_000             # + j × 40_000 + f × 1_000 + t
    probe_seed0: int = 62_000_000                   # + c × 4_000 + f × 100 + k (f, k < 32)
    train_seed0: int = 64_000_000                   # + c × 40_000 + f × 1_000 + t
    oc_seed: int = 77_000_000
    smoke_probe_seed0: int = 77_100_000
    smoke_train_seed0: int = 77_110_000
    smoke_oracle_seed0: int = 77_150_000
    precheck_seed: int = 77_200_000
    compare_seed: int = 77_300_000
    # ---- budget, pool, paths, CLI -------------------------------------------------------------------------------------
    budget_h: float = 24.0
    workers: int = 16
    pool_timeout_s: float = 3600.0
    summary: str = "results/summary/y_learning.json"
    raw_dir: str = "results/y"
    cache_dir: str = "results/y/cache"
    oracle_detail: str = "results/y/oracle.json"
    progress_dir: str = "results/y/progress"
    archive_root: str = "~/flymon-archive/y"
    cli_print_chars: int = 2000
    # ================================================================ phase A (Y.7 orders 0–5); 0p above unchanged
    # ---- W / X numbers the imported functions read (pinned to x_spec.SPEC; k_cap is Y's) ----------------------------
    bar: float = 1.0
    band_width: float = 0.2
    mech_min: float = 0.75
    min_gate_pairs: int = 4
    min_pass_pairs: int = 3
    p_set_grid: tuple = (0.5, 0.625, 0.75, 0.875, 1.0)
    q_grid: tuple = (0.5, 0.625, 0.75)
    k_grid: tuple = (8, 16)
    f_min: int = 8
    f_max: int = 32
    k_min: int = 4
    k_cap: int = 10                                 # the largest k_hi of k_ranges (X's is 8)
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
    chol_jitter: float = 1e-9
    w_rejection_tries: int = 200
    synth_reps: int = 1000
    synth_null_max: float = 0.02
    synth_big_min: float = 0.98
    synth_big_dprime: float = 4.0
    synth_drift_dprime_min: float = 1.5
    simple_normal_fs: tuple = (8, 16, 24, 32)
    simple_normal_reps: int = 2000
    synth_base: tuple = (40.0, 90.0)
    synth_sd: float = 4.0
    synth_corr: float = 0.8
    synth_learn: tuple = (20.0, 10.0)
    synth_drift: float = 15.0
    het_scales: tuple = (0.5, 1.0, 2.0)
    het_low_dprime: float = 0.5
    het_all_dprime: float = 1.0
    precheck_reps: int = 4000
    p26_reps: int = 200                             # stage0's bit-identity run on W θ̂ (experiments per g)
    # ---- Y calibration (Y.6.2 as amended by Y.9.2 P1-2) --------------------------------------------------------------
    grid_steps: int = 200                           # coarse h = hi_W / 200
    widen: tuple = (1, 2, 4)                        # ×1 → ×2 → ×4 (stage end = grid_steps × mult)
    refine_delta: float = 0.25                      # refine a coarse cell whose larger end is ≥ −0.25
    refine_parts: int = 32                          # 32 equal parts per refined cell
    knob_tol: float = 1e-3                          # knob width ending the bisection (spikes)
    coarse_step_flag: float = 0.1                   # d_hi − d_lo > 0.1 → coarse_step (mark only)
    # ---- generator (Y.6.3, Y.9.2 P1-4 · P1-5 · P3-14) ----------------------------------------------------------------
    tries: int = 2000
    fill_max: float = 0.01
    # ---- order 5 extras (Y.9.2 P2-12 · P2-10, Y.0) and fixture 8's Monte-Carlo -----------------------------------
    small_boot_draws: int = 50
    floor_share: float = 0.10
    floor_resid_n: int = 1024
    thr_step: float = 0.25
    thr_expected: tuple = (19.25, 42.0)             # Y.0's values on W θ̂ with X's power calibration
    mix_reps: int = 4000
    sigma2_scale: float = 2.0                       # records variant R-Σ2 (Y.6.4)
    # ---- phase B numbers fixed now (Y.5, Y.9.2 P2-8 · P2-9); select_y / tests use cost_margin in phase A ------------
    cost_margin: float = 1.3
    reconfirm_reps: int = 1600
    reconfirm_max: int = 5
    # ---- the filter-candidate comparison (Y.3.4, record only) --------------------------------------------------------
    compare_reps: int = 4000
    # ---- the pilot (Y.4) ---------------------------------------------------------------------------------------------
    pilot_flies: int = 8
    pilot_probes: int = 8
    pilot_w_pairs: tuple = ("a|4|Rock Slide|Strength", "b|2|Mega Drain vs Machamp|Mega Drain vs Tentacruel",
                            "b|10|Surf vs Slowbro|Surf vs Venusaur")
    pilot_v_pairs: tuple = ("b|17|Thunderbolt vs Nidoran-M|Thunderbolt vs Clefairy", "a|218|Earthquake|Strength",
                            "a|305|Earthquake|Rock Slide", "a|223|Surf|Earthquake")
    pilot_v_oracle: tuple = ((-0.146, 48.0, 128.5), (0.170, 46.0, 109.0), (0.214, 46.0, 109.0),
                             (-0.884, 42.0, 137.0))     # Y.4's printed V oracle (d_pre, L_A^or, L_P^or)
    pilot_w_naive: tuple = ((0.077, 31.0, 107.0), (-0.052, 38.0, 114.5), (-0.318, 34.5, 121.0))
    fact_digits: int = 3
    flip_strict: tuple = ("b|10|Surf vs Slowbro|Surf vs Venusaur",
                          "b|2|Mega Drain vs Machamp|Mega Drain vs Tentacruel")      # Y.9.2 P1-6 facts
    flip_lenient: tuple = ("b|4|Rock Slide vs Venusaur|Rock Slide vs Charizard",
                           "b|2|Mega Drain vs Machamp|Mega Drain vs Tentacruel")
    # ---- calibration fixtures (Y.6.6 + Y.9.2 P1-2 (d)); synthetic f in d′ units relative to the target ------------
    fix_hi_w: float = 100.0                         # hi_W of every fixture (h = 0.5)
    fix_target: float = 1.5
    fix_step: tuple = (37.3, -0.2, 0.1)             # 1: step at 37.3 from −0.2 to +0.1 (a 0.3 jump)
    fix_cross: tuple = (30.0, 120.0, 250.0, 380.0)  # 2: up, down, up, down (below the target at the ×4 end 400)
    fix_slope: float = 0.1                          # d′ per spike of the linear pieces (2, 3)
    fix_x2: float = 150.0                           # 3: the only crossing (inside ×2)
    fix_below: float = -1.0                         # 3, 4: f far below the target
    fix_above: float = 0.05                         # 5: f(0) = +0.05 > cal_tol
    fix_narrow: tuple = (9.7, 0.1, 0.05, -0.1, 40.0, 0.2)   # 7: window [9.7, 9.7 + h·0.1] at +0.05, else −0.1;
                                                            #    coarse crossing at 40 to +0.2
    fix_b_steps: tuple = (20.0, 30.0, -0.3, 0.05)   # 8: b's step at 20 in a_lo's state (−0.2 → +0.1), at 30 in
                                                    #    a_hi's (−0.3 → +0.05)
    fix_exact: float = 1e-12                        # fixture 1 / 9 tolerance
    fix_mean: float = 1e-9                          # fixture 8 mean tolerance
    fix_se: float = 3.0                             # fixture 8 Monte-Carlo standard errors
    fix_design: tuple = (1.0, 0.75, 8, 8)           # fixture 8's one design (p_set, q, K, F)
    set_fixtures: tuple = (((4, 2, 0, 0.5), "FAIL"), ((4, 3, 0, 0.75), "PASS"), ((4, 3, 0, 0.875), "FAIL"),
                           ((6, 3, 0, 0.625), "FAIL"), ((5, 3, 1, 0.5), "PASS"), ((4, 2, 2, 0.5), "UNDECIDED"),
                           ((3, 3, 1, 0.5), "UNDECIDED"), ((4, 1, 1, 0.5), "FAIL"), ((8, 8, 0, 1.0), "PASS"),
                           ((8, 7, 0, 1.0), "FAIL"), ((4, 3, 2, 0.75), "UNDECIDED"), ((3, 0, 1, 0.5), "UNDECIDED"),
                           ((10, 5, 0, 0.5), "PASS"), ((10, 9, 0, 1.0), "FAIL"))   # X.4.6 + two k = 10 cases
    m_table: tuple = ((0.5, (3, 3, 3, 4, 4, 5, 5)), (0.625, (3, 4, 4, 5, 5, 6, 7)), (0.75, (3, 4, 5, 6, 6, 7, 8)),
                      (0.875, (4, 5, 6, 7, 7, 8, 9)), (1.0, (4, 5, 6, 7, 8, 9, 10)))   # Y.5, b = 0, k 4–10
    block_list_max: int = 20                        # a block lists at most 20 passing designs (all in the detail)
    # ---- X facts (Y.7 1) and phase-A details -------------------------------------------------------------------------
    x_summary: str = "results/summary/x_learning.json"
    x_commits: tuple = (("stage0", "868771a"), ("precheck", "4b81035"))
    w_commits: tuple = (("reuse", "744d1bc"), ("path", "5620f95"), ("pilot", "5fbc4c8"), ("oc", "f30ae35"))   # Y.7 1
    stage0_detail: str = "results/y/stage0.json"
    pilot_detail: str = "results/y/pilot.json"
    precheck_detail: str = "results/y/precheck.json"
    reconfirm_seed: int = 78_000_000
    small_boot_seed: int = 78_100_000
    records_seed: int = 78_200_000
    # ================================================================ phase B (Y.7 orders 6–12); everything above unchanged
    # ---- details, caches and markers (results/y/, git-ignored; y_store guards the writes) --------------------------
    oc_detail: str = "results/y/oc.json"
    smoke_detail: str = "results/y/smoke.json"
    gates_detail: str = "results/y/gates.json"
    learn_detail: str = "results/y/learn.json"
    band_detail: str = "results/y/band.json"
    records_detail: str = "results/y/records.json"
    judge_detail: str = "results/y/judge.json"
    smoke_cache_dir: str = "results/y/smoke/cache"
    judge_marker: str = "results/y/judge_read.json"
    reread_marker: str = "results/y/judge_reread.json"
    done_marker: str = "results/y/judge_done.json"
    # ---- the bootstrap's process pool (plan Reading 2): one draw per task, BLAS single-threaded in the workers ----------
    thread_env: tuple = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    pct_scale: float = 100.0                        # numpy percentiles: lower 100·(1 − level), upper 100·level
    # ---- the smoke (Y.7 7): Y pilot pair j 0 (b|17), one fly, the smoke seed block 77_1xx_xxx ------------------------
    smoke_pair: int = 0
    smoke_flies: int = 1
    smoke_block: tuple = (77_100_000, 77_200_000)   # Y.2's smoke block [77_100_000, 77_199_999]
    # ---- the budget arithmetic (Y.5, Y.7 7a / 8a, Y.9.2 P2-9) --------------------------------------------------------
    s_per_h: float = 3600.0
    cost_fields: tuple = ("trial_s", "presentation_s", "oracle_round_s")   # max(pilot, smoke) per field (Reading 7)


SPEC = YSpec()
