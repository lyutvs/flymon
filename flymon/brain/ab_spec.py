"""Every number of spec appendix AB (lines 6578–7011; the body AB.0–AB.9.1 is the rule, AB.9.2 lists the red-team
changes). Y's filter numbers and V's KC numbers are read from y_spec / v_spec and never restated; the facts AB checks
against committed blocks (commits, digests, sha256, counts) are restated here because AB.2 / AB.3 declare them.
- Seeds (AB.2): AB's own roots carry "seed" in their field names so the collectors in test_p_spec see them; V's KC
  strength seeds (24_002_000 + i) are reused on purpose and come from v_spec.SPEC.kc_seeds().
- Streams (AB.2 "흐름 만들기"): every stream is aa_estimate.stream(root, *tags); the five test vectors are sealed.
- The futility gate (AB.7 0f, AB.9.3): its model constants, reps, threshold and grid are sealed (seal_names).
- seal_fields(): the numbers the seal (AB.7 0d) hashes — AB.9.1's fixed list as far as ab_spec holds it (Y's filter
  and V's KC numbers are sealed through their own files); seal_files: the AB rule files and every flymon file they
  import (the closure is added by ab_runner.seal_closure and checked by the runner tests)."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

SD4 = (0.5, 1.0, 2.0, 3.0)


@dataclass(frozen=True)
class ABSpec:
    # ---- the chain (AB.7 순서 요약; plan Reading 1) --------------------------------------------------------------
    stages: tuple = ("stage0", "reuse", "generate", "seal_code", "cal_gate", "futility", "kc_input", "set", "smoke",
                     "oracle", "screen", "calibrate", "learn", "verdict", "records")
    stage_labels: tuple = (("stage0", "0a"), ("reuse", "0b"), ("generate", "0c"), ("seal_code", "0d"),
                           ("cal_gate", "0e"), ("futility", "0f"), ("kc_input", "3"), ("set", "4"), ("smoke", "5"), ("oracle", "6"),
                           ("screen", "7"), ("calibrate", "7b"), ("learn", "8"), ("verdict", "9"), ("records", "10"))
    fly_stages: tuple = ("kc_input", "smoke", "oracle", "screen", "learn")      # FlyPool (engine) stages
    first_measure: str = "kc_input"                                           # 2nd-gen first measurement (AB.7 3)
    records_stages: tuple = ("records",)                                      # the records ledger (AB.7 예산)
    decl_commit: str = "ea408a2"                                              # "이 부록 커밋" (code = HEAD 5e3eb7e)
    # ---- reuse facts (AB.2 재사용 대조, order 0b) ----------------------------------------------------------------
    v_commits: tuple = (("z", "928eaad"), ("kc_input", "7dc199d"), ("set", "cf0b3b2"), ("judge", "a279a56"),
                        ("kc_band", "69f7b2d"))
    kc_input_sha256: str = "f197492aca8e1b23721a0c121f80255e019264e3d90d675ef4543a56dff2b85d"
    w_digest_keys: str = "65dbf001a61ea7f484cf4e61a212a3973fde8709c5225e15e2221b0777a5712a"
    w_n_b: int = 167
    w_n_a: int = 82
    w_last_turn: int = 1967
    aa_blocks: tuple = (("stage0", "e7a1dd9"), ("reuse", "7a62d4b"), ("smoke", "9490fc5"), ("seal_code", "25b8a24"),
                        ("screen", "8f097d5"), ("coverage", "d1bb46c"), ("learn", "621e220"), ("estimate", "7e475e8"),
                        ("records", "51a4b9d"))
    aa_trained: int = 31
    w_pilot_pairs: int = 16
    y_pilot_pairs: int = 7
    y_pilot_split: tuple = (("even", 3), ("v_set", 4))
    v_set_rows: int = 64
    lv_odour_n: int = 108
    reuse_globs: tuple = ("t_*.py", "v_*.py", "w_*.py", "y_*.py", "aa_*.py")
    reuse_extra: tuple = ("flymon/brain/z_split.py",)
    # ---- the generator (AB.3; declared values, key level) ---------------------------------------------------------
    gen2_num: tuple = (152, 251)
    pool_types: tuple = ("DRAGON", "FIGHTING", "FIRE", "FLYING", "GRASS", "GROUND", "ICE", "NORMAL", "POISON",
                         "PSYCHIC", "ROCK", "WATER")
    n_opp: int = 73
    n_opp_type_out: int = 27
    n_new_type_species: int = 15
    new_type_sets: tuple = ("DRAGON+WATER", "FIRE+ROCK", "FLYING+GRASS", "FLYING+GROUND", "FLYING+PSYCHIC",
                            "GROUND+ICE", "GROUND+WATER", "NORMAL+PSYCHIC", "ROCK")
    n_combos: int = 1168
    shuffle_seed: int = 20261007
    reasons: tuple = ("cap", "collision", "e1", "used", "glom_dup", "in_set", "pool_only", "lv_odour", "kc_input")
    decl_skips: tuple = (("used", 2765), ("in_set", 1994), ("pool_only", 242), ("cap", 138), ("collision", 0),
                         ("e1", 0), ("glom_dup", 0))                         # the walk without lv_odour (Reading 2)
    decl_lv: tuple = (("a", 9), ("b", 118))                                  # rows of that walk with an lv odour
    decl_lv_kc_out: tuple = (("a", 9), ("b", 18))
    decl_pre_kc: tuple = (("a", 143), ("b", 66))                             # KC 전 세트 209
    decl_turns: tuple = (1, 1150)
    decl_last_turn: tuple = (("a", 1150), ("b", 972))
    decl_odours: tuple = (("all", 107), ("v_cache", 36), ("new", 71))
    decl_vcache_drop: tuple = (("a", 4), ("b", 21))
    decl_vcache_drop_odours: int = 8
    decl_post_kc_max: tuple = (("a", 139), ("b", 45))                        # KC 뒤 상한 184
    decl_needs_kc: tuple = (("a", 139), ("b", 40))
    decl_vcache_pass: tuple = (("a", 0), ("b", 5))
    decl_a_x_groups: tuple = (("Earthquake", 39), ("Surf", 35), ("Psychic", 24), ("Flamethrower", 18),
                              ("Rock Slide", 9), ("Mega Drain", 9), ("Thunderbolt", 5))
    decl_a_t_groups: tuple = (("DRAGON+WATER", 16), ("FIRE+ROCK", 16), ("FLYING+GRASS", 16), ("FLYING+GROUND", 16),
                              ("FLYING+PSYCHIC", 16), ("GROUND+ICE", 16), ("GROUND+WATER", 16),
                              ("NORMAL+PSYCHIC", 16), ("ROCK", 11))
    decl_b_x_groups: int = 45
    decl_b_t_groups: int = 13
    decl_gen1_same_keys: int = 23                                            # record (AB.3 기록)
    selftest_first_turn: int = 306                                           # W.2's first turn
    kc_repro: tuple = ("ELECTRIC|FIGHTING", "ELECTRIC|FLYING+ICE", "ELECTRIC|GROUND", "ELECTRIC|GROUND+ROCK")
    # ---- the design (AB.1, AB.7 5, 8) -----------------------------------------------------------------------------
    flies: int = 8                                                           # F
    probes: int = 8                                                          # K
    smoke_pair: str = "b|17|Thunderbolt vs Nidoran-M|Thunderbolt vs Clefairy"   # Y pilot pair b|17 (AB.7 5)
    smoke_flies: int = 1
    noplast_pairs: int = 2
    noplast_flies: int = 2
    n_oracle_seeds: int = 8
    # ---- AB seed roots (AB.2 AB 시드 블록) ------------------------------------------------------------------------
    oracle_act_seed0: int = 88_000_000
    oracle_select_seed0: int = 88_000_100
    oracle_report_seed0: int = 88_000_200
    boot_seed: int = 88_010_000
    cal_seed: int = 88_020_000
    synth_seed: int = 88_030_000
    smoke_probe_seed0: int = 88_040_000
    smoke_train_seed0: int = 88_050_000
    smoke_oracle_seed0: int = 88_060_000
    probe_seed0: int = 88_100_000
    train_seed0: int = 89_500_000
    seed_blocks: tuple = ((88_000_000, 88_010_000), (88_010_000, 88_020_000), (88_020_000, 88_030_000),
                          (88_030_000, 88_040_000), (88_040_000, 88_050_000), (88_050_000, 88_060_000),
                          (88_060_000, 88_070_000), (88_100_000, 89_300_000), (89_500_000, 101_500_000))
    smoke_probe_span: int = 100
    smoke_fly_span: int = 32
    c_max: int = 300                                                         # W_SPEC.n_cand_max (c < 184 ≤ 300)
    stream_vectors: tuple = (
        (("boot", "pool", "S1"), (951363967822883318, 2182253266640778925, 736143899316884423)),
        (("cal", "cal", "g5", "sel", 1), (7454298250045295999, 3533929901597276528, 5564414599299931819)),
        (("cal", "cal", "S1", "ver", 24), (395433227751014368, 5991647161679921843, 5562313390102814311)),
        (("cal", "cal", "g7", "truth", 23), (4163674912002594046, 7997771487875728230, 5231196438215320182)),
        (("synth", "fut", "g6", "icc"), (4360801369187812025, 3076614880039035012, 6312001925096678629)))
    # ---- the criterion (AB.4; sealed) -------------------------------------------------------------------------------
    winsor: float = 10.0
    bar: float = 1.0                                                         # Hedges scale (D, D_fin)
    raw_min: float = 0.25                                                    # z units (R)
    k_min: int = 8
    g_min: int = 5
    round_digits: int = 9
    mech_min: float = 0.75
    min_flies: int = 2                                                       # aa_estimate.pair_record (records)
    ci_level: float = 0.95                                                   # aa_estimate.t_interval (records)
    hedges_j_declared: float = 0.8889
    c7_declared: float = 1.1259
    delta_declared: float = 0.999228
    # ---- intervals and calibration (AB.5; sealed) -------------------------------------------------------------------
    boot_b: int = 10_000
    boot_chunk: int = 1_000
    p_grid: tuple = (0.025, 0.0125, 0.005, 0.0025, 0.001, 0.0005, 0.00025, 0.0001)
    f_grid: tuple = (0.00625, 0.0025, 0.001, 0.0005, 0.00025, 0.0001, 0.00005, 0.00002, 0.00001)
    p_target: float = 0.025
    f_target: float = 0.00625
    cp_level: float = 0.975
    n_sel: int = 5_000
    n_ver: int = 10_000
    truth_pairs: int = 1_000_000
    truth_chunk: int = 50_000
    cal_chunk: int = 250                                                     # reps per checkpoint (plan Reading 9)
    cells: tuple = ((("none", 0.0),) + tuple(("X", v) for v in SD4) + tuple(("T", v) for v in SD4)
                    + tuple(("pair", v) for v in SD4) + (("XT", 1.0), ("XT", 2.0), ("XT", 3.0), ("Xpair", 2.0),
                    ("skX", 1.0), ("skX", -1.0), ("skT", 1.0), ("skT", -1.0), ("fly", 1.0), ("floor", 1.0),
                    ("floor", -1.0)))                                         # cells (1)–(24), 1-based numbers
    skew_sd: float = 2.0
    floor_const: float = 0.05
    floor_flies: int = 2
    n_type_sets: int = 9                                                     # representative structures (해석 13)
    rep_structures: tuple = (("g5", (5, 4, 3, 3, 2)), ("g6", (5, 4, 3, 3, 2, 1)), ("g7", (7, 6, 4, 3, 2, 2, 1)))
    synth_struct: str = "g7"
    synth_cells: tuple = (1, 4, 12, 19)
    synth_reps: int = 1_000
    synth_effects: tuple = (1.5, 2.5, 3.5)
    synth_other: float = 3.0
    bench_cells: tuple = (17, 23)
    bench_reps: int = 200
    # ---- records (AB.4 기록, AB.6) -----------------------------------------------------------------------------------
    floor_cut: float = 0.10
    floor_qs: tuple = (0.0, 0.25, 0.5, 0.75, 1.0)
    compare_refs: tuple = (0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5)
    bar_ref: float = 1.0
    pct: tuple = (2.5, 97.5)                                                 # record-only CIs (pool_fly / pool_pair)
    pct_scale: float = 100.0                                                 # α → numpy percentile
    sens_min: int = 2
    # ---- budget (AB.7 예산) ------------------------------------------------------------------------------------------
    core_cap_h: float = 24.0
    records_cap_h: float = 4.0
    cost_margin: float = 2.0
    s_per_h: float = 3600.0
    smoke_k: int = 26
    smoke_lenient: int = 40
    prior_kc_s_per_odour: float = 756.9 / 173                                # V kc_input (two engines × 8 seeds)
    prior_oracle_s_per_row: float = 11_039.0 / 249                           # Y 0p (16 workers)
    prior_screen_s_per_pair: float = 1_034.0 / 31                            # AA screen
    prior_learn_s_per_pair: float = 18_505.0 / 31                            # AA learn (16 workers)
    prior_fixed_h: float = 0.5                                               # 0a pytest + smoke (AB.7 table, upper)
    prior_verdict_h: float = 0.1
    # ---- the futility gate (AB.7 0f, AB.9.3; sealed with 0d) --------------------------------------------------------
    fut_src_pairs: int = 16                                                  # AA S1 (records.s1_records.keys)
    fut_src_units: int = 384                                                 # 16 × 8 flies × R · N · RN
    fut_x_sizes: tuple = (5, 2, 2, 2, 2, 2, 1)                               # AA S1 X-label groups (merge_groups)
    fut_decl_point: tuple = (5.263, -3.984, 6.021, -4.517)                   # s1_records.point, GATES order (3 dp)
    fut_decl_raw_assoc: tuple = (2.573, -1.449)                              # s1_records.raw.point (reward, punish)
    fut_decl_tau: tuple = (2.485, 1.676, 2.117, 2.038)                       # τ of D: sd · DL · sd · DL (3 dp)
    fut_decl_tau_src: tuple = ("sd", "dl", "sd", "dl")
    fut_chol_eps: float = 1e-9                                               # C + 10^−9 I (Cholesky)
    fut_tags: tuple = ("g5", "g6", "g7")                                     # the 0e representative structures
    fut_allocs: tuple = ("icc", "pair", "group")                             # ρ allocation · ρ = 0 · ρ = 1
    fut_judge: tuple = ("g6", "icc")                                         # the judgement row (해석 38)
    fut_reps: int = 2_000                                                    # M
    fut_threshold: float = 0.5                                               # P̂ < 0.5 → STOP_FUTILE (point, 해석 39)
    fut_ci_tails: tuple = (0.025, 0.975)                                     # the record CP 95 % two-sided
    fut_grid_g: tuple = (5, 6, 7, 8, 9, 10, 11, 12)
    fut_grid_k: tuple = (8, 12, 16, 20, 24, 28, 32, 36, 40)
    fut_grid_reps: int = 500
    fut_grid_alloc: str = "icc"
    fut_scenarios: tuple = (("all", ()), ("noskew", (18, 19, 20, 21)), ("sd2", (5, 9, 13, 16)),
                            ("sd1", (4, 5, 8, 9, 12, 13, 15, 16, 17, 18, 19, 20, 21)))   # (tag, excluded cells)
    fut_grid_cells: int = 272                                                # 4 scenarios × 68 (k ≥ g)
    fut_dir: str = "results/ab/fut"
    fut_detail: str = "results/ab/futility.json"
    # ---- pool, paths, CLI -------------------------------------------------------------------------------------------
    workers: int = 16
    pool_timeout_s: float = 3600.0
    thread_env: tuple = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    summary: str = "results/summary/ab_learning.json"
    raw_dir: str = "results/ab"
    cache_dir: str = "results/ab/cache"
    smoke_cache_dir: str = "results/ab/smoke/cache"
    progress_dir: str = "results/ab/progress"
    cal_dir: str = "results/ab/cal"
    tests_log: str = "results/ab/tests_0a.log"
    tests_ok_line: str = "exit 0"
    archive_root: str = "~/flymon-archive/ab"
    aa_summary: str = "results/summary/aa_learning.json"
    aa_learn_detail: str = "results/aa/learn.json"
    aa_records_detail: str = "results/aa/records.json"
    aa_estimate_detail: str = "results/aa/estimate.json"
    aa_cache_dir: str = "results/aa/cache"
    aa_probe_seed0: int = 62_000_000                                         # Y's main-set seeds (AA's; diff test)
    aa_train_seed0: int = 64_000_000
    aa_boot_seed: int = 87_000_000                                           # AA's stream root (TS diff test)
    diff_pairs: int = 2
    cli_print_chars: int = 2000
    seal_files: tuple = ("flymon/brain/ab_estimate.py", "flymon/brain/ab_spec.py", "flymon/brain/ab_rules.py",
                         "flymon/brain/ab_pairs.py")      # + the import closure, added by seal_closure() in ab_runner
    seal_names: tuple = ("flies", "probes", "winsor", "bar", "raw_min", "k_min", "g_min", "round_digits", "mech_min",
                         "hedges_j_declared", "c7_declared", "delta_declared", "boot_b", "boot_chunk", "p_grid",
                         "f_grid", "p_target", "f_target", "cp_level", "n_sel", "n_ver", "truth_pairs", "truth_chunk",
                         "cal_chunk", "cells", "skew_sd", "floor_const", "floor_flies", "n_type_sets",
                         "rep_structures", "synth_struct", "synth_cells", "synth_reps", "synth_effects", "synth_other",
                         "boot_seed", "cal_seed", "synth_seed", "probe_seed0", "train_seed0", "stream_vectors",
                         "floor_cut", "compare_refs", "noplast_pairs", "noplast_flies", "shuffle_seed", "pct", "pct_scale",
                         "fut_src_pairs", "fut_src_units", "fut_x_sizes", "fut_chol_eps", "fut_tags", "fut_allocs",
                         "fut_judge", "fut_reps", "fut_threshold", "fut_ci_tails", "fut_grid_g", "fut_grid_k",
                         "fut_grid_reps", "fut_grid_alloc", "fut_scenarios", "fut_grid_cells",
                         # AB.9.1 fixed list: the generator and its declared values (AB.3), the KC repro odours,
                         # the budget (AB.7 예산), the seed blocks and every AB seed root (AB.2), the order (stages)
                         "n_opp", "n_combos", "lv_odour_n", "decl_skips", "decl_lv", "decl_lv_kc_out", "decl_pre_kc",
                         "decl_turns", "decl_last_turn", "decl_odours", "decl_vcache_drop", "decl_vcache_drop_odours",
                         "decl_post_kc_max", "decl_needs_kc", "decl_vcache_pass", "decl_a_x_groups",
                         "decl_a_t_groups", "decl_b_x_groups", "decl_b_t_groups", "decl_gen1_same_keys", "kc_repro",
                         "core_cap_h", "records_cap_h", "cost_margin", "smoke_k", "smoke_lenient", "seed_blocks",
                         "n_oracle_seeds", "oracle_act_seed0", "oracle_select_seed0", "oracle_report_seed0",
                         "smoke_probe_seed0", "smoke_train_seed0", "smoke_oracle_seed0", "stages")

    def seal_fields(self) -> dict:
        return {k: getattr(self, k) for k in self.seal_names}

    def stage_label(self, stage: str) -> str:
        return dict(self.stage_labels).get(stage, stage)

    def sealed_stages(self) -> tuple:
        return self.stages[self.stages.index("seal_code") + 1:]

    def oracle_seeds(self) -> dict:
        n = self.n_oracle_seeds
        return dict(act=list(range(self.oracle_act_seed0, self.oracle_act_seed0 + n)),
                    select=list(range(self.oracle_select_seed0, self.oracle_select_seed0 + n)),
                    report=list(range(self.oracle_report_seed0, self.oracle_report_seed0 + n)))

    def smoke_oracle_seeds(self) -> dict:
        o, n = self.smoke_oracle_seed0, self.n_oracle_seeds
        return dict(act=list(range(o, o + n)), select=list(range(o + 100, o + 100 + n)),
                    report=list(range(o + 200, o + 200 + n)))

    def root(self, name: str) -> int:
        return dict(boot=self.boot_seed, cal=self.cal_seed, synth=self.synth_seed)[name]

    def cell(self, ci: int) -> tuple:
        """Cell (ci) of AB.5, 1-based (the spec's (1)–(24))."""
        if not 1 <= int(ci) <= len(self.cells):
            raise ValueError(f"cell {ci} outside 1..{len(self.cells)}")
        return self.cells[int(ci) - 1]

    def rep_sizes(self, tag: str) -> tuple:
        return dict(self.rep_structures)[tag]


SPEC = ABSpec()


def small(spec: ABSpec = SPEC, **kw) -> ABSpec:
    """Test scale only (never used by the CLI): smaller B / reps; every rule number stays."""
    return dataclasses.replace(spec, **kw)
