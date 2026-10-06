"""Every number of spec appendix AA (lines 6138–6388; the body AA.0–AA.9.2 is the rule, AA.9.3 lists the red-team
changes). Y's numbers that the imported Y functions read are passed by dataclasses.replace(y_spec.SPEC, …) and never
restated, except the facts AA checks against committed blocks (oracle sha256, N_len, the block commits).
- Seeds (AA.2 6174–6180): AA's own roots carry "seed" in their field names so the collectors in test_p_spec see them;
  the main-set judged probe / training roots (62_000_000 / 64_000_000) and the Y pilot roots of the differential test
  (60_000_000 / 61_000_000) are Y's (y_spec.SPEC) and are not declared here.
- Streams (plan Reading 4): SeedSequence([boot_seed, *tags]); set_flow maps each set to its two-stage flow tag.
- seal_fields(): the estimation numbers the seal (AA.7 3, 6254) hashes; seal_files: aa_estimate and every flymon file it
  imports (test_aa_spec checks the import closure)."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass


@dataclass(frozen=True)
class AASpec:
    # ---- the chain (AA.7 6245–6264; plan Reading 1) -------------------------------------------------------------
    stages: tuple = ("stage0", "reuse", "smoke", "seal_code", "screen", "coverage", "learn", "estimate", "records")
    stage_labels: tuple = (("stage0", "0a"), ("reuse", "1"), ("smoke", "2"), ("seal_code", "3"), ("screen", "4"),
                           ("coverage", "4b"), ("learn", "5"), ("estimate", "6"), ("records", "7"))
    pool_stages: tuple = ("smoke", "screen", "learn", "records")
    records_stages: tuple = ("records",)                          # the records ledger (AA.7 예산)
    sealed_stages: tuple = ("coverage", "estimate", "records")    # check the seal hash first (AA.7 3)
    # ---- reuse facts (AA.2 6170–6172, AA.7 1 6252) -----------------------------------------------------------------
    decl_commit: str = "c87f4df"                                  # plan Reading 2 (code identical at 12fd9cd)
    y_blocks: tuple = (("digest", "36df057"), ("oracle", "2a9425a"), ("stage0", "b1279d2"), ("reuse", "2b82a8f"),
                       ("pilot", "eaffc5a"), ("precheck", "279e916"), ("oc", "547c3b0"))
    y_outcomes: tuple = (("digest", "PASS"), ("oracle", "PASS"), ("stage0", "PASS"), ("reuse", "PASS"),
                         ("pilot", "PASS"), ("precheck", "PASS"), ("oc", "STOP_OC_UNREACHABLE"))
    y_absent_blocks: tuple = ("gates", "learn")
    y_absent_files: tuple = ("results/y/gates.json", "results/y/learn.json", "results/y/progress/gates.json",
                             "results/y/progress/learn.json")    # plan Reading 22
    z_blocks: tuple = (("stage0", "94c6935"), ("ydiag", "300938f"), ("reuse", "2603508"), ("sens", "403b4a3"))
    z_outcomes: tuple = (("stage0", "PASS"), ("ydiag", "PASS"), ("reuse", "PASS"), ("sens", "STOP_PLAN_UNREACHABLE"))
    z_summary: str = "results/summary/z_learning.json"
    oracle_sha256: str = "fa4750ac80883bcdc571e8e5b6ba7cc3d25f320549988dc0c3ab8823f75768bb"
    n_len: int = 31
    n_len_axes: tuple = (("b", 11), ("a", 20))
    n_main: int = 249
    # ---- the design (AA.1 6162, AA.3 6189–6191) ---------------------------------------------------------------------
    flies: int = 8                                                # F
    probes: int = 8                                               # K
    smoke_pair: int = 0                                           # Y pilot V pair j 0 = b|17 (AA.7 2)
    smoke_flies: int = 1
    noplast_pairs: int = 2                                        # AA.7 7
    noplast_flies: int = 2
    # ---- AA seed roots (AA.2 6174–6180) ------------------------------------------------------------------------------
    boot_seed: int = 87_000_000
    smoke_probe_seed0: int = 87_100_000
    smoke_train_seed0: int = 87_110_000
    g6_seed: int = 87_200_000
    synth_seed: int = 87_300_000
    seed_blocks: tuple = ((87_000_000, 87_100_000), (87_100_000, 87_200_000), (87_200_000, 87_300_000),
                          (87_300_000, 87_400_000))
    smoke_probe_span: int = 100                                   # smoke probe seeds: root + f·100 + k (W layout)
    smoke_fly_span: int = 32
    # ---- estimation (AA.4, AA.5, AA.6; sealed) -----------------------------------------------------------------------
    winsor: float = 10.0
    min_flies: int = 2
    boot_b: int = 10_000
    boot_chunk: int = 1_000
    pct: tuple = (2.5, 97.5)
    ci_level: float = 0.95
    min_groups: int = 5                                           # two-stage primary iff groups ≥ 5 (k ≥ 5 implied)
    set_flow: tuple = (("S1", "pool"), ("S31", "pool31"), ("out", "pool_out"), ("sens", "sens_floor"))
    floor_cut: float = 0.10
    floor_qs: tuple = (0.0, 0.25, 0.5, 0.75, 1.0)               # φ quantiles (AA.6 6239)
    compare_refs: tuple = (0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5)
    bar_ref: float = 1.0                                          # the ±1 rows carry the "마리 관문 막대" note (6284)
    c7_declared: float = 1.1259
    hedges_j_declared: float = 0.8889
    # ---- coverage (AA.5 6227–6231, AA.9.3 해석 14) and the order-0 synthetic validation (6247) --------------------
    cov_deltas: tuple = (0.0, 1.0, 1.5)
    cov_hets: tuple = ("none", "group", "pair")
    cov_het_sd: float = 0.5
    cov_reps: int = 1000
    cov_b: int = 2000
    cov_bar: float = 0.90
    cov_methods: tuple = ("two_stage", "fly", "pair", "raw_two_stage", "raw_fly")
    synth_pairs: int = 20
    synth_deltas: tuple = (0.0, 1.0, 1.5)
    synth_sds: tuple = (0.0, 0.5)
    synth_reps: int = 1000
    # ---- G.6 recompute record (AA.9.1 6299–6310) ---------------------------------------------------------------------
    g6_rows: tuple = ("point", "lo", "hi")
    g6_scales: tuple = ("hedges", "raw")                          # Hedges first (6305 "주 행")
    d_false: float = 0.5
    g6_design: tuple = (0.5, 0.5, 16, 8, (4, 8))                   # p_set, q, K, F, k range (6309)
    # ---- budget (AA.7 6267–6270) -------------------------------------------------------------------------------------
    core_cap_h: float = 12.0
    records_cap_h: float = 4.0
    g6_cap_h: float = 2.0
    cost_margin: float = 1.3
    s_per_h: float = 3600.0
    round_digits: int = 9
    cov_timing_reps: int = 20                                     # smoke: one coverage cell's timed reps
    # ---- pool, paths, CLI ---------------------------------------------------------------------------------------------
    workers: int = 16
    pool_timeout_s: float = 3600.0
    thread_env: tuple = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    summary: str = "results/summary/aa_learning.json"
    raw_dir: str = "results/aa"
    cache_dir: str = "results/aa/cache"
    smoke_cache_dir: str = "results/aa/smoke/cache"
    progress_dir: str = "results/aa/progress"
    tests_log: str = "results/aa/tests_0a.log"
    tests_ok_line: str = "exit 0"
    archive_root: str = "~/flymon-archive/aa"
    cli_print_chars: int = 2000
    seal_files: tuple = ("flymon/brain/aa_estimate.py", "flymon/brain/aa_spec.py", "flymon/brain/w_verdict.py",
                         "flymon/brain/y_rules.py", "flymon/brain/y_oc.py", "flymon/brain/x_oc.py",
                         "flymon/brain/x_verdict.py", "flymon/brain/w_oc.py", "flymon/brain/w_records.py",
                         "flymon/brain/y_spec.py", "flymon/brain/w_spec.py", "flymon/brain/w_rules.py")
    seal_names: tuple = ("winsor", "min_flies", "boot_b", "boot_chunk", "pct", "ci_level", "min_groups", "boot_seed",
                         "synth_seed", "set_flow", "floor_cut", "floor_qs", "compare_refs", "bar_ref", "cov_deltas", "cov_hets",
                         "cov_het_sd", "cov_reps", "cov_b", "cov_bar", "cov_methods", "synth_pairs", "synth_deltas",
                         "synth_sds", "synth_reps", "g6_rows", "g6_scales", "d_false", "g6_design", "g6_seed",
                         "g6_cap_h", "flies", "probes")

    def seal_fields(self) -> dict:
        return {k: getattr(self, k) for k in self.seal_names}

    def stage_label(self, stage: str) -> str:
        return dict(self.stage_labels).get(stage, stage)

    def flow(self, set_tag: str) -> str:
        return dict(self.set_flow)[set_tag]


SPEC = AASpec()


def small(spec: AASpec = SPEC, **kw) -> AASpec:
    """Test scale only (never used by the CLI): smaller B / reps; every rule number stays."""
    return dataclasses.replace(spec, **kw)
