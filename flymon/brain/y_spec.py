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
  in a sentence comes from here. Y.9.2's new seed roots (78_000_000 · 78_100_000 · 78_200_000) are phase A's to append."""
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


SPEC = YSpec()
