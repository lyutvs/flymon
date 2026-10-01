"""Every number of the encoder-redesign track (docs/superpowers/specs/2026-10-01-encoder-redesign-design.md).
No other module restates one; smoke runs and tests use dataclasses.replace of SPEC."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass


@dataclass(frozen=True)
class ESpec:
    exclude: tuple = ("ORN_DA1", "ORN_V")
    k2_alphabet: int = 40
    band_lo: float = 400.0
    band_hi: float = 3200.0
    # codebook (3.3)
    dsatur_tie_seeds: int = 1000
    exact_node_budget: int = 10_000_000
    anneal_iters: int = 200_000
    anneal_restarts: int = 8
    anneal_seed0: int = 20261001
    t_start: float = 2.0
    t_end: float = 0.03
    w_dup: float = 1e6
    w_soft: float = 1e3
    # measurement windows (4.1, 4.4)
    drive_strength: float = 0.35
    e0_strength: float = 0.35
    settle_ms: float = 800.0
    read_ms: float = 600.0
    window_ms: int = 200
    drive_seeds: tuple = tuple(range(24_001_000, 24_001_008))
    strength_seeds: tuple = tuple(range(24_002_000, 24_002_008))
    smoke_seeds: tuple = tuple(range(24_009_000, 24_009_100))
    # strength calibration (4.3)
    s_grid: tuple = (0.175, 0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0)
    kc_target: float = 0.0554
    kc_band: tuple = (0.05, 0.09)
    tail_lo: float = 0.03
    tail_hi: float = 0.15
    tail_share_max: float = 0.10
    # grid and selection (4.2, 4.4)
    configs: tuple = ("k3-full", "k3-norm", "k2-full", "k2-norm")
    alphas: tuple = (0.2, 0.5, 0.8)
    even_act_seeds: tuple = tuple(range(500, 508))
    even_select_seeds: tuple = tuple(range(600, 608))
    even_report_seeds: tuple = tuple(range(608, 616))
    testable_min: float = 2.0
    naive_max: float = 0.5
    bar_b: int = 11
    f_a_min: int = 2
    tie_pairs: int = 2
    n_b: int = 21
    # judgement (5.1-5.5)
    l_rng_seed: int = 20260927
    l_total_turns: int = 210
    judge_first_turn: int = 64
    judge_last_turn: int = 209
    judge_act_seeds: tuple = tuple(range(24_100_000, 24_100_008))
    judge_select_seeds: tuple = tuple(range(24_100_100, 24_100_108))
    judge_report_seeds: tuple = tuple(range(24_100_200, 24_100_208))
    margin: int = 2
    oc_q: tuple = (0.33, 0.5, 0.6, 0.7)
    oc_c: tuple = (2, 4, 7)
    oc_c_delta: int = 2
    oc_naive_max: int = 12
    oc_n_a_rows: tuple = (18, 32)
    # paths
    raw_dir: str = "results/encoder"
    m0d_summary: str = "results/summary/m0d.json"

    @staticmethod
    def k_of(config: str) -> int:
        return int(config[1])

    @staticmethod
    def dual_rule(config: str) -> str:
        return config.split("-")[1]


SPEC = ESpec()


def smoke(spec: ESpec = SPEC) -> ESpec:
    """Two seeds per block from the smoke range, two strengths, few iterations; raw dir results/encoder/smoke."""
    s = spec.smoke_seeds
    return dataclasses.replace(spec, drive_seeds=s[0:2], strength_seeds=s[2:4], even_act_seeds=s[4:6],
                               even_select_seeds=s[6:8], even_report_seeds=s[8:10], s_grid=(0.35, 0.7),
                               anneal_iters=20_000, anneal_restarts=2, raw_dir="results/encoder/smoke")


def track_seeds(spec: ESpec = SPEC) -> set:
    return set(spec.drive_seeds) | set(spec.strength_seeds) | set(spec.smoke_seeds) | set(spec.judge_act_seeds) \
        | set(spec.judge_select_seeds) | set(spec.judge_report_seeds)
