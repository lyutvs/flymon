"""Constants of the re-scoped claim (docs/superpowers/specs/2026-09-28-rescoped-claim-design.md, section 10 overrides)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RSpec:
    k: int = 8
    pair_seeds: tuple = (1000, 1001, 1002, 1003, 1004, 1005)
    gen_start: int = 1000
    n_pairs: int = 6
    used_design_seeds: tuple = (0, 7, 23)
    pairs_digest: str = "2c5d8076c912e069bdc61b29221c53007bea7f1d065b5e20d9ddeb1952d14133"
    control: str = "seed0"
    strength: float = 0.35
    n_flies: int = 8
    n_probe: int = 8
    trials: int = 20
    pulse_ms: float = 400.0
    reward_dan: str = "PAM08"
    punish_dan: str = "PPL105"
    probe_settle_ms: float = 800.0
    probe_read_ms: float = 600.0
    train_settle_ms: float = 800.0
    train_gap_ms: float = 200.0
    a_type: str = "MBON13"
    p_type: str = "MBON05"
    z_a: tuple = (10.78125, 9.412096743243064)
    z_p: tuple = (26.25, 19.30889259728101)
    x_tie: str = "b"
    floor_spikes: float = 5.0
    floor_silent_max: float = 0.125   # rule B (spec 10.3 amendment 2026-09-28): share of qualification seeds < floor_spikes
    valid_min: int = 6
    assoc_min: float = 1.0
    spill_max: float = 0.5
    sign_min: float = 0.75
    oracle_min: float = 2.0
    oracle_alphas: tuple = (0.2, 0.5, 0.8)
    oracle_settle_ms: float = 800.0
    oracle_read_ms: float = 600.0
    kc_window_ms: float = 200.0
    n_qual_seeds: int = 8
    min_pairs: int = 4
    fp_cap: float = 0.10
    q_null: float = 0.2
    rho: float = 0.3
    probe_base: int = 900_000
    train_base: int = 9_000_000
    train_base_n2: int = 9_800_000
    qual_base: int = 980_000
    recovery_grid: tuple = (0.0, 0.001, 0.002, 0.005, 0.01, 0.02)
    median_floor: float = 0.5
    taurec_taught_floor_max: float = 0.5   # spec 10.6 amendment 2026-09-28: alt floor_frac_taught path max <= this
    taurec_pulses: int = 1000
    taurec_odours: int = 50
    taurec_gen_seed: int = 990_000
    taurec_seed_base: int = 991_000
    taurec_sample_every: int = 10
    taurec_reward_ms: float = 600.0
    taurec_punish_ms: float = 400.0
    residual_max: float = 0.05
    boot_draws: int = 10_000
    boot_seed: int = 20260928
    power_draws: int = 20_000
    mie: float = 0.05
    power_target: float = 0.8
    var_margin: float = 1.5
    budget_hours: float = 60.0
    learn_battles: int = 40
    retry_max: int = 3
    schedule_seeds: tuple = (("pilot", 101, 102), ("judge", 201, 202))

    def pair_names(self) -> tuple:
        return (self.control,) + tuple(f"p{s}" for s in self.pair_seeds)

    def pair_index(self, name: str) -> int:
        return self.pair_names().index(name)

    def probe_seeds(self, name: str, fly: int) -> list:
        p = self.pair_index(name)
        return [self.probe_base + 10_000 * p + 100 * fly + k for k in range(self.n_probe)]

    def train_seed(self, name: str, fly: int, trial: int, second_null: bool = False) -> int:
        base = self.train_base_n2 if second_null else self.train_base
        return base + 100_000 * self.pair_index(name) + 1000 * fly + trial

    def qual_seeds(self, name: str) -> dict:
        b = self.qual_base + 1000 * self.pair_index(name)
        return {key: [b + off + i for i in range(self.n_qual_seeds)]
                for key, off in (("act", 0), ("select", 100), ("report", 200))}


SPEC = RSpec()
