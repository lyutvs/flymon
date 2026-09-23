"""The configuration object of spec appendix J.12 (B's numeric section): the three design pairs, F.2's protocol on C0,
the readout, the statistics' constants and the calibration pilot's threshold rule.

H.3a.11's rule carries over: every number the B runner, its rules and its records use is a field of `SPEC` (or of a
`dataclasses.replace` of it, for smoke runs and tests), and no other module restates one.
"""
from __future__ import annotations

from dataclasses import dataclass

PAIRS = ("calibration", "exploration", "confirmation")


@dataclass(frozen=True)
class BSpec:
    # ---- J.12.1: the pairs (design_odor_pair(k, seed)); index p = PAIRS.index(name) -------------------------------
    k: int = 8
    pair_seeds: tuple = (("calibration", 23), ("exploration", 0), ("confirmation", 7))
    fixed_x: tuple = (("exploration", "b"),)     # F.4: MBON13 answers odour a 0-2 on every M0c seed
    x_tie: str = "b"                             # the naive rule's tie
    # ---- J.12.2: protocol (F.2) on C0 = Params() -------------------------------------------------------------------
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
    probe_base: int = 400_000                    # + 10_000 p + 100 f + k
    train_base: int = 4_000_000                  # + 100_000 p + 1000 f + t (R, N and noplast share it)
    train_base_n2: int = 4_500_000               # N' (calibration only)
    # ---- the readout (F.3) ----------------------------------------------------------------------------------------------
    a_type: str = "MBON13"
    p_type: str = "MBON05"
    z_a: tuple = (21.8293, 18.1032)
    z_p: tuple = (40.6433, 24.0891)
    # ---- J.12.3: statistics ----------------------------------------------------------------------------------------------
    floor_spikes: float = 5.0                    # naive X median of each readout type (F.4's floor)
    valid_min: int = 6                           # of n_flies
    spill_max: float = 0.5                       # Y specificity
    # ---- J.12.5: the calibration pilot's threshold rule ---------------------------------------------------------------
    null_max: float = 0.05
    power_min: float = 0.9
    boot_draws: int = 10_000
    boot_seed: int = 20260923
    grid_dprime: float = 0.05
    grid_choice: float = 0.125
    grid_max_dprime: float = 50.0

    def pair_index(self, name: str) -> int:
        return PAIRS.index(name)

    def pair_seed(self, name: str) -> int:
        return dict(self.pair_seeds)[name]

    def probe_seeds(self, name: str, fly: int) -> list:
        return [self.probe_base + 10_000 * self.pair_index(name) + 100 * fly + k for k in range(self.n_probe)]

    def train_seed(self, name: str, fly: int, trial: int, second_null: bool = False) -> int:
        base = self.train_base_n2 if second_null else self.train_base
        return base + 100_000 * self.pair_index(name) + 1000 * fly + trial


SPEC = BSpec()
