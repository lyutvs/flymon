"""The configuration of spec appendix O as amended by O.7 (O.7 wins over O.2-O.5): C3's two-state map (O1) and
presentation-evoked depression (O2) on N's real-odour rig. A characterisation, no learning claim (O.1).
Every O number is a field here. N's numbers (Hallem pins and mixtures, the g grid and stimuli, probe windows, training
timings and the train-seed rule, the bootstrap draws and seed, the state threshold P < 5) are read through `n` or
assigned from N_SPEC in a default: none is restated."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from .n_spec import SPEC as N_SPEC, NSpec, train_seed


@dataclass(frozen=True)
class OSpec:
    n: NSpec = N_SPEC
    # ---- O1 (O.2, O.7.1, O.7.2) ----------------------------------------------------------------------------------------
    o1_conditions: tuple = (("on", "none"), ("kc", "apl_to_kc_zero"), ("nonkc", "apl_to_nonkc_zero"),
                            ("all", "apl_all_zero"))                 # O.7.1: the first is the APL-on rig
    o1_g_grid: tuple = N_SPEC.g_grid               # "N0f와 같음"
    o1_stimuli: tuple = N_SPEC.n0f_stimuli
    o1_c_delta: float = 8.0                        # N.8a's value, δ-DL only
    o1_seed0: int = 22_000_000
    o1_n_seeds: int = 64
    mixed_min: float = 0.10                        # O.7.2-1: silent and firing share both >= 0.10 (7 of 64)
    mixed_cells_min: int = 2                       # fewer -> STOP_NO_SILENT_STATE
    mid_band: tuple = (N_SPEC.state_p_min, 20)     # (a): P in [state_p_min, 20); the lower bound is N's threshold, read
    mid_max: float = 0.05
    auc_min: float = 0.9                           # (b) APL, (c) KC
    nature_share: tuple = (2, 3)                   # >= 2/3 of mixed cells
    path_ratio: float = 0.25                       # O.7.2-3: q_c <= 0.25 q_on
    ci_level: float = 0.95
    slope_ci_level: float = 0.99                   # O.7.2-4: Bonferroni over five stimuli
    flat_range: float = 0.15
    phi_band: tuple = (0.1, 0.9)                   # O.2 record: cell pairs with silent share in [0.1, 0.9]
    record_quantiles: tuple = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)   # per-cell APL / KC distributions (record)
    # ---- O2 (O.3, O.7.3, O.7.4) ----------------------------------------------------------------------------------------
    o2_point: tuple = (0.25, 8.0)                  # (g, c_δ): N.8a's operating point
    o2_pairs: tuple = (("4:1", "1:4", "sim"), ("dDL", "4:1", "dis"))   # (X, Y, the N.9 pair whose oracle o sets e)
    o2_arms: tuple = (("plastic", False, True, False), ("frozen", False, False, False),
                      ("punish", True, True, False), ("da_zero", False, True, True))   # (name, punish, plastic, da_zero)
    o2_seed0: int = 22_001_000
    o2_n_seeds: int = 32
    depression_frac: float = 0.25                  # b = 0.25 x mean pre A_X of arm 1
    sep_frac: float = 0.25                         # e = 0.25 x |o|
    oracle_o: tuple = (("sim", -2.348), ("dis", -2.433))   # O.7.4: N.9's oracle raw effects
    oracle_o_tol: float = 0.0005                   # reading 13: block n1's exact o must round to these
    flip_max_frac: float = 0.125                   # O.7.4-4: more than 1/8 of seeds (4 of 32)
    # ---- bootstrap (O.7.2, O.7.4) ---------------------------------------------------------------------------------------
    boot_draws: int = N_SPEC.boot_draws
    boot_seed: int = N_SPEC.boot_seed
    # ---- smoke (O.5: 22_009_xxx) and runs -------------------------------------------------------------------------------
    smoke_seed0: int = 22_009_000
    smoke_o1_n: int = 8
    smoke_o2_offset: int = 100
    smoke_o2_n: int = 4
    smoke_boot_draws: int = 200
    workers: int = N_SPEC.workers
    pool_timeout_s: float = N_SPEC.pool_timeout_s
    spec_path: str = N_SPEC.spec_path

    @property
    def o1_seeds(self) -> tuple:
        return tuple(range(self.o1_seed0, self.o1_seed0 + self.o1_n_seeds))

    @property
    def o2_seeds(self) -> tuple:
        return tuple(range(self.o2_seed0, self.o2_seed0 + self.o2_n_seeds))

    @property
    def on_edit(self) -> str:
        return self.o1_conditions[0][1]

    def o2_train_seeds(self) -> tuple:
        n = self.n
        return tuple(train_seed(s, t, n.train_seed_base, n.train_seed_stride) for s in self.o2_seeds
                     for t in range(int(n.h4.teach_trials)))

    def n_o1_presentations(self) -> int:
        return len(self.o1_conditions) * len(self.o1_g_grid) * len(self.o1_stimuli) * self.o1_n_seeds

    def n_o2_arms(self) -> int:
        return len(self.o2_pairs) * len(self.o2_arms) * self.o2_n_seeds


SPEC = OSpec()


def smoke(spec: OSpec) -> OSpec:
    """Scale only (reading 20): O1 on smoke_o1_n seeds from smoke_seed0 at two g of the grid (a slope needs two), O2 on
    smoke_o2_n seeds from smoke_seed0 + smoke_o2_offset, smoke_boot_draws draws. Every threshold stays."""
    return dataclasses.replace(spec, o1_g_grid=spec.o1_g_grid[1:3], o1_seed0=spec.smoke_seed0,
                               o1_n_seeds=spec.smoke_o1_n, o2_seed0=spec.smoke_seed0 + spec.smoke_o2_offset,
                               o2_n_seeds=spec.smoke_o2_n, boot_draws=spec.smoke_boot_draws)
