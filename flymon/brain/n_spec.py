"""The configuration of spec appendix N as amended by N.8 (N.8 wins where they differ): C3 unchanged; Hallem 2006 real
odours (odor_real, data/odor/ pinned by sha256); N0f's feasibility grid and operating-point rule (N.8.3); N0's
similarity gate (N.8.4); N1's punish-only oracle (N.3, N.8.5); N2.0's pilot and joint-rule OC and N2's absolute
punishment conditioning x APL->KC block (N.4, N.8.6). `l` carries C3's m0d path and readout; H.4 / H.3 constants come
through `h4` / `h3` (= l.j.h4 / l.j.h4.h3). The data pins (DoOR repo, commit, column, odorant CAS, receptor count,
output sha256) live only in scripts/fetch_door_hallem.py's constant block and are loaded from it by file path. Nothing
here repeats one of them."""
from __future__ import annotations

import dataclasses
import importlib.util
from dataclasses import dataclass
from pathlib import Path

from .l_spec import SPEC as L_SPEC, LSpec

_FETCH_PATH = Path(__file__).resolve().parents[2] / "scripts" / "fetch_door_hallem.py"


def _load_fetch():
    """scripts/fetch_door_hallem.py as a module (its main() is behind `if __name__ == "__main__"`; importing is inert)."""
    spec = importlib.util.spec_from_file_location("_n_fetch_door_hallem", _FETCH_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_FETCH = _load_fetch()


def train_seed(seed: int, trial: int, base: int, stride: int) -> int:
    """The training reset seed, conditioning.train_block's rule (base + seed x stride + trial): the one formula (NSpec and
    n_jobs call it). Python ints: ~2.1e10 reaches default_rng unchanged."""
    return int(base) + int(seed) * int(stride) + int(trial)


@dataclass(frozen=True)
class NSpec:
    l: LSpec = L_SPEC
    # ---- data (N.8.2); pins loaded from the fetch script, never restated ----------------------------------------------
    data_dir: str = "data/odor"
    data_sha256: tuple = tuple(_FETCH.OUTPUT_SHA256.items())
    door_repo: str = _FETCH.REPO
    door_commit: str = _FETCH.SHA
    door_column: str = _FETCH.COLUMN
    odorant_cas: tuple = _FETCH.ODORANTS                     # (key, CAS): IA, EB, dDL
    n_receptors: int = _FETCH.N_RECEPTORS
    lin_totals: tuple = (("IA", 2030.0), ("EB", 1860.0), ("dDL", 286.0))   # Lin et al. 2014, a record (N.8.2)
    lin_tol: float = 0.01
    # ---- stimuli (N.2) -------------------------------------------------------------------------------------------------
    mixtures: tuple = (("4:1", (("IA", 0.8), ("EB", 0.2))), ("1:4", (("IA", 0.2), ("EB", 0.8))),
                       ("dDL", (("dDL", 1.0),)), ("IA", (("IA", 1.0),)), ("EB", (("EB", 1.0),)))
    n0f_stimuli: tuple = ("4:1", "1:4", "dDL", "IA", "EB")
    judged_stimuli: tuple = ("4:1", "1:4", "dDL")
    pairs: tuple = (("sim", "4:1", "1:4"), ("dis", "4:1", "dDL"))
    # ---- N0f (N.8.3) ---------------------------------------------------------------------------------------------------
    g_grid: tuple = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)
    c_delta_grid: tuple = (1.0, 2.0, 4.0, 8.0)
    conditions: tuple = (("on", "none"), ("block", "apl_to_kc_zero"), ("all", "apl_all_zero"))
    n0f_seeds: tuple = tuple(range(21_003_000, 21_003_008))
    kc_band: tuple = (0.03, 0.15)                 # (1) on: each judged stimulus' KC median, inclusive
    kc_target: float = 0.0554                     # C3's H.3 reference
    kc_block_max: float = 0.30                    # (2) on and block
    runaway_hz: float = 150.0                     # D.6 (a) sub-window threshold
    runaway_share_max: float = 0.125              # (2) presentations above it, at most 1/8
    state_p_min: int = 5                          # N.8.8: P < 5 silent, >= 5 firing
    state_diff_flag: float = 0.25                 # state-share difference above it is flagged, not a stop
    # ---- N0 (N.8.4) ----------------------------------------------------------------------------------------------------
    act_seeds: tuple = tuple(range(21_000_000, 21_000_016))    # i < 16 (N.8.4); reading 1
    boot_draws: int = 10_000                      # N0's seed bootstrap and N2's judgement bootstrap
    boot_seed: int = 20260930
    ci_level: float = 0.95                        # N0 go, N2 (1) and (2): lower bound of the 95% CI
    equiv_ci_level: float = 0.90                  # N2 (3): D_dis 90% CI inside [-eps, eps]
    # ---- N1 (N.3, N.8.5) -----------------------------------------------------------------------------------------------
    select_seeds: tuple = tuple(range(21_000_100, 21_000_108))
    report_seeds: tuple = tuple(range(21_000_200, 21_000_216))
    # ---- N2.0 / N2 (N.4, N.8.6) -----------------------------------------------------------------------------------------
    pilot_seeds: tuple = tuple(range(21_001_000, 21_001_016))
    judge_seed0: int = 21_002_000
    n2_conditions: tuple = (("sim_on", "sim", "none"), ("sim_off", "sim", "apl_to_kc_zero"),
                            ("dis_on", "dis", "none"), ("dis_off", "dis", "apl_to_kc_zero"))
    n2_record_conditions: tuple = (("sim_all", "sim", "apl_all_zero"), ("dis_all", "dis", "apl_all_zero"))
    plumbing_seeds: int = 2                       # reading 9
    train_seed_base: int = 1_000_000              # conditioning.train_block's rule
    train_seed_stride: int = 1000
    c1_frac: float = 0.25
    delta_min_frac: float = 0.5
    eps_frac: float = 0.25
    alt_frac: float = 0.75
    n_grid: tuple = (16, 24, 32, 48, 64)
    sd_mults: tuple = (1.0, 2.0)
    null_max: float = 0.05
    power_min: float = 0.8
    oc_draws: int = 2000                          # reading 7: experiments per (n, spread, hypothesis) cell
    oc_boot: int = 1000                           # reading 7: bootstrap draws inside one simulated experiment
    oc_seed: int = 20261001
    budget_h: float = 48.0
    budget_workers: int = 16                      # reading 11: the budget's assumed worker count
    # ---- runs ------------------------------------------------------------------------------------------------------------
    workers: int = 16                             # the stage CLIs' --workers default
    pool_timeout_s: float = 3600.0                # FlyPool timeout per call (m_cli's: teach jobs run whole)
    # ---- where -----------------------------------------------------------------------------------------------------------
    spec_path: str = "docs/superpowers/specs/2026-09-14-flymon-design.md"
    smoke_point: tuple = (1.0, 1.0)               # reading 13

    @property
    def h4(self):
        return self.l.j.h4

    @property
    def h3(self):
        return self.l.j.h4.h3

    @property
    def zero_share_max(self) -> float:
        return self.h4.react_zero_share_max

    @property
    def testable_min(self) -> float:
        return self.h4.testable_min

    def sha_pins(self) -> dict | None:
        return dict(self.data_sha256) if self.data_sha256 else None

    def mixture(self, name: str) -> dict:
        return dict(dict(self.mixtures)[name])

    def mixtures_dict(self) -> dict:
        return {n: dict(w) for n, w in self.mixtures}

    def edit_of(self, cond: str) -> str:
        return dict(self.conditions)[cond]

    def pair_stimuli(self) -> dict:
        return {n: (x, y) for n, x, y in self.pairs}

    def judge_seeds(self, n: int) -> tuple:
        return tuple(range(self.judge_seed0, self.judge_seed0 + int(n)))

    def train_seed(self, seed: int, trial: int) -> int:
        return train_seed(seed, trial, self.train_seed_base, self.train_seed_stride)


SPEC = NSpec()


def smoke(spec: NSpec) -> NSpec:
    """A 1 x 1 grid, a seed block of its own (21_009_xxx, never a declared stage's), 2-3 seeds per role, n = 3, few
    bootstrap draws and a tiny OC. The data pins stay: smoke runs on the real data."""
    return dataclasses.replace(
        spec, g_grid=(1.0,), c_delta_grid=(1.0,), n0f_seeds=(21_009_000, 21_009_001),
        act_seeds=(21_009_010, 21_009_011, 21_009_012), select_seeds=(21_009_020, 21_009_021),
        report_seeds=(21_009_030, 21_009_031, 21_009_032), pilot_seeds=(21_009_040, 21_009_041, 21_009_042),
        judge_seed0=21_009_050, n_grid=(3,), plumbing_seeds=1, boot_draws=200, oc_draws=20, oc_boot=50)
