"""Every number of spec appendix AD (AD.9 folded into AD.0-AD.8): AC's stage-2 amendment. FLY-L 12 and the
opponent-shuffled control FLY-OS 12 on the combined-lever brain L_V (one pool; global flies 24-35 / 36-47, pool indices
0-23, ids DL / DE), 120 learning battles (fallback 80, futility check at 40), situation points every 20 battles, the
fresh 20-pair descriptive set at 0 / 40 / T, seeds 306 (learning schedule) / 307 (OS draws) / 308 (new pairs), the
budget left after stage 1 (60 h - 41 695.3 s - 1 h reserve). AD has no PASS / FAIL (AD.7); every result carries AC.5's
LABEL.
- smoke(): AD.5 1's smoke (FLY-L 2 + FLY-OS 2, 2 learning + 2 evaluation battles, points 0 / 1 / 2, futility check at 1,
  4 + 4 pairs, seeds derived from 305, ids XL / XE). Every threshold stays.
- bench(n): AD.4's re-bench attempt n (24 flies, 16 workers, 2 learning battles, the point-0 situation evaluations, no
  evaluation block, ids YL / YE), one output directory per attempt."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from ..ac.spec import FORBIDDEN, LABEL  # noqa: F401  (AD keeps AC.5's label and AC.8's forbidden words)
from ..ac.spec import SPEC as AC
from ..agent.policy import derive_seed

ARM_TEXT = {"FLYL": "FLY-L", "FLYOS": "FLY-OS", "COFF": "C-off", "FLY1": "1단계 FLY", "RND": "RND", "MAX": "MAX",
            "FLYL_OS": "FLY-L(OS 평가)"}
STOPS = ("STOP_SAFETY", "STOP_SMOKE", "STOP_BALANCE", "STOP_REUSE", "STOP_MANIFEST", "STOP_BENCH", "STOP_BUDGET",
         "STOP_FUTILE", "STOP_INFRA")


@dataclass(frozen=True)
class ADSpec:
    # ---- arms (AD.1) ------------------------------------------------------------------------------------------------
    arms: tuple = (("FLYL", 12), ("FLYOS", 12))
    first_fly: int = 24
    min_valid: tuple = (("FLYL", 9), ("FLYOS", 9), ("COFF", 4), ("FLY1", 9), ("RND", 12), ("MAX", 12))
    # ---- learning length, futility check, points (AD.2) -------------------------------------------------------------
    learn_battles: int = 120
    fallback_battles: int = 80
    futility_point: int = 40
    point_step: int = 20
    eval_battles: int = AC.eval_battles
    retry_max: int = AC.retry_max
    # ---- sets (AD.2) ------------------------------------------------------------------------------------------------
    n_pairs: int = AC.n_pairs
    n_new_pairs: int = 20
    # ---- seeds (AD.2, AD.8 1) ---------------------------------------------------------------------------------------
    gen_seed: int = AC.gen_seed
    eval_seed: int = AC.eval_seed
    boot_seed: int = AC.boot_seed
    tie_seed: int = AC.tie_seed
    learn_seed: int = 306
    os_seed: int = 307
    new_seed: int = 308
    # ---- statistics and readings (AD.3) -----------------------------------------------------------------------------
    boot_draws: int = AC.boot_draws
    min_effect: float = AC.min_effect
    kc_ratio: float = AC.kc_ratio
    floor_ratio: float = AC.floor_ratio
    floor_read: float = 0.5
    taurec_pulses: int = 1000
    # ---- OS gates (AD.5 3-4, AD.8 5-6) ------------------------------------------------------------------------------
    os_gate_draws: int = 16_000
    os_freq_tol: float = 0.01
    os_chi2_p: float = 0.001
    strength_tol: float = 1e-9
    kc_band_min: float = 0.15
    kc_band_sd: float = 3.0
    calib_reps: tuple = (0, 1, 2)
    smoke_max_tie: float = AC.smoke_max_tie
    smoke_max_a0: float = AC.smoke_max_a0
    smoke_max_p0: float = AC.smoke_max_p0
    # ---- budget (AD.4, AD.8 11) -------------------------------------------------------------------------------------
    budget_total_h: float = AC.budget_total_h
    stage1_spent_s: float = 41_695.3
    reserve_h: float = 1.0
    session_cap_h: float = AC.session_cap_h
    rescope_s_per_batch_battle: float = AC.rescope_s_per_batch_battle
    margin: float = AC.margin
    bench_battles: int = AC.bench_battles
    bench_workers: int = AC.bench_workers
    bench_valid: int = 3
    bench_max_attempts: int = 6
    busy_cpu: float = 50.0
    busy_samples: int = 3
    worst_factor: float = 2.0
    # ---- the brain (AC.2, unchanged) --------------------------------------------------------------------------------
    lever_edit: str = AC.lever_edit
    lever_sha: str = AC.lever_sha
    p_type: str = AC.p_type
    codebook_config: str = AC.codebook_config
    dual_rule: str = AC.dual_rule
    # ---- ids, paths, stage-1 records reused ---------------------------------------------------------------------------
    tags: tuple = ("DL", "DE")
    mode: str = "run"
    out_root: str = "results/m4/ad"
    bench_attempt: int = 0
    stage1_root: str = AC.out_root
    coff_flies: tuple = (12, 13, 14, 15, 16, 17)
    fly1_flies: tuple = tuple(range(12))
    stage1_point: int = 40

    def arm_sizes(self) -> dict:
        return dict(self.arms)

    def min_valid_of(self) -> dict:
        return dict(self.min_valid)

    def n_flies(self) -> int:
        return sum(n for _, n in self.arms)

    def n_rows(self) -> int:
        return max(n for _, n in self.arms)

    def layout(self) -> list:
        out = []
        for arm, n in self.arms:
            for k in range(n):
                out.append((arm, k, self.first_fly + len(out)))
        return out

    def _t(self, T) -> int:
        return int(self.learn_battles if T is None else T)

    def points(self, T=None) -> tuple:
        return tuple(range(0, self._t(T) + 1, self.point_step))

    def new_points(self, T=None) -> tuple:
        pts = self.points(T)
        return tuple(p for p in pts if p in (0, self.futility_point, pts[-1]))

    def futility_checked(self, T=None) -> bool:
        return self.futility_point < self._t(T) and self.futility_point in self.points(T)

    def seeds(self) -> dict:
        return {"gen": self.gen_seed, "eval": self.eval_seed, "boot": self.boot_seed, "tie": self.tie_seed,
                "learn": self.learn_seed, "os": self.os_seed, "new": self.new_seed}

    def budget_limit_h(self) -> float:
        return self.budget_total_h - self.stage1_spent_s / 3600.0 - self.reserve_h

    def sit_evals_at(self, point: int, T, new_set: bool) -> int:
        if int(point) not in self.points(T):
            return 0
        n = self.n_flies() + self.arm_sizes()["FLYL"]
        return n + (self.n_flies() if new_set and int(point) in self.new_points(T) else 0)

    def sit_evals(self, T, new_set: bool) -> int:
        return sum(self.sit_evals_at(p, T, new_set) for p in self.points(T))

    def batches(self, T) -> int:
        return int(T) + int(self.eval_battles)


SPEC = ADSpec()


def smoke(spec: ADSpec = SPEC) -> ADSpec:
    return dataclasses.replace(
        spec, arms=(("FLYL", 2), ("FLYOS", 2)),
        min_valid=(("FLYL", 1), ("FLYOS", 1), ("COFF", 1), ("FLY1", 1), ("RND", 1), ("MAX", 1)),
        learn_battles=2, fallback_battles=2, futility_point=1, point_step=1, eval_battles=2, n_pairs=4, n_new_pairs=4,
        gen_seed=spec.tie_seed, learn_seed=derive_seed(spec.tie_seed, "ad-smoke-learn"),
        eval_seed=derive_seed(spec.tie_seed, "smoke-eval"), new_seed=derive_seed(spec.tie_seed, "ad-smoke-new"),
        tags=("XL", "XE"), mode="smoke", out_root="results/m4-smoke/ad")


def bench(attempt: int, spec: ADSpec = SPEC) -> ADSpec:
    return dataclasses.replace(
        spec, learn_battles=spec.bench_battles, fallback_battles=spec.bench_battles, eval_battles=0,
        gen_seed=spec.tie_seed, learn_seed=derive_seed(spec.tie_seed, "ad-bench-learn"),
        eval_seed=derive_seed(spec.tie_seed, "smoke-eval"), new_seed=derive_seed(spec.tie_seed, "ad-bench-new"),
        tags=("YL", "YE"), mode="bench", out_root=f"results/m4-bench/ad-{int(attempt)}", bench_attempt=int(attempt))
