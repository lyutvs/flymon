"""Every number of spec appendix AC (AC.9 folded into AC.0-AC.8): the stage-1 arms on the combined-lever brain L_V,
the M3 gates, the confirmation set and the schedules, the M4 stage-1 run and its reading. AC has no PASS / FAIL
(AC.0, AC.8); every result carries LABEL.
- smoke(): AC.7 2b's smoke (F = 2 per arm x 2 battles, situation evaluation after battles 1 and 2, a 4-pair set drawn
  with the tie / smoke seed 305, schedules from seeds derived from 305). Every threshold stays.
- bench(): AC.6's representative-size benchmark (the 24-fly brain group, 16 workers, 2 learning battles, the point-0
  situation evaluation on a 20-pair set drawn with 305, no evaluation block)."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from ..agent.policy import derive_seed
from ..brain.v_spec import LEVER_V
from ..brain.v_spec import SPEC as V_SPEC

LABEL = ("M2 정식 판정 없음 — 근거는 AA 효과 크기 추정(보상 연합 d′ 6.02 [4.83, 7.49], 처벌 −4.52 [−5.89, −3.23], "
         "넓힌 풀 16쌍, 커버리지 0.855 명목 미달)뿐. 탐색 단계(부록 AC).")
FORBIDDEN = ("PASS", "FAIL", "학습된다", "학습 안 됨")


@dataclass(frozen=True)
class ACSpec:
    # ---- the brain (AC.2) -------------------------------------------------------------------------------------------
    lever_edit: str = LEVER_V
    lever_sha: str = V_SPEC.sha_combined
    a_type: str = "MBON13"
    p_type: str = "MBON05"
    codebook_config: str = "k2-norm"
    dual_rule: str = "norm"
    strength: float = 1.0
    settle_ms: float = 800.0
    read_ms: float = 600.0
    reward_dan: str = "PAM08"
    punish_dan: str = "PPL105"
    # ---- seeds (AC.4) -----------------------------------------------------------------------------------------------
    gen_seed: int = 301
    learn_seed: int = 302
    eval_seed: int = 303
    boot_seed: int = 304
    tie_seed: int = 305
    # ---- arms, battles, evaluation points (AC.2, AC.5) --------------------------------------------------------------
    brain_arms: tuple = (("FLY", 12), ("COFF", 6), ("TB", 6))
    nobrain_arms: tuple = (("RND", 16), ("MAX", 16))
    min_valid: tuple = (("FLY", 9), ("COFF", 4), ("TB", 4), ("RND", 12), ("MAX", 12))
    learn_battles: int = 40
    eval_battles: int = 20
    sit_points: tuple = (0, 10, 20, 30, 40)
    retry_max: int = 3
    # ---- the confirmation set (AC.4) --------------------------------------------------------------------------------
    n_pairs: int = 20
    # ---- statistics (AC.5) ------------------------------------------------------------------------------------------
    boot_draws: int = 10_000
    min_effect: float = 0.15
    # ---- gates and the shadow record (AC.3, AC.7 2b) ----------------------------------------------------------------
    smoke_max_tie: float = 0.5
    smoke_max_a0: float = 0.5
    smoke_max_p0: float = 0.5
    kc_ratio: float = 2.0
    floor_ratio: float = 0.2
    # ---- budget (AC.6) ----------------------------------------------------------------------------------------------
    budget_total_h: float = 60.0
    budget_stage1_h: float = 48.0
    session_cap_h: float = 24.0
    rescope_s_per_batch_battle: float = 431.1
    margin: float = 1.3
    bench_battles: int = 2
    bench_workers: int = 16
    # ---- ids and paths ----------------------------------------------------------------------------------------------
    tags: tuple = ("AL", "AE")
    mode: str = "run"
    out_root: str = "results/m4/stage1"

    def arm_sizes(self) -> dict:
        return dict(self.brain_arms + self.nobrain_arms)

    def min_valid_of(self) -> dict:
        return dict(self.min_valid)

    def n_brain(self) -> int:
        return sum(n for _, n in self.brain_arms)

    def learn_rows(self) -> int:
        return max(n for _, n in self.brain_arms)

    def eval_rows(self) -> int:
        # controller ruling (AC.4): ONE evaluation schedule of eval_battles battles, played by every fly of every arm
        return 1

    def seeds(self) -> dict:
        return {"gen": self.gen_seed, "learn": self.learn_seed, "eval": self.eval_seed, "boot": self.boot_seed,
                "tie": self.tie_seed}


SPEC = ACSpec()


def smoke(spec: ACSpec = SPEC) -> ACSpec:
    return dataclasses.replace(
        spec, brain_arms=(("FLY", 2), ("COFF", 2), ("TB", 2)), nobrain_arms=(("RND", 2), ("MAX", 2)),
        min_valid=(("FLY", 1), ("COFF", 1), ("TB", 1), ("RND", 1), ("MAX", 1)), learn_battles=2, eval_battles=2,
        sit_points=(0, 1, 2), n_pairs=4, gen_seed=spec.tie_seed,
        learn_seed=derive_seed(spec.tie_seed, "smoke-learn"), eval_seed=derive_seed(spec.tie_seed, "smoke-eval"),
        tags=("SL", "SE"), mode="smoke", out_root="results/m4-smoke")


def bench(spec: ACSpec = SPEC) -> ACSpec:
    s = smoke(spec)
    return dataclasses.replace(
        spec, learn_battles=spec.bench_battles, eval_battles=0, sit_points=(0,), gen_seed=s.gen_seed,
        learn_seed=s.learn_seed, eval_seed=s.eval_seed, tags=("BL", "BE"), mode="bench", out_root="results/m4-bench")
