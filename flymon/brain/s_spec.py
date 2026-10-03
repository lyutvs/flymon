"""Every number of spec appendix S as amended by S.9 (S.9 wins over S.0-S.8): the same lever (APL->MBON05 removal) on
the last unused judgement set (L generator turns 104-209), one M2 judgement.
- LastSetSpec subclasses R's LeverSpec, so R's records and bands (r_records, r_rules.read_band) read S's numbers
  through the same field names; only what S changes is restated here (plan Reading 2).
- S's new seed blocks are FIELDS (judgement 24_300_xxx, smoke 24_309_xxx, gate ② 25_100_000+i and smoke 25_109_xxx in
  the nested P specs), so every collision collector sees them (S.9.6). R's reused blocks stay in LeverSpec's methods.
- Gate ② (S.3 ④, S.9.3, S.9.6) runs two P specs on the same new block: P_L (the lever, the edit p_judge's gate expects
  through spec.o.on_edit) and P_C (no edit, P's own first condition). They differ in that one edit only.
- g_fail_drop is S's net-drop threshold (G_fail_S ⇔ pun_C − pun_L ≥ 3 on an axis); g_fail_flips is None because S has
  no pass->fail branch (r_rules.g_fail / g_prob, R's rule, would fail loudly on an S spec)."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from ..agent.e_spec import SPEC as E
from .p_spec import SPEC as P_SPEC
from .p_spec import PSpec
from .p_spec import smoke as p_smoke
from .q_jobs import MBON05 as LEVER_EDIT
from .r_spec import LEVER, LeverSpec
from .r_spec import SPEC as R_SPEC

_S_O = dict(o2_seed0=25_100_000, smoke_seed0=25_109_000)
P_C = dataclasses.replace(P_SPEC, o=dataclasses.replace(P_SPEC.o, **_S_O))
P_L = dataclasses.replace(P_SPEC, o=dataclasses.replace(
    P_SPEC.o, **_S_O, o1_conditions=((LEVER, LEVER_EDIT),) + P_SPEC.o.o1_conditions[1:]))


@dataclass(frozen=True)
class LastSetSpec(LeverSpec):
    # ---- the set (S.2) ------------------------------------------------------------------------------------------
    first_turn: int = 104
    set_last_turn: int = E.judge_last_turn          # 209: the L generator's last turn (encoder 5.1)
    n_a: int = 43
    last_turn: int = 177                            # T, declared before any measurement
    digest_e0_b: str = "2b92f40a448fe5552598ac8ab265fb5cdc50424e8de945a8a5be587c44553456"
    digest_e0_a: str = "7b53467569821e3c3ffdbe2eb317843b22de033041717b9ec8d48fbc40cf8cc7"
    digest_keys: str = "884d49cc5d7c29a9a37e17166a1220c52ccbc1675b79f14f7e5abe2e0ca91231"
    # ---- the judgement seeds (S.3 ⑤) and S's smoke (S.3 ②) ------------------------------------------------------
    judge_act_seeds: tuple = tuple(range(24_300_000, 24_300_008))
    judge_select_seeds: tuple = tuple(range(24_300_100, 24_300_108))
    judge_report_seeds: tuple = tuple(range(24_300_200, 24_300_208))
    smoke_seeds: tuple = tuple(range(24_309_000, 24_309_100))
    # ---- gate ② (S.3 ④, S.9.3, S.9.6, S.9.7) ----------------------------------------------------------------------
    p: PSpec = P_L
    p_c: PSpec = P_C
    p_ratio_min: float = 0.5
    gate2_oc_rhos: tuple = (0.4, 0.5, 0.61, 0.83, 1.0)
    # ---- the guard (S.4, S.9.1) --------------------------------------------------------------------------------------
    g_fail_drop: int = 3
    g_fail_flips: object = None
    # ---- the operating characteristic (S.6) ---------------------------------------------------------------------------
    oc_harm: tuple = ((0.1, 0.02), (0.15, 0.02), (0.2, 0.05), (0.1, 0.05))
    # ---- R's reused gates (S.3 ①, S.9.5) -----------------------------------------------------------------------------
    r_shared_key: str = "3c2699c7d8eddfe0e1243ec9a4fe8c76ddb9c71bfa1a73cb3e554957df378ccc"
    r_summary: str = R_SPEC.summary
    r_reused: tuple = ("repro", "gate1", "gate3")
    r_commits: tuple = (("repro", "22c934b"), ("gate1", "6ad2201"), ("gate2", "78f514a"), ("gate3", "a83977f"))
    # ---- paths ------------------------------------------------------------------------------------------------------
    cache_dir: str = "results/s/cache"
    smoke_cache_dir: str = "results/s/smoke/cache"
    smoke_detail: str = "results/s/smoke.json"
    summary: str = "results/summary/s_lever.json"
    archive_root: str = "~/flymon-archive/s"

    def judge_seeds(self) -> dict:
        if self.smoke:
            raise ValueError("smoke never measures the judgement set (S.3 ②)")
        return dict(act=list(self.judge_act_seeds), select=list(self.judge_select_seeds),
                    report=list(self.judge_report_seeds))


SPEC = LastSetSpec()


def smoke(spec: LastSetSpec = SPEC) -> LastSetSpec:
    """Scale only: P's smoke derivation on both gate-② P specs (4 seeds 25_109_100+), S's smoke seeds through the
    inherited methods, 4 workers. Every threshold stays."""
    return dataclasses.replace(spec, p=p_smoke(spec.p), p_c=p_smoke(spec.p_c), smoke=True, workers=4)
