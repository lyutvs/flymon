"""The configuration of spec appendix M as amended by M.10: C3 unchanged; the taught DAN compartments re-chosen among
21 cell-level core candidates (PAM reward x PPL1 punish, 19 scanned), read through their core-cell population; an
offline specificity pre-check, a per-arm scan and a joint top-2 x 2 check on even pairs, and a judgement on the L set's
turns >= 4 against the same-set C3. `j` carries every J/H constant, `k` K's min_weight, `l` L's pair generator, the
pinned ceiling file, m0d.json's path, C3's readout and the attempt rows I-K. Nothing here repeats one of them."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from .j_spec import SPEC as J_SPEC, JSpec
from .k_spec import SPEC as K_SPEC, KSpec
from .l_spec import SPEC as L_SPEC, LSpec


@dataclass(frozen=True)
class MSpec:
    j: JSpec = J_SPEC
    k: KSpec = K_SPEC
    l: LSpec = L_SPEC
    # ---- candidates (M.10.1): representatives of the distinct core cell sets ----------------------------------------
    reward_family: str = "PAM"
    punish_family: str = "PPL1"
    reward_candidates: tuple = ("PAM01", "PAM02", "PAM03", "PAM04", "PAM05", "PAM06", "PAM08", "PAM10", "PAM11",
                                "PAM12", "PAM13", "PAM14", "PAM15")   # PAM07 = PAM08, PAM09 = PAM10 (same core cells)
    punish_candidates: tuple = ("PPL101", "PPL102", "PPL103", "PPL104", "PPL105", "PPL106", "PPL107", "PPL108")
    incumbent_reward: str = "PAM08"
    incumbent_punish: str = "PPL105"
    # ---- stages 1-2 (M.3, M.10.3) ------------------------------------------------------------------------------------
    top_k: int = 2
    # ---- judgement set (M.10.4) --------------------------------------------------------------------------------------
    judge_from_turn: int = 4
    judge_n_b: int = 21
    judge_n_a: int = 18
    judge_b_digest: str = "f55be2dfd868ab7ec1ab2913a2d1cbf3ba6823f1ca4b307fe42392149461d162"
    judge_a_digest: str = "a978f054fb497d23ca620354b8b92e2c4871ad17770fbec990b10a0828350dc1"
    # ---- smoke sizes (reading 17) ------------------------------------------------------------------------------------
    n_even_b: int | None = None           # None = all declared even (b) pairs; smoke uses 2
    n_even_a: int | None = None
    n_odd: int | None = None
    # ---- operating characteristics (M.10.6) -------------------------------------------------------------------------
    oc_q_b: tuple = (0.33, 0.5, 0.6, 0.7)
    oc_ratio: tuple = (0.5, 1.0)
    oc_naive_a: tuple = tuple(range(0, 9))
    oc_c: tuple = (2, 4, 7)
    # ---- attempt rows after L's own I-K (L.12 result, this declaration) ---------------------------------------------
    attempts_new: tuple = (("L", "naive-readout screen SCREEN_IMPRECISE 7/15", "c3d2e25"),
                           ("M", "this declaration", "1441eba"))

    @property
    def m0d_path(self) -> str:
        return self.l.m0d_path

    @property
    def readout_c3(self) -> tuple:
        """Block h4's C3 single readout (reading 6) = L's (H.4's) readout."""
        return self.l.readout

    @property
    def attempts(self) -> tuple:
        """I-K as L recorded them (L's own "this declaration" row replaced by its result), then L and M."""
        return tuple(a for a in self.l.attempts if a[0] != "L") + self.attempts_new

    def dict_readout_c3(self) -> dict:
        return dict(self.readout_c3)


SPEC = MSpec()


def smoke(spec: MSpec) -> MSpec:
    """Two candidates per arm (incumbent + one, a filter of the declared lists in declared order: m_cands checks the
    representatives against the declared SPEC lists first, then keeps these), two even (b) / (a) pairs, two odd pairs,
    and a judgement list taken from the L set's last turns (turn >= 60 of 64, M.10.8; never the declared turns 4-12 nor
    the F v4 confirmation candidates, turns 13-59) with no digests: a smoke run can never write a judged block."""
    keep_r, keep_p = {"PAM08", "PAM10"}, {"PPL103", "PPL105"}
    return dataclasses.replace(spec, reward_candidates=tuple(n for n in spec.reward_candidates if n in keep_r),
                               punish_candidates=tuple(n for n in spec.punish_candidates if n in keep_p),
                               n_even_b=2, n_even_a=2, n_odd=2, judge_from_turn=60, judge_n_b=2, judge_n_a=0,
                               judge_b_digest="", judge_a_digest="")
