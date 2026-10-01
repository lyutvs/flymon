"""The configuration of spec appendix P as amended by P.6 (P.6 wins over P.1-P.4): the first learning judgement of the
project — punishment learning on N's real-odour dissimilar pair (IA:EB 4:1 vs δ-DL), both directions, on new seeds. A
confirmatory test declared after O2 (P.0).
The O-shaped part `o` is o_spec.SPEC derived with dataclasses.replace (P.6.7): only the pairs (the two directions), the
arms (①②③ of O's four, with O's flags) and the seed blocks differ. N's and O's numbers are read through `o` / `o.n`,
never restated; the P-only numbers are this dataclass's own fields."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from .n_spec import smoke as n_smoke
from .o_spec import SPEC as O_SPEC, OSpec, smoke as o_smoke

ARMS = ("plastic", "frozen", "punish")            # P.2: ① no punish, plastic; ② no punish, frozen; ③ punish, plastic
DIRECTIONS = ("r1", "r2")                         # P.2: r1 X = 4:1, Y = δ-DL; r2 X = δ-DL, Y = 4:1
C1_PAIR = "dis"                                   # P.3: c1 from N1's dissimilar-pair oracle (N.8.6's c1 rule)
_X, _Y = dict((p, (x, y)) for p, x, y in O_SPEC.n.pairs)[C1_PAIR]

P_O = dataclasses.replace(
    O_SPEC,
    o2_pairs=((_X, _Y, C1_PAIR), (_Y, _X, C1_PAIR)),                 # (X, Y, the N1 pair whose o sets c1), r1 then r2
    o2_arms=tuple(a for a in O_SPEC.o2_arms if a[0] in ARMS),       # O's (name, punish, plastic, da_zero), arm 4 dropped
    o2_seed0=23_000_000,                                             # P.2: 23_000_000 + i, i < o2_n_seeds (32, O's)
    smoke_seed0=23_009_000)                                          # P.2: smoke 23_009_xxx (O's offset 100 -> 23_009_100+)


@dataclass(frozen=True)
class PSpec:
    o: OSpec = P_O
    directions: tuple = DIRECTIONS
    arms: tuple = ARMS
    c1_pair: str = C1_PAIR
    concentration_frac: float = 0.5               # P.6.2 (iii): |mean dt| <= 0.5 |s_r|
    oc_draws: int = O_SPEC.n.oc_draws             # P.6.4: simulated experiments (N's OC count)
    oc_seed: int = O_SPEC.n.oc_seed
    smoke_oc_draws: int = n_smoke(O_SPEC.n).oc_draws
    o2_check_tol: float = 1e-9                    # P.6.4: recomputed O2 D vs the committed block's D

    @property
    def seeds(self) -> tuple:
        return self.o.o2_seeds

    @property
    def c1_frac(self) -> float:
        return self.o.n.c1_frac

    def train_seeds(self) -> tuple:
        return self.o.o2_train_seeds()

    def pairs(self) -> dict:
        return {d: (x, y) for d, (x, y, _) in zip(self.directions, self.o.o2_pairs)}

    def flags(self) -> dict:
        return {a: (pu, pl, dz) for a, pu, pl, dz in self.o.o2_arms}

    def n_arms(self) -> int:
        return len(self.directions) * len(self.arms) * self.o.o2_n_seeds


SPEC = PSpec()


def smoke(spec: PSpec) -> PSpec:
    """Scale only: O's smoke (4 seeds from 23_009_100, 200 bootstrap draws) and N's smoke OC count."""
    return dataclasses.replace(spec, o=o_smoke(spec.o), oc_draws=spec.smoke_oc_draws)
