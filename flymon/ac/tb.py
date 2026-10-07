"""FLY-TB's encoder (spec AC.2, red team AC.9 1): the odour of move type m is the uniform mean, over the 16 POOL
species' type sets T, of the E-grid odour odour(m, T), rescaled so its total strength equals the mean total strength of
those 16 FLY odours (the same move type). It does not depend on the opponent, differs between move types and is
never empty (a TB odour with the opponent channel simply removed would be no odour at all - a blind control)."""
from __future__ import annotations

import numpy as np

from ..agent.encode_grid import GridEncoder, odour
from ..battle.pool import POOL


class TBEncoder(GridEncoder):
    def __init__(self, pops, cb, dual_rule: str):
        super().__init__(pops, cb, dual_rule)
        self._tb: dict = {}
        self._fly_total: dict = {}

    def _fly_odours(self, move_type: str) -> list:
        return [odour(self.rc, self.cb, move_type, tuple(sorted(self.species_types[m.species])), self.rule)
                for m in POOL]

    def fly_mean_total(self, move_type: str) -> float:
        if move_type not in self._fly_total:
            self._fly_total[move_type] = float(np.mean([sum(o.values()) for o in self._fly_odours(move_type)]))
        return self._fly_total[move_type]

    def tb_odour(self, move_type: str) -> dict:
        if move_type not in self._tb:
            fly = self._fly_odours(move_type)
            acc: dict = {}
            for o in fly:
                for g, s in o.items():
                    acc[g] = acc.get(g, 0.0) + s / len(fly)
            k = self.fly_mean_total(move_type) / sum(acc.values())
            self._tb[move_type] = {g: float(s * k) for g, s in sorted(acc.items())}
        return dict(self._tb[move_type])

    def odour(self, battle, move) -> dict:
        return self.tb_odour(self.move_info[self.move_by_id[move.id]][0])
