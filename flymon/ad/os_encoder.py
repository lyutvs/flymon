"""FLY-OS's encoder (spec AD.0 5, AD.1, AD.5 4): the odour is odour(rc, cb, move type, the drawn species' type set,
rule), the FLY encoder's own function. The drawn species is one of the 16 POOL species, uniform and independent of the
true opponent (the true one included: drawing from the other 15 would leave reverse information, AD.9 1), drawn fresh
for every presentation. A battle presentation is one decision - every candidate of one turn shares the draw - keyed by
derive_seed(307, "os", global fly, schedule battle id, battle.turn): the schedule id the player set with set_battle,
never Showdown's battle tag, so a retried battle draws the same. A situation presentation is (global fly, pair, side)
under key "os-sit" / "os-sit-paired" / "os-sit-new", no point (AC.5: changes between points come from the weights).
Under the `norm` rule every E-grid odour has total strength k (single type: k glomeruli of mean 1; dual: 2k of 1/2), so
FLY-OS's total equals FLY's at every presentation (AD.1)."""
from __future__ import annotations

import numpy as np

from ..agent.encode_grid import GridEncoder, odour
from ..agent.policy import derive_seed
from ..battle.pool import POOL

SPECIES = tuple(m.species for m in POOL)


def draw_index(seed: int) -> int:
    return int(np.random.default_rng(int(seed)).integers(0, len(SPECIES)))


def battle_seed(os_seed: int, gfly: int, battle_id: str, turn: int) -> int:
    return derive_seed(int(os_seed), "os", int(gfly), str(battle_id), int(turn))


def situation_seed(os_seed: int, key: str, gfly: int, pair: int, side: int) -> int:
    return derive_seed(int(os_seed), str(key), int(gfly), int(pair), int(side))


def odour_for_species(enc: GridEncoder, move_type: str, species: str) -> dict:
    return odour(enc.rc, enc.cb, move_type, tuple(sorted(enc.species_types[species])), enc.rule)


def same_types(enc: GridEncoder, a: str, b: str) -> bool:
    return sorted(enc.species_types[a]) == sorted(enc.species_types[b])


class OSEncoder(GridEncoder):
    def __init__(self, pops, cb, dual_rule: str, gfly: int, os_seed: int):
        super().__init__(pops, cb, dual_rule)
        self.gfly, self.os_seed = int(gfly), int(os_seed)
        self.battle_id = None

    def set_battle(self, battle_id: str) -> None:
        self.battle_id = str(battle_id)

    def drawn(self, battle) -> str:
        if self.battle_id is None:
            raise RuntimeError("OSEncoder: set_battle(schedule battle id) must run before the first decision")
        return SPECIES[draw_index(battle_seed(self.os_seed, self.gfly, self.battle_id, battle.turn))]

    def odour_for(self, move_type: str, species: str) -> dict:
        return odour_for_species(self, move_type, species)

    def odour(self, battle, move) -> dict:
        return self.odour_for(self.move_info[self.move_by_id[move.id]][0], self.drawn(battle))

    def shadow(self, battle) -> dict:
        true, d = self._species(battle.opponent_active_pokemon), self.drawn(battle)
        return dict(true=true, drawn=d, same_types=same_types(self, true, d))


def os_pair_odours(enc: GridEncoder, pairs, gfly: int, os_seed: int, key: str) -> tuple:
    ods, draws = [], []
    for i, p in enumerate(pairs):
        sides = []
        for side, opp in ((0, p["o1"]), (1, p["o2"])):
            d = SPECIES[draw_index(situation_seed(os_seed, key, gfly, i, side))]
            sides.append([odour_for_species(enc, enc.move_info[m][0], d) for m in p["cands"]])
            draws.append(dict(pair=i, side=side, true=opp, drawn=d, same_types=same_types(enc, opp, d)))
        ods.append((sides[0], sides[1]))
    return ods, draws


def gate_table(os_seed: int, draws: int) -> np.ndarray:
    t = np.zeros((len(SPECIES), len(SPECIES)), np.int64)
    for r, sp in enumerate(SPECIES):
        for n in range(int(draws)):
            t[r, draw_index(derive_seed(int(os_seed), "os-gate", sp, n))] += 1
    return t


def gate(table, tol: float, p_min: float) -> dict:
    from scipy.stats import chi2_contingency
    t = np.asarray(table)
    freq = t / t.sum(1, keepdims=True)
    dev = float(np.abs(freq - 1.0 / t.shape[1]).max())
    p = float(chi2_contingency(t)[1])
    return dict(ok=bool(dev <= tol and p >= p_min), max_freq_dev=dev, chi2_p=p, tol=float(tol), p_min=float(p_min))


def total_strength_error(enc: GridEncoder, species=SPECIES) -> float:
    mtypes = sorted({v[0] for v in enc.move_info.values()})
    return max(abs(sum(odour_for_species(enc, m, s).values()) - enc.cb.k) for m in mtypes for s in species)
