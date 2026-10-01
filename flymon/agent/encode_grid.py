"""Spec 3.4: E-grid situation odours. odour(move type, opponent types) = union of the cells' codewords; strengths
proportional to 1/receptor count with mean 1 (the E0 convention, h4_pairs.odour); `norm` halves a dual-typed odour
(k / |glomeruli|). My type, HP and power are not encoded."""
from __future__ import annotations

import numpy as np
from poke_env.data.normalize import to_id_str

from ..battle.pool import POOL


class Codebook:
    def __init__(self, cells, words):
        self.cells = [tuple(c) for c in cells]
        self.words = [tuple(w) for w in words]
        self._ix = {c: i for i, c in enumerate(self.cells)}
        self.k = len(self.words[0])

    def word(self, move_type: str, opp_type: str) -> tuple:
        return self.words[self._ix[(move_type, opp_type)]]


def glomeruli(cb: Codebook, move_type: str, opp_types) -> tuple:
    out = [g for t in opp_types for g in cb.word(move_type, t)]
    if len(set(out)) != len(out):
        raise ValueError(f"glomerulus collision for {move_type} vs {opp_types}")
    return tuple(out)


def odour(receptor_counts: dict, cb: Codebook, move_type: str, opp_types, dual_rule: str) -> dict:
    gl = glomeruli(cb, move_type, opp_types)
    inv = np.array([1.0 / receptor_counts[g] for g in gl]); inv /= inv.mean()
    if dual_rule == "norm" and len(opp_types) == 2:
        inv *= cb.k / len(gl)
    elif dual_rule not in ("full", "norm"):
        raise ValueError(dual_rule)
    return {g: float(s) for g, s in zip(gl, inv)}


def reachable(species_types, move_types) -> list:
    opp_sets = sorted({tuple(sorted(v)) for v in species_types.values()})
    return [(m, o) for o in opp_sets for m in move_types]


def unique_odours(cb: Codebook, species_types, move_types) -> bool:
    sets = [frozenset(glomeruli(cb, m, o)) for m, o in reachable(species_types, move_types)]
    return len(set(sets)) == len(sets)


def cap_ok(receptor_counts, cb, species_types, move_types, dual_rule, s, max_rate_hz, cap_hz) -> bool:
    return all(max_rate_hz * s * v <= cap_hz
               for m, o in reachable(species_types, move_types)
               for v in odour(receptor_counts, cb, m, o, dual_rule).values())


class GridEncoder:
    """encode.Encoder's interface on E-grid odours."""

    def __init__(self, pops, cb: Codebook, dual_rule: str):
        from ..brain.h4_pairs import pool_vocabulary
        self.rc = {str(t): len(v) for t, v in pops.receptor_types.items()}
        self.cb, self.rule = cb, dual_rule
        self.species_types, self.move_info, _, _ = pool_vocabulary()
        self.species_by_id = {to_id_str(m.species): m.species for m in POOL}
        self.move_by_id = {to_id_str(a): a for m in POOL for a in m.attacks}

    def _species(self, mon) -> str:
        return self.species_by_id[to_id_str(mon.species)]

    def odour(self, battle, move) -> dict:
        opp = self._species(battle.opponent_active_pokemon)
        mtype = self.move_info[self.move_by_id[move.id]][0]
        return odour(self.rc, self.cb, mtype, tuple(sorted(self.species_types[opp])), self.rule)

    def situation_key(self, battle, cands) -> tuple:
        return (self._species(battle.active_pokemon), self._species(battle.opponent_active_pokemon),
                tuple(sorted(m.id for m in cands)))