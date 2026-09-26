"""Battle state -> E0 odour (spec 3.3), with exactly H.4's vocabulary, channel assignment and odour function
(`flymon.brain.h4_pairs`), so a battle situation and an H.4 / L pair with the same content are the same odour."""
from __future__ import annotations

from poke_env.data.normalize import to_id_str

from ..battle.pool import POOL
from ..brain import h4_pairs


class Encoder:
    def __init__(self, pops):
        self.pops = pops
        self.species_types, self.move_info, mon_t, mv_t = h4_pairs.pool_vocabulary()
        self.chan = h4_pairs.e0_channels(pops, mon_t, mv_t)
        self.species_by_id = {to_id_str(m.species): m.species for m in POOL}
        self.move_by_id = {to_id_str(a): a for m in POOL for a in m.attacks}

    def _species(self, mon) -> str:
        return self.species_by_id[to_id_str(mon.species)]

    def hp_bins(self, battle) -> tuple[str, str]:
        return (h4_pairs.hp_bin(battle.active_pokemon.current_hp_fraction),
                h4_pairs.hp_bin(battle.opponent_active_pokemon.current_hp_fraction))

    def odour(self, battle, move) -> dict:
        me, opp = self._species(battle.active_pokemon), self._species(battle.opponent_active_pokemon)
        mtype, bp = self.move_info[self.move_by_id[move.id]]
        return h4_pairs.odour(self.pops, self.chan, list(self.species_types[me]), list(self.species_types[opp]), mtype,
                              bp, battle.active_pokemon.current_hp_fraction,
                              battle.opponent_active_pokemon.current_hp_fraction)

    def situation_key(self, battle, cands) -> tuple:
        my_bin, op_bin = self.hp_bins(battle)
        return (self._species(battle.active_pokemon), self._species(battle.opponent_active_pokemon), my_bin, op_bin,
                tuple(sorted(m.id for m in cands)))
