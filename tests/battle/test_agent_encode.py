"""flymon.agent config + encoder: the battle state becomes the same E0 odour H.4 used."""
import numpy as np
import pytest

from flymon.agent.config import AgentConfig, config_hash
from flymon.agent.encode import Encoder
from flymon.battle.pool import by_species
from flymon.brain import h4_pairs
from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from pathlib import Path

NPZ = Path("data/malecns.npz")
needs_data = pytest.mark.skipif(not NPZ.exists(), reason="data/malecns.npz not built")


def test_config_hash_changes_with_any_field():
    a = AgentConfig(params=Params(), z={"A": (1.0, 2.0), "P": (3.0, 4.0)}, readout={"A": "MBON13", "P": "MBON05"})
    b = AgentConfig(params=Params(), z={"A": (1.0, 2.0), "P": (3.0, 4.5)}, readout={"A": "MBON13", "P": "MBON05"})
    assert config_hash(a) == config_hash(a)
    assert config_hash(a) != config_hash(b)


@needs_data
def test_battle_odour_equals_h4_odour(make_battle):
    pops = Populations.from_connectome(Connectome.load(str(NPZ)))
    enc = Encoder(pops)
    battle = make_battle(by_species["Blastoise"], "Charizard")
    move = [m for m in battle.available_moves if m.id == "surf"][0]
    st, mi, mon_t, mv_t = h4_pairs.pool_vocabulary()
    chan = h4_pairs.e0_channels(pops, mon_t, mv_t)
    my_hp = battle.active_pokemon.current_hp_fraction
    op_hp = battle.opponent_active_pokemon.current_hp_fraction
    want = h4_pairs.odour(pops, chan, list(st["Blastoise"]), list(st["Charizard"]), mi["Surf"][0], mi["Surf"][1], my_hp, op_hp)
    assert enc.odour(battle, move) == want


@needs_data
def test_situation_key_is_species_hp_bins_and_sorted_candidates(make_battle):
    enc = Encoder(Populations.from_connectome(Connectome.load(str(NPZ))))
    battle = make_battle(by_species["Blastoise"], "Charizard")
    cands = [m for m in battle.available_moves if m.id in ("surf", "earthquake", "strength")]
    key = enc.situation_key(battle, cands[::-1])
    assert key == ("Blastoise", "Charizard", "high", "high", ("earthquake", "strength", "surf"))
