import asyncio

from flymon.battle.moves import gen1
from flymon.battle.pool import by_species
from flymon.battle.providers import MaxDamageProvider, RandomProvider


def test_random_provider_is_seeded(make_battle):
    b = make_battle(by_species["Lapras"], "Venusaur")
    cands = [gen1(m) for m in ("Surf", "Psychic", "Thunderbolt")]
    a = [asyncio.run(RandomProvider(3).decide(b, cands, {})) for _ in range(10)]
    c = [asyncio.run(RandomProvider(3).decide(b, cands, {})) for _ in range(10)]
    assert a == c and set(a) <= {0, 1, 2}


def test_max_damage_provider_prefers_super_effective(make_battle):
    b = make_battle(by_species["Lapras"], "Venusaur")
    cands = [gen1(m) for m in ("Surf", "Psychic", "Thunderbolt")]
    assert asyncio.run(MaxDamageProvider().decide(b, cands, {})) == 1   # psychic vs grass/poison
