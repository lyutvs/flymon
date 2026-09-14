"""The 16-Pokémon Gen 1 OU-legal team pool. Attacks are the fly's candidates; support is coach-only."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PoolMon:
    species: str
    attacks: tuple
    support: tuple
    base_speed: int


POOL = [
    PoolMon("Blastoise", ("Surf", "Earthquake", "Strength"), ("Reflect",), 78),
    PoolMon("Snorlax", ("Earthquake", "Surf", "Strength"), ("Amnesia",), 30),
    PoolMon("Exeggutor", ("Psychic", "Mega Drain"), ("Reflect",), 55),
    PoolMon("Chansey", ("Thunderbolt", "Psychic"), ("Soft-Boiled", "Thunder Wave"), 50),
    PoolMon("Rhydon", ("Earthquake", "Rock Slide", "Strength"), (), 40),
    PoolMon("Slowbro", ("Surf", "Psychic"), ("Amnesia", "Thunder Wave"), 30),
    PoolMon("Lapras", ("Surf", "Psychic", "Thunderbolt"), (), 60),
    PoolMon("Venusaur", ("Mega Drain", "Cut"), ("Swords Dance",), 80),
    PoolMon("Charizard", ("Flamethrower", "Earthquake", "Strength"), ("Swords Dance",), 100),
    PoolMon("Nidoking", ("Earthquake", "Thunderbolt", "Surf"), (), 85),
    PoolMon("Kangaskhan", ("Earthquake", "Surf", "Strength"), (), 90),
    PoolMon("Hypno", ("Psychic", "Tri Attack"), ("Thunder Wave", "Reflect"), 67),
    PoolMon("Dragonite", ("Surf", "Thunderbolt"), ("Agility",), 80),
    PoolMon("Machamp", ("Earthquake", "Rock Slide", "Strength"), (), 55),
    PoolMon("Tentacruel", ("Surf", "Mega Drain"), ("Swords Dance",), 100),
    PoolMon("Poliwrath", ("Surf", "Earthquake", "Psychic"), ("Amnesia",), 70),
]
by_species = {m.species: m for m in POOL}


def export_text(mon: PoolMon) -> str:
    return "\n".join([mon.species] + [f"- {m}" for m in mon.attacks + mon.support])


def team_export(mons: list) -> str:
    return "\n\n".join(export_text(m) for m in mons)
