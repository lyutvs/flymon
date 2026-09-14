"""Opponents the fly plays against: poke-env's stock baselines, one account per index."""
from __future__ import annotations

from poke_env.player import Player, RandomPlayer, SimpleHeuristicsPlayer
from poke_env.ps_client import AccountConfiguration

KINDS = {"random": RandomPlayer, "heuristic": SimpleHeuristicsPlayer}


def make_opponent(kind: str, n: int, server_configuration, team: str) -> Player:
    cls = KINDS[kind]
    return cls(account_configuration=AccountConfiguration(f"fm-{kind}-{n}", None), battle_format="gen1ou",
               server_configuration=server_configuration, team=team, max_concurrent_battles=1)
