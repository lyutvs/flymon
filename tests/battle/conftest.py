"""Offline battle builder: a poke-env Battle populated by hand, no Showdown server needed."""
import logging

import pytest
from poke_env.battle import Battle

from flymon.battle.moves import gen1
from flymon.battle.pool import POOL, PoolMon

_STATS = {"atk": 200, "def": 200, "spa": 200, "spd": 200, "spe": 200}


def _request(my: PoolMon, bench: list) -> dict:
    ids = [gen1(m).id for m in my.attacks + my.support]
    active = {"moves": [{"move": m, "id": gen1(m).id, "pp": 16, "maxpp": 16, "target": "normal", "disabled": False}
                        for m in my.attacks + my.support]}
    side = [{"ident": f"p1: {my.species}", "details": f"{my.species}, L100", "condition": "300/300", "active": True,
             "stats": _STATS, "moves": ids, "baseAbility": "none", "item": "", "pokeball": "pokeball"}]
    for b in bench:
        side.append({"ident": f"p1: {b.species}", "details": f"{b.species}, L100", "condition": "300/300", "active": False,
                     "stats": _STATS, "moves": [gen1(m).id for m in b.attacks + b.support], "baseAbility": "none",
                     "item": "", "pokeball": "pokeball"})
    return {"active": [active], "side": {"name": "p1", "id": "p1", "pokemon": side}, "rqid": 2}


@pytest.fixture
def make_battle():
    def _make(my: PoolMon, opp_species: str, bench: list | None = None) -> Battle:
        bench = bench if bench is not None else [m for m in POOL if m.species != my.species][:2]
        b = Battle("battle-gen1ou-t", "p1", logging.getLogger("t"), gen=1)
        b.parse_message(["", "switch", f"p2a: {opp_species}", f"{opp_species}, L100", "300/300"])
        b.parse_request(_request(my, bench))
        return b
    return _make
