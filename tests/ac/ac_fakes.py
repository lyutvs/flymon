"""Shared fakes for tests/ac: a stub E-grid codebook on 49 synthetic glomeruli (tests/agent/test_encode_grid.py's),
stub battles and moves; Task 12 adds a fake pool and fake battles."""
from types import SimpleNamespace

from flymon.ac.tb import TBEncoder
from flymon.agent import e_codebook as ecb
from flymon.agent import encode_grid as eg
from flymon.agent.e_spec import SPEC as E_SPEC
from flymon.agent.e_spec import smoke as e_smoke
from flymon.brain.h4_pairs import pool_vocabulary

ST, MI, MON, MOVE = pool_vocabulary()
GLOMS = [f"ORN_G{i:02d}" for i in range(49)]
RC = {g: 1 + (i % 5) for i, g in enumerate(GLOMS)}
DRIVE = {g: float(10 + 50 * i) for i, g in enumerate(GLOMS)}
CB = eg.Codebook(ecb.cells(MOVE, MON), ecb.build(DRIVE, 2, e_smoke(E_SPEC))["codebook"])


class StubPops:
    receptor_types = {g: list(range(RC[g])) for g in GLOMS}


def Mon(species, hp=1.0):
    return SimpleNamespace(species=species, current_hp_fraction=hp)


def Mv(move_id):
    return SimpleNamespace(id=move_id)


def battle(me, opp):
    return SimpleNamespace(active_pokemon=Mon(me), opponent_active_pokemon=Mon(opp))


def stub_grid():
    return eg.GridEncoder(StubPops(), CB, "norm")


def stub_tb():
    return TBEncoder(StubPops(), CB, "norm")
