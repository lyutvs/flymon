import numpy as np
import pytest
from flymon.agent import encode_grid as eg, e_codebook as cb
from flymon.agent.e_spec import SPEC, smoke
from flymon.brain.h4_pairs import pool_vocabulary

ST, MI, MON, MOVE = pool_vocabulary()
GLOMS = [f"ORN_G{i:02d}" for i in range(49)]
RC = {g: 1 + (i % 5) for i, g in enumerate(GLOMS)}
DRIVE = {g: float(10 + 50 * i) for i, g in enumerate(GLOMS)}
BUILT = cb.build(DRIVE, 2, smoke(SPEC))
CB = eg.Codebook(cb.cells(MOVE, MON), BUILT["codebook"])

def test_reachable_is_112():
    r = eg.reachable(ST, MOVE)
    assert len(r) == 112 and len(set(r)) == 112
    singles = {o for _, o in r if len(o) == 1}
    assert singles == {("FIGHTING",), ("NORMAL",), ("PSYCHIC",), ("WATER",)}

def test_odour_full_and_norm():
    single = eg.odour(RC, CB, "WATER", ("NORMAL",), "full")
    assert len(single) == 2 and np.isclose(np.mean(list(single.values())), 1.0)
    dual_f = eg.odour(RC, CB, "WATER", ("GRASS", "POISON"), "full")
    dual_n = eg.odour(RC, CB, "WATER", ("GRASS", "POISON"), "norm")
    assert len(dual_f) == 4 and np.isclose(np.mean(list(dual_f.values())), 1.0)
    assert all(np.isclose(dual_n[g], 0.5 * dual_f[g]) for g in dual_f)
    assert eg.odour(RC, CB, "WATER", ("NORMAL",), "norm") == single
    inv = {g: 1 / RC[g] for g in single}; m = np.mean(list(inv.values()))
    assert all(np.isclose(single[g], inv[g] / m) for g in single)

def test_axis_pairs_share_no_glomerulus():
    a = set(eg.glomeruli(CB, "WATER", ("GRASS", "POISON")))
    b = set(eg.glomeruli(CB, "WATER", ("FIRE", "FLYING")))
    c = set(eg.glomeruli(CB, "GROUND", ("GRASS", "POISON")))
    assert not (a & b) and not (a & c)

def test_unique_odours():
    assert eg.unique_odours(CB, ST, MOVE)

def test_cap_ok():
    assert eg.cap_ok(RC, CB, ST, MOVE, "full", 0.35, 200.0, 333.3)
    assert not eg.cap_ok(RC, CB, ST, MOVE, "full", 2.0, 200.0, 333.3)

def test_grid_encoder_matches_odour():
    class Mon:
        def __init__(self, species, hp=1.0): self.species, self.current_hp_fraction = species, hp
    class Mv:
        def __init__(self, id): self.id = id
    class B:
        active_pokemon = Mon("Blastoise"); opponent_active_pokemon = Mon("Venusaur")
    class P:
        receptor_types = {g: list(range(RC[g])) for g in GLOMS}
    enc = eg.GridEncoder(P(), CB, "norm")
    assert enc.odour(B(), Mv("surf")) == eg.odour(RC, CB, "WATER", ("GRASS", "POISON"), "norm")