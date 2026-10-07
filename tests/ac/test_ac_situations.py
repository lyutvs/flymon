"""Situation-pair evaluation (AC.5): argmax with the seeded tie rule on the confirmation set, plasticity and exploration
off, weights unchanged; a fly's switch rate = share of pairs where it picks the best move in both situations."""
import numpy as np
import pytest

from flymon.ac import situations as si
from flymon.agent import encode_grid as eg
from tests.ac.ac_fakes import CB, MI, ST, Mv, battle, stub_grid, stub_tb

Z = {"A": (0.0, 1.0), "P": (0.0, 1.0)}           # V = A - P


class Cfg:
    z, strength, settle_ms, read_ms = Z, 1.0, 800.0, 600.0


PAIRS = [dict(me="Lapras", o1="Charizard", o2="Exeggutor", cands=["Surf", "Psychic", "Thunderbolt"],
              best1="Surf", best2="Psychic"),
         dict(me="Lapras", o1="Gyarados", o2="Exeggutor", cands=["Surf", "Psychic", "Thunderbolt"],
              best1="Thunderbolt", best2="Psychic")]


class FakePool:
    """decide_batch returns per request [n_cand, 1 + 1 + 2] counts (A cell, P cell, two KCs) from a script."""

    def __init__(self, script, mutate=False):
        self.script, self.mutate = script, mutate
        self.w = {f: np.ones(4, np.float32) for f in range(3)}  # flies 0-2 (tie test uses fly_seed % 3)
        self.calls = []

    def decide_batch(self, reqs, strength, settle_ms, read_ms, idx):
        self.calls.append(reqs)
        if self.mutate:
            self.w[0] = self.w[0] * np.float32(0.5)
        return [np.asarray(self.script[(i // 2, i % 2)], float) for i, _ in enumerate(reqs)]


def _counts(winner, n=3, tie=False):
    rows = [[0, 0, 1, 1] for _ in range(n)]
    rows[winner][0] = 5
    if tie:
        rows[(winner + 1) % n][0] = 5
    return rows


def _eval(pool):
    ods = [([{}] * 3, [{}] * 3)] * 2
    return si.evaluate(pool, 0, 10, PAIRS, ods, Cfg(), np.array([0]), np.array([1]), np.array([2, 3]), 305)


def test_switch_rate_counts_pairs_right_in_both_situations():
    pool = FakePool({(0, 0): _counts(0), (0, 1): _counts(1), (1, 0): _counts(0), (1, 1): _counts(1)})
    rec = _eval(pool)
    assert rec["switched"] == [1, 0] and rec["rate"] == 0.5 and rec["point"] == 10 and rec["fly"] == 0
    assert rec["frozen"] and rec["ties"] == 0 and rec["n_candidates"] == 12
    assert rec["a_zero"] == 8 and rec["p_zero"] == 12
    seeds = [s for _, _, s in pool.calls[0]]
    assert len(set(seeds)) == 4 and [f for f, _, _ in pool.calls[0]] == [0, 0, 0, 0]


def test_situation_tie_is_seeded_and_counted():
    picks = set()
    for fly_seed in range(12):
        pool = FakePool({k: _counts(0, tie=True) for k in [(0, 0), (0, 1), (1, 0), (1, 1)]})
        rec = si.evaluate(pool, fly_seed % 3, 0, PAIRS, [([{}] * 3, [{}] * 3)] * 2, Cfg(), np.array([0]),
                          np.array([1]), np.array([2, 3]), 305 + fly_seed)
        assert rec["ties"] == 4
        picks |= {s["pick"] for s in rec["situations"]}
    assert picks == {0, 1}


def test_weight_change_is_flagged():
    pool = FakePool({k: _counts(0) for k in [(0, 0), (0, 1), (1, 0), (1, 1)]}, mutate=True)
    assert _eval(pool)["frozen"] is False


def test_offline_odour_equals_the_battle_odour():
    g, tb = stub_grid(), stub_tb()
    assert si.offline_odour(g, "Surf", "Charizard") == g.odour(battle("Lapras", "Charizard"), Mv("surf"))
    assert si.offline_odour(g, "Surf", "Charizard") == eg.odour(g.rc, CB, MI["Surf"][0], tuple(sorted(ST["Charizard"])),
                                                                "norm")
    assert si.offline_odour(tb, "Surf", "Charizard") == tb.odour(battle("Lapras", "Exeggutor"), Mv("surf"))
    o1, o2 = si.pair_odours(g, PAIRS[:1])[0]
    assert len(o1) == len(o2) == 3 and o1[0] == si.offline_odour(g, "Surf", "Charizard")


def test_floor_contact():
    class P:
        flies = [type("F", (), {"shuffle_seed": None})()]
        w0 = {None: np.array([1.0, 1.0, 1.0, 1.0, 1.0], np.float32)}
        w = {0: np.array([0.2, 0.5, 1.0, 0.2, 0.1], np.float32)}
    masks = {"PAM08": np.array([1, 1, 1, 0, 0], bool), "PPL105": np.array([0, 0, 0, 1, 0], bool)}
    assert si.floor_contact(P(), 0, masks, ("PAM08", "PPL105"), 0.2) == pytest.approx(2 / 3)
    P.w = {0: P.w0[None].copy()}
    assert si.floor_contact(P(), 0, masks, ("PAM08", "PPL105"), 0.2) is None
