"""The M4 confirmation set (spec AC.4): situations (M, O) with all of M's attacks as candidates and a unique
best-multiplier move; pairs with the same M, different opponent type sets and different best moves; no (M, O) of the
L set; one pair per dedupe key; exactly 20 drawn with seed 301, else STOP_SET."""
import dataclasses

from flymon.ac import confirm
from flymon.ac.spec import SPEC
from flymon.battle.pool import by_species
from flymon.brain.h4_pairs import pool_vocabulary

ST, MI, _, _ = pool_vocabulary()


def test_type_mult_and_best_move():
    assert confirm.type_mult("ELECTRIC", ("WATER",)) == 2.0
    assert confirm.type_mult("GROUND", ("FIRE", "FLYING")) == 0.0
    assert confirm.type_mult("WATER", ("FIRE", "FLYING")) == 2.0
    assert confirm.best_move("Blastoise", "Charizard", ST, MI) == "Surf"
    assert confirm.best_move("Snorlax", "Snorlax", ST, MI) is None          # every attack x1: no unique best


def test_candidate_counts_are_the_plan_time_counts():
    c = confirm.candidate_pairs()["counts"]
    assert tuple(c) == confirm.FILTERS
    assert c == {"all": 1680, "not_l": 909, "unique_best": 183, "distinct_types": 183, "different_best": 102,
                 "deduped": 102}


def test_every_candidate_pair_obeys_ac4():
    lset = confirm.l_set_combos(ST, MI)
    assert len(lset) == 64
    rows = confirm.candidate_pairs()["pairs"]
    keys = set()
    for r in rows:
        assert r["o1"] != r["o2"] and r["me"] not in (r["o1"], r["o2"])
        assert (r["me"], r["o1"]) not in lset and (r["me"], r["o2"]) not in lset
        assert r["cands"] == list(by_species[r["me"]].attacks)
        assert sorted(r["o1_types"]) != sorted(r["o2_types"]) and r["best1"] != r["best2"]
        for o, b, m in ((r["o1"], r["best1"], r["mults1"]), (r["o2"], r["best2"], r["mults2"])):
            assert confirm.best_move(r["me"], o, ST, MI) == b
            assert m.count(max(m)) == 1 and r["cands"][m.index(max(m))] == b
        k = (r["me"], tuple(tuple(x) for x in r["key"][1]))
        assert k not in keys
        keys.add(k)


def test_exactly_20_drawn_with_301_and_deterministic():
    s = confirm.confirmation_set()
    rows = confirm.candidate_pairs()["pairs"]
    assert s["status"] == "OK" and len(s["pairs"]) == 20 and s["seed"] == 301 and s["n_candidates"] == 102
    assert all(p in rows for p in s["pairs"])
    assert s == confirm.confirmation_set() and len(s["digest"]) == 64
    assert s["l_set"] == {"n_turns": 64, "rng_seed": 20260927}
    assert confirm.draw(rows, 20, 301) == s["pairs"]
    assert confirm.draw(rows, 20, 302) != s["pairs"]


def test_too_few_candidates_is_stop_set():
    s = confirm.confirmation_set(dataclasses.replace(SPEC, n_pairs=103))
    assert s["status"] == "STOP_SET" and s["pairs"] == [] and s["digest"] is None and s["n_candidates"] == 102


def test_situations_of_candidates():
    sits = confirm.situations_of(confirm.candidate_pairs()["pairs"])
    assert len(sits) == 79 and len(set(sits)) == 79
    assert all(c == tuple(by_species[m].attacks) for m, _, c in sits)
