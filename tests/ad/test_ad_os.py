"""FLY-OS's encoder (AD.1, AD.5 4 - a smoke gate): the opponent-type part of the E-grid odour is replaced by the type
set of one of the 16 POOL species drawn uniformly and independently of the true opponent, fresh per presentation (one
decision = one turn's candidates); the key is (global fly, schedule battle id, battle.turn), never the Showdown tag."""
import json

import numpy as np

from flymon.ad import os_encoder as oe
from flymon.ad.spec import SPEC
from flymon.agent.policy import derive_seed
from tests.ac.ac_fakes import CB, MI, Mv, StubPops, battle, stub_grid

MOVES = ["Surf", "Psychic", "Thunderbolt"]


def enc(gfly=40):
    return oe.OSEncoder(StubPops(), CB, "norm", gfly, SPEC.os_seed)


def turn_battle(opp, turn, tag="battle-gen1ou-1"):
    b = battle("Lapras", opp)
    b.turn, b.battle_tag = turn, tag
    return b


def mv(name):
    return Mv(next(k for k, v in stub_grid().move_by_id.items() if v == name))


def test_species_are_the_16_pool_species():
    assert len(oe.SPECIES) == 16 and oe.SPECIES[0] == "Blastoise" and len(set(oe.SPECIES)) == 16


def test_draw_keys_on_global_fly_schedule_id_and_turn():
    e = enc(40)
    e.set_battle("DL-f40-b007")
    b = turn_battle("Charizard", 3)
    want = oe.SPECIES[int(np.random.default_rng(derive_seed(307, "os", 40, "DL-f40-b007", 3)).integers(0, 16))]
    assert e.drawn(b) == want
    seeds = {oe.battle_seed(307, g, bid, t) for g, bid, t in ((40, "DL-f40-b007", 3), (41, "DL-f40-b007", 3),
                                                                (40, "DL-f40-b008", 3), (40, "DL-f40-b007", 4))}
    assert len(seeds) == 4                                  # every key part changes the draw stream
    assert e.drawn(turn_battle("Charizard", 3)) == want     # the same key reproduces the same species


def test_retry_with_a_new_showdown_tag_draws_the_same():
    e = enc()
    e.set_battle("DL-f40-b007")
    a = [e.drawn(turn_battle("Charizard", t, "battle-gen1ou-11")) for t in range(1, 30)]
    b = [e.drawn(turn_battle("Charizard", t, "battle-gen1ou-99")) for t in range(1, 30)]
    assert a == b


def test_candidates_of_one_turn_share_the_draw_and_turns_redraw():
    e = enc()
    e.set_battle("DL-f40-b000")
    turns = []
    for t in range(1, 40):
        b = turn_battle("Charizard", t)
        d = e.drawn(b)
        for m in MOVES:
            o = e.odour(b, mv(m))
            assert o == e.odour_for(MI[m][0], d)            # every candidate of the turn uses the same species
        turns.append(d)
    assert len(set(turns)) > 1                              # independent per turn (record-only diversity elsewhere)


def test_os_odour_is_the_fly_encoder_odour_of_the_drawn_species_byte_identical():
    e, g = enc(), stub_grid()
    e.set_battle("DL-f40-b002")
    for t in range(1, 25):
        b = turn_battle("Charizard", t)
        d = e.drawn(b)
        for m in MOVES:
            assert json.dumps(e.odour(b, mv(m))) == json.dumps(g.odour(battle("Lapras", d), mv(m)))


def test_shadow_records_true_drawn_and_type_match():
    e = enc()
    e.set_battle("DL-f40-b001")
    b = turn_battle("Snorlax", 5)
    sh = e.shadow(b)
    assert sh["true"] == "Snorlax" and sh["drawn"] == e.drawn(b)
    assert sh["same_types"] == (sorted(e.species_types["Snorlax"]) == sorted(e.species_types[sh["drawn"]]))


def test_set_battle_is_required():
    e = enc()
    try:
        e.drawn(turn_battle("Charizard", 1))
    except RuntimeError as err:
        assert "set_battle" in str(err)
    else:
        raise AssertionError("drawn() without set_battle must raise")


def test_uniform_and_independent_of_the_true_species():
    t = oe.gate_table(SPEC.os_seed, SPEC.os_gate_draws)
    assert t.shape == (16, 16) and (t.sum(1) == SPEC.os_gate_draws).all()
    g = oe.gate(t, SPEC.os_freq_tol, SPEC.os_chi2_p)
    assert g["ok"] and g["max_freq_dev"] <= 0.01 and g["chi2_p"] >= 0.001
    assert round(g["max_freq_dev"], 5) == 0.00625                     # the plan-time reading


def test_gate_fails_a_biased_table():
    t = np.full((16, 16), 1000)
    t[np.arange(16), np.arange(16)] += 3000                           # drawn == true too often
    assert not oe.gate(t, 0.01, 0.001)["ok"]


def test_situation_odours_use_their_key_and_record_draws():
    g = stub_grid()
    pairs = [dict(me="Lapras", o1="Charizard", o2="Exeggutor", cands=MOVES)]
    ods, draws = oe.os_pair_odours(g, pairs, 40, 307, "os-sit")
    assert len(ods) == 1 and len(ods[0][0]) == 3 and len(draws) == 2
    d0 = oe.SPECIES[oe.draw_index(derive_seed(307, "os-sit", 40, 0, 0))]
    assert draws[0] == dict(pair=0, side=0, true="Charizard", drawn=d0,
                            same_types=sorted(g.species_types["Charizard"]) == sorted(g.species_types[d0]))
    assert ods[0][0] == [oe.odour_for_species(g, MI[m][0], d0) for m in MOVES]
    _, other = oe.os_pair_odours(g, pairs, 40, 307, "os-sit-paired")
    _, again = oe.os_pair_odours(g, pairs, 40, 307, "os-sit")
    assert again == draws and [x["drawn"] for x in other] == [
        oe.SPECIES[oe.draw_index(derive_seed(307, "os-sit-paired", 40, 0, s))] for s in (0, 1)]


def test_total_strength_equals_k():
    assert oe.total_strength_error(stub_grid()) <= SPEC.strength_tol
