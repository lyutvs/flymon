"""Fixed protocol logs -> attribution outcomes (spec 3.4)."""
from flymon.battle.attribution import TurnAttributor


def _lines(text):
    return [l.split("|") for l in text.strip().splitlines()]


CASE_DIRECT_SE_KO = """
|turn|3
|move|p1a: Lapras|Psychic|p2a: Venusaur
|-supereffective|p2a: Venusaur
|-damage|p2a: Venusaur|0 fnt
|faint|p2a: Venusaur
|upkeep
"""
CASE_RESIDUAL_KO = """
|turn|4
|move|p1a: Charizard|Flamethrower|p2a: Exeggutor
|-supereffective|p2a: Exeggutor
|-damage|p2a: Exeggutor|12/100
|move|p2a: Exeggutor|Psychic|p1a: Charizard
|-damage|p1a: Charizard|150/300
|-damage|p2a: Exeggutor|0 fnt|[from] brn
|faint|p2a: Exeggutor
|upkeep
"""
CASE_SUBSTITUTE = """
|turn|5
|move|p1a: Snorlax|Earthquake|p2a: Chansey
|-activate|p2a: Chansey|Substitute|[damage]
|upkeep
"""
CASE_MISS = """
|turn|6
|move|p1a: Rhydon|Rock Slide|p2a: Dragonite|[miss]
|-miss|p1a: Rhydon|p2a: Dragonite
|upkeep
"""
CASE_CANT = """
|turn|7
|cant|p1a: Slowbro|par
|move|p2a: Machamp|Earthquake|p1a: Slowbro
|-damage|p1a: Slowbro|100/300
|upkeep
"""
CASE_SWITCH_TURN = """
|turn|8
|switch|p1a: Chansey|Chansey, L100|300/300
|move|p2a: Machamp|Earthquake|p1a: Chansey
|-damage|p1a: Chansey|150/300
|upkeep
"""
CASE_DOUBLE_FAINT = """
|turn|9
|switch|p2a: Gengar|Gengar, L100|100/100
|move|p1a: Jolteon|Thunderbolt|p2a: Gengar
|-damage|p2a: Gengar|0 fnt
|faint|p2a: Gengar
|-damage|p1a: Jolteon|0 fnt|[from] psn
|faint|p1a: Jolteon
|upkeep
"""
CASE_EFFECTIVENESS_TAG = """
|turn|10
|switch|p2a: Starmie|Starmie, L100|100/100
|move|p1a: Zapdos|Thunderbolt|p2a: Starmie
|-supereffective|p2a: Starmie
|-damage|p2a: Starmie|40/100
|upkeep
"""


def run(case, **kwargs):
    a = TurnAttributor("p1")
    for l in _lines(case):
        a.feed(l)
    return a.end_turn(**kwargs)


def test_direct_super_effective_ko():
    o = run(CASE_DIRECT_SE_KO)
    assert o.move_id == "psychic" and o.effectiveness == "super" and o.target_fainted_by_me and o.dealt_frac == 1.0 and not o.uncertain


def test_residual_ko_not_attributed():
    o = run(CASE_RESIDUAL_KO)
    assert o.move_id == "flamethrower" and o.dealt_frac == 0.88 and not o.target_fainted_by_me and "residual" in " ".join(o.notes)


def test_substitute_gives_no_signal():
    o = run(CASE_SUBSTITUTE)
    assert o.direct_damage == 0 and o.dealt_frac == 0.0 and o.uncertain


def test_miss_and_cant_are_no_signal():
    assert run(CASE_MISS).missed and run(CASE_CANT).no_action


def test_switch_turn_has_no_move():
    o = run(CASE_SWITCH_TURN)
    assert o.move_id is None and o.no_action


def test_double_faint_attributes_only_the_target():
    o = run(CASE_DOUBLE_FAINT)
    assert o.move_id == "thunderbolt"
    assert o.target_fainted_by_me and o.dealt_frac == 1.0 and not o.uncertain


def test_effectiveness_mismatch_with_expected_multiplier():
    o = run(CASE_EFFECTIVENESS_TAG, expected_multiplier=0.5)
    assert o.effectiveness == "super" and o.uncertain
    assert "effectiveness mismatch" in " ".join(o.notes)


def test_effectiveness_agreement_with_expected_multiplier():
    o = run(CASE_EFFECTIVENESS_TAG, expected_multiplier=2.0)
    assert o.effectiveness == "super" and not o.uncertain
    assert "effectiveness mismatch" not in " ".join(o.notes)


def test_has_block_tracks_my_move_within_the_turn():
    a = TurnAttributor("p1")
    assert not a.has_block()
    for l in _lines(CASE_SWITCH_TURN):
        a.feed(l)
    assert not a.has_block()
    for l in _lines(CASE_MISS):
        a.feed(l)
    assert a.has_block()
    a.end_turn()
    assert not a.has_block()


def test_hp_tracking_persists_across_turns_until_reset():
    a = TurnAttributor("p1")
    for l in _lines(CASE_EFFECTIVENESS_TAG):
        a.feed(l)
    a.end_turn()
    for l in _lines("""
|turn|11
|move|p1a: Zapdos|Thunderbolt|p2a: Starmie
|-damage|p2a: Starmie|10/100
|upkeep
"""):
        a.feed(l)
    o = a.end_turn()
    assert o.target_hp_before == 40 and o.direct_damage == 30 and o.dealt_frac == 0.3
    assert not any("assumed" in n for n in o.notes)
