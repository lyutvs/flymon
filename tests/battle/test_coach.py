from flymon.battle.coach import Coach
from flymon.battle.pool import by_species


def test_coach_attacks_with_best_move_in_good_matchup(make_battle):
    b = make_battle(by_species["Blastoise"], "Charizard")
    d = Coach().decide(b)
    assert d.kind == "attack" and d.move.id == "surf"      # water vs fire


def test_weak_coach_never_switches_or_supports(make_battle):
    b = make_battle(by_species["Chansey"], "Machamp")       # bad matchup, switches available
    d = Coach(weak=True).decide(b)
    assert d.kind == "attack"


def test_weak_coach_attacks_where_strong_coach_switches(make_battle):
    b = make_battle(by_species["Charizard"], "Rhydon")   # fire/flying into ground/rock
    assert Coach().decide(b).kind == "switch"
    d = Coach(weak=True).decide(b)
    assert d.kind == "attack" and d.move.id == "earthquake"


def test_weak_coach_still_switches_when_forced(make_battle):
    b = make_battle(by_species["Chansey"], "Machamp", force_switch=True)
    assert b.force_switch
    assert Coach(weak=True).decide(b).kind == "switch"
