from flymon.battle.coach import Coach
from flymon.battle.pool import PoolMon, by_species
from flymon.battle.router import candidates


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


def test_weak_coach_uses_the_same_filter_as_the_router(make_battle):
    """A disallowed attack (Body Slam: 30% par) is never the weak coach's pick, as for the router."""
    mon = PoolMon("Snorlax", ("Body Slam", "Strength"), (), 30)
    b = make_battle(mon, "Gengar", bench=[])
    assert {m.id for m in b.available_moves} == {"bodyslam", "strength"}
    assert [m.id for m in candidates(b)] == ["strength"]
    d = Coach(weak=True).decide(b)
    assert d.kind == "attack" and d.move.id == "strength"
