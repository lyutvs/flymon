from flymon.battle.coach import Coach, CoachDecision
from flymon.battle.pool import by_species
from flymon.battle.router import candidates, route


def test_candidates_are_attacks_only(make_battle):
    b = make_battle(by_species["Chansey"], "Machamp")
    assert [m.id for m in candidates(b)] == ["thunderbolt", "psychic"]


def test_route_fly_when_attack_and_two_plus(make_battle):
    b = make_battle(by_species["Lapras"], "Venusaur")
    d = Coach().decide(b)
    who, cands = route(d, candidates(b))
    assert who == "fly" and len(cands) == 3


def test_route_coach_when_single_candidate_or_non_attack(make_battle):
    b = make_battle(by_species["Venusaur"], "Charizard")
    only = candidates(b)[:1]
    assert route(CoachDecision(None, "attack", only[0]), only)[0] == "coach"
    assert route(CoachDecision(None, "switch", None), candidates(b))[0] == "coach"
