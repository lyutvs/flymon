from flymon.battle.pool import by_species
from flymon.battle.schedule import load, make_schedule, save


def test_schedule_is_deterministic_and_well_formed(tmp_path):
    a = make_schedule(n_flies=3, n_battles=4, seed=7)
    b = make_schedule(n_flies=3, n_battles=4, seed=7)
    assert [x.__dict__ for x in a] == [x.__dict__ for x in b]
    assert len(a) == 12
    for sb in a:
        assert len(sb.my_team) == 6 and len(set(sb.my_team)) == 6 and all(s in by_species for s in sb.my_team)
        assert len(sb.opp_team) == 6 and len(set(sb.opp_team)) == 6
        assert sb.opponent == "heuristic"
    assert make_schedule(3, 4, seed=8)[0].my_team != a[0].my_team
    p = tmp_path / "s.json"; save(p, a)
    assert [x.__dict__ for x in load(p)] == [x.__dict__ for x in a]
