from flymon.battle.moves import attack_allowed, gen1, support_allowed
from flymon.battle.pool import POOL, export_text, team_export


def test_pool_size_and_shape():
    assert len(POOL) == 16
    for mon in POOL:
        assert 2 <= len(mon.attacks) <= 3 and 0 <= len(mon.support) <= 2
        assert 3 <= len(mon.attacks) + len(mon.support) <= 4
        assert mon.base_speed <= 100


def test_every_attack_allowed_and_types_distinct():
    for mon in POOL:
        types = [gen1(m).type for m in mon.attacks]
        assert len(set(types)) == len(types), (mon.species, types)
        for m in mon.attacks:
            ok, why = attack_allowed(gen1(m))
            assert ok, (mon.species, m, why)
        for s in mon.support:
            assert support_allowed(gen1(s))[0], (mon.species, s)


def test_attack_type_budget():
    types = {gen1(m).type for mon in POOL for m in mon.attacks}
    assert len(types) <= 12


def test_export_text_format():
    mon = POOL[0]
    text = export_text(mon)
    assert text.splitlines()[0] == mon.species
    assert all(line.startswith("- ") for line in text.splitlines()[1:])
    team = team_export(POOL[:6])
    assert team.count("\n\n") == 5
