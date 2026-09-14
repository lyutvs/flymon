import pytest

from flymon.battle.moves import attack_allowed, fly_choosable, gen1, is_attack, support_allowed


@pytest.mark.parametrize("mid,ok,why", [
    ("surf", True, ""), ("earthquake", True, ""), ("strength", True, ""),
    ("thunderbolt", True, ""),            # par 10% — mild secondary allowed
    ("psychic", True, ""),                # 33% special drop — stat-drop secondary allowed
    ("bodyslam", False, "secondary"),     # par 30%
    ("blizzard", False, "secondary"),     # frz
    ("explosion", False, "selfdestruct"),
    ("wrap", False, "volatileStatus"),
    ("thrash", False, "volatileStatus"),
    ("hyperbeam", False, "recharge"),
    ("fly", False, "charge"),
    ("slash", False, "critRatio"),
    ("seismictoss", False, "damage"),
    ("superfang", False, "damage"),       # damageCallback: halves current HP
    ("counter", False, "damage"),         # damageCallback: returns damage taken
    ("doubleedge", False, "recoil"),
    ("megakick", False, "accuracy"),
    ("stomp", False, "secondary"),        # flinch
    ("thunderwave", False, "category"),
])
def test_attack_allowed(mid, ok, why):
    allowed, reason = attack_allowed(gen1(mid))
    assert allowed is ok, (mid, reason)
    if not ok:
        assert why in reason


@pytest.mark.parametrize("mid,ok", [("thunderwave", True), ("reflect", True), ("softboiled", True), ("amnesia", True),
                                    ("swordsdance", True), ("agility", True), ("substitute", False), ("toxic", False),
                                    ("rest", False), ("confuseray", False), ("sleeppowder", False), ("surf", False)])
def test_support_allowed(mid, ok):
    assert support_allowed(gen1(mid))[0] is ok


def test_is_attack():
    assert is_attack(gen1("surf")) and not is_attack(gen1("thunderwave")) and not is_attack(gen1("seismictoss"))
    assert not is_attack(gen1("superfang")) and not is_attack(gen1("counter"))


def test_fly_choosable_is_the_conjunction():
    assert fly_choosable(gen1("surf"))
    assert not fly_choosable(gen1("superfang"))
    assert not fly_choosable(gen1("bodyslam"))   # an attack, but not allowed
