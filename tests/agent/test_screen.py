from types import SimpleNamespace

import pytest

from flymon.agent.screen import ScreenTable, screened_route

KEY = ("Blastoise", "Charizard", "high", "mid", ("earthquake", "surf"))


def test_missing_key_uses_default():
    assert ScreenTable({}, default="coach").route(KEY) == "coach"
    assert ScreenTable({}, default="fly").route(KEY) == "fly"


def test_entries_override_default_and_values_are_checked():
    t = ScreenTable({KEY: "fly"}, default="coach")
    assert t.route(KEY) == "fly"
    with pytest.raises(ValueError, match="fly.*coach"):
        ScreenTable({KEY: "maybe"}, default="coach")
    with pytest.raises(ValueError):
        ScreenTable({}, default="nobody")


def test_json_round_trip():
    t = ScreenTable({KEY: "fly"}, default="coach")
    assert ScreenTable.from_json(t.to_json()).route(KEY) == "fly"
    assert ScreenTable.from_json(t.to_json()).default == "coach"


def test_screen_only_narrows_the_base_router():
    atk = SimpleNamespace(kind="attack", order="o", move=None)
    sup = SimpleNamespace(kind="support", order="o", move=None)
    cands = ["m1", "m2"]
    assert screened_route(atk, cands, KEY, ScreenTable.allow_all()) == ("fly", cands)
    assert screened_route(atk, cands, KEY, ScreenTable({}, default="coach")) == ("coach", [])
    assert screened_route(sup, cands, KEY, ScreenTable.allow_all()) == ("coach", [])      # the base router says coach
    assert screened_route(atk, ["m1"], KEY, ScreenTable.allow_all()) == ("coach", [])     # one candidate
