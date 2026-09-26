import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("run_m3_smoke", Path("scripts/run_m3_smoke.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def test_out_must_be_under_results_m3():
    with pytest.raises(SystemExit):
        m.parse_args(["--out", "results/m0d/x"])
    assert m.parse_args(["--out", "results/m3/smoke/x"]).flies == 2


def test_opponent_name_is_per_battle_and_fits_showdown():
    names = {m.opponent_name(f"f{f:02d}-b{b:03d}") for f in range(3) for b in range(3)}
    assert len(names) == 9 and all(len(n) <= 18 for n in names)
    assert m.opponent_name("f01-b002") == "fm-h-f01-b002"
    with pytest.raises(ValueError):
        m.opponent_name("x" * 20)


def test_battle_opponent_uses_the_kind_class_with_the_per_battle_account(monkeypatch):
    made = []

    class Fake:
        def __init__(self, **kw):
            made.append(kw)
    monkeypatch.setitem(m.KINDS, "heuristic", Fake)
    opp = m.make_battle_opponent("heuristic", "f00-b001", "srv", "team")
    assert isinstance(opp, Fake)
    kw = made[0]
    assert kw["account_configuration"].username == "fm-h-f00-b001"
    assert (kw["battle_format"], kw["server_configuration"], kw["team"], kw["max_concurrent_battles"]) == \
        ("gen1ou", "srv", "team", 1)
