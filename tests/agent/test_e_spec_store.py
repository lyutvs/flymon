import importlib, inspect, json, os
import pytest
from flymon.agent import e_spec, e_store
from flymon.agent.e_spec import SPEC

def _seed_ints(obj):
    out = set()
    if isinstance(obj, bool):
        return out
    if isinstance(obj, int):
        out.add(obj)
    elif isinstance(obj, (tuple, list, range, set, frozenset)):
        for v in obj:
            out |= _seed_ints(v)
    return out

def _module_seeds(modname):
    m = importlib.import_module(modname)
    out = set()
    for name, cls in inspect.getmembers(m, inspect.isclass):
        if cls.__module__ != m.__name__ or not hasattr(cls, "__dataclass_fields__"):
            continue
        for f, fld in cls.__dataclass_fields__.items():
            if "seed" in f:
                default = fld.default if fld.default is not inspect._empty else None
                if default is None and fld.default_factory is not inspect._empty:  # pragma: no cover
                    default = fld.default_factory()
                out |= _seed_ints(default)
    for name, v in vars(m).items():
        if "SEED" in name.upper():
            out |= _seed_ints(v)
    return out

def test_track_seeds_disjoint_from_every_spec_module():
    import pkgutil, flymon.brain, flymon.rescope
    mods = [f"flymon.brain.{n}" for _, n, _ in pkgutil.iter_modules(flymon.brain.__path__) if n.endswith("_spec")]
    mods += ["flymon.rescope.spec", "flymon.rescope.blocks", "flymon.brain.d6a"]
    other = set().union(*(_module_seeds(m) for m in mods))
    other |= set(range(500, 616))
    mine = e_spec.track_seeds(SPEC)
    assert mine and not (mine & other)

def test_spec_numbers():
    assert SPEC.s_grid == (0.175, 0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0)
    assert SPEC.configs == ("k3-full", "k3-norm", "k2-full", "k2-norm")
    assert SPEC.drive_seeds == tuple(range(24_001_000, 24_001_008))
    assert SPEC.judge_act_seeds == tuple(range(24_100_000, 24_100_008))
    assert SPEC.even_report_seeds == tuple(range(608, 616))
    assert SPEC.bar_b == 11 and SPEC.f_a_min == 2 and SPEC.margin == 2 and SPEC.tie_pairs == 2

def test_guard_refuses_outside(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        e_store.write_json("results/m0d/x.json", {"a": 1}, [])
    p = e_store.write_json("results/encoder/x.json", {"a": 1}, [])
    assert json.loads(p.read_text()) == {"a": 1}
    e_store.write_summary_block(e_store.SUMMARY, "set", {"n": 1}, [])
    e_store.write_summary_block(e_store.SUMMARY, "drive", {"m": 2}, [])
    assert e_store.read_summary() == {"set": {"n": 1}, "drive": {"m": 2}}

def test_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    c = e_store.ECache("results/encoder/cache", {"key": "abc"})
    calls = []
    f = lambda: calls.append(1) or {"v": 3}
    assert c.get_or_compute("oracle", {"x": 1}, f, []) == {"v": 3}
    assert c.get_or_compute("oracle", {"x": 1}, f, []) == {"v": 3}
    assert calls == [1] and c.hits == 1 and c.misses == 1
    assert c.get("oracle", {"x": 2}) is None
