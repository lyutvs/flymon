"""Y's 0p numbers (Y.2 · Y.3.2 · Y.3.3 · Y.7 0p): thresholds, keys, the main-set facts, seed roots that collide with
nothing declared, and no Y file but y_spec holding a number (literal guard)."""
import ast
import dataclasses
import importlib
import json
from pathlib import Path

import pytest

from flymon.brain import d6a
from flymon.brain.v_spec import SPEC as V
from flymon.brain.w_spec import SPEC as W
from flymon.brain.y_spec import SPEC, YSpec
from tests.brain.test_p_spec import MODULES, _collect

ROOT = Path(__file__).resolve().parents[2]


def test_y_spec_is_enumerated_for_the_collectors():
    assert MODULES.get("flymon/brain/y_spec.py") == "flymon.brain.y_spec"


def test_thresholds():
    assert (SPEC.naive_max, SPEC.c_a, SPEC.c_p) == (0.5, 20.0, 43.0)
    assert (SPEC.lenient_d, SPEC.lenient_a, SPEC.lenient_p) == (1.0, 16.0, 34.4)
    assert round(0.8 * SPEC.c_a, 9) == SPEC.lenient_a and round(0.8 * SPEC.c_p, 9) == SPEC.lenient_p
    assert SPEC.round_digits == 9 and V.testable_min == 2.0
    assert SPEC.f2_0 == (8.875, 51.640625) and SPEC.f2_25 == (30.7421875, 94.421875)
    assert (SPEC.user_a, SPEC.user_onesided_d) == (31.0, -1.0)
    assert SPEC.k_ranges == ((4, 8), (6, 10)) and min(lo for lo, _ in SPEC.k_ranges) == 4
    assert SPEC.budget_h == 24.0


def test_f2_values_are_xs_precheck_diag():
    p = ROOT / "results/x/precheck_diag.json"
    if not p.exists():
        pytest.skip("X detail not present in this checkout")
    f = json.loads(p.read_text())["filters"]
    assert SPEC.f2_0 == (f["F2(0)"]["pass_counts"]["c_A"], f["F2(0)"]["pass_counts"]["c_P"])
    assert SPEC.f2_25 == (f["F2(25)"]["pass_counts"]["c_A"], f["F2(25)"]["pass_counts"]["c_P"])


def test_keys_and_main_set_facts():
    assert SPEC.u_measure_key == W.u_measure_key_u == \
        "8a4e09302f7f2f7e7d69f3540d8fc392da582cb1246bfebeb150348776e26523"
    assert SPEC.w_measure_key == "761274e0ed09f9ec7043079e477588b4a2eef76c8076614782542573b2138df1"
    assert SPEC.main_digest_keys == "65dbf001a61ea7f484cf4e61a212a3973fde8709c5225e15e2221b0777a5712a"
    assert (SPEC.main_n_b, SPEC.main_n_a, SPEC.main_last_turn) == (167, 82, 1967)
    assert (SPEC.summary, SPEC.raw_dir, SPEC.cache_dir, SPEC.oracle_detail) == (
        "results/summary/y_learning.json", "results/y", "results/y/cache", "results/y/oracle.json")


def test_seed_roots_are_new():
    out, seen = set(), set()
    _collect(SPEC, out, seen)
    assert out == {60_000_000, 61_000_000, 62_000_000, 64_000_000, 77_000_000, 77_100_000, 77_110_000,
                   77_150_000, 77_200_000, 77_300_000}
    assert not any(dataclasses.is_dataclass(getattr(SPEC, f.name)) for f in dataclasses.fields(YSpec))
    declared, seen = set(d6a.SEEDS), set()
    for path, mod in MODULES.items():
        if path == "flymon/brain/y_spec.py":
            continue
        m = importlib.import_module(mod)
        _collect(m.SPEC, declared, seen)
        if hasattr(m, "smoke"):
            _collect(m.smoke(m.SPEC), declared, seen)
    assert not any(60_000_000 <= s < 77_400_000 for s in declared)
    o = W.oracle_seeds()
    assert not any(60_000_000 <= s < 77_400_000 for s in o["act"] + o["select"] + o["report"])


def _numbers(path: Path) -> set:
    return {n.value for n in ast.walk(ast.parse(path.read_text()))
            if isinstance(n, ast.Constant) and type(n.value) in (int, float)}


def test_no_y_file_but_y_spec_holds_a_number():
    files = [p for p in sorted((ROOT / "flymon/brain").glob("y_*.py")) if p.name != "y_spec.py"]
    if (ROOT / "scripts/run_y.py").exists():
        files.append(ROOT / "scripts/run_y.py")
    for p in files:
        bad = {v for v in _numbers(p) if (type(v) is int and abs(v) > 16) or (type(v) is float and v not in (0.0, 1.0))}
        assert not bad, (p.name, bad)
