"""AC.7 2b's smoke gates as pure functions (the script adds the pytest runs and the naive L_V pool)."""
import ast
import json
from pathlib import Path

import numpy as np

from flymon.ac import smoke
from flymon.ac.spec import LABEL, SPEC

Z = {"A": (0.0, 1.0), "P": (0.0, 1.0)}
ROOT = Path(__file__).resolve().parents[2]


def _test_names(path: Path) -> set:
    tree = ast.parse(path.read_text())
    return {n.name for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test_")}


def test_gate_test_ids_name_real_tests():
    # every gate id must point at a real test file (and, with ::name, a real test function in it), so a renamed or
    # deleted test cannot let a gate pass by collecting nothing
    assert len(smoke.SAFETY_TESTS) == 4 and len(smoke.EQUALITY_TESTS) == 2 and len(smoke.TB_TESTS) >= 1
    for tid in smoke.SAFETY_TESTS + smoke.EQUALITY_TESTS + smoke.TB_TESTS:
        file, _, name = tid.partition("::")
        p = ROOT / file
        assert p.is_file(), tid
        names = _test_names(p)
        if name:
            assert name in names, tid
        else:
            assert names, tid
    assert len(set(smoke.SAFETY_TESTS)) == 4 and len(set(smoke.EQUALITY_TESTS)) == 2


def test_pool_response_ratios():
    c = [np.array([[3, 1, 1], [3, 1, 1]]),           # tie (equal V), A > 0, P > 0
         np.array([[0, 0, 1], [2, 0, 1], [0, 5, 0]])]   # no tie; A = 0 twice, P = 0 twice
    r = smoke.pool_response(c, 1, 1, Z)
    assert r == {"n_situations": 2, "n_candidates": 5, "tie_ratio": 0.5, "a0_ratio": 0.4, "p0_ratio": 0.4}


def test_response_gates_are_inclusive_at_half():
    g = smoke.response_gates({"tie_ratio": 0.5, "a0_ratio": 0.5, "p0_ratio": 0.51}, SPEC)
    assert g == {"pool_tie": True, "pool_a0": True, "pool_p0": False}


def test_schema_and_kc_ratio(tmp_path):
    d = tmp_path / "BRAIN"
    (d / "logs/eval").mkdir(parents=True)
    ok = {"schema": 1, "kind": "decision", "fly": 0, "battle_id": "SL-f00-b000", "battle_tag": "t", "turn": 1,
          "decider": "fly", "coach_kind": "attack", "candidates": ["surf", "psychic"], "chosen": "surf",
          "v": [0.1, 0.2], "a": [1, 2], "p": [0, 1], "kc_active": [10, 30], "tau": 1.0, "seed": 3,
          "kc_ratio_gt2": True}
    coach = dict(ok, decider="coach", kc_ratio_gt2=False)
    for k in ("v", "a", "p", "kc_active", "tau", "seed"):
        coach.pop(k)
    (d / "logs/fly00.jsonl").write_text(json.dumps(ok) + "\n" + json.dumps(dict(ok, kc_ratio_gt2=False)) + "\n" +
                                        json.dumps(coach) + "\n")
    (d / "logs/eval/fly00.jsonl").write_text(json.dumps({k: v for k, v in ok.items() if k != "tau"}) + "\n")
    paths = smoke.log_paths(d)
    assert [p.name for p in paths] == ["fly00.jsonl", "fly00.jsonl"]
    errs = smoke.schema_errors(paths)
    assert len(errs) == 1 and "tau" in errs[0]
    assert smoke.kc_ratio_frac(paths) == {"n_fly_turns": 3, "n_gt2": 2, "frac": 2 / 3}
    assert smoke.kc_ratio_frac([]) == {"n_fly_turns": 0, "n_gt2": 0, "frac": None}


def test_resumed_and_status(tmp_path):
    (tmp_path / "wall_clock.json").write_text(json.dumps({"sessions": [{"complete": False, "status": "ok"},
                                                                       {"complete": True, "status": "ok"}]}))
    assert smoke.resumed(tmp_path)
    (tmp_path / "wall_clock.json").write_text(json.dumps({"sessions": [{"complete": True, "status": "ok"}]}))
    assert not smoke.resumed(tmp_path)
    assert smoke.status({"a": True, "b": True}) == "OK" and smoke.status({"a": True, "b": False}) == "STOP_SMOKE"


def test_script_parses_and_carries_label():
    src = (ROOT / "scripts/check_ac_smoke.py").read_text()
    ast.parse(src)
    assert "label=LABEL" in src and isinstance(LABEL, str) and LABEL
