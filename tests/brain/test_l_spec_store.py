"""Spec L / L.11: the one configuration object and the guarded writes."""
import json
from pathlib import Path

import pytest

from flymon.brain.l_spec import SPEC, smoke
from flymon.brain.l_store import guard, write_json, write_summary_block


def test_the_declared_configuration():
    s = SPEC
    assert (s.cov_min_stage1, s.gate_min_passes, s.gate_min_precision) == (0.35, 10, 0.6)
    assert (s.n_pass, s.max_screened, s.n_turns, s.a_turns) == (21, 84, 64, 8)
    assert (s.rng_seed, s.lift_seed, s.n_lift) == (20260927, 20260928, 10)
    assert s.guard_types == ("MBON13", "MBON05") and s.guard_min == 5
    assert s.family_order == ("G", "S", "f2", "f3")
    assert dict(s.directions) == {"S": ">", "f2": ">", "f3": "<"}
    assert dict(s.readout) == {"A": "MBON13", "P": "MBON05"}
    hash(s)                                                  # frozen and hashable
    assert s.ceiling_sha256.startswith("81bfab20") and s.k_act_sha256.startswith("93d86ee0")
    assert s.j.stage2_n_b == 21 and s.j.h4.select_seeds == tuple(range(600, 608))


def test_smoke_shrinks_the_run_not_the_rule():
    m = smoke(SPEC)
    assert m.n_pass < SPEC.n_pass and m.n_turns < SPEC.n_turns
    assert m.cov_min_stage1 == SPEC.cov_min_stage1 and m.family_order == SPEC.family_order
    assert m.j == SPEC.j                    # declared act / select / report seeds kept: loaders and self-checks need them


def test_the_guard_refuses_a_modified_engine_on_a_reference_path(tmp_path, monkeypatch, capsys):
    """The engine branch (pool_bench's refusals) fires before the path rule and names the engine."""
    from flymon.brain.config import Params
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        guard("results/m0c/x.json", [Params(apl_input_scale=0.6)])
    assert "M0d modes" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        guard("results/m0/x.json", [Params(kc_kc_scale=0.0)])
    assert "kc_kc_scale=0.0" in capsys.readouterr().err
    guard("results/m0d/l/x.json", [Params(apl_input_scale=0.6, kc_kc_scale=0.0)])   # the allowed path passes


@pytest.mark.parametrize("path", ["results/m0d/l/runs/x.json", "results/summary/l_screen.json"])
def test_allowed_paths(path, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    guard(path, [])


@pytest.mark.parametrize("path", ["results/summary/k_engine.json", "results/m0d/k/x.json", "results/m0d/lx/a.json",
                                  "elsewhere.json"])
def test_refused_paths(path, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        guard(path, [])


def test_summary_blocks_are_merged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_summary_block("results/summary/l_screen.json", "stage0", {"a": 1}, [])
    write_summary_block("results/summary/l_screen.json", "stage1", {"b": 2}, [])
    doc = json.loads(Path("results/summary/l_screen.json").read_text())
    assert doc == {"stage0": {"a": 1}, "stage1": {"b": 2}}
    write_json("results/m0d/l/runs/r.json", {"x": [1, 2]}, [])
    assert json.loads(Path("results/m0d/l/runs/r.json").read_text()) == {"x": [1, 2]}
