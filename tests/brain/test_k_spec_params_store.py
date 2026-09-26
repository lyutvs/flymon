"""Spec K.8: the configuration object, the engine builder and the guarded writes."""
import dataclasses
import json
from pathlib import Path

import pytest

from flymon.brain.config import Params
from flymon.brain.h3_runner import c1_params
from flymon.brain.j_params import StdParams
from flymon.brain.j_spec import SPEC as J_SPEC
from flymon.brain.k_params import k_make, with_kc
from flymon.brain.k_spec import SPEC, smoke
from flymon.brain.k_store import guard, write_json, write_summary_block


def test_the_declared_configuration():
    assert SPEC.grid == (-0.05, -0.1, -0.2, -0.4, -0.8)
    assert (SPEC.target_type, SPEC.reward_type, SPEC.min_weight) == ("MBON13", "MBON05", 5)
    assert (SPEC.tie_tol, SPEC.gate_ratio, SPEC.raster_g) == (0.01, 1.0, -0.8)
    assert SPEC.j == J_SPEC and SPEC.j.first_kc == 1.65 and SPEC.j.homeo_target == 0.060
    assert smoke(SPEC) != SPEC and smoke(SPEC).grid == (-0.8,)
    assert SPEC.even_pair_uses == ("H.4 C0-C3", "H.4a.8 ceiling freq", "H.4a.8 ceiling all", "J.13")


def test_with_kc_zero_is_the_engine_itself():
    base = c1_params(J_SPEC.h4.h3, 1.65, 1.0)
    assert base.kc_kc_scale == 0.0 and with_kc(base, 0.0) == base
    assert with_kc(base, -0.2).kc_kc_scale == -0.2
    assert dataclasses.replace(with_kc(base, -0.2), kc_kc_scale=0.0) == base


@pytest.mark.parametrize("g", [0.1, float("nan"), float("-inf")])
def test_with_kc_refuses_excitation_and_non_finite(g):
    with pytest.raises(ValueError):
        with_kc(Params(), g)


def test_with_kc_refuses_depression_engines():
    with pytest.raises(TypeError):
        with_kc(StdParams(receptor_scale=2.0), -0.1)
    with pytest.raises(ValueError):
        with_kc(Params(orn_std=True), -0.1)


def test_k_make_is_c3s_rule_with_g():
    make = k_make(J_SPEC.h4.h3, -0.4)
    assert make(1.6) == dataclasses.replace(c1_params(J_SPEC.h4.h3, 1.6, 1.0), kc_kc_scale=-0.4)


@pytest.mark.parametrize("path", ["results/m0d/k/run/runs/x.json", "results/summary/k_engine.json"])
def test_allowed_paths(path, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_json(path, {"a": 1}, [Params(), with_kc(Params(), -0.2)])
    assert json.loads(Path(path).read_text()) == {"a": 1}


@pytest.mark.parametrize("path", ["results/summary/m2_engine.json", "results/summary/std_scan.json",
                                  "results/m0d/j/std/runs/x.json", "results/summary/m0d.json", "results/m0c/x.json",
                                  "results/m0d/x.json", "results/k/x.json", "x.json"])
def test_refused_paths(path, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        guard(path, [with_kc(Params(), -0.2)])
    assert not Path(path).exists()


def test_summary_blocks_are_merged_atomically(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = "results/summary/k_engine.json"
    write_summary_block(p, "scan", {"outcome": "scan_go"}, [Params()])
    write_summary_block(p, "judge", {"outcome": "B"}, [Params()])
    assert json.loads(Path(p).read_text()) == {"scan": {"outcome": "scan_go"}, "judge": {"outcome": "B"}}
    assert not list(Path("results/summary").glob(".*.tmp"))
