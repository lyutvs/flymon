"""Spec J.11's CLIs: every refusal comes before any measurement or write (the runs themselves are controller steps)."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


SCAN, JUDGE = _load("run_j_std_scan"), _load("run_j_std_judge")


@pytest.mark.parametrize("cli", [SCAN, JUDGE])
def test_outside_the_repository_root_is_refused(cli, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert cli.main([]) == 2 and "repository root" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("cli", [SCAN, JUDGE])
def test_dirty_hashed_files_are_refused(cli, monkeypatch, capsys):
    monkeypatch.setattr(cli, "git_state", lambda files: dict(commit="x", dirty_hashed=["flymon/brain/j_rules.py"],
                                                             dirty_other=[]))
    assert cli.main([], require_root=False) == 2 and "dirty" in capsys.readouterr().err


def test_a_foreign_connectome_is_refused(tmp_path, monkeypatch, capsys):
    npz = tmp_path / "other.npz"
    npz.write_bytes(b"not the connectome")
    monkeypatch.setattr(SCAN, "git_state", lambda files: dict(commit="x", dirty_hashed=[], dirty_other=[]))
    assert SCAN.main(["--npz", str(npz)], require_root=False) == 2 and "declared connectome" in capsys.readouterr().err


@pytest.mark.skipif(not (ROOT / "data/malecns.npz").exists(), reason="MaleCNS connectome not built")
def test_the_judgement_refuses_a_scan_from_other_code_or_an_incomplete_scan(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(JUDGE, "git_state", lambda files: dict(commit="x", dirty_hashed=[], dirty_other=[]))
    scan = tmp_path / "scan.json"
    scan.write_text(json.dumps({"measure_key": "0" * 64, "scan": {"outcome": "SCAN_COMPLETE"}}))
    assert JUDGE.main(["--scan", str(scan)], require_root=False) == 2 and "J key" in capsys.readouterr().err
    key = JUDGE.code_key("data/malecns.npz", files=JUDGE.MEASURE_FILES)["key"]
    scan.write_text(json.dumps({"measure_key": key, "scan": {"outcome": "NO_FEASIBLE_SETTING"}}))
    assert JUDGE.main(["--scan", str(scan)], require_root=False) == 2 and "SCAN_COMPLETE" in capsys.readouterr().err


@pytest.mark.skipif(not (ROOT / "data/malecns.npz").exists(), reason="MaleCNS connectome not built")
def test_the_judgement_refuses_a_smoke_or_dirty_scan_outside_smoke(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(JUDGE, "git_state", lambda files: dict(commit="x", dirty_hashed=[], dirty_other=[]))
    key = JUDGE.code_key("data/malecns.npz", files=JUDGE.MEASURE_FILES)["key"]
    scan = tmp_path / "scan.json"
    scan.write_text(json.dumps({"measure_key": key, "scan": {"outcome": "SCAN_COMPLETE"}, "smoke": True}))
    assert JUDGE.main(["--scan", str(scan)], require_root=False) == 2 and "a smoke run" in capsys.readouterr().err
    scan.write_text(json.dumps({"measure_key": key, "scan": {"outcome": "SCAN_COMPLETE"}, "smoke": False,
                                "git": {"dirty_hashed": ["x"]}}))
    assert JUDGE.main(["--scan", str(scan)], require_root=False) == 2 and "dirty hashed files" in capsys.readouterr().err


def test_pairs_below_one_is_refused(capsys):
    assert JUDGE.main(["--pairs", "0"], require_root=False) == 2 and "--pairs" in capsys.readouterr().err


def test_the_smoke_specs_differ_from_the_declared_one():
    """A smoke run can never write a summary: its spec is not SPEC."""
    assert SCAN.smoke_spec(SCAN.SPEC) != SCAN.SPEC and JUDGE.smoke_spec(JUDGE.SPEC) != JUDGE.SPEC
