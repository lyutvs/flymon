"""Spec K's CLIs: every refusal comes before any measurement or write (the runs themselves are controller steps)."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
needs_npz = pytest.mark.skipif(not (ROOT / "data/malecns.npz").exists(), reason="MaleCNS connectome not built")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


SCAN, JUDGE = _load("run_k_scan"), _load("run_k_judge")
CLEAN = lambda files: dict(commit="x", dirty_hashed=[], dirty_other=[])


@pytest.mark.parametrize("cli", [SCAN, JUDGE])
def test_outside_the_repository_root_is_refused(cli, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert cli.main([]) == 2 and "repository root" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("cli", [SCAN, JUDGE])
def test_dirty_hashed_files_are_refused(cli, monkeypatch, capsys):
    monkeypatch.setattr(cli, "git_state", lambda files: dict(commit="x", dirty_hashed=["flymon/brain/k_rules.py"],
                                                             dirty_other=[]))
    assert cli.main([], require_root=False) == 2 and "dirty" in capsys.readouterr().err


def test_a_foreign_connectome_is_refused(tmp_path, monkeypatch, capsys):
    npz = tmp_path / "other.npz"
    npz.write_bytes(b"not the connectome")
    monkeypatch.setattr(SCAN, "git_state", CLEAN)
    assert SCAN.main(["--npz", str(npz)], require_root=False) == 2 and "declared connectome" in capsys.readouterr().err


def _scan(key, manifest, **kw):
    doc = {"measure_key": key, "code": {"key": manifest}, "smoke": False, "git": {"dirty_hashed": []},
           "spec": json.loads(JUDGE.canonical(JUDGE.SPEC)),
           "stage1": {"outcome": "scan_go", "order": [0], "settings": [
               {"g": -0.2, "reconverge": {"adopted": {"params": {"kc_thresh": 1.65, "kc_thresh_file": "missing.npz",
                                                                  "kc_thresh_sha256": "0" * 64}},
                                          "guard": {}}}]}}
    doc.update(kw)
    return doc


@needs_npz
def test_the_judgement_refuses_other_code_a_stopped_scan_and_a_missing_threshold_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(JUDGE, "git_state", CLEAN)
    key = JUDGE.code_key("data/malecns.npz", files=JUDGE.MEASURE_FILES)["key"]
    manifest = JUDGE.code_key("data/malecns.npz", files=JUDGE.HASHED_FILES)["key"]
    f = tmp_path / "scan.json"
    f.write_text(json.dumps(_scan("0" * 64, manifest)))
    assert JUDGE.main(["--scan", str(f)], require_root=False) == 2 and "K key" in capsys.readouterr().err
    f.write_text(json.dumps(_scan(key, manifest, stage1={"outcome": "stop_no_target_gain"})))
    assert JUDGE.main(["--scan", str(f)], require_root=False) == 2 and "K.8.4" in capsys.readouterr().err
    f.write_text(json.dumps(_scan(key, "0" * 64)))
    assert JUDGE.main(["--scan", str(f)], require_root=False) == 2 and "procedure code" in capsys.readouterr().err
    f.write_text(json.dumps(_scan(key, manifest, smoke=True)))
    assert JUDGE.main(["--scan", str(f)], require_root=False) == 2 and "smoke" in capsys.readouterr().err
    f.write_text(json.dumps(_scan(key, manifest)))
    assert JUDGE.main(["--scan", str(f)], require_root=False) == 2 and "threshold file" in capsys.readouterr().err


@needs_npz
def test_the_judgement_reads_the_scan_block_of_the_summary(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(JUDGE, "git_state", CLEAN)
    manifest = JUDGE.code_key("data/malecns.npz", files=JUDGE.HASHED_FILES)["key"]
    f = tmp_path / "k_engine.json"
    f.write_text(json.dumps({"scan": _scan("0" * 64, manifest)}))
    assert JUDGE.main(["--scan", str(f)], require_root=False) == 2 and "K key" in capsys.readouterr().err


def test_the_smoke_spec_differs_from_the_declared_one():
    assert SCAN.smoke(SCAN.SPEC) != SCAN.SPEC
