"""scripts/run_m2_d6a.py (spec G.8) refuses before measuring anything: off the repository root, an existing output,
dirty tracked files on a full run, and a full run without E.1's record for the self-check."""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("run_m2_d6a", Path("scripts/run_m2_d6a.py").resolve())
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A fake repository root: the files main() checks for, nothing else."""
    (tmp_path / "flymon/brain").mkdir(parents=True); (tmp_path / "flymon/brain/d6a.py").write_text("")
    (tmp_path / "data").mkdir(); (tmp_path / "data/malecns.npz").write_bytes(b"")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run, "git_state", lambda: {"commit": "abc", "dirty": []})
    monkeypatch.setattr(run.Connectome, "load", lambda *a, **k: pytest.fail("measured after a refusal"))
    return tmp_path


def test_refuses_off_the_repository_root(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert run.main([]) == 2 and "repository root" in capsys.readouterr().err


@pytest.mark.parametrize("smoke, name", [(False, "g8.json"), (True, "smoke.json")])
def test_never_overwrites_an_output(root, capsys, smoke, name):
    (root / "results/m2/d6a").mkdir(parents=True); (root / "results/m2/d6a" / name).write_text("{}")
    assert run.main(["--smoke"] if smoke else []) == 2 and "exists" in capsys.readouterr().err


def test_full_run_refuses_dirty_tracked_files(root, monkeypatch, capsys):
    monkeypatch.setattr(run, "git_state", lambda: {"commit": "abc", "dirty": ["flymon/brain/d6a.py"]})
    assert run.main([]) == 2 and "dirty" in capsys.readouterr().err


def test_full_run_needs_e1s_record(root, capsys):
    assert run.main([]) == 2 and "self-check" in capsys.readouterr().err
