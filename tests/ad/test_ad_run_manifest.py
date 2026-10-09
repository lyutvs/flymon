"""The run manifest (AD.4): T, the points, the learning-prefix digest used, the new-set flag, the budget and input
manifest sha256 and the code commit, verified at every run and resume (STOP_MANIFEST); its sha256 is part of the
checkpoint key."""
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

from flymon.ad import budget, inputs, run_manifest as rm, store
from flymon.ad.spec import SPEC, smoke

ROOT = Path(__file__).resolve().parents[2]


def repo(tmp_path):
    run = lambda *a: subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)
    run("init", "-q"); run("config", "user.email", "t@t"); run("config", "user.name", "t")
    (tmp_path / "flymon").mkdir()
    (tmp_path / "flymon/x.py").write_text("x = 1\n")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/run_ad.py").write_text("# runner\n")
    run("add", "."); run("commit", "-qm", "code")
    return run


def files(tmp_path, spec, T=120):
    _, man_p = inputs.input_paths(spec)
    store.write_json(man_p, {"status": "OK", "fixed": {}})
    bp = Path(budget.BUDGET) if spec.mode == "run" else None
    if bp is not None:
        store.write_json(bp, {"status": "OK", "T": T, "new_set": True})
    return bp, man_p


def test_build_and_check(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    repo(tmp_path)
    bp, man_p = files(tmp_path, SPEC)
    m = rm.build(SPEC, T=120, new_set=True, budget_path=bp, inputs_manifest_path=man_p, commit=store.head_commit())
    assert m["points"] == [0, 20, 40, 60, 80, 100, 120] and m["new_points"] == [0, 40, 120]
    assert m["learn_prefix_digest"] == "4a19075eacf8a74904b5e41d1e0221d3cc7cf3d49006c883ce6aeff9f4374867"
    assert rm.check(m, SPEC) == []
    m80 = rm.build(SPEC, T=80, new_set=False, budget_path=bp, inputs_manifest_path=man_p, commit=store.head_commit())
    assert m80["learn_prefix_digest"].startswith("6036ac80bae4") and m80["new_points"] == []


def test_check_detects_changed_budget_code_and_points(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = repo(tmp_path)
    bp, man_p = files(tmp_path, SPEC)
    m = rm.build(SPEC, T=120, new_set=True, budget_path=bp, inputs_manifest_path=man_p, commit=store.head_commit())
    assert rm.check(dict(m, points=[0, 40, 120]), SPEC) == ["points"]
    assert rm.check(dict(m, T=100), SPEC) == ["T", "points", "new_points", "learn_prefix_digest", "budget_decision"]
    store.write_json(bp, {"status": "OK", "T": 120, "new_set": False})
    assert rm.check(m, SPEC) == ["budget_sha256", "budget_decision"]
    store.write_json(bp, {"status": "OK", "T": 120, "new_set": True})
    (tmp_path / "flymon/x.py").write_text("x = 2\n")
    assert rm.check(m, SPEC) == ["code_changed: flymon/x.py"]
    assert rm.check(m, SPEC, check_code=False) == []
    run("commit", "-qam", "edit")
    assert rm.check(m, SPEC) == ["code_changed: flymon/x.py"]


def test_checkpoint_key_moves_with_the_manifest():
    k = dict(cfg_hash="c", layout=[["FLYL", 0, 24]], input_digests={"learn": "a"}, mode="run", lever_sha="l")
    assert rm.checkpoint_key(**k, run_manifest_sha256="1") != rm.checkpoint_key(**k, run_manifest_sha256="2")


def load_script():
    s = importlib.util.spec_from_file_location("warm", ROOT / "scripts/write_ad_run_manifest.py")
    mod = importlib.util.module_from_spec(s); s.loader.exec_module(mod); return mod


def test_script_refuses_without_ok_budget_or_with_dirty_code(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    repo(tmp_path)
    w = load_script()
    _, man_p = inputs.input_paths(SPEC)
    with pytest.raises(SystemExit, match="ad_inputs_manifest"):
        w.main([])
    store.write_json(man_p, {"status": "OK", "fixed": {}})
    with pytest.raises(SystemExit, match="ad_budget"):
        w.main([])
    bp = Path(budget.BUDGET)
    store.write_json(bp, {"status": "STOP_BUDGET", "T": None})
    with pytest.raises(SystemExit, match="STOP_BUDGET"):
        w.main([])
    store.write_json(bp, {"status": "OK", "T": 80, "new_set": True})
    (tmp_path / "flymon/x.py").write_text("x = 3\n")
    with pytest.raises(SystemExit, match="uncommitted"):
        w.main([])
    subprocess.run(["git", "commit", "-qam", "c"], cwd=tmp_path, check=True)
    assert w.main([]) == 0
    m = json.loads((tmp_path / rm.RUN_MANIFEST).read_text())
    assert m["T"] == 80 and m["points"] == [0, 20, 40, 60, 80] and m["new_points"] == [0, 40, 80]


def test_smoke_manifest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    repo(tmp_path)
    files(tmp_path, smoke())
    assert load_script().main(["--smoke"]) == 0
    m = json.loads((tmp_path / "results/m4-smoke/ad/run_manifest.json").read_text())
    assert (m["T"], m["points"], m["budget_path"], m["mode"]) == (2, [0, 1, 2], None, "smoke")
