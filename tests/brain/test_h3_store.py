"""Resume keys, guarded writes and the measurement cache (spec H.3a.11, H.2 write guards)."""
import json
import os
from pathlib import Path

import pytest

from flymon.brain import h3_store as S
from flymon.brain.config import Params

GRADED = Params(apl_mode="graded", apl_input_scale=0.5)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A fake repository root with the hashed files, and the working directory inside it."""
    files = ("a.py", "b.py", "uv.lock")
    for f in files:
        (tmp_path / f).write_text(f"# {f}\n")
    (tmp_path / "conn.npz").write_bytes(b"npz")
    monkeypatch.chdir(tmp_path)
    return tmp_path, files


def test_code_key_follows_hashed_content_and_nothing_else(repo):
    root, files = repo
    k0 = S.code_key(root / "conn.npz", root, files)["key"]
    (root / "docs.md").write_text("a note")
    assert S.code_key(root / "conn.npz", root, files)["key"] == k0
    (root / "b.py").write_text("# changed\n")
    assert S.code_key(root / "conn.npz", root, files)["key"] != k0


def test_code_key_follows_the_npz(repo):
    root, files = repo
    k0 = S.code_key(root / "conn.npz", root, files)["key"]
    (root / "conn.npz").write_bytes(b"other")
    assert S.code_key(root / "conn.npz", root, files)["key"] != k0


def test_git_state_blocks_on_hashed_files_only(tmp_path):
    import subprocess
    for args in (["init", "-q"], ["config", "user.email", "t@t"], ["config", "user.name", "t"]):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True)
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "doc.md").write_text("doc\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "init"], check=True)
    (tmp_path / "doc.md").write_text("doc changed\n")
    st = S.git_state(tmp_path, ("a.py",))
    assert st["dirty_hashed"] == [] and st["dirty_other"] == ["doc.md"]
    (tmp_path / "a.py").write_text("x = 2\n")
    assert S.git_state(tmp_path, ("a.py",))["dirty_hashed"] == ["a.py"]


def test_every_write_calls_both_engine_guards(repo, monkeypatch):
    calls = []
    monkeypatch.setattr(S, "refuse_old_engine_output", lambda out, k: calls.append(("old", out, k)))
    monkeypatch.setattr(S, "refuse_modified_engine_output", lambda out, p: calls.append(("mod", out, p)))
    S.write_json("results/m0d/h3/x.json", {"a": 1}, [Params(), GRADED])
    assert [c[0] for c in calls] == ["old", "mod", "old", "mod"]
    assert {c[1] for c in calls} == {"results/m0d/h3/x.json"}


@pytest.mark.parametrize("path", ["results/m0c/sparsity.json", "results/m0/x.json", "results/summary/m0c.json",
                                  "results/m0d_other/x.json", "elsewhere.json"])
def test_writes_outside_the_m0d_tree_are_refused_even_for_the_default_engine(repo, path):
    with pytest.raises(SystemExit) as e:
        S.write_json(path, {}, [Params()])
    assert e.value.code == 2 and not Path(path).exists()


def test_the_modified_engine_guard_fires_first_on_reference_trees(repo, capsys):
    with pytest.raises(SystemExit):
        S.write_json("results/m0c/x.json", {}, [GRADED])
    assert "M0d modes on" in capsys.readouterr().err


def test_writes_are_atomic_and_leave_no_temporaries(repo):
    p = S.write_json("results/m0d/h3/a/b.json", {"x": [1, 2]}, [Params()])
    assert json.loads(p.read_text()) == {"x": [1, 2]}
    assert [f.name for f in p.parent.iterdir()] == ["b.json"]


def test_the_cache_computes_once_and_serves_the_same_result(repo):
    root, files = repo
    code = S.code_key(root / "conn.npz", root, files)
    c = S.MeasureCache("results/m0d/h3/cache", code, "run1")
    n = []

    def compute():
        n.append(1)
        return [dict(x=1.5, seeds=(1, 2))]
    a = c.get_or_compute("kind", dict(params=GRADED, seeds=[1, 2]), compute, [GRADED])
    b = S.MeasureCache("results/m0d/h3/cache", code, "run2").get_or_compute(
        "kind", dict(params=GRADED, seeds=[1, 2]), compute, [GRADED])
    assert a == b == [dict(x=1.5, seeds=[1, 2])] and len(n) == 1


def test_the_cache_key_changes_with_params_inputs_and_code(repo):
    root, files = repo
    code = S.code_key(root / "conn.npz", root, files)
    c = S.MeasureCache("results/m0d/h3/cache", code, "r")
    k = c.key("reference", dict(params=GRADED))
    assert c.key("reference", dict(params=Params(apl_mode="graded", apl_input_scale=0.25))) != k
    assert c.key("rest", dict(params=GRADED)) != k
    other = S.MeasureCache("results/m0d/h3/cache", dict(code, key="0" * 64), "r")
    assert other.key("reference", dict(params=GRADED)) != k


def test_a_cache_file_with_another_key_is_recomputed(repo):
    root, files = repo
    code = S.code_key(root / "conn.npz", root, files)
    c = S.MeasureCache("results/m0d/h3/cache", code, "r")
    c.get_or_compute("k", {"i": 1}, lambda: 1, [Params()])
    path = next(Path("results/m0d/h3/cache/k").iterdir())
    d = json.loads(path.read_text())
    path.write_text(json.dumps(dict(d, key="stale", result=99)))
    assert c.get_or_compute("k", {"i": 1}, lambda: 2, [Params()]) == 2


def test_summary_block_replacement_keeps_other_blocks(repo):
    S.write_json("results/summary/m0d.json", {"h4": {"x": 1}}, [Params()])
    S.replace_summary_block("results/summary/m0d.json", "h3", {"y": 2}, [Params()])
    assert json.loads(Path("results/summary/m0d.json").read_text()) == {"h4": {"x": 1}, "h3": {"y": 2}}


def test_the_measurement_key_leaves_procedure_files_out_and_the_dirty_check_keeps_them():
    assert set(S.MEASURE_FILES) < set(S.HASHED_FILES)
    for f in ("flymon/brain/h3_rules.py", "flymon/brain/h3_runner.py", "flymon/brain/h3_c3.py",
              "flymon/brain/h3_records.py", "flymon/brain/h3_spec.py", "scripts/run_m0d_h3.py"):
        assert f in S.HASHED_FILES and f not in S.MEASURE_FILES
    for f in ("flymon/brain/engine_cpu.py", "flymon/brain/h3_jobs.py", "flymon/brain/measure.py", "uv.lock"):
        assert f in S.MEASURE_FILES


def test_the_adopted_c3_thresholds_may_be_copied_next_to_the_summary(repo):
    S.write_bytes("results/summary/m0d_h3_c3_thresholds.npz", b"x", [Params()])
    with pytest.raises(SystemExit):
        S.write_bytes("results/summary/other.npz", b"x", [Params()])
