"""Z's writer (Z.9.2 P1-5 T3): results/z/ and the two Z summaries only, atomic writes, ledgers, archives never
overwritten."""
import dataclasses
import json
from pathlib import Path

import pytest

from flymon.brain import z_store as S
from flymon.brain.config import Params
from flymon.brain.z_spec import SPEC as Z

PL = [Params()]


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.parametrize("path", ["results/y/x.json", "results/summary/y_learning.json", "results/w/a.json",
                                  "flymon/brain/z_spec.py", "results/zz/a.json", "results/summary/x.json"])
def test_guard_refuses(cwd, path):
    with pytest.raises(SystemExit) as e:
        S.write_json(path, {}, PL)
    assert e.value.code == 2 and not Path(path).exists()


def test_writes_allowed_atomically(cwd):
    for p in ("results/z/a/b.json", S.SUMMARY, S.DRAWS):
        S.write_json(p, dict(x=1), PL)
        assert json.loads(Path(p).read_text()) == {"x": 1}
    assert not list(Path("results/z/a").glob(".*.tmp"))
    assert sorted(x.name for x in Path("results/z/a").iterdir()) == ["b.json"]          # nothing beside the file
    assert sorted(x.name for x in Path("results/summary").iterdir()) == sorted(Path(q).name for q in (S.SUMMARY, S.DRAWS))


def test_symlink_escape_refused(cwd, tmp_path):
    (cwd / "results").mkdir(exist_ok=True)
    other = tmp_path.parent / f"{tmp_path.name}_y"
    other.mkdir()
    (cwd / "results/z").symlink_to(other)
    with pytest.raises(SystemExit):
        S.write_json("results/z/a.json", {}, PL)


def test_ledgers(cwd):
    S.write_summary_block(S.SUMMARY, "stage0", dict(outcome="PASS"), PL, dict(stage="stage0", wall_s=2.0),
                          dict(stage="stage0", wall_s=1.0))
    S.append_ledger(S.SUMMARY, dict(stage="ydiag", env_mismatch=["numpy"]), PL)
    d = S.read_summary()
    assert d["stage0"] == {"outcome": "PASS"} and len(d["budget"]["ledger"]) == 2
    assert d["budget"]["records_ledger"] == [{"stage": "stage0", "wall_s": 1.0}]


def test_archive_never_overwrites(cwd, tmp_path):
    zs = dataclasses.replace(Z, archive_root=str(tmp_path / "arch"))
    S.write_json("results/z/split.json", dict(a=1), PL)
    m = S.archive(["results/z/split.json"], "split", zs)
    assert Path(m[0]["dst"]).exists()
    assert S.archive(["results/z/split.json"], "split", zs) == m           # idempotent repeat
    S.write_json("results/z/split.json", dict(a=2), PL)
    with pytest.raises(SystemExit):
        S.archive(["results/z/split.json"], "split", zs)
    with pytest.raises(SystemExit):
        S.archive(["results/z/missing.json"], "other", zs)
    assert S.archive([], "empty", zs) == []


def test_archive_refuses_another_file_set(cwd, tmp_path):
    zs = dataclasses.replace(Z, archive_root=str(tmp_path / "arch"))
    for n in ("a", "b"):
        S.write_json(f"results/z/{n}.json", dict(n=n), PL)
    S.archive(["results/z/a.json", "results/z/b.json"], "pair", zs)
    with pytest.raises(SystemExit) as e:                          # a subset of the archived set
        S.archive(["results/z/a.json"], "pair", zs)
    assert e.value.code == 2
    S.archive(["results/z/a.json"], "one", zs)
    with pytest.raises(SystemExit) as e:                          # a superset of the archived set
        S.archive(["results/z/a.json", "results/z/b.json"], "one", zs)
    assert e.value.code == 2
