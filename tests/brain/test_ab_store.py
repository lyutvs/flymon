"""ab_store: AB writes only under results/ab/ and results/summary/ab_learning.json (symlink escapes refused), atomic;
two ledgers and the candidates block; ABCache never overwrites a raw entry; ReadCache never writes; archives are never
overwritten."""
import dataclasses
import json
import os
from pathlib import Path

import pytest

from flymon.brain import ab_store as S
from flymon.brain.ab_spec import SPEC
from flymon.brain.config import Params

P = [Params()]


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.parametrize("bad", ["results/aa/x.json", "results/aa/cache/w_learn/x.json", "results/y/x.json",
                                 "results/z/x.json", "results/summary/aa_learning.json", "results/summary/y_learning.json",
                                 "results/summary/z_learning.json", "results/abx/x.json", "elsewhere.json"])
def test_guard_refuses_outside(cwd, bad):
    with pytest.raises(SystemExit) as e:
        S.write_json(bad, {}, P)
    assert e.value.code == 2 and not Path(bad).exists()


def test_symlink_escape_refused(cwd, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    (cwd / "results").mkdir()
    (cwd / "results/ab").symlink_to(other)
    with pytest.raises(SystemExit) as e:
        S.write_json("results/ab/x.json", {}, P)
    assert e.value.code == 2 and list(other.iterdir()) == []


@pytest.mark.parametrize("name", ["aa", "y"])
def test_symlinked_old_results_dirs_refused_by_resolution(cwd, tmp_path, name):
    """results/<aa|y> are symlinks to directories outside the checkout, and results/ab/via_<name> links back to them:
    the lexical path results/ab/via_<name>/x.json looks allowed, but the guard resolves it and refuses (exit 2); the
    target directory is untouched."""
    target = tmp_path / "outside" / name
    target.mkdir(parents=True)
    (target / "keep.json").write_text("{}")
    (cwd / "results/ab").mkdir(parents=True)
    (cwd / "results" / name).symlink_to(target)
    (cwd / "results/ab" / f"via_{name}").symlink_to(Path("..") / name)
    for path in (f"results/{name}/x.json", f"results/ab/via_{name}/x.json"):
        with pytest.raises(SystemExit) as e:
            S.write_json(path, {}, P)
        assert e.value.code == 2, path
    assert sorted(p.name for p in target.iterdir()) == ["keep.json"] and (target / "keep.json").read_text() == "{}"


def test_write_is_atomic_on_failure(cwd, monkeypatch):
    S.write_json("results/ab/a.json", {"x": 1}, P)
    before = Path("results/ab/a.json").read_bytes()

    def boom(*a, **k):
        raise OSError("simulated failure mid-write")
    monkeypatch.setattr(S.os, "replace", boom)
    with pytest.raises(OSError):
        S.write_json("results/ab/a.json", {"x": 2}, P)
    assert Path("results/ab/a.json").read_bytes() == before                    # old content intact
    assert sorted(p.name for p in Path("results/ab").iterdir()) == ["a.json"]   # no temporary file left
    with pytest.raises(OSError):
        S.write_json("results/ab/b.json", {"x": 3}, P)
    assert not Path("results/ab/b.json").exists()                              # no partial target


def test_exclusive_write_refuses_existing(cwd):
    S.write_json("results/ab/e.json", {"x": 1}, P, exclusive=True)
    before = Path("results/ab/e.json").read_bytes()
    with pytest.raises(SystemExit) as e:
        S.write_json("results/ab/e.json", {"x": 2}, P, exclusive=True)
    assert e.value.code == 2 and Path("results/ab/e.json").read_bytes() == before
    assert sorted(p.name for p in Path("results/ab").iterdir()) == ["e.json"]


def test_ledgers_append_and_candidates_replace(cwd):
    S.write_summary_block(S.SUMMARY, "a", 1, P, dict(stage="a"), candidates=[dict(key="k1")])
    S.write_summary_block(S.SUMMARY, "b", 2, P, dict(stage="b"), candidates=[dict(key="k2")])
    d = S.read_summary()
    assert [e["stage"] for e in d["budget"]["ledger"]] == ["a", "b"] and d["candidates"] == [dict(key="k2")]
    S.write_summary_block(S.SUMMARY, "c", 3, P, dict(stage="c"))                # candidates=None keeps it
    d = S.read_summary()
    assert [e["stage"] for e in d["budget"]["ledger"]] == ["a", "b", "c"] and d["candidates"] == [dict(key="k2")]
    assert (d["a"], d["b"], d["c"]) == (1, 2, 3)


def test_ledgers_and_candidates(cwd):
    S.write_summary_block(S.SUMMARY, "stage0", dict(outcome="PASS"), P, dict(stage="stage0", wall_s=1.0),
                          candidates=[dict(key="k", c=0, axis="b", state="untouched")])
    S.append_ledger(S.SUMMARY, dict(stage="reuse", env_mismatch=["numpy"], wall_s=0.0), P)
    S.write_summary_block(S.SUMMARY, "records", dict(outcome="PASS"), P, None, dict(stage="records", wall_s=2.0))
    d = S.read_summary()
    assert [e["stage"] for e in d["budget"]["ledger"]] == ["stage0", "reuse"]
    assert d["budget"]["records_ledger"][0]["wall_s"] == 2.0 and d["candidates"][0]["state"] == "untouched"


def test_abcache_put_once_then_refuses(cwd):
    c = S.ABCache(SPEC.cache_dir, {"key": "w"}, smoke_seeds=())
    ins = dict(probe_seeds=[88_100_000], pair="p", brain="R", block="screen")
    c.put("w_learn", ins, {"v": 1}, P)
    assert c.get("w_learn", ins) == {"v": 1}
    with pytest.raises(SystemExit) as e:
        c.put("w_learn", ins, {"v": 2}, P)                       # raw cache immutable
    assert e.value.code == 2 and c.get("w_learn", ins) == {"v": 1}


def test_abcache_concurrent_put_refused_not_overwritten(cwd, monkeypatch):
    """Another writer lands the entry between put's exists-check and its publish: the put refuses, the other entry
    stays."""
    c = S.ABCache(SPEC.cache_dir, {"key": "w"}, smoke_seeds=())
    ins = dict(probe_seeds=[88_100_000], pair="p", brain="R", block="screen")
    target = c._path("w_learn", ins)
    real_link = os.link

    def racing_link(src, dst):
        Path(dst).write_text(json.dumps({"key": c.key("w_learn", ins), "kind": "w_learn", "result": {"v": "other"}}))
        return real_link(src, dst)
    monkeypatch.setattr(S.os, "link", racing_link)
    with pytest.raises(SystemExit) as e:
        c.put("w_learn", ins, {"v": "mine"}, P)
    assert e.value.code == 2 and c.get("w_learn", ins) == {"v": "other"}
    assert [p.name for p in target.parent.iterdir()] == [target.name]          # temporary file removed


def test_abcache_smoke_scope(cwd):
    smoke = frozenset([88_040_000])
    with pytest.raises(SystemExit):
        S.ABCache(SPEC.cache_dir, {"key": "w"}, smoke).put("w_learn", dict(probe_seeds=[88_040_000]), {}, P)
    with pytest.raises(SystemExit):
        S.ABCache(SPEC.smoke_cache_dir, {"key": "w"}, smoke).put("w_learn", dict(probe_seeds=[88_100_000]), {}, P)


def test_readcache_never_writes(cwd):
    r = S.ReadCache("results/aa/cache", {"key": "w"})
    with pytest.raises(SystemExit):
        r.put("w_learn", dict(probe_seeds=[62_000_000]), {}, P)
    assert not Path("results/aa").exists()


def test_archive_never_overwrites(cwd, tmp_path):
    s = dataclasses.replace(SPEC, archive_root=str(tmp_path / "arch"))
    S.write_json("results/ab/a.json", {"x": 1}, P)
    m = S.archive(["results/ab/a.json"], "screen", s)
    assert len(m) == 1 and S.archive(["results/ab/a.json"], "screen", s) == m      # idempotent repeat
    S.write_json("results/ab/a.json", {"x": 2}, P)
    with pytest.raises(SystemExit):
        S.archive(["results/ab/a.json"], "screen", s)


def test_archive_different_file_set_refused(cwd, tmp_path):
    s = dataclasses.replace(SPEC, archive_root=str(tmp_path / "arch"))
    S.write_json("results/ab/a.json", {"x": 1}, P)
    S.write_json("results/ab/b.json", {"x": 2}, P)
    S.archive(["results/ab/a.json"], "screen", s)
    with pytest.raises(SystemExit) as e:
        S.archive(["results/ab/a.json", "results/ab/b.json"], "screen", s)
    assert (tmp_path / "arch/screen/ab/a.json").exists()                      # the live layout <root>/<dest>/ab/
    assert e.value.code == 2 and not (tmp_path / "arch/screen/ab/b.json").exists()


@pytest.mark.parametrize("exists", [False, True])
def test_archive_refuses_escape_from_root(cwd, tmp_path, exists):
    s = dataclasses.replace(SPEC, archive_root=str(tmp_path / "arch"))
    S.write_json("results/ab/a.json", {"x": 1}, P)
    (tmp_path / "arch").mkdir()
    if exists:                                     # an existing destination outside the root holding the same set
        (tmp_path / "x/ab").mkdir(parents=True)
        (tmp_path / "x/ab/a.json").write_bytes(Path("results/ab/a.json").read_bytes())
    with pytest.raises(SystemExit) as e:
        S.archive(["results/ab/a.json"], "../x", s)
    assert e.value.code == 2
    assert (tmp_path / "x").exists() == exists and list((tmp_path / "arch").iterdir()) == []
