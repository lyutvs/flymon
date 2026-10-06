"""aa_store: AA writes only under results/aa/ and results/summary/aa_learning.json (symlink escapes refused), atomic;
two ledgers and the candidates block; AACache never overwrites a raw entry; ReadCache never writes; archives are never
overwritten."""
import dataclasses
import json
import os
from pathlib import Path

import pytest

from flymon.brain import aa_store as S
from flymon.brain.aa_spec import SPEC
from flymon.brain.config import Params

P = [Params()]


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.parametrize("bad", ["results/y/x.json", "results/z/x.json", "results/summary/y_learning.json",
                                 "results/summary/z_learning.json", "results/aax/x.json", "elsewhere.json"])
def test_guard_refuses_outside(cwd, bad):
    with pytest.raises(SystemExit) as e:
        S.write_json(bad, {}, P)
    assert e.value.code == 2 and not Path(bad).exists()


def test_symlink_escape_refused(cwd, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    (cwd / "results").mkdir()
    (cwd / "results/aa").symlink_to(other)
    with pytest.raises(SystemExit) as e:
        S.write_json("results/aa/x.json", {}, P)
    assert e.value.code == 2 and list(other.iterdir()) == []


def test_write_is_atomic_on_failure(cwd, monkeypatch):
    S.write_json("results/aa/a.json", {"x": 1}, P)
    before = Path("results/aa/a.json").read_bytes()

    def boom(*a, **k):
        raise OSError("simulated failure mid-write")
    monkeypatch.setattr(S.os, "replace", boom)
    with pytest.raises(OSError):
        S.write_json("results/aa/a.json", {"x": 2}, P)
    assert Path("results/aa/a.json").read_bytes() == before                    # old content intact
    assert sorted(p.name for p in Path("results/aa").iterdir()) == ["a.json"]   # no temporary file left
    with pytest.raises(OSError):
        S.write_json("results/aa/b.json", {"x": 3}, P)
    assert not Path("results/aa/b.json").exists()                              # no partial target


def test_exclusive_write_refuses_existing(cwd):
    S.write_json("results/aa/e.json", {"x": 1}, P, exclusive=True)
    before = Path("results/aa/e.json").read_bytes()
    with pytest.raises(SystemExit) as e:
        S.write_json("results/aa/e.json", {"x": 2}, P, exclusive=True)
    assert e.value.code == 2 and Path("results/aa/e.json").read_bytes() == before
    assert sorted(p.name for p in Path("results/aa").iterdir()) == ["e.json"]


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


def test_aacache_put_once_then_refuses(cwd):
    c = S.AACache(SPEC.cache_dir, {"key": "w"}, smoke_seeds=())
    ins = dict(probe_seeds=[62_000_000], pair="p", brain="R", block="screen")
    c.put("w_learn", ins, {"v": 1}, P)
    assert c.get("w_learn", ins) == {"v": 1}
    with pytest.raises(SystemExit) as e:
        c.put("w_learn", ins, {"v": 2}, P)                       # AA.7 3: raw cache immutable
    assert e.value.code == 2 and c.get("w_learn", ins) == {"v": 1}


def test_aacache_concurrent_put_refused_not_overwritten(cwd, monkeypatch):
    """Another writer lands the entry between put's exists-check and its publish: the put refuses, the other entry
    stays."""
    c = S.AACache(SPEC.cache_dir, {"key": "w"}, smoke_seeds=())
    ins = dict(probe_seeds=[62_000_000], pair="p", brain="R", block="screen")
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


def test_aacache_smoke_scope(cwd):
    smoke = frozenset([87_100_000])
    with pytest.raises(SystemExit):
        S.AACache(SPEC.cache_dir, {"key": "w"}, smoke).put("w_learn", dict(probe_seeds=[87_100_000]), {}, P)
    with pytest.raises(SystemExit):
        S.AACache(SPEC.smoke_cache_dir, {"key": "w"}, smoke).put("w_learn", dict(probe_seeds=[62_000_000]), {}, P)


def test_readcache_never_writes(cwd):
    r = S.ReadCache("results/y/cache", {"key": "w"})
    with pytest.raises(SystemExit):
        r.put("w_learn", dict(probe_seeds=[60_000_000]), {}, P)
    assert not Path("results/y").exists()


def test_archive_never_overwrites(cwd, tmp_path):
    s = dataclasses.replace(SPEC, archive_root=str(tmp_path / "arch"))
    S.write_json("results/aa/a.json", {"x": 1}, P)
    m = S.archive(["results/aa/a.json"], "screen", s)
    assert len(m) == 1 and S.archive(["results/aa/a.json"], "screen", s) == m      # idempotent repeat
    S.write_json("results/aa/a.json", {"x": 2}, P)
    with pytest.raises(SystemExit):
        S.archive(["results/aa/a.json"], "screen", s)


def test_archive_different_file_set_refused(cwd, tmp_path):
    s = dataclasses.replace(SPEC, archive_root=str(tmp_path / "arch"))
    S.write_json("results/aa/a.json", {"x": 1}, P)
    S.write_json("results/aa/b.json", {"x": 2}, P)
    S.archive(["results/aa/a.json"], "screen", s)
    with pytest.raises(SystemExit) as e:
        S.archive(["results/aa/a.json", "results/aa/b.json"], "screen", s)
    assert e.value.code == 2 and not (tmp_path / "arch/screen/aa/b.json").exists()


@pytest.mark.parametrize("exists", [False, True])
def test_archive_refuses_escape_from_root(cwd, tmp_path, exists):
    s = dataclasses.replace(SPEC, archive_root=str(tmp_path / "arch"))
    S.write_json("results/aa/a.json", {"x": 1}, P)
    (tmp_path / "arch").mkdir()
    if exists:                                     # an existing destination outside the root holding the same set
        (tmp_path / "x/aa").mkdir(parents=True)
        (tmp_path / "x/aa/a.json").write_bytes(Path("results/aa/a.json").read_bytes())
    with pytest.raises(SystemExit) as e:
        S.archive(["results/aa/a.json"], "../x", s)
    assert e.value.code == 2
    assert (tmp_path / "x").exists() == exists and list((tmp_path / "arch").iterdir()) == []
