"""V's writer and caches (V.5, V.8): writes only under results/v/ and results/summary/v_lever.json, atomically — every
other track's path (results/u/, results/t/, results/r/, results/s/, results/p/, their summaries) refuses; VCache is
RCache with V's writer and V's smoke seeds; RReadCache is T's (R's root, read only); r_store's load_manifest and
archive_copy are reused."""
import json
from pathlib import Path

import pytest

from flymon.brain import r_store, t_store
from flymon.brain import v_store as V
from flymon.brain.v_spec import SPEC, smoke


def test_guard_refuses_every_other_track(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for bad in ("results/u/x.json", "results/u/cache/r_act/a.json", "results/t/z.json", "results/r/cache/a.json",
                "results/s/x.json", "results/p/x.json", "results/summary/u_lever.json", "results/summary/t_lever.json",
                "results/summary/r_lever.json", "results/vx/a.json", "elsewhere.json"):
        with pytest.raises(SystemExit) as e:
            V.write_json(bad, {"a": 1}, [])
        assert e.value.code == 2, bad
    assert json.loads(V.write_json("results/v/x.json", {"a": 1}, []).read_text()) == {"a": 1}
    V.write_summary_block(V.SUMMARY, "reuse", {"n": 1}, [])
    V.write_summary_block(V.SUMMARY, "path", {"m": 2}, [])
    assert V.read_summary() == {"reuse": {"n": 1}, "path": {"m": 2}}
    assert not list(Path("results/summary").glob(".*.tmp"))
    assert V.SUMMARY == SPEC.summary and V.ALLOWED_DIR == "results/v/"


def test_guard_refuses_a_symlink_escape(tmp_path, monkeypatch):
    root, outside = tmp_path / "repo", tmp_path / "outside"
    (root / "results/v").mkdir(parents=True)
    outside.mkdir()
    (root / "results/v/link").symlink_to(outside, target_is_directory=True)
    monkeypatch.chdir(root)
    with pytest.raises(SystemExit):
        V.write_json("results/v/link/x.json", {"a": 1}, [])
    assert not (outside / "x.json").exists()


def test_vcache_scope_and_writer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    real, sm = V.VCache(SPEC.cache_dir, {"key": "u"}), V.VCache(SPEC.smoke_cache_dir, {"key": "u"})
    assert isinstance(real, r_store.RCache) and {k for k in vars(V.VCache) if not k.startswith("__")} == {"put"}
    assert V.SMOKE_SEEDS == frozenset(SPEC.smoke_seeds) | frozenset(smoke(SPEC).p.seeds)
    real.put("r_oracle", {"act_seeds": [24_600_000]}, {"v": 1}, [])
    assert real.get("r_oracle", {"act_seeds": [24_600_000]}) == {"v": 1}
    with pytest.raises(SystemExit):
        real.get("r_arm", {"seed": 25_409_100})                   # V's P smoke seed under the real root
    with pytest.raises(SystemExit):
        sm.get("r_oracle", {"act_seeds": [24_600_000]})           # a real seed under the smoke root
    bad = V.VCache("results/u/cache", {"key": "u"})
    with pytest.raises(SystemExit):
        bad.put("r_act", {"edit": "x"}, {"v": 1}, [])
    assert V.RReadCache is t_store.RReadCache and V.load_manifest is r_store.load_manifest
    assert V.archive_copy is r_store.archive_copy
