import json
import numpy as np
import pytest
from flymon.brain.config import Params
from flymon.rescope import store

def test_guard_allows_only_rescope_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for ok in ("results/rescope/a.json", "results/rescope-smoke/x/y.json", "results/summary/rescope_primary.json"):
        assert store.guard(ok, [Params()])
    for bad in ("results/b/a.json", "results/summary/m0d.json", "results/m0d/x.json", "elsewhere.json"):
        with pytest.raises(SystemExit):
            store.guard(bad, [Params()])

def test_write_json_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = store.write_json("results/rescope/t.json", {"b": 1, "a": [1, 2]}, [Params()])
    assert json.loads(p.read_text()) == {"a": [1, 2], "b": 1}

def test_checkpoint_roundtrip_and_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ck = store.Checkpoint("results/rescope/pairX", "key1", [Params()])
    assert ck.load() is None
    st = dict(records=[{"brain": "Rr"}], x="a", done=["pre"], provenance={"commit": "c"},
              weights={"flies": [dict(enabled=True, shuffle_seed=None, w=np.arange(3, dtype=np.float32))]})
    ck.save(st)
    got = ck.load()
    assert got["x"] == "a" and got["done"] == ["pre"] and got["records"] == [{"brain": "Rr"}]
    assert np.array_equal(got["weights"]["flies"][0]["w"], np.arange(3, dtype=np.float32))
    assert store.Checkpoint("results/rescope/pairX", "key2", [Params()]).load() is None
