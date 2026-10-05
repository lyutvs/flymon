"""W's writer (W.8): writes only under results/w/ and results/summary/w_learning.json (SystemExit 2 otherwise, R / S /
T / U / V trees included), atomically; the budget ledger is appended with a block; WCache keeps smoke and real seeds
apart; V's cache is read only."""
import json

import pytest

from flymon.brain import w_store as WS
from flymon.brain.config import Params
from flymon.brain.w_spec import SPEC


def test_guard(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    WS.write_json("results/w/x.json", {"a": 1}, [Params()])
    WS.write_summary_block(WS.SUMMARY, "reuse", {"b": 2}, [Params()], ledger=dict(stage="reuse", wall_s=1.0))
    WS.write_summary_block(WS.SUMMARY, "path", {"c": 3}, [Params()], ledger=dict(stage="path", wall_s=2.0))
    d = json.loads((tmp_path / WS.SUMMARY).read_text())
    assert d["reuse"] == {"b": 2} and [e["stage"] for e in d["ledger"]] == ["reuse", "path"]
    for bad in ("results/v/x.json", "results/summary/v_lever.json", "results/u/x.json", "results/r/cache/x.json",
                "x.json", "results/wx/a.json"):
        with pytest.raises(SystemExit) as e:
            WS.write_json(bad, {}, [Params()])
        assert e.value.code == 2
    assert not list((tmp_path / "results/w").glob(".*.tmp"))


def test_caches(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    c = WS.WCache("results/w/cache", {"key": "k"})
    s = WS.WCache("results/w/smoke/cache", {"key": "k"})
    real = dict(probe_seeds=SPEC.probe_seeds(0, 0, 2))
    sm = dict(probe_seeds=SPEC.smoke_probe_seeds(0, 2))
    c.put("w_learn", real, {"r": 1}, [Params()])
    assert c.get("w_learn", real) == {"r": 1}
    s.put("w_learn", sm, {"r": 2}, [Params()])
    for cache, ins in ((c, sm), (s, real)):
        with pytest.raises(SystemExit):
            cache.put("w_learn", ins, {}, [Params()])
    v = WS.VReadCache("results/v/cache", {"key": "u"})
    assert v.get("r_arm", dict(seed=25_400_000)) is None
    with pytest.raises(SystemExit):
        v.put("r_arm", dict(seed=25_400_000), {}, [Params()])
