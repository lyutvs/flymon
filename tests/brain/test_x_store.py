"""X's writer (X.2 · X.8): results/x/ and results/summary/x_learning.json only, atomic, the ledger appended."""
import json

import pytest

from flymon.brain import x_store as XS
from flymon.brain.config import Params


def test_guard(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    XS.write_json("results/x/a.json", {"a": 1}, [Params()])
    XS.write_summary_block(XS.SUMMARY, "stage0", {"b": 2}, [Params()], ledger=dict(stage="stage0", wall_s=1.0))
    d = json.loads((tmp_path / XS.SUMMARY).read_text())
    assert d["stage0"] == {"b": 2} and d["ledger"][0]["stage"] == "stage0"
    for bad in ("results/w/oc.json", "results/summary/w_learning.json", "results/v/x.json", "x.json",
                "results/xx/a.json"):
        with pytest.raises(SystemExit) as e:
            XS.write_json(bad, {}, [Params()])
        assert e.value.code == 2
    assert not list((tmp_path / "results/x").glob(".*.tmp"))


def test_numpy_values_are_written_as_plain_json(tmp_path, monkeypatch):
    import numpy as np
    monkeypatch.chdir(tmp_path)
    obj = dict(ok=np.bool_(True), f=np.float64(0.25), i=np.int64(7), a=np.arange(3), t=(np.float32(0.5), 1),
               nested={1: [np.bool_(False)]})
    want = dict(ok=True, f=0.25, i=7, a=[0, 1, 2], t=[0.5, 1], nested={"1": [False]})
    assert XS.to_json(obj) == want
    XS.write_json("results/x/n.json", obj, [Params()])
    assert json.loads((tmp_path / "results/x/n.json").read_text()) == want
