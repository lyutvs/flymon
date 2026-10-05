"""Y's writer: results/y/ and results/summary/y_learning.json only, atomic, the ledger under budget; YCache's put goes
through it."""
import json
from pathlib import Path

import pytest

from flymon.brain import y_store as YS
from flymon.brain.config import Params


def test_guard_and_ledger(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    YS.write_json("results/y/a.json", {"a": 1}, [Params()])
    YS.write_summary_block(YS.SUMMARY, "digest", {"b": 2}, [Params()], ledger=dict(stage="digest", wall_s=1.0))
    YS.write_summary_block(YS.SUMMARY, "oracle", {"c": 3}, [Params()], ledger=dict(stage="oracle", wall_s=2.0))
    d = json.loads((tmp_path / YS.SUMMARY).read_text())
    assert d["digest"] == {"b": 2} and [e["stage"] for e in d["budget"]["ledger"]] == ["digest", "oracle"]
    for bad in ("results/w/x.json", "results/x/x.json", "results/v/cache/x.json", "results/summary/x_learning.json",
                "results/yy/a.json", "y.json"):
        with pytest.raises(SystemExit) as e:
            YS.write_json(bad, {}, [Params()])
        assert e.value.code == 2
    assert not list((tmp_path / "results/y").glob(".*.tmp"))


def test_ycache_round_trip_and_refuses_outside(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    c = YS.YCache("results/y/cache", {"key": "k" * 64})
    ins = dict(pair="b|1|x|y", report_seeds=[24_700_200])
    assert c.get("r_oracle", ins) is None
    c.put("r_oracle", ins, {"v": 1}, [Params()])
    assert c.get("r_oracle", ins) == {"v": 1}
    with pytest.raises(SystemExit):
        YS.YCache("results/w/cache", {"key": "k" * 64}).put("r_oracle", ins, {"v": 1}, [Params()])


def test_archive_stage_fixed_set_idempotent_and_never_overwrites(tmp_path, monkeypatch):
    """Y red-team P3-13 (as amended by Y.9.2) + Task 1 review M2: each stage archives a fixed file set (digest: none;
    oracle: oracle.json) at every block commit; a repeat with the identical set is a no-op returning the same manifest,
    anything else refuses; a missing source refuses."""
    import dataclasses

    from flymon.brain.h3_store import sha256_file
    from flymon.brain.y_spec import SPEC as Y
    monkeypatch.chdir(tmp_path)
    ys = dataclasses.replace(Y, archive_root=str(tmp_path / "archive/y"))
    assert YS.stage_files("digest", ys) == [] and YS.stage_files("oracle", ys) == [Y.oracle_detail]
    assert YS.archive_stage("digest", ys) == [] and not (tmp_path / "archive/y/digest").exists()
    with pytest.raises(SystemExit) as e:
        YS.archive_stage("oracle", ys)                                 # oracle.json missing
    assert e.value.code == 2 and not (tmp_path / "archive/y/oracle").exists()
    orc = YS.write_json(Y.oracle_detail, {"pairs": []}, [Params()])
    m = YS.archive_stage("oracle", ys)
    assert [Path(e["dst"]).relative_to(tmp_path) for e in m] == [Path("archive/y/oracle/y/oracle.json")]
    assert m[0]["sha256"] == sha256_file(orc) == sha256_file(m[0]["dst"])
    assert YS.archive_stage("oracle", ys) == m == YS.archive_stage("oracle", ys)    # idempotent
    assert YS.archive_stage("digest", ys) == []
    (tmp_path / "archive/y/oracle/y/extra.json").write_text("{}")      # not the fixed set -> refuse
    with pytest.raises(SystemExit):
        YS.archive_stage("oracle", ys)
    (tmp_path / "archive/y/oracle/y/extra.json").unlink()
    YS.write_json(Y.oracle_detail, {"pairs": [1]}, [Params()])         # differing source -> refuse
    with pytest.raises(SystemExit) as e:
        YS.archive_stage("oracle", ys)
    assert e.value.code == 2 and sha256_file(m[0]["dst"]) == m[0]["sha256"]
    for bad in ("../outside", "pilot"):
        with pytest.raises(SystemExit):
            YS.archive_stage(bad, ys)
