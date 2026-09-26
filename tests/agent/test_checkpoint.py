import json

import numpy as np
import pytest

from flymon.agent.checkpoint import CheckpointStore, CrashForTest, filter_log


def _state(x):
    return {"flies": [{"enabled": True, "shuffle_seed": None, "w": np.full(5, x, np.float32)}]}


def test_commit_and_load_round_trip(tmp_path):
    s = CheckpointStore(tmp_path, "h")
    assert s.load() is None
    s.commit("f00-b000", _state(1.0), {"0": 1})
    got = CheckpointStore(tmp_path, "h").load()
    assert got["completed"] == ["f00-b000"] and got["cursor"] == {"0": 1}
    assert np.array_equal(got["pool_state"]["flies"][0]["w"], np.full(5, 1.0, np.float32))


def test_only_two_generations_are_kept(tmp_path):
    s = CheckpointStore(tmp_path, "h")
    for i in range(4):
        s.commit(f"f00-b{i:03d}", _state(float(i)), {"0": i + 1})
    assert sorted(p.name for p in tmp_path.glob("state_*.npz")) == ["state_000003.npz", "state_000004.npz"]
    assert CheckpointStore(tmp_path, "h").load()["completed"] == [f"f00-b{i:03d}" for i in range(4)]


def test_crash_while_writing_state_keeps_previous(tmp_path):
    s = CheckpointStore(tmp_path, "h")
    s.commit("f00-b000", _state(1.0), {"0": 1})
    s.fault = "state"
    with pytest.raises(CrashForTest):
        s.commit("f00-b001", _state(2.0), {"0": 2})
    got = CheckpointStore(tmp_path, "h").load()
    assert got["completed"] == ["f00-b000"] and got["pool_state"]["flies"][0]["w"][0] == 1.0


def test_crash_before_manifest_rename_resumes_previous(tmp_path):
    s = CheckpointStore(tmp_path, "h")
    s.commit("f00-b000", _state(1.0), {"0": 1})
    s.fault = "before_manifest"
    with pytest.raises(CrashForTest):
        s.commit("f00-b001", _state(2.0), {"0": 2})
    assert CheckpointStore(tmp_path, "h").load()["completed"] == ["f00-b000"]


def test_corrupt_newest_state_falls_back_one_generation(tmp_path):
    s = CheckpointStore(tmp_path, "h")
    s.commit("f00-b000", _state(1.0), {"0": 1})
    s.commit("f00-b001", _state(2.0), {"0": 2})
    (tmp_path / "state_000002.npz").write_bytes(b"garbage")
    got = CheckpointStore(tmp_path, "h").load()
    assert got["generation"] == 1 and got["completed"] == ["f00-b000"]


def test_config_hash_mismatch_refuses(tmp_path):
    CheckpointStore(tmp_path, "h").commit("f00-b000", _state(1.0), {"0": 1})
    with pytest.raises(ValueError, match="config hash"):
        CheckpointStore(tmp_path, "other").load()


def test_resume_drops_uncommitted_log_records(tmp_path):
    log = tmp_path / "fly00.jsonl"
    recs = [{"battle_id": "f00-b000", "kind": "decision"}, {"battle_id": "f00-b001", "kind": "decision"},
            {"battle_id": "f00-b001", "kind": "outc"}]
    log.write_text("".join(json.dumps(r) + "\n" for r in recs) + '{"battle_id": "f00-b00')    # torn last line
    assert filter_log(log, {"f00-b000"}) == 3
    assert [json.loads(x)["battle_id"] for x in log.read_text().splitlines()] == ["f00-b000"]
