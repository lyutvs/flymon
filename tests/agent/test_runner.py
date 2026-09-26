import asyncio
import json
import shutil

import numpy as np
import pytest

from flymon.agent.checkpoint import CheckpointStore
from flymon.agent.runner import run_cohort
from flymon.battle.schedule import make_schedule


def _state(w):
    return {"flies": [{"enabled": True, "shuffle_seed": None, "w": w[i]} for i in range(len(w))]}


def _zeros(n=2):
    return {i: np.zeros(3, np.float32) for i in range(n)}


async def test_resume_skips_committed_battles_and_continues_identically(tmp_path):
    sched = make_schedule(2, 3, seed=0)
    w = {0: np.zeros(3, np.float32), 1: np.zeros(3, np.float32)}

    async def play(sb):
        w[sb.fly_id] = w[sb.fly_id] + np.float32(1 + int(sb.battle_id[-3:]))

    store = CheckpointStore(tmp_path / "ck", "h")
    full = await run_cohort(sched, store, tmp_path / "logs", play, lambda: _state(w))
    want = {k: v.copy() for k, v in w.items()}

    w = {0: np.zeros(3, np.float32), 1: np.zeros(3, np.float32)}
    store2 = CheckpointStore(tmp_path / "ck2", "h")
    await run_cohort(sched, store2, tmp_path / "logs2", play, lambda: _state(w), stop_after=3)
    ck = CheckpointStore(tmp_path / "ck2", "h").load()
    w = {i: np.asarray(f["w"]).copy() for i, f in enumerate(ck["pool_state"]["flies"])}
    res = await run_cohort(sched, CheckpointStore(tmp_path / "ck2", "h"), tmp_path / "logs2", play, lambda: _state(w))
    assert len(res["skipped"]) == 3 and len(res["skipped"]) + len(res["played"]) == len(full["played"])
    assert all(np.array_equal(w[k], want[k]) for k in want)


class _SpyStore(CheckpointStore):
    """Records every committed pool state and snapshots the checkpoint dir after the first commit."""

    def __init__(self, root, h, snap):
        super().__init__(root, h)
        self.snap, self.commits = snap, []

    def commit(self, battle_id, pool_state, cursor):
        gen = super().commit(battle_id, pool_state, cursor)
        self.commits.append((battle_id, [np.asarray(f["w"]).copy() for f in pool_state["flies"]]))
        if len(self.commits) == 1:
            shutil.copytree(self.root, self.snap)
        return gen


def _interleaved_play(w):
    """Fly 0's battles are short; fly 1's battle changes its weights in several steps with yields in between, so
    fly 0 finishes (and commits) while fly 1 is mid-battle."""

    async def play(sb):
        f = sb.fly_id
        steps = 1 if f == 0 else 4
        for _ in range(steps):
            w[f] = w[f] + np.float32(1 + int(sb.battle_id[-3:]))
            await asyncio.sleep(0)

    return play


async def test_commit_holds_only_committed_state_of_other_flies(tmp_path):
    """Ruling 2: a commit made after fly 0's battle must not carry fly 1's half-played battle, and resuming from that
    checkpoint reproduces the uninterrupted final weights."""
    sched = make_schedule(2, 2, seed=0)
    w = _zeros()
    ref = await run_cohort(sched, CheckpointStore(tmp_path / "ref", "h"), tmp_path / "ref_logs",
                           _interleaved_play(w), lambda: _state(w))
    want = {k: v.copy() for k, v in w.items()}
    assert len(ref["played"]) == 4

    w = _zeros()
    spy = _SpyStore(tmp_path / "ck", "h", tmp_path / "snap")
    await run_cohort(sched, spy, tmp_path / "logs", _interleaved_play(w), lambda: _state(w))
    first_id, first_ws = spy.commits[0]
    assert first_id == "f00-b000"
    assert np.array_equal(first_ws[0], np.full(3, 1, np.float32))
    assert np.array_equal(first_ws[1], np.zeros(3, np.float32))   # fly 1 was mid-battle: not in the commit

    ck = CheckpointStore(tmp_path / "snap", "h").load()
    assert ck["completed"] == ["f00-b000"]
    w = {i: np.asarray(f["w"]).copy() for i, f in enumerate(ck["pool_state"]["flies"])}
    res = await run_cohort(sched, CheckpointStore(tmp_path / "snap", "h"), tmp_path / "logs_r",
                           _interleaved_play(w), lambda: _state(w))
    assert res["skipped"] == ["f00-b000"] and len(res["played"]) == 3
    assert all(np.array_equal(w[k], want[k]) for k in want)


async def test_stop_after_counts_started_battles_across_concurrent_flies(tmp_path):
    """Ruling 3: stop_after=1 with two concurrent flies plays exactly one battle; resume plays the rest."""
    sched = make_schedule(2, 2, seed=0)
    w = _zeros()
    play = _interleaved_play(w)
    part1 = await run_cohort(sched, CheckpointStore(tmp_path / "ck", "h"), tmp_path / "logs", play,
                             lambda: _state(w), stop_after=1)
    assert len(part1["played"]) == 1 and part1["skipped"] == []
    ck = CheckpointStore(tmp_path / "ck", "h").load()
    w2 = {i: np.asarray(f["w"]).copy() for i, f in enumerate(ck["pool_state"]["flies"])}
    part2 = await run_cohort(sched, CheckpointStore(tmp_path / "ck", "h"), tmp_path / "logs",
                             _interleaved_play(w2), lambda: _state(w2))
    assert len(part2["skipped"]) == 1 and len(part2["played"]) == 3
    assert len(CheckpointStore(tmp_path / "ck", "h").load()["completed"]) == 4


async def test_committed_state_keeps_plain_types_and_filters_logs(tmp_path):
    sched = make_schedule(1, 2, seed=0)
    w = _zeros(1)
    logs = tmp_path / "logs"
    logs.mkdir()

    async def play(sb):
        with open(logs / "fly00.jsonl", "a") as fh:
            fh.write(json.dumps({"battle_id": sb.battle_id}) + "\n")
        w[0] = w[0] + np.float32(1)

    await run_cohort(sched, CheckpointStore(tmp_path / "ck", "h"), logs, play, lambda: _state(w), stop_after=1)
    # an uncommitted battle's record and a torn line, as after a crash mid-battle
    with open(logs / "fly00.jsonl", "a") as fh:
        fh.write(json.dumps({"battle_id": "f00-b001"}) + "\n" + '{"battle_id": "f00')
    ck = CheckpointStore(tmp_path / "ck", "h").load()
    assert type(ck["pool_state"]["flies"][0]["enabled"]) is bool
    w = {0: np.asarray(ck["pool_state"]["flies"][0]["w"]).copy()}
    await run_cohort(sched, CheckpointStore(tmp_path / "ck", "h"), logs, play, lambda: _state(w))
    ids = [json.loads(line)["battle_id"] for line in (logs / "fly00.jsonl").read_text().splitlines()]
    assert ids == ["f00-b000", "f00-b001"]


async def test_resume_builds_on_the_loaded_generation(tmp_path):
    """run_cohort loads before its first commit, so the new generation extends the completed list on disk."""
    sched = make_schedule(1, 3, seed=0)
    w = _zeros(1)

    async def play(sb):
        w[0] = w[0] + np.float32(1)

    await run_cohort(sched, CheckpointStore(tmp_path / "ck", "h"), tmp_path / "logs", play, lambda: _state(w),
                     stop_after=2)
    m = json.loads((tmp_path / "ck" / "manifest.json").read_text())
    # corrupt the newest generation: resume must fall back to the older one and build on it
    (tmp_path / "ck" / m["generations"][-1]["file"]).write_bytes(b"torn")
    ck = CheckpointStore(tmp_path / "ck", "h").load()
    assert ck["completed"] == ["f00-b000"]
    w = {0: np.asarray(ck["pool_state"]["flies"][0]["w"]).copy()}
    res = await run_cohort(sched, CheckpointStore(tmp_path / "ck", "h"), tmp_path / "logs", play, lambda: _state(w))
    assert res["played"] == ["f00-b001", "f00-b002"]
    assert CheckpointStore(tmp_path / "ck", "h").load()["completed"] == ["f00-b000", "f00-b001", "f00-b002"]
    assert np.array_equal(w[0], np.full(3, 3, np.float32))


@pytest.mark.parametrize("damage", ["all_checksums", "config_hash"])
async def test_unusable_checkpoint_is_fatal_not_a_fresh_start(tmp_path, damage):
    sched = make_schedule(1, 2, seed=0)
    w = _zeros(1)
    played = []

    async def play(sb):
        played.append(sb.battle_id)

    await run_cohort(sched, CheckpointStore(tmp_path / "ck", "h"), tmp_path / "logs", play, lambda: _state(w))
    h = "h"
    if damage == "all_checksums":
        for p in (tmp_path / "ck").glob("state_*.npz"):
            p.write_bytes(b"torn")
    else:
        h = "other"
    played.clear()
    with pytest.raises(ValueError):
        await run_cohort(sched, CheckpointStore(tmp_path / "ck", h), tmp_path / "logs", play, lambda: _state(w))
    assert played == []
