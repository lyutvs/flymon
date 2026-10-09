"""run_ad_arm (AD.2, AD.4) on the fake pool: learning in two phases split at the futility point, one decision per
session, STOP_FUTILE going straight to the evaluation block, resume equivalence across the split, the per-battle start
check, a changed checkpoint key refused as STOP_MANIFEST, and the bench's eval-less completion."""
import asyncio
from types import SimpleNamespace

import pytest

from flymon.ad import inputs
from flymon.ad.runner import run_ad_arm
from flymon.ad.spec import smoke
from flymon.rescope import blocks
from tests.ac.ac_fakes import FakePool, fake_attempt_for, snapshot

S = smoke()
DOC = {"learn": blocks.schedule_rows(inputs.learn_canonical(S)), "eval": blocks.schedule_rows(inputs.eval_canonical(S))}
LEARN, EVAL = inputs.schedules_for(S, DOC, S.learn_battles)
N = S.n_flies()


def go(out, *, status="CONTINUE", calls=None, key="h" * 64, resume=False, stop_after=None, gate=None, eval_=EVAL,
       pool=None):
    calls = calls if calls is not None else []

    def decide():
        calls.append(1)
        return {"status": status}
    pool = pool or FakePool(N)
    run = asyncio.run(run_ad_arm(out, learn=LEARN, eval_=eval_, attempt_for=fake_attempt_for(out, pool), cfg_hash=key,
                                 retry_max=3, resume=resume, pool=pool, swarm=SimpleNamespace(mode="learn"),
                                 reset_player=lambda b, f: None, stop_after=stop_after, should_stop_battle=gate,
                                 futility_point=S.futility_point, decide_futility=decide))
    return run, calls, pool


def learned(out):
    return sorted(r["battle_id"] for r in blocks.read_jsonl(out / "logs" / "battles.jsonl"))


def test_continue_plays_every_learning_battle_then_eval(tmp_path):
    run, calls, _ = go(tmp_path)
    assert run["complete"] and calls == [1] and run["futility"] == {"status": "CONTINUE"}
    assert learned(tmp_path) == sorted(sb.battle_id for sb in LEARN) and len(run["played"]["E"]) == len(EVAL)


def test_stop_futile_skips_the_rest_and_runs_eval(tmp_path):
    run, calls, _ = go(tmp_path, status="STOP_FUTILE")
    assert run["complete"] and calls == [1]
    assert learned(tmp_path) == sorted(sb.battle_id for sb in LEARN if inputs.battle_index(sb.battle_id) < 1)
    assert run["learn_ids"] == [sb.battle_id for sb in LEARN if sb.battle_id.endswith("-b000")]
    assert len(run["played"]["E"]) == len(EVAL)


def test_resume_across_the_split_equals_one_session(tmp_path):
    full = tmp_path / "full"
    go(full)
    part = tmp_path / "part"
    calls = []
    run, _, pool = go(part, stop_after=6, calls=calls)              # phase 1 (4) + 2 of phase 2
    assert not run["complete"] and calls == [1]
    run, _, _ = go(part, resume=True, calls=calls, pool=pool)
    assert run["complete"] and calls == [1, 1]
    assert snapshot(full) == snapshot(part)


def test_per_battle_gate_stops_at_a_batch_boundary(tmp_path):
    seen = []

    def gate(block, sb):
        seen.append((block, inputs.battle_index(sb.battle_id)))
        return block == "L" and inputs.battle_index(sb.battle_id) >= 1
    run, calls, _ = go(tmp_path, gate=gate)
    assert not run["complete"] and calls == [1]
    assert learned(tmp_path) == sorted(sb.battle_id for sb in LEARN if sb.battle_id.endswith("-b000"))
    assert ("L", 1) in seen and run["played"]["E"] == []


def test_changed_checkpoint_key_is_stop_manifest(tmp_path):
    go(tmp_path, stop_after=3)
    with pytest.raises(SystemExit, match="STOP_MANIFEST"):
        go(tmp_path, resume=True, key="g" * 64)


def test_refuses_existing_checkpoints_without_resume(tmp_path):
    go(tmp_path, stop_after=3)
    with pytest.raises(SystemExit, match="--resume"):
        go(tmp_path)


def test_bench_without_eval_block_completes(tmp_path):
    run, _, _ = go(tmp_path, eval_=[])
    assert run["complete"] and run["played"]["E"] == [] and run["after"] == run["before"]
