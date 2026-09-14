import asyncio

import pytest

from flymon.battle.barrier import BatchBarrier, Request


def _batcher(calls):
    async def run_batch(reqs):
        calls.append([r.player_id for r in reqs])
        return [0] * len(reqs)
    return run_batch


def _now() -> float:
    return asyncio.get_running_loop().time()


async def test_runs_immediately_when_all_active_players_wait():
    calls = []
    bar = BatchBarrier(_batcher(calls), deadline_ms=10_000)   # long deadline: must not be what triggers
    bar.register("a"); bar.register("b")
    t0 = _now()
    ra, rb = await asyncio.gather(bar.submit("a", None, ["x", "y"], {}), bar.submit("b", None, ["x", "y"], {}))
    assert (ra, rb) == (0, 0)
    assert calls in ([["a", "b"]], [["b", "a"]])
    assert _now() - t0 < 1.0


async def test_single_remaining_player_runs_immediately():
    seen: list[Request] = []

    async def run_batch(reqs):
        seen.extend(reqs)
        return [0] * len(reqs)

    bar = BatchBarrier(run_batch, deadline_ms=10_000)
    bar.register("a"); bar.register("b")
    bar.unregister("b")                                  # b's battle ended: a is now the only active player
    t0 = _now()
    assert await bar.submit("a", None, ["x", "y"], {"turn": 3}) == 0
    assert _now() - t0 < 1.0
    assert [(r.player_id, r.battle, r.candidates, r.context) for r in seen] == [("a", None, ["x", "y"], {"turn": 3})]


async def test_deadline_runs_batch_without_all_players():
    calls = []
    bar = BatchBarrier(_batcher(calls), deadline_ms=50)
    bar.register("a"); bar.register("b")                 # b stays active but never submits
    t0 = _now()
    assert await bar.submit("a", None, ["x", "y"], {}) == 0
    elapsed = _now() - t0
    assert 0.04 <= elapsed < 1.0
    assert calls == [["a"]]


async def test_request_arriving_during_running_batch_is_not_lost():
    calls = []

    async def run_batch(reqs):
        calls.append([r.player_id for r in reqs])
        await asyncio.sleep(0.05)                        # batch is slow: "b" arrives while it runs
        return [0] * len(reqs)

    bar = BatchBarrier(run_batch, deadline_ms=30)
    bar.register("a"); bar.register("b")
    ta = asyncio.create_task(bar.submit("a", None, ["x"], {}))
    await asyncio.sleep(0.04)                            # deadline fired, batch for "a" is in flight
    tb = asyncio.create_task(bar.submit("b", None, ["x"], {}))
    assert await ta == 0
    assert await tb == 0
    assert all(call for call in calls)                   # no empty batch
    assert sorted(pid for call in calls for pid in call) == ["a", "b"]   # each player served exactly once


async def test_batch_exception_propagates_to_all_waiters():
    async def boom(reqs):
        raise RuntimeError("swarm failed")
    bar = BatchBarrier(boom, deadline_ms=20)
    bar.register("a"); bar.register("b")
    with pytest.raises(RuntimeError):
        await asyncio.gather(bar.submit("a", None, ["x"], {}), bar.submit("b", None, ["x"], {}))


async def test_cancel_wakes_waiter():
    calls = []
    bar = BatchBarrier(_batcher(calls), deadline_ms=10_000)
    bar.register("a"); bar.register("b")
    task = asyncio.create_task(bar.submit("a", None, ["x"], {}))
    await asyncio.sleep(0.01)
    bar.cancel("a")
    with pytest.raises(asyncio.CancelledError):
        await task
    assert calls == []


async def test_all_players_terminating_cancels_pending_request():
    calls = []
    bar = BatchBarrier(_batcher(calls), deadline_ms=10_000)
    bar.register("a"); bar.register("b")
    task = asyncio.create_task(bar.submit("a", None, ["x"], {}))
    await asyncio.sleep(0.01)
    bar.unregister("a")                                  # a's battle ended while its request was pending
    bar.unregister("b")
    with pytest.raises(asyncio.CancelledError):
        await task
    assert calls == []                                   # never run with an empty batch
