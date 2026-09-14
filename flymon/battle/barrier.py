"""Collect the fly's decisions across concurrent battles and run them as one batch (the swarm in M3)."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


@dataclass
class Request:
    player_id: str
    battle: Any
    candidates: list
    context: dict
    future: asyncio.Future = field(default_factory=lambda: asyncio.get_running_loop().create_future())


class BatchBarrier:
    """One batch runs when the deadline since the first pending request expires, or when every
    active player is waiting -- whichever happens first."""

    def __init__(self, run_batch: Callable[[list], Awaitable[list]], deadline_ms: float = 100.0):
        self.run_batch, self.deadline = run_batch, deadline_ms / 1000.0
        self.active: set = set()
        self.pending: dict = {}
        self._inflight: dict = {}                                  # requests handed to a running run_batch
        self._timer: asyncio.Task | None = None
        self._tasks: set = set()                                   # strong refs: the loop only weakly holds tasks
        self._lock = asyncio.Lock()

    def register(self, player_id: str) -> None:
        self.active.add(player_id)

    def unregister(self, player_id: str) -> None:
        """The player left (battle over, exception): drop it from the active set and wake its request."""
        self.active.discard(player_id)
        self.cancel(player_id)
        self._maybe_flush_soon()

    def cancel(self, player_id: str) -> None:
        """Wake the player's request with CancelledError, whether it is queued or already in a running batch."""
        req = self.pending.pop(player_id, None) or self._inflight.get(player_id)
        if req is not None and not req.future.done():
            req.future.cancel()
        self._stop_timer_if_idle()

    async def submit(self, player_id: str, battle, candidates: list, context: dict) -> int:
        """One request at a time per player: a fly plays its battles sequentially, so a second
        concurrent submit is a bug and must not silently orphan the first one's waiter."""
        if player_id in self.pending or player_id in self._inflight:
            raise RuntimeError(f"{player_id} already has a pending request")
        req = Request(player_id, battle, candidates, context)
        self.pending[player_id] = req
        if self._timer is None:                                    # deadline runs from the first pending request
            self._timer = asyncio.create_task(self._deadline())
        self._maybe_flush_soon()
        return await req.future

    def _maybe_flush_soon(self) -> None:
        if self.pending and self.active <= set(self.pending):      # every active player is waiting
            task = asyncio.create_task(self._flush())
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    def _stop_timer_if_idle(self) -> None:
        """No pending request left: drop the timer so it neither fires on an empty set nor outlives the loop."""
        if not self.pending and self._timer is not None:
            if asyncio.current_task() is not self._timer:
                self._timer.cancel()
            self._timer = None

    async def _deadline(self) -> None:
        await asyncio.sleep(self.deadline)
        await self._flush()

    async def _flush(self) -> None:
        async with self._lock:
            reqs = [r for r in self.pending.values() if not r.future.done()]
            self.pending.clear()
            self._stop_timer_if_idle()       # requests that arrive from here on start a fresh deadline
            if not reqs:
                return
            self._inflight.update({r.player_id: r for r in reqs})
            try:
                results = await self.run_batch(reqs)
                if len(results) != len(reqs):                       # the swarm broke its contract
                    raise ValueError(f"run_batch returned {len(results)} results for {len(reqs)} requests")
                for r, idx in zip(reqs, results):
                    if not r.future.done():  # a waiter cancelled mid-batch never gets a result
                        r.future.set_result(int(idx))
            except asyncio.CancelledError:   # the flush itself was cancelled: do not leave waiters hanging
                for r in reqs:
                    if not r.future.done():
                        r.future.cancel()
                raise
            except Exception as e:           # run_batch *or* result dispatch: propagate to every waiter
                for r in reqs:
                    if not r.future.done():
                        r.future.set_exception(e)
            finally:
                for r in reqs:
                    if self._inflight.get(r.player_id) is r:
                        del self._inflight[r.player_id]
