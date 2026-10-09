"""OverlapBarrier: BatchBarrier whose flushes may overlap (speedup brief fix 3). battle/barrier.py holds its asyncio lock
across run_batch, so a batch never starts before the previous one (and its slowest job) is done. Here the lock only
guards taking the pending set; run_batch then runs outside it, so a later batch can go to the workers while an earlier
one is still in flight. Every request still gets exactly its own result: a player has one request at a time
(BatchBarrier.submit refuses a second), so a fly's decide / reinforce steps keep their order and each request's result
depends only on its fly's weights, odours and seed, never on which batch carried it."""
from __future__ import annotations

import asyncio

from ..battle.barrier import BatchBarrier


class OverlapBarrier(BatchBarrier):
    # copied from flymon/battle/barrier.py:BatchBarrier._flush (the lock released before run_batch)
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
            if len(results) != len(reqs):                           # the swarm broke its contract
                raise ValueError(f"run_batch returned {len(results)} results for {len(reqs)} requests")
            for r, idx in zip(reqs, results):
                if not r.future.done():      # a waiter cancelled mid-batch never gets a result
                    r.future.set_result(int(idx))
        except asyncio.CancelledError:       # the flush itself was cancelled: do not leave waiters hanging
            for r in reqs:
                if not r.future.done():
                    r.future.cancel()
            raise
        except Exception as e:               # run_batch *or* result dispatch: propagate to every waiter
            for r in reqs:
                if not r.future.done():
                    r.future.set_exception(e)
        finally:
            for r in reqs:
                if self._inflight.get(r.player_id) is r:
                    del self._inflight[r.player_id]
