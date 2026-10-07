"""Cohort loop (spec 3.6): each fly plays its own schedule sequentially; flies run concurrently; after every battle a
cohort checkpoint is committed (one generation per battle, serialized by a lock). Resume skips battle ids the
manifest lists and drops log records of uncommitted battles.

What a commit holds: per fly, the state as of that fly's last committed battle. The live pool state is not committed
whole, because other flies may be mid-battle and a resume would replay their half-played battle on top of its
reinforcements. So the runner keeps one committed entry per fly, taken from swarm_state() once at start (after the
caller's resume load) and, after fly F's battle, replaces only entry F.

stop_after reserves its budget when a battle starts, so exactly N battles are played even with concurrent flies.
A checkpoint that cannot be used (every generation fails its checksum, or a config-hash mismatch) raises ValueError
from store.load(); that is fatal, never a fresh start.
should_stop() is asked when a battle would start (before stop_after reserves); True ends that fly's loop."""
from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np

from .checkpoint import filter_log


def _entry(f: dict) -> dict:
    """A private copy of one fly's state, so later in-place weight updates cannot leak into a committed entry."""
    return dict(f, w=np.array(f["w"], np.float32, copy=True))


async def run_cohort(sched, store, logs_dir, play_one, swarm_state, stop_after: int | None = None, should_stop=None) -> dict:
    logs_dir = Path(logs_dir); logs_dir.mkdir(parents=True, exist_ok=True)
    ck = store.load()   # before any commit: commit() builds on the generation load() returned; ValueError propagates
    done = set(ck["completed"]) if ck else set()
    for p in sorted(logs_dir.glob("fly*.jsonl")):
        filter_log(p, done)
    committed = [_entry(f) for f in swarm_state()["flies"]]   # per-fly committed entries
    cursor = dict(ck["cursor"]) if ck else {}
    lock = asyncio.Lock()
    played, skipped = [], [sb.battle_id for sb in sched if sb.battle_id in done]
    budget = {"left": stop_after}

    async def fly_loop(fly, sbs):
        for sb in sbs:
            if sb.battle_id in done:
                continue
            async with lock:
                if should_stop is not None and should_stop():
                    return                                  # a session cap (AC.6): nothing started, nothing reserved
                if budget["left"] is not None:
                    if budget["left"] <= 0:
                        return
                    budget["left"] -= 1   # reserve at start
            await play_one(sb)
            async with lock:
                committed[fly] = _entry(swarm_state()["flies"][fly])
                cursor[str(fly)] = sb.battle_id
                store.commit(sb.battle_id, {"flies": list(committed)}, dict(cursor))
                played.append(sb.battle_id)

    by_fly: dict = {}
    for sb in sched:
        by_fly.setdefault(sb.fly_id, []).append(sb)
    await asyncio.gather(*(fly_loop(f, s) for f, s in by_fly.items()))
    return {"played": played, "skipped": skipped}
