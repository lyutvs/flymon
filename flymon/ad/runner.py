"""run_ad_arm: rescope's run_arm for AD (spec AD.2, AD.4). The learning block runs in two phases split at the futility
point: phase 1 = learning battles with index < 40 (every fly), then decide_futility() once per session (it persists
the decision, Task 8), then phase 2 = the rest - or nothing under STOP_FUTILE - and the evaluation block after the
effective learning end. Phase boundaries are a barrier: a fly ahead waits for the slowest before battle 40, so the
check reads every fly at 40 and no fly learns past 40 before the decision. should_stop_battle(block, sb) is asked at
every battle start (AD.4's per-batch reservation). A checkpoint written under another key (the run manifest's sha256 is
part of it) is STOP_MANIFEST. Without an evaluation schedule (the bench) the run is complete when learning is."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..agent.checkpoint import CheckpointStore
from ..rescope import blocks
from .inputs import battle_index


def _complete(ck, sched) -> bool:
    return ck is not None and set(blocks.ids_of(sched)) <= set(ck["completed"])


def _load(store):
    try:
        return store.load()
    except ValueError as e:
        raise SystemExit(f"STOP_MANIFEST: {store.root}: {e} (the checkpoint key holds the run manifest's sha256, AD.4)")


# copied from flymon/rescope/blocks.py:run_arm (+ two learning phases split at the futility point, the per-battle start
# check, STOP_MANIFEST on a foreign checkpoint key, an eval-less bench; no yoked queues)
async def run_ad_arm(out, *, learn, eval_, attempt_for, cfg_hash: str, retry_max: int, resume: bool, pool, swarm,
                     reset_player=None, stop_after=None, on_block_end=None, after_battle=None, should_stop=None,
                     should_stop_battle=None, futility_point=None, decide_futility=None) -> dict:
    out = Path(out)
    stores = {"L": CheckpointStore(out / "checkpoints" / "learn", cfg_hash),
              "E": CheckpointStore(out / "checkpoints" / "eval", cfg_hash)}
    cks = {b: _load(s) for b, s in stores.items()}
    if not resume and any(cks.values()):
        raise SystemExit(f"{out} already has checkpoints; pass --resume")
    budget = {"left": stop_after}
    played: dict = {"L": [], "E": []}
    state = {"fut": None, "learn_ids": None}

    def snapshot(sb):
        return dict(w=np.array(pool.w[sb.fly_id], copy=True))

    def rollback_for(block):
        def rollback(sb, snap, i, res):
            if reset_player:
                reset_player(block, sb.fly_id)              # AD: retires the attempt token before the restore
            pool.w[sb.fly_id] = snap["w"].copy()
        return rollback

    def after_for(block):
        def after(sb, rec):
            if after_battle is not None:
                after_battle(block, sb, rec)
        return after

    async def block_run(block, sched, logs_dir):
        gate = None if should_stop_battle is None else (lambda sb: should_stop_battle(block, sb))
        r = await blocks.run_block(sched, stores[block], logs_dir, attempt_for(block), pool.state, retry_max,
                                   block=block, snapshot=snapshot, rollback=rollback_for(block),
                                   after=after_for(block), stop_after=budget["left"], should_stop=should_stop,
                                   should_stop_battle=gate)
        played[block] += r["played"]
        if on_block_end is not None:
            await on_block_end(block)
        if budget["left"] is not None:
            budget["left"] -= len(r["played"])
        return _load(stores[block])

    def partial(before=None) -> dict:
        return dict(complete=False, played=played, before=before or {}, after={}, futility=state["fut"],
                    learn_ids=state["learn_ids"])

    ck = cks["L"]
    if ck:
        pool.load_state(ck["pool_state"])
    phase1 = learn if futility_point is None else [sb for sb in learn
                                                   if battle_index(sb.battle_id) < int(futility_point)]
    eff = phase1
    if not _complete(ck, phase1):
        if cks["E"]:
            raise SystemExit(f"{out}: an eval checkpoint exists but the learning block is incomplete")
        ck = await block_run("L", phase1, out / "logs")
        if not _complete(ck, phase1):
            return partial()
    if len(phase1) < len(learn):
        state["fut"] = decide_futility() if decide_futility is not None else {"status": "CONTINUE"}
        if state["fut"]["status"] != "STOP_FUTILE":
            eff = learn
            if not _complete(ck, eff):
                if cks["E"]:
                    raise SystemExit(f"{out}: an eval checkpoint exists but the learning block is incomplete")
                ck = await block_run("L", eff, out / "logs")
                if not _complete(ck, eff):
                    return partial()
    state["learn_ids"] = [sb.battle_id for sb in eff]
    pool.load_state(ck["pool_state"])                    # the committed end of learning (== the live pool)
    before = {f: blocks.weights_sha(e["w"]) for f, e in enumerate(ck["pool_state"]["flies"])}
    if not eval_:
        return dict(complete=True, played=played, before=before, after=dict(before), futility=state["fut"],
                    learn_ids=state["learn_ids"])
    if cks["E"]:
        pool.load_state(cks["E"]["pool_state"])
    blocks.enter_eval(swarm, pool)
    if not _complete(cks["E"], eval_) and ((should_stop is not None and should_stop())
                                          or (budget["left"] is not None and budget["left"] <= 0)):
        return partial(before)
    eck = await block_run("E", eval_, out / "logs" / "eval")
    after = {f: blocks.weights_sha(pool.w[f]) for f in range(pool.n_flies)}
    return dict(complete=_complete(eck, eval_), played=played, before=before, after=after, futility=state["fut"],
                learn_ids=state["learn_ids"])
