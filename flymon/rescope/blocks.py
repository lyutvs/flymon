"""Learn / eval blocks for the M4 arms (spec 4.5, 10.7-10.8).

A brain arm plays its learning block (FLY: own pulses; RS: the donor's queue) and then its evaluation block with
plasticity off, argmax decisions and no pulse delivered (EvalPlayer queues none, and the eval reinforcement barrier
refuses any that would arrive). C-off and the no-brain arms (RND, MAX) play the evaluation block only.

Each block is one run_cohort over its own CheckpointStore (out/checkpoints/{learn,eval}) and log directory
(out/logs for learning - the RS donor log - and out/logs/eval), so a resume in one block never filters the other's
records. Every battle goes through play_with_retries: an attempt that ends unfinished (server error) is rolled back
- the fly's weights and yoked-queue state are restored to their pre-battle values, the player's undelivered pulses
are dropped and the attempt's log records move to <logs>/retries/ - and replayed with the same schedule entry, up to
retry_max times; beyond that the battle (and so the fly) is INVALID. One record per committed battle goes to
<logs>/battles.jsonl (filtered to the committed battles on resume, like the fly logs).

RS queue state: YokedQueue marks a bundle used at pop(), so its state is saved (out/yoke_state.json) only after the
battle's drain, keyed by battle id; a resume restores each fly's state after its last committed learning battle
(the checkpoint cursor), and refuses a state taken from another donor log (FLY k rerun -> RS k must be rerun)."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np

from ..agent.checkpoint import CheckpointStore, filter_log
from ..agent.player import AgentPlayer
from ..agent.policy import derive_seed
from ..agent.runner import run_cohort
from ..battle.fly_coach_player import FlyCoachPlayer
from ..battle.providers import RandomProvider
from ..battle.schedule import ScheduledBattle, make_schedule

PHASE_TAG = {"pilot": "P", "judge": "J"}
BLOCKS = ("PL", "PE", "JL", "JE")          # phase (P pilot / J judge) + block (L learn / E eval), plan R6 supplement
_ID = re.compile(r"^(PL|PE|JL|JE)-f(\d{2})-b(\d{3})$")


# ---- schedules ------------------------------------------------------------------------------
def block_tag(phase: str, block: str) -> str:
    """The battle-id prefix of a phase's block: pilot learn PL, pilot eval PE, judge learn JL, judge eval JE. The
    phase is in the id so derive_seed (keyed on the battle id) never repeats between the pilot and the judge run."""
    if phase not in PHASE_TAG or block not in ("L", "E"):
        raise ValueError(f"phase must be one of {tuple(PHASE_TAG)} and block L / E, got {phase!r} / {block!r}")
    return PHASE_TAG[phase] + block


def block_schedule(n_flies, n_battles, seed, tag) -> list:
    """make_schedule with battle ids <tag>-fNN-bNNN, tag one of BLOCKS (block_tag(phase, block))."""
    if tag not in BLOCKS:
        raise ValueError(f"block tag must be one of {BLOCKS}, got {tag!r}")
    return [dataclasses.replace(sb, battle_id=f"{tag}-{sb.battle_id}")
            for sb in make_schedule(n_flies, n_battles, "heuristic", seed=seed)]


def schedule_rows(sched) -> list:
    return [dict(battle_id=s.battle_id, fly_id=int(s.fly_id), my_team=[str(x) for x in s.my_team],
                 opp_team=[str(x) for x in s.opp_team], opponent=str(s.opponent)) for s in sched]


def schedule_digest(sched) -> str:
    return hashlib.sha256(json.dumps(schedule_rows(sched), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def schedule_from_rows(rows) -> list:
    return [ScheduledBattle(**d) for d in rows]


def assert_disjoint(learn, eval_) -> None:
    key = lambda s: (tuple(s.my_team), tuple(s.opp_team))
    both = {key(s) for s in learn} & {key(s) for s in eval_}
    if both:
        raise ValueError(f"{len(both)} learn/eval battles share teams and order")


def battle_index(battle_id: str) -> int:
    """The within-block battle number of a <tag>-fNN-bNNN id (tau's battle index); any other id is a ValueError."""
    m = _ID.match(str(battle_id))
    if m is None:
        raise ValueError(f"battle id {battle_id!r} is not <PL|PE|JL|JE>-fNN-bNNN")
    return int(m.group(3))


def ids_of(sched, fly=None) -> list:
    return [s.battle_id for s in sched if fly is None or s.fly_id == fly]


# ---- eval block ------------------------------------------------------------------------------
def enter_eval(swarm, pool) -> None:
    for f in range(pool.n_flies):
        pool.set_enabled(f, False)
    swarm.mode = "eval"


class EvalPlayer(AgentPlayer):
    """An AgentPlayer for the evaluation block: a fly-turn outcome queues no pulse and writes no reinforce record
    (spec 4.5: plasticity off in E). Decisions and battle summaries are AgentPlayer's."""

    def _on_outcome(self, tag: str, turn: int, outcome) -> None:
        self._choice.pop(tag, None)

    def _log(self, battle, who, decision, cands, chosen, detail=None) -> None:
        """A fly decision record also carries `multipliers`: each candidate's type multiplier against the opponent's
        active Pokemon (poke-env `damage_multiplier`, the coach's attack_score type factor and the spec 3.x
        effectiveness check), in candidate order - the input of the information-turn type match (spec 4.5, recorded
        only; logschema allows extra fields)."""
        if who == "fly":
            detail = dict(detail or {}, multipliers=type_multipliers(battle, cands))
        super()._log(battle, who, decision, cands, chosen, detail)


def type_multipliers(battle, cands) -> list | None:
    opp = battle.opponent_active_pokemon
    if opp is None:
        return None
    return [float(opp.damage_multiplier(m)) for m in cands]


async def refuse_pulses(reqs) -> list:
    """The eval block's reinforcement barrier: any pulse reaching it is a leak of learning into evaluation."""
    raise RuntimeError(f"eval block: {len(reqs)} reinforcement request(s) reached the swarm (no pulse in E)")


class NoBrainPlayer(FlyCoachPlayer):
    """FlyCoachPlayer whose log records carry fly and battle_id (the checkpoint key run_cohort's filter_log and the
    retry rollback use); decisions are its provider's. A RandomProvider is reseeded at every battle start from
    derive_seed("rnd", phase, fly, battle_id), so a resumed or retried RND fly draws exactly what an uninterrupted one
    would (no generator state carried between battles)."""

    def __init__(self, fly: int, phase: str = None, **kw):
        super().__init__(**kw)
        self.fly, self.phase, self.battle_id, self.battle_index = int(fly), phase, None, 0

    def start_battle(self, battle_id: str, battle_index: int) -> None:
        self.battle_id, self.battle_index = battle_id, int(battle_index)
        if isinstance(self.provider, RandomProvider):
            if self.phase is None:
                raise ValueError("a RandomProvider NoBrainPlayer needs its phase (the per-battle seed)")
            self.provider.rng = np.random.default_rng(rnd_seed(self.phase, self.fly, battle_id))

    def _write(self, rec: dict) -> None:
        super()._write({"fly": self.fly, "battle_id": self.battle_id, **rec})


def rnd_seed(phase: str, fly: int, battle_id: str) -> int:
    return derive_seed("rnd", phase, int(fly), battle_id)


# ---- per-battle bookkeeping ------------------------------------------------------------------
def weights_sha(w) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(w, np.float32)).tobytes()).hexdigest()


def file_sha(path) -> str | None:
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def _atomic_text(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=p.name + ".")
    with os.fdopen(fd, "w") as fh:
        fh.write(text)
    os.replace(tmp, p)


def read_jsonl(path) -> list:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def append_jsonl(path, rec: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


def divert_attempt_log(log_path, battle_id: str, dest, attempt: int) -> int:
    """Move the records of a failed attempt (this battle id) out of the fly log into dest, tagged with the attempt
    number; returns how many moved. The fly log keeps only committed battles' and the current attempt's records."""
    p = Path(log_path)
    if not p.exists():
        return 0
    keep, moved = [], []
    for line in p.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue                                   # a torn line (the attempt was cut mid-write)
        if rec.get("battle_id") == battle_id:
            moved.append(dict(rec, attempt=int(attempt)))
        else:
            keep.append(line)
    _atomic_text(p, "".join(x + "\n" for x in keep))
    for rec in moved:
        append_jsonl(dest, rec)
    return len(moved)


async def play_with_retries(play, sb, retry_max, on_unfinished=None) -> dict:
    """play(sb) -> dict with "finished". An unfinished attempt calls on_unfinished(sb, attempt, res) (the rollback)
    and is replayed with the same entry, up to retry_max times; beyond that the result is invalid."""
    res = {}
    for attempt in range(retry_max + 1):
        res = await play(sb)
        if res.get("finished", False):
            return dict(res, retries=attempt, invalid=False)
        if on_unfinished is not None:
            await on_unfinished(sb, attempt, res)
    return dict(res, retries=retry_max, invalid=True)


def fly_result(fly, eval_records, queue=None, spec=None, invalid=False) -> dict:
    out = dict(fly=fly, eval_battles=eval_records, invalid=bool(invalid))
    if queue is not None:
        out.update(residual_frac=queue.residual_frac(), donor_sha256=queue.donor_sha256)
        out["invalid"] = out["invalid"] or queue.residual_frac() > spec.residual_max
    return out


# ---- RS queue state --------------------------------------------------------------------------
class YokeBook:
    """Per-fly yoked queues and each queue's state after every committed learning battle (out/yoke_state.json).
    record() runs after the battle's drain and before run_cohort commits it; load() restores the state after the
    fly's last committed battle and refuses a state from another donor log."""

    def __init__(self, path, queues: dict):
        self.path, self.queues = Path(path), dict(queues)
        self.hist = {f: {} for f in self.queues}

    def load(self, cursor: dict) -> None:
        d = json.loads(self.path.read_text()) if self.path.exists() else {"flies": {}}
        for f, q in self.queues.items():
            ent = d["flies"].get(str(f))
            last = cursor.get(str(f))
            if ent is None:
                if last is not None:
                    raise SystemExit(f"RS fly {f}: checkpoint at {last} but no yoked-queue state in {self.path}")
                continue
            if ent["donor_sha256"] != q.donor_sha256:
                if last is None:
                    continue                           # nothing committed yet: a fresh start overwrites the file
                raise SystemExit(f"RS fly {f}: donor log changed since this RS fly started "
                                 f"({ent['donor_sha256'][:12]} -> {q.donor_sha256[:12]}); FLY {f} was rerun, "
                                 f"so RS {f} must be rerun from scratch")
            self.hist[f] = dict(ent["after"])
            if last is not None:
                if last not in self.hist[f]:
                    raise SystemExit(f"RS fly {f}: no yoked-queue state after committed battle {last}")
                try:
                    q.load_state(self.hist[f][last])
                except ValueError as e:
                    raise SystemExit(f"RS fly {f}: {e}")

    def record(self, fly: int, battle_id: str) -> None:
        self.hist[fly][battle_id] = self.queues[fly].state()
        self.save()

    def save(self) -> None:
        d = {"flies": {str(f): {"donor_sha256": q.donor_sha256, "after": self.hist[f]} for f, q in self.queues.items()}}
        _atomic_text(self.path, json.dumps(d, sort_keys=True))


# ---- the block and the arm -------------------------------------------------------------------
def _complete(ck, sched) -> bool:
    return ck is not None and set(ids_of(sched)) <= set(ck["completed"])


async def run_block(sched, store, logs_dir, attempt, swarm_state, retry_max, *, block, snapshot=None, rollback=None,
                    after=None, stop_after=None, should_stop=None) -> dict:
    """run_cohort over one block. attempt(sb, n) plays attempt n (0 = first) of a battle and returns a dict with
    finished / won / ...; snapshot(sb) is taken before the first attempt, rollback(sb, snap, n, res) undoes an
    unfinished attempt, after(sb, res) runs once the battle is settled (before the commit)."""
    logs_dir = Path(logs_dir)
    ck = store.load()
    done = set(ck["completed"]) if ck else set()
    filter_log(logs_dir / "battles.jsonl", done)
    for p in sorted((logs_dir / "retries").glob("fly*.jsonl")):
        filter_log(p, done)

    async def play_one(sb):
        snap = snapshot(sb) if snapshot else None
        n = {"i": 0}

        async def play(s):
            r = await attempt(s, n["i"])
            n["i"] += 1
            return r

        async def undo(s, i, res):
            divert_attempt_log(logs_dir / f"fly{s.fly_id:02d}.jsonl", s.battle_id,
                               logs_dir / "retries" / f"fly{s.fly_id:02d}.jsonl", i)
            if rollback:
                rollback(s, snap, i, res)

        res = await play_with_retries(play, sb, retry_max, on_unfinished=undo)
        rec = dict(battle_id=sb.battle_id, fly=int(sb.fly_id), block=block, won=res.get("won"),
                   finished=bool(res.get("finished", False)), retries=int(res["retries"]), invalid=bool(res["invalid"]),
                   **{k: res.get(k) for k in ("battle_tag", "turns", "fly_turns", "coach_turns") if k in res})
        if after:
            after(sb, rec)
        append_jsonl(logs_dir / "battles.jsonl", rec)

    return await run_cohort(sched, store, logs_dir, play_one, swarm_state, stop_after=stop_after,
                            should_stop=should_stop)


def _nobrain_state(n_flies):
    return lambda: {"flies": [dict(enabled=False, shuffle_seed=None, w=np.zeros(0, np.float32)) for _ in range(n_flies)]}


async def run_arm(out, *, eval_, attempt_for, cfg_hash: str, retry_max: int, resume: bool, n_flies: int,
                  learn=None, pool=None, swarm=None, yoke: YokeBook | None = None, reset_player=None,
                  stop_after=None, on_block_end=None, after_battle=None, should_stop=None) -> dict:
    """Learning block (if `learn`), then enter_eval and the evaluation block. attempt_for(block) -> attempt(sb, n);
    reset_player(block, fly) drops a player's undelivered pulses after an unfinished attempt; on_block_end(block) is
    awaited after each block's cohort (the players log out). Returns
    {complete, played, before, after}: per-fly weight sha256 entering and leaving the eval block (brain arms).
    after_battle(block, sb, rec) runs once per settled battle, before its commit; should_stop() is asked at every battle
    start and before the evaluation block (a session cap)."""
    out = Path(out)
    stores = {"E": CheckpointStore(out / "checkpoints" / "eval", cfg_hash)}
    if learn:
        stores["L"] = CheckpointStore(out / "checkpoints" / "learn", cfg_hash)
    cks = {b: s.load() for b, s in stores.items()}
    if not resume and any(cks.values()):
        raise SystemExit(f"{out} already has checkpoints; pass --resume")
    state_fn = pool.state if pool is not None else _nobrain_state(n_flies)
    budget = {"left": stop_after}
    played = {}

    def snapshot_for(block):
        def snapshot(sb):
            return dict(w=None if pool is None else np.array(pool.w[sb.fly_id], copy=True),
                        yoke=yoke.queues[sb.fly_id].state() if (yoke and block == "L") else None)
        return snapshot

    def rollback_for(block):
        def rollback(sb, snap, i, res):
            if reset_player:
                reset_player(block, sb.fly_id)
            if snap["w"] is not None:
                pool.w[sb.fly_id] = snap["w"].copy()
            if snap["yoke"] is not None:
                yoke.queues[sb.fly_id].load_state(snap["yoke"])
        return rollback

    def after_for(block):
        def after(sb, rec):
            if yoke and block == "L":
                yoke.record(sb.fly_id, sb.battle_id)
            if after_battle is not None:
                after_battle(block, sb, rec)                # AC: fly-level INVALID, situation-pair evaluation
        return after

    async def block_run(block, sched, logs_dir):
        r = await run_block(sched, stores[block], logs_dir, attempt_for(block), state_fn, retry_max, block=block,
                            snapshot=snapshot_for(block), rollback=rollback_for(block), after=after_for(block),
                            stop_after=budget["left"], should_stop=should_stop)
        played[block] = r["played"]
        if on_block_end is not None:
            await on_block_end(block)
        if budget["left"] is not None:
            budget["left"] -= len(r["played"])
        return stores[block].load()

    before = {}
    if learn:
        ck = cks["L"]
        if ck:
            pool.load_state(ck["pool_state"])
        if yoke:
            yoke.load(ck["cursor"] if ck else {})
        if not _complete(ck, learn):
            if cks["E"]:
                raise SystemExit(f"{out}: an eval checkpoint exists but the learning block is incomplete")
            ck = await block_run("L", learn, out / "logs")
            if not _complete(ck, learn):
                return dict(complete=False, played=played, before={}, after={})
        pool.load_state(ck["pool_state"])              # the committed end of learning (== the live pool)
        before = {f: weights_sha(e["w"]) for f, e in enumerate(ck["pool_state"]["flies"])}
    elif pool is not None:
        before = {f: weights_sha(e["w"]) for f, e in enumerate(pool.state()["flies"])}   # C-off: naive weights

    if pool is not None:
        if cks["E"]:
            pool.load_state(cks["E"]["pool_state"])
        enter_eval(swarm, pool)
    if should_stop is not None and should_stop() and not _complete(cks["E"], eval_):
        return dict(complete=False, played=played, before=before, after={})
    if budget["left"] is not None and budget["left"] <= 0 and not _complete(cks["E"], eval_):
        return dict(complete=False, played=played, before=before, after={})
    eck = await block_run("E", eval_, out / "logs" / "eval")
    after = {} if pool is None else {f: weights_sha(pool.w[f]) for f in range(pool.n_flies)}
    return dict(complete=_complete(eck, eval_), played=played, before=before, after=after)


def arm_per_fly(out, *, n_flies: int, eval_, run: dict, learn=None, yoke: YokeBook | None = None, spec=None,
                brain: bool = True, extra: dict | None = None, weight_medians: dict | None = None) -> list:
    """per_fly entries of result.json from the committed battle records (both blocks) and the run's weight hashes.
    Learning arms also get learn_records (per battle, schedule order: the learning-block win curve); weight_medians
    (fly -> plastic weight median / w0 at the arm's end, brain arms) goes to final_weight_median_ratio."""
    out = Path(out)
    erecs = {r["battle_id"]: r for r in read_jsonl(out / "logs" / "eval" / "battles.jsonl")}
    lrecs = {r["battle_id"]: r for r in read_jsonl(out / "logs" / "battles.jsonl")} if learn else {}
    rows = []
    for f in range(n_flies):
        ev = [erecs[b] for b in ids_of(eval_, f)]
        le = [lrecs[b] for b in ids_of(learn, f)] if learn else []
        eval_records = [dict(battle_id=r["battle_id"], won=r["won"], finished=r["finished"], retries=r["retries"])
                        for r in ev]
        bad = any(r["invalid"] for r in ev + le)
        row = fly_result(f, eval_records, queue=yoke.queues[f] if yoke else None, spec=spec, invalid=bad)
        row["battle_invalid"] = bad
        row["weights_bit_identical_across_eval"] = (run["before"].get(f) == run["after"].get(f)) if brain else None
        if brain and not row["weights_bit_identical_across_eval"]:
            row["invalid"] = True
        row["eval_retries"] = sum(r["retries"] for r in ev)
        if learn:
            row.update(learn_battles=len(le), learn_wins=sum(r["won"] is True for r in le),
                       learn_retries=sum(r["retries"] for r in le),
                       learn_log_sha256=file_sha(out / "logs" / f"fly{f:02d}.jsonl"),
                       learn_records=[dict(battle_id=r["battle_id"], won=r["won"], finished=r["finished"],
                                           retries=r["retries"]) for r in le])
        if weight_medians is not None:
            row["final_weight_median_ratio"] = weight_medians.get(f)
        if yoke:
            row["yoke"] = yoke.queues[f].summary(spec.residual_max)
        if extra and f in extra:
            row.update(extra[f])
        rows.append(row)
    return rows
