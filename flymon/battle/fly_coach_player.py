"""poke-env player: coach decides strategy, a DecisionProvider picks among attack candidates."""
from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from poke_env.battle import AbstractBattle
from poke_env.player import Player

from ..live.events import make_event
from .attribution import TurnAttributor
from .barrier import BatchBarrier
from .coach import Coach
from .providers import DecisionProvider
from .router import candidates, route


class FlyCoachPlayer(Player):
    """Every turn: the coach decides; when it attacks with >= 2 candidates, the provider picks one.

    Raw protocol lines are mirrored into a per-battle `TurnAttributor`, so each turn's `Outcome`
    (direct damage, effectiveness, KO by my move) is available as a reinforcement signal.

    The JSONL log interleaves two record kinds, both carrying `battle_tag` and `turn`:
    `"decision"` (what I chose that turn) and `"outcome"` (what the move I used that turn did).
    They are written at different moments -- a turn's outcome is only known once the turn closes --
    so they are separate records and are joined on `(battle_tag, turn)`.

    A forced switch (after a faint) makes the server ask again within the same `turn`, so that turn
    carries a second `"decision"` record with the same `(battle_tag, turn)`: the join between
    decision and outcome records is many-to-one, not one-to-one.

    With an `event_sink` (flymon.live), every JSONL record is also emitted as a viewer event with the same
    content, bracketed by `battle_start`/`battle_end`; a provider that fills `context["detail"]` has it
    carried on the decision event. `turn_delay_s` holds each order back so a person can watch the battle.
    Neither changes what is decided or logged; a sink that raises is counted in `emit_errors`.
    """

    def __init__(self, provider: DecisionProvider, coach: Coach, barrier: BatchBarrier | None = None,
                 log_path: Path | None = None, event_sink=None, turn_delay_s: float = 0.0, **player_kwargs):
        super().__init__(**player_kwargs)
        self.provider, self.coach, self.barrier = provider, coach, barrier
        self.event_sink, self.turn_delay_s = event_sink, float(turn_delay_s)
        self.emit_errors = 0
        self._started: set[str] = set()
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.attributors: dict[str, TurnAttributor] = {}
        self.outcomes: dict[str, list] = {}
        self.turn_log: dict[str, list[dict]] = {}
        self.turns: dict[str, int] = {}                 # last `|turn|N` seen, per battle
        self.player_id = self.username
        if barrier is not None:
            barrier.register(self.player_id)

    # ---- raw protocol -> attribution ----------------------------------------------------
    async def _handle_battle_message(self, split_messages):
        tag = split_messages[0][0].lstrip(">")
        if not split_messages[0][0].startswith(">game"):  # best-of rooms carry no battle protocol
            if tag not in self._started:
                self._started.add(tag)
                self._emit("battle_start", tag, 0)
            att = self.attributors.setdefault(tag, TurnAttributor())
            self._emit_protocol(tag, split_messages[1:])
            battle = self._battles.get(tag)
            if battle is not None and battle.player_role:
                att.my_side = battle.player_role
            for line in split_messages[1:]:
                event = line[1] if len(line) > 1 else ""
                if event in ("turn", "upkeep", "win", "tie"):
                    self._close_turn(tag)   # a block closed at `|turn|N+1` still belongs to turn N
                att.feed(line)
                if event == "turn" and len(line) > 2:
                    self.turns[tag] = int(line[2])
        # a `|request|` line in this message makes super() call choose_move synchronously
        await super()._handle_battle_message(split_messages)

    def _emit_protocol(self, battle_tag: str, lines: list) -> None:
        """Mirror the raw protocol to the viewer, which replays it in the official battle renderer.
        `|request|` carries my own team sheet and the renderer ignores it, so it stays out."""
        if self.event_sink is None:
            return
        out = ["|".join(line) for line in lines if len(line) >= 2 and line[1] != "request"]
        if out:
            self._emit("protocol", battle_tag, self.turns.get(battle_tag, 0), lines=out)

    def _close_turn(self, battle_tag: str) -> None:
        """End the turn's attribution block, if I acted, and log its `Outcome` against that turn."""
        att = self.attributors.get(battle_tag)
        if att is None or not att.has_block():
            return
        outcome = att.end_turn()
        self.outcomes.setdefault(battle_tag, []).append(outcome)
        rec = {"battle_tag": battle_tag, "turn": self.turns.get(battle_tag, 0), "kind": "outcome",
               "outcome": asdict(outcome)}
        self._write(rec)
        self._emit("outcome", battle_tag, rec["turn"], outcome=rec["outcome"])

    # ---- decision -------------------------------------------------------------------------
    async def choose_move(self, battle: AbstractBattle):
        decision = self.coach.decide(battle)
        who, cands = route(decision, candidates(battle))
        chosen, detail = None, None
        if who == "fly":
            ctx = {"battle_tag": battle.battle_tag, "turn": battle.turn}
            if self.barrier is not None:
                self.barrier.register(self.player_id)  # a finished battle unregistered us
                idx = await self.barrier.submit(self.player_id, battle, cands, ctx)
            else:
                idx = await self.provider.decide(battle, cands, ctx)
            if not 0 <= idx < len(cands):
                raise ValueError(f"provider returned index {idx} for {len(cands)} candidates")
            chosen = cands[idx]
            detail = ctx.get("detail")
            order = self.create_order(chosen)
        else:
            order = decision.order
        self._log(battle, who, decision, cands, chosen, detail)
        if self.turn_delay_s > 0:
            await asyncio.sleep(self.turn_delay_s)
        return order

    def _log(self, battle, who, decision, cands, chosen, detail=None) -> None:
        rec = {"battle_tag": battle.battle_tag, "turn": battle.turn, "kind": "decision", "decider": who,
               "coach_kind": decision.kind, "candidates": [m.id for m in cands],
               "chosen": chosen.id if chosen else (decision.move.id if decision.move else None)}
        self.turn_log.setdefault(battle.battle_tag, []).append(rec)
        self._write(rec)
        fields = {k: rec[k] for k in ("decider", "coach_kind", "candidates", "chosen")}
        if detail is not None:
            fields["detail"] = detail
        self._emit("decision", rec["battle_tag"], rec["turn"], **fields)

    def _write(self, rec: dict) -> None:
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    def _emit(self, etype: str, battle_tag: str, turn: int, **fields) -> None:
        if self.event_sink is None:
            return
        try:
            self.event_sink.emit(make_event(etype, self.username, battle_tag, turn, **fields))
        except Exception:
            self.emit_errors += 1

    # ---- stats ------------------------------------------------------------------------------
    def battle_stats(self, battle_tag: str) -> dict:
        recs = self.turn_log.get(battle_tag, [])
        b = self._battles.get(battle_tag)
        return {"fly_turns": sum(r["decider"] == "fly" for r in recs),
                "coach_turns": sum(r["decider"] == "coach" for r in recs),
                "won": bool(b.won) if b is not None and b.won is not None else None,
                "finished": bool(b.finished) if b is not None else False}

    def n_candidates_mean(self, battle_tag: str) -> float:
        """Mean number of attack candidates over the turns the fly decided (0.0 if it never did)."""
        counts = [len(r["candidates"]) for r in self.turn_log.get(battle_tag, []) if r["decider"] == "fly"]
        return sum(counts) / len(counts) if counts else 0.0

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        self._close_turn(battle.battle_tag)   # a battle that ends without `|upkeep|` still logs its last turn
        self._emit("battle_end", battle.battle_tag, battle.turn, won=battle.won, turns=battle.turn)
        if self.barrier is not None:
            self.barrier.unregister(self.player_id)
