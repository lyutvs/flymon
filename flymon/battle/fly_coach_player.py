"""poke-env player: coach decides strategy, a DecisionProvider picks among attack candidates."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from poke_env.battle import AbstractBattle
from poke_env.player import Player

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
    """

    def __init__(self, provider: DecisionProvider, coach: Coach, barrier: BatchBarrier | None = None,
                 log_path: Path | None = None, **player_kwargs):
        super().__init__(**player_kwargs)
        self.provider, self.coach, self.barrier = provider, coach, barrier
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
            att = self.attributors.setdefault(tag, TurnAttributor())
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

    def _close_turn(self, battle_tag: str) -> None:
        """End the turn's attribution block, if I acted, and log its `Outcome` against that turn."""
        att = self.attributors.get(battle_tag)
        if att is None or not att.has_block():
            return
        outcome = att.end_turn()
        self.outcomes.setdefault(battle_tag, []).append(outcome)
        self._write({"battle_tag": battle_tag, "turn": self.turns.get(battle_tag, 0), "kind": "outcome",
                     "outcome": asdict(outcome)})

    # ---- decision -------------------------------------------------------------------------
    async def choose_move(self, battle: AbstractBattle):
        decision = self.coach.decide(battle)
        who, cands = route(decision, candidates(battle))
        chosen = None
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
            order = self.create_order(chosen)
        else:
            order = decision.order
        self._log(battle, who, decision, cands, chosen)
        return order

    def _log(self, battle, who, decision, cands, chosen) -> None:
        rec = {"battle_tag": battle.battle_tag, "turn": battle.turn, "kind": "decision", "decider": who,
               "coach_kind": decision.kind, "candidates": [m.id for m in cands],
               "chosen": chosen.id if chosen else (decision.move.id if decision.move else None)}
        self.turn_log.setdefault(battle.battle_tag, []).append(rec)
        self._write(rec)

    def _write(self, rec: dict) -> None:
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

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
        if self.barrier is not None:
            self.barrier.unregister(self.player_id)
