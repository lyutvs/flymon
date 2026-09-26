"""AgentPlayer: FlyCoachPlayer with the M3 brain loop — screened routing (L.7), the fly's odours and V in the
decision record, and spec 3.4 reinforcement of fly-decided turns only (safety constraint 3). Reinforcement for turn N
is queued when N's attribution block closes and delivered before the fly's next decision (and by drain() at battle
end), through its own BatchBarrier so concurrent flies share one FlyPool batch."""
from __future__ import annotations

from dataclasses import asdict

from ..battle.fly_coach_player import FlyCoachPlayer
from ..battle.router import candidates
from . import logschema
from .policy import derive_seed
from .pulses import pulses_for
from .screen import screened_route


class _NoProvider:
    async def decide(self, battle, candidates, context):
        raise RuntimeError("AgentPlayer decides through its barrier")


class AgentPlayer(FlyCoachPlayer):
    def __init__(self, fly: int, encoder, table, rbarrier, **kw):
        if kw.get("barrier") is None:
            raise ValueError("AgentPlayer needs a decision barrier")
        super().__init__(provider=_NoProvider(), **kw)
        self.fly, self.encoder, self.table, self.rbarrier = int(fly), encoder, table, rbarrier
        self.battle_id, self.battle_index = None, 0
        self._choice: dict = {}        # battle_tag -> {"turn", "odour", "move"} of the fly's latest move choice
        self._queue: dict = {}         # battle_tag -> [pulse context]
        self._k: dict = {}             # battle_tag -> decisions so far (seed part)
        self.no_signal: dict = {}      # battle_tag -> fly turns whose outcome gave no pulse
        self.fatal: BaseException | None = None   # first exception raised inside choose_move (SafetyStop, pool ...)

    def start_battle(self, battle_id: str, battle_index: int) -> None:
        self.battle_id, self.battle_index = battle_id, int(battle_index)

    # ---- records --------------------------------------------------------------------------
    def _write(self, rec: dict) -> None:
        super()._write({"schema": logschema.SCHEMA, "fly": self.fly, "battle_id": self.battle_id, **rec})

    def _log(self, battle, who, decision, cands, chosen, detail=None) -> None:
        rec = {"battle_tag": battle.battle_tag, "turn": battle.turn, "kind": "decision", "decider": who,
               "coach_kind": decision.kind, "candidates": [m.id for m in cands],
               "chosen": chosen.id if chosen else (decision.move.id if decision.move else None)}
        if detail is not None:
            rec.update(detail)
        self.turn_log.setdefault(battle.battle_tag, []).append(rec)
        self._write(rec)
        self._emit("decision", rec["battle_tag"], rec["turn"],
                   **{k: rec[k] for k in ("decider", "coach_kind", "candidates", "chosen")})

    # ---- decision -------------------------------------------------------------------------
    async def choose_move(self, battle):
        """poke-env runs this in a message-handler task nobody awaits, so an exception here would only be logged and
        the battle would hang with no move sent. Record it on self.fatal (first one wins), forfeit the battle so
        battle_against returns, then re-raise; the caller re-raises self.fatal (spec 4.3: a safety failure stops the run)."""
        try:
            return await self._choose_move(battle)
        except Exception as e:
            if self.fatal is None:
                self.fatal = e
            try:
                await self.ps_client.send_message("/forfeit", battle.battle_tag)
            except Exception:                         # a dead socket: the run stops on self.fatal anyway
                self.logger.exception("forfeit of %s failed", battle.battle_tag)
            raise

    async def _choose_move(self, battle):
        tag = battle.battle_tag
        await self._flush(tag)
        decision = self.coach.decide(battle)
        cands = candidates(battle)
        key = self.encoder.situation_key(battle, cands) if len(cands) >= 2 else None
        who, cands = screened_route(decision, cands, key, self.table) if key else ("coach", [])
        chosen = detail = None
        if who == "fly":
            k = self._k.get(tag, 0); self._k[tag] = k + 1
            odours = [self.encoder.odour(battle, m) for m in cands]
            ctx = {"battle_tag": tag, "turn": battle.turn, "fly": self.fly, "battle_id": self.battle_id,
                   "battle_index": self.battle_index, "k": k, "odours": odours}
            self.barrier.register(self.player_id)
            idx = await self.barrier.submit(self.player_id, battle, cands, ctx)
            if not 0 <= idx < len(cands):
                raise ValueError(f"swarm returned index {idx} for {len(cands)} candidates")
            chosen, detail = cands[idx], ctx["detail"]
            self._choice[tag] = {"turn": battle.turn, "odour": odours[idx], "move": chosen.id}
            order = self.create_order(chosen)
        else:
            if not battle.force_switch:
                self._choice.pop(tag, None)
            order = decision.order
        self._log(battle, who, decision, cands, chosen, detail)
        return order

    # ---- reinforcement --------------------------------------------------------------------
    def _close_turn(self, battle_tag: str) -> None:
        n = len(self.outcomes.get(battle_tag, []))
        super()._close_turn(battle_tag)
        outs = self.outcomes.get(battle_tag, [])
        if len(outs) > n:
            self._on_outcome(battle_tag, self.turns.get(battle_tag, 0), outs[-1])

    def _on_outcome(self, tag: str, turn: int, outcome) -> None:
        choice = self._choice.pop(tag, None)
        if choice is None or choice["turn"] != turn:
            return                                    # a coach turn: no DAN (spec 4.3 safety constraint 3)
        pulses = pulses_for(outcome)
        if not pulses:
            self.no_signal[tag] = self.no_signal.get(tag, 0) + 1
        for j, (dan, ms) in enumerate(pulses):
            self._queue.setdefault(tag, []).append({"fly": self.fly, "odour": choice["odour"], "dan": dan, "ms": float(ms),
                                                    "seed": derive_seed("reinforce", self.fly, self.battle_id, turn, j)})
        self._write({"battle_tag": tag, "turn": turn, "kind": "reinforce", "odour_move": choice["move"],
                     "pulses": [[d, float(ms)] for d, ms in pulses]})

    def pending_pulses(self, tag: str) -> list:
        return list(self._queue.get(tag, []))

    async def _flush(self, tag: str) -> None:
        """Deliver the battle's queued pulses in order. If a submit raises (SafetyStop, pool failure) the player still
        leaves rbarrier, and the pulses not yet sent are dropped: that exception stops the run."""
        q = self._queue.pop(tag, [])
        if not q:
            return
        try:
            for ctx in q:
                self.rbarrier.register(self.player_id)
                await self.rbarrier.submit(self.player_id, None, [ctx["dan"]], ctx)
        finally:
            self.rbarrier.unregister(self.player_id)

    async def drain(self) -> None:
        for tag in list(self._queue):
            await self._flush(tag)

    def write_battle_summary(self, tag: str, fracs: dict) -> None:
        st = self.battle_stats(tag)
        self._write({"battle_tag": tag, "turn": self.turns.get(tag, 0), "kind": "battle", "won": st["won"],
                     "fly_turns": st["fly_turns"], "coach_turns": st["coach_turns"],
                     "no_signal_turns": self.no_signal.get(tag, 0), "compartment_fracs": fracs})
