"""AD's battle players (spec AD.1, AD.5 2). Both carry the fly's global number gfly (24-47; every derive_seed key) and
its current attempt token. ADAgentPlayer's pulse requests carry the attempt token and the fly's weight generation
read when the fly made the choice they reinforce, so a request of an attempt that was later rolled back - even one whose
outcome arrives after the retry began - is refused at submission (ADSwarm) and at write-back (LeverFlyPool.reinforce_batch). A fly decision of a fly whose encoder has a shadow (FLY-OS) records the true
opponent, the drawn species and whether their type sets match. Every record carries gfly (fly stays the pool index)."""
from __future__ import annotations

from ..agent.player import AgentPlayer
from ..agent.policy import derive_seed
from ..agent.pulses import pulses_for
from ..rescope.blocks import EvalPlayer


class _ADCommon:
    def __init__(self, *args, gfly: int, attempts, gen_of, **kw):
        super().__init__(*args, **kw)
        self.gfly, self.attempts, self.gen_of = int(gfly), attempts, gen_of
        self.attempt = None

    def start_attempt(self, battle_id: str, battle_index: int, attempt: int) -> None:
        self.start_battle(battle_id, battle_index)
        self.attempt = self.attempts.begin(self.fly, battle_id, attempt)
        if hasattr(self.encoder, "set_battle"):
            self.encoder.set_battle(battle_id)

    def _write(self, rec: dict) -> None:
        super()._write({"gfly": self.gfly, **rec})

    def _log(self, battle, who, decision, cands, chosen, detail=None) -> None:
        if who == "fly" and hasattr(self.encoder, "shadow"):
            detail = dict(detail or {}, os=self.encoder.shadow(battle))
        super()._log(battle, who, decision, cands, chosen, detail)


class ADAgentPlayer(_ADCommon, AgentPlayer):
    async def _choose_move(self, battle):
        """AgentPlayer._choose_move, then the fly's choice is stamped with the attempt token, battle id and weight
        generation read when the decision began, so its pulse carries the choosing attempt's values even if its outcome
        arrives after a rollback and the retry's start_attempt (AD.5 2: the old room may still deliver messages)."""
        token = None if self.attempt is None else list(self.attempt)
        bid = self.battle_id
        gen = int(self.gen_of(self.fly))   # read without the pool lock; ADSwarm and the pool re-check it under the lock
        order = await super()._choose_move(battle)
        choice = self._choice.get(battle.battle_tag)
        if choice is not None and "attempt" not in choice:
            choice.update(attempt=token, battle_id=bid, gen=gen)
        return order

    # copied from flymon/agent/player.py:AgentPlayer._on_outcome (+ the global fly in the pulse seed; the attempt token,
    # battle id and weight generation stamped on the choice when the fly decided, AD.5 2)
    def _on_outcome(self, tag: str, turn: int, outcome) -> None:
        choice = self._choice.pop(tag, None)
        if choice is None or choice["turn"] != turn:
            return                                    # a coach turn: no DAN (spec 4.3 safety constraint 3)
        pulses = pulses_for(outcome)
        if not pulses:
            self.no_signal[tag] = self.no_signal.get(tag, 0) + 1
        token, gen = choice.get("attempt"), choice.get("gen")    # the choosing attempt's, stamped in _choose_move
        bid = choice.get("battle_id", self.battle_id)
        for j, (dan, ms) in enumerate(pulses):
            self._queue.setdefault(tag, []).append({"fly": self.fly, "odour": choice["odour"], "dan": dan, "ms": float(ms),
                                                    "seed": derive_seed("reinforce", self.gfly, bid, turn, j),
                                                    "attempt": token, "gen": gen})
        self._write({"battle_tag": tag, "turn": turn, "kind": "reinforce", "odour_move": choice["move"],
                     "pulses": [[d, float(ms)] for d, ms in pulses], "attempt": token, "gen": gen})


class ADEvalPlayer(_ADCommon, EvalPlayer):
    """EvalPlayer (no pulse, multipliers logged) with gfly, the attempt token and the FLY-OS shadow."""
