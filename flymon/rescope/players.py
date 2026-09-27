"""FLY-RS: an AgentPlayer whose fly-turn reinforcement comes from the donor's queue, not from its own outcome
(spec 10.8). Everything else - routing, decisions, coach turns (no DAN, safety constraint 3), delivery through
rbarrier - is AgentPlayer's."""
from __future__ import annotations

from ..agent.player import AgentPlayer
from ..agent.policy import derive_seed


class YokedPlayer(AgentPlayer):
    def __init__(self, fly, encoder, table, rbarrier, queue, **kw):
        super().__init__(fly, encoder, table, rbarrier, **kw)
        self.yoke = queue

    def _on_outcome(self, tag: str, turn: int, outcome) -> None:
        """Mirrors AgentPlayer._on_outcome; only the pulse source differs (one queue bundle per fly turn)."""
        choice = self._choice.pop(tag, None)
        if choice is None or choice["turn"] != turn:
            return                                    # a coach turn: no DAN, and the queue is not consumed
        pulses = self.yoke.pop() or []
        if not pulses:
            self.no_signal[tag] = self.no_signal.get(tag, 0) + 1
        for j, (dan, ms) in enumerate(pulses):
            self._queue.setdefault(tag, []).append({"fly": self.fly, "odour": choice["odour"], "dan": dan, "ms": float(ms),
                                                    "seed": derive_seed("reinforce", self.fly, self.battle_id, turn, j)})
        self._write({"battle_tag": tag, "turn": turn, "kind": "reinforce", "odour_move": choice["move"],
                     "pulses": [[d, float(ms)] for d, ms in pulses], "yoked": True,
                     "donor_sha256": self.yoke.donor_sha256})
