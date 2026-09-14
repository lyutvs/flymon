"""Decision providers: who picks among the fly's attack candidates. The brain provider arrives in M3."""
from __future__ import annotations

from typing import Protocol

import numpy as np
from poke_env.battle import AbstractBattle, Move

from .coach import attack_score


class DecisionProvider(Protocol):
    async def decide(self, battle: AbstractBattle, candidates: list[Move], context: dict) -> int: ...


class RandomProvider:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    async def decide(self, battle, candidates, context) -> int:
        return int(self.rng.integers(len(candidates)))


class MaxDamageProvider:
    async def decide(self, battle, candidates, context) -> int:
        scores = [attack_score(battle, m) for m in candidates]
        return int(np.argmax(scores))
