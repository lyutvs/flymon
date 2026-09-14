"""Router: the fly chooses only when the coach wants to attack and there is a real choice."""
from __future__ import annotations

from poke_env.battle import AbstractBattle, Move

from .coach import CoachDecision
from .moves import attack_allowed, is_attack


def candidates(battle: AbstractBattle) -> list[Move]:
    """The fly's attack candidates, in request order."""
    return [m for m in battle.available_moves if is_attack(m) and attack_allowed(m)[0]]


def route(coach_decision: CoachDecision, cands: list[Move]) -> tuple[str, list[Move]]:
    if coach_decision.kind == "attack" and len(cands) >= 2:
        return "fly", cands
    return "coach", []
