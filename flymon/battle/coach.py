"""Coach v1: SimpleHeuristicsPlayer's singles rules, classified so the router knows what it chose."""
from __future__ import annotations

from dataclasses import dataclass

from poke_env.battle import AbstractBattle, Move, MoveCategory, Pokemon
from poke_env.player import Player, SimpleHeuristicsPlayer
from poke_env.player.battle_order import BattleOrder

from .moves import is_attack


@dataclass
class CoachDecision:
    order: BattleOrder
    kind: str
    move: Move | None


def attack_score(battle: AbstractBattle, move: Move) -> float:
    """The coach's own attack scoring (SimpleHeuristicsPlayer): bp x STAB x stat ratio x accuracy x type."""
    active, opp = battle.active_pokemon, battle.opponent_active_pokemon
    phys = SimpleHeuristicsPlayer._stat_estimation(active, "atk") / SimpleHeuristicsPlayer._stat_estimation(opp, "def")
    spec = SimpleHeuristicsPlayer._stat_estimation(active, "spa") / SimpleHeuristicsPlayer._stat_estimation(opp, "spd")
    ratio = phys if move.category == MoveCategory.PHYSICAL else spec
    stab = 1.5 if move.type in active.types else 1.0
    return move.base_power * stab * ratio * move.accuracy * opp.damage_multiplier(move)


class Coach:
    def __init__(self, weak: bool = False):
        self.weak = weak

    def decide(self, battle: AbstractBattle) -> CoachDecision:
        if self.weak and not battle.force_switch:
            attacks = [m for m in battle.available_moves if is_attack(m)]
            if attacks:
                best = max(attacks, key=lambda m: attack_score(battle, m))
                return CoachDecision(Player.create_order(best), "attack", best)
        order, _ = SimpleHeuristicsPlayer.choose_singles_move(battle)
        target = getattr(order, "order", None)
        if isinstance(target, Move):
            return CoachDecision(order, "attack" if is_attack(target) else "support", target)
        if isinstance(target, Pokemon):
            return CoachDecision(order, "switch", None)
        return CoachDecision(order, "default", None)
