"""Which Gen 1 moves the fly may choose among (attacks) and which the coach may use (support)."""
from __future__ import annotations

from functools import lru_cache

from poke_env.battle import Move
from poke_env.data.normalize import to_id_str

SUPPORT_ALLOWED = {"thunderwave", "reflect", "lightscreen", "recover", "softboiled", "amnesia", "swordsdance", "agility"}
BAD_SECONDARY_STATUS = {"frz", "slp"}
MILD_STATUS = {"par", "brn", "psn"}


@lru_cache(maxsize=None)
def gen1(move_id: str) -> Move:
    """Look up a Gen 1 move by id or display name ("Mega Drain", "Soft-Boiled", ...)."""
    return Move(to_id_str(move_id), gen=1)


def is_attack(move: Move) -> bool:
    e = move.entry
    return e.get("category") in ("Physical", "Special") and e.get("basePower", 0) > 0 and "damage" not in e


def attack_allowed(move: Move) -> tuple[bool, str]:
    e = move.entry
    if e.get("category") not in ("Physical", "Special"):
        return False, "category: not an attack"
    if e.get("basePower", 0) <= 0 or "damage" in e:
        return False, "damage: fixed/level damage or zero power"
    flags = e.get("flags", {})
    for f in ("charge", "recharge"):
        if f in flags:
            return False, f"flags: {f}"
    if e.get("recoil"):
        return False, "recoil"
    if e.get("selfdestruct"):
        return False, "selfdestruct"
    if e.get("volatileStatus") or e.get("self", {}).get("volatileStatus"):
        return False, "volatileStatus: trapping/locking/bide"
    sec = e.get("secondary")
    if sec:
        if sec.get("volatileStatus"):
            return False, f"secondary volatileStatus {sec['volatileStatus']}"
        st = sec.get("status")
        if st in BAD_SECONDARY_STATUS:
            return False, f"secondary status {st}"
        if st in MILD_STATUS and sec.get("chance", 100) > 10:
            return False, f"secondary status {st} chance {sec.get('chance')} > 10"
        if "boosts" in sec and sec.get("chance", 100) > 33:
            return False, f"secondary boosts chance {sec.get('chance')} > 33"
        if st not in MILD_STATUS and "boosts" not in sec:
            return False, f"secondary: {sec}"
    acc = e.get("accuracy", 100)
    acc = 100 if acc is True else acc
    if acc < 90:
        return False, f"accuracy {acc} < 90"
    if e.get("critRatio", 1) != 1:
        return False, f"critRatio {e['critRatio']}"
    return True, ""


def support_allowed(move: Move) -> tuple[bool, str]:
    if is_attack(move):
        return False, "attack"
    return (move.id in SUPPORT_ALLOWED), ("" if move.id in SUPPORT_ALLOWED else "not in support allow-list")
