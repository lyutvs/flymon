"""Attribute what MY move did in a turn, straight from raw Showdown protocol lines (spec 3.4).

Pure and offline: no poke-env Battle object, no server. Feed every split protocol line in
order, call `end_turn()` at `|turn|`/`|upkeep|`, and get one `Outcome` describing the direct
damage my move dealt, its type effectiveness and whether it KO'd the target -- so that only
clean signals reinforce the brain.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from poke_env.data.normalize import to_id_str

DEFAULT_MAX_HP = 100  # opponent HP arrives as a percentage; used when a mon's max HP is unseen


@dataclass
class Outcome:
    """What my move did this turn. `uncertain`/`missed`/`no_action` mean "do not learn from it"."""

    move_id: str | None = None
    direct_damage: int = 0
    target_hp_before: int = 0
    target_max_hp: int = 0
    dealt_frac: float = 0.0
    effectiveness: str = "unknown"  # super | neutral | resisted | immune | unknown
    target_fainted_by_me: bool = False
    missed: bool = False
    no_action: bool = False
    uncertain: bool = False
    notes: list[str] = field(default_factory=list)


def _side(ident: str) -> str:
    """`"p1a: Lapras"` -> `"p1"` (the player id, without the slot letter)."""
    return ident[:2]


def _is_ident(token: str) -> bool:
    return len(token) > 4 and token[:1] == "p" and token[1:2].isdigit() and token[3:5] == ": "


def _parse_hp(token: str) -> tuple[int, int | None]:
    """`"0 fnt"` -> (0, None); `"12/100"` -> (12, 100); `"150/300"` -> (150, 300)."""
    token = token.split(" ")[0]
    if "/" in token:
        cur, _, mx = token.partition("/")
        return int(cur), int(mx)
    return int(token), None


def _multiplier_class(multiplier: float) -> str:
    if multiplier == 0:
        return "immune"
    if multiplier < 1:
        return "resisted"
    if multiplier == 1:
        return "neutral"
    return "super"


class TurnAttributor:
    """Stateful, per-battle parser. `my_side` is public: Task 8 refreshes it per message."""

    def __init__(self, my_side: str = "p1") -> None:
        self.my_side = my_side
        self.reset()

    def reset(self) -> None:
        """Clear everything, including HP tracking (new battle)."""
        self._hp: dict[str, tuple[int, int]] = {}
        self._reset_turn()

    def _reset_turn(self) -> None:
        self._out = Outcome()
        self._has_block = False
        self._in_block = False
        self._target: str | None = None
        self._after_direct = False
        self._any_direct = False

    def has_block(self) -> bool:
        """True iff a `|move|` by my side has been seen since the last `end_turn()`/`reset()`."""
        return self._has_block

    def feed(self, split_message: list[str]) -> None:
        """Consume one raw protocol line, already split on `|`."""
        if len(split_message) < 2:
            return
        event, args = split_message[1], split_message[2:]
        handler = getattr(self, f"_on_{event.lstrip('-').replace('-', '_')}", None)
        if event in ("turn", "upkeep"):
            self._in_block = False
        elif handler is not None:
            handler(args)
        # Any other event (`-hitcount`, `-message`, `-crit`, `-anim`, `-hint`, ...) is cosmetic or an
        # annotation: it carries no HP change, so it leaves faint adjacency untouched. Every event that
        # can change HP or end my action block has a handler above and breaks adjacency there.

    # -- protocol events -------------------------------------------------

    def _on_move(self, args: list[str]) -> None:
        if not args or not _is_ident(args[0]) or _side(args[0]) != self.my_side:
            self._in_block = False  # a foreign move ends my action block
            return
        if self._has_block:
            self._out.uncertain = True
            self._out.notes.append("multiple moves by my side in one turn")
            self._in_block = False
            return
        self._has_block = True
        self._in_block = True
        self._after_direct = False
        self._out.move_id = to_id_str(args[1]) if len(args) > 1 else None
        self._out.effectiveness = "neutral"
        if len(args) > 2 and _is_ident(args[2]):
            self._target = args[2]
        else:
            self._out.uncertain = True
            self._out.notes.append("target could not be identified")

    def _on_damage(self, args: list[str]) -> None:
        if not args or not _is_ident(args[0]):
            return
        ident = args[0]
        residual = any(a.startswith("[from]") for a in args[2:])
        before = self._hp.get(ident)
        cur, mx = self._parse_and_track(ident, args[1])
        if ident != self._target or not (self._in_block or self._has_block):
            self._after_direct = False
            return
        if residual or not self._in_block:
            self._after_direct = False
            why = "[from] residual" if residual else "residual (outside my action block)"
            self._out.notes.append(f"damage on target ignored: {why}")
            return
        hp_before, max_hp = before if before is not None else (mx or DEFAULT_MAX_HP, mx or DEFAULT_MAX_HP)
        if before is None:
            self._out.notes.append(f"target HP before was unseen; assumed max {max_hp}")
        self._out.direct_damage += max(0, hp_before - cur)
        if not self._any_direct:  # multi-hit: keep the HP the target had before the *first* hit
            self._out.target_hp_before = hp_before
        self._out.target_max_hp = max_hp
        self._after_direct = True
        self._any_direct = True

    def _on_heal(self, args: list[str]) -> None:
        if args and _is_ident(args[0]):
            self._parse_and_track(args[0], args[1])
        if self._in_block:
            self._after_direct = False

    def _on_sethp(self, args: list[str]) -> None:
        self._on_heal(args)

    def _on_switch(self, args: list[str]) -> None:
        if len(args) > 2 and _is_ident(args[0]):
            self._parse_and_track(args[0], args[2])
        if self._in_block:
            self._after_direct = False

    def _on_drag(self, args: list[str]) -> None:
        self._on_switch(args)

    def _on_activate(self, args: list[str]) -> None:
        if self._in_block and len(args) > 1 and args[0] == self._target and to_id_str(args[1]) == "substitute":
            self._out.uncertain = True
            self._out.notes.append("substitute absorbed the hit; no damage signal")
        self._after_direct = False

    def _on_miss(self, args: list[str]) -> None:
        if self._in_block:
            self._out.missed = True
        self._after_direct = False

    def _on_faint(self, args: list[str]) -> None:
        if args and _is_ident(args[0]):
            self._hp[args[0]] = (0, self._hp.get(args[0], (0, DEFAULT_MAX_HP))[1])
        if self._in_block and args and args[0] == self._target and self._after_direct:
            self._out.target_fainted_by_me = True
        self._after_direct = False

    def _on_cant(self, args: list[str]) -> None:
        if args and _is_ident(args[0]) and _side(args[0]) == self.my_side:
            self._out.no_action = True
            self._out.notes.append(f"cant: {args[1] if len(args) > 1 else 'unknown'}")
        self._in_block = False

    def _on_supereffective(self, args: list[str]) -> None:
        self._tag(args, "super")

    def _on_resisted(self, args: list[str]) -> None:
        self._tag(args, "resisted")

    def _on_immune(self, args: list[str]) -> None:
        self._tag(args, "immune")

    # -- helpers ---------------------------------------------------------

    def _tag(self, args: list[str], effectiveness: str) -> None:
        if self._in_block and args and _is_ident(args[0]) and _side(args[0]) != self.my_side:
            self._out.effectiveness = effectiveness

    def _parse_and_track(self, ident: str, token: str) -> tuple[int, int | None]:
        cur, mx = _parse_hp(token)
        known_max = mx if mx is not None else self._hp.get(ident, (0, None))[1]
        if known_max is not None:
            self._hp[ident] = (cur, known_max)
        return cur, mx

    def end_turn(self, expected_multiplier: float | None = None) -> Outcome:
        """Finish the turn: return its `Outcome` and clear per-turn state (HP tracking persists)."""
        out = self._out
        if not self._has_block:
            out.no_action = True
        if out.target_max_hp:
            out.dealt_frac = min(out.direct_damage, out.target_hp_before) / out.target_max_hp
        if self._has_block and expected_multiplier is not None:
            expected = _multiplier_class(expected_multiplier)
            if expected != out.effectiveness:
                out.uncertain = True
                out.notes.append(f"effectiveness mismatch: protocol {out.effectiveness}, expected {expected}")
        self._reset_turn()
        return out
