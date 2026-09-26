"""The router's screen table (appendix L.7 / L.11.6): (my species, opponent species, my HP bin, opponent HP bin,
sorted attack-candidate ids) -> "fly" | "coach". Which situations go to the fly is decided by the M3/M4 declaration
(L.11 decision 11), not here; the table is an input and the smoke uses allow_all()."""
from __future__ import annotations

from ..battle.router import route

WHO = ("fly", "coach")


class ScreenTable:
    def __init__(self, entries: dict, default: str):
        bad = {k: v for k, v in entries.items() if v not in WHO}
        if bad or default not in WHO:
            raise ValueError(f"screen table values must be 'fly' or 'coach'; got default={default!r}, bad={bad}")
        self.entries, self.default = {tuple(k): v for k, v in entries.items()}, default

    def route(self, key: tuple) -> str:
        return self.entries.get(tuple(key), self.default)

    @classmethod
    def allow_all(cls) -> "ScreenTable":
        return cls({}, default="fly")

    def to_json(self) -> dict:
        return {"default": self.default, "entries": [[list(k[:4]) + [list(k[4])], v] for k, v in self.entries.items()]}

    @classmethod
    def from_json(cls, d: dict) -> "ScreenTable":
        return cls({(*k[:4], tuple(k[4])): v for k, v in d["entries"]}, default=d["default"])


def screened_route(coach_decision, cands: list, key: tuple, table: ScreenTable) -> tuple[str, list]:
    """The base router first (the coach attacks and there are >= 2 candidates); the table can only hand a fly turn
    back to the coach, never create one."""
    who, c = route(coach_decision, cands)
    if who == "fly" and table.route(key) == "fly":
        return "fly", c
    return "coach", []
