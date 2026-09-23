"""Viewer events: one JSON object per thing that happened, joined on (fly, battle_tag, turn).

This is the contract the brain plugs into (live viewer design, section 4). Adding a field does not
bump VERSION; the page ignores fields and types it does not know.
"""
from __future__ import annotations

import json

import numpy as np

VERSION = 1
REQUIRED = {
    "battle_start": (),
    "decision": ("decider", "coach_kind", "candidates", "chosen"),
    "outcome": ("outcome",),
    "protocol": ("lines",),
    "trace": ("phase", "slot", "series"),
    "battle_end": ("won", "turns"),
}
PHASES = ("decide", "reinforce")
SERIES_KINDS = ("scalar", "bars", "text")


def make_event(etype: str, fly: str, battle_tag: str, turn: int, **fields) -> dict:
    if etype not in REQUIRED:
        raise ValueError(f"unknown event type {etype!r}")
    missing = [k for k in REQUIRED[etype] if k not in fields]
    if missing:
        raise ValueError(f"{etype} event is missing {missing}")
    if etype == "protocol":
        lines = fields["lines"]
        if not isinstance(lines, list) or not all(isinstance(x, str) for x in lines):
            raise ValueError("protocol lines must be a list of strings")
    if etype == "trace":
        if fields["phase"] not in PHASES:
            raise ValueError(f"trace phase must be one of {PHASES}, got {fields['phase']!r}")
        for s in fields["series"]:
            if (not isinstance(s, dict) or s.get("kind") not in SERIES_KINDS or not isinstance(s.get("path"), str)
                    or not isinstance(s.get("points"), list)):
                raise ValueError(f"bad trace series {s!r}")
    return {"v": VERSION, "fly": fly, "type": etype, "battle_tag": battle_tag, "turn": int(turn), **fields}


def _plain(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")


def to_json(obj) -> str:
    """json.dumps that also accepts numpy scalars and arrays (the brain's detail values are numpy)."""
    return json.dumps(obj, default=_plain, allow_nan=False)
