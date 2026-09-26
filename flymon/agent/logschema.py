"""M3 JSONL log schema (spec 3.6). Every record: schema, kind, fly, battle_id (the checkpoint commit key),
battle_tag, turn. Per kind the required extra fields below; unknown extra fields are allowed."""
from __future__ import annotations

SCHEMA = 1
KINDS = ("decision", "outcome", "reinforce", "battle")
COMMON = ("schema", "kind", "fly", "battle_id", "battle_tag", "turn")
REQUIRED = {
    "decision": ("decider", "coach_kind", "candidates", "chosen"),
    "outcome": ("outcome",),
    "reinforce": ("pulses", "odour_move"),
    "battle": ("won", "fly_turns", "coach_turns", "no_signal_turns", "compartment_fracs"),
}
FLY_DECISION = ("v", "a", "p", "kc_active", "tau", "seed")


def validate(rec: dict) -> None:
    if rec.get("kind") not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {rec.get('kind')!r}")
    need = COMMON + REQUIRED[rec["kind"]]
    if rec["kind"] == "decision" and rec.get("decider") == "fly":
        need += FLY_DECISION
    missing = [k for k in need if k not in rec]
    if missing:
        raise ValueError(f"{rec['kind']} record is missing {missing}")
    if rec["schema"] != SCHEMA:
        raise ValueError(f"schema {rec['schema']} != {SCHEMA}")
