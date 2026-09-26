"""Spec 3.4's reinforcement table: my move's Outcome -> DAN pulses. A resisted or immune hit that also dealt damage
gets both channels, as two presentations, reward first (one DAN per presentation; presentation.reinforce)."""
from __future__ import annotations


def pulses_for(outcome, reward_type: str = "PAM08", punish_type: str = "PPL105") -> list:
    if outcome.uncertain or outcome.missed or outcome.no_action:
        return []
    out = []
    reward_ms = 400.0 * outcome.dealt_frac if outcome.dealt_frac > 0 else 0.0
    if outcome.target_fainted_by_me:
        reward_ms += 200.0
    if reward_ms > 0:
        out.append((reward_type, reward_ms))
    if outcome.effectiveness == "immune":
        out.append((punish_type, 400.0))
    elif outcome.effectiveness == "resisted":
        out.append((punish_type, 200.0))
    return out
