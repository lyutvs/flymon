"""The rules of spec appendix K (K.8.3–K.8.5): rank the ADOPTED engines by N (ties within tie_tol relative -> larger S
median, then smaller |g|), gate on N / N_C3 > gate_ratio, and name the closing state of a judgement. Pure functions."""
from __future__ import annotations

from .h3_rules import COMBO_ADOPTED
from .j_rules import COMPUTE_ABORTED, INVALID, STOP_MULTI_TYPE

SCAN_GO = "scan_go"
STOP_NO_QUALIFIED_SETTING = "stop_no_qualified_setting"
STOP_NO_TARGET_GAIN = "stop_no_target_gain"
DROPPED_NO_READOUT = "dropped_no_readout"


def select_order(settings: list, n_c3: float, tie_tol: float) -> list:
    if not n_c3 > 0:
        raise ValueError(f"N_C3 = {n_c3!r}: the ratio N / N_C3 is undefined (stop and ask the user, K.8.3)")
    left = [i for i, s in enumerate(settings) if s["status"] == COMBO_ADOPTED]
    order = []
    while left:
        best = max(settings[i]["metrics"]["N"] for i in left)
        tied = [i for i in left if settings[i]["metrics"]["N"] >= best * (1.0 - tie_tol)]
        pick = max(tied, key=lambda i: (settings[i]["metrics"]["S_median"], -abs(settings[i]["g"])))
        order.append(pick)
        left.remove(pick)
    return order


def scan_outcome(settings: list, order: list, n_c3: float, gate_ratio: float) -> str:
    if not order:
        return STOP_NO_QUALIFIED_SETTING
    return SCAN_GO if settings[order[0]]["metrics"]["N"] / n_c3 > gate_ratio else STOP_NO_TARGET_GAIN


def closing_state(judge: dict) -> str:
    o = judge["outcome"]
    if o in (STOP_MULTI_TYPE, INVALID, COMPUTE_ABORTED):
        return o
    rd = judge.get("reading")
    if rd is None:
        return DROPPED_NO_READOUT                     # j_runner.judge: B without a band (no readout in a pool)
    if rd.get("band") is None:
        raise ValueError(f"reading without a band: {rd!r} (off the declared 21 pairs is not a judgement)")
    return rd["band"]
