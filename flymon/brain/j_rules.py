"""Spec appendix J's decision rules as pure functions over measured rows (J.11.3 scan, J.11.4 reading, J.11.4-5 D.6).

Nothing here runs the engine. Editing this module never invalidates a cached measurement (it is outside
j_measure.MEASURE_FILES).
"""
from __future__ import annotations

import math

import numpy as np

from . import d6a

# ---- outcomes -----------------------------------------------------------------------------------------------------
SCAN_COMPLETE = "SCAN_COMPLETE"
NO_FEASIBLE_SETTING = "NO_FEASIBLE_SETTING"     # no setting both restorable and inside the KC band: stage 2 cannot run
SELECTED = "SELECTED"                           # testable_b >= 11/21 and F_a >= 2 (the M2 bar, G.14.4; J.12.9) -> H.5
B = "B"                                         # below the bar (J.11.5), with a band (J.12.9 decision 2):
B_TB = "B_Tb"                                   #   testable_b <= 7 (no rise over C3's 7/21): J.11.5's closing sentence
B_NO_CONCLUSION = "B_NO_CONCLUSION"             #   testable_b 8-10: rose over C3 but below the bar; neither closes nor selects
B_FA = "B_Fa"                                   #   testable_b >= 11 but F_a < 2: past the (b) bottleneck, not closed
INVALID = "INVALID"                             # G.14.4 row 1 on the oracle rows
STOP_MULTI_TYPE = "stop_multiple_types"         # two readout types in one pool (H.4 does not combine them)
STOP_NO_OPERATING_POINT = "STOP_NO_OPERATING_POINT"   # every tried setting failed the C3 re-convergence (J.11.4-1)
COMPUTE_ABORTED = "COMPUTE_ABORTED"             # the deadline passed (J.10.4): resumable, not a verdict
MEASURED = ("kc", "kc_on", "pn", "uni_hz", "orn_hz", "std_r")


# ================================================================ stage 1 (J.11.3)
def glom_means(rows: list) -> dict:
    """{glomerulus: {measure: seed mean, or None where a measure is None (no uniPN, no depression)}}."""
    by = {}
    for r in rows:
        by.setdefault(r["g"], []).append(r)
    out = {}
    for g, rs in by.items():
        out[g] = {k: (None if any(x.get(k) is None for x in rs) else float(np.mean([x[k] for x in rs])))
                  for k in MEASURED}
    return out


def log10_var(values, floor: float) -> float:
    x = np.log10(np.maximum(np.asarray(list(values), float), floor))
    return float(x.var())


def scan_metrics(rows: list, floor: float) -> dict:
    """The per-setting record: seed means per glomerulus, the median ALPN (the gain the scale restores), and the log10
    variance over glomeruli of every measure (uniPN rate over the glomeruli that have uniPNs; the rest are named)."""
    means = glom_means(rows)
    gl = sorted(means)
    var = {k: log10_var([means[g][k] for g in gl], floor) for k in ("kc", "kc_on", "pn")}
    with_uni = [g for g in gl if means[g]["uni_hz"] is not None]
    var["uni_hz"] = log10_var([means[g]["uni_hz"] for g in with_uni], floor) if with_uni else None
    return dict(n_glomeruli=len(gl), alpn_median=float(np.median([means[g]["pn"] for g in gl])),
                log10_var=var, no_uni=[g for g in gl if means[g]["uni_hz"] is None], per_glomerulus=means)


def lit_distance(f: float, tau: float, spec) -> float:
    return abs(math.log(f / spec.lit_f)) + abs(math.log(tau / spec.lit_tau_ms))


def select_order(settings: list, spec) -> list:
    """Indices of the settings that are restorable and inside the KC band, in the declared order: repeatedly the
    largest relative kc_on log10-variance reduction, a difference of at most tie_tol going to the setting closer to
    the literature constants (then the earlier one). Every setting is kept in the record; only these are judged."""
    live = [i for i, s in enumerate(settings) if s["restorable"] and s["feasible"]]
    order = []
    while live:
        best = max(settings[i]["reduction"] for i in live)
        tied = [i for i in live if best - settings[i]["reduction"] <= spec.tie_tol]
        pick = min(tied, key=lambda i: (lit_distance(settings[i]["f"], settings[i]["tau_ms"], spec), i))
        order.append(pick)
        live.remove(pick)
    return order


# ================================================================ stage 2 (J.11.4-3, bands by J.12.9 decision 2)
def stage2_reading(agg: dict, spec) -> dict:
    """The M2 bar H.4 applied to C0-C3 (G.14.4), read in COUNTS on the declared pair list (J.12.9): SELECTED iff
    testable_b >= stage2_select_testable_b and F_a >= f_a_min, otherwise B with a band — B_Tb (testable_b <=
    stage2_close_max_testable_b), B_NO_CONCLUSION (between the two), B_Fa (testable_b at the bar, F_a below it).
    `spec` is the J spec (H.4's bar through spec.h4). When n_b is not the declared stage2_n_b (smoke runs with fewer
    pairs) the band is None — not a judgement — and the outcome falls back to T_b >= t_b_min and F_a >= f_a_min.
    With fewer than f_a_min naive-balanced (a) pairs F_a cannot reach the bar; that is recorded for the B sentence."""
    spec4 = spec.h4
    tb, n_b, fa = int(agg["testable_b"]), int(agg["n_b"]), int(agg["F_a"])
    fa_ok = fa >= spec4.f_a_min
    note = None
    if n_b != spec.stage2_n_b:
        ok = bool(agg["T_b"] >= spec4.t_b_min and fa_ok)
        band, note = None, "pair list is not the declared one: not a judgement"
    elif tb >= spec.stage2_select_testable_b:
        ok = fa_ok
        band = SELECTED if ok else B_FA
    else:
        ok = False
        band = B_TB if tb <= spec.stage2_close_max_testable_b else B_NO_CONCLUSION
    out = dict(outcome=SELECTED if ok else B, band=band, T_b=float(agg["T_b"]), testable_b=tb, n_b=n_b, F_a=fa,
               naive_a=int(agg["naive_a"]), f_a_possible=bool(agg["naive_a"] >= spec4.f_a_min),
               bar=dict(t_b_min=spec4.t_b_min, f_a_min=spec4.f_a_min, n_b=spec.stage2_n_b,
                        select_testable_b=spec.stage2_select_testable_b,
                        close_max_testable_b=spec.stage2_close_max_testable_b))
    if note is not None:
        out["band_note"] = note
    return out


# ================================================================ D.6 on the judged engine (J.11.4-5)
def judge_d6a_declared(raw: dict, declared: dict) -> dict:
    """G.8's D.6 (a) rule on a declared engine. d6a.judge has exactly two engine-specific checks — the engine is
    Params() and E.2's self-check (E.1's cells re-presented on that engine) passed. This entry replaces both with one:
    the record's params equal the declared engine's (`declared`, the JSON form of its Params). Every other G.8 check
    (full run, seeds 400-463, E.1's 41 odours, every cell once, the window constants) is d6a.judge's, unchanged;
    d6a.judge itself stays G.8's C0-only entry."""
    if raw.get("params") != declared:
        raise ValueError("engine is not the declared engine")
    return d6a.judge({**raw, "params": d6a.engine_params(),
                      "e2_check": {"ok": True, "cells": "not applicable: declared engine (spec J.11.4-5)"}})


def d6b(raw: dict, limit: float) -> dict:
    """E.2's (b) on a d6a record: per (turn, seed) the candidates' read-window KC spike ratio max / max(1, min); a turn
    is over when its largest ratio > limit; (b) fires when any turn is over."""
    by = {}
    for r in raw["rows"]:
        s = [int(x) for x in r["kc_spikes"]]
        by.setdefault(int(r["turn"]), []).append(max(s) / max(1, min(s)))
    ratios = {t: {"mean": float(np.mean(v)), "max": float(np.max(v))} for t, v in sorted(by.items())}
    over = [t for t, v in ratios.items() if v["max"] > limit]
    return {"condition": f"within-turn candidate KC spike-count ratio > {limit}", "fired": bool(over),
            "turns_over": over, "n_turns_over": len(over), "n_turns": len(ratios),
            "ratio_max": float(max(v["max"] for v in ratios.values())),
            "ratio_mean_over_turns": float(np.mean([v["mean"] for v in ratios.values()])),
            "per_turn": {str(t): v for t, v in ratios.items()}}


# ================================================================ self-checks (J.11.6)
C3_FIELDS = ("kc_thresh", "apl_input_scale", "mbon_hold_frac", "kc_thresh_sha256", "apl_mode", "apl_r_max")


def c3_mismatch(got: dict, want: dict) -> list:
    """Self-check (ii): the fields of C3's adopted Params (JSON form) the re-converged engine does not reproduce. The
    threshold file's path differs by construction (J writes its own), so its sha256 is compared, not the path."""
    return [f for f in C3_FIELDS if got.get(f) != want.get(f)]


def cycle_divergence(got_cells: list, want_cells: list) -> dict | None:
    """Where the re-convergence parted from H.3's C3 record: the first (cell, cycle) whose accepted scale or whose
    end-of-homeostasis threshold sha256 differs; None when every recorded cycle agrees."""
    for ci, (g, w) in enumerate(zip(got_cells, want_cells)):
        for k, (gc, wc) in enumerate(zip(g.get("cycles", []), w.get("cycles", []))):
            gx, wx = gc["search"]["accepted"]["x"], wc["search"]["accepted"]["x"]
            gh = (gc.get("homeostasis") or {}).get("params")
            wh = (wc.get("homeostasis") or {}).get("params")
            gs = None if gh is None else (gh["kc_thresh_sha256"] if isinstance(gh, dict) else gh.kc_thresh_sha256)
            ws = None if wh is None else (wh["kc_thresh_sha256"] if isinstance(wh, dict) else wh.kc_thresh_sha256)
            if gx != wx or gs != ws:
                return dict(cell=ci, cycle=k + 1, scale=[gx, wx], threshold_sha256=[gs, ws])
        if len(g.get("cycles", [])) != len(w.get("cycles", [])):
            return dict(cell=ci, cycles=[len(g.get("cycles", [])), len(w.get("cycles", []))])
    return None


def pair_mismatch(got: dict, want: dict, keys=("d_pre", "r", "p", "m", "testable")) -> list:
    """Self-check (i): the statistics of one oracle pair that differ from H.4's recorded C3 pair."""
    return [k for k in keys if got.get(k) != want.get(k)]
