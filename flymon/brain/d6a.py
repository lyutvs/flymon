"""Spec G.8: the D.6 (a) re-judgement (E.7 #4) — a D.6 ruling, not calibration.

The 41 candidate odours of the first M2 measurement (E.1: the provisional spec-3.3 encoder, 16 turns, candidates in
turn order) x seeds 400-463 (G.3). Each is presented for settle 800 + read 600 ms on the default engine (Params() =
the M0c engine = H.3's C0, every H.2 mode off), plasticity off, weights reset, every candidate of a turn from the same
reset seed (E.1's paired noise). Per KC, the spikes in a 200 ms window sliding by 1 ms over the whole presentation
(h4_jobs._present_kc, the window G.14 and H.4 recorded); a presentation exceeds if some KC has >= 31 spikes in some
window (> 150 Hz). D.6 (a) is met iff any presentation exceeds.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json

import numpy as np

from . import h4_pairs
from .config import Params
from .h4_jobs import _present_kc

N_TURNS = 16
N_ODOURS = 41
SEEDS = tuple(range(400, 464))                  # G.3: the D.6 (a) re-judgement block
STRENGTH, SETTLE_MS, READ_MS = 0.35, 800.0, 600.0
WINDOW_MS = 200
OVER_SPIKES = 31                                # > 150 Hz in 200 ms
ODOURS_DIGEST = "29992673f80a629ba09dbb3cf43eb257846c55bc654cbc75f3a74f4af3d6c0c9"
CONDITION = "some KC has >= 31 spikes in a 200 ms window sliding by 1 ms over settle + read (> 150 Hz)"


def candidate_odours(pops) -> list:
    """[{turn, move, odor}] for E.1's 41 candidates, bit-exact to m2_probe.turn_odors (shared my/opp/HP keys first,
    then move and power; strengths 1/receptor count over the odour's glomeruli, mean 1)."""
    species_types, move_info, mon_types, move_types = h4_pairs.pool_vocabulary()
    chan = h4_pairs.e0_channels(pops, mon_types, move_types)
    rows = []
    for t in h4_pairs.build_turns(species_types, move_info, N_TURNS):
        shared = ([f"my:{x}" for x in t["my_types"]] + [f"opp:{x}" for x in t["opp_types"]]
                  + [f"myhp:{h4_pairs.hp_bin(t['my_hp'])}", f"opphp:{h4_pairs.hp_bin(t['opp_hp'])}"])
        for c in t["candidates"]:
            keys = shared + [f"move:{c['type']}", f"pow:{h4_pairs.power_bin(c['bp'])}"]
            glom = [chan[k][0] for k in keys]
            if len(set(glom)) != len(glom):
                raise ValueError(f"channel collision in {keys}")
            inv = np.array([1.0 / len(pops.receptor_types[g]) for g in glom])
            inv = inv / inv.mean()
            rows.append({"turn": int(t["turn"]), "move": c["move"], "odor": {g: float(s) for g, s in zip(glom, inv)}})
    return rows


def odours_digest(rows: list) -> str:
    blob = json.dumps([[int(r["turn"]), r["move"], sorted((k, float(v)) for k, v in r["odor"].items())] for r in rows],
                      separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def engine_params() -> dict:
    """Params() as it round-trips through JSON (tuples become lists)."""
    return json.loads(json.dumps(dataclasses.asdict(Params())))


def d6a_job(eng, pl, pops, comps, ro, odors: list, seed: int, strength: float = STRENGTH, settle_ms: float = SETTLE_MS,
            read_ms: float = READ_MS, window_ms: int = WINDOW_MS) -> dict:
    """One turn, one seed: every candidate from the same reset seed; plasticity off and weights reset around it."""
    pl.reset_weights(); pl.set_enabled(False)
    try:
        out = [_present_kc(eng, pl, pops, o, int(seed), strength, settle_ms, read_ms, int(window_ms)) for o in odors]
    finally:
        pl.reset_weights(); pl.set_enabled(True)
    return {"seed": int(seed), "max_win": [int(o["max_win"]) for o in out],
            "kc_active_frac": [float((o["read"] > 0).mean()) for o in out],
            "kc_spikes": [int(o["read"].sum()) for o in out]}


def judge(raw: dict) -> dict:
    """D.6 (a) from a raw G.8 record. ValueError names every way the record is not a complete G.8 run."""
    problems = []
    if raw.get("smoke") is not False:
        problems.append("not a full run (smoke is not False)")
    if list(raw.get("seeds", [])) != list(SEEDS):
        problems.append("seeds are not 400-463")
    if not (raw.get("e2_check") or {}).get("ok"):
        problems.append("the E.2 self-check is missing or failed")
    if raw.get("params") != engine_params():
        problems.append("engine is not Params() (the M0c engine)")
    fixed = dict(strength=STRENGTH, settle_ms=SETTLE_MS, read_ms=READ_MS, window_ms=WINDOW_MS, over_spikes=OVER_SPIKES)
    problems += [f"{k} {raw.get(k)!r} != {v!r}" for k, v in fixed.items() if raw.get(k) != v]
    odours = raw.get("odours", [])
    if len(odours) != N_ODOURS or odours_digest(odours) != ODOURS_DIGEST:
        problems.append(f"odours are not E.1's 41 candidates (n {len(odours)})")
    moves = {}
    for o in odours:
        moves.setdefault(int(o["turn"]), []).append(o["move"])
    rows = raw.get("rows", [])
    keys = [(int(r["turn"]), int(r["seed"])) for r in rows]
    if len(keys) != len(set(keys)):
        problems.append("duplicate (turn, seed) rows")
    want = {(t, s) for t in moves for s in SEEDS}
    if set(keys) != want:
        problems.append(f"rows cover {len(set(keys) & want)} of {len(want)} (turn, seed) cells, "
                        f"{len(set(keys) - want)} foreign")
    for r in rows:
        w = r.get("max_win", [])
        if len(w) != len(moves.get(int(r["turn"]), [])) or any(not isinstance(x, int) or x < 0 for x in w):
            problems.append(f"row (turn {r['turn']}, seed {r['seed']}): max_win {w!r} does not match its candidates")
    if problems:
        raise ValueError("; ".join(problems))
    over = [{"turn": int(r["turn"]), "move": moves[int(r["turn"])][j], "seed": int(r["seed"]), "max_win": int(w)}
            for r in rows for j, w in enumerate(r["max_win"]) if w >= OVER_SPIKES]
    top = max(w for r in rows for w in r["max_win"])
    return {"condition": CONDITION, "fired": bool(over), "n_presentations": sum(len(r["max_win"]) for r in rows),
            "n_over": len(over), "max_win_spikes": int(top), "max_win_hz": top * 1000.0 / WINDOW_MS,
            "margin_spikes": OVER_SPIKES - 1 - int(top), "over": sorted(over, key=lambda o: -o["max_win"])[:50]}
