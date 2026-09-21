"""The oracle's pair list for M0d H.4 (spec H.4 step 2 = G.14.3: encoder E0, even turns, (a) 18 pairs and (b) 21 pairs).

Ported from the committed calibration code with the same calls in the same order — the turns and vocabulary from
`m2-calibration-g/m2_probe.py` (`pool_vocabulary`, `assign_channels`, `build_turns`) and the E0 encoder and the pairs
from `m2-calibration-g/m2_encoder_compare.py` (`build_encoder("E0")`, `odour`, `alternate_opponent`, `pairs_for`),
which produced G.12's and G.14's pair lists. Only E0 is ported (H.4 names no other encoder). tests/brain/test_h4_pairs.py
checks the port against those modules and pins the list by digest (`H4Spec.pairs_digest`).
"""
from __future__ import annotations

import hashlib
import itertools
import json

import numpy as np

from ..battle.pool import POOL
from .circuits import Populations

EXCLUDE = ("ORN_DA1", "ORN_V")          # spec 3.3: cVA and CO2 channels are not free to reassign
POWER, HP = ["lt60", "60to89", "ge90"], ["low", "mid", "high"]
POWER_EDGES = (60, 90)
HP_EDGES = (0.34, 0.67)


def pool_vocabulary():
    """Species types, move (type, base power), the sorted mon types and move types of the 16-mon pool (Gen 1)."""
    from poke_env.battle import Move
    from poke_env.data import GenData
    from poke_env.data.normalize import to_id_str
    gd = GenData.from_gen(1)
    species_types, move_info = {}, {}
    for m in POOL:
        species_types[m.species] = tuple(t.upper() for t in gd.pokedex[to_id_str(m.species)]["types"])
        for a in m.attacks:
            mv = Move(to_id_str(a), gen=1)
            move_info[a] = (mv.type.name, int(mv.base_power))
    mon_types = sorted({t for v in species_types.values() for t in v})
    move_types = sorted({v[0] for v in move_info.values()})
    return species_types, move_info, mon_types, move_types


def assign_channels(pops: Populations, groups: list) -> dict:
    """m2_probe.assign_channels: receptor types by receptor count, trimmed symmetrically, spread per group."""
    cand = sorted((t for t in pops.receptor_types if t not in EXCLUDE),
                  key=lambda t: (len(pops.receptor_types[t]), str(t)))
    need = sum(len(names) for _, names in groups)
    if need > 45:
        raise ValueError(f"spec 3.3 allows at most 45 channels, this vocabulary needs {need}")
    if need > len(cand):
        raise ValueError(f"need {need} receptor types, have {len(cand)}")
    lo = (len(cand) - need) // 2
    band = cand[lo:lo + need]
    free = set(range(need))
    out = {}
    for gname, names in sorted(groups, key=lambda g: (len(g[1]), g[0])):
        q = len(names)
        for j, nm in enumerate(names):
            ideal = (j + 0.5) * need / q
            pos = min(free, key=lambda p: (abs(p + 0.5 - ideal), p))
            free.discard(pos)
            out[f"{gname}:{nm}"] = str(band[pos])
    return out


def power_bin(bp: int) -> str:
    return "lt60" if bp < POWER_EDGES[0] else ("60to89" if bp < POWER_EDGES[1] else "ge90")


def hp_bin(frac: float) -> str:
    return "low" if frac <= HP_EDGES[0] else ("mid" if frac <= HP_EDGES[1] else "high")


def build_turns(species_types, move_info, n_turns: int) -> list:
    hps = [(0.9, 0.9), (0.9, 0.3), (0.5, 0.6), (0.2, 0.8), (0.6, 0.2), (0.3, 0.5)]
    turns = []
    for i in range(n_turns):
        me, opp = POOL[i % len(POOL)], POOL[(i * 5 + 3) % len(POOL)]
        my_hp, op_hp = hps[i % len(hps)]
        cands = [{"move": a, "type": move_info[a][0], "bp": move_info[a][1]} for a in me.attacks][:4]
        turns.append({"turn": i, "me": me.species, "opp": opp.species,
                      "my_types": list(species_types[me.species]), "opp_types": list(species_types[opp.species]),
                      "my_hp": my_hp, "opp_hp": op_hp, "candidates": cands})
    return turns


def e0_channels(pops: Populations, mon_types, move_types) -> dict:
    """m2_encoder_compare.build_encoder("E0"): {"group:symbol": [receptor type]} by receptor count."""
    syms = ([("my", t) for t in mon_types] + [("opp", t) for t in mon_types] + [("move", t) for t in move_types]
            + [("pow", b) for b in POWER] + [("myhp", b) for b in HP] + [("opphp", b) for b in HP])
    groups = {}
    for g, sym in syms:
        groups.setdefault(g, []).append(sym)
    flat = assign_channels(pops, [(g, v) for g, v in groups.items()])
    return {k: [v] for k, v in flat.items()}


def odour(pops, chan, my_types, opp_types, move, bp, my_hp, opp_hp) -> dict:
    """m2_encoder_compare.odour with every E0 group present: strengths proportional to 1/receptor count, mean 1."""
    keys = ([f"my:{t}" for t in my_types] + [f"opp:{t}" for t in opp_types] + [f"move:{move}", f"pow:{power_bin(bp)}",
            f"myhp:{hp_bin(my_hp)}", f"opphp:{hp_bin(opp_hp)}"])
    glom = [g for k in keys for g in chan[k]]
    if len(set(glom)) != len(glom):
        raise ValueError(f"glomerulus collision in {keys}")
    inv = np.array([1.0 / len(pops.receptor_types[g]) for g in glom]); inv /= inv.mean()
    return {g: float(s) for g, s in zip(glom, inv)}


def alternate_opponent(turn_index: int, me: str, opp_types, species_types) -> str:
    """G.11: POOL[(5i + 3 + k) mod 16], the first k >= 1 with no shared type and not my own species."""
    for k in range(1, len(POOL)):
        cand = POOL[(5 * turn_index + 3 + k) % len(POOL)].species
        if cand != me and not (set(species_types[cand]) & set(opp_types)):
            return cand
    raise ValueError(f"turn {turn_index}: no type-disjoint alternate opponent")


def even_pairs(pops: Populations, n_turns: int = 16) -> list:
    """[{axis, turn, x, y, odor_x, odor_y}] for the even turns, in G.12/G.14 order ((a) then (b) within each turn)."""
    species_types, move_info, mon_types, move_types = pool_vocabulary()
    chan = e0_channels(pops, mon_types, move_types)
    out = []
    for t in build_turns(species_types, move_info, n_turns):
        if t["turn"] % 2:
            continue
        cands = t["candidates"]
        od = lambda c, opp_types: odour(pops, chan, t["my_types"], opp_types, c["type"], c["bp"], t["my_hp"], t["opp_hp"])
        for i, j in itertools.combinations(range(len(cands)), 2):
            out.append({"axis": "a", "turn": t["turn"], "x": cands[i]["move"], "y": cands[j]["move"],
                        "odor_x": od(cands[i], t["opp_types"]), "odor_y": od(cands[j], t["opp_types"])})
        alt = alternate_opponent(t["turn"], t["me"], t["opp_types"], species_types)
        for c in cands:
            out.append({"axis": "b", "turn": t["turn"], "x": f"{c['move']} vs {t['opp']}", "y": f"{c['move']} vs {alt}",
                        "odor_x": od(c, t["opp_types"]), "odor_y": od(c, list(species_types[alt]))})
    return out


def pair_key(p: dict) -> tuple:
    return (p["axis"], int(p["turn"]), p["x"], p["y"])


def pairs_digest(pairs: list) -> str:
    rows = [[p["axis"], int(p["turn"]), p["x"], p["y"], sorted((k, float(v)) for k, v in p["odor_x"].items()),
             sorted((k, float(v)) for k, v in p["odor_y"].items())] for p in pairs]
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()
