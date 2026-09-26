"""The new pair set of spec appendix L (L.4, L.11.3): ordered (my species, opponent) combos of the 16-mon pool that
neither build_turns (turns 0-15) nor H.5's confirmation formula uses, permuted once by default_rng(20260927), the first
64 taken. Turn j: HP from build_turns' 6-cycle at j, candidates = my attacks (at most 4), G.11's type-disjoint alternate
opponent stepping from the opponent's pool position. (b) pairs per candidate; (a) pairs (G.10) only for the first
a_turns turns (F_a's fixed sample). A pair whose normalised odour pair equals one of the 16 old turns (both axes) or an
earlier new pair is skipped and recorded. Independence from the old sets is partial (same species pool, K.8.1)."""
from __future__ import annotations

import itertools

import numpy as np

from ..battle.pool import POOL
from .h4_pairs import alternate_opponent, build_turns, e0_channels, odour, pairs_digest, pool_vocabulary
from .k_pairs import odour_key

HPS = [(0.9, 0.9), (0.9, 0.3), (0.5, 0.6), (0.2, 0.8), (0.6, 0.2), (0.3, 0.5)]   # build_turns' list


def excluded_combos() -> set:
    """build_turns' (POOL[i], POOL[(5i+3)%16]) and H.5's (POOL[i], POOL[(7i+5+k)%16]), first k >= 0 off my species."""
    n = len(POOL)
    out = {(POOL[i].species, POOL[(5 * i + 3) % n].species) for i in range(n)}
    for i in range(n):
        k = 0
        while POOL[(7 * i + 5 + k) % n].species == POOL[i].species:
            k += 1
        out.add((POOL[i].species, POOL[(7 * i + 5 + k) % n].species))
    return out


def new_turns(species_types, move_info, n_turns: int, seed: int) -> list:
    """build_turns-shaped turns plus "pos" (the order index j); combos in POOL order, permuted once by the seed."""
    ex = excluded_combos()
    combos = [(a, b) for a in POOL for b in POOL if a.species != b.species and (a.species, b.species) not in ex]
    order = np.random.default_rng(seed).permutation(len(combos))
    turns = []
    for j, ix in enumerate(order[:n_turns]):
        me, opp = combos[int(ix)]
        my_hp, op_hp = HPS[j % len(HPS)]
        cands = [{"move": a, "type": move_info[a][0], "bp": move_info[a][1]} for a in me.attacks][:4]
        turns.append({"turn": j, "pos": j, "me": me.species, "opp": opp.species,
                      "my_types": list(species_types[me.species]), "opp_types": list(species_types[opp.species]),
                      "my_hp": my_hp, "opp_hp": op_hp, "candidates": cands})
    return turns


def alternate_from(opp: str, me: str, opp_types, species_types, turn: int) -> str:
    """G.11 from the opponent's pool position: the first k >= 1 with no shared type and not my species; raises."""
    names = [m.species for m in POOL]
    start = names.index(opp)
    for k in range(1, len(names)):
        cand = names[(start + k) % len(names)]
        if cand != me and not (set(species_types[cand]) & set(opp_types)):
            return cand
    raise ValueError(f"turn {turn}: no type-disjoint alternate opponent")


def _both(p) -> frozenset:
    return frozenset((odour_key(p["odor_x"]), odour_key(p["odor_y"])))


def _turn_pairs(pops, chan, t, alt: str, species_types, with_a: bool) -> list:
    """even_pairs' rows for one turn: (a) combinations (if with_a), then (b) per candidate against `alt`."""
    od = lambda c, opp_types: odour(pops, chan, t["my_types"], opp_types, c["type"], c["bp"], t["my_hp"], t["opp_hp"])
    cands, out = t["candidates"], []
    if with_a:
        for i, j in itertools.combinations(range(len(cands)), 2):
            out.append({"axis": "a", "turn": t["turn"], "x": cands[i]["move"], "y": cands[j]["move"],
                        "odor_x": od(cands[i], t["opp_types"]), "odor_y": od(cands[j], t["opp_types"])})
    for c in cands:
        out.append({"axis": "b", "turn": t["turn"], "x": f"{c['move']} vs {t['opp']}", "y": f"{c['move']} vs {alt}",
                    "odor_x": od(c, t["opp_types"]), "odor_y": od(c, list(species_types[alt]))})
    return out


def old_pairs(pops, chan, species_types, move_info) -> list:
    """Every (a) and (b) pair of build_turns' 16 turns (even and odd), for the odour-level exclusion."""
    out = []
    for t in build_turns(species_types, move_info, 16):
        alt = alternate_opponent(t["turn"], t["me"], t["opp_types"], species_types)
        out += _turn_pairs(pops, chan, t, alt, species_types, with_a=True)
    return out


def new_pairs(pops, spec) -> dict:
    """{b, a, skipped: [{key, reason in (old_set, in_set)}]} in declared order; (a) only for turns < spec.a_turns."""
    species_types, move_info, mon_types, move_types = pool_vocabulary()
    chan = e0_channels(pops, mon_types, move_types)
    old = {_both(p) for p in old_pairs(pops, chan, species_types, move_info)}
    seen = set(old)
    out = dict(b=[], a=[], skipped=[])
    for t in new_turns(species_types, move_info, spec.n_turns, spec.rng_seed):
        alt = alternate_from(t["opp"], t["me"], t["opp_types"], species_types, t["turn"])
        for p in _turn_pairs(pops, chan, t, alt, species_types, with_a=t["turn"] < spec.a_turns):
            k = _both(p)
            if k in seen:
                out["skipped"].append({"key": [p["axis"], p["turn"], p["x"], p["y"]],
                                       "reason": "old_set" if k in old else "in_set"})
                continue
            seen.add(k)
            out[p["axis"]].append(p)
    return out


def check_digests(s: dict, spec) -> dict:
    """Refuses an unpinned (empty) digest or a list whose digest differs from spec.b_digest / spec.a_digest."""
    for ax in ("b", "a"):
        want = getattr(spec, f"{ax}_digest")
        if not want:
            raise ValueError(f"spec.{ax}_digest is empty: the new ({ax}) list is not pinned")
        got = pairs_digest(s[ax])
        if got != want:
            raise ValueError(f"new ({ax}) list digest {got} differs from spec.{ax}_digest {want}")
    return s
