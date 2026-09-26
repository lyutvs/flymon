"""The odd-turn (b) pairs of spec appendix K (K.2 / K.8.1): h4_pairs.even_pairs' (b) rule (E0 channels, G.11's
alternate opponent) on the odd turns 1, 3, ..., 15, used only to select the engine (no learning). Independence from the
even pairs is partial — the opponents are species the even turns also meet — so the check is at odour level: identical
normalised (odour X, odour Y) pairs and shared odours are reported, never used to drop a pair."""
from __future__ import annotations

from .h4_pairs import alternate_opponent, build_turns, e0_channels, odour, pair_key, pool_vocabulary


def odd_pairs(pops, n_turns: int = 16) -> list:
    species_types, move_info, mon_types, move_types = pool_vocabulary()
    chan = e0_channels(pops, mon_types, move_types)
    out = []
    for t in build_turns(species_types, move_info, n_turns):
        if not t["turn"] % 2:
            continue
        alt = alternate_opponent(t["turn"], t["me"], t["opp_types"], species_types)   # raises with the turn number
        for c in t["candidates"]:
            od = lambda opp_types, c=c: odour(pops, chan, t["my_types"], opp_types, c["type"], c["bp"], t["my_hp"],
                                             t["opp_hp"])
            out.append({"axis": "b", "turn": t["turn"], "x": f"{c['move']} vs {t['opp']}", "y": f"{c['move']} vs {alt}",
                        "odor_x": od(t["opp_types"]), "odor_y": od(list(species_types[alt]))})
    return out


def odour_key(odor: dict) -> tuple:
    return tuple(sorted((str(g), round(float(v), 12)) for g, v in odor.items()))


def overlap_report(odd: list, others: dict) -> dict:
    """{name: {pairs: [{odd: key, other: key}], n_shared_odours}} — a pair matches when its two odours equal the other
    pair's two odours in either order."""
    def both(p):
        return frozenset((odour_key(p["odor_x"]), odour_key(p["odor_y"])))
    odd_odours = {odour_key(p[k]) for p in odd for k in ("odor_x", "odor_y")}
    rep = {}
    for name, pairs in others.items():
        index = {}
        for q in pairs:
            index.setdefault(both(q), []).append(q)
        hits = [{"odd": list(pair_key(p)), "other": list(pair_key(q))} for p in odd for q in index.get(both(p), [])]
        shared = odd_odours & {odour_key(q[k]) for q in pairs for k in ("odor_x", "odor_y")}
        rep[name] = {"pairs": hits, "n_shared_odours": len(shared)}
    return rep
