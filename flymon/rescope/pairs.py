"""New designed odour pairs (spec 4.2): 16 glomeruli drawn per seed from the whole candidate set, not a band."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from ..brain.stimuli import channel_strengths, design_odor_pair

EXCLUDE = ("ORN_DA1", "ORN_V")


def candidates(pops, exclude=EXCLUDE) -> list:
    cand = [t for t in pops.receptor_types if t not in exclude]
    return sorted(cand, key=lambda t: (len(pops.receptor_types[t]), t))


def sample_pair(pops, seed: int, k: int = 8) -> tuple:
    cand = candidates(pops)
    pick = sorted(np.random.default_rng(seed).choice(len(cand), size=2 * k, replace=False).tolist())
    band = [cand[i] for i in pick]
    return band[0::2], band[1::2]


def used_sets(pops, spec) -> list:
    out = []
    for s in spec.used_design_seeds:
        a, b = design_odor_pair(pops, k=spec.k, seed=s)
        out.append(frozenset(a) | frozenset(b))
    return out


def new_pairs(pops, spec) -> list:
    seen, pairs, seed = used_sets(pops, spec), [], spec.gen_start
    while len(pairs) < spec.n_pairs:
        a, b = sample_pair(pops, seed, spec.k)
        u = frozenset(a) | frozenset(b)
        if u not in seen:
            seen.append(u)
            pairs.append({"seed": seed, "a": list(a), "b": list(b)})
        seed += 1
    return pairs


def pairs_digest(pairs: list) -> str:
    canon = json.dumps([{"seed": p["seed"], "a": p["a"], "b": p["b"]} for p in pairs], sort_keys=True,
                       separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()


def pair_odors(pops, name: str, spec) -> dict:
    if name == spec.control:
        a, b = design_odor_pair(pops, k=spec.k, seed=0)
        return {"a": a, "b": b}
    seed = int(name[1:])
    p = next(p for p in new_pairs(pops, spec) if p["seed"] == seed)
    return {"a": channel_strengths(pops, p["a"]), "b": channel_strengths(pops, p["b"])}
