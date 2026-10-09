"""Situation-pair evaluation (spec AC.5): at learning battles 0 / 10 / 20 / 30 / 40 (C-off: 0 only, AC.2) each
confirmation-set situation is presented with plasticity and exploration off and decided by argmax with the AC.2 tie
rule. A fly's switch rate = the share of its pairs where it picked the best move in both situations. Decision seed
derive_seed(tie_seed, "sit", fly, pair, side), tie seed derive_seed(tie_seed, "sit-tie", fly, pair, side): no point
in the seed, so a change between points comes from the weights only. The fly's weight sha256 is recorded before and
after (frozen must hold). floor_contact is AC.3's shadow: the floor (w/w0 <= 0.2) share of the taught edges (PAM08 /
PPL105 core-compartment plastic edges with w != w0), taurec's tolerance.
seed_fly / seed_key (AD.1, AD.2): the global fly and the key of the new set ("sit-new"); defaults are AC's."""
from __future__ import annotations

import numpy as np

from ..agent import policy
from ..agent.encode_grid import odour
from ..rescope.blocks import weights_sha


def offline_odour(enc, move: str, opp: str) -> dict:
    mtype = enc.move_info[move][0]
    if hasattr(enc, "tb_odour"):
        return enc.tb_odour(mtype)
    return odour(enc.rc, enc.cb, mtype, tuple(sorted(enc.species_types[opp])), enc.rule)


def pair_odours(enc, pairs) -> list:
    return [([offline_odour(enc, m, p["o1"]) for m in p["cands"]], [offline_odour(enc, m, p["o2"]) for m in p["cands"]])
            for p in pairs]


def evaluate(pool, fly: int, point: int, pairs, odours, cfg, a_idx, p_idx, kc_idx, tie_seed: int, seed_fly=None,
             seed_key: str = "sit") -> dict:
    a_idx, p_idx, kc_idx = np.asarray(a_idx), np.asarray(p_idx), np.asarray(kc_idx)
    idx = np.concatenate([a_idx, p_idx, kc_idx])
    na, npp = len(a_idx), len(p_idx)
    sf = int(fly) if seed_fly is None else int(seed_fly)
    reqs, keys = [], []
    for i, (o1, o2) in enumerate(odours):
        for side, ods in ((0, o1), (1, o2)):
            reqs.append((int(fly), list(ods), policy.derive_seed(tie_seed, seed_key, sf, i, side)))
            keys.append((i, side))
    before = weights_sha(pool.w[fly])
    counts = pool.decide_batch(reqs, cfg.strength, cfg.settle_ms, cfg.read_ms, idx)
    after = weights_sha(pool.w[fly])
    sits, ok = [], {}
    for (i, side), cnt in zip(keys, counts):
        cnt = np.asarray(cnt)
        a, p = cnt[:, :na].sum(1), cnt[:, na:na + npp].sum(1)
        v = policy.values(a, p, cfg.z)
        pick, tied = policy.argmax_tiebreak(v, policy.derive_seed(tie_seed, f"{seed_key}-tie", sf, i, side))
        pr = pairs[i]
        best = pr["best1"] if side == 0 else pr["best2"]
        ok[(i, side)] = pr["cands"][pick] == best
        sits.append(dict(pair=i, side=side, v=[float(x) for x in v], a=[int(x) for x in a], p=[int(x) for x in p],
                         kc_active=[int(x) for x in (cnt[:, na + npp:] > 0).sum(1)], pick=int(pick), tie=bool(tied),
                         correct=bool(ok[(i, side)])))
    switched = [int(ok[(i, 0)] and ok[(i, 1)]) for i in range(len(odours))]
    return dict(kind="situation_eval", fly=int(fly), point=int(point), rate=float(np.mean(switched)),
                switched=switched, situations=sits, ties=sum(s["tie"] for s in sits),
                a_zero=sum(x == 0 for s in sits for x in s["a"]), p_zero=sum(x == 0 for s in sits for x in s["p"]),
                n_candidates=sum(len(s["a"]) for s in sits), weights_sha_before=before, weights_sha_after=after,
                frozen=before == after)


def floor_contact(pool, fly: int, edge_masks: dict, types, ratio: float):
    w0 = pool.w0[pool.flies[fly].shuffle_seed]
    w = pool.w[fly]
    m = np.zeros(np.shape(w), bool)
    for t in types:
        m |= np.asarray(edge_masks[t], bool)
    taught = m & (w != w0)
    if not taught.any():
        return None
    return float(np.mean((w[taught] / w0[taught]) <= ratio * (1 + 1e-6)))
