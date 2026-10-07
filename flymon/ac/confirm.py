"""The M4 confirmation set (spec AC.4; spec 4.4's confirmation set is built by this).
- A situation is (my Pokemon M, opponent O != M); its candidates are all of M's attacks (POOL has 2-3), its best move
  the one with the largest Gen 1 type multiplier against O's types (poke-env's gen-1 type_chart[defender][attacker]),
  which must be unique (tied situations are dropped).
- A pair is (M, O1, O2) with O1 before O2 in POOL order, different opponent type sets and different best moves. A pair
  touching an (M, O) of the L set (l_pairs.new_turns(..., 64, 20260927), L.4 2890) is dropped. Dedupe key =
  (M, {O1's type set, O2's type set}); the first pair in (M in POOL order, O1, O2) order is kept.
- Exactly n_pairs are drawn: sorted(default_rng(gen_seed).choice(n, n_pairs, replace=False)); fewer candidates ->
  STOP_SET. Each filter's surviving count is recorded (FILTERS).
The set is not separated from the learning schedule (AC.4: an E-grid odour depends only on (move type, opponent types),
so a fly may have met the same odours while learning; the result sentence says so)."""
from __future__ import annotations

import functools

import numpy as np

from ..battle.pool import POOL, by_species
from ..brain import l_pairs
from ..brain.h4_pairs import pool_vocabulary
from ..brain.l_spec import SPEC as L_SPEC
from .spec import SPEC
from .store import digest

FILTERS = ("all", "not_l", "unique_best", "distinct_types", "different_best", "deduped")


@functools.lru_cache(maxsize=1)
def _chart() -> dict:
    from poke_env.data import GenData
    return GenData.from_gen(1).type_chart


def type_mult(move_type: str, opp_types) -> float:
    m = 1.0
    for t in opp_types:
        m *= float(_chart()[t][move_type])
    return m


def _mults(me: str, opp: str, species_types, move_info) -> list:
    return [type_mult(move_info[a][0], species_types[opp]) for a in by_species[me].attacks]


def best_move(me: str, opp: str, species_types, move_info):
    m = _mults(me, opp, species_types, move_info)
    top = max(m)
    if m.count(top) != 1:
        return None
    return by_species[me].attacks[m.index(top)]


def l_set_combos(species_types, move_info) -> set:
    return {(t["me"], t["opp"]) for t in l_pairs.new_turns(species_types, move_info, L_SPEC.n_turns, L_SPEC.rng_seed)}


def candidate_pairs() -> dict:
    st, mi, _, _ = pool_vocabulary()
    lset = l_set_combos(st, mi)
    names = [m.species for m in POOL]
    counts = {f: 0 for f in FILTERS}
    seen, rows = set(), []
    for me in names:
        opps = [o for o in names if o != me]
        for i, o1 in enumerate(opps):
            for o2 in opps[i + 1:]:
                counts["all"] += 1
                if (me, o1) in lset or (me, o2) in lset:
                    continue
                counts["not_l"] += 1
                b1, b2 = best_move(me, o1, st, mi), best_move(me, o2, st, mi)
                if b1 is None or b2 is None:
                    continue
                counts["unique_best"] += 1
                t1, t2 = sorted(st[o1]), sorted(st[o2])
                if t1 == t2:
                    continue
                counts["distinct_types"] += 1
                if b1 == b2:
                    continue
                counts["different_best"] += 1
                key = [me, sorted([t1, t2])]
                hk = (me, tuple(tuple(x) for x in key[1]))
                if hk in seen:
                    continue
                seen.add(hk)
                counts["deduped"] += 1
                rows.append(dict(me=me, o1=o1, o2=o2, o1_types=t1, o2_types=t2, cands=list(by_species[me].attacks),
                                 best1=b1, best2=b2, mults1=_mults(me, o1, st, mi), mults2=_mults(me, o2, st, mi),
                                 key=key))
    return dict(pairs=rows, counts=counts)


def draw(rows: list, n: int, seed: int) -> list:
    idx = sorted(int(i) for i in np.random.default_rng(int(seed)).choice(len(rows), int(n), replace=False))
    return [rows[i] for i in idx]


def confirmation_set(spec=SPEC) -> dict:
    cand = candidate_pairs()
    rows = cand["pairs"]
    out = dict(counts=cand["counts"], n_candidates=len(rows), n_pairs=spec.n_pairs, seed=spec.gen_seed,
               l_set={"n_turns": L_SPEC.n_turns, "rng_seed": L_SPEC.rng_seed})
    if len(rows) < spec.n_pairs:
        return dict(out, status="STOP_SET", pairs=[], digest=None)
    pairs = draw(rows, spec.n_pairs, spec.gen_seed)
    return dict(out, status="OK", pairs=pairs, digest=digest(pairs))


def situations_of(pairs) -> list:
    out, seen = [], set()
    for p in pairs:
        for o in (p["o1"], p["o2"]):
            s = (p["me"], o, tuple(p["cands"]))
            if s not in seen:
                seen.add(s)
                out.append(s)
    return out
