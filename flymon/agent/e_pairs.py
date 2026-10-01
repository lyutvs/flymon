"""Spec 4.4 and 5.1: the even selection situations (H.4's list, re-encoded) and the judgement set from the L
generator's turns 64-209, deduplicated by E-grid key against every earlier pair and within itself."""
from __future__ import annotations

import hashlib
import itertools
import json

from ..brain import h4_pairs, l_pairs
from ..brain.h4_spec import SPEC as H4
from .e_spec import SPEC
from .encode_grid import glomeruli, odour


def _rows_for_turn(pops, chan, t, alt, species_types, with_a: bool) -> list:
    """l_pairs._turn_pairs' rows (same labels, same order) with the E-grid situation fields added."""
    e0 = l_pairs._turn_pairs(pops, chan, t, alt, species_types, with_a)
    cands = t["candidates"]; opp = tuple(sorted(t["opp_types"])); alt_t = tuple(sorted(species_types[alt]))
    meta = []
    if with_a:
        for i, j in itertools.combinations(range(len(cands)), 2):
            meta.append((cands[i]["type"], opp, cands[j]["type"], opp))
    for c in cands:
        meta.append((c["type"], opp, c["type"], alt_t))
    assert len(meta) == len(e0)
    return [dict(axis=r["axis"], turn=int(r["turn"]), x=r["x"], y=r["y"], move_x=mx, opp_x=ox, move_y=my, opp_y=oy,
                 odor_x_e0=r["odor_x"], odor_y_e0=r["odor_y"]) for r, (mx, ox, my, oy) in zip(e0, meta)]


def _e0_digest(rows) -> str:
    return h4_pairs.pairs_digest([dict(axis=r["axis"], turn=r["turn"], x=r["x"], y=r["y"], odor_x=r["odor_x_e0"],
                                       odor_y=r["odor_y_e0"]) for r in rows])


def even_situations(pops) -> list:
    st, mi, mon, move = h4_pairs.pool_vocabulary()
    chan = h4_pairs.e0_channels(pops, mon, move)
    out = []
    for t in h4_pairs.build_turns(st, mi, 16):
        if t["turn"] % 2:
            continue
        alt = h4_pairs.alternate_opponent(t["turn"], t["me"], t["opp_types"], st)
        out += _rows_for_turn(pops, chan, t, alt, st, with_a=True)
    if _e0_digest(out) != H4.pairs_digest:
        raise ValueError("even situations do not reproduce H4Spec.pairs_digest")
    return out


def egrid_key(r) -> frozenset:
    """Unordered pair of odour keys (move type, sorted opponent types)."""
    return frozenset({(r["move_x"], tuple(r["opp_x"])), (r["move_y"], tuple(r["opp_y"]))})


def used_situations(pops, spec=SPEC) -> list:
    """Every (a)/(b) row of build_turns' 16 turns and of L turns 0-63 (every L turn with (a) pairs)."""
    st, mi, mon, move = h4_pairs.pool_vocabulary()
    chan = h4_pairs.e0_channels(pops, mon, move)
    out = []
    for t in h4_pairs.build_turns(st, mi, 16):
        alt = h4_pairs.alternate_opponent(t["turn"], t["me"], t["opp_types"], st)
        out += _rows_for_turn(pops, chan, t, alt, st, with_a=True)
    for t in l_pairs.new_turns(st, mi, spec.judge_first_turn, spec.l_rng_seed):
        alt = l_pairs.alternate_from(t["opp"], t["me"], t["opp_types"], st, t["turn"])
        out += _rows_for_turn(pops, chan, t, alt, st, with_a=True)
    return out


def _skip(r, reason) -> dict:
    return {"key": [r["axis"], r["turn"], r["x"], r["y"]], "reason": reason}


def judgement_set(pops, spec) -> dict:
    """L turns judge_first_turn..judge_last_turn in order; a row is skipped if its E-grid key is used or already taken.
    (b) rows are kept until spec.n_b; the rest of that turn is dropped ("after_21st")."""
    st, mi, mon, move = h4_pairs.pool_vocabulary()
    chan = h4_pairs.e0_channels(pops, mon, move)
    used = {egrid_key(r) for r in used_situations(pops, spec)}
    taken = set()
    b, a, skipped, last_turn = [], [], [], None
    turns = l_pairs.new_turns(st, mi, spec.l_total_turns, spec.l_rng_seed)
    for t in turns[spec.judge_first_turn:spec.judge_last_turn + 1]:
        alt = l_pairs.alternate_from(t["opp"], t["me"], t["opp_types"], st, t["turn"])
        rows = _rows_for_turn(pops, chan, t, alt, st, with_a=True)
        for ix, r in enumerate(rows):
            k = egrid_key(r)
            if k in used:
                skipped.append(_skip(r, "used")); continue
            if k in taken:
                skipped.append(_skip(r, "in_set")); continue
            taken.add(k)
            (b if r["axis"] == "b" else a).append(r)
            if len(b) == spec.n_b:
                skipped += [_skip(q, "after_21st") for q in rows[ix + 1:]]
                break
        last_turn = t["turn"]
        if len(b) == spec.n_b:
            break
    status = "OK" if len(b) == spec.n_b else "STOP_SET_SHORT"
    keys = [[r["axis"], r["turn"], sorted([m, list(o)] for m, o in egrid_key(r))] for r in b + a]
    return dict(b=b, a=a, n_a=len(a), last_turn=last_turn, skipped=skipped, status=status,
                digest_e0_b=_e0_digest(b), digest_e0_a=_e0_digest(a),
                digest_keys=hashlib.sha256(json.dumps(keys, separators=(",", ":")).encode()).hexdigest())


def attach_odours(rows, rc, cb, dual_rule) -> list:
    return [dict(r, odor_x=odour(rc, cb, r["move_x"], tuple(r["opp_x"]), dual_rule),
                 odor_y=odour(rc, cb, r["move_y"], tuple(r["opp_y"]), dual_rule)) for r in rows]


def _gl_key(r, cb) -> frozenset:
    return frozenset({frozenset(glomeruli(cb, r["move_x"], tuple(r["opp_x"]))),
                      frozenset(glomeruli(cb, r["move_y"], tuple(r["opp_y"])))})


def overlap_report(judge_rows, used_rows, rc, cb, dual_rule) -> dict:
    """Glomerulus-set level for one config: cross = judgement pairs whose set pair equals a used pair's; within =
    duplicates among the judgement pairs. rc and dual_rule do not change glomerulus sets (kept for one signature)."""
    used = {_gl_key(r, cb) for r in used_rows}
    jk = [_gl_key(r, cb) for r in judge_rows]
    return dict(cross=sum(k in used for k in jk), within=len(jk) - len(set(jk)))


def shared_odour_count(judge_b, even_rows) -> int:
    """Judgement (b) pairs with at least one odour key (move, opp) equal to an odour key of an even pair."""
    even = {k for r in even_rows for k in egrid_key(r)}
    return sum(bool(egrid_key(r) & even) for r in judge_b)
