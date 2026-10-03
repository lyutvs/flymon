"""S's judgement set (S.2), list only — nothing here runs the engine.
- s_set: L generator turns first_turn..set_last_turn in declared order; a row whose E-grid key is in
  e_pairs.used_situations (H.4's 16 turns and L turns 0-63), in R's judgement set (turns 64-103, (b) 21 · (a) 32 — its
  three digests re-checked against R's declared values and the encoder block set first, r_pairs.check_set) or already
  in the set is skipped; (b) rows are kept until n_b and the rest of that turn is dropped (encoder 5.1,
  e_pairs.judgement_set's loop); (a) = every remaining (a) row of turns first_turn..T. Status STOP_SET_SHORT when the
  (b) rows run out before set_last_turn.
- check_s_set: the generated set against S's declared T, n_a and digests (S.2: a mismatch refuses).
- judgement_rows: the checked set's (b) 21 then (a) 43 rows with E-grid k2-norm odours; refused for a smoke spec. Only
  S's judgement stages call it (plan Global Constraints)."""
from __future__ import annotations

import hashlib
import json

from ..agent import e_pairs
from ..agent.e_spec import SPEC as E
from . import h4_pairs, l_pairs, q_pairs, r_pairs
from .r_spec import SPEC as R_SPEC

STOP_SET_SHORT = "STOP_SET_SHORT"
OK = "OK"


def r_set_keys(pops, enc: dict) -> set:
    """The E-grid keys of R's judgement set, after its digests, n_a and last turn match R's declared values and the
    encoder block set (ValueError otherwise)."""
    js = e_pairs.judgement_set(pops, E)
    bad = r_pairs.check_set(js, enc, R_SPEC)
    if bad:
        raise ValueError("R's judgement set does not reproduce: " + "; ".join(bad))
    return {e_pairs.egrid_key(r) for r in js["b"] + js["a"]}


def s_set(pops, enc: dict, spec) -> dict:
    st, mi, mon, move = h4_pairs.pool_vocabulary()
    chan = h4_pairs.e0_channels(pops, mon, move)
    used = {e_pairs.egrid_key(r) for r in e_pairs.used_situations(pops, E)}
    r_keys = r_set_keys(pops, enc)
    taken, b, a, skipped, last_turn = set(), [], [], {"used": 0, "r_set": 0, "in_set": 0, "after_21st": 0}, None
    turns = l_pairs.new_turns(st, mi, E.l_total_turns, E.l_rng_seed)
    for t in turns[spec.first_turn:spec.set_last_turn + 1]:
        alt = l_pairs.alternate_from(t["opp"], t["me"], t["opp_types"], st, t["turn"])
        rows = e_pairs._rows_for_turn(pops, chan, t, alt, st, with_a=True)
        for ix, r in enumerate(rows):
            k = e_pairs.egrid_key(r)
            if k in used:
                skipped["used"] += 1
                continue
            if k in r_keys:
                skipped["r_set"] += 1
                continue
            if k in taken:
                skipped["in_set"] += 1
                continue
            taken.add(k)
            (b if r["axis"] == "b" else a).append(r)
            if len(b) == spec.n_b:
                skipped["after_21st"] += len(rows) - ix - 1
                break
        last_turn = t["turn"]
        if len(b) == spec.n_b:
            break
    keys = [[r["axis"], r["turn"], sorted([m, list(o)] for m, o in e_pairs.egrid_key(r))] for r in b + a]
    return dict(b=b, a=a, n_b=len(b), n_a=len(a), last_turn=last_turn, skipped=skipped,
                status=OK if len(b) == spec.n_b else STOP_SET_SHORT,
                digest_e0_b=e_pairs._e0_digest(b), digest_e0_a=e_pairs._e0_digest(a),
                digest_keys=hashlib.sha256(json.dumps(keys, separators=(",", ":")).encode()).hexdigest())


def check_s_set(js: dict, spec) -> list:
    """S.2: [] when the generated set equals the declared one (status OK, (b) n_b, T, n_a, three digests)."""
    bad = []
    if js["status"] != OK or js["n_b"] != spec.n_b:
        bad.append(f"S set status {js['status']}, (b) {js['n_b']}, declared {spec.n_b}")
    for k, want in (("last_turn", spec.last_turn), ("n_a", spec.n_a), ("digest_e0_b", spec.digest_e0_b),
                    ("digest_e0_a", spec.digest_e0_a), ("digest_keys", spec.digest_keys)):
        if js[k] != want:
            bad.append(f"S set {k}: generated {js[k]!r}, declared {want!r}")
    return bad


def summary(js: dict) -> dict:
    """What block `set` records: counts, T, skips, digests (the declared values, already public in S.2)."""
    return {k: js[k] for k in ("status", "n_b", "n_a", "last_turn", "skipped", "digest_e0_b", "digest_e0_a",
                               "digest_keys")}


def judgement_rows(pops, rc: dict, enc: dict, spec) -> list:
    if spec.smoke:
        raise ValueError("smoke never uses the judgement set (S.3 ②)")
    js = s_set(pops, enc, spec)
    bad = check_s_set(js, spec)
    if bad:
        raise ValueError("; ".join(bad))
    return e_pairs.attach_odours(js["b"] + js["a"], rc, q_pairs.codebook(enc, spec), E.dual_rule(spec.config))
