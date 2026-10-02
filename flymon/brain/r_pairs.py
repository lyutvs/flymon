"""R's rows and references (R.1-R.3, R.9.6). Nothing here runs the engine.
- calibration_odours: encoder 4.3's 112 reachable situation odours (e_runner.Runner._odour_table's construction).
- even_rows: H.4's 8 even turns, (a) 18 · (b) 21, E-grid k2-norm odours (e_pairs.even_situations checks the E0
  digest; q_pairs.codebook checks the committed codebook digest). E0 odours stay on each row (odor_x_e0 / odor_y_e0).
- judgement_rows: the encoder's judgement set (L generator turns 64-103) with the three digests, n_a and the last turn
  checked against R's declared values AND block set (R.3); refused for a smoke spec (R.8). Only R's judgement stages
  call it (plan Global Constraints).
- the reproduction gate's references (Reading 4): the encoder ③ activity entry's file name (EMeasurer's key, computed
  only), and P's committed p_arm entries (kind p_arm, block p's measure key)."""
from __future__ import annotations

import json
from pathlib import Path

from ..agent import e_pairs
from ..agent.e_spec import SPEC as E
from ..agent.encode_grid import cap_ok, odour, reachable
from . import q_pairs
from .h4_pairs import pool_vocabulary


def row_key(r) -> str:
    return f"{r['axis']}|{int(r['turn'])}|{r['x']}|{r['y']}"


def okey(m, opp) -> str:
    """e_runner._okey's odour id ("ELECTRIC|DRAGON+FLYING")."""
    return f"{m}|{'+'.join(opp)}"


def calibration_odours(rc: dict, cb, rule: str) -> tuple:
    """({id: odour}, single-type ids, dual-type ids) over every reachable (move type, opponent types)."""
    st, _, _, move = pool_vocabulary()
    odours, single, dual = {}, [], []
    for m, o in reachable(st, move):
        oid = okey(m, o)
        odours[oid] = odour(rc, cb, m, o, rule)
        (single if len(o) == 1 else dual).append(oid)
    return odours, single, dual


def cap_at(rc: dict, cb, rule: str, s: float, max_rate_hz: float, cap_hz: float) -> bool:
    """Encoder 4.3 condition 4 (computed, lever-independent): no glomerulus of any reachable odour above the cap."""
    st, _, _, move = pool_vocabulary()
    return bool(cap_ok(rc, cb, st, move, rule, s, max_rate_hz, cap_hz))


def even_rows(pops, rc: dict, enc: dict, spec) -> list:
    rows = e_pairs.attach_odours(e_pairs.even_situations(pops), rc, q_pairs.codebook(enc, spec),
                                 E.dual_rule(spec.config))
    nb, na = sum(r["axis"] == "b" for r in rows), sum(r["axis"] == "a" for r in rows)
    if (nb, na) != (spec.n_b, spec.n_a_even):
        raise ValueError(f"even rows (b) {nb} · (a) {na}, declared {spec.n_b} · {spec.n_a_even}")
    return rows


def check_set(js: dict, enc: dict, spec) -> list:
    """R.3: the generated set's digests, n_a and last turn equal R's declared values and block set's; [] when so."""
    st = enc["set"]
    bad = []
    for k, want in (("digest_e0_b", spec.digest_e0_b), ("digest_e0_a", spec.digest_e0_a),
                    ("digest_keys", spec.digest_keys), ("n_a", spec.n_a), ("last_turn", spec.last_turn)):
        if js[k] != want or st[k] != want:
            bad.append(f"judgement set {k}: generated {js[k]!r}, block set {st[k]!r}, declared {want!r}")
    if js["status"] != "OK" or len(js["b"]) != spec.n_b or len(js["a"]) != spec.n_a:
        bad.append(f"judgement set status {js['status']}, (b) {len(js['b'])}, (a) {len(js['a'])}")
    return bad


def judgement_rows(pops, rc: dict, enc: dict, spec) -> list:
    """The judgement set's (b) 21 then (a) 32 rows with E-grid k2-norm odours; ValueError on any mismatch."""
    if spec.smoke:
        raise ValueError("smoke never uses the judgement set (R.8)")
    js = e_pairs.judgement_set(pops, E)
    bad = check_set(js, enc, spec)
    if bad:
        raise ValueError("; ".join(bad))
    return e_pairs.attach_odours(js["b"] + js["a"], rc, q_pairs.codebook(enc, spec), E.dual_rule(spec.config))


def cond_odours(row: dict, cond) -> tuple:
    if cond.odour == "egrid":
        return row["odor_x"], row["odor_y"]
    if cond.odour == "e0":
        return row["odor_x_e0"], row["odor_y_e0"]
    raise ValueError(f"unknown odour kind {cond.odour!r}")


def encoder_activity_ref(oid: str, odour_: dict, params, n_kc: int, enc_code: dict, spec) -> str:
    """The basename of encoder ③'s activity entry for this odour (k2-norm, s = spec.strength, the strength seeds):
    EMeasurer's own key over its own inputs — computed, nothing is read or written."""
    from ..agent.e_measure import EMeasurer
    from ..agent.e_store import ECache
    m = EMeasurer(None, ECache(f"{E.raw_dir}/cache", enc_code), params, n_kc, E, guard_params=False)
    return m.cache._path("activity", m._act_inputs(oid, odour_, spec.strength, E.strength_seeds)).name


def p_reference(cache_dir, measure_key: str, wanted: set) -> dict:
    """{(direction, arm, seed): result} of P's committed run: kind p_arm and code_key = block p's measure key; an
    unreadable file is skipped."""
    out = {}
    for f in sorted(Path(cache_dir).glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if d.get("kind") != "p_arm" or d.get("code_key") != measure_key:
            continue
        i = d["inputs"]
        k = (i["direction"], i["arm"], int(i["seed"]))
        if k in wanted:
            out[k] = d["result"]
    return out
