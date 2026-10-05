"""W's pairs, list only — nothing here runs the engine.
- Pilot (W.3 3, W.9.4): the POOL even pairs (H.4's even declaration, r_pairs.even_rows) that V's block even found
  L_V-testable on axis (b), plus a|4 Rock Slide|Strength, in the even rows' declared order; pilot number j = the
  position in that list. The pilot is records-only and never gates (G.2, G.5).
- Main set (W.2): V's generator (v_pairs._Gen: T's permutation, T's used rows and T's set as used) with V's set rows
  (regenerated from block kc_input's values of V's summary and checked against V's block set) added to the used keys
  and glomerulus keys; turns 306-1985, every (b) and (a) row that passes V's skips in V's order (cap → collision → E1 →
  used → glomerulus duplicate → in-set → POOL-only → kc_input on either engine), no axis cap. Declared order (W.9.6
  P3-15): generator turn, then (b) before (a) within a turn, then the generator's own order; candidate number c = the
  position (W.9.6 P2-11: c < 300). The block `set` fixes its digests, counts, last turn, skip counts and clusters
  before any oracle on it; later stages regenerate it and refuse unless it equals the block."""
from __future__ import annotations

import hashlib
import json
from collections import Counter

from ..agent import e_pairs
from ..agent.e_spec import SPEC as E
from . import q_pairs, t_pairs, v_pairs
from .r_pairs import okey, row_key
from .v_pairs import REASONS, _Gen, _kc_out, sides
from .v_spec import SPEC as V_SPEC

OK = "OK"
SUMMARY_KEYS = ("n_b", "n_a", "n", "first_turn", "last_turn", "skipped", "clusters_b", "clusters_a", "digest_keys",
                "digest_e0_b", "digest_e0_a", "n_odours", "all_off_pool")


def pilot_rows(even_rows: list, v_even_pairs: list, spec) -> list:
    """The pilot pairs in the even rows' declared order: V-even (b) testable on L_V, plus spec.pilot_extra."""
    testable_b = {p["key"] for p in v_even_pairs if p["axis"] == "b" and p["testable"]}
    keep = testable_b | set(spec.pilot_extra)
    rows = [r for r in even_rows if row_key(r) in keep]
    missing = keep - {row_key(r) for r in rows}
    if missing:
        raise ValueError(f"pilot pairs not among the even rows: {sorted(missing)}")
    if len(rows) > spec.n_pilot_max:
        raise ValueError(f"{len(rows)} pilot pairs exceed the seed layout's {spec.n_pilot_max}")
    return rows


def _order(rows: list) -> list:
    return sorted(rows, key=lambda r: (int(r["turn"]), 0 if r["axis"] == "b" else 1))


def w_set(pops, enc: dict, params, kc: dict, v_block: dict, spec) -> dict:
    """The main set and its record (module docstring). kc = V's block kc_input values {"none", "lever"}; v_block = V's
    block set's "set" record. ValueError when V's set does not reproduce or an odour has no KC value."""
    js_v = v_pairs.v_set(pops, enc, params, kc, V_SPEC)
    bad = v_pairs.check_v_set(js_v, v_block)
    if bad:
        raise ValueError("V's set does not reproduce V's block set: " + "; ".join(bad[:3]))
    g = _Gen(pops, enc, params)
    v_rows = js_v["b"] + js_v["a"]
    g.used |= {e_pairs.egrid_key(r) for r in v_rows}
    g.used_g |= {t_pairs._gkey(g.cb, r) for r in v_rows}
    taken, rows, skipped = set(), [], Counter({k: 0 for k in REASONS})
    for j in range(spec.first_turn, spec.last_turn + 1):
        for r in g.rows(j):
            k = e_pairs.egrid_key(r)
            why = g.fixed_reason(r)
            if why is None and k in taken:
                why = "in_set"
            if why is None and g.pool_only(r):
                why = "pool_only"
            if why is None and any(_kc_out(kc, okey(m, o), V_SPEC.valid_band) for m, o in sides(r)):
                why = "kc_input"
            if why is not None:
                skipped[why] += 1
            else:
                taken.add(k)
                rows.append(r)
    rows = _order(rows)
    b, a = [r for r in rows if r["axis"] == "b"], [r for r in rows if r["axis"] == "a"]
    keys = [[r["axis"], r["turn"], sorted([m, list(o)] for m, o in e_pairs.egrid_key(r))] for r in rows]
    clusters = Counter(t_pairs.cluster_of(r, g.pool_sets) for r in rows)
    odours = sorted({s for r in rows for s in sides(r)})
    return dict(rows=rows, n_b=len(b), n_a=len(a), n=len(rows), first_turn=spec.first_turn,
                last_turn=max((r["turn"] for r in rows), default=None), skipped=dict(skipped),
                clusters_b=sorted([*c, n] for c, n in clusters.items() if len(c) == 2),
                clusters_a=sorted([*c, n] for c, n in clusters.items() if len(c) == 1),
                digest_keys=hashlib.sha256(json.dumps(keys, separators=(",", ":")).encode()).hexdigest(),
                digest_e0_b=e_pairs._e0_digest(b), digest_e0_a=e_pairs._e0_digest(a), n_odours=len(odours),
                all_off_pool=all(not g.pool_only(r) for r in rows), keys=[row_key(r) for r in rows])


def set_summary(js: dict) -> dict:
    return {k: js[k] for k in SUMMARY_KEYS}


def check_w_set(js: dict, block: dict) -> list:
    """[] when the regenerated set equals block set's record (every SUMMARY_KEYS value and the key list)."""
    bad = [f"W set {k}: regenerated {js[k]!r}, block set {block.get(k)!r}" for k in SUMMARY_KEYS
           if json.loads(json.dumps(js[k])) != block.get(k)]
    if js["keys"] != block.get("keys"):
        bad.append("W set keys differ from block set")
    return bad


def main_rows(pops, rc: dict, enc: dict, params, kc: dict, v_block: dict, block: dict, spec) -> list:
    """The checked main set's rows with E-grid odours and candidate numbers c (declared order)."""
    if spec.smoke:
        raise ValueError("smoke never uses the main set (W.9.5)")
    js = w_set(pops, enc, params, kc, v_block, spec)
    bad = check_w_set(js, block)
    if bad:
        raise ValueError("; ".join(bad[:3]))
    rows = e_pairs.attach_odours(js["rows"], rc, q_pairs.codebook(enc, V_SPEC), E.dual_rule(V_SPEC.config))
    return [dict(r, c=i) for i, r in enumerate(rows)]


def cluster_labels(rows: list) -> dict:
    return t_pairs.clusters(rows)
