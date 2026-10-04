"""V's judgement set (V.2 as amended by V.9.1 / V.9.5), list only — nothing here runs the engine.
- The generator is T's (T.2 / T.9.1, t_pairs, T's own spec: the widened Gen-1 pool, the 1,986 combos permuted once by
  T's set_rng_seed, turn j's HP, my moves, the alternate, e_pairs._rows_for_turn's rows), walked from turn 0.
- used = T's used rows (e_pairs.used_situations, L turns 0-209 — R's and S's sets too) + T's set (64 rows), whose keys
  and glomerulus keys join used. T's set is reproduced first (t_pairs.t_set / check_t_set, which reproduces S's and R's
  first); a mismatch refuses (ValueError).
- Skips in the declared order (V.9.5 P2-10 / P3-12): ORN cap → glomerulus collision → E1 clash (a side's glomerulus set
  equals a different reachable odour's) → used key → glomerulus-level duplicate → key already in the set → both opponent
  type sets in POOL → kc_input (last): a row is skipped when any of its two E-grid odours has a per-odour KC median
  outside valid_band on either engine (the values of block kc_input only). Then (b) rows until n_b, (a) until n_a.
- candidate_odours (V.3 3, V.9.5 P2-10): the E-grid odours of every row of the whole permutation that passes the skips
  independent of the filling (cap, collision, E1, used, glomerulus duplicate, POOL-only), with each odour's ORN-cap
  arithmetic (max_rate_hz × s × max v against cap_hz).
- set_summary: what block set fixes before any oracle on the set (digests, last turns, odour count, skip counts, the
  cluster table); later stages regenerate the set from block kc_input's values and refuse unless it equals block set.
- judgement_rows / set_odours / set_e0_odours: the checked set's rows with E-grid odours, its distinct E-grid odours
  (the KC band's re-check) and its distinct E0 odours (a record, V.9.5 P2-11); refused for a smoke spec.
- kc_record: the filter's record (V.6): dropped odours per engine, candidate rows using them, and R's, S's and T's set
  rows using them."""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np
from poke_env.data.normalize import to_id_str

from ..agent import e_pairs
from ..agent.e_spec import SPEC as E
from ..agent.encode_grid import glomeruli, odour, reachable
from ..battle.pool import POOL
from . import h4_pairs, l_pairs, odor_real, q_pairs, r_pairs, s_pairs, t_pairs
from .r_pairs import okey, row_key
from .r_spec import SPEC as R_SPEC
from .s_spec import SPEC as S_SPEC
from .t_spec import SPEC as T_SPEC

STOP_SET_SHORT = "STOP_SET_SHORT"
OK = "OK"
REASONS = ("cap", "collision", "e1", "used", "glom_dup", "in_set", "pool_only", "kc_input")
ENGINES = ("none", "lever")
SUMMARY_KEYS = ("status", "n_b", "n_a", "last_turn", "last_turn_b", "last_turn_a", "skipped", "n_opp", "n_combos",
                "n_odours", "e1_clashes", "cap_fails", "all_off_pool", "clusters_b", "clusters_a", "digest_e0_b",
                "digest_e0_a", "digest_keys")


def sides(r) -> tuple:
    return (r["move_x"], tuple(r["opp_x"])), (r["move_y"], tuple(r["opp_y"]))


class _Gen:
    """T's generator state (t_pairs.t_set's construction on T's own spec) plus T's set as used."""

    def __init__(self, pops, enc: dict, params):
        js_t = t_pairs.t_set(pops, enc, T_SPEC, params)
        bad = t_pairs.check_t_set(js_t, T_SPEC)
        if bad:
            raise ValueError("T's judgement set does not reproduce: " + "; ".join(bad))
        self.js_t = js_t
        st, mi, mon, move = h4_pairs.pool_vocabulary()
        self.st, self.mi = st, mi
        self.chan = h4_pairs.e0_channels(pops, mon, move)
        self.pops = pops
        self.rc = {str(t): len(v) for t, v in pops.receptor_types.items()}
        self.cb, self.rule = q_pairs.codebook(enc, T_SPEC), E.dual_rule(T_SPEC.config)
        self.max_rate, self.cap = float(params.max_rate_hz), float(odor_real.cap_hz(params))
        opp = t_pairs.opponents(mon, T_SPEC)
        self.species = dict(st)
        self.species.update({n: ts for n, ts in opp})
        self.names = [n for n, _ in opp]
        self.pool_sets = {tuple(sorted(v)) for v in st.values()}
        ex = l_pairs.excluded_combos()
        self.combos = [(m, n) for m in POOL for n in self.names
                       if to_id_str(n) != to_id_str(m.species) and (m.species, n) not in ex]
        used_rows = list(e_pairs.used_situations(pops, E))
        for t in l_pairs.new_turns(st, mi, E.judge_last_turn + 1, E.l_rng_seed):
            alt = l_pairs.alternate_from(t["opp"], t["me"], t["opp_types"], st, t["turn"])
            used_rows += e_pairs._rows_for_turn(pops, self.chan, t, alt, st, with_a=True)
        used_rows += js_t["b"] + js_t["a"]
        self.used = {e_pairs.egrid_key(r) for r in used_rows}
        self.used_g = {t_pairs._gkey(self.cb, r) for r in used_rows}
        self.reach = {frozenset(glomeruli(self.cb, m, o)): (m, o) for m, o in reachable(st, move)}
        self.order = np.random.default_rng(T_SPEC.set_rng_seed).permutation(len(self.combos))

    def rows(self, j: int) -> list:
        me, on = self.combos[int(self.order[j])]
        st, mi, names, species = self.st, self.mi, self.names, self.species
        my_hp, op_hp = l_pairs.HPS[j % len(l_pairs.HPS)]
        cands = [{"move": x, "type": mi[x][0], "bp": mi[x][1]} for x in me.attacks][:4]
        t = {"turn": j, "pos": j, "me": me.species, "opp": on, "my_types": list(st[me.species]),
             "opp_types": list(species[on]), "my_hp": my_hp, "opp_hp": op_hp, "candidates": cands}
        s0, alt = names.index(on), None
        for k in range(1, len(names)):
            c = names[(s0 + k) % len(names)]
            if to_id_str(c) != to_id_str(me.species) and not (set(species[c]) & set(species[on])):
                alt = c
                break
        if alt is None:
            raise ValueError(f"turn {j}: no type-disjoint alternate opponent in the widened pool")
        return e_pairs._rows_for_turn(self.pops, self.chan, t, alt, species, with_a=True)

    def cap_fails(self, m, o) -> bool:
        return t_pairs._cap_fails(self.rc, self.cb, m, o, self.rule, T_SPEC.strength, self.max_rate, self.cap)

    def e1(self, m, o) -> bool:
        g = t_pairs._glom(self.cb, m, o)
        return g is not None and self.reach.get(g, (m, o)) != (m, o)

    def fixed_reason(self, r) -> str | None:
        """The first skip independent of the filling (cap, collision, e1, used, glom_dup), else None."""
        sd = sides(r)
        if any(self.cap_fails(m, o) for m, o in sd):
            return "cap"
        if any(t_pairs._glom(self.cb, m, o) is None for m, o in sd):
            return "collision"
        if any(self.e1(m, o) for m, o in sd):
            return "e1"
        if e_pairs.egrid_key(r) in self.used:
            return "used"
        if t_pairs._gkey(self.cb, r) in self.used_g:
            return "glom_dup"
        return None

    def pool_only(self, r) -> bool:
        return tuple(r["opp_x"]) in self.pool_sets and tuple(r["opp_y"]) in self.pool_sets

    def odour(self, m, o) -> dict:
        return odour(self.rc, self.cb, m, o, self.rule)


def candidate_odours(pops, enc: dict, params) -> dict:
    """V.3 3 / V.9.5 P2-10: {odours: {id: E-grid odour}, cap: {id: max Hz}, cap_hz, n_rows, turns}."""
    g = _Gen(pops, enc, params)
    ids, n_rows = set(), 0
    for j in range(len(g.order)):
        for r in g.rows(j):
            if g.fixed_reason(r) is None and not g.pool_only(r):
                n_rows += 1
                ids.update(sides(r))
    od = {okey(m, o): g.odour(m, o) for m, o in sorted(ids)}
    return dict(odours=od, cap={k: g.max_rate * float(T_SPEC.strength) * max(v.values()) for k, v in od.items()},
                cap_hz=g.cap, n_rows=n_rows, n_turns=len(g.order))


def _kc_out(kc: dict, oid: str, band) -> bool:
    lo, hi = band
    vals = []
    for e in ENGINES:
        if oid not in kc[e]:
            raise ValueError(f"odour {oid} has no {e} value in block kc_input")
        vals.append(kc[e][oid])
    return any(not lo <= v <= hi for v in vals)


def v_set(pops, enc: dict, params, kc: dict, spec) -> dict:
    """The set and its records (module docstring). kc = {"none": {id: median}, "lever": {id: median}} from block
    kc_input. ValueError when T's set does not reproduce or a candidate odour has no value."""
    g = _Gen(pops, enc, params)
    b, a, taken, skipped, last = [], [], set(), Counter({k: 0 for k in REASONS}), None
    for j in range(spec.first_turn, len(g.order)):
        for r in g.rows(j):
            k = e_pairs.egrid_key(r)
            why = g.fixed_reason(r)
            if why is None and k in taken:
                why = "in_set"
            if why is None and g.pool_only(r):
                why = "pool_only"
            if why is None and any(_kc_out(kc, okey(m, o), spec.valid_band) for m, o in sides(r)):
                why = "kc_input"
            if why is not None:
                skipped[why] += 1
            elif r["axis"] == "b" and len(b) < spec.n_b:
                taken.add(k)
                b.append(r)
            elif r["axis"] == "a" and len(a) < spec.n_a:
                taken.add(k)
                a.append(r)
        last = j
        if len(b) == spec.n_b and len(a) == spec.n_a:
            break
    keys = [[r["axis"], r["turn"], sorted([m, list(o)] for m, o in e_pairs.egrid_key(r))] for r in b + a]
    odours = sorted({s for r in b + a for s in sides(r)})
    clusters = Counter(t_pairs.cluster_of(r, g.pool_sets) for r in b + a)
    return dict(b=b, a=a, n_b=len(b), n_a=len(a), last_turn=last,
                last_turn_b=max((r["turn"] for r in b), default=None),
                last_turn_a=max((r["turn"] for r in a), default=None),
                status=OK if (len(b), len(a)) == (spec.n_b, spec.n_a) else STOP_SET_SHORT, skipped=dict(skipped),
                n_opp=len(g.names), n_combos=len(g.combos), n_odours=len(odours),
                e1_clashes=[okey(m, o) for m, o in odours if g.e1(m, o)],
                cap_fails=[okey(m, o) for m, o in odours if g.cap_fails(m, o)],
                all_off_pool=all(not g.pool_only(r) for r in b + a),
                clusters_b=sorted([*c, n] for c, n in clusters.items() if len(c) == 2),
                clusters_a=sorted([*c, n] for c, n in clusters.items() if len(c) == 1),
                digest_e0_b=e_pairs._e0_digest(b), digest_e0_a=e_pairs._e0_digest(a),
                digest_keys=hashlib.sha256(json.dumps(keys, separators=(",", ":")).encode()).hexdigest(),
                odour_ids=[okey(m, o) for m, o in odours])


def set_summary(js: dict) -> dict:
    return {k: js[k] for k in SUMMARY_KEYS}


def check_v_set(js: dict, block: dict) -> list:
    """[] when the regenerated set equals block set's record (every SUMMARY_KEYS value)."""
    return [f"V set {k}: regenerated {js[k]!r}, block set {block.get(k)!r}" for k in SUMMARY_KEYS
            if json.loads(json.dumps(js[k])) != block.get(k)]


def _checked(pops, enc: dict, params, kc: dict, block: dict, spec) -> dict:
    if spec.smoke:
        raise ValueError("smoke never uses the judgement set (V.3 8)")
    js = v_set(pops, enc, params, kc, spec)
    bad = check_v_set(js, block)
    if js["status"] != OK or bad:
        raise ValueError("; ".join(bad) or f"V set status {js['status']}")
    return js


def judgement_rows(pops, rc: dict, enc: dict, params, kc: dict, block: dict, spec) -> list:
    js = _checked(pops, enc, params, kc, block, spec)
    return e_pairs.attach_odours(js["b"] + js["a"], rc, q_pairs.codebook(enc, spec), E.dual_rule(spec.config))


def set_odours(pops, rc: dict, enc: dict, params, kc: dict, block: dict, spec) -> dict:
    """{odour id: E-grid odour} of the checked set (V.3 6's re-check)."""
    js = _checked(pops, enc, params, kc, block, spec)
    cb, rule = q_pairs.codebook(enc, spec), E.dual_rule(spec.config)
    return {okey(m, o): odour(rc, cb, m, o, rule) for m, o in sorted({s for r in js["b"] + js["a"] for s in sides(r)})}


def set_e0_odours(pops, enc: dict, params, kc: dict, block: dict, spec) -> dict:
    """{"<pair key>|x" / "|y": E0 odour} of the checked set (V.9.5 P2-11's record), distinct by content."""
    js = _checked(pops, enc, params, kc, block, spec)
    out, seen = {}, set()
    for r in js["b"] + js["a"]:
        for s, o in (("x", r["odor_x_e0"]), ("y", r["odor_y_e0"])):
            c = json.dumps(o, sort_keys=True)
            if c not in seen:
                seen.add(c)
                out[f"{row_key(r)}|{s}"] = o
    return out


def kc_record(pops, enc: dict, params, kc: dict, spec) -> dict:
    """V.6's KC input record: odours outside valid_band per engine, candidate rows using a dropped odour, and R's, S's
    and T's judgement-set rows using one (records only)."""
    lo, hi = spec.valid_band
    out = {e: sorted(o for o, v in kc[e].items() if not lo <= v <= hi) for e in ENGINES}
    dropped = set(out["none"]) | set(out["lever"])
    g = _Gen(pops, enc, params)
    n_cand = 0
    for j in range(len(g.order)):
        for r in g.rows(j):
            if g.fixed_reason(r) is None and not g.pool_only(r) and any(okey(m, o) in dropped for m, o in sides(r)):
                n_cand += 1
    js_r = e_pairs.judgement_set(pops, E)
    bad = r_pairs.check_set(js_r, enc, R_SPEC)
    if bad:
        raise ValueError("R's judgement set does not reproduce: " + "; ".join(bad))
    js_s = s_pairs.s_set(pops, enc, S_SPEC)
    sets = dict(R=js_r["b"] + js_r["a"], S=js_s["b"] + js_s["a"], T=g.js_t["b"] + g.js_t["a"])

    def using(rows):
        return sum(any(okey(m, o) in dropped for m, o in sides(r)) for r in rows)
    return dict(outside=out, dropped=sorted(dropped), candidate_rows_dropped=n_cand,
                set_rows={k: dict(n=len(v), using_dropped=using(v)) for k, v in sets.items()})


def cluster_labels(rows: list) -> dict:
    """{pair key: cluster label} (T.9.1's labels: "X 대 Y" on (b), the type set on (a))."""
    return t_pairs.clusters(rows)
