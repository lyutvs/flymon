"""T's judgement set (T.2 as replaced by T.9.1), list only — nothing here runs the engine.
- opponents: Gen-1 dex numbers 1..opp_num_max without `forme`, every type in the codebook's 12 types, dex order (127).
- t_set: combos (me ∈ POOL, opp ∈ the widened pool, opp ≠ me, not in l_pairs.excluded_combos) in POOL order × the
  widened order, permuted once by default_rng(set_rng_seed); turn j's HP = l_pairs.HPS[j mod 6]; my moves = POOL's first
  four attacks (l_pairs.new_turns' turn shape); the alternate = G.11 over the widened order (first k ≥ 1, not my
  species, type-disjoint with the opponent); rows = e_pairs._rows_for_turn (L's and the encoder's row order and E0
  odours). Each row is checked in the declared order — ORN cap → glomerulus collision → used key
  (e_pairs.used_situations and L turns 0-209, so R's and S's sets too) → glomerulus-level duplicate of any used row →
  key already in the set → both opponent type sets in POOL — and a row failing one is counted under that reason; then
  (b) rows are taken until n_b and (a) rows until n_a (a full axis ignores its rows). Before any of it, S's set (and through it R's) must
  reproduce its declared digests with the same code (s_pairs.s_set / check_s_set).
- ORN cap (T.9.1): every glomerulus value v of the row's E-grid k2-norm odour (encode_grid.odour, the config's dual
  rule) satisfies max_rate_hz × s × v ≤ cap_hz (encode_grid.cap_ok's inequality); an odour that cannot be encoded is
  left to the collision check (plan Reading 6).
- check_t_set: the generated set against T.9.1's declared values (turns, counts, digests, skip counts, the cluster
  table, 55 odours, no E1 clash with the 112 reachable odours, every row off-POOL, every odour under the cap).
- judgement_rows: the checked set's (b) 21 then (a) 43 rows with E-grid odours; refused for a smoke spec. set_odours:
  its 55 distinct E-grid odours (gate ①'s supplement). Only T's judgement stages and the gate-① supplement call them
  (plan Global Constraints)."""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np
from poke_env.data import GenData
from poke_env.data.normalize import to_id_str

from ..agent import e_pairs
from ..agent.e_spec import SPEC as E
from ..agent.encode_grid import glomeruli, odour, reachable
from ..battle.pool import POOL
from . import h4_pairs, l_pairs, odor_real, q_pairs, s_pairs
from .r_pairs import okey, row_key
from .s_spec import SPEC as S_SPEC

STOP_SET_SHORT = "STOP_SET_SHORT"
OK = "OK"
REASONS = ("cap", "collision", "used", "glom_dup", "in_set", "pool_only")


def opponents(mon_types, spec) -> list:
    """[(name, types)] of the widened pool, dex order."""
    dex = GenData.from_gen(1).pokedex
    out = []
    for _, v in sorted(dex.items(), key=lambda kv: kv[1].get("num", 0)):
        if not (1 <= v.get("num", 0) <= spec.opp_num_max) or v.get("forme"):
            continue
        ts = tuple(t.upper() for t in v["types"])
        if set(ts) <= set(mon_types):
            out.append((v["name"], ts))
    return out


def _glom(cb, m, o):
    try:
        return frozenset(glomeruli(cb, m, o))
    except ValueError:
        return None


def _gkey(cb, r) -> frozenset:
    return frozenset({_glom(cb, r["move_x"], tuple(r["opp_x"])), _glom(cb, r["move_y"], tuple(r["opp_y"]))})


def _cap_fails(rc, cb, m, o, rule, s, max_rate_hz, cap_hz) -> bool:
    try:
        od = odour(rc, cb, m, o, rule)
    except ValueError:
        return False                                       # not encodable: the collision check takes it
    return not all(max_rate_hz * s * v <= cap_hz for v in od.values())


def _tag(types, pool_sets) -> str:
    return "+".join(types) + ("" if tuple(types) in pool_sets else "*")


def cluster_of(r, pool_sets) -> tuple:
    """T.9.1's cluster: (b) (X's opponent type set, Y's); (a) the opponent type set (X's = Y's). * = off POOL."""
    x = _tag(tuple(r["opp_x"]), pool_sets)
    return (x, _tag(tuple(r["opp_y"]), pool_sets)) if r["axis"] == "b" else (x,)


def pool_type_sets() -> set:
    st, _, _, _ = h4_pairs.pool_vocabulary()
    return {tuple(sorted(v)) for v in st.values()}


def t_set(pops, enc: dict, spec, params) -> dict:
    """The set and its records (module docstring). ValueError when S's (or R's) set does not reproduce first."""
    js_s = s_pairs.s_set(pops, enc, S_SPEC)
    bad = s_pairs.check_s_set(js_s, S_SPEC)
    if bad:
        raise ValueError("S's judgement set does not reproduce: " + "; ".join(bad))
    st, mi, mon, move = h4_pairs.pool_vocabulary()
    chan = h4_pairs.e0_channels(pops, mon, move)
    rc = {str(t): len(v) for t, v in pops.receptor_types.items()}
    cb, rule = q_pairs.codebook(enc, spec), E.dual_rule(spec.config)
    max_rate, cap = float(params.max_rate_hz), float(odor_real.cap_hz(params))
    opp = opponents(mon, spec)
    species = dict(st)
    species.update({n: ts for n, ts in opp})
    names = [n for n, _ in opp]
    pool_sets = {tuple(sorted(v)) for v in st.values()}
    ex = l_pairs.excluded_combos()
    combos = [(m, n) for m in POOL for n in names if to_id_str(n) != to_id_str(m.species) and (m.species, n) not in ex]
    used_rows = list(e_pairs.used_situations(pops, E))
    for t in l_pairs.new_turns(st, mi, E.judge_last_turn + 1, E.l_rng_seed):
        alt = l_pairs.alternate_from(t["opp"], t["me"], t["opp_types"], st, t["turn"])
        used_rows += e_pairs._rows_for_turn(pops, chan, t, alt, st, with_a=True)
    used = {e_pairs.egrid_key(r) for r in used_rows}
    used_g = {_gkey(cb, r) for r in used_rows}
    order = np.random.default_rng(spec.set_rng_seed).permutation(len(combos))
    b, a, taken, skipped, last = [], [], set(), Counter({k: 0 for k in REASONS}), None
    for j, ix in enumerate(order[:spec.set_last_turn + 1]):
        me, on = combos[int(ix)]
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
        for r in e_pairs._rows_for_turn(pops, chan, t, alt, species, with_a=True):
            sides = ((r["move_x"], tuple(r["opp_x"])), (r["move_y"], tuple(r["opp_y"])))
            k = e_pairs.egrid_key(r)
            if any(_cap_fails(rc, cb, m, o, rule, spec.strength, max_rate, cap) for m, o in sides):
                skipped["cap"] += 1
            elif any(_glom(cb, m, o) is None for m, o in sides):
                skipped["collision"] += 1
            elif k in used:
                skipped["used"] += 1
            elif _gkey(cb, r) in used_g:
                skipped["glom_dup"] += 1
            elif k in taken:
                skipped["in_set"] += 1
            elif tuple(r["opp_x"]) in pool_sets and tuple(r["opp_y"]) in pool_sets:
                skipped["pool_only"] += 1
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
    reach = {frozenset(glomeruli(cb, m, o)): (m, o) for m, o in reachable(st, move)}
    odours = sorted({side for r in b + a for side in ((r["move_x"], tuple(r["opp_x"])),
                                                       (r["move_y"], tuple(r["opp_y"])))})
    e1 = [okey(m, o) for m, o in odours if reach.get(frozenset(glomeruli(cb, m, o)), (m, o)) != (m, o)]
    cap_bad = [okey(m, o) for m, o in odours if _cap_fails(rc, cb, m, o, rule, spec.strength, max_rate, cap)]
    clusters = Counter(cluster_of(r, pool_sets) for r in b + a)
    return dict(b=b, a=a, n_b=len(b), n_a=len(a), last_turn=last,
                last_turn_b=max((r["turn"] for r in b), default=None),
                last_turn_a=max((r["turn"] for r in a), default=None),
                status=OK if (len(b), len(a)) == (spec.n_b, spec.n_a) else STOP_SET_SHORT, skipped=dict(skipped),
                n_opp=len(names), n_combos=len(combos), n_odours=len(odours), e1_clashes=e1, cap_fails=cap_bad,
                all_off_pool=all(not (tuple(r["opp_x"]) in pool_sets and tuple(r["opp_y"]) in pool_sets)
                                 for r in b + a),
                clusters_b=sorted([*c, n] for c, n in clusters.items() if len(c) == 2),
                clusters_a=sorted([*c, n] for c, n in clusters.items() if len(c) == 1),
                digest_e0_b=e_pairs._e0_digest(b), digest_e0_a=e_pairs._e0_digest(a),
                digest_keys=hashlib.sha256(json.dumps(keys, separators=(",", ":")).encode()).hexdigest())


def check_t_set(js: dict, spec) -> list:
    """T.9.1: [] when the generated set equals the declared one."""
    bad = []
    if js["status"] != OK or (js["n_b"], js["n_a"]) != (spec.n_b, spec.n_a):
        bad.append(f"T set status {js['status']}, n_b {js['n_b']} · n_a {js['n_a']}, declared {spec.n_b} · {spec.n_a}")
    for k, want in (("last_turn_b", spec.last_turn_b), ("last_turn", spec.last_turn), ("last_turn_a", spec.last_turn),
                    ("n_opp", spec.n_opp), ("n_combos", spec.n_combos), ("n_odours", spec.n_set_odours),
                    ("digest_e0_b", spec.digest_e0_b), ("digest_e0_a", spec.digest_e0_a),
                    ("digest_keys", spec.digest_keys), ("skipped", dict(spec.skipped_declared)),
                    ("clusters_b", sorted(list(c) for c in spec.clusters_b)),
                    ("clusters_a", sorted(list(c) for c in spec.clusters_a)),
                    ("e1_clashes", []), ("cap_fails", []), ("all_off_pool", True)):
        if js[k] != want:
            bad.append(f"T set {k}: generated {js[k]!r}, declared {want!r}")
    return bad


def summary(js: dict) -> dict:
    """What block `set` records (the declared values, already public in T.9.1)."""
    return {k: js[k] for k in ("status", "n_b", "n_a", "last_turn", "last_turn_b", "last_turn_a", "skipped", "n_opp",
                               "n_combos", "n_odours", "e1_clashes", "cap_fails", "all_off_pool", "clusters_b",
                               "clusters_a", "digest_e0_b", "digest_e0_a", "digest_keys")}


def _checked(pops, enc: dict, spec, params) -> dict:
    if spec.smoke:
        raise ValueError("smoke never uses the judgement set (T.3 4)")
    js = t_set(pops, enc, spec, params)
    bad = check_t_set(js, spec)
    if bad:
        raise ValueError("; ".join(bad))
    return js


def judgement_rows(pops, rc: dict, enc: dict, spec, params) -> list:
    js = _checked(pops, enc, spec, params)
    return e_pairs.attach_odours(js["b"] + js["a"], rc, q_pairs.codebook(enc, spec), E.dual_rule(spec.config))


def set_odours(pops, rc: dict, enc: dict, spec, params) -> dict:
    """{odour id: E-grid odour} of the checked set's distinct (move type, opponent types) — gate ①'s supplement."""
    js = _checked(pops, enc, spec, params)
    cb, rule = q_pairs.codebook(enc, spec), E.dual_rule(spec.config)
    sides = sorted({side for r in js["b"] + js["a"] for side in ((r["move_x"], tuple(r["opp_x"])),
                                                                  (r["move_y"], tuple(r["opp_y"])))})
    return {okey(m, o): odour(rc, cb, m, o, rule) for m, o in sides}


def clusters(rows: list) -> dict:
    """{pair key: cluster label} (T.9.1's table: "X 대 Y" on (b), the type set on (a))."""
    ps = pool_type_sets()
    return {row_key(r): " 대 ".join(cluster_of(r, ps)) for r in rows}
