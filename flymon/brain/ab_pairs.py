"""AB's generator (AB.3 6670–6693), list only — nothing here runs the engine.
- Opponents: poke_env GenData.from_gen(2).pokedex in dex order, numbers 152–251, no `forme`, the base-species rule
  (`baseSpecies` absent or to_id_str(baseSpecies) == its own id) and every type among POOL's 12 (73 species).
- Combos (me ∈ POOL, opp ∈ the 73, opp id ≠ my species id) in POOL order × the list order, permuted once by
  default_rng(20261007); turn j's HP = l_pairs.HPS[j mod 6], my moves = POOL's first four attacks (Move(gen=1)), the
  alternate = G.11 over the Gen-2 list (first k ≥ 1 after the opponent, not my species, type-disjoint), rows =
  e_pairs._rows_for_turn(with_a=True). The loop is re-implemented here (v_pairs._Gen's opponents / combos / order
  are T's); v_pairs._Gen is used for its codebook, channels, used rows and its cap / E1 / POOL-only predicates only.
- used = v_pairs._Gen's used (T's: build_turns, L 0–209, T's set) + V's set (64) + W's main set (249), keys and
  glomerulus keys. Skips in the declared order cap → collision → e1 → used → glom_dup → in_set → pool_only →
  lv_odour → kc_input; a skipped row's key never enters `taken`; no axis cap, turns 0–1167.
- Declared-form tally (plan Reading 2): the AB.3 table counts the walk WITHOUT lv_odour (cap … pool_only) and then
  the rows of that walk carrying an lv odour; the resulting rows are identical to the in-order walk's (a skipped key
  never enters `taken`, and a key's odours are its rows' odours). Both tallies are returned; 0c compares the declared
  form and records the in-order one.
- Self-test (AB.7 0): T's 127 opponents, T's seed 20261004, T's combos (l_pairs.excluded_combos out), used = T's +
  V's set, turns 306–1985, V's KC values as kc_input, lv_odour off → W's main set (b 167 · a 82, last turn 1967,
  digest 65dbf001…).
- lv_sources: the 108 odours of AA's 31 trained pairs, W's 16 pilot pairs, Y's 7 pilot pairs and V's 64 set rows
  (keys only, never values)."""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np
from poke_env.data import GenData
from poke_env.data.normalize import to_id_str

from ..agent import e_pairs
from ..agent.e_spec import SPEC as E
from ..battle.pool import POOL
from . import h4_pairs, l_pairs, q_pairs, t_pairs, v_pairs, w_pairs
from .ab_spec import SPEC as AB
from .r_pairs import okey, row_key
from .t_spec import SPEC as T_SPEC
from .v_spec import SPEC as V_SPEC
from .w_spec import SPEC as W_SPEC


def gen2_opponents(mon_types, s=AB) -> tuple:
    """([(name, types)], excluded-by-type count) — AB.3 상대 목록 (dex order)."""
    lo, hi = s.gen2_num
    out, n_out = [], 0
    for k, v in sorted(GenData.from_gen(2).pokedex.items(), key=lambda kv: kv[1].get("num", 0)):
        if not (lo <= v.get("num", 0) <= hi) or v.get("forme"):
            continue
        if "baseSpecies" in v and to_id_str(v["baseSpecies"]) != k:
            continue
        ts = tuple(t.upper() for t in v["types"])
        if set(ts) <= set(mon_types):
            out.append((v["name"], ts))
        else:
            n_out += 1
    return out, n_out


def ts_key(r) -> str:
    """The opponent type-set group of a row (AB.5: the X side's type set, sorted, "+"-joined; 해석 3)."""
    return "+".join(sorted(r["opp_x"]))


def digest_keys(rows) -> str:
    keys = [[r["axis"], r["turn"], sorted([m, list(o)] for m, o in e_pairs.egrid_key(r))] for r in rows]
    return hashlib.sha256(json.dumps(keys, separators=(",", ":")).encode()).hexdigest()


def declared_order(rows) -> list:
    """AB.3 (W.9.6 P3-15): turn, then (b) before (a), then _rows_for_turn's order (a stable sort)."""
    return sorted(rows, key=lambda r: (int(r["turn"]), 0 if r["axis"] == "b" else 1))


class Walker:
    """The generator over a base v_pairs._Gen (its codebook, channels, used rows and predicates)."""

    def __init__(self, base, opp: list, seed: int, extra_used_rows: list, excluded=frozenset()):
        self.g = base
        self.names = [n for n, _ in opp]
        self.alts = list(self.names)                       # G.11 over the Gen-2 list only (AB.3)
        self.species = dict(base.species)                  # POOL + Gen-1 types (only Gen-2 names are looked up)
        self.species.update({n: ts for n, ts in opp})
        self.combos = [(m, n) for m in POOL for n in self.names
                       if to_id_str(n) != to_id_str(m.species) and (m.species, n) not in excluded]
        self.order = np.random.default_rng(seed).permutation(len(self.combos))
        self.used = set(base.used) | {e_pairs.egrid_key(r) for r in extra_used_rows}
        self.used_g = set(base.used_g) | {t_pairs._gkey(base.cb, r) for r in extra_used_rows}

    def rows(self, j: int) -> list:
        g = self.g
        me, on = self.combos[int(self.order[j])]
        my_hp, op_hp = l_pairs.HPS[j % len(l_pairs.HPS)]
        cands = [{"move": x, "type": g.mi[x][0], "bp": g.mi[x][1]} for x in me.attacks][:4]
        t = {"turn": j, "pos": j, "me": me.species, "opp": on, "my_types": list(g.st[me.species]),
             "opp_types": list(self.species[on]), "my_hp": my_hp, "opp_hp": op_hp, "candidates": cands}
        s0, alt = self.alts.index(on), None
        for k in range(1, len(self.alts)):
            c = self.alts[(s0 + k) % len(self.alts)]
            if to_id_str(c) != to_id_str(me.species) and not (set(self.species[c]) & set(self.species[on])):
                alt = c
                break
        if alt is None:
            raise ValueError(f"turn {j}: no type-disjoint alternate opponent in the Gen-2 list")
        return e_pairs._rows_for_turn(g.pops, g.chan, t, alt, self.species, with_a=True)

    def fixed_reason(self, r):
        g, sd = self.g, v_pairs.sides(r)
        if any(g.cap_fails(m, o) for m, o in sd):
            return "cap"
        if any(t_pairs._glom(g.cb, m, o) is None for m, o in sd):
            return "collision"
        if any(g.e1(m, o) for m, o in sd):
            return "e1"
        if e_pairs.egrid_key(r) in self.used:
            return "used"
        if t_pairs._gkey(g.cb, r) in self.used_g:
            return "glom_dup"
        return None

    def walk(self, first: int, last: int, lv=None, kc=None, band=None, s=AB) -> dict:
        lv = None if lv is None else set(lv)
        taken, rows, sk = set(), [], Counter({k: 0 for k in s.reasons})
        for j in range(first, last + 1):
            for r in self.rows(j):
                k = e_pairs.egrid_key(r)
                why = self.fixed_reason(r)
                if why is None and k in taken:
                    why = "in_set"
                if why is None and self.g.pool_only(r):
                    why = "pool_only"
                if why is None and lv is not None and any(okey(m, o) in lv for m, o in v_pairs.sides(r)):
                    why = "lv_odour"
                if why is None and kc is not None and any(v_pairs._kc_out(kc, okey(m, o), band)
                                                          for m, o in v_pairs.sides(r)):
                    why = "kc_input"
                if why is not None:
                    sk[why] += 1
                else:
                    taken.add(k)
                    rows.append(r)
        rows = declared_order(rows)
        return dict(rows=rows, skipped=dict(sk), digest_keys=digest_keys(rows))


def _odours(rows) -> list:
    return sorted({okey(m, o) for r in rows for m, o in v_pairs.sides(r)})


def _by_axis(rows) -> dict:
    c = Counter(r["axis"] for r in rows)
    return {"a": c.get("a", 0), "b": c.get("b", 0)}


def lv_sources(aa_doc: dict, w_doc: dict, y_doc: dict, w_rows: list, even_rows: list, v_rows: list, s=AB) -> dict:
    """AB.2 / AB.3 `lv_odour` 출처 (keys only): AA candidates `trained` → W main rows; W pilot manifest unit keys (first
    four fields) → even rows; Y pilot admission keys → even rows or V set rows; V's 64 set rows. Returns the sorted
    odour ids, their sha256, the per-source counts and the reasons (empty = as declared)."""
    wk = {row_key(r): r for r in w_rows}
    ek = {row_key(r): r for r in even_rows}
    vk = {row_key(r): r for r in v_rows}
    trained = sorted(x["key"] for x in aa_doc.get("candidates") or [] if x.get("state") == "trained")
    w_pilot = sorted({"|".join(m["key"].split("|")[:4]) for m in (w_doc.get("pilot") or {}).get("manifest") or []})
    y_pilot = sorted((y_doc.get("pilot") or {}).get("admission") or {})
    why = []
    if len(trained) != s.aa_trained or any(k not in wk for k in trained):
        why.append(f"AA trained 키 {len(trained)}개(주 세트 밖 {sum(k not in wk for k in trained)}) ≠ {s.aa_trained}")
    if len(w_pilot) != s.w_pilot_pairs or any(k not in ek for k in w_pilot):
        why.append(f"W 파일럿 쌍 키 {len(w_pilot)}개(짝수 행 밖 {sum(k not in ek for k in w_pilot)}) ≠ {s.w_pilot_pairs}")
    split = dict(even=sum(k in ek for k in y_pilot), v_set=sum(k in vk and k not in ek for k in y_pilot))
    if len(y_pilot) != s.y_pilot_pairs or split != dict(s.y_pilot_split):
        why.append(f"Y 파일럿 키 {len(y_pilot)}개, 풀림 {split} ≠ {dict(s.y_pilot_split)}")
    if len(v_rows) != s.v_set_rows:
        why.append(f"V 세트 {len(v_rows)}행 ≠ {s.v_set_rows}")
    rows = ([wk[k] for k in trained if k in wk] + [ek[k] for k in w_pilot if k in ek]
            + [ek.get(k) or vk[k] for k in y_pilot if k in ek or k in vk] + list(v_rows))
    ids = _odours(rows)
    if len(ids) != s.lv_odour_n:
        why.append(f"lv_odour 냄새 {len(ids)}개 ≠ {s.lv_odour_n}")
    per = dict(aa31=len(_odours([wk[k] for k in trained if k in wk])),
               learned=len(_odours(rows[:len(rows) - len(v_rows)])), v_set=len(_odours(v_rows)))
    return dict(odours=ids, sha256=hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
                n=len(ids), per_source=per, keys=dict(aa31=trained, w_pilot=w_pilot, y_pilot=y_pilot), reasons=why)


def base_used_rows(pops, enc: dict, params, kc_v: dict, v_block: dict) -> tuple:
    """(base _Gen, V's set rows, W's main rows) — V's and W's sets regenerated and checked (ValueError)."""
    base = v_pairs._Gen(pops, enc, params)
    js_v = v_pairs.v_set(pops, enc, params, kc_v, V_SPEC)
    bad = v_pairs.check_v_set(js_v, v_block)
    if bad:
        raise ValueError("V's set does not reproduce V's block set: " + "; ".join(bad[:3]))
    js_w = w_pairs.w_set(pops, enc, params, kc_v, v_block, W_SPEC)
    return base, js_v["b"] + js_v["a"], js_w


def selftest(base, v_rows: list, kc_v: dict, s=AB) -> dict:
    """AB.7 0 생성기 자가 시험 (the Gen-1 configuration through this module's loop)."""
    st, _mi, mon, _mv = h4_pairs.pool_vocabulary()
    w = Walker(base, t_pairs.opponents(mon, T_SPEC), T_SPEC.set_rng_seed, v_rows, l_pairs.excluded_combos())
    got = w.walk(s.selftest_first_turn, len(w.combos) - 1, None, kc_v, V_SPEC.valid_band, s)
    rows = got["rows"]
    return dict(n_b=_by_axis(rows)["b"], n_a=_by_axis(rows)["a"], last_turn=max(r["turn"] for r in rows),
                digest_keys=got["digest_keys"], skipped=got["skipped"],
                ok=(got["digest_keys"] == s.w_digest_keys and _by_axis(rows) == {"a": s.w_n_a, "b": s.w_n_b}
                    and max(r["turn"] for r in rows) == s.w_last_turn))


def gen2_walker(base, v_rows: list, w_rows: list, s=AB) -> Walker:
    _st, _mi, mon, _mv = h4_pairs.pool_vocabulary()
    opp, _n_out = gen2_opponents(mon, s)
    return Walker(base, opp, s.shuffle_seed, list(v_rows) + list(w_rows))


def generate(base, v_rows: list, w_rows: list, lv: list, kc_v: dict, s=AB) -> dict:
    """Order 0c: the KC-pre set (rows in declared order) and every AB.3 declared value, computed from keys only. kc_v
    = V's kc_input values (the cache; used for the V-cache fates of the table, never as the filter here)."""
    st, _mi, mon, _mv = h4_pairs.pool_vocabulary()
    opp, n_out = gen2_opponents(mon, s)
    w = Walker(base, opp, s.shuffle_seed, list(v_rows) + list(w_rows))
    inorder = w.walk(0, len(w.combos) - 1, lv, None, None, s)
    plain = w.walk(0, len(w.combos) - 1, None, None, None, s)
    lvs = set(lv)
    has_lv = [r for r in plain["rows"] if any(okey(m, o) in lvs for m, o in v_pairs.sides(r))]
    rows = inorder["rows"]
    known = set(kc_v["none"]) & set(kc_v["lever"])
    out_known = {o for o in _odours(rows) if o in known and v_pairs._kc_out(kc_v, o, V_SPEC.valid_band)}

    def kc_out_known(r):
        return any(okey(m, o) in known and v_pairs._kc_out(kc_v, okey(m, o), V_SPEC.valid_band)
                   for m, o in v_pairs.sides(r))
    alive = [r for r in rows if not kc_out_known(r)]
    needs = [r for r in alive if any(okey(m, o) not in known for m, o in v_pairs.sides(r))]
    g1_keys = set()
    g1 = Walker(base, t_pairs.opponents(mon, T_SPEC), T_SPEC.set_rng_seed, [], l_pairs.excluded_combos())
    for j in range(len(g1.combos)):
        g1_keys |= {e_pairs.egrid_key(r) for r in g1.rows(j)}
    pool_sets = {tuple(sorted(v)) for v in st.values()}
    new_sets = sorted({"+".join(sorted(ts)) for _n, ts in opp} - {"+".join(t) for t in pool_sets}
                      - {"+".join(sorted(ts)) for _n, ts in t_pairs.opponents(mon, T_SPEC)})
    a_rows = [r for r in alive if r["axis"] == "a"]
    b_rows = [r for r in alive if r["axis"] == "b"]
    return dict(
        rows=rows, n_opp=len(opp), n_opp_type_out=n_out, opponents=[n for n, _ in opp],
        n_new_type_species=sum("+".join(sorted(ts)) in new_sets for _n, ts in opp), new_type_sets=new_sets,
        n_combos=len(w.combos), skipped_inorder=inorder["skipped"],
        skipped_declared={k: plain["skipped"][k] for k in ("used", "in_set", "pool_only", "cap", "collision", "e1",
                                                           "glom_dup")},
        lv=_by_axis(has_lv), lv_kc_out=_by_axis([r for r in has_lv if kc_out_known(r)]),
        pre_kc=_by_axis(rows), turns=[min(r["turn"] for r in rows), max(r["turn"] for r in rows)],
        last_turn={ax: max(r["turn"] for r in rows if r["axis"] == ax) for ax in ("a", "b")},
        odours=dict(all=len(_odours(rows)), v_cache=len(set(_odours(rows)) & known),
                    new=len(set(_odours(rows)) - known)),
        odour_ids=_odours(rows), new_odours=sorted(set(_odours(rows)) - known),
        v_cache_odours=sorted(set(_odours(rows)) & known), v_cache_out=sorted(out_known),
        vcache_drop=_by_axis([r for r in rows if kc_out_known(r)]), post_kc_max=_by_axis(alive),
        needs_kc=_by_axis(needs), vcache_pass=_by_axis([r for r in alive if r not in needs]),
        a_x_groups=dict(Counter(r["x"] for r in a_rows)), a_t_groups=dict(Counter(map(ts_key, a_rows))),
        b_x_groups=len(Counter(r["x"] for r in b_rows)), b_t_groups=len(Counter(map(ts_key, b_rows))),
        gen1_same_keys=sum(e_pairs.egrid_key(r) in g1_keys for r in plain["rows"]),
        digest_keys=inorder["digest_keys"], keys=[row_key(r) for r in rows])


def declared_reasons(gen: dict, s=AB) -> list:
    """0c: every AB.3 declared value against the regenerated one ([] = as declared)."""
    want = dict(n_opp=s.n_opp, n_opp_type_out=s.n_opp_type_out, n_new_type_species=s.n_new_type_species,
                new_type_sets=list(s.new_type_sets), n_combos=s.n_combos, skipped_declared=dict(s.decl_skips),
                lv=dict(s.decl_lv), lv_kc_out=dict(s.decl_lv_kc_out), pre_kc=dict(s.decl_pre_kc),
                turns=list(s.decl_turns), last_turn=dict(s.decl_last_turn), odours=dict(s.decl_odours),
                vcache_drop=dict(s.decl_vcache_drop), post_kc_max=dict(s.decl_post_kc_max),
                needs_kc=dict(s.decl_needs_kc), vcache_pass=dict(s.decl_vcache_pass),
                a_x_groups=dict(s.decl_a_x_groups), a_t_groups=dict(s.decl_a_t_groups),
                b_x_groups=s.decl_b_x_groups, b_t_groups=s.decl_b_t_groups, gen1_same_keys=s.decl_gen1_same_keys)
    why = [f"{k}: 생성 {json.loads(json.dumps(gen[k]))!r} ≠ 선언 {v!r}" for k, v in want.items()
           if json.loads(json.dumps(gen[k])) != json.loads(json.dumps(v))]
    if len(gen["v_cache_out"]) != s.decl_vcache_drop_odours:
        why.append(f"V 캐시 KC 탈락 냄새 {len(gen['v_cache_out'])} ≠ {s.decl_vcache_drop_odours}")
    if gen["v_cache_odours"][:4] != list(s.kc_repro):
        why.append(f"KC 재현 냄새 {gen['v_cache_odours'][:4]} ≠ {list(s.kc_repro)}")
    return why


def kc_merge(kc_v: dict, new_per_odour: dict) -> dict:
    """V's cache values plus the new odours' medians per engine (order 4's filter values)."""
    return {e: dict(kc_v[e], **new_per_odour[e]) for e in v_pairs.ENGINES}


def ab_set(pre_rows: list, kc: dict) -> list:
    """Order 4: the KC-pre rows whose two odours are inside valid_band on both engines, declared order, c = position."""
    keep = [r for r in pre_rows if not any(v_pairs._kc_out(kc, okey(m, o), V_SPEC.valid_band)
                                           for m, o in v_pairs.sides(r))]
    return [dict(r, c=i) for i, r in enumerate(declared_order(keep))]


def set_record(rows: list, kc: dict, pre_rows=None) -> dict:
    """Order 4's block record: digest, counts by axis, both group tables, dropped odours per engine. pre_rows = the
    209 KC-pre rows: the dropped list then holds only the odours those rows carry (kc also holds V's whole cache, whose
    other odours never met this set); None keeps every out-of-band odour of kc (the old form)."""
    lo, hi = V_SPEC.valid_band
    seen = None if pre_rows is None else set(_odours(pre_rows))
    return dict(digest_keys=digest_keys(rows), n=len(rows), by_axis=_by_axis(rows),
                x_groups=sorted(Counter(r["x"] for r in rows).items()),
                t_groups=sorted(Counter(map(ts_key, rows)).items()),
                dropped={e: sorted(o for o, v in kc[e].items() if not lo <= v <= hi and (seen is None or o in seen))
                         for e in v_pairs.ENGINES},
                keys=[row_key(r) for r in rows])


def attach(rows: list, pops, enc: dict) -> list:
    """The rows with their E-grid odours (V's codebook and dual rule — the W / V judgement path)."""
    rc = {str(t): len(v) for t, v in pops.receptor_types.items()}
    return e_pairs.attach_odours(rows, rc, q_pairs.codebook(enc, V_SPEC), E.dual_rule(V_SPEC.config))


def new_odour_inputs(base, ids: list) -> dict:
    """{odour id: E-grid odour} for order 3's activity jobs (v_pairs._Gen.odour; id = 'MOVE|TYPE+TYPE')."""
    out = {}
    for oid in ids:
        m, o = oid.split("|")
        out[oid] = base.odour(m, tuple(o.split("+")))
    return out
