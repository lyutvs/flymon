# flymon/brain/r_records.py
"""R's per-pair values and records (R.3, R.4, R.6, R.9.5, R.9.7). Nothing here runs the engine or decides a band.
- strip_job: a job result without timing (`wall_s`, any depth) and R's top-level keys — what the reproduction gate
  compares bit for bit with the reference entries (Reading 4).
- raw_check: completeness / uniqueness / probe lengths / edit / per-cell sums / finite values / one CSC sha of one
  condition's oracle results, WITHOUT any pair statistic: the only content of a judgement measurement block (jm:*), so
  nothing band-relevant is visible before the seal (R.9.7, Reading 10).
- cond_summary: raw_check plus h4_formula.pair_stats per pair (m = min(r, −p), testable ⇔ m ≥ 2, naive ⇔ |d_pre| <
  0.5; an undefined or NaN d′ is a reason, ±inf is kept as the encoder's _measure keeps it), arm_aggregate, per-axis
  pass counts and R.6's records.
- transitions / compare: C → L (and E0 → L) tables and the records gate ③ and the judgement write.
- gate1_record: gate ①'s per-odour activity under the lever beside encoder ③'s values (same seeds).
- preread_validity: R.9.7's checks over the three judgement conditions (INVALID: L edges != 2; reasons otherwise),
  including every raw entry's stored `inputs` against the measurer's inputs for its condition × pair × judgement
  seeds (judge_inputs), so a manifest that points one condition at another's files is caught (R.9.7 completeness over
  condition × pair × seed)."""
from __future__ import annotations

import json
import math
from collections import Counter
from statistics import median

import numpy as np

from ..agent import e_rules
from ..agent.e_spec import SPEC as E
from . import d6a
from .h3_store import canonical
from .h4_formula import arm_aggregate, pair_stats
from .h4_rules import z_constants
from .h4_spec import SPEC as H4
from .o_rules import finite
from .r_pairs import row_key

STRIP = ("wall_s",)
JUDGE_BLOCK = "judge"                   # the block name stage_jm passes to RMeasurer.oracle for the judgement set
R_KEYS = ("r", "direction", "x", "y", "point")
CRITERIA = ("testable", "reward_pass", "punish_pass")


def strip_job(x: dict, top=R_KEYS) -> dict:
    def rec(v):
        if isinstance(v, dict):
            return {k: rec(w) for k, w in v.items() if k not in STRIP}
        if isinstance(v, list):
            return [rec(w) for w in v]
        return v
    return {k: rec(v) for k, v in x.items() if k not in top and k not in STRIP}


def cell_sums_ok(res: dict) -> bool:
    r = res.get("r")
    if r is None:
        return False
    pre = res["report"]["pre"]["P"]
    for j, side in enumerate(("x", "y")):
        cells = r["p_cells"][side]
        if len(cells) != len(pre) or any(int(sum(c)) != int(row[j]) for c, row in zip(cells, pre)):
            return False
    return True


def stats_ok(st) -> bool:
    return st is not None and all(isinstance(st[f], (int, float)) and not math.isnan(st[f])
                                  for f in ("d_pre", "r", "p", "m"))


def pair_row(key: str, res: dict, st: dict, spec) -> dict:
    tm = spec.testable_min
    return dict(key=key, axis=key.split("|", 1)[0], d_pre=float(st["d_pre"]), r=float(st["r"]), p=float(st["p"]),
                m=float(st["m"]), testable=bool(st["testable"]), reward_pass=bool(st["r"] >= tm),
                punish_pass=bool(-st["p"] >= tm), naive=bool(abs(st["d_pre"]) < spec.naive_max),
                alpha_reward=float(res["alpha_reward"]), alpha_punish=float(res["alpha_punish"]))


def aggregate(pairs: list, spec) -> dict | None:
    """h4_formula.arm_aggregate over the pair rows; None without a (b) pair (T_b would divide by 0)."""
    if not any(p["axis"] == "b" for p in pairs):
        return None
    return arm_aggregate({(p["axis"], p["key"]): p for p in pairs}, spec.naive_max, spec.bar_b / spec.n_b,
                         spec.f_a_min)


def counts(pairs: list) -> dict:
    out = {}
    for ax in ("b", "a"):
        ps = [p for p in pairs if p["axis"] == ax]
        out[ax] = dict(n=len(ps), **{c: sum(p[c] for p in ps) for c in CRITERIA}, naive=sum(p["naive"] for p in ps))
    return out


def _short(res: dict, n_rep: int, n_act: int) -> bool:
    q, r = res.get("q"), res.get("r")
    try:
        return (q is None or r is None
                or any(len(res["report"][ph][k]) != n_rep for ph in ("pre", "R1", "R2") for k in ("A", "P"))
                or any(len(r["p_cells"][s]) != n_rep for s in ("x", "y"))
                or any(len(res["kc"][s]["frac"]) != n_act or len(res["kc"][s]["max_win"]) != n_act
                       for s in ("x", "y")))
    except (KeyError, TypeError):
        return True


def raw_check(got: list, cond, expected: list, seeds: dict) -> dict:
    n_rep, n_act = len(seeds["report"]), len(seeds["act"])
    reasons = []
    keys = [g["key"] for g in got]
    if len(set(keys)) != len(keys):
        reasons.append("duplicate pair rows")
    exp = set(expected)
    if set(keys) != exp:
        reasons.append(f"pairs differ from the declared list (missing {len(exp - set(keys))}, "
                       f"extra {len(set(keys) - exp)})")
    for g in got:
        res = g["result"]
        if _short(res, n_rep, n_act):
            reasons.append(f"{g['key']}: probes are not the {n_rep} report / {n_act} activity seeds, or a record "
                           f"is missing")
            continue
        q, r = res["q"], res["r"]
        if q.get("edit") != cond.edit:
            reasons.append(f"{g['key']}: ran edit {q.get('edit')}, condition {cond.name} declares {cond.edit}")
        if not cell_sums_ok(res):
            reasons.append(f"{g['key']}: per-cell {r.get('p_type')} counts do not sum to the type count")
        if not (finite(res["report"]) and finite(res["kc"]) and finite(r["p_cells"]) and finite(q["apl_out"])):
            reasons.append(f"{g['key']}: a non-finite value")
    qs = [g["result"]["q"] for g in got if isinstance(g["result"].get("q"), dict)]
    edges = sorted({int(q["edit_edges"]) for q in qs})
    shas = sorted({q["csc_sha256"] for q in qs})
    if len(shas) > 1:
        reasons.append(f"CSC sha256 differs between rows: {shas}")
    return dict(reasons=reasons, edit_edges=edges, csc_sha256=shas[0] if len(shas) == 1 else shas, n_pairs=len(got),
                manifest=[dict(key=g["key"], cache_key=g.get("cache_key"), cache_file=g.get("cache_file"))
                          for g in got])


def saturation(got: list, spec) -> dict | None:
    """R.9.7: every cell × report seed × X/Y of the naive per-cell probe (w0); rate = count × 1000 / read window;
    ceiling = 1000 / (refrac_steps × dt) of one cell; shares at ≥ each sat_frac × ceiling (Reading 11)."""
    rs = [g["result"]["r"] for g in got if isinstance(g["result"].get("r"), dict)]
    if not rs:
        return None
    rates = [c * spec.ms_per_s / r["read_ms"] for r in rs for s in ("x", "y") for seed in r["p_cells"][s] for c in seed]
    caps = sorted({spec.ms_per_s / (r["refrac_steps"] * r["dt"]) for r in rs})
    if len(caps) != 1 or not rates:
        return dict(cap_hz=caps, n=len(rates), why="mixed ceilings or no cell")
    x, cap = np.asarray(rates, float), caps[0]
    return dict(cap_hz=cap, n=int(x.size), median=float(np.median(x)),
                q95=float(np.percentile(x, spec.sat_quantile)), max=float(x.max()),
                share={str(f): float((x >= f * cap).mean()) for f in spec.sat_fracs})


def _rows_under(got: list, z: dict, spec) -> list:
    out = []
    for g in got:
        st = pair_stats(g["result"]["report"], z, spec.testable_min)
        if stats_ok(st):
            out.append(pair_row(g["key"], g["result"], st, spec))
    return out


def z_renorm(got: list, spec, z_h4: dict) -> dict:
    """R.9.7, record only (Reading 12): z re-estimated on this condition's own naive report-seed probes (pairs × seeds
    × X/Y, type sums; ddof = H.4's z_ddof) and what it would give, beside median |d_pre| under block h4's z."""
    ref = [{"types": {"A": float(a), "P": float(p)}} for g in got
           for ra, rp in zip(g["result"]["report"]["pre"]["A"], g["result"]["report"]["pre"]["P"])
           for a, p in zip(ra, rp)]
    try:
        zr = z_constants(ref, {"A": "A", "P": "P"}, H4.z_ddof)
    except ValueError as e:
        return dict(z=None, why=str(e))
    pr, ph = _rows_under(got, zr, spec), _rows_under(got, z_h4, spec)
    agg = aggregate(pr, spec) or {}

    def med(ps):
        return float(median(abs(p["d_pre"]) for p in ps)) if ps else None

    return dict(z={k: [float(v) for v in zr[k]] for k in zr}, testable_b=agg.get("testable_b"), F_a=agg.get("F_a"),
                naive_a=agg.get("naive_a"), abs_d_pre_median=med(pr), abs_d_pre_median_h4=med(ph))


def cond_summary(got: list, cond, spec, z: dict, expected: list, seeds: dict) -> dict:
    rc = raw_check(got, cond, expected, seeds)
    n_rep, n_act = len(seeds["report"]), len(seeds["act"])
    ok = [g for g in got if not _short(g["result"], n_rep, n_act)]
    pairs, und = [], []
    for g in ok:
        st = pair_stats(g["result"]["report"], z, spec.testable_min)
        if not stats_ok(st):
            und.append(g["key"])
            continue
        pairs.append(pair_row(g["key"], g["result"], st, spec))
    reasons = list(rc["reasons"]) + ([f"{len(und)} pair(s) with an undefined d′: {und[:3]}"] if und else [])
    fr = [x for g in ok for s in ("x", "y") for x in g["result"]["kc"][s]["frac"]]
    wins = [w for g in ok for s in ("x", "y") for w in g["result"]["kc"][s]["max_win"]]
    apl = [a for g in ok for s in ("x", "y") for a in g["result"]["q"]["apl_out"][s]]
    a_naive = [c for g in ok for row in g["result"]["report"]["pre"]["A"] for c in row]
    # R.6 "순진 P_X·A_X 중앙값": X's column only (column 0 = odor_x: the oracle probes [odor_x, odor_y], h4_formula.dv
    # takes V[:, 0] − V[:, 1]), every report seed of every pair, naive weights (w0).
    a_x = [row[0] for g in ok for row in g["result"]["report"]["pre"]["A"]]
    p_x = [row[0] for g in ok for row in g["result"]["report"]["pre"]["P"]]
    return dict(rc, reasons=reasons, pairs=pairs, aggregate=aggregate(pairs, spec), counts=counts(pairs),
                kc_median=float(np.median(fr)) if fr else None,
                d6a_over_share=float(np.mean([w >= d6a.OVER_SPIKES for w in wins])) if wins else None,
                d6a_condition=d6a.CONDITION, apl_out_median=float(np.median(apl)) if apl else None,
                naive_A_median=float(np.median(a_naive)) if a_naive else None,
                naive_A_X_median=float(np.median(a_x)) if a_x else None,
                naive_P_X_median=float(np.median(p_x)) if p_x else None,
                alpha_reward=dict(Counter(str(p["alpha_reward"]) for p in pairs)),
                alpha_punish=dict(Counter(str(p["alpha_punish"]) for p in pairs)),
                saturation=saturation(ok, spec), z_renorm=z_renorm(ok, spec, z),
                condition=dict(name=cond.name, edit=cond.edit, odour=cond.odour, strength=cond.strength))


def transitions(to: list, frm: list) -> dict:
    """{axis: {criterion: {"pp","pf","fp","ff"}, "minus_p": {key: [−p from, −p to]}}}: first letter = `frm` (C or E0)
    passes, second = `to` (L) passes (R.4, R.6, R.9.5)."""
    tm, fm = {p["key"]: p for p in to}, {p["key"]: p for p in frm}
    out = {}
    for ax in ("b", "a"):
        ks = sorted(k for k in fm if k in tm and fm[k]["axis"] == ax)
        t = {}
        for c in CRITERIA:
            cell = {"pp": 0, "pf": 0, "fp": 0, "ff": 0}
            for k in ks:
                cell[("p" if fm[k][c] else "f") + ("p" if tm[k][c] else "f")] += 1
            t[c] = cell
        t["minus_p"] = {k: [-fm[k]["p"], -tm[k]["p"]] for k in ks}
        out[ax] = t
    return out


KEEP = ("aggregate", "counts", "kc_median", "d6a_over_share", "apl_out_median", "naive_A_median", "naive_A_X_median",
        "naive_P_X_median", "alpha_reward", "alpha_punish", "saturation", "z_renorm", "edit_edges", "csc_sha256",
        "reasons")


def compare(L: dict, C: dict, E0: dict | None, spec) -> dict:
    names = spec.cond_names
    conds = {names[0]: L, names[1]: C, **({names[2]: E0} if E0 else {})}
    return dict(transitions_C_to_L=transitions(L["pairs"], C["pairs"]),
                transitions_E0_to_L=transitions(L["pairs"], E0["pairs"]) if E0 else None,
                C_F_a=(C["aggregate"] or {}).get("F_a"), C_naive_a=(C["aggregate"] or {}).get("naive_a"),
                conditions={n: {k: s.get(k) for k in KEEP} for n, s in conds.items()})


def gate1_record(act: dict, single: list, dual: list, enc_per: dict, spec) -> dict:
    """R.9.2: the per-odour activity (median over seeds, e_rules.odour_activity) under the lever, the 112-odour median,
    the ORIGINAL qualification (encoder 4.3 conditions 1-3, e_rules.strength_ok on E's numbers) as a record, and the
    paired comparison with encoder ③'s per-odour values on the same seeds."""
    per = e_rules.odour_activity({o: act[o]["frac"] for o in act})
    sv, dv_ = [per[o] for o in single], [per[o] for o in dual]
    wins = [int(w) for v in act.values() for w in v["max_win"]]
    enc = {o: float(enc_per[o]) for o in per if o in enc_per}
    return dict(median=float(median(sv + dv_)), n_odours=len(per),
                n_seeds=sorted({len(v["frac"]) for v in act.values()}), original=e_rules.strength_ok(sv, dv_, E),
                encoder_median=float(median(enc.values())) if enc else None, per_odour=per, encoder_per_odour=enc,
                missing_in_encoder=sorted(set(per) - set(enc)), decreased=sum(per[o] < enc[o] for o in enc),
                ratio_median=(float(median(per[o] / enc[o] for o in enc if enc[o] > 0)) if enc else None),
                d6a_over=sum(w >= d6a.OVER_SPIKES for w in wins), d6a_presentations=len(wins),
                d6a_condition=d6a.CONDITION, edit_edges=sorted({int(e) for v in act.values() for e in v["edit_edges"]}),
                csc_sha256=sorted({s for v in act.values() for s in v["csc_sha256"]}))


def judge_inputs(measurer, rows: list, spec, block: str = JUDGE_BLOCK) -> dict:
    """{condition: {pair key: RMeasurer.inputs(row, cond, block, judge_seeds)}} — what every judgement raw entry must
    have stored as its `inputs` (preread_validity's `want`)."""
    seeds = spec.judge_seeds()
    return {n: {row_key(r): measurer.inputs(r, spec.cond(n), block, seeds) for r in rows} for n in spec.cond_names}


def _stored_inputs(path):
    try:
        with open(path) as f:
            return json.load(f)["inputs"]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def inputs_reasons(name: str, got: list, want: dict) -> list:
    """Each raw entry's stored `inputs` (read from its cache file) must canonical-JSON equal want[pair key]."""
    bad = []
    for g in got:
        exp, have = want.get(g["key"]), _stored_inputs(g.get("cache_file"))
        if exp is None or have is None or canonical(have) != canonical(exp):
            bad.append(g["key"])
    return [f"{name}: {len(bad)} raw entr(y/ies) whose stored inputs are not condition {name}'s for that pair and the "
            f"judgement seeds: {bad[:3]}"] if bad else []


def preread_validity(blocks: dict, raws: dict, spec, z: dict, expected: list, code_key: str, repro_sha: str,
                     want: dict) -> dict:
    """R.9.7 (Reading 10) over the judgement conditions: block code key and seeds, raw_check, defined d′ per pair,
    every raw entry's stored inputs = want[condition][pair] (judge_inputs), L exactly lever_edges CSC edges (else
    `invalid`), C and E0 none, L ≠ C sha, C = E0 = the repro's unedited sha."""
    reasons, invalid, checks = [], [], {}
    seeds = spec.judge_seeds()
    n_rep, n_act = len(seeds["report"]), len(seeds["act"])
    for n in spec.cond_names:
        cond, b = spec.cond(n), blocks[n]
        if b.get("code_key") != code_key:
            reasons.append(f"{n}: block code key {b.get('code_key')} is not the current {code_key}")
        if b.get("seeds") != seeds:
            reasons.append(f"{n}: block seeds are not the judgement oracle seeds")
        rc = raw_check(raws[n], cond, expected, seeds)
        reasons += [f"{n}: {x}" for x in rc["reasons"]]
        und = [g["key"] for g in raws[n] if not _short(g["result"], n_rep, n_act)
               and not stats_ok(pair_stats(g["result"]["report"], z, spec.testable_min))]
        if und:
            reasons.append(f"{n}: {len(und)} pair(s) with an undefined d′")
        reasons += inputs_reasons(n, raws[n], want.get(n, {}))
        checks[n] = dict(edit_edges=rc["edit_edges"], csc_sha256=rc["csc_sha256"], n_pairs=rc["n_pairs"])
    nl, nc, ne = spec.cond_names
    if checks[nl]["edit_edges"] != [spec.lever_edges]:
        invalid.append(f"L changed {checks[nl]['edit_edges']} CSC edges, declared exactly {spec.lever_edges} (R.6)")
    for n in (nc, ne):
        if checks[n]["edit_edges"] != [0]:
            reasons.append(f"{n} changed {checks[n]['edit_edges']} CSC edges, declared none")
    if checks[nl]["csc_sha256"] == checks[nc]["csc_sha256"]:
        reasons.append("L and C ran on the same CSC weights")
    if not (checks[nc]["csc_sha256"] == checks[ne]["csc_sha256"] == repro_sha):
        reasons.append(f"C / E0 CSC sha256 {checks[nc]['csc_sha256']} / {checks[ne]['csc_sha256']} is not the "
                       f"unedited engine's {repro_sha}")
    return dict(reasons=reasons, invalid=invalid, checks=checks)
