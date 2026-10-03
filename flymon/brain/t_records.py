"""T's numbers that no R / S module returns — records and the values the z gates read; no engine.
- z_side (T.1, T.9.4): one engine variant's reference-set z (h4_rules.z_constants over the readout types' summed
  counts, ddof 0; None with the reason when an SD is 0) and H.4's readout guard per readout type
  (h3_rules.mbon_type_stats: median(read − same-seed rest), zero share) with the edges, CSC and counts.
- z_ratios: σ_lever / σ_h4 per readout (T.9.4's record).
- gate1s_record (T.9.1): per-odour KC activity medians (e_rules.odour_activity) under the lever and unedited, on the
  T set's odours, with edges, CSC and seed counts.
- p_zlever (T.9.3, record only): P_L's p_judge read on z_lever and ℓ_L(z_lever) / ℓ_C(h4 z) per direction.
- clusters (T.6): testable / reward / punish passes per T.9.1 cluster and condition.
- alpha_fixed (T.9.2, record only — "α 고정 민감도"): L's judgement raw (α chosen under z_lever) read on h4 z: n, F_a,
  naive_a, G_fail_S against C and the band that reading would give.
- ratio / gate2_oc: s_records' (gate ② on h4 z, T.9.3)."""
from __future__ import annotations

from ..agent import e_rules
from . import r_records, s_records, s_rules
from .h3_rules import mbon_type_stats
from .h4_rules import z_constants
from .p_rules import p_judge

ratio = s_records.ratio
gate2_oc = s_records.gate2_oc


def z_side(ref: list, rest: list, readout: dict, spec) -> dict:
    rb = {int(r["seed"]): r for r in rest}
    guard = {t: mbon_type_stats(ref, rb, t, spec.z_guard_med_min, spec.z_guard_zero_max) for t in readout.values()}
    try:
        z, why = z_constants(ref, readout, spec.z_ddof), None
    except ValueError as e:
        z, why = None, str(e)
    zero_sd = [t for t in readout.values() if len({r["types"][t] for r in ref}) < 2]
    return dict(z=None if z is None else {k: [float(v[0]), float(v[1])] for k, v in z.items()}, why=why, guard=guard,
                zero_sd=zero_sd, n_ref=len(ref), n_rest=len(rest),
                edit_edges=sorted({int(r["edit_edges"]) for r in ref + rest}),
                csc_sha256=sorted({r["csc_sha256"] for r in ref + rest}))


def z_ratios(z_lever: dict, z_h4: dict) -> dict:
    return {k: float(z_lever[k][1]) / float(z_h4[k][1]) for k in z_h4}


def gate1s_record(act_l: dict, act_n: dict, spec) -> dict:
    per = e_rules.odour_activity({o: act_l[o]["frac"] for o in act_l})
    per_n = e_rules.odour_activity({o: act_n[o]["frac"] for o in act_n})
    lo, hi = spec.valid_band
    return dict(per_odour=per, per_odour_none=per_n, n_odours=len(per),
                n_seeds=sorted({len(v["frac"]) for v in act_l.values()}),
                outside=sorted(o for o, v in per.items() if not lo <= v <= hi),
                outside_none=sorted(o for o, v in per_n.items() if not lo <= v <= hi),
                min=min(per.values()), max=max(per.values()),
                edit_edges=sorted({int(e) for v in act_l.values() for e in v["edit_edges"]}),
                edit_edges_none=sorted({int(e) for v in act_n.values() for e in v["edit_edges"]}),
                csc_sha256=sorted({s for v in act_l.values() for s in v["csc_sha256"]}),
                csc_sha256_none=sorted({s for v in act_n.values() for s in v["csc_sha256"]}))


def p_zlever(rows_l: list, z_lever: dict, c1: float, res_c: dict, spec) -> dict:
    res = p_judge(rows_l, z_lever, c1, spec.p)
    d = res.get("directions") or {}
    dc = res_c.get("directions") or {}
    return dict(label=res.get("label"), outcome=res.get("outcome"),
                ell={k: v.get("ell") for k, v in d.items()},
                ratio_to_C_h4={k: (d[k]["ell"] / dc[k]["ell"] if k in dc and dc[k].get("ell") else None) for k in d})


def clusters(pairs: dict, labels: dict) -> dict:
    """{cluster: {condition: {n, testable, reward_pass, punish_pass}}} over the judgement pairs (T.6)."""
    out = {}
    for cond, ps in pairs.items():
        for p in ps:
            cl = labels[p["key"]]
            c = out.setdefault(cl, {}).setdefault(cond, dict(n=0, testable=0, reward_pass=0, punish_pass=0))
            c["n"] += 1
            for k in ("testable", "reward_pass", "punish_pass"):
                c[k] += int(bool(p[k]))
    return out


def alpha_fixed(raws_l: list, C: dict, keys: list, seeds: dict, z_h4: dict, spec) -> dict:
    """L's raw (α chosen under z_lever) on h4 z, against C (h4 z) — a record, never the judgement (T.9.2)."""
    L = r_records.cond_summary(raws_l, spec.cond(spec.cond_names[0]), spec, z_h4, keys, seeds)
    aL, aC = L["aggregate"] or {}, C["aggregate"] or {}
    gf = s_rules.g_fail_s(L["pairs"], C["pairs"], spec)
    band = None
    if L["aggregate"] and C["aggregate"]:
        band = s_rules.read_band(aL["testable_b"], aC["testable_b"], aL["F_a"], gf["g_fail"], aL["n_b"], aL["n_a"],
                                 aL["naive_a"], spec)["band"]
    return dict(n=aL.get("testable_b"), c=aC.get("testable_b"), F_a=aL.get("F_a"), naive_a=aL.get("naive_a"),
                g_fail=gf, band=band, note="α 고정 민감도: z_lever가 고른 α의 원자료를 h4 z로 읽은 값 — 판정 아님 (T.9.2)")
