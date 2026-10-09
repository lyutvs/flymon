"""Record-only diagnosis of M4 stage 1's criterion 1 (spec AC.11; no verdict, no rule change). Pure helpers over the
existing R4 records: situation records (per fly, point, pair x side: v / a / p / kc_active / pick / correct), the
learning / evaluation decision logs (candidates, chosen, v, egrid, multipliers) and the reinforce / outcome records.
Nothing here evaluates a brain. Every number is descriptive (AC.5 label; no PASS / FAIL wording, AC.8)."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


def mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def unique_argmax(vals):
    """Index of the unique maximum, or None when the maximum is shared."""
    vals = [float(x) for x in vals]
    top = max(vals)
    return vals.index(top) if vals.count(top) == 1 else None


def margin(v) -> float:
    """Best minus second-best V (0 for a single candidate)."""
    s = sorted((float(x) for x in v), reverse=True)
    return s[0] - s[1] if len(s) > 1 else 0.0


def centred(v) -> np.ndarray:
    a = np.asarray(v, float)
    return a - a.mean()


def chance_switch(pairs) -> list:
    """Per pair, the switch probability of a picker uniform over the candidates on each side independently, 1/k^2.
    An opponent-blind deterministic picker switches with probability 0: the two sides' best moves differ."""
    return [1.0 / len(p["cands"]) ** 2 for p in pairs]


def sides(rec) -> dict:
    return {(s["pair"], s["side"]): s for s in rec["situations"]}


def best_contrast(rec, pairs) -> list:
    """Per pair: d = [V(best1) - V(best2)] at side 0 minus the same at side 1. With only the opponent changing, an
    opponent-conditioned V gives d > 0 (side 0 favours best1 more than side 1 does); identical inputs give E[d] = 0."""
    sd, out = sides(rec), []
    for i, p in enumerate(pairs):
        b1, b2 = p["cands"].index(p["best1"]), p["cands"].index(p["best2"])
        v0, v1 = sd[(i, 0)]["v"], sd[(i, 1)]["v"]
        out.append((v0[b1] - v0[b2]) - (v1[b1] - v1[b2]))
    return out


def pick_changed(rec, n_pairs: int) -> list:
    sd = sides(rec)
    return [int(sd[(i, 0)]["pick"] != sd[(i, 1)]["pick"]) for i in range(n_pairs)]


def side_correct(rec) -> list:
    return [int(s["correct"]) for s in rec["situations"]]


def abs_side_delta(rec, n_pairs: int, key: str, centre: bool = False) -> list:
    """Per pair x candidate |x(side 1) - x(side 0)| for x = v / a / p / kc_active (centred within a side if asked)."""
    sd, out = sides(rec), []
    for i in range(n_pairs):
        x0, x1 = np.asarray(sd[(i, 0)][key], float), np.asarray(sd[(i, 1)][key], float)
        if centre:
            x0, x1 = x0 - x0.mean(), x1 - x1.mean()
        out.extend(np.abs(x1 - x0).tolist())
    return out


def mult_class(m: float) -> str:
    if m == 0:
        return "immune"
    if m < 1:
        return "not_very"
    if m > 1:
        return "super"
    return "neutral"


def rule_picks(powers, mults) -> dict:
    """Unique-argmax picks (None if tied) of three opponent-aware or opponent-blind rules."""
    return {"power": unique_argmax(powers), "mult": unique_argmax(mults),
            "power_x_mult": unique_argmax([p * m for p, m in zip(powers, mults)])}


def rule_agreement(items) -> dict:
    """items: (chosen index, rule_picks dict). Share of items, per rule, where the chosen index equals the rule's
    pick, over items where the rule's pick is defined; and the conflict split (power pick != mult pick)."""
    agree, n = defaultdict(int), defaultdict(int)
    conflict = dict(n=0, power=0, mult=0, other=0)
    for chosen, rp in items:
        for k, r in rp.items():
            if r is not None:
                n[k] += 1
                agree[k] += int(chosen == r)
        if rp["power"] is not None and rp["mult"] is not None and rp["power"] != rp["mult"]:
            conflict["n"] += 1
            conflict["power" if chosen == rp["power"] else "mult" if chosen == rp["mult"] else "other"] += 1
    out = {k: dict(n=n[k], agree=(agree[k] / n[k] if n[k] else None)) for k in ("power", "mult", "power_x_mult")}
    c = conflict
    out["conflict"] = dict(c, power_share=(c["power"] / c["n"] if c["n"] else None),
                           mult_share=(c["mult"] / c["n"] if c["n"] else None))
    return out


def pulse_ms(pulses, reward: str = "PAM08", punish: str = "PPL105") -> tuple:
    r = sum(float(ms) for d, ms in pulses if d == reward)
    p = sum(float(ms) for d, ms in pulses if d == punish)
    return r, p


def join_turns(records) -> list:
    """Fly decisions joined with their outcome and reinforce records by (battle_id, turn)."""
    by = defaultdict(dict)
    for r in records:
        k = r.get("kind")
        if k == "decision" and r.get("decider") == "fly":
            by[(r["battle_id"], r["turn"])]["decision"] = r
        elif k in ("outcome", "reinforce"):
            by[(r["battle_id"], r["turn"])][k] = r
    return [dict(v, battle_id=b, turn=t) for (b, t), v in sorted(by.items()) if "decision" in v]


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 3 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _ranks(x) -> np.ndarray:
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(x.size)
    r[order] = np.arange(x.size, dtype=float)
    for v in np.unique(x):
        m = x == v
        r[m] = r[m].mean()
    return r


def spearman(x, y):
    if len(x) < 3:
        return None
    return pearson(_ranks(x), _ranks(y))


def log2_mult(m: float, floor: float = 0.125) -> float:
    """log2 of a multiplier with immune (0) put at log2(floor) (= -3, one step below 0.25)."""
    return math.log2(max(float(m), floor))


def glom_jaccard(cb, move_type: str, t1, t2) -> float:
    """Glomerulus-set Jaccard between the E-grid odours of one move type against two opponent type sets (FLY's
    odours; FLY-TB's are identical across opponents, Jaccard 1 by construction)."""
    g1 = {g for t in t1 for g in cb.word(move_type, t)}
    g2 = {g for t in t2 for g in cb.word(move_type, t)}
    return len(g1 & g2) / len(g1 | g2)


def weight_summary(w, w0, floor: float = 0.2) -> dict:
    w, w0 = np.asarray(w, float), np.asarray(w0, float)
    ch = w != w0
    nz = ch & (w0 != 0)
    ratio = w[nz] / w0[nz]
    return dict(n_changed=int(ch.sum()), frac_changed=float(ch.mean()),
                frac_floor_of_changed=(float(np.mean(ratio <= floor * (1 + 1e-6))) if ratio.size else None),
                median_ratio_changed=(float(np.median(ratio)) if ratio.size else None),
                frac_up_of_changed=(float(np.mean(ratio > 1)) if ratio.size else None))


def noise_spread(recs, n_pairs: int) -> dict:
    """Records of flies with the same weights and the same odours (only the decision seed differs): the mean, over
    situations x candidates, of the across-fly V standard deviation; the mean modal-pick share; the mean margin."""
    sd_v, modal = [], []
    for i in range(n_pairs):
        for side in (0, 1):
            vs = np.array([sides(r)[(i, side)]["v"] for r in recs], float)
            sd_v.extend(vs.std(axis=0, ddof=1).tolist())
            picks = [sides(r)[(i, side)]["pick"] for r in recs]
            modal.append(max(picks.count(k) for k in set(picks)) / len(picks))
    return dict(n_flies=len(recs), v_sd_across_flies_same_input=mean(sd_v), modal_pick_share=mean(modal),
                margin_mean=mean(margin(s["v"]) for r in recs for s in r["situations"]))
