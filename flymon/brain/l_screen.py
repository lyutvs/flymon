# flymon/brain/l_screen.py
"""Spec L.11.2, stage 1 as pure functions: the naive features of a (b) pair, the candidate rules (G alone, G and one
threshold on S / f2 / f3), the choice under a coverage floor, the leave-one-turn-out gate and its result codes. Nothing
here runs the engine."""
from __future__ import annotations

import numpy as np

from .k_metrics import pair_metrics

SCREEN_GO = "SCREEN_GO"
SCREEN_FEW = "SCREEN_FEW"                 # LOTO passes < gate_min_passes: precision cannot be judged (user decides)
SCREEN_IMPRECISE = "SCREEN_IMPRECISE"     # LOTO precision < gate_min_precision: closes the naive-screen claim
SCREEN_NO_RULE = "SCREEN_NO_RULE"         # no rule meets the coverage floor on all 41 pairs (user decides)


def features(fx, fy, pre: dict, w13, w05, spec) -> dict:
    """G on every select seed's [x, y]; S from K's metric; f2 = min of MBON13 medians; f3 = Jaccard of the KCs active on
    any activity seed (fx, fy are per-KC firing fractions over the 8 seeds, so > 0 is the union)."""
    fx, fy = np.asarray(fx, float), np.asarray(fy, float)
    g = all(min(xy) >= spec.guard_min for t in spec.guard_types for xy in pre[t])
    x13 = [xy[0] for xy in pre["MBON13"]]; y13 = [xy[1] for xy in pre["MBON13"]]
    on_x, on_y = fx > 0, fy > 0
    return dict(G=bool(g), S=float(pair_metrics(fx, fy, w13, w05)["S"]),
                f2=float(min(np.median(x13), np.median(y13))),
                f3=float((on_x & on_y).sum() / max(int((on_x | on_y).sum()), 1)))


def passes(rule, feat: dict) -> bool:
    if rule is None or not feat["G"]:
        return False
    if rule["family"] == "G":
        return True
    v = feat[rule["family"]]
    return v > rule["t"] if rule["op"] == ">" else v < rule["t"]


def rules_for(train: list, spec) -> list:
    out = [{"family": "G"}]
    for fam in spec.family_order[1:]:
        op = dict(spec.directions)[fam]
        vals = sorted({p["feat"][fam] for p in train if p["feat"]["G"]}, reverse=(op == "<"))
        ts = [(a + b) / 2.0 for a, b in zip(vals, vals[1:])]
        out += [{"family": fam, "op": op, "t": float(t)} for t in ts]
    return out


def _score(rule, train):
    hit = [p["testable"] for p in train if passes(rule, p["feat"])]
    return len(hit), sum(hit)


def choose(train: list, spec) -> dict | None:
    best, best_key = None, None
    fam_rank = {f: i for i, f in enumerate(spec.family_order)}
    for i, r in enumerate(rules_for(train, spec)):
        n, k = _score(r, train)
        cov = n / len(train) if train else 0.0
        if n == 0 or cov < spec.cov_min_stage1:
            continue
        key = (k / n, cov, -fam_rank[r["family"]], -i)
        if best_key is None or key > best_key:
            best, best_key = dict(rule=r, precision=k / n, coverage=cov, n_pass=n), key
    return best


def loto(pairs: list, spec) -> dict:
    folds, n_pass, n_hit = [], 0, 0
    for t in sorted({p["turn"] for p in pairs}):
        c = choose([p for p in pairs if p["turn"] != t], spec)
        rule = None if c is None else c["rule"]
        held = [p for p in pairs if p["turn"] == t]
        pred = [passes(rule, p["feat"]) for p in held]
        n_pass += sum(pred); n_hit += sum(pr and p["testable"] for pr, p in zip(pred, held))
        folds.append(dict(turn=t, rule=rule, predicted=pred))
    return dict(folds=folds, n_pass=n_pass, n_hit=n_hit, precision=(n_hit / n_pass) if n_pass else None)


def auc(values, labels, higher_passes: bool) -> float | None:
    """P(a testable pair scores better than a non-testable one), ties 1/2 (Mann-Whitney)."""
    pos = [v for v, l in zip(values, labels) if l]; neg = [v for v, l in zip(values, labels) if not l]
    if not pos or not neg:
        return None
    s = sum((1.0 if (a > b) == higher_passes and a != b else 0.5 if a == b else 0.0) for a in pos for b in neg)
    return s / (len(pos) * len(neg))


def gate(pairs: list, spec) -> dict:
    final = choose(pairs, spec)
    lo = loto(pairs, spec)
    if final is None:
        outcome = SCREEN_NO_RULE
    elif lo["n_pass"] < spec.gate_min_passes or lo["precision"] is None:   # 0 passes: precision None, not 0
        outcome = SCREEN_FEW
    elif lo["precision"] < spec.gate_min_precision:
        outcome = SCREEN_IMPRECISE
    else:
        outcome = SCREEN_GO
    lab = [p["testable"] for p in pairs]
    rec = dict(auc={f: auc([p["feat"][f] for p in pairs], lab, dict(spec.directions)[f] == ">")
                    for f in spec.family_order[1:]},
               by_set={s: _by_set(pairs, final, s) for s in ("even", "odd")},
               loto_rule_counts=_rule_counts(lo["folds"]),
               species_folds="identical to the turn folds: turn i's species is POOL[i] for i < 16 (plan reading 6)")
    return dict(outcome=outcome, final=final, loto=lo, records=rec)


def _by_set(pairs, final, s):
    sub = [p for p in pairs if p["set"] == s]
    rule = None if final is None else final["rule"]
    pas = [p for p in sub if passes(rule, p["feat"])]
    pos = [p for p in sub if p["testable"]]
    return dict(n=len(sub), n_pass=len(pas), precision=(sum(p["testable"] for p in pas) / len(pas)) if pas else None,
                recall=(sum(passes(rule, p["feat"]) for p in pos) / len(pos)) if pos else None)


def _rule_counts(folds):
    out = {}
    for f in folds:
        r = f["rule"]
        k = "none" if r is None else (r["family"] if r["family"] == "G" else f"{r['family']}{r['op']}{r['t']:.6g}")
        out[k] = out.get(k, 0) + 1
    return out
