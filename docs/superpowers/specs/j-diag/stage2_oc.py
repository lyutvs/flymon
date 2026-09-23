"""Spec J.11.4-6: the operating characteristic of stage 2's reading, computed before stage 2 runs (a record shown to the
user; the rule changes only by a dated amendment before the run).

The reading is j_rules.stage2_reading itself. The generative model keeps H.4's C3 record's structure (block "h4" of
results/summary/m0d.json, committed): its 8 even turns with their (b) pairs and their (a) pairs, and each (a) pair's
naive balance (|d_pre| < 0.5). For a true testable share q, a draw resamples the 8 turns with replacement, gives each
turn the testable probability logistic(logit(q) + u_turn) with u_turn the turn's effect in the C3 record (add-0.5
smoothing), and draws every (b) and (a) pair of the turn testable with that probability (the same q on both axes).
The binomial reference ignores the turns: testable (b) ~ Bin(21, q), F_a ~ Bin(naive (a) count, q).
    uv run python docs/superpowers/specs/j-diag/stage2_oc.py [--draws 10000] [--out results/j/diag/stage2_oc.json]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from flymon.brain.j_rules import SELECTED, stage2_reading
from flymon.brain.j_spec import SPEC

QS = (0.33, 0.5, 0.6, 0.7)
SEED = 20260923


def logit(p):
    return math.log(p / (1 - p))


def turns_of(pairs: list) -> dict:
    out = {}
    for p in pairs:
        t = out.setdefault(int(p["turn"]), dict(b=0, b_testable=0, a_naive=[]))
        if p["axis"] == "b":
            t["b"] += 1
            t["b_testable"] += bool(p["testable"])
        else:
            t["a_naive"].append(bool(abs(p["d_pre"]) < SPEC.h4.naive_max))
    return out


def simulate(turns: dict, q: float, draws: int, rng) -> dict:
    keys = sorted(turns)
    nb = sum(t["b"] for t in turns.values())
    overall = (sum(t["b_testable"] for t in turns.values()) + 0.5) / (nb + 1.0)
    u = {k: logit((t["b_testable"] + 0.5) / (t["b"] + 1.0)) - logit(overall) for k, t in turns.items()}
    n_sel = 0
    for _ in range(draws):
        pick = rng.choice(keys, size=len(keys), replace=True)
        tb = n_b = fa = naive = 0
        for k in pick:
            t = turns[k]
            p = 1.0 / (1.0 + math.exp(-(logit(q) + u[k])))
            tb += int(rng.binomial(t["b"], p)) if t["b"] else 0
            n_b += t["b"]
            for nv in t["a_naive"]:
                hit = rng.random() < p
                fa += int(nv and hit)
                naive += int(nv)
        if n_b == 0:
            continue
        agg = dict(T_b=tb / n_b, testable_b=tb, n_b=n_b, F_a=fa, naive_a=naive)
        n_sel += stage2_reading(agg, SPEC.h4)["outcome"] == SELECTED
    return dict(p_selected=n_sel / draws, p_b=1 - n_sel / draws)


def binomial(nb: int, na: int, q: float, draws: int, rng) -> float:
    n_sel = 0
    for _ in range(draws):
        tb, fa = int(rng.binomial(nb, q)), int(rng.binomial(na, q))
        agg = dict(T_b=tb / nb, testable_b=tb, n_b=nb, F_a=fa, naive_a=na)
        n_sel += stage2_reading(agg, SPEC.h4)["outcome"] == SELECTED
    return n_sel / draws


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", default="results/summary/m0d.json")
    ap.add_argument("--draws", type=int, default=10_000)
    ap.add_argument("--out", default="results/j/diag/stage2_oc.json")
    a = ap.parse_args(argv)
    rec = json.loads(Path(a.summary).read_text())["h4"]["h4"]["combos"]["C3"]["oracle"]
    turns = turns_of(rec["pairs"])
    nb = sum(t["b"] for t in turns.values())
    na = sum(sum(t["a_naive"]) for t in turns.values())
    rng = np.random.default_rng(SEED)
    rows = []
    for q in QS:
        sim = simulate(turns, q, a.draws, rng)
        rows.append(dict(q=q, turn_bootstrap=sim, binomial_p_selected=binomial(nb, na, q, a.draws, rng)))
        print(f"q {q:.2f}: P(SELECTED) turn bootstrap {sim['p_selected']:.3f}, binomial {rows[-1]['binomial_p_selected']:.3f}")
    res = dict(what="spec J.11.4-6 operating characteristic of stage 2's reading", seed=SEED, draws=a.draws,
               c3_record=dict(n_b=nb, testable_b=rec["aggregate"]["testable_b"], naive_a=na, F_a=rec["aggregate"]["F_a"],
                              turns={str(k): dict(b=t["b"], b_testable=t["b_testable"], a_naive=sum(t["a_naive"]),
                                                   a=len(t["a_naive"])) for k, t in sorted(turns.items())}),
               bar=dict(t_b_min=SPEC.h4.t_b_min, f_a_min=SPEC.h4.f_a_min), rows=rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
