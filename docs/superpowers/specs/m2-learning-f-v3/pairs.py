"""Candidate-pair table for the M2 learning unit test (appendix F v3), derived from the exploration set.

Reads results/m2/candidate_map.json (16 turns x seeds 200-207). For every within-turn candidate pair:
signed naive d' of dV = V(X) - V(Y) (X = earlier candidate in the turn's list; ddof = 1), the number of
channels in which the two candidates differ, and eligibility under two guards:
  loose   MBON13 median >= 5 for both candidates                               (F v1)
  floor   every seed >= K spikes on MBON13 AND on MBON05, both candidates      (F v3: catches near-silent
          dropouts such as t0 Strength MBON05 [38,37,2,2,38,38,41,40] that "no silent seed" let through)
"""
from __future__ import annotations

import itertools, json, sys
from collections import defaultdict

import numpy as np

Z = {"MBON13": (21.8293, 18.1032), "MBON05": (40.6433, 24.0891)}   # frozen, F.3


def pbin(bp: int) -> str:
    return "lt60" if bp < 60 else ("60to89" if bp < 90 else "ge90")


def build(cmap_path: str, meta_path: str, k_floor: int = 5) -> list:
    d = json.load(open(cmap_path)); meta = json.load(open(meta_path))
    mt = np.array(d["mbon_types"]); turns = {t["turn"]: t for t in meta["turns"]}
    by = defaultdict(dict)
    for (tn, sd), r in zip(d["index"], d["results"]):
        by[tn][sd] = np.array([c["mbon_counts"] for c in r["candidates"]], float)
    rows = []
    for tn in sorted(by):
        seeds = sorted(by[tn]); c = turns[tn]["candidates"]
        A = np.stack([by[tn][s][:, mt == "MBON13"].sum(1) for s in seeds])
        P = np.stack([by[tn][s][:, mt == "MBON05"].sum(1) for s in seeds])
        V = (A - Z["MBON13"][0]) / Z["MBON13"][1] - (P - Z["MBON05"][0]) / Z["MBON05"][1]
        for i, j in itertools.combinations(range(len(c)), 2):
            dv = V[:, i] - V[:, j]
            sd = float(dv.std(ddof=1))
            rows.append({
                "turn": tn, "mon": turns[tn]["me"], "x": c[i]["move"], "y": c[j]["move"],
                "n_channels": int((c[i]["type"] != c[j]["type"]) + (pbin(c[i]["bp"]) != pbin(c[j]["bp"]))),
                "d_signed": float(dv.mean() / sd) if sd > 0 else None, "mean": float(dv.mean()), "sd": sd,
                "loose": bool(np.median(A[:, i]) >= 5 and np.median(A[:, j]) >= 5),
                "floor": bool((A[:, [i, j]] >= k_floor).all() and (P[:, [i, j]] >= k_floor).all()),
                "min_A": int(A[:, [i, j]].min()), "min_P": int(P[:, [i, j]].min()),
            })
    return rows


if __name__ == "__main__":
    rows = build("results/m2/candidate_map.json", "results/m2/candidate_map_meta.json")
    json.dump(rows, open(sys.argv[1], "w"), indent=1)
    fl = [r for r in rows if r["floor"]]
    print(f"pairs {len(rows)} | loose {sum(r['loose'] for r in rows)} | floor(all seeds >= 5 on MBON13 and MBON05) {len(fl)}")
    print(f"{'t':>3} {'mon':<11} {'X':<12} {'Y':<12} {'ch':>2} {'d_signed':>9} {'sd':>6} {'minA':>5} {'minP':>5}")
    for r in sorted(fl, key=lambda r: abs(r["d_signed"])):
        print(f"{r['turn']:>3} {r['mon']:<11} {r['x']:<12} {r['y']:<12} {r['n_channels']:>2} {r['d_signed']:>9.2f} {r['sd']:>6.3f} {r['min_A']:>5} {r['min_P']:>5}")
