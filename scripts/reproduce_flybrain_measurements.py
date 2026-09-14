#!/usr/bin/env python3
"""Reproduce the three flybrain measurements that gate the M0 engine.

  sparsity      Kenyon-cell sparsity/overlap for a designed odour pair + MBON baseline
  conditioning  paired-seed olfactory conditioning with reversal (Task 9)
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.engine_cpu import Engine
from flymon.brain.measure import chance_jaccard, jaccard, kc_sparsity, mbon_baseline
from flymon.brain.stimuli import design_odor_pair, total_drive


def cmd_sparsity(a):
    conn = Connectome.load(a.npz)
    pops = Populations.from_connectome(conn)
    odor_a, odor_b = design_odor_pair(pops, k=a.k, seed=a.odor_seed)
    grid = []
    for kc_thresh, apl in itertools.product(a.kc_thresh, a.apl_scale):
        p = Params(kc_thresh=kc_thresh, apl_scale=apl)
        eng = Engine(conn, pops, p, seed=0)
        rows = []
        for s in range(a.seeds):
            ra = kc_sparsity(eng, pops, odor_a, a.strength, seed=100 + s)
            rb = kc_sparsity(eng, pops, odor_b, a.strength, seed=100 + s)
            rows.append({"frac_active_A": ra["frac_active"], "frac_active_B": rb["frac_active"],
                         "jaccard": jaccard(ra["active"], rb["active"]),
                         "chance": chance_jaccard(ra["frac_active"], rb["frac_active"]),
                         "mbon_hz_A": ra["mbon_hz"], "mbon_hz_B": rb["mbon_hz"]})
        mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        rest = mbon_baseline(eng, pops, seed=100)  # baseline depends on kc_thresh/apl_scale too
        grid.append({"kc_thresh": kc_thresh, "apl_scale": apl, **mean,
                     "mbon_hz_rest": rest["mbon_hz"], "mbon_types_active_rest": rest["n_types_active"]})
        print(json.dumps(grid[-1]), flush=True)
    base = mbon_baseline(Engine(conn, pops, Params(), seed=0), pops, seed=100)
    out = {"odor_A": odor_a, "odor_B": odor_b, "drive_A": total_drive(pops, odor_a), "drive_B": total_drive(pops, odor_b),
           "strength": a.strength, "grid": grid, "mbon_hz_rest": base["mbon_hz"], "mbon_types_active_rest": base["n_types_active"]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sparsity")
    s.add_argument("--npz", default="data/malecns.npz"); s.add_argument("--out", default="results/m0/sparsity.json")
    s.add_argument("--kc-thresh", type=float, nargs="+", default=[1.0, 1.5]); s.add_argument("--apl-scale", type=float, nargs="+", default=[0.1, 0.2])
    s.add_argument("--strength", type=float, default=0.35); s.add_argument("--seeds", type=int, default=3)
    s.add_argument("--k", type=int, default=8); s.add_argument("--odor-seed", type=int, default=0)
    s.set_defaults(fn=cmd_sparsity)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
