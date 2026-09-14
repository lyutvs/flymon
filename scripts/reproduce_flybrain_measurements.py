#!/usr/bin/env python3
"""Reproduce the three flybrain measurements that gate the M0 engine.

  sparsity      Kenyon-cell sparsity/overlap for a designed odour pair + MBON baseline
  conditioning  paired-seed olfactory conditioning with reversal (Task 9)
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np

from flymon.brain.circuits import Populations, compartments, export_compartments, validate_populations
from flymon.brain.conditioning import Readout, arms, run_arm, summarise
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.engine_cpu import Engine
from flymon.brain.measure import chance_jaccard, jaccard, kc_sparsity, mbon_baseline_multi
from flymon.brain.plasticity import Plasticity
from flymon.brain.stimuli import design_odor_pair, total_drive


COMPARTMENTS_OUT = Path("results/summary/compartments.json")


def cmd_sparsity(a):
    conn = Connectome.load(a.npz)
    pops = Populations.from_connectome(conn)
    comps = compartments(conn, pops, Params().core_frac)
    validate_populations(conn, pops, comps)
    COMPARTMENTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    export_compartments(comps, conn, COMPARTMENTS_OUT)
    odor_a, odor_b = design_odor_pair(pops, k=a.k, seed=a.odor_seed)
    grid = []
    for kc_thresh, apl, hold in itertools.product(a.kc_thresh, a.apl_scale, a.mbon_hold):
        p = Params(kc_thresh=kc_thresh, apl_scale=apl, mbon_hold_frac=hold)
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
        # baseline depends on every Params in the grid; several seeds because the raw mean is
        # unstable when the FR1 clique saturates a few MBONs (the gate reads the trimmed mean)
        rest = mbon_baseline_multi(eng, pops, seeds=range(100, 100 + a.rest_seeds), ms=a.rest_ms)
        grid.append({"kc_thresh": kc_thresh, "apl_scale": apl, "mbon_hold_frac": hold, **mean, **rest})
        print(json.dumps(grid[-1]), flush=True)
    d = Params()
    default_row = [g for g in grid if (g["kc_thresh"], g["apl_scale"], g["mbon_hold_frac"])
                   == (d.kc_thresh, d.apl_scale, d.mbon_hold_frac)]
    base = (default_row[0] if default_row else
            mbon_baseline_multi(Engine(conn, pops, d, seed=0), pops,
                                seeds=range(100, 100 + a.rest_seeds), ms=a.rest_ms))
    out = {"odor_A": odor_a, "odor_B": odor_b, "drive_A": total_drive(pops, odor_a), "drive_B": total_drive(pops, odor_b),
           "strength": a.strength, "grid": grid,
           "mbon_hz_rest": base["mbon_hz_rest"], "mbon_hz_rest_trimmed": base["mbon_hz_rest_trimmed"],
           "mbon_hz_rest_trimmed_sd": base["mbon_hz_rest_trimmed_sd"], "mbon_n_saturated": base["mbon_n_saturated"],
           "mbon_types_active_rest": base["mbon_types_active_rest"]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))


def _cond_worker(args):
    npz, params_dict, seed, strength, k, odor_seed, punish_type, reward_type, kw = args
    conn = Connectome.load(npz)
    pops = Populations.from_connectome(conn)
    p = Params(**params_dict)
    eng = Engine(conn, pops, p, seed=seed)
    comps = compartments(conn, pops, p.core_frac)
    validate_populations(conn, pops, comps, punish_type, reward_type)
    pl = Plasticity(eng, pops, comps)
    ro = Readout.from_compartments(comps, punish_type, reward_type)
    a, b = design_odor_pair(pops, k=k, seed=odor_seed)
    return seed, {arm: run_arm(eng, pl, pops, ro, a, b, strength, seed, arm,
                               punish_type=punish_type, reward_type=reward_type, **kw)
                  for arm in arms(punish_type, reward_type)}


def cmd_conditioning(a):
    from multiprocessing import Pool
    params_dict = dict(learn_rate=a.learn_rate, kc_trace_scale=a.kc_trace_scale, da_trace_scale=a.da_trace_scale,
                       recovery_per_pulse=a.recovery, kc_thresh=a.kc_thresh, apl_scale=a.apl_scale)
    kw = dict(trials=a.trials, present_ms=a.present_ms)
    jobs = [(a.npz, params_dict, s, a.strength, a.k, a.odor_seed, a.punish_type, a.reward_type, kw)
            for s in range(a.seeds)]
    with Pool(a.jobs) as pool:
        per_seed = dict(pool.map(_cond_worker, jobs))
    params_out = dict(params_dict, punish_type=a.punish_type, reward_type=a.reward_type)
    out = {"params": params_out, "strength": a.strength, "trials": a.trials, **summarise(per_seed)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("n_seeds", "n_flip", "noplast_max_abs_dD")}), json.dumps(out["arms"], indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sparsity")
    s.add_argument("--npz", default="data/malecns.npz"); s.add_argument("--out", default="results/m0/sparsity.json")
    s.add_argument("--kc-thresh", type=float, nargs="+", default=[1.0, 1.5]); s.add_argument("--apl-scale", type=float, nargs="+", default=[0.1, 0.2])
    s.add_argument("--mbon-hold", type=float, nargs="+", default=[0.85])   # third grid axis: MBON baseline band is 3-4 Hz
    s.add_argument("--strength", type=float, default=0.35); s.add_argument("--seeds", type=int, default=3)
    s.add_argument("--k", type=int, default=8); s.add_argument("--odor-seed", type=int, default=0)
    s.add_argument("--rest-seeds", type=int, default=3); s.add_argument("--rest-ms", type=float, default=3000.0)
    s.set_defaults(fn=cmd_sparsity)
    c = sub.add_parser("conditioning")
    c.add_argument("--npz", default="data/malecns.npz"); c.add_argument("--out", default="results/m0/conditioning.json")
    c.add_argument("--seeds", type=int, default=8)
    c.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))   # each worker holds a full connectome
    c.add_argument("--trials", type=int, default=12); c.add_argument("--present-ms", type=float, default=800.0)
    c.add_argument("--strength", type=float, default=0.35); c.add_argument("--k", type=int, default=8); c.add_argument("--odor-seed", type=int, default=0)
    c.add_argument("--learn-rate", type=float, default=Params().learn_rate)
    c.add_argument("--kc-trace-scale", type=float, default=Params().kc_trace_scale)
    c.add_argument("--da-trace-scale", type=float, default=Params().da_trace_scale)
    c.add_argument("--recovery", type=float, default=0.0)
    # flybrain used PPL105 punishment / PAM08 reward; the real-data gate overrides punishment to PPL101
    c.add_argument("--punish-type", default="PPL105"); c.add_argument("--reward-type", default="PAM08")
    c.add_argument("--kc-thresh", type=float, default=Params().kc_thresh); c.add_argument("--apl-scale", type=float, default=Params().apl_scale)
    c.set_defaults(fn=cmd_conditioning)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
