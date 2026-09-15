#!/usr/bin/env python3
"""M0b: pool throughput, the M0 measurements reproduced through the pool (bit-for-bit), and the gate summary.

  throughput  per-worker ms/step of the decision and reinforcement phases at 4/8/16 workers, 3 repeats
  reproduce   M0 conditioning 5 arms x 8 seeds as pool jobs (== results/m0/conditioning.json per seed),
              sparsity seeds 100-102 (== results/m0/sparsity.json default row), resting baseline + runaway
              set, and one in-process vs pool decide() equality check
  summary     results/summary/m0b.json: budget table (spec C.6), exact-match results, the M0b gate
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np

from flymon.brain.circuits import Populations, compartments
from flymon.brain.conditioning import channel_specific_seeds, summarise
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.engine_cpu import Engine
from flymon.brain.plasticity import Plasticity
from flymon.brain.stimuli import design_odor_pair
from flymon.brain.fly_pool import FlyPool, FlySpec
from flymon.brain.pool_bench import budget_table, exact_match, m0b_gate, throughput_row
from flymon.brain.pool_jobs import baseline_job, conditioning_arm_job, rss_job, sparsity_job
from flymon.brain.presentation import decide

ARMS = ("both", "reversed", "noplast", "punish_only", "reward_only")


def _npz(a) -> str:
    if not Path(a.npz).exists():
        print(f"SKIP: {a.npz} not found (real data is user-downloaded, see docs/data.md)")
        raise SystemExit(2)
    return a.npz


def _write(path: str, obj: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1))
    print(f"wrote {path}")


def cmd_throughput(a):
    npz = _npz(a)
    conn = Connectome.load(npz)
    pops = Populations.from_connectome(conn)
    odor_a, _ = design_odor_pair(pops, k=a.k, seed=a.odor_seed)
    rows = []
    for W in a.workers:
        with FlyPool(npz, Params(), [FlySpec()] * W, workers=W) as pool:
            rows.append(throughput_row(pool, odor_a, a.strength, a.steps, a.warm, a.repeats, idx=pops.mbon))
        print(json.dumps(rows[-1]), flush=True)
    _write(a.out, {"steps": a.steps, "warm": a.warm, "repeats": a.repeats, "strength": a.strength, "odor_seed": a.odor_seed, "rows": rows})


def cmd_reproduce(a):
    npz = _npz(a)
    t0 = time.perf_counter()
    m0_cond = json.loads(Path(a.m0_conditioning).read_text()) if Path(a.m0_conditioning).exists() else None
    m0_sp = json.loads(Path(a.m0_sparsity).read_text()) if Path(a.m0_sparsity).exists() else None
    p = Params()
    with FlyPool(npz, p, [FlySpec()] * a.workers, workers=a.workers) as pool:
        jobs = [dict(seed=s, arm=arm, strength=a.strength, k=a.k, odor_seed=a.odor_seed, trials=a.trials,
                     present_ms=a.present_ms, settle_ms=a.settle_ms, punish_type=a.punish_type, reward_type=a.reward_type)
                for s in range(a.seeds) for arm in ARMS]
        res = pool.run_jobs(conditioning_arm_job, jobs)
        per_seed: dict = {}
        for job, r in zip(jobs, res):
            per_seed.setdefault(job["seed"], {})[job["arm"]] = r
        cond = summarise(per_seed)
        cond["channel_specific_seeds"] = channel_specific_seeds(cond["per_seed"])
        sp_rows = pool.run_jobs(sparsity_job, [dict(seed=100 + s, strength=a.strength, k=a.k, odor_seed=a.odor_seed) for s in range(a.rest_seeds)])
        base_rows = pool.run_jobs(baseline_job, [dict(seed=100 + s, ms=a.rest_ms) for s in range(a.rest_seeds)])
        rss_before = pool.run_jobs(rss_job, [{}] * a.workers)
        rss_after = pool.run_jobs(rss_job, [{}] * a.workers, shuffle_seed=1_000_003)      # builds one C-shuf variant per worker
        memory = {"worker_rss_GB": float(np.mean(rss_before)), "worker_rss_GB_max": float(max(rss_before)),
                  "variant_rss_GB": float(np.mean(np.subtract(rss_after, rss_before)))}
        conn = Connectome.load(npz)
        pops = Populations.from_connectome(conn)
        odor_a, odor_b = design_odor_pair(pops, k=a.k, seed=a.odor_seed)
        eng = Engine(conn, pops, p, seed=0)
        pl = Plasticity(eng, pops, compartments(conn, pops, p.core_frac))
        local = decide(eng, pl, pops, [odor_a, odor_b], a.strength, seed=7, settle_ms=200.0, read_ms=600.0, idx=pops.mbon)
        remote = pool.decide_batch([(0, [odor_a, odor_b], 7)], a.strength, settle_ms=200.0, read_ms=600.0, idx=pops.mbon)[0]
    decide_equal = bool(np.array_equal(local, remote))
    sparsity = {k: float(np.mean([r[k] for r in sp_rows])) for k in sp_rows[0]}
    sparsity["per_seed"] = sp_rows
    trimmed = np.array([r["mbon_hz_trimmed"] for r in base_rows])
    baseline = {"mbon_hz_rest": float(np.mean([r["mbon_hz"] for r in base_rows])), "mbon_hz_rest_trimmed": float(trimmed.mean()),
                "mbon_hz_rest_trimmed_sd": float(trimmed.std()), "per_seed": base_rows,
                "runaway": {k: float(np.mean([r["runaway"][k] for r in base_rows])) for k in ("n_over_sat", "n_kc_over_sat", "spike_share_over_sat")}}
    cond_match = exact_match(per_seed, {k: v for k, v in m0_cond["per_seed"].items() if int(k) < a.seeds}) if m0_cond else {"ok": False, "note": "no M0 file"}
    sp_match = {"ok": False, "note": "no M0 file"}
    if m0_sp:
        row = [g for g in m0_sp["grid"] if (g["kc_thresh"], g["apl_scale"], g.get("mbon_hold_frac")) == (p.kc_thresh, p.apl_scale, p.mbon_hold_frac)][0]
        keys = ("frac_active_A", "frac_active_B", "jaccard", "chance", "mbon_hz_A", "mbon_hz_B")
        diffs = {k: abs(sparsity[k] - row[k]) for k in keys}
        diffs["mbon_hz_rest_trimmed"] = abs(baseline["mbon_hz_rest_trimmed"] - row["mbon_hz_rest_trimmed"])
        sp_match = {"diffs": diffs, "max_abs_diff": max(diffs.values()), "ok": max(diffs.values()) == 0.0, "cpu_row": {k: row[k] for k in keys + ("mbon_hz_rest_trimmed",)}}
    out = {"params": {**dataclasses.asdict(p), "punish_type": a.punish_type, "reward_type": a.reward_type, "settle_ms": a.settle_ms,
                      "present_ms": a.present_ms, "trials": a.trials}, "strength": a.strength, "odor_seed": a.odor_seed,
           "workers": a.workers, "conditioning": cond, "conditioning_match": cond_match, "sparsity": sparsity, "sparsity_match": sp_match,
           "baseline": baseline, "memory": memory, "decide_equal": decide_equal, "wall_clock_s": time.perf_counter() - t0}
    print(json.dumps({"n_seeds": cond["n_seeds"], "n_flip": cond["n_flip"], "channel_specific": cond["channel_specific_seeds"],
                      "conditioning_exact": cond_match.get("ok"), "sparsity_exact": sp_match.get("ok"), "decide_equal": decide_equal,
                      "runaway": baseline["runaway"], "memory": memory, "wall_s": round(out["wall_clock_s"])}), flush=True)
    _write(a.out, out)


def cmd_summary(a):
    th = json.loads(Path(a.throughput).read_text())
    rp = json.loads(Path(a.reproduce).read_text())
    budget = budget_table(th["rows"], decisions=a.decisions, eval_decisions=a.eval_decisions, limit_hours=a.limit_hours)
    gate = m0b_gate(budget, rp["conditioning_match"], rp["sparsity_match"], rp["decide_equal"])
    co = rp["conditioning"]
    out = {"params_frozen": dataclasses.asdict(Params()), "throughput": th["rows"], "budget": budget,
           "conditioning": {k: co[k] for k in ("n_seeds", "n_flip", "n_flip_disc", "noplast_max_abs_dD", "channel_specific_seeds", "arms")},
           "conditioning_match": rp["conditioning_match"], "sparsity_match": rp["sparsity_match"], "decide_equal": rp["decide_equal"],
           "runaway": rp["baseline"]["runaway"], "memory": rp.get("memory"), "gate": gate, "generated_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    print(f"M0b gate {'PASS' if gate['passed'] else 'FAIL'}: baseline {budget['baseline_hours']:.1f} h (limit {budget['limit_hours']}), "
          f"conditioning_exact={gate['conditioning_exact_ok']} sparsity_exact={gate['sparsity_exact_ok']} decide_equal={gate['decide_equal_ok']}")
    _write(a.out, out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("throughput")
    t.add_argument("--npz", default="data/malecns.npz"); t.add_argument("--out", default="results/m0b/throughput.json")
    t.add_argument("--workers", type=int, nargs="+", default=[4, 8, 16]); t.add_argument("--steps", type=int, default=300)
    t.add_argument("--warm", type=int, default=100); t.add_argument("--repeats", type=int, default=3)
    t.add_argument("--strength", type=float, default=0.35); t.add_argument("--k", type=int, default=8); t.add_argument("--odor-seed", type=int, default=0)
    t.set_defaults(fn=cmd_throughput)
    r = sub.add_parser("reproduce")
    r.add_argument("--npz", default="data/malecns.npz"); r.add_argument("--out", default="results/m0b/reproduce.json")
    r.add_argument("--m0-conditioning", default="results/m0/conditioning.json"); r.add_argument("--m0-sparsity", default="results/m0/sparsity.json")
    r.add_argument("--workers", type=int, default=16); r.add_argument("--seeds", type=int, default=8)
    r.add_argument("--rest-seeds", type=int, default=3); r.add_argument("--rest-ms", type=float, default=3000.0)
    r.add_argument("--trials", type=int, default=12); r.add_argument("--present-ms", type=float, default=800.0); r.add_argument("--settle-ms", type=float, default=800.0)
    r.add_argument("--strength", type=float, default=0.35); r.add_argument("--k", type=int, default=8); r.add_argument("--odor-seed", type=int, default=0)
    r.add_argument("--punish-type", default="PPL105"); r.add_argument("--reward-type", default="PAM08")
    r.set_defaults(fn=cmd_reproduce)
    s = sub.add_parser("summary")
    s.add_argument("--throughput", default="results/m0b/throughput.json"); s.add_argument("--reproduce", default="results/m0b/reproduce.json")
    s.add_argument("--out", default="results/summary/m0b.json"); s.add_argument("--decisions", type=int, default=52_000)
    s.add_argument("--eval-decisions", type=int, default=81_600); s.add_argument("--limit-hours", type=float, default=60.0)
    s.set_defaults(fn=cmd_summary)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
