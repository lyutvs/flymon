"""M0d H.3a: the two measurements that align the amendment's rules with the runs that produced the adopted point.

Diagnostic only: calibration data, not an H.3 selection. No M2 turns, no candidate odours, no oracle,
plasticity off. No rule is proposed here — numbers and their implications only.

Red-team round 5 found two places where the H.3a rules quote seed sets that were never measured:

1. **D.4 on 8 and 16 seeds.** The amendment specifies an 8-seed D.4, but every D.4 run on disk
   (`d4_margin_protocol.py` e936434, `candidate_feasibility.py` 90c36f9, `slim_inputs.py` bd20a59) used the
   three ranking seeds 100-102. This item re-runs the D.4 gate measurement for the adopted candidate **and**
   C0 on seeds **100-115** at the gate's own protocol (`kc_sparsity` default settle 200 ms, read 600 ms,
   design pair `design_odor_pair(pops, k=8, seed=0)`), and reports both clauses and both margins at 3, 8 and
   16 seeds, with a 95% bootstrap CI of the overlap margin at 8 and 16 seeds.
2. **The MBON baseline over both seed blocks.** The amendment gates the M0c trimmed baseline on seeds
   132-147, but the candidates were only ever measured on 116-131, and `slim_inputs.py` found C0 differing by
   about 1 Hz between the two blocks. This item measures the M0c trimmed baseline
   (`flymon.brain.measure.mbon_baseline_multi`, 3000 ms, 100 Hz trim) for the adopted candidate and C0 on
   116-131, on 132-147, and pooled over all 32 seeds 116-147.

Nothing is reimplemented. The D.4 gate job and its per-seed statistics come from `d4_margin_protocol.py`
(`job_design`, `design_stats`, `GATE_SETTLE_MS`, `READ_MS`), the MBON baseline job and band from
`guard_after_baseline.py` (`job_baseline`, `REST_MS`, `SAT_HZ`, `BASELINE_BAND_HZ`), the adopted point's final
Params and C0's frozen Params from `slim_inputs.py` (`ADOPTED_*`, `C0_PARAMS`, `C0_SCALE`, `BASE_BLOCKS`), and
the CSC-into-APL scaling and NPZ path from `apl_input_scale_sweep.py`. The 16-seed D.4 needs no new code path:
`job_design` takes an arbitrary seed list, so only the seed list and the aggregation subsets differ; here the
seeds are split one per job so the 16 workers run them in parallel, and `design_stats` is applied to the first
3, the first 8 and all 16 per-seed rows.

Writes results/m0d/diag/align_inputs.{json,md}; --smoke runs 2 seeds per item into results/m0d/diag/smoke/
and prints a projection of the full run.
"""
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, "docs/superpowers/specs/m0d-diag")

from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.fly_pool import FlyPool

from apl_input_scale_sweep import NPZ, STRENGTH
from d4_margin_protocol import GATE_SETTLE_MS, READ_MS, design_stats, job_design
from guard_after_baseline import BASELINE_BAND_HZ, REST_MS, SAT_HZ, job_baseline
from slim_inputs import (ADOPTED_LABEL, ADOPTED_PARAMS, ADOPTED_SCALE, BASE_BLOCKS, C0_PARAMS, C0_SCALE)
from two_checks import D4_BAND, DESIGN_K, DESIGN_ODOR_SEED

# ---- the two configurations, measured identically in both items --------------------------------
C0_LABEL = "C0"
CONFIGS = ((ADOPTED_LABEL, ADOPTED_PARAMS, ADOPTED_SCALE),      # graded, r_max 0.333, kc_thresh 1.6,
           (C0_LABEL, C0_PARAMS, C0_SCALE))                     # scale 0.11863, hold 0.84082 / C0 at hold 0.85

# ---- item 1: D.4 on 16 seeds -------------------------------------------------------------------
D4_SEEDS = tuple(range(100, 116))            # 100-115; the ranking seeds 100-102 are the first three
SEED_COUNTS = (3, 8, 16)                     # the subsets reported: 100-102, 100-107, 100-115
BOOT_COUNTS = (8, 16)                        # the subsets that get a bootstrap CI
N_BOOT, BOOT_SEED = 10_000, 20260918

# ---- item 2: the MBON baseline over both blocks ------------------------------------------------
POOLED_NAME = "116-147"

FULL_UNITS = dict(item1=len(CONFIGS) * len(D4_SEEDS),
                  item2=len(CONFIGS) * sum(len(s) for _, s in BASE_BLOCKS))


def table(header, rows):
    L = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    L += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return L


def yn(x):
    return "yes" if x else "**NO**"


# ================================================================ item 1: D.4 at 3 / 8 / 16 seeds
def boot_overlap(margins):
    """95% percentile bootstrap CI of the mean overlap margin, resampling the seeds (N_BOOT draws, BOOT_SEED).

    A fresh `default_rng(BOOT_SEED)` per call, so each (config, seed-count) CI is reproducible on its own.
    """
    x = np.asarray(margins, float)
    rng = np.random.default_rng(BOOT_SEED)
    means = x[rng.integers(0, x.size, size=(N_BOOT, x.size))].mean(axis=1)
    lo, hi = (float(v) for v in np.percentile(means, [2.5, 97.5]))
    return dict(n_seeds=int(x.size), n_boot=N_BOOT, boot_seed=BOOT_SEED, mean=float(x.mean()),
                ci=[lo, hi], straddles_zero=bool(lo <= 0.0 <= hi), boot_mean=float(means.mean()),
                p_le_zero=float((means <= 0.0).mean()))


def subset_stats(rows, n, with_boot):
    """`design_stats` over the first n per-seed rows, plus the seed-to-seed SDs and (optionally) the CI."""
    sub = rows[:n]
    d = design_stats(sub)
    d["n_seeds"] = len(sub)
    d["seeds"] = [int(r["seed"]) for r in sub]
    margins = [float(r["chance"]) - float(r["jaccard"]) for r in sub]
    d["overlap_margin_per_seed"] = margins
    for k in ("frac_active_A", "frac_active_B", "jaccard", "chance"):
        d[k + "_sd"] = float(np.std([r[k] for r in sub], ddof=1)) if len(sub) > 1 else None
    d["pct_A_sd"] = 100 * d["frac_active_A_sd"] if d["frac_active_A_sd"] is not None else None
    d["pct_B_sd"] = 100 * d["frac_active_B_sd"] if d["frac_active_B_sd"] is not None else None
    d["overlap_margin_sd"] = float(np.std(margins, ddof=1)) if len(sub) > 1 else None
    d["overlap_margin_se"] = (d["overlap_margin_sd"] / math.sqrt(len(sub))) if d["overlap_margin_sd"] else None
    d["boot"] = boot_overlap(margins) if (with_boot and len(sub) > 1) else None
    d.pop("per_seed", None)
    return d


def item1(pool, seeds, acc):
    """One job per (config, seed): `job_design` at the gate protocol, exactly as the 3-seed runs called it."""
    jobs, owners = [], []
    for label, params, scale in CONFIGS:
        for s in seeds:
            owners.append((label, int(s)))
            jobs.append(dict(params=params, scale=scale, settle_ms=float(GATE_SETTLE_MS), seeds=[int(s)]))
    t0 = time.time()
    raw = pool.run_jobs(job_design, jobs)
    acc("item1", len(jobs), time.time() - t0)

    per_config = {}
    for label, _, _ in CONFIGS:
        rows = [r for (lab, _s), part in zip(owners, raw) if lab == label for r in part]
        rows.sort(key=lambda r: r["seed"])
        counts = [n for n in SEED_COUNTS if n <= len(rows)] or [len(rows)]
        per_config[label] = dict(per_seed=rows, seeds=[int(r["seed"]) for r in rows],
                                 by_count={str(n): subset_stats(rows, n, n in BOOT_COUNTS) for n in counts},
                                 seed_counts=counts)
        for n in counts:
            d = per_config[label]["by_count"][str(n)]
            b = d["boot"]
            print(f"  item 1: {label} n={n} -> A {d['pct_A']:.3f}% B {d['pct_B']:.3f}%, sparsity margin "
                  f"{d['margin_pp']:+.3f} pp, chance-J {d['overlap_margin']:+.5f}"
                  + (f", CI [{b['ci'][0]:+.5f}, {b['ci'][1]:+.5f}]" if b else "")
                  + f", D.4 {'PASS' if d['d4_ok'] else 'FAIL'}", flush=True)
    return dict(settle_ms=GATE_SETTLE_MS, read_ms=READ_MS, strength=STRENGTH, design_k=DESIGN_K,
                design_odor_seed=DESIGN_ODOR_SEED, d4_band=list(D4_BAND), seeds=[int(s) for s in seeds],
                seed_counts=list(SEED_COUNTS), boot_counts=list(BOOT_COUNTS), n_boot=N_BOOT,
                boot_seed=BOOT_SEED, configs=per_config)


# ================================================================ item 2: the baseline over both blocks
def baseline_block(pool, params, scale, seeds, acc):
    """`guard_after_baseline.job_baseline` (i.e. `mbon_baseline_multi`) one seed per job, mean over the seeds."""
    jobs = [dict(params=params, scale=scale, seeds=[int(s)]) for s in seeds]
    t0 = time.time()
    rows = [r for part in pool.run_jobs(job_baseline, jobs) for r in part]
    acc("item2", len(jobs), time.time() - t0)
    return rows


def block_stats(rows, name):
    x = np.array([r["trimmed"] for r in rows], float)
    sd = float(x.std(ddof=1)) if x.size > 1 else 0.0
    return dict(name=name, n=int(x.size), seeds=[int(r["seed"]) for r in rows],
                per_seed=[float(v) for v in x], mean_hz=float(x.mean()), sd_hz=sd,
                se_hz=float(sd / math.sqrt(x.size)) if x.size else None,
                in_band=bool(BASELINE_BAND_HZ[0] <= float(x.mean()) <= BASELINE_BAND_HZ[1]),
                raw_hz=float(np.mean([r["raw"] for r in rows])),
                n_saturated=float(np.mean([r["n_saturated"] for r in rows])),
                n_types_active=float(np.mean([r["n_types_active"] for r in rows])))


def item2(pool, blocks, acc):
    per_config = {}
    for label, params, scale in CONFIGS:
        by_block, all_rows = {}, []
        for name, seeds in blocks:
            rows = baseline_block(pool, params, scale, seeds, acc)
            rows.sort(key=lambda r: r["seed"])
            by_block[name] = block_stats(rows, name)
            all_rows += rows
            print(f"  item 2: {label} seeds {name} -> {by_block[name]['mean_hz']:.4f} Hz "
                  f"(SE {by_block[name]['se_hz']:.4f}), in {BASELINE_BAND_HZ[0]:g}-{BASELINE_BAND_HZ[1]:g} Hz: "
                  f"{by_block[name]['in_band']}", flush=True)
        all_rows.sort(key=lambda r: r["seed"])
        pooled = block_stats(all_rows, POOLED_NAME)
        diff = None
        if len(by_block) == 2:
            a, b = (by_block[n] for n, _ in blocks)
            d = a["mean_hz"] - b["mean_hz"]
            se = math.sqrt((a["se_hz"] or 0.0) ** 2 + (b["se_hz"] or 0.0) ** 2)
            diff = dict(blocks=[a["name"], b["name"]], diff_hz=float(d), se_hz=float(se),
                        z=float(d / se) if se else None)
        per_config[label] = dict(params=params, scale=float(scale), blocks=by_block, pooled=pooled,
                                 block_diff=diff,
                                 all_in_band=bool(all(b["in_band"] for b in by_block.values())
                                                  and pooled["in_band"]))
        print(f"  item 2: {label} pooled {pooled['name']} -> {pooled['mean_hz']:.4f} Hz "
              f"(SE {pooled['se_hz']:.4f}), in band: {pooled['in_band']}", flush=True)
    return dict(rest_ms=REST_MS, sat_hz=SAT_HZ, band_hz=list(BASELINE_BAND_HZ),
                blocks={n: [int(s) for s in s_] for n, s_ in blocks}, pooled_name=POOLED_NAME,
                configs=per_config)


# ================================================================ report
def report(res, wall_s):
    meta, i1, i2 = res["meta"], res["item1"], res["item2"]
    lo_pct, hi_pct = 100 * D4_BAND[0], 100 * D4_BAND[1]
    L = ["# M0d H.3a: 16-seed D.4 and both baseline blocks at the adopted point", "",
         f"Diagnostic only (calibration data, not an H.3 selection; no M2 turns, candidate odours or oracle; "
         f"plasticity off). No rule changes are proposed here — numbers and implications only. Both items are "
         f"measured on the same two configurations: the adopted candidate **{ADOPTED_LABEL}** at its final "
         f"Params (`apl_mode` graded, `apl_r_max` {ADOPTED_PARAMS['apl_r_max']:g}, `kc_thresh` "
         f"{ADOPTED_PARAMS['kc_thresh']:g}, `apl_input_scale` {ADOPTED_SCALE:.5f}, `mbon_hold_frac` "
         f"{ADOPTED_PARAMS['mbon_hold_frac']:.5f}) and **C0** (`apl_mode` spiking, `kc_thresh` "
         f"{C0_PARAMS['kc_thresh']:g}, no CSC-into-APL scaling, `mbon_hold_frac` "
         f"{C0_PARAMS['mbon_hold_frac']:g}). Commit {meta['commit']} (dirty tree: "
         f"{str(meta['dirty']).lower()})" + (", **--smoke run (reduced settings)**." if meta["smoke"] else "."),
         "",
         "## Item 1 — D.4 at the gate protocol on 16 seeds", "",
         f"The D.4 gate measurement exactly as the gate calls it (`flymon.brain.measure.kc_sparsity` at its "
         f"default settle {GATE_SETTLE_MS:g} ms, read {READ_MS:g} ms, strength {STRENGTH}, design pair "
         f"`design_odor_pair(pops, k={DESIGN_K}, seed={DESIGN_ODOR_SEED})`), on seeds "
         f"{i1['seeds'][0]}-{i1['seeds'][-1]} ({len(i1['seeds'])} seeds). The job and the per-seed statistics "
         f"are `d4_margin_protocol.job_design` / `design_stats`, the same code path the 3-seed runs used; only "
         f"the seed list and the aggregation subsets differ. D.4: KC active {lo_pct:g}-{hi_pct:g}% for **both** "
         f"odours and Jaccard ≤ chance. Sparsity margin = min({hi_pct:g} − max odour, min odour − {lo_pct:g}) "
         f"in pp; overlap margin = chance − J."]

    for label in (ADOPTED_LABEL, C0_LABEL):
        c = i1["configs"][label]
        L += ["", f"### {label} — per seed", ""]
        L += table(["seed", "KC % A", "KC % B", "Jaccard", "chance", "chance − J"],
                   [[r["seed"], f"{100 * r['frac_active_A']:.3f}", f"{100 * r['frac_active_B']:.3f}",
                     f"{r['jaccard']:.5f}", f"{r['chance']:.5f}",
                     f"{r['chance'] - r['jaccard']:+.5f}"] for r in c["per_seed"]])
        L += ["", f"### {label} — by seed count", ""]
        rows = []
        for n in c["seed_counts"]:
            d = c["by_count"][str(n)]
            b = d["boot"]
            rows.append([f"{n} ({d['seeds'][0]}-{d['seeds'][-1]})",
                         f"{d['pct_A']:.3f}" + (f" ± {d['pct_A_sd']:.3f}" if d["pct_A_sd"] is not None else ""),
                         f"{d['pct_B']:.3f}" + (f" ± {d['pct_B_sd']:.3f}" if d["pct_B_sd"] is not None else ""),
                         f"{d['jaccard']:.5f}", f"{d['chance']:.5f}",
                         f"{d['margin_pp']:+.3f}", f"{d['overlap_margin']:+.5f}"
                         + (f" ± {d['overlap_margin_sd']:.5f}" if d["overlap_margin_sd"] else ""),
                         (f"[{b['ci'][0]:+.5f}, {b['ci'][1]:+.5f}]" if b else "—"),
                         (yn(not b["straddles_zero"]) if b else "—"),
                         yn(d["d4_sparsity_ok"]), yn(d["d4_overlap_ok"]),
                         "**PASS**" if d["d4_ok"] else "**FAIL**"])
        L += table(["seeds", f"KC % A (mean ± SD)", "KC % B (mean ± SD)", "Jaccard", "chance",
                    "sparsity margin pp", "overlap margin (mean ± SD)", "overlap 95% CI",
                    "CI excludes 0?", f"{lo_pct:g}-{hi_pct:g}% both", "J ≤ chance", "D.4"], rows)

    L += ["", f"## Item 2 — the M0c trimmed MBON baseline over both seed blocks", "",
          f"`flymon.brain.measure.mbon_baseline_multi` through `guard_after_baseline.job_baseline`, "
          f"{REST_MS:g} ms of rest per seed, cells above {SAT_HZ:g} Hz excluded from the mean; the statistic "
          f"is the mean over the per-seed trimmed rates, as `mbon_baseline_multi` aggregates them. Blocks: "
          + "; ".join(f"**{n}** ({len(s)} seeds)" for n, s in i2["blocks"].items())
          + f"; pooled = all {sum(len(s) for s in i2['blocks'].values())} seeds. Band: "
          f"{BASELINE_BAND_HZ[0]:g}-{BASELINE_BAND_HZ[1]:g} Hz.", ""]
    rows = []
    for label in (ADOPTED_LABEL, C0_LABEL):
        c = i2["configs"][label]
        for name in list(c["blocks"]) + [POOLED_NAME]:
            b = c["blocks"][name] if name in c["blocks"] else c["pooled"]
            rows.append([label, name if name != POOLED_NAME else f"{name} (pooled)", b["n"],
                         f"{b['mean_hz']:.4f}", f"{b['sd_hz']:.4f}",
                         f"{b['se_hz']:.4f}" if b["se_hz"] is not None else "—", yn(b["in_band"])])
    L += table(["config", "seeds", "n", "mean Hz", "SD", "SE", f"in {BASELINE_BAND_HZ[0]:g}-"
                f"{BASELINE_BAND_HZ[1]:g} Hz?"], rows)
    L += ["", "Block difference (first block − second block):", ""]
    for label in (ADOPTED_LABEL, C0_LABEL):
        d = i2["configs"][label]["block_diff"]
        L.append(f"- **{label}**: " + ("not computable (one block only)." if d is None else
                                       f"{d['blocks'][0]} − {d['blocks'][1]} = {d['diff_hz']:+.4f} Hz "
                                       f"(SE {d['se_hz']:.4f}"
                                       + (f", z {d['z']:+.2f}" if d["z"] is not None else "") + ")."))

    L += ["", "## Summary", ""]
    c = i1["configs"][ADOPTED_LABEL]
    parts = []
    for n in c["seed_counts"]:
        d = c["by_count"][str(n)]
        b = d["boot"]
        parts.append(f"{n} seeds {'PASS' if d['d4_ok'] else 'FAIL'} (sparsity {d['margin_pp']:+.3f} pp, "
                     f"overlap {d['overlap_margin']:+.5f}"
                     + (f", CI [{b['ci'][0]:+.5f}, {b['ci'][1]:+.5f}] "
                        f"{'straddles' if b['straddles_zero'] else 'excludes'} 0" if b else "") + ")")
    L += [f"- (a) D.4 at the adopted point: " + "; ".join(parts) + ".",
          "- (b) Baseline band: " + "; ".join(
              f"{label} " + ", ".join(
                  f"{name} {(i2['configs'][label]['blocks'][name] if name in i2['configs'][label]['blocks'] else i2['configs'][label]['pooled'])['mean_hz']:.3f} Hz "
                  f"{'in' if (i2['configs'][label]['blocks'][name] if name in i2['configs'][label]['blocks'] else i2['configs'][label]['pooled'])['in_band'] else 'OUT of'} band"
                  for name in list(i2["configs"][label]["blocks"]) + [POOLED_NAME])
              for label in (ADOPTED_LABEL, C0_LABEL)) + ".", ""]

    L += [f"Wall time of the run: {wall_s:.0f} s ({wall_s / 60:.1f} min); engine work "
          + ", ".join(f"{k} {v['units']} units in {v['wall_s']:.0f} s" for k, v in meta["cost"].items())
          + f". Commit {meta['commit']}, dirty tree: {str(meta['dirty']).lower()}.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    t0 = time.time()
    conn = Connectome.load(NPZ)
    pops = Populations.from_connectome(conn)
    del conn

    if smoke:
        d4_seeds = D4_SEEDS[:2]
        blocks = [(n, s[:2]) for n, s in BASE_BLOCKS]
        workers, out_dir = 4, "results/m0d/diag/smoke"
    else:
        d4_seeds = D4_SEEDS
        blocks = list(BASE_BLOCKS)
        workers, out_dir = 16, "results/m0d/diag"

    cost = {}

    def acc(step, units, wall):
        c = cost.setdefault(step, dict(units=0, wall_s=0.0))
        c["units"] += int(units)
        c["wall_s"] += float(wall)

    print(f"item 1: D.4 on {len(d4_seeds)} seeds x {len(CONFIGS)} configs at settle {GATE_SETTLE_MS:g} ms; "
          f"item 2: baseline on " + " + ".join(f"{len(s)}" for _, s in blocks)
          + f" seeds x {len(CONFIGS)} configs; {workers} workers", flush=True)

    with FlyPool(NPZ, Params(), [{} for _ in range(workers)], workers=workers) as pool:
        i1 = item1(pool, d4_seeds, acc)
        i2 = item2(pool, blocks, acc)

    wall = time.time() - t0
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    res = dict(meta=dict(npz=NPZ, smoke=smoke, commit=commit, dirty=dirty, strength=STRENGTH,
                         gate_settle_ms=GATE_SETTLE_MS, read_ms=READ_MS, adopted_label=ADOPTED_LABEL,
                         adopted_params=ADOPTED_PARAMS, adopted_scale=ADOPTED_SCALE, c0_params=C0_PARAMS,
                         c0_scale=C0_SCALE, d4_seeds=[int(s) for s in d4_seeds],
                         baseline_blocks={n: [int(x) for x in s] for n, s in blocks},
                         workers=workers, wall_s=wall, cost=cost, plasticity=False),
               item1=i1, item2=i2)

    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/align_inputs.json", "w") as f:
        json.dump(res, f, indent=1)
    with open(f"{out_dir}/align_inputs.md", "w") as f:
        f.write(report(res, wall))
    print(report(res, wall))
    print(f"wrote {out_dir}/align_inputs.json / .md in {wall:.0f} s", flush=True)

    if smoke:
        proj = 0.0
        for k, full in FULL_UNITS.items():
            c = cost.get(k)
            if not c or not c["units"]:
                continue
            per = c["wall_s"] / c["units"]
            est = per * full * (workers / 16.0)       # the smoke runs `workers`, the full run 16
            proj += est
            print(f"projection {k}: {c['units']} units in {c['wall_s']:.1f} s -> {full} units at 16 workers "
                  f"≈ {est:.0f} s", flush=True)
        overhead = wall - sum(c["wall_s"] for c in cost.values())
        print(f"projected full run ≈ {proj + overhead:.0f} s ({(proj + overhead) / 60:.1f} min), including "
              f"{overhead:.0f} s of non-engine overhead measured here", flush=True)
