"""M0d H.3a: higher-power clamped-release control. Calibration only, diagnostic.

`two_checks.py` (check 2) compared the calibrated graded engine (A) against the same engine with its release
clamped to A's mean (B) on the H.3 48-odour reference set. All four metrics landed inside the noise band, but
every point estimate favoured A. With only 48 odour clusters and 2 all51 seeds the CIs were wide. This run
repeats exactly that comparison with about four times the odours (192) and four times the all51 seeds (8), to
tell a real small effect from nothing.

**The extended 192-odour set used here is a diagnostic calibration set. It does NOT replace the H.3 48-odour
reference set** — it uses the same generator rule with a different draw (rng 800001), and no H.3 selection,
M2 turn, candidate odour or oracle is involved. Plasticity is off throughout.

Pre-registered decision rule, unchanged from `two_checks.py` and fixed before this run: the primary metric is
the odour-to-odour IQR of KC active %. The effect counts as measured only if the 95% bootstrap CI of the
A - B difference excludes 0. Secondary metrics (all51 log10 variance of per-glomerulus KC drive, all51
zero-drive glomerulus count, median KC active %) are reported the same way and do not override the primary.

Definitions are imported, not re-invented: the odour generator rule, the CSC-into-APL scaling and the read
window come from apl_input_scale_sweep.py; the all51 protocol and its statistics from apl_cap_disparity.py;
the presentation job, the metric definitions and the paired bootstrap from two_checks.py.

Writes results/m0d/diag/clamp_control_power.json (raw) and .md (report); --smoke runs a reduced version
(8 odours, 4 glomeruli) into results/m0d/diag/smoke/ and prints a projection of the full run.
"""
import json
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
from flymon.brain.stimuli import channel_strengths

from apl_cap_disparity import EXCLUDE, all51_stats, job_all51
from apl_input_scale_sweep import NPZ, READ_STEPS, SETTLE_MS, STRENGTH
from two_checks import BOOT_DRAWS, BOOT_SEED, boot_ci, iqr, job_ref, per_odor_kc_pct

EXT_SEED = 800_001                      # a different draw of the same generator rule as the H.3 reference set
N_ODORS_EXT = 192
EXT_PROBE_SEED0 = 810_100               # odour j gets probe seeds (810100 + 2j, 810101 + 2j)
ALL51_SEEDS_8 = tuple(range(200, 208))  # 8 seeds instead of apl_cap_disparity.ALL51_SEEDS' 2
CAND_PARAMS = dict(apl_mode="graded", apl_r_max=0.333, kc_thresh=1.5)
CAND_SCALE = 0.0799
RELEASE_TOL = 0.02
PRIOR_JSON = "results/m0d/diag/two_checks.json"
MAX_PROJECTED_MIN = 25.0


def extended_odors(pops, n=N_ODORS_EXT, seed=EXT_SEED):
    """The reference-set generator rule (apl_input_scale_sweep.reference_odors) with a different draw."""
    cand = [t for t in pops.receptor_types if t not in EXCLUDE]                  # dict order = sorted names
    assert len(cand) == 51
    rng = np.random.default_rng(seed)
    odors = []
    for j in range(n):
        k = int(rng.integers(6, 10))                                             # 6..9 glomeruli
        types = sorted(str(t) for t in rng.choice(cand, size=k, replace=False))
        odors.append(dict(name=f"X{j:03d}", types=types, strengths=channel_strengths(pops, types),
                          seeds=[EXT_PROBE_SEED0 + 2 * j, EXT_PROBE_SEED0 + 1 + 2 * j]))
    assert len({tuple(o["types"]) for o in odors}) == n, "the extended odour type-sets are not distinct"
    return odors


def log10_var(K):
    return float(np.var(np.log10(np.maximum(np.asarray(K, float), 1))))          # all51_stats' definition


def zero_count(K):
    return float((np.asarray(K, float) == 0).sum())


METRICS = (("odour-to-odour IQR of KC active %", True, "odour", iqr),
           ("all51 log10 variance of per-glomerulus KC drive", False, "all51", log10_var),
           ("all51 zero-drive glomerulus count", False, "all51", zero_count),
           ("median KC active % (extended set)", False, "odour", lambda x: float(np.median(x))))
PRIOR_NAMES = {"odour-to-odour IQR of KC active %": "odour-to-odour IQR of KC active %",
               "all51 log10 variance of per-glomerulus KC drive":
                   "all51 log10 variance of per-glomerulus KC drive",
               "all51 zero-drive glomerulus count": "all51 zero-drive glomerulus count",
               "median KC active % (extended set)": "median KC active % (reference set)"}


def prior_metrics():
    """The 48-odour / 2-seed numbers from two_checks.py, keyed by this run's metric names."""
    try:
        r = json.load(open(PRIOR_JSON))
    except (OSError, ValueError):
        return {}
    by_name = {m["name"]: m for m in r["check2"]["metrics"]}
    out = {}
    for name, _, _, _ in METRICS:
        m = by_name.get(PRIOR_NAMES[name])
        if m:
            out[name] = dict(a=m["a"], b=m["b"], diff=m["diff"], ci=m["ci"], within_noise=m["within_noise"],
                             n_clusters=m["n_clusters"])
    out["_r_bar"] = r["check2"]["r_bar"]
    out["_release_rel_diff"] = r["check2"]["release_rel_diff"]
    return out


def report(res, wall_s):
    m2, meta, prior = res["comparison"], res["meta"], res["prior"]
    L = ["# M0d H.3a: higher-power clamped-release control", "",
         f"Diagnostic calibration only (no H.3 selection, no M2 turns, no candidate odours, no oracle; plasticity "
         f"off). **The {meta['n_odors']}-odour set used here is a diagnostic extension and does not replace the "
         f"H.3 48-odour reference set.** It follows the same generator rule (6-9 glomeruli drawn without "
         f"replacement from the 51 assignable types, DA1/V excluded, strength ∝ 1/receptor count) with a "
         f"different draw, `np.random.default_rng({EXT_SEED})`; odour j gets probe seeds "
         f"({EXT_PROBE_SEED0} + 2j, {EXT_PROBE_SEED0 + 1} + 2j). All {meta['n_odors']} type-sets are distinct. "
         f"Presentation strength {STRENGTH}, settle {SETTLE_MS} ms, read {READ_STEPS} steps. "
         f"Commit {meta['commit']} (dirty tree: {str(meta['dirty']).lower()}).", "",
         f"Engine A: graded APL as calibrated (apl_r_max 0.333, kc_thresh 1.5, apl_input_scale {CAND_SCALE}, "
         f"everything else default). Engine B: identical Params except `apl_slope = 1e9` (the sigmoid becomes a "
         f"constant 0.5) and `apl_r_max = 2·R̄·0.333 = {m2['r_max_b']:.6g}`. "
         f"**R̄ = {m2['r_bar']:.6f}** is engine A's mean release fraction over this extended set — all "
         f"{m2['n_presentations_a']} presentations ({meta['n_odors']} odours x 2 seeds) x {m2['n_apl']} APL cells "
         f"x {READ_STEPS} read steps. (On the 48-odour set `two_checks.py` measured R̄ = "
         f"{prior['_r_bar']:.6f}.)", "",
         "## Release-match validation (pre-comparison)", "",
         f"Time-integrated APL→KC input is mean release x read steps x the (identical) APL→KC weights, so the "
         f"engines are compared on mean release per cell per step in spike-equivalents: "
         f"A = {m2['release_a']:.6f}, B = {m2['release_b']:.6f}, relative difference "
         f"**{100 * m2['release_rel_diff']:.4f}%** "
         f"({'within' if m2['release_match_ok'] else 'ABOVE'} the {100 * RELEASE_TOL:g}% tolerance; the 48-odour "
         f"run reported {100 * prior['_release_rel_diff']:.4f}%)."]
    if not m2["release_match_ok"]:
        L += ["", f"**The release match exceeds {100 * RELEASE_TOL:g}%, so the comparison is not reported.**", ""]
    else:
        L += ["", "## Metrics (A − B, 95% bootstrap CI), this run vs the 48-odour run", "",
              f"Bootstrap: {BOOT_DRAWS} draws, seed {BOOT_SEED}, clusters resampled with replacement and paired "
              f"between A and B. The odour metrics resample the {meta['n_odors']} odours (each odour's 2 seeds "
              f"averaged first); the all51 metrics have no odour clusters, so they resample the "
              f"{m2['n_glom']} glomeruli of the all51 protocol with the same draws and seed. The all51 protocol "
              f"is unchanged (51 single-glomerulus stimuli, channel strength c_norm/receptor count with "
              f"c_norm = {m2['c_norm']:g}, strength {STRENGTH}, settle {SETTLE_MS} ms, read {READ_STEPS} steps) "
              f"except for the seeds: {list(ALL51_SEEDS_8)} instead of the earlier 2. "
              f"\"Excludes 0\" = the 95% CI of the difference does not contain 0.", "",
              "| metric | A | B | A − B | 95% CI | excludes 0? | 48-odour A − B | 48-odour 95% CI | 48-odour "
              "excludes 0? |", "|---|---|---|---|---|---|---|---|---|"]
        for m in m2["metrics"]:
            p = prior.get(m["name"])
            pc = (f"{p['diff']:.4f} | [{p['ci'][0]:.4f}, {p['ci'][1]:.4f}] | "
                  f"{'no' if p['within_noise'] else 'YES'}") if p else "— | — | —"
            L.append(f"| {m['name']}{' (primary)' if m['primary'] else ''} | {m['a']:.4f} | {m['b']:.4f} | "
                     f"{m['diff']:.4f} | [{m['ci'][0]:.4f}, {m['ci'][1]:.4f}] | "
                     f"{'no' if m['within_noise'] else 'YES'} | {pc} |")
        prim = [m for m in m2["metrics"] if m["primary"]][0]
        sec = [m for m in m2["metrics"] if not m["primary"]]
        agree = [m for m in sec if p_sign(m["diff"]) == p_sign(prim["diff"]) and p_sign(m["diff"]) != 0]
        excl = [m for m in sec if not m["within_noise"]]
        sign_notes = [f"{m['name']} {m['diff']:+.4f}" for m in sec]
        L += ["", f"- Pre-registered rule: the primary metric ({prim['name']}) difference is {prim['diff']:.4f} "
                  f"with 95% CI [{prim['ci'][0]:.4f}, {prim['ci'][1]:.4f}], which "
                  f"**{'CONTAINS' if prim['within_noise'] else 'EXCLUDES'} 0**, so at "
                  f"{prim['n_clusters']} odour clusters the effect "
                  f"**{'is not measured' if prim['within_noise'] else 'IS measured'}** by the rule fixed before "
                  f"the run. The CI width goes from "
                  f"{prior[prim['name']]['ci'][1] - prior[prim['name']]['ci'][0]:.4f} at 48 odours to "
                  f"{prim['ci'][1] - prim['ci'][0]:.4f} here.",
              f"- Secondary metrics: {len(agree)} of {len(sec)} have the same sign as the primary difference "
              f"({'; '.join(sign_notes)}). "
              f"Secondary metrics whose CI excludes 0: "
              + (", ".join(m["name"] for m in excl) or "none") + "."]
    L += ["", f"Wall time of the run: {wall_s:.0f} s ({wall_s / 60:.1f} min). Commit {meta['commit']}, "
              f"dirty tree: {str(meta['dirty']).lower()}.", ""]
    return "\n".join(L)


def p_sign(x):
    return int(np.sign(x))


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    t0 = time.time()
    conn = Connectome.load(NPZ)
    pops = Populations.from_connectome(conn)
    odors_all = extended_odors(pops)
    rts_all = sorted(t for t in pops.receptor_types if t not in EXCLUDE)
    c_norm = float(np.median([len(pops.receptor_types[t]) for t in rts_all]))
    n_apl = len(pops.apl)
    del conn

    if smoke:
        odors, rts, seeds = odors_all[:8], rts_all[:4], ALL51_SEEDS_8[:2]
        chunk, workers, out_dir = 2, 4, "results/m0d/diag/smoke"
    else:
        odors, rts, seeds = odors_all, rts_all, ALL51_SEEDS_8
        chunk, workers, out_dir = 6, 16, "results/m0d/diag"

    t_sim0 = time.time()
    with FlyPool(NPZ, Params(), [{} for _ in range(workers)], workers=workers) as pool:
        # ---- engine A over the extended set (also fixes R̄)
        a_pres = []
        for part in pool.run_jobs(job_ref, [dict(params=CAND_PARAMS, scale=CAND_SCALE, odors=odors[c:c + chunk])
                                            for c in range(0, len(odors), chunk)]):
            a_pres.extend(part)
        r_bar = float(np.mean([p["release_frac"] for p in a_pres]))
        release_a = float(np.mean([p["release_mean"] for p in a_pres]))
        r_max_b = 2.0 * r_bar * 0.333
        b_params = dict(CAND_PARAMS, apl_slope=1e9, apl_r_max=r_max_b)

        # ---- engine B: release clamped to A's mean
        b_pres = []
        for part in pool.run_jobs(job_ref, [dict(params=b_params, scale=CAND_SCALE, odors=odors[c:c + chunk])
                                            for c in range(0, len(odors), chunk)]):
            b_pres.extend(part)
        release_b = float(np.mean([p["release_mean"] for p in b_pres]))
        rel_diff = abs(release_b - release_a) / release_a if release_a else float("inf")
        match_ok = bool(rel_diff <= RELEASE_TOL)

        # ---- all51 for both engines, 8 seeds
        items = [(r, s) for r in rts for s in seeds]
        b_chunk = max(1, len(items) // workers + (1 if len(items) % workers else 0))
        all51 = {}
        for name, prm in (("A", CAND_PARAMS), ("B", b_params)):
            jobs = [dict(params=prm, scale=CAND_SCALE, items=items[i:i + b_chunk], c_norm=c_norm)
                    for i in range(0, len(items), b_chunk)]
            all51[name] = [r for part in pool.run_jobs(job_all51, jobs) for r in part]
    t_sim = time.time() - t_sim0

    # ---- statistics (metric definitions identical to two_checks.py)
    names_a, kc_a = per_odor_kc_pct(a_pres)
    names_b, kc_b = per_odor_kc_pct(b_pres)
    assert names_a == names_b
    gs = sorted({r["g"] for r in all51["A"]})
    drive = {n: np.array([np.mean([r["kc"] for r in all51[n] if r["g"] == g]) for g in gs]) for n in ("A", "B")}
    vals = dict(odour=(kc_a, kc_b), all51=(drive["A"], drive["B"]))

    metrics = []
    if match_ok:
        for name, primary, cluster, stat in METRICS:
            av, bv = vals[cluster]
            m = boot_ci(av, bv, stat)
            metrics.append(dict(name=name, primary=primary, cluster=cluster,
                                a=float(stat(av)), b=float(stat(bv)), **m))

    wall = time.time() - t0
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    n_pres = len(a_pres) + len(b_pres) + 2 * len(items)
    res = dict(
        meta=dict(npz=NPZ, smoke=smoke, commit=commit, dirty=dirty, strength=STRENGTH, settle_ms=SETTLE_MS,
                  read_steps=READ_STEPS, n_odors=len(odors), ext_seed=EXT_SEED,
                  ext_probe_seed0=EXT_PROBE_SEED0, all51_seeds=list(seeds),
                  boot=dict(draws=BOOT_DRAWS, seed=BOOT_SEED), workers=workers, wall_s=wall, sim_s=t_sim,
                  n_presentations=n_pres, s_per_presentation_worker=t_sim * workers / n_pres,
                  odors=[{k: v for k, v in o.items() if k != "strengths"} for o in odors]),
        prior=prior_metrics(),
        comparison=dict(r_bar=r_bar, r_max_b=float(r_max_b), n_apl=n_apl, n_presentations_a=len(a_pres),
                        n_glom=len(gs), c_norm=c_norm, release_a=release_a, release_b=release_b,
                        release_rel_diff=float(rel_diff), release_match_ok=match_ok, release_tol=RELEASE_TOL,
                        params_a=CAND_PARAMS, params_b=b_params, scale=CAND_SCALE, metrics=metrics,
                        all51_stats=dict(A=all51_stats(all51["A"]), B=all51_stats(all51["B"])),
                        per_odor_kc_pct=dict(odors=names_a, A=kc_a.tolist(), B=kc_b.tolist()),
                        per_glom_drive=dict(glomeruli=gs, A=drive["A"].tolist(), B=drive["B"].tolist()),
                        presentations_a=a_pres, presentations_b=b_pres),
    )
    md = report(res, wall)
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/clamp_control_power.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    with open(f"{out_dir}/clamp_control_power.md", "w") as f:
        f.write(md)
    print(md)
    print(f"sim {t_sim:.1f} s for {n_pres} presentations on {workers} workers; "
          f"{res['meta']['s_per_presentation_worker']:.2f} worker-s/presentation")
    if smoke:
        full_pres = 2 * 2 * N_ODORS_EXT + 2 * 51 * len(ALL51_SEEDS_8)
        proj = res["meta"]["s_per_presentation_worker"] * full_pres / 16 / 60
        print(f"projection for the full run ({full_pres} presentations on 16 workers): {proj:.1f} min "
              f"({'OK' if proj <= MAX_PROJECTED_MIN else 'ABOVE'} the {MAX_PROJECTED_MIN:g} min ceiling)")
