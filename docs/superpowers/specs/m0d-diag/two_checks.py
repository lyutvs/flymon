"""M0d H.3a: two pre-registered checks before the third H.3a draft. Diagnostic only.

Calibration data, not an H.3 selection: no M2 turns, no candidate odours, no oracle, plasticity off.

Check 1 — reference set vs design-pair sparsity (D.4 gate compatibility). Stage 1 of H.3a qualifies an
operating point on the H.3 reference odour set (median KC active 6-8%), while stage 3 applies the spec-5/D.4
sparsity gate to the *design odour pair* and demands 3-7% for both odours. For C0 (spiking, kc_thresh 1.5,
no scaling) and six graded apl_r_max 0.333 points this script measures both protocols and reports the
conversion ratio (design-pair mean over the two odours) / (reference-set median), the D.4 verdict per config,
and — per the interpretation rule fixed before the run — the reference-set KC % band whose design-pair image
lands inside 3-7% with at least 0.5 pp margin on both sides.

Check 2 — gain-control control. At the preserved candidate (graded, apl_r_max 0.333, kc_thresh 1.5,
apl_input_scale 0.0799): engine A is the calibrated graded APL; engine B clamps the release to a constant
equal to A's mean release (apl_slope 1e9 makes the sigmoid a constant 0.5, apl_r_max = 2 * R_bar * 0.333,
R_bar = A's mean release fraction over all presentations, cells and read steps). Primary metric (fixed before
the run): the odour-to-odour IQR of KC active % on the reference set. Secondary: all51 log10 variance of
per-glomerulus KC drive, all51 zero-drive glomerulus count, median KC active %. "Within noise" = the 95%
bootstrap CI of the A - B difference contains 0 (10,000 draws, seed 20260918, clusters resampled paired).

Definitions are imported, not re-invented: the reference odour set, the CSC-into-APL scaling and the all51
protocol come from apl_input_scale_sweep.py / apl_cap_disparity.py; the D.4 sparsity/overlap definition comes
from the M0 gate code (flymon.brain.measure.kc_sparsity / jaccard / chance_jaccard, gate thresholds as in
scripts/write_m0_summary.gate_ok, design pair from flymon.brain.stimuli.design_odor_pair).

Writes results/m0d/diag/two_checks.json (raw) and .md (report); --smoke runs a reduced version into
results/m0d/diag/smoke/.
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
from flymon.brain.measure import chance_jaccard, jaccard
from flymon.brain.stimuli import design_odor_pair, present

from apl_cap_disparity import ALL51_SEEDS, EXCLUDE, all51_stats, job_all51
from apl_input_scale_sweep import NPZ, READ_STEPS, SETTLE_MS, STRENGTH, make_engine, reference_odors

# The six graded points are sweep grid values (geomspace(0.005, 1, 10)) except the preserved candidate
# 0.0799 (apl_cap_disparity.CAP0_SCALE, the sweep's interpolated s at 11 mV). Exact values, not the
# 4-significant-digit labels, so the reference-set numbers reproduce the earlier runs bit for bit.
S = tuple(float(s) for s in np.geomspace(0.005, 1.0, 10))
CANDIDATE = (1.5, 0.0799)
CONFIGS = [
    ("C0", dict(kc_thresh=1.5), 1.0),
    ("G(1.5, 0.0527)", dict(apl_mode="graded", apl_r_max=0.333, kc_thresh=1.5), S[4]),
    ("G(1.5, 0.0799)", dict(apl_mode="graded", apl_r_max=0.333, kc_thresh=1.5), 0.0799),
    ("G(1.5, 0.0949)", dict(apl_mode="graded", apl_r_max=0.333, kc_thresh=1.5), S[5]),
    ("G(1.5, 0.1710)", dict(apl_mode="graded", apl_r_max=0.333, kc_thresh=1.5), S[6]),
    ("G(1.25, 0.0292)", dict(apl_mode="graded", apl_r_max=0.333, kc_thresh=1.25), S[3]),
    ("G(1.25, 0.0527)", dict(apl_mode="graded", apl_r_max=0.333, kc_thresh=1.25), S[4]),
]
CAND_LABEL = "G(1.5, 0.0799)"
DESIGN_SEEDS = (100, 101, 102)
DESIGN_K, DESIGN_ODOR_SEED = 8, 0
D4_BAND = (0.03, 0.07)          # scripts/write_m0_summary.gate_ok, spec 5 / D.4
D4_MARGIN_PP = 0.5              # the interpretation rule's margin, percentage points
BOOT_DRAWS, BOOT_SEED = 10_000, 20260918
PRIOR = {"sweep": "results/m0d/diag/apl_input_scale_sweep.json",
         "cap": "results/m0d/diag/apl_cap_disparity.json"}


# ---- jobs (module level so spawn can pickle them) ----------------------------------------------
def job_ref(eng, pl, pops, comps, ro, params, scale, odors):
    """One reference-odour block: KC active fraction and the mean APL release per cell per read step."""
    e, _ = make_engine(eng, pops, params, scale)
    graded = e.p.apl_mode == "graded"
    n_apl = len(pops.apl)
    out = []
    for o in odors:
        for seed in o["seeds"]:
            e.reset(seed); e.clear_drive(); present(e, pops, o["strengths"], STRENGTH); e.run(SETTLE_MS)
            counts = np.zeros(e.N, np.int32)
            rel = 0.0
            for _ in range(READ_STEPS):
                counts[e.step()] += 1
                if graded:
                    rel += float(e.apl_release(e.v[pops.apl]).astype(np.float64).sum())
            kc = counts[pops.kc]
            out.append(dict(odor=o["name"], seed=int(seed), kc_active_frac=float((kc > 0).mean()),
                            kc_spikes=int(kc.sum()),
                            release_mean=(rel / (READ_STEPS * n_apl)) if graded else None,
                            release_frac=(rel / (READ_STEPS * n_apl) / e.p.apl_r_max) if graded else None))
    return out


def job_design(eng, pl, pops, comps, ro, params, scale, seeds):
    """The M0 sparsity/overlap protocol on the design odour pair (flymon.brain.measure definitions),
    with the M0d read window: settle 800 ms, read 600 steps."""
    e, _ = make_engine(eng, pops, params, scale)
    a, b = design_odor_pair(pops, k=DESIGN_K, seed=DESIGN_ODOR_SEED)
    out = []
    for seed in seeds:
        act = []
        for odor in (a, b):
            e.reset(int(seed)); e.clear_drive(); present(e, pops, odor, STRENGTH); e.run(SETTLE_MS)
            counts = np.zeros(e.N, np.int32)
            for _ in range(READ_STEPS):
                counts[e.step()] += 1
            act.append(counts[pops.kc] > 0)
        fa, fb = float(act[0].mean()), float(act[1].mean())
        out.append(dict(seed=int(seed), frac_active_A=fa, frac_active_B=fb,
                        jaccard=jaccard(act[0], act[1]), chance=chance_jaccard(fa, fb)))
    return out


# ---- statistics --------------------------------------------------------------------------------
def per_odor_kc_pct(pres):
    """KC active % per odour, averaging that odour's seeds (the bootstrap cluster)."""
    names = sorted({p["odor"] for p in pres})
    return names, np.array([100 * np.mean([p["kc_active_frac"] for p in pres if p["odor"] == n]) for n in names])


def iqr(x):
    q1, q3 = np.percentile(np.asarray(x, float), [25, 75])
    return float(q3 - q1)


def boot_ci(a, b, stat, draws=BOOT_DRAWS, seed=BOOT_SEED):
    """95% percentile CI of stat(A) - stat(B), resampling the shared clusters with replacement (paired)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, size=(draws, n))
    d = np.array([stat(a[i]) - stat(b[i]) for i in idx])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(diff=float(stat(a) - stat(b)), ci=[float(lo), float(hi)], within_noise=bool(lo <= 0.0 <= hi),
                n_clusters=int(n), draws=int(draws), seed=int(seed))


def design_stats(rows):
    """Mean over seeds, as scripts/reproduce_flybrain_measurements.cmd_sparsity averages its seed rows."""
    m = {k: float(np.mean([r[k] for r in rows])) for k in ("frac_active_A", "frac_active_B", "jaccard", "chance")}
    m["d4_sparsity_ok"] = bool(D4_BAND[0] <= m["frac_active_A"] <= D4_BAND[1]
                               and D4_BAND[0] <= m["frac_active_B"] <= D4_BAND[1])
    m["d4_overlap_ok"] = bool(m["jaccard"] <= m["chance"])
    m["d4_ok"] = bool(m["d4_sparsity_ok"] and m["d4_overlap_ok"])
    m["per_seed"] = rows
    return m


def prior_reference_kc_pct(label, params, scale):
    """The reference-set median KC active % already on record for this config, if any."""
    try:
        if label == "C0" or label == CAND_LABEL:
            r = json.load(open(PRIOR["cap"]))
            name = "C0" if label == "C0" else "C1@0.333"
            c = r["configs"][name]
            if abs(float(c["scale"]) - scale) < 1e-9:
                return float(c["stats"]["median_kc_pct"]), PRIOR["cap"] + f" [{name}]"
            return None
        r = json.load(open(PRIOR["sweep"]))
        for row in r["rows"]:
            if row["params"]["kc_thresh"] == params["kc_thresh"] and abs(row["scale"] - scale) < 1e-12:
                return 100 * float(np.median([p["kc_active_frac"] for p in row["presentations"]])), \
                       PRIOR["sweep"] + f" [kc_thresh {params['kc_thresh']:g}, s {row['scale']:.4g}]"
    except (OSError, KeyError, ValueError):
        return None
    return None


# ---- report ------------------------------------------------------------------------------------
def report(res, wall_s):
    c1, c2, meta = res["check1"], res["check2"], res["meta"]
    L = ["# M0d H.3a: two pre-registered checks (D.4 conversion, clamped-release control)", "",
         f"Diagnostic only (not an H.3 selection; no M2 turns, candidate odours or oracle; plasticity off). "
         f"Reference set: the H.3 48-odour set (rng 800000, 6-9 glomeruli, DA1/V excluded, strength ∝ 1/receptor "
         f"count), {meta['n_odors']} odours x 2 seeds. Design pair: `design_odor_pair(pops, k={DESIGN_K}, "
         f"seed={DESIGN_ODOR_SEED})`, seeds {list(meta['design_seeds'])}. Both protocols: strength {STRENGTH}, "
         f"settle {SETTLE_MS} ms, read {READ_STEPS} steps. s multiplies every CSC edge into APL. "
         f"Commit {meta['commit']} (dirty tree: {str(meta['dirty']).lower()}).", "",
         "## Check 1 — reference set vs design pair (D.4 gate compatibility)", "",
         "D.4 / spec-5 definitions taken from the M0 gate code: KC active fraction = share of Kenyon cells with "
         "at least one read-window spike, Jaccard over the two active sets, chance = pa·pb/(pa+pb−pa·pb) "
         "(`flymon.brain.measure`), sparsity band 3-7% for **both** odours and Jaccard ≤ chance "
         f"(`scripts/write_m0_summary.gate_ok`). Design-pair rows are the mean over the "
         f"{len(meta['design_seeds'])} seeds, as "
         "`reproduce_flybrain_measurements.py` averages its seed rows.", "",
         "| config | s | reference median KC % | prior on record | design A KC % | design B KC % | design mean KC % |"
         " ratio design/reference | Jaccard | chance | J ≤ chance | D.4 3-7% both |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in c1["configs"]:
        d = r["design"]
        prior = f"{r['prior_kc_pct']:.2f}" + (" ✓" if r["prior_match"] else " ✗") if r["prior_kc_pct"] is not None else "—"
        L.append(f"| {r['label']} | {r['scale']:.4g} | {r['ref_median_kc_pct']:.2f} | {prior} | "
                 f"{100 * d['frac_active_A']:.2f} | {100 * d['frac_active_B']:.2f} | {100 * r['design_mean_frac']:.2f} | "
                 f"{r['ratio']:.3f} | {d['jaccard']:.4f} | {d['chance']:.4f} | {'yes' if d['d4_overlap_ok'] else 'NO'} | "
                 f"{'yes' if d['d4_sparsity_ok'] else 'NO'} |")
    p = c1["prior_check"]
    cand = c1["candidate"]
    band = c1["band"]
    L += ["", f"Reproduction of the numbers already on record: {p['n_matched']}/{p['n_compared']} configs agree with "
              f"their earlier reference-set median to within {p['tol_pp']:g} pp "
              f"(max |Δ| = {p['max_abs_diff_pp']:.4f} pp" + (f", worst {p['worst']}" if p["worst"] else "") + "). "
              f"The six graded points reproduce the sweep and C0/G(1.5, 0.0799) reproduce apl_cap_disparity.", "",
          f"Conversion ratio (design-pair mean over the two odours) / (reference-set median): "
          f"{c1['ratio_min']:.3f}-{c1['ratio_max']:.3f} over the {len(c1['configs'])} configs, "
          f"median {c1['ratio_median']:.3f}; at the preserved candidate {CAND_LABEL} it is {cand['ratio']:.3f}.", "",
          "### The pre-registered rule", "",
          "> If the preserved candidate (1.5, 0.0799) fails D.4's 3-7% on either odour, the stage-1 target band "
          "cannot stay at 6-8% with a 7% target; report the reference-set KC % band whose design-pair image lands "
          "inside 3-7% with at least 0.5 pp margin on both sides, using the measured ratio.", "",
          f"- Preserved candidate: reference median {cand['ref_median_kc_pct']:.2f}%, design pair A "
          f"{100 * cand['design']['frac_active_A']:.2f}% / B {100 * cand['design']['frac_active_B']:.2f}% — D.4 3-7% "
          f"**{'HOLDS' if cand['design']['d4_sparsity_ok'] else 'FAILS'}** on "
          f"{'both odours' if cand['design']['d4_sparsity_ok'] else 'at least one odour'}; Jaccard "
          f"{cand['design']['jaccard']:.4f} vs chance {cand['design']['chance']:.4f} "
          f"({'at or below' if cand['design']['d4_overlap_ok'] else 'ABOVE'} chance).",
          f"- Rule conclusion: {c1['rule_conclusion']}",
          f"- Reference-set KC % band mapping into {100 * D4_BAND[0]:g}-{100 * D4_BAND[1]:g}% with "
          f"{D4_MARGIN_PP:g} pp margin on both sides (target image {band['image_lo_pct']:g}-{band['image_hi_pct']:g}%, "
          f"ratio {band['ratio']:.3f}): **{band['lo_pct']:.2f}-{band['hi_pct']:.2f}%** "
          f"(midpoint {band['mid_pct']:.2f}%). With the extreme ratios measured here "
          f"({c1['ratio_min']:.3f} / {c1['ratio_max']:.3f}) the band would be "
          f"{band['lo_pct_ratio_max']:.2f}-{band['hi_pct_ratio_min']:.2f}% if it must hold for every config.", "",
          "## Check 2 — clamped-release control at the preserved candidate", "",
          f"Engine A: graded APL as calibrated (apl_r_max 0.333, kc_thresh 1.5, s 0.0799). "
          f"Engine B: identical Params except `apl_slope = 1e9` (sigmoid ≡ 0.5) and "
          f"`apl_r_max = 2·R̄·0.333 = {c2['r_max_b']:.6g}`. "
          f"**R̄ = {c2['r_bar']:.6f}** is the mean release fraction over all presentations, cells and read steps "
          f"of engine A on the reference set — the reference population is the {c2['n_presentations_a']} reference-set "
          f"presentations ({meta['n_odors']} odours x 2 seeds) x {c2['n_apl']} APL cells x {READ_STEPS} read steps.", "",
          "### Release-match validation (pre-comparison)", "",
          f"Time-integrated APL→KC input is mean release x read steps x the (identical) APL→KC weights, so the "
          f"engines are compared on mean release per cell per step in spike-equivalents: "
          f"A = {c2['release_a']:.6f}, B = {c2['release_b']:.6f}, relative difference "
          f"**{100 * c2['release_rel_diff']:.3f}%** "
          f"({'within' if c2['release_match_ok'] else 'ABOVE'} the 2% tolerance)."]
    if not c2["release_match_ok"]:
        L += ["", "**The release match exceeds 2%, so the comparison below is not reported.**", ""]
    else:
        L += ["", "### Metrics (A − B, 95% bootstrap CI)", "",
              f"Bootstrap: {BOOT_DRAWS} draws, seed {BOOT_SEED}, clusters resampled with replacement and paired "
              f"between A and B. Reference-set metrics resample the {meta['n_odors']} odours (each odour's 2 seeds "
              f"averaged first); the all51 metrics have no odour clusters, so they resample the "
              f"{c2['n_glom']} glomeruli of the same protocol with the same draws and seed. "
              f"\"Within noise\" = the 95% CI of the difference contains 0.", "",
              "| metric | A | B | A − B | 95% CI | within noise |", "|---|---|---|---|---|---|"]
        for m in c2["metrics"]:
            L.append(f"| {m['name']}{' (primary)' if m['primary'] else ''} | {m['a']:.4f} | {m['b']:.4f} | "
                     f"{m['diff']:.4f} | [{m['ci'][0]:.4f}, {m['ci'][1]:.4f}] | "
                     f"{'yes' if m['within_noise'] else 'NO'} |")
        prim = [m for m in c2["metrics"] if m["primary"]][0]
        L += ["", f"- Rule conclusion: the primary metric ({prim['name']}) difference is "
                  f"{prim['diff']:.4f} with CI [{prim['ci'][0]:.4f}, {prim['ci'][1]:.4f}], so clamping the release "
                  f"**{'does not change' if prim['within_noise'] else 'CHANGES'}** the primary dispersion metric "
                  f"beyond noise. Secondary metrics outside noise: "
                  + (", ".join(m["name"] for m in c2["metrics"] if not m["primary"] and not m["within_noise"]) or "none")
                  + "."]
    L += ["", f"Wall time of the run: {wall_s:.0f} s ({wall_s / 60:.1f} min). Commit {meta['commit']}, "
              f"dirty tree: {str(meta['dirty']).lower()}.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    t0 = time.time()
    conn = Connectome.load(NPZ); pops = Populations.from_connectome(conn)
    odors_all = reference_odors(pops)
    rts_all = sorted(t for t in pops.receptor_types if t not in EXCLUDE)
    c_norm = float(np.median([len(pops.receptor_types[t]) for t in rts_all]))
    n_apl = len(pops.apl)
    del conn

    if smoke:
        configs = [c for c in CONFIGS if c[0] == CAND_LABEL]
        odors, rts, seeds = odors_all[:4], rts_all[:4], DESIGN_SEEDS[:1]
        chunk, workers, out_dir = 2, 4, "results/m0d/diag/smoke"
    else:
        configs = CONFIGS
        odors, rts, seeds = odors_all, rts_all, DESIGN_SEEDS
        chunk, workers, out_dir = 6, 16, "results/m0d/diag"

    with FlyPool(NPZ, Params(), [{} for _ in range(workers)], workers=workers) as pool:
        # ---- check 1: both protocols for every config
        ref_jobs, ref_owner = [], []
        for label, params, scale in configs:
            for c in range(0, len(odors), chunk):
                ref_owner.append(label)
                ref_jobs.append(dict(params=params, scale=scale, odors=odors[c:c + chunk]))
        ref_pres = {}
        for label, part in zip(ref_owner, pool.run_jobs(job_ref, ref_jobs)):
            ref_pres.setdefault(label, []).extend(part)
        design_rows = dict(zip([c[0] for c in configs],
                               pool.run_jobs(job_design, [dict(params=p, scale=s, seeds=list(seeds))
                                                          for _, p, s in configs])))

        c1_rows = []
        for label, params, scale in configs:
            pres = ref_pres[label]
            ref_med = 100 * float(np.median([p["kc_active_frac"] for p in pres]))
            d = design_stats(design_rows[label])
            dmean = 0.5 * (d["frac_active_A"] + d["frac_active_B"])
            prior = prior_reference_kc_pct(label, params, scale)
            c1_rows.append(dict(label=label, params=params, scale=float(scale), ref_median_kc_pct=ref_med,
                                prior_kc_pct=prior[0] if prior else None, prior_source=prior[1] if prior else None,
                                prior_match=bool(prior and abs(prior[0] - ref_med) <= 0.01),
                                design=d, design_mean_frac=float(dmean),
                                ratio=float(100 * dmean / ref_med) if ref_med > 0 else float("nan"),
                                presentations=pres))

        # ---- check 2: engine A is the candidate's reference run; B clamps the release
        cand_label, cand_params, cand_scale = [c for c in configs if c[0] == CAND_LABEL][0]
        a_pres = ref_pres[cand_label]
        r_bar = float(np.mean([p["release_frac"] for p in a_pres]))
        release_a = float(np.mean([p["release_mean"] for p in a_pres]))
        r_max_b = 2.0 * r_bar * 0.333
        b_params = dict(cand_params, apl_slope=1e9, apl_r_max=r_max_b)
        b_pres = []
        for part in pool.run_jobs(job_ref, [dict(params=b_params, scale=cand_scale, odors=odors[c:c + chunk])
                                            for c in range(0, len(odors), chunk)]):
            b_pres.extend(part)
        release_b = float(np.mean([p["release_mean"] for p in b_pres]))
        rel_diff = abs(release_b - release_a) / release_a if release_a else float("inf")
        match_ok = bool(rel_diff <= 0.02)

        # all51 for both engines (secondary metrics)
        items = [(r, seed) for r in rts for seed in ALL51_SEEDS]
        b_chunk = max(1, len(items) // workers + (1 if len(items) % workers else 0))
        all51 = {}
        for name, prm in (("A", cand_params), ("B", b_params)):
            jobs = [dict(params=prm, scale=cand_scale, items=items[i:i + b_chunk], c_norm=c_norm)
                    for i in range(0, len(items), b_chunk)]
            all51[name] = [r for part in pool.run_jobs(job_all51, jobs) for r in part]

    # ---- check 2 statistics
    names_a, kc_a = per_odor_kc_pct(a_pres)
    names_b, kc_b = per_odor_kc_pct(b_pres)
    assert names_a == names_b
    gs = sorted({r["g"] for r in all51["A"]})
    drive = {n: np.array([np.mean([r["kc"] for r in all51[n] if r["g"] == g]) for g in gs]) for n in ("A", "B")}
    st_a, st_b = all51_stats(all51["A"]), all51_stats(all51["B"])

    def log10_var(K):
        return float(np.var(np.log10(np.maximum(np.asarray(K, float), 1))))   # all51_stats' definition

    def zero_count(K):
        return float((np.asarray(K, float) == 0).sum())

    metrics = []
    for name, primary, a_vals, b_vals, stat in (
            ("odour-to-odour IQR of KC active %", True, kc_a, kc_b, iqr),
            ("all51 log10 variance of per-glomerulus KC drive", False, drive["A"], drive["B"], log10_var),
            ("all51 zero-drive glomerulus count", False, drive["A"], drive["B"], zero_count),
            ("median KC active % (reference set)", False, kc_a, kc_b, lambda x: float(np.median(x)))):
        m = boot_ci(a_vals, b_vals, stat)
        metrics.append(dict(name=name, primary=primary, a=float(stat(a_vals)), b=float(stat(b_vals)), **m))

    # ---- check 1 statistics: ratio and the band the rule asks for
    ratios = [r["ratio"] for r in c1_rows]
    cand_row = [r for r in c1_rows if r["label"] == CAND_LABEL][0]
    ratio = cand_row["ratio"]
    img = (100 * D4_BAND[0] + D4_MARGIN_PP, 100 * D4_BAND[1] - D4_MARGIN_PP)
    band = dict(ratio=ratio, image_lo_pct=img[0], image_hi_pct=img[1], lo_pct=img[0] / ratio, hi_pct=img[1] / ratio,
                mid_pct=0.5 * (img[0] + img[1]) / ratio, ratio_min=min(ratios), ratio_max=max(ratios),
                lo_pct_ratio_max=img[0] / max(ratios), hi_pct_ratio_min=img[1] / min(ratios))
    cand_fails = not cand_row["design"]["d4_sparsity_ok"]
    rule_conclusion = (
        f"the preserved candidate FAILS D.4's 3-7% (A {100 * cand_row['design']['frac_active_A']:.2f}%, "
        f"B {100 * cand_row['design']['frac_active_B']:.2f}%), so by the rule fixed before the run the stage-1 "
        f"target band cannot stay at 6-8% with a 7% target; the band below is the reported replacement number."
        if cand_fails else
        f"the preserved candidate PASSES D.4's 3-7% on both odours (A {100 * cand_row['design']['frac_active_A']:.2f}%, "
        f"B {100 * cand_row['design']['frac_active_B']:.2f}%), so the rule's conditional does not fire and the "
        f"stage-1 6-8% band with a 7% target stands; the band below is reported for reference.")

    prior_cmp = [r for r in c1_rows if r["prior_kc_pct"] is not None]
    diffs = [abs(r["prior_kc_pct"] - r["ref_median_kc_pct"]) for r in prior_cmp]
    worst = max(prior_cmp, key=lambda r: abs(r["prior_kc_pct"] - r["ref_median_kc_pct"]))["label"] if prior_cmp else None

    wall = time.time() - t0
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    res = dict(
        meta=dict(npz=NPZ, smoke=smoke, commit=commit, dirty=dirty, strength=STRENGTH, settle_ms=SETTLE_MS,
                  read_steps=READ_STEPS, n_odors=len(odors), design_seeds=list(seeds), design_k=DESIGN_K,
                  design_odor_seed=DESIGN_ODOR_SEED, d4_band=list(D4_BAND), d4_margin_pp=D4_MARGIN_PP,
                  boot=dict(draws=BOOT_DRAWS, seed=BOOT_SEED), workers=workers, wall_s=wall, odors=odors),
        check1=dict(configs=c1_rows, ratio_min=float(min(ratios)), ratio_max=float(max(ratios)),
                    ratio_median=float(np.median(ratios)), candidate=cand_row, band=band,
                    rule_conclusion=rule_conclusion,
                    prior_check=dict(n_compared=len(prior_cmp), tol_pp=0.01,
                                     n_matched=sum(1 for r in prior_cmp if r["prior_match"]),
                                     max_abs_diff_pp=float(max(diffs)) if diffs else 0.0, worst=worst)),
        check2=dict(r_bar=r_bar, r_max_b=float(r_max_b), n_apl=n_apl, n_presentations_a=len(a_pres),
                    n_glom=len(gs), release_a=release_a, release_b=release_b, release_rel_diff=float(rel_diff),
                    release_match_ok=match_ok, params_a=cand_params, params_b=b_params, scale=float(cand_scale),
                    metrics=metrics, all51_stats=dict(A=st_a, B=st_b),
                    per_odor_kc_pct=dict(odors=names_a, A=kc_a.tolist(), B=kc_b.tolist()),
                    per_glom_drive=dict(glomeruli=gs, A=drive["A"].tolist(), B=drive["B"].tolist()),
                    presentations_b=b_pres),
    )
    md = report(res, wall)
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/two_checks.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    with open(f"{out_dir}/two_checks.md", "w") as f:
        f.write(md)
    print(md)
    n_pres = sum(len(r["presentations"]) for r in c1_rows) + len(b_pres)
    print(f"wall {wall:.1f} s; {n_pres} reference presentations + {2 * len(seeds) * len(configs)} design-pair + "
          f"{2 * len(items)} all51 presentations on {workers} workers")
