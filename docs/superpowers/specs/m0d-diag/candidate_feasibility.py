"""M0d H.3a: is there an operating point that satisfies D.4 and the readout-floor guard at once? Calibration only.

Diagnostic only: no H.3 selection, no M2 turns, no candidate odours, no oracle, plasticity off.

Why this exists. Two constraints pull in opposite directions and no measured point satisfies both:

- **D.4 at the gate's own protocol** (`kc_sparsity` at its default settle 200 ms) needs the reference-set KC
  median around 5.42-6.94% (`d4_margin_protocol.md`, e936434); the previously preserved candidate
  (`kc_thresh` 1.5, `apl_input_scale` 0.0799, reference median 7.65%) fails the 3-7% clause there (odour A 7.17%).
- **The readout-floor guard** gets harder as KC drive falls: at reference 7.55% MBON13 already sits at
  zero-share 0.188 against a limit of 0.25 (`guard_after_baseline.md`, 5fe7f7b).

The operating-characteristic run (`gate_oc.md`, 744943d) fixed the decision rules used here: **point estimates**
(zero-share <= 0.25 and median delta >= 5; CIs recorded but not gating), **16 seeds** for the MBON baseline, and
a **384-odour** gain-control control. This script walks the five steps of that plan and reports whether a point
exists that passes everything.

Nothing is reimplemented: the odour generators, the CSC-into-APL scaling, the presentation jobs, the D.4
protocol, the MBON baseline bisection protocol, the guard's statistics and the clamped-release control are all
imported from the committed diagnostics (`apl_input_scale_sweep.py` 8b51594, `apl_cap_disparity.py` 780dd13,
`readout_floor_guard.py` cde8bff, `guard_after_baseline.py` 5fe7f7b, `d4_margin_protocol.py` e936434,
`clamp_control_power.py` c2a0f32), so their numbers are unchanged by construction.

Steps:
1. Candidates around the D.4-compatible band: for kc_thresh in {1.55, 1.60, 1.65, 1.70}, bisect
   `apl_input_scale` in [0.005, 1.0] (log, 8 iterations, both ends checked first) to a reference-set median APL
   membrane of 11 mV; accept only if |median - 11| <= 1.0 mV.
2. D.4 at the gate protocol (settle 200 ms, seeds 100-102) for every step-1 candidate plus the old candidate.
3. MBON baseline: bisect `mbon_hold_frac` over [0.5, 1.0], 8 iterations, no early stop, 16 seeds (100-115),
   accept the final interval midpoint; stage-3 check on independent seeds 116-131 against 3-4 Hz.
4. Readout guard at the recalibrated baseline, point rules (48 reference odours x 2 seeds + same-seed resting).
5. Clamped-release gain control at 384 odours for the best surviving candidate; pass = 95% CI upper bound < 0.

Early stops: if nothing survives step 2 (or 3, or 4) the run stops there. `--no-step5` leaves step 5 out.

Writes results/m0d/diag/candidate_feasibility.{json,md}; --smoke runs 1 kc_thresh x 2 bisection steps x 4 odours
into results/m0d/diag/smoke/ and prints a projection of the full run.
"""
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, "docs/superpowers/specs/m0d-diag")

from flymon.brain.circuits import Populations, compartments
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.fly_pool import FlyPool

from apl_cap_disparity import EXCLUDE, all51_stats, job_all51
from apl_input_scale_sweep import NPZ, READ_STEPS, SETTLE_MS, STRENGTH, TARGET_MV, reference_odors
from apl_input_scale_sweep import job as job_sweep
from clamp_control_power import ALL51_SEEDS_8, EXT_PROBE_SEED0, EXT_SEED, RELEASE_TOL, extended_odors, log10_var
from clamp_control_power import zero_count
from d4_margin_protocol import GATE_SETTLE_MS, READ_MS, design_stats, job_design
from guard_after_baseline import (BASELINE_BAND_HZ, BASELINE_TARGET_HZ, BOOT_SEED as GUARD_BOOT_SEED, HOLD_BRACKET,
                                  N_BOOT, REST_MS, SAT_HZ, bootstrap_type, job_baseline, job_stim_kc)
from readout_floor_guard import (CALLOUT, MED_DELTA_MIN, PUNISH_TYPE, REWARD_TYPE, ZERO_SHARE_MAX, job_rest, mark,
                                 mbon_type_index, pool_types, type_stats)
from two_checks import BOOT_DRAWS, BOOT_SEED, D4_BAND, DESIGN_SEEDS, boot_ci, iqr, job_ref, per_odor_kc_pct

# ---- step 1 ------------------------------------------------------------------------------------
APL_R_MAX = 0.333                       # graded APL cap held fixed by the task
KC_THRESHES = (1.55, 1.60, 1.65, 1.70)
S_BRACKET = (0.005, 1.0)                # apl_input_scale bisection interval (log scale)
N_BISECT_S = 8
MV_TOL = 1.0                            # accept a step-1 candidate only if |median mV - 11| <= this
SMOKE_MV_TOL = 5.0                      # smoke only: a loose tolerance, so 2 bisection steps still exercise 2-5
REL_QL = (0.269, 0.731)                 # quasi-linear range of the release sigmoid, as a fraction of apl_r_max

# ---- step 2 ------------------------------------------------------------------------------------
OLD_LABEL = "old G(1.5, 0.0799)"
OLD_PARAMS = dict(apl_mode="graded", apl_r_max=APL_R_MAX, kc_thresh=1.5)
OLD_SCALE = 0.0799
D4_SOURCE = "results/m0d/diag/d4_margin_protocol.json"

# ---- step 3 ------------------------------------------------------------------------------------
N_BISECT_HOLD = 8
CAL_SEEDS = tuple(range(100, 116))      # 16 seeds for the calibration baseline
CHECK_SEEDS = tuple(range(116, 132))    # 16 independent seeds for the stage-3 check

# ---- step 5 ------------------------------------------------------------------------------------
N_ODORS_GC = 384
MAX_TOTAL_MIN = 60.0                    # if the projection exceeds this, step 5 is left out

# full-run unit counts, for the smoke projection
FULL_UNITS = dict(step1=len(KC_THRESHES) * (2 + N_BISECT_S) * 96,
                  step2=6 * (len(KC_THRESHES) + 1),
                  step3=(2 + N_BISECT_HOLD + 2) * len(CAL_SEEDS),
                  step4=2 * 96,
                  step5=2 * 2 * N_ODORS_GC + 2 * 51 * len(ALL51_SEEDS_8))


def fmt(x, n=2):
    return "—" if x is None else f"{x:.{n}f}"


def table(header, rows):
    L = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    L += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return L


# ================================================================ step 1: apl_input_scale bisection
def point_stats(pres, scale):
    """Reference-set summary of one (kc_thresh, s) point, from `apl_input_scale_sweep.job` presentations."""
    v = np.array([p["apl_v_mean"] for p in pres], float)
    kc = 100.0 * np.array([p["kc_active_frac"] for p in pres], float)
    rel = np.array([p["release_frac"] for p in pres], float)
    q1, q3 = (float(x) for x in np.percentile(v, [25, 75]))
    return dict(scale=float(scale), n_pres=len(pres), median_mv=float(np.median(v)), q1_mv=q1, q3_mv=q3,
                iqr_mv=q3 - q1, median_kc_pct=float(np.median(kc)), kc_q1_pct=float(np.percentile(kc, 25)),
                kc_q3_pct=float(np.percentile(kc, 75)), median_release_frac=float(np.median(rel)),
                rel_share_below=float((rel < REL_QL[0]).mean()),
                rel_share_in=float(((rel >= REL_QL[0]) & (rel <= REL_QL[1])).mean()),
                rel_share_above=float((rel > REL_QL[1]).mean()))


def eval_scale(pool, kc_thresh, scale, odors, chunk, acc):
    params = dict(apl_mode="graded", apl_r_max=APL_R_MAX, kc_thresh=float(kc_thresh))
    jobs = [dict(params=params, scale=float(scale), odors=odors[c:c + chunk]) for c in range(0, len(odors), chunk)]
    t0 = time.time()
    pres = [p for part in pool.run_jobs(job_sweep, jobs) for p in part]
    acc("step1", len(pres), time.time() - t0)
    st = point_stats(pres, scale)
    st["presentations"] = pres
    return st


def bisect_scale(pool, kc_thresh, odors, chunk, n_bisect, acc, mv_tol=MV_TOL):
    """Log bisection of `apl_input_scale` for a reference-set median APL membrane of TARGET_MV.

    s multiplies every CSC edge into APL, so the APL membrane *rises* with s (1.07 mV at s = 0.005 to 78.6 mV at
    s = 1 for kc_thresh 1.5, `apl_input_scale_sweep.md`) while the KC active % falls. The bracket check therefore
    requires mv(lo) <= TARGET_MV <= mv(hi). The accepted point is the evaluated s closest to TARGET_MV — it has
    to be an evaluated point, because the acceptance test |median - 11| <= tolerance is a test on a measured
    median.
    """
    lo, hi = S_BRACKET
    ends = {}
    for x in (lo, hi):
        ends[x] = eval_scale(pool, kc_thresh, x, odors, chunk, acc)
        print(f"    kc_thresh {kc_thresh:g}: endpoint s {x:.5g} -> {ends[x]['median_mv']:.2f} mV, "
              f"{ends[x]['median_kc_pct']:.2f}% KC", flush=True)
    bracketed = bool(ends[lo]["median_mv"] <= TARGET_MV <= ends[hi]["median_mv"])
    trace = []
    for i in range(n_bisect):
        mid = float(math.exp(0.5 * (math.log(lo) + math.log(hi))))
        st = eval_scale(pool, kc_thresh, mid, odors, chunk, acc)
        st["step"] = i + 1
        st["bracket_before"] = [lo, hi]
        if st["median_mv"] < TARGET_MV:
            lo = mid
        else:
            hi = mid
        st["bracket_after"] = [lo, hi]
        trace.append(st)
        print(f"    kc_thresh {kc_thresh:g}: bisect {i + 1}/{n_bisect}: s {mid:.5g} -> "
              f"{st['median_mv']:.2f} mV, {st['median_kc_pct']:.2f}% KC", flush=True)
    evaluated = [ends[S_BRACKET[0]], ends[S_BRACKET[1]]] + trace
    best = min(evaluated, key=lambda s: abs(s["median_mv"] - TARGET_MV))
    return dict(kc_thresh=float(kc_thresh), trace=trace, endpoints={f"{x:g}": ends[x] for x in S_BRACKET},
                bracketed=bracketed, final_bracket=[lo, hi], accepted=best,
                accepted_scale=float(best["scale"]), median_mv=float(best["median_mv"]),
                mv_error=float(best["median_mv"] - TARGET_MV),
                mv_tol=float(mv_tol), accepted_ok=bool(abs(best["median_mv"] - TARGET_MV) <= mv_tol))


# ================================================================ step 2: D.4 at the gate protocol
def d4_at_gate(pool, cands, seeds, acc):
    jobs = [dict(params=c["params"], scale=c["scale"], settle_ms=float(GATE_SETTLE_MS), seeds=list(seeds))
            for c in cands]
    t0 = time.time()
    rows = pool.run_jobs(job_design, jobs)
    acc("step2", 2 * len(seeds) * len(cands), time.time() - t0)
    return [design_stats(r) for r in rows]


def old_candidate_reference_median():
    """The old candidate's reference-set median KC %, read from the committed d4_margin_protocol run."""
    try:
        r = json.load(open(D4_SOURCE))
        return float(r["candidate"]["ref_median_kc_pct"]), D4_SOURCE
    except (OSError, KeyError, ValueError, TypeError):
        return None, None


# ================================================================ step 3: MBON baseline, 16 seeds
def measure_baseline(pool, params, scale, hold, seeds, acc):
    """`mbon_baseline_multi` over `seeds` through `guard_after_baseline.job_baseline` (one seed per job)."""
    p = dict(params, mbon_hold_frac=float(hold))
    jobs = [dict(params=p, scale=float(scale), seeds=[int(s)]) for s in seeds]
    t0 = time.time()
    rows = [r for part in pool.run_jobs(job_baseline, jobs) for r in part]
    acc("step3", len(rows), time.time() - t0)
    trimmed = np.array([r["trimmed"] for r in rows], float)
    return dict(hold=float(hold), hz=float(trimmed.mean()), sd=float(trimmed.std(ddof=1)),
                se=float(trimmed.std(ddof=1) / math.sqrt(len(trimmed))), n_seeds=len(trimmed),
                seeds=[int(s) for s in seeds], per_seed=[float(x) for x in trimmed],
                in_band=bool(BASELINE_BAND_HZ[0] <= float(trimmed.mean()) <= BASELINE_BAND_HZ[1]))


def bisect_hold(pool, params, scale, n_bisect, cal_seeds, check_seeds, acc):
    """Bisect `mbon_hold_frac` over HOLD_BRACKET for BASELINE_TARGET_HZ; the baseline rises with the hold.

    No early stop, 16 calibration seeds, and the accepted value is the final interval's midpoint (the rule the
    operating-characteristic run selected). The accepted hold is then measured on the 16 calibration seeds and,
    as the stage-3 check, on 16 independent seeds.
    """
    lo, hi = HOLD_BRACKET
    ends = {}
    for x in (lo, hi):
        ends[x] = measure_baseline(pool, params, scale, x, cal_seeds, acc)
        print(f"    endpoint hold {x:.4f} -> {ends[x]['hz']:.3f} Hz", flush=True)
    bracketed = bool(ends[lo]["hz"] <= BASELINE_TARGET_HZ <= ends[hi]["hz"])
    trace = []
    for i in range(n_bisect):
        mid = 0.5 * (lo + hi)
        st = measure_baseline(pool, params, scale, mid, cal_seeds, acc)
        st["step"] = i + 1
        st["bracket_before"] = [float(lo), float(hi)]
        if st["hz"] < BASELINE_TARGET_HZ:
            lo = mid
        else:
            hi = mid
        st["bracket_after"] = [float(lo), float(hi)]
        trace.append(st)
        print(f"    bisect {i + 1}/{n_bisect}: hold {mid:.5f} -> {st['hz']:.3f} Hz", flush=True)
    hold = 0.5 * (lo + hi)
    cal = measure_baseline(pool, params, scale, hold, cal_seeds, acc)
    chk = measure_baseline(pool, params, scale, hold, check_seeds, acc)
    print(f"    accepted hold {hold:.5f}: calibration {cal['hz']:.3f} Hz, independent-seed check "
          f"{chk['hz']:.3f} Hz ({'pass' if chk['in_band'] else 'FAIL'})", flush=True)
    return dict(trace=trace, endpoints={f"{x:g}": ends[x] for x in HOLD_BRACKET}, bracketed=bracketed,
                final_bracket=[float(lo), float(hi)], hold=float(hold), calibration=cal, check=chk,
                stage3_ok=bool(chk["in_band"]), delta_vs_default=float(hold - Params().mbon_hold_frac))


# ================================================================ step 4: readout guard, point rules
def run_guard(pool, params, scale, odors, seeds, chunk, mbon_types, pool_all, odor_names, acc):
    jobs = [dict(params=params, scale=float(scale), odors=odors[c:c + chunk]) for c in range(0, len(odors), chunk)]
    t0 = time.time()
    stim = [p for part in pool.run_jobs(job_stim_kc, jobs) for p in part]
    r_chunk = max(1, 2 * chunk)
    rjobs = [dict(params=params, scale=float(scale), seeds=seeds[c:c + r_chunk])
             for c in range(0, len(seeds), r_chunk)]
    rest = [p for part in pool.run_jobs(job_rest, rjobs) for p in part]
    acc("step4", len(stim) + len(rest), time.time() - t0)
    rest_by_seed = {p["seed"]: p for p in rest}
    assert set(rest_by_seed) == set(seeds), "resting runs must cover every presentation seed"
    rng = np.random.default_rng(GUARD_BOOT_SEED)
    boot = {n: bootstrap_type(stim, rest_by_seed, n, odor_names, rng) for n in pool_all}
    return dict(types={n: type_stats(stim, rest_by_seed, n) for n in mbon_types}, boot=boot,
                median_kc_active=float(np.median([p["kc_active_frac"] for p in stim])),
                median_kc=float(np.median([p["kc_spikes"] for p in stim])), stim=stim, rest=rest)


# ================================================================ step 5: clamped-release gain control
def gain_control(pool, params, scale, odors, rts, seeds, c_norm, chunk, workers, acc):
    t0 = time.time()
    a_pres = []
    for part in pool.run_jobs(job_ref, [dict(params=params, scale=scale, odors=odors[c:c + chunk])
                                        for c in range(0, len(odors), chunk)]):
        a_pres.extend(part)
    r_bar = float(np.mean([p["release_frac"] for p in a_pres]))
    release_a = float(np.mean([p["release_mean"] for p in a_pres]))
    r_max_b = 2.0 * r_bar * APL_R_MAX
    b_params = dict(params, apl_slope=1e9, apl_r_max=r_max_b)
    b_pres = []
    for part in pool.run_jobs(job_ref, [dict(params=b_params, scale=scale, odors=odors[c:c + chunk])
                                        for c in range(0, len(odors), chunk)]):
        b_pres.extend(part)
    release_b = float(np.mean([p["release_mean"] for p in b_pres]))
    rel_diff = abs(release_b - release_a) / release_a if release_a else float("inf")
    match_ok = bool(rel_diff <= RELEASE_TOL)

    items = [(r, s) for r in rts for s in seeds]
    b_chunk = max(1, len(items) // workers + (1 if len(items) % workers else 0))
    all51 = {}
    for name, prm in (("A", params), ("B", b_params)):
        jobs = [dict(params=prm, scale=scale, items=items[i:i + b_chunk], c_norm=c_norm)
                for i in range(0, len(items), b_chunk)]
        all51[name] = [r for part in pool.run_jobs(job_all51, jobs) for r in part]
    acc("step5", len(a_pres) + len(b_pres) + 2 * len(items), time.time() - t0)

    names_a, kc_a = per_odor_kc_pct(a_pres)
    names_b, kc_b = per_odor_kc_pct(b_pres)
    assert names_a == names_b
    gs = sorted({r["g"] for r in all51["A"]})
    drive = {n: np.array([np.mean([r["kc"] for r in all51[n] if r["g"] == g]) for g in gs]) for n in ("A", "B")}
    vals = dict(odour=(kc_a, kc_b), all51=(drive["A"], drive["B"]))
    metrics = []
    if match_ok:
        for name, primary, cluster, stat in (("odour-to-odour IQR of KC active %", True, "odour", iqr),
                                             ("all51 log10 variance of per-glomerulus KC drive", False, "all51",
                                              log10_var),
                                             ("all51 zero-drive glomerulus count", False, "all51", zero_count),
                                             ("median KC active % (extended set)", False, "odour",
                                              lambda x: float(np.median(x)))):
            av, bv = vals[cluster]
            m = boot_ci(av, bv, stat)
            metrics.append(dict(name=name, primary=primary, cluster=cluster, a=float(stat(av)), b=float(stat(bv)),
                                sign_correct_pass=bool(m["ci"][1] < 0.0), **m))
    prim = [m for m in metrics if m["primary"]]
    return dict(r_bar=r_bar, r_max_b=float(r_max_b), n_odors=len(odors), n_presentations_a=len(a_pres),
                n_glom=len(gs), c_norm=float(c_norm), release_a=release_a, release_b=release_b,
                release_rel_diff=float(rel_diff), release_match_ok=match_ok, release_tol=RELEASE_TOL,
                params_a=params, params_b=b_params, scale=float(scale), all51_seeds=[int(s) for s in seeds],
                metrics=metrics, all51_stats=dict(A=all51_stats(all51["A"]), B=all51_stats(all51["B"])),
                per_odor_kc_pct=dict(odors=names_a, A=kc_a.tolist(), B=kc_b.tolist()),
                per_glom_drive=dict(glomeruli=gs, A=drive["A"].tolist(), B=drive["B"].tolist()),
                passes=bool(match_ok and prim and prim[0]["ci"][1] < 0.0),
                wall_s=time.time() - t0)


# ================================================================ report
def report(res):
    m = res["meta"]
    L = ["# M0d H.3a: joint feasibility of D.4 and the readout-floor guard", "",
         f"Diagnostic calibration only (not an H.3 selection; no M2 turns, candidate odours or oracle; plasticity "
         f"off). The question: **does an operating point exist that passes the D.4 gate at its own protocol and the "
         f"readout-floor guard at the same time?** Decision rules are the ones the operating-characteristic run "
         f"(`gate_oc.md`) selected: point estimates for the guard (zero-share <= {ZERO_SHARE_MAX:g} and median "
         f"delta >= {MED_DELTA_MIN:g}, CIs recorded but not gating), {len(CAL_SEEDS)} seeds for the MBON baseline "
         f"with the final-bracket-midpoint rule, and a {N_ODORS_GC}-odour gain-control control with the "
         f"sign-correct rule. Every protocol, odour generator and statistic is imported from the committed "
         f"diagnostics, so their numbers are reproduced rather than recomputed by hand. "
         f"Commit {m['commit']} (dirty tree: {str(m['dirty']).lower()}), wall time {m['wall_s'] / 60:.1f} min on "
         f"{m['workers']} workers.", ""]

    # ---- step 1
    s1 = res["step1"]
    L += ["## Step 1 — candidates around the D.4-compatible band", "",
          f"Graded APL, `apl_r_max` {APL_R_MAX:g}. For each `kc_thresh`, `apl_input_scale` is bisected on a log "
          f"scale in [{S_BRACKET[0]:g}, {S_BRACKET[1]:g}] ({N_BISECT_S} iterations, both ends checked first) to "
          f"bring the reference-set median APL membrane to {TARGET_MV:g} mV; a candidate is accepted only if "
          f"|median − {TARGET_MV:g}| <= {m['mv_tol']:g} mV. The accepted point is the evaluated s closest to "
          f"{TARGET_MV:g} mV (the acceptance test is a test on a measured median, so it has to be an evaluated "
          f"point). Reference set: {m['n_odors']} odours x 2 seeds, strength {STRENGTH}, settle {SETTLE_MS} ms, "
          f"read {READ_STEPS} steps ({m['n_odors'] * 2} presentations per evaluation). Release occupancy is the "
          f"share of presentations whose mean release fraction lies in the sigmoid's quasi-linear range "
          f"[{REL_QL[0]:g}, {REL_QL[1]:g}] of `apl_r_max`, with the two tails (same convention as the "
          f"`apl_input_scale_sweep` release-share columns).", ""]
    L += table(["kc_thresh", "bracketed?", "accepted s", "median APL mV", "|Δ| vs 11 mV", "IQR APL mV",
                "reference median KC %", "median release frac", f"rel < {REL_QL[0]:g}",
                f"rel in [{REL_QL[0]:g}, {REL_QL[1]:g}]", f"rel > {REL_QL[1]:g}", "accepted?"],
               [[f"{b['kc_thresh']:g}", "yes" if b["bracketed"] else "NO", f"{b['accepted_scale']:.5g}",
                 fmt(b["accepted"]["median_mv"]), fmt(abs(b["mv_error"])), fmt(b["accepted"]["iqr_mv"]),
                 fmt(b["accepted"]["median_kc_pct"]), fmt(b["accepted"]["median_release_frac"], 3),
                 fmt(b["accepted"]["rel_share_below"], 3), fmt(b["accepted"]["rel_share_in"], 3),
                 fmt(b["accepted"]["rel_share_above"], 3), "**yes**" if b["accepted_ok"] else "no"]
                for b in s1["bisections"]])
    L += ["", "Bisection traces (s → median APL mV / median KC active %):", ""]
    for b in s1["bisections"]:
        L.append(f"- **kc_thresh {b['kc_thresh']:g}**: endpoints " + "; ".join(
            f"{float(k):.5g} → {v['median_mv']:.2f} mV / {v['median_kc_pct']:.2f}%"
            for k, v in b["endpoints"].items()) + ". Bisection: " + "; ".join(
            f"{s['scale']:.5g} → {s['median_mv']:.2f} / {s['median_kc_pct']:.2f}%" for s in b["trace"])
            + f". Final bracket [{b['final_bracket'][0]:.5g}, {b['final_bracket'][1]:.5g}].")
    L += ["", f"{len(s1['candidates'])} of {len(s1['bisections'])} kc_thresh values produced an accepted "
              f"candidate.", ""]

    # ---- step 2
    s2 = res.get("step2")
    L += ["## Step 2 — D.4 at the gate's own protocol (settle 200 ms)", ""]
    if s2 is None:
        L += ["Not run: step 1 accepted no candidate.", ""]
    else:
        L += [f"`design_odor_pair(pops, k=8, seed=0)`, `kc_sparsity` at the gate's default settle "
              f"{GATE_SETTLE_MS:g} ms, read {READ_MS:g} ms, seeds {list(m['design_seeds'])} averaged (as "
              f"`cmd_sparsity` averages its seed rows). D.4: KC active {100 * D4_BAND[0]:g}-{100 * D4_BAND[1]:g}% "
              f"for **both** odours and Jaccard <= chance. `margin` = min(7 − max odour, min odour − 3). The old "
              f"candidate {OLD_LABEL} is carried as a reference row; its reference median is read from "
              f"`{m.get('old_ref_source') or D4_SOURCE}` and not re-measured.", ""]
        L += table(["candidate", "kc_thresh", "s", "reference median KC %", "design A KC %", "design B KC %",
                    "Jaccard", "chance", "chance − J", "J <= chance", "3-7% both", "margin pp", "D.4"],
                   [[r["label"], f"{r['params']['kc_thresh']:g}", f"{r['scale']:.5g}",
                     fmt(r["ref_median_kc_pct"]), fmt(r["d4"]["pct_A"]), fmt(r["d4"]["pct_B"]),
                     fmt(r["d4"]["jaccard"], 4), fmt(r["d4"]["chance"], 4), f"{r['d4']['overlap_margin']:+.4f}",
                     "yes" if r["d4"]["d4_overlap_ok"] else "NO",
                     "yes" if r["d4"]["d4_sparsity_ok"] else "NO", f"{r['d4']['margin_pp']:+.2f}",
                     "**PASS**" if r["d4"]["d4_ok"] else "FAIL"] for r in s2["rows"]])
        L += ["", f"Passing step 2 (new candidates only): "
                  + (", ".join(r["label"] for r in s2["rows"] if r["d4"]["d4_ok"] and not r["is_old"]) or "none")
                  + ".", ""]

    # ---- step 3
    s3 = res.get("step3")
    L += ["## Step 3 — MBON baseline recalibration, 16 seeds", ""]
    if s3 is None:
        L += ["Not run: no candidate passed step 2." if s2 is not None else "Not run: step 1 accepted no "
              "candidate.", ""]
    else:
        L += [f"`mbon_hold_frac` bisected over [{HOLD_BRACKET[0]:g}, {HOLD_BRACKET[1]:g}], {N_BISECT_HOLD} "
              f"iterations, **no early stop**, target {BASELINE_TARGET_HZ:g} Hz under the M0c definition "
              f"(`flymon.brain.measure.mbon_baseline_multi`, {REST_MS:g} ms, cells above {SAT_HZ:g} Hz excluded "
              f"from the mean), measured on {len(CAL_SEEDS)} seeds {CAL_SEEDS[0]}-{CAL_SEEDS[-1]}. The accepted "
              f"value is the **final interval midpoint**. The stage-3 check re-measures the accepted hold on "
              f"{len(CHECK_SEEDS)} independent seeds {CHECK_SEEDS[0]}-{CHECK_SEEDS[-1]} and requires "
              f"{BASELINE_BAND_HZ[0]:g}-{BASELINE_BAND_HZ[1]:g} Hz.", ""]
        L += table(["candidate", "hold @0.5 (Hz)", "hold @1.0 (Hz)", "bracketed?", "accepted `mbon_hold_frac`",
                    "16-seed baseline (Hz)", "SE (Hz)", f"stage-3 check, seeds {CHECK_SEEDS[0]}-{CHECK_SEEDS[-1]}",
                    "SE (Hz)", f"in {BASELINE_BAND_HZ[0]:g}-{BASELINE_BAND_HZ[1]:g} Hz?"],
                   [[r["label"], fmt(r["cal"]["endpoints"]["0.5"]["hz"], 3),
                     fmt(r["cal"]["endpoints"]["1"]["hz"], 3), "yes" if r["cal"]["bracketed"] else "NO",
                     f"**{r['cal']['hold']:.5f}**", fmt(r["cal"]["calibration"]["hz"], 3),
                     fmt(r["cal"]["calibration"]["se"], 3), fmt(r["cal"]["check"]["hz"], 3),
                     fmt(r["cal"]["check"]["se"], 3), "**pass**" if r["cal"]["stage3_ok"] else "**FAIL**"]
                    for r in s3["rows"]])
        L += ["", "Bisection traces (hold → 16-seed baseline Hz):", ""]
        for r in s3["rows"]:
            L.append(f"- **{r['label']}**: " + "; ".join(f"{s['hold']:.5f} → {s['hz']:.3f}"
                                                         for s in r["cal"]["trace"])
                     + f". Final bracket [{r['cal']['final_bracket'][0]:.5f}, "
                       f"{r['cal']['final_bracket'][1]:.5f}], midpoint {r['cal']['hold']:.5f}.")
        L += ["", "Surviving step 3: " + (", ".join(r["label"] for r in s3["rows"] if r["cal"]["stage3_ok"])
                                          or "none") + ".", ""]

    # ---- step 4
    s4 = res.get("step4")
    L += ["## Step 4 — readout-floor guard at the recalibrated baseline, point rules", ""]
    if s4 is None:
        L += ["Not run (an earlier step left nothing to test).", ""]
    else:
        L += [f"Guard protocol exactly as `guard_after_baseline.py`: the H.3 reference set ({m['n_odors']} odours "
              f"x 2 seeds, strength {STRENGTH}, settle {SETTLE_MS} ms, read {READ_STEPS} steps) plus a same-seed "
              f"resting run for each of the {m['n_odors'] * 2} seeds, counts summed per MBON *type* over both "
              f"hemispheres. **Point rule** (the operating-characteristic run's choice): median of (stimulated − "
              f"same-seed resting) >= {MED_DELTA_MIN:g} **and** zero-share <= {ZERO_SHARE_MAX:g}. Odour-cluster "
              f"bootstrap CIs ({N_BOOT} draws, seed {GUARD_BOOT_SEED}) are recorded but do **not** gate. Pools "
              f"from `compartments(conn, pops, Params().core_frac)`: **A ({PUNISH_TYPE} core)** = "
              f"{', '.join(res['pools']['A']) or 'none'}; **P ({REWARD_TYPE} core)** = "
              f"{', '.join(res['pools']['P']) or 'none'}. A candidate passes the guard when at least one type "
              f"passes in **each** pool.", ""]
        for key, title in (("A", f"A — {PUNISH_TYPE} core"), ("P", f"P — {REWARD_TYPE} core")):
            L += ["", f"### Pool {title}", ""]
            L += table(["MBON type", "candidate", "med Δ", "med Δ 95% CI", "zero-share", "zero-share 95% CI",
                        "med stim", "med rest", "point rule"],
                       [[n, r["label"], fmt(r["guard"]["types"][n]["median_delta"], 1),
                         f"[{r['guard']['boot'][n]['median_delta_ci'][0]:.1f}, "
                         f"{r['guard']['boot'][n]['median_delta_ci'][1]:.1f}]",
                         fmt(r["guard"]["types"][n]["zero_share"], 3),
                         f"[{r['guard']['boot'][n]['zero_share_ci'][0]:.2f}, "
                         f"{r['guard']['boot'][n]['zero_share_ci'][1]:.2f}]",
                         fmt(r["guard"]["types"][n]["median_stim"], 1),
                         fmt(r["guard"]["types"][n]["median_rest"], 1),
                         f"**{mark(r['guard']['types'][n]['passes'])}**"]
                        for n in res["pools"][key] for r in s4["rows"]])
        L += ["", "### The G.14.8 readout: MBON05 and MBON13", ""]
        for n in CALLOUT:
            where = [k for k in ("A", "P") if n in res["pools"][k]]
            loc = f"pool {' and '.join(where)}" if where else "neither pool"
            if n not in res["mbon_types"]:
                L.append(f"- **{n}**: not present as an MBON type in this connectome.")
                continue
            cells = []
            for r in s4["rows"]:
                t = r["guard"]["types"][n]
                b = r["guard"]["boot"].get(n)
                ci_m = (f" [{b['median_delta_ci'][0]:.1f}, {b['median_delta_ci'][1]:.1f}]" if b else "")
                ci_z = (f" [{b['zero_share_ci'][0]:.2f}, {b['zero_share_ci'][1]:.2f}]" if b else "")
                cells.append(f"{r['label']} med Δ {t['median_delta']:.1f}{ci_m}, zero-share "
                             f"{t['zero_share']:.3f}{ci_z} → {mark(t['passes'])}")
            L.append(f"- **{n}** ({loc}): " + "; ".join(cells) + ".")
        L += ["", "### Guard verdict", ""]
        L += table(["candidate", "median KC active % (reference, recalibrated)", "passing types in A",
                    "passing types in P", "guard", "D.4 margin pp"],
                   [[r["label"], fmt(100 * r["guard"]["median_kc_active"]),
                     f"{len(r['pass_A'])}/{len(res['pools']['A'])}"
                     + (f" ({', '.join(r['pass_A'])})" if r["pass_A"] else ""),
                     f"{len(r['pass_P'])}/{len(res['pools']['P'])}"
                     + (f" ({', '.join(r['pass_P'])})" if r["pass_P"] else ""),
                     "**PASS**" if r["passes"] else "FAIL", f"{r['margin_pp']:+.2f}"] for r in s4["rows"]])
        L += ["", "Surviving step 4: " + (", ".join(r["label"] for r in s4["rows"] if r["passes"]) or "none")
              + ".", ""]

    # ---- step 5
    s5 = res.get("step5")
    L += [f"## Step 5 — clamped-release gain control, {N_ODORS_GC} odours", ""]
    if s5 is None:
        L += [res.get("step5_note", "Not run (an earlier step left nothing to test)."), ""]
    else:
        L += [f"Best surviving candidate by D.4 margin: **{s5['label']}**. Diagnostic {s5['gc']['n_odors']}-odour "
              f"set: the reference-set generator rule with `np.random.default_rng({EXT_SEED})`, odour j probed "
              f"with seeds ({EXT_PROBE_SEED0} + 2j, {EXT_PROBE_SEED0 + 1} + 2j) — the same extension "
              f"`clamp_control_power.py` used, widened to {N_ODORS_GC}. Engine A is the candidate at its "
              f"recalibrated `mbon_hold_frac`; engine B is identical except `apl_slope = 1e9` (the sigmoid becomes "
              f"a constant 0.5) and `apl_r_max = 2·R̄·{APL_R_MAX:g} = {s5['gc']['r_max_b']:.6g}`, with "
              f"**R̄ = {s5['gc']['r_bar']:.6f}** engine A's mean release fraction over all "
              f"{s5['gc']['n_presentations_a']} presentations x APL cells x {READ_STEPS} read steps.", "",
              f"Release-match validation: A = {s5['gc']['release_a']:.6f}, B = {s5['gc']['release_b']:.6f}, "
              f"relative difference **{100 * s5['gc']['release_rel_diff']:.4f}%** "
              f"({'within' if s5['gc']['release_match_ok'] else 'ABOVE'} the {100 * RELEASE_TOL:g}% tolerance).",
              ""]
        if not s5["gc"]["release_match_ok"]:
            L += [f"**The release match exceeds {100 * RELEASE_TOL:g}%, so the comparison is not reported.**", ""]
        else:
            L += [f"Bootstrap: {BOOT_DRAWS} draws, seed {BOOT_SEED}, clusters resampled with replacement and "
                  f"paired between A and B; the odour metrics resample the {s5['gc']['n_odors']} odours (each "
                  f"odour's 2 seeds averaged first), the all51 metrics resample the {s5['gc']['n_glom']} "
                  f"glomeruli (51 single-glomerulus stimuli, seeds {s5['gc']['all51_seeds']}). **Pass = the 95% "
                  f"CI upper bound of A − B is below 0** (sign-correct rule).", ""]
            L += table(["metric", "A", "B", "A − B", "95% CI", "CI upper < 0?"],
                       [[mm["name"] + (" (primary)" if mm["primary"] else ""), fmt(mm["a"], 4), fmt(mm["b"], 4),
                         fmt(mm["diff"], 4), f"[{mm['ci'][0]:.4f}, {mm['ci'][1]:.4f}]",
                         "**yes**" if mm["sign_correct_pass"] else "no"] for mm in s5["gc"]["metrics"]])
            a51, b51 = s5["gc"]["all51_stats"]["A"], s5["gc"]["all51_stats"]["B"]
            L += ["", f"- all51 log10 variance of per-glomerulus KC drive: A {a51['log10_var']:.4f}, "
                      f"B {b51['log10_var']:.4f}; zero-drive glomerulus count: "
                      f"A {int(sum(1 for v in s5['gc']['per_glom_drive']['A'] if v == 0))}, "
                      f"B {int(sum(1 for v in s5['gc']['per_glom_drive']['B'] if v == 0))} "
                      f"(of {s5['gc']['n_glom']}).",
                  f"- Gain-control verdict: **{'PASS' if s5['gc']['passes'] else 'FAIL'}** by the sign-correct "
                  f"rule.", ""]

    # ---- summary
    L += ["## Summary", ""]
    for k in ("a", "b", "c", "d"):
        L.append(f"- **({k})** {res['answers'][k]}")
    L += ["", f"Wall time of the run: {m['wall_s']:.0f} s ({m['wall_s'] / 60:.1f} min) on {m['workers']} workers. "
              f"Commit {m['commit']}, dirty tree: {str(m['dirty']).lower()}.", ""]
    return "\n".join(L) + "\n"


# ================================================================ driver
def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(x) for x in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def main():
    smoke = "--smoke" in sys.argv
    no_step5 = "--no-step5" in sys.argv
    t0 = time.time()

    cost = {}

    def acc(step, units, seconds):
        c = cost.setdefault(step, dict(units=0, wall_s=0.0))
        c["units"] += int(units)
        c["wall_s"] += float(seconds)

    conn = Connectome.load(NPZ)
    pops = Populations.from_connectome(conn)
    core_frac = Params().core_frac
    comps = compartments(conn, pops, core_frac)
    pools = dict(A=pool_types(conn, comps, PUNISH_TYPE), P=pool_types(conn, comps, REWARD_TYPE))
    mbon_types = sorted(mbon_type_index(conn, pops))
    odors_all = reference_odors(pops)
    ext_all = extended_odors(pops, n=N_ODORS_GC, seed=EXT_SEED)
    rts_all = sorted(t for t in pops.receptor_types if t not in EXCLUDE)
    c_norm = float(np.median([len(pops.receptor_types[t]) for t in rts_all]))
    del conn

    if smoke:
        kc_list, n_bis_s, odors = (KC_THRESHES[0],), 2, odors_all[:4]
        design_seeds, cal_seeds, check_seeds, n_bis_hold = DESIGN_SEEDS[:1], CAL_SEEDS[:4], CHECK_SEEDS[:4], 2
        ext, rts, all51_seeds = ext_all[:8], rts_all[:4], ALL51_SEEDS_8[:2]
        chunk, workers, out_dir, mv_tol = 2, 4, "results/m0d/diag/smoke", SMOKE_MV_TOL
    else:
        kc_list, n_bis_s, odors = KC_THRESHES, N_BISECT_S, odors_all
        design_seeds, cal_seeds, check_seeds, n_bis_hold = DESIGN_SEEDS, CAL_SEEDS, CHECK_SEEDS, N_BISECT_HOLD
        ext, rts, all51_seeds = ext_all, rts_all, ALL51_SEEDS_8
        chunk, workers, out_dir, mv_tol = 6, 16, "results/m0d/diag", MV_TOL
    seeds = [int(s) for o in odors for s in o["seeds"]]
    odor_names = [o["name"] for o in odors]
    pool_all = list(pools["A"]) + list(pools["P"])

    print(f"pools: A ({PUNISH_TYPE} core) = {pools['A']}; P ({REWARD_TYPE} core) = {pools['P']}")
    print(f"step 1: {len(kc_list)} kc_thresh x ({n_bis_s} + 2) evaluations x {len(odors) * 2} presentations",
          flush=True)

    res = dict(pools=pools, mbon_types=mbon_types)
    answers = {}
    step5_note = None

    with FlyPool(NPZ, Params(), [{} for _ in range(workers)], workers=workers,
                 punish_type=PUNISH_TYPE, reward_type=REWARD_TYPE) as pool:
        # ---------------- step 1
        bisections = []
        for k in kc_list:
            print(f"step 1: kc_thresh {k:g}", flush=True)
            bisections.append(bisect_scale(pool, k, odors, chunk, n_bis_s, acc, mv_tol))
        cands = []
        for b in bisections:
            if not b["accepted_ok"]:
                continue
            cands.append(dict(label=f"G({b['kc_thresh']:g}, {b['accepted_scale']:.4g})",
                              params=dict(apl_mode="graded", apl_r_max=APL_R_MAX, kc_thresh=float(b["kc_thresh"])),
                              scale=float(b["accepted_scale"]),
                              ref_median_kc_pct=float(b["accepted"]["median_kc_pct"]),
                              median_mv=float(b["accepted"]["median_mv"]), is_old=False))
        res["step1"] = dict(bisections=bisections, candidates=cands)
        print(f"step 1 accepted {len(cands)} candidate(s): {[c['label'] for c in cands]}", flush=True)

        # ---------------- step 2
        old_ref, old_ref_src = old_candidate_reference_median()
        rows2 = None
        if cands:
            all_c = cands + [dict(label=OLD_LABEL, params=OLD_PARAMS, scale=OLD_SCALE,
                                  ref_median_kc_pct=old_ref, median_mv=None, is_old=True)]
            print(f"step 2: D.4 at settle {GATE_SETTLE_MS:g} ms for {len(all_c)} configs", flush=True)
            stats = d4_at_gate(pool, all_c, design_seeds, acc)
            rows2 = []
            for c, d in zip(all_c, stats):
                rows2.append(dict(c, d4=d))
                print(f"    {c['label']}: A {d['pct_A']:.2f}% B {d['pct_B']:.2f}%, J {d['jaccard']:.4f} vs "
                      f"chance {d['chance']:.4f} -> {'PASS' if d['d4_ok'] else 'FAIL'} "
                      f"(margin {d['margin_pp']:+.2f} pp)", flush=True)
            res["step2"] = dict(rows=rows2, design_seeds=list(design_seeds), settle_ms=float(GATE_SETTLE_MS))
        pass2 = [r for r in (rows2 or []) if r["d4"]["d4_ok"] and not r["is_old"]]

        # ---------------- step 3
        pass3 = []
        if pass2:
            rows3 = []
            for c in pass2:
                print(f"step 3: {c['label']} — bisect mbon_hold_frac on {len(cal_seeds)} seeds", flush=True)
                cal = bisect_hold(pool, c["params"], c["scale"], n_bis_hold, cal_seeds, check_seeds, acc)
                rows3.append(dict(c, cal=cal))
            res["step3"] = dict(rows=rows3, cal_seeds=list(cal_seeds), check_seeds=list(check_seeds))
            pass3 = [r for r in rows3 if r["cal"]["stage3_ok"]]

        # ---------------- step 4
        pass4 = []
        if pass3:
            rows4 = []
            for c in pass3:
                hold = c["cal"]["hold"]
                params = dict(c["params"], mbon_hold_frac=hold)
                print(f"step 4: {c['label']} — guard at hold {hold:.5f}", flush=True)
                g = run_guard(pool, params, c["scale"], odors, seeds, chunk, mbon_types, pool_all, odor_names, acc)
                pa = [n for n in pools["A"] if g["types"][n]["passes"]]
                pp = [n for n in pools["P"] if g["types"][n]["passes"]]
                rows4.append(dict(c, params_run=params, guard=g, pass_A=pa, pass_P=pp,
                                  passes=bool(pa and pp), margin_pp=float(c["d4"]["margin_pp"])))
                print(f"    {c['label']}: A {pa or 'none'}, P {pp or 'none'} -> "
                      f"{'PASS' if (pa and pp) else 'FAIL'}", flush=True)
            res["step4"] = dict(rows=rows4)
            pass4 = [r for r in rows4 if r["passes"]]

        # ---------------- step 5
        if pass4 and not no_step5:
            best = max(pass4, key=lambda r: r["margin_pp"])
            params = dict(best["params"], mbon_hold_frac=best["cal"]["hold"])
            print(f"step 5: gain control at {best['label']} on {len(ext)} odours", flush=True)
            gc = gain_control(pool, params, best["scale"], ext, rts, all51_seeds, c_norm, chunk, workers, acc)
            res["step5"] = dict(label=best["label"], params=params, scale=best["scale"], gc=gc)
            print(f"    release match {100 * gc['release_rel_diff']:.4f}%, "
                  f"{'PASS' if gc['passes'] else 'FAIL'}", flush=True)
        elif pass4 and no_step5:
            step5_note = ("Not run: the smoke projection of the full run exceeded the "
                          f"{MAX_TOTAL_MIN:g}-minute ceiling, so steps 1-4 were run and step 5 left out "
                          "(`--no-step5`).")
        elif not pass4:
            step5_note = "Not run: no candidate survived step 4."
    if step5_note:
        res["step5_note"] = step5_note

    # ---------------- answers
    s5 = res.get("step5")
    if pass4:
        best = max(pass4, key=lambda r: r["margin_pp"])
        answers["a"] = (f"**Yes** — {len(pass4)} operating point(s) pass D.4 (both clauses) at the gate's protocol "
                        f"and the readout-floor guard at the same time: "
                        + ", ".join(r["label"] for r in pass4) + ".")
        answers["b"] = (f"Best point: **{best['label']}** (`kc_thresh` {best['params']['kc_thresh']:g}, "
                        f"`apl_input_scale` {best['scale']:.5g}, `apl_r_max` {APL_R_MAX:g}, `mbon_hold_frac` "
                        f"{best['cal']['hold']:.5f}) — reference median KC {best['ref_median_kc_pct']:.2f}%, APL "
                        f"membrane {best['median_mv']:.2f} mV, D.4 design pair {best['d4']['pct_A']:.2f}% / "
                        f"{best['d4']['pct_B']:.2f}% with margin {best['d4']['margin_pp']:+.2f} pp and "
                        f"chance − J {best['d4']['overlap_margin']:+.4f}, 16-seed baseline "
                        f"{best['cal']['calibration']['hz']:.3f} Hz (stage-3 check "
                        f"{best['cal']['check']['hz']:.3f} Hz), guard passing types A "
                        f"{', '.join(best['pass_A'])} / P {', '.join(best['pass_P'])}.")
        if s5 and s5["gc"]["release_match_ok"] and s5["gc"]["metrics"]:
            prim = [mm for mm in s5["gc"]["metrics"] if mm["primary"]][0]
            answers["c"] = (f"Gain control at {s5['label']} on {s5['gc']['n_odors']} odours: IQR difference "
                            f"{prim['diff']:.4f} with 95% CI [{prim['ci'][0]:.4f}, {prim['ci'][1]:.4f}], so the "
                            f"sign-correct rule (CI upper < 0) is "
                            f"**{'PASSED' if prim['sign_correct_pass'] else 'NOT passed'}**.")
        else:
            answers["c"] = res.get("step5_note", "Gain control was not run.")
        answers["d"] = "Not applicable: a point exists (see (a) and (b))."
    else:
        stop = ("step 1" if not res["step1"]["candidates"] else
                "step 2" if not pass2 else "step 3" if not pass3 else "step 4")
        answers["a"] = f"**No** — no operating point passes both; the run stopped after {stop}."
        if not res["step1"]["candidates"]:
            worst = min(res["step1"]["bisections"], key=lambda b: abs(b["mv_error"]))
            answers["b"] = (f"No candidate reached the {TARGET_MV:g} ± {mv_tol:g} mV membrane target; the closest "
                            f"was kc_thresh {worst['kc_thresh']:g} at s {worst['accepted_scale']:.5g} with "
                            f"{worst['median_mv']:.2f} mV.")
            answers["d"] = (f"The incompatible pair is the membrane target itself and the `kc_thresh` range "
                            f"tried: the best |Δ| vs {TARGET_MV:g} mV was "
                            f"{abs(worst['mv_error']):.2f} mV against a {mv_tol:g} mV tolerance.")
        elif not pass2:
            new = [r for r in res["step2"]["rows"] if not r["is_old"]]
            best = max(new, key=lambda r: r["d4"]["margin_pp"])
            answers["b"] = ("No point passes; the least-failing step-1 candidate was "
                            f"{best['label']} (reference median {best['ref_median_kc_pct']:.2f}%, design pair "
                            f"{best['d4']['pct_A']:.2f}% / {best['d4']['pct_B']:.2f}%, margin "
                            f"{best['d4']['margin_pp']:+.2f} pp, chance − J "
                            f"{best['d4']['overlap_margin']:+.4f}).")
            spars = [r for r in new if not r["d4"]["d4_sparsity_ok"]]
            over = [r for r in new if not r["d4"]["d4_overlap_ok"]]
            answers["d"] = (f"The incompatible pair is the 11 mV APL membrane target (which fixes the "
                            f"reference-set KC median at "
                            f"{min(r['ref_median_kc_pct'] for r in new):.2f}-"
                            f"{max(r['ref_median_kc_pct'] for r in new):.2f}% over the candidates) and D.4 at the "
                            f"gate protocol: {len(spars)}/{len(new)} candidates fail the 3-7% clause and "
                            f"{len(over)}/{len(new)} the overlap clause; the best sparsity margin is "
                            f"{max(r['d4']['margin_pp'] for r in new):+.2f} pp and the best overlap margin "
                            f"{max(r['d4']['overlap_margin'] for r in new):+.4f}.")
        elif not pass3:
            answers["b"] = ("No point passes; " + ", ".join(
                f"{r['label']} reached {r['cal']['calibration']['hz']:.3f} Hz on the calibration seeds and "
                f"{r['cal']['check']['hz']:.3f} Hz on the independent seeds" for r in res["step3"]["rows"]) + ".")
            answers["d"] = (f"The incompatible pair is D.4 at the gate protocol and the "
                            f"{BASELINE_BAND_HZ[0]:g}-{BASELINE_BAND_HZ[1]:g} Hz MBON baseline: every D.4-passing "
                            f"candidate misses the band on independent seeds by at least "
                            f"{min(max(BASELINE_BAND_HZ[0] - r['cal']['check']['hz'], r['cal']['check']['hz'] - BASELINE_BAND_HZ[1]) for r in res['step3']['rows']):.3f} Hz.")
        else:
            rows4 = res["step4"]["rows"]
            answers["b"] = ("No point passes; " + "; ".join(
                f"{r['label']} passes {len(r['pass_A'])} type(s) in A and {len(r['pass_P'])} in P"
                for r in rows4) + ".")
            empt = set()
            for r in rows4:
                if not r["pass_A"]:
                    empt.add("A")
                if not r["pass_P"]:
                    empt.add("P")
            worst_lines = []
            for r in rows4:
                for n in CALLOUT:
                    if n in r["guard"]["types"]:
                        t = r["guard"]["types"][n]
                        worst_lines.append(f"{r['label']} {n} zero-share {t['zero_share']:.3f} "
                                           f"(limit {ZERO_SHARE_MAX:g}), med Δ {t['median_delta']:.1f} "
                                           f"(limit {MED_DELTA_MIN:g})")
            answers["d"] = (f"The incompatible pair is D.4 at the gate protocol and the readout-floor guard: "
                            f"every D.4-passing candidate leaves pool {'/'.join(sorted(empt))} empty under the "
                            f"point rules — " + "; ".join(worst_lines) + ".")
        answers["c"] = res.get("step5_note", "Gain control was not run (no surviving candidate).")

    res["answers"] = answers

    wall = time.time() - t0
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    res["meta"] = dict(npz=NPZ, smoke=smoke, commit=commit, dirty=dirty, strength=STRENGTH, settle_ms=SETTLE_MS,
                       read_steps=READ_STEPS, n_odors=len(odors), n_seeds=len(seeds), kc_threshes=list(kc_list),
                       s_bracket=list(S_BRACKET), n_bisect_scale=n_bis_s, target_mv=TARGET_MV, mv_tol=mv_tol,
                       rel_quasi_linear=list(REL_QL), apl_r_max=APL_R_MAX, design_seeds=list(design_seeds),
                       gate_settle_ms=float(GATE_SETTLE_MS), d4_band=list(D4_BAND), cal_seeds=list(cal_seeds),
                       check_seeds=list(check_seeds), n_bisect_hold=n_bis_hold, hold_bracket=list(HOLD_BRACKET),
                       baseline_target_hz=BASELINE_TARGET_HZ, baseline_band_hz=list(BASELINE_BAND_HZ),
                       rest_ms=REST_MS, sat_hz=SAT_HZ, med_delta_min=MED_DELTA_MIN,
                       zero_share_max=ZERO_SHARE_MAX, n_boot=N_BOOT, guard_boot_seed=GUARD_BOOT_SEED,
                       boot=dict(draws=BOOT_DRAWS, seed=BOOT_SEED), n_odors_gc=len(ext),
                       all51_seeds=list(all51_seeds), core_frac=float(core_frac), workers=workers,
                       no_step5=no_step5, wall_s=wall, cost=cost, old_ref_source=old_ref_src,
                       teachability_evaluated=False, plasticity=False)

    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/candidate_feasibility.json", "w") as f:
        json.dump(jsonable(res), f, indent=1, default=float)
    md = report(res)
    with open(f"{out_dir}/candidate_feasibility.md", "w") as f:
        f.write(md)
    print(md)
    print(f"wrote {out_dir}/candidate_feasibility.json / .md in {wall:.0f} s ({wall / 60:.1f} min) "
          f"on {workers} workers")

    if smoke:
        print("\nprojection for the full run (per-step worker-seconds per unit x full unit counts / 16 workers;"
              " worst case: every kc_thresh survives to steps 3 and 4):")
        mins = {}
        for step, full in FULL_UNITS.items():
            c = cost.get(step)
            if not c or not c["units"]:
                print(f"  {step}: not exercised in the smoke run")
                continue
            per_unit = c["wall_s"] * workers / c["units"]
            n_full = full * len(KC_THRESHES) if step in ("step3", "step4") else full
            mins[step] = per_unit * n_full / 16 / 60
            print(f"  {step}: {per_unit:.2f} worker-s/unit x {n_full} units -> {mins[step]:.1f} min")
        total = sum(mins.values())
        steps14 = total - mins.get("step5", 0.0)
        print(f"  steps 1-4 only: {steps14:.1f} min; TOTAL with step 5: {total:.1f} min "
              f"({'OK' if total <= MAX_TOTAL_MIN else 'ABOVE'} the {MAX_TOTAL_MIN:g} min ceiling)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
