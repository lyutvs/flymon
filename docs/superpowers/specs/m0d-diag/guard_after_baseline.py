"""M0d H.3a calibration: the readout-floor guard re-run AFTER per-configuration MBON baseline recalibration.

Diagnostic only: no M2 turns, no candidate odours, no oracle, plasticity off.

Why this exists. `readout_floor_guard.py` (cde8bff) decided `apl_r_max` = 0.333 with `mbon_hold_frac` left at its
default 0.85 — i.e. before H.3 stage 2 rebisects that value to a 3.5 Hz MBON baseline. Spec H.1 defines C1 as
"graded APL **+ MBON rebalancing**", so the cap-1.0 rejection was decided on an engine that had not been
rebalanced. This run redoes the guard at each configuration's own recalibrated baseline.

Step 1 — spec H.3 stage 2, verbatim: bisect `mbon_hold_frac` over [0.5, 1.0], 8 iterations, targeting a trimmed
MBON baseline of 3.5 Hz under the M0c definition (`flymon.brain.measure.mbon_baseline_multi`, seeds 100-107,
3000 ms, cells above 100 Hz excluded from the mean). Outside 3-4 Hz = that configuration fails stage 2.
Step 2 — the readout-floor guard protocol of `readout_floor_guard.py` at the recalibrated `mbon_hold_frac`.
Step 3 — 95% bootstrap CIs (odour-clustered, 10,000 draws, seed 20260918) for the zero-share and the median
difference of every MBON type in pools A and P.
Step 4 — median KC active % on the reference set at the recalibrated `mbon_hold_frac`.

Writes results/m0d/diag/guard_after_baseline.{json,md}; --smoke runs 1 config x 2 bisection steps x 4 odours
into results/m0d/diag/smoke/.
"""
import json
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
from flymon.brain.measure import mbon_baseline_multi
from flymon.brain.stimuli import present

from apl_input_scale_sweep import NPZ, READ_STEPS, SETTLE_MS, STRENGTH, make_engine, reference_odors
from readout_floor_guard import (CALLOUT, CAP1_KC_THRESH, CAP1_SCALE, CAP0_SCALE, CONFIGS, MED_DELTA_MIN,
                                 PUNISH_TYPE, REWARD_TYPE, ZERO_SHARE_MAX, job_rest, mark, mbon_type_index,
                                 pool_types, type_stats)

# ---- H.3 stage 2 (spec, verbatim) --------------------------------------------------------------
BASELINE_TARGET_HZ = 3.5                 # bisection target
BASELINE_BAND_HZ = (3.0, 4.0)            # outside the band = stage 2 failure
HOLD_BRACKET = (0.5, 1.0)                # bisection interval
N_BISECT = 8                             # bisection iterations
REST_SEEDS = tuple(range(100, 108))      # M0c definition: seeds 100-107
REST_MS = 3000.0                         # M0c definition: 3 s
SAT_HZ = 100.0                           # mbon_hz_trimmed: cells above 100 Hz excluded
DEFAULT_HOLD = Params().mbon_hold_frac   # 0.85, the value the first guard run used

# ---- step 3 ------------------------------------------------------------------------------------
N_BOOT = 10_000
BOOT_SEED = 20260918
KC_SHIFT_CALLOUT_PP = 0.3                # step 4: flag a KC active % move larger than this


# ---- jobs (module level so spawn can pickle them) ----------------------------------------------
def job_baseline(eng, pl, pops, comps, ro, params, scale, seeds):
    """M0c resting MBON baseline for these seeds, through `mbon_baseline_multi` itself (no reimplementation)."""
    e, _ = make_engine(eng, pops, params, scale)
    out = []
    for s in seeds:
        r = mbon_baseline_multi(e, pops, [int(s)], ms=REST_MS, sat_hz=SAT_HZ)
        out.append(dict(seed=int(s), trimmed=float(r["mbon_hz_rest_trimmed"]), raw=float(r["mbon_hz_rest"]),
                        n_saturated=float(r["mbon_n_saturated"]), n_types_active=float(r["mbon_types_active_rest"])))
    return out


def job_stim_kc(eng, pl, pops, comps, ro, params, scale, odors):
    """`readout_floor_guard.job_stim`'s protocol, plus the KC active fraction needed by step 4.

    The engine call sequence (reset/clear_drive/present/run/step x READ_STEPS) is identical to job_stim, so
    the per-type counts are bit-identical; `kc_active_frac` is derived from the same counts afterwards.
    """
    e, _ = make_engine(eng, pops, params, scale)
    mt = mbon_type_index(eng.conn, pops)
    out = []
    for o in odors:
        for seed in o["seeds"]:
            e.reset(int(seed)); e.clear_drive(); present(e, pops, o["strengths"], STRENGTH); e.run(SETTLE_MS)
            counts = np.zeros(e.N, np.int32)
            for _ in range(READ_STEPS):
                counts[e.step()] += 1
            kc = counts[pops.kc]
            out.append(dict(odor=o["name"], seed=int(seed), kc_spikes=int(kc.sum()),
                            kc_active_frac=float((kc > 0).mean()),
                            types={n: int(counts[idx].sum()) for n, idx in mt.items()}))
    return out


# ---- step 1: bisection -------------------------------------------------------------------------
def measure_baseline(pool, params, scale, hold):
    """`mbon_baseline_multi` over REST_SEEDS, one seed per worker; the statistic is the mean of the per-seed
    trimmed rates, exactly as `mbon_baseline_multi` aggregates them."""
    p = dict(params, mbon_hold_frac=float(hold))
    jobs = [dict(params=p, scale=scale, seeds=[s]) for s in REST_SEEDS]
    rows = [r for part in pool.run_jobs(job_baseline, jobs) for r in part]
    trimmed = np.array([r["trimmed"] for r in rows], float)
    return dict(hold=float(hold), hz=float(trimmed.mean()), sd=float(trimmed.std()),
                per_seed=[float(x) for x in trimmed],
                raw_hz=float(np.mean([r["raw"] for r in rows])),
                n_saturated=float(np.mean([r["n_saturated"] for r in rows])),
                n_types_active=float(np.mean([r["n_types_active"] for r in rows])))


def bisect_hold(pool, params, scale, n_bisect):
    """Bisect `mbon_hold_frac` over HOLD_BRACKET for BASELINE_TARGET_HZ; the baseline rises with the hold."""
    lo, hi = HOLD_BRACKET
    trace = []
    ends = {}
    for x in (lo, hi):
        e = measure_baseline(pool, params, scale, x)
        ends[x] = e
        print(f"    endpoint hold {x:.4f} -> {e['hz']:.3f} Hz", flush=True)
    bracketed = bool(ends[lo]["hz"] <= BASELINE_TARGET_HZ <= ends[hi]["hz"])
    for i in range(n_bisect):
        mid = 0.5 * (lo + hi)
        st = measure_baseline(pool, params, scale, mid)
        st["step"] = i + 1
        st["bracket_before"] = [float(lo), float(hi)]
        if st["hz"] < BASELINE_TARGET_HZ:
            lo = mid
        else:
            hi = mid
        st["bracket_after"] = [float(lo), float(hi)]
        trace.append(st)
        print(f"    bisect {i + 1}/{n_bisect}: hold {mid:.5f} -> {st['hz']:.3f} Hz", flush=True)
    best = min(trace, key=lambda s: abs(s["hz"] - BASELINE_TARGET_HZ))
    return dict(trace=trace, endpoints={f"{x:g}": ends[x] for x in HOLD_BRACKET}, bracketed=bracketed,
                final_bracket=[float(lo), float(hi)], hold=float(best["hold"]), baseline_hz=float(best["hz"]),
                baseline_sd=float(best["sd"]), per_seed=best["per_seed"],
                in_band=bool(BASELINE_BAND_HZ[0] <= best["hz"] <= BASELINE_BAND_HZ[1]),
                delta_vs_default=float(best["hold"] - DEFAULT_HOLD))


# ---- step 2 ------------------------------------------------------------------------------------
def run_config(pool, params, scale, odors, seeds, chunk):
    """`readout_floor_guard.run_config`, with job_stim_kc in place of job_stim (same protocol + KC active %)."""
    jobs = [dict(params=params, scale=scale, odors=odors[c:c + chunk]) for c in range(0, len(odors), chunk)]
    stim = [p for part in pool.run_jobs(job_stim_kc, jobs) for p in part]
    r_chunk = max(1, 2 * chunk)
    rjobs = [dict(params=params, scale=scale, seeds=seeds[c:c + r_chunk]) for c in range(0, len(seeds), r_chunk)]
    rest = [p for part in pool.run_jobs(job_rest, rjobs) for p in part]
    return stim, rest


# ---- step 3: odour-clustered bootstrap ---------------------------------------------------------
def bootstrap_type(stim, rest_by_seed, name, odor_names, rng):
    """95% percentile CIs for the zero-share and the median difference, resampling the 48 odour clusters."""
    by_odor = {o: [] for o in odor_names}
    for p in stim:
        by_odor[p["odor"]].append((float(p["types"][name]),
                                   float(rest_by_seed[p["seed"]]["types"][name])))
    cl_s = np.array([[v[0] for v in by_odor[o]] for o in odor_names], float)   # (n_odors, n_pres)
    cl_r = np.array([[v[1] for v in by_odor[o]] for o in odor_names], float)
    n = len(odor_names)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    s = cl_s[idx].reshape(N_BOOT, -1)
    d = (cl_s - cl_r)[idx].reshape(N_BOOT, -1)
    zero = (s == 0).mean(axis=1)
    med = np.median(d, axis=1)
    zl, zh = (float(x) for x in np.percentile(zero, [2.5, 97.5]))
    ml, mh = (float(x) for x in np.percentile(med, [2.5, 97.5]))
    return dict(zero_share_ci=[zl, zh], zero_share_boot_mean=float(zero.mean()),
                limit_inside_zero_ci=bool(zl <= ZERO_SHARE_MAX <= zh),
                median_delta_ci=[ml, mh], median_delta_boot_mean=float(med.mean()),
                limit_inside_median_ci=bool(ml <= MED_DELTA_MIN <= mh),
                n_clusters=n, n_boot=N_BOOT, boot_seed=BOOT_SEED)


# ---- report ------------------------------------------------------------------------------------
def ci(lo, hi, fmt="{:.2f}"):
    return f"[{fmt.format(lo)}, {fmt.format(hi)}]"


def pool_block(L, title, types, res, configs):
    L += ["", f"### Pool {title}", "",
          "| MBON type | config | med Δ | med Δ 95% CI | zero-share | zero-share 95% CI | 0.25 inside CI? | "
          "med stim | med rest | flag |", "|---|---|---|---|---|---|---|---|---|---|"]
    for n in types:
        for c in configs:
            st = res[c]["types"][n]
            b = res[c]["boot"][n]
            L.append(f"| {n} | {c} | {st['median_delta']:.1f} | {ci(*b['median_delta_ci'], fmt='{:.1f}')} | "
                     f"{st['zero_share']:.3f} | {ci(*b['zero_share_ci'])} | "
                     f"{'yes' if b['limit_inside_zero_ci'] else 'no'} | {st['median_stim']:.1f} | "
                     f"{st['median_rest']:.1f} | **{mark(st['passes'])}** |")
    return L


def report(res, cal, pools, mbon_types, guard, configs, wall_s, meta):
    L = ["# M0d H.3a calibration: the readout-floor guard after MBON baseline recalibration", "",
         "Diagnostic only (not an H.3 selection; no M2 turns, candidate odours or oracle; plasticity off). The "
         "first guard run (`readout_floor_guard.py`, commit cde8bff) measured every configuration at the default "
         f"`mbon_hold_frac` = {DEFAULT_HOLD:g}, i.e. **before** H.3 stage 2 rebisects it to a "
         f"{BASELINE_TARGET_HZ:g} Hz MBON baseline. Spec H.1 defines C1 as graded APL **plus MBON rebalancing**, "
         "so this run redoes the guard at each configuration's own recalibrated baseline.", "",
         "## Step 1 — `mbon_hold_frac` recalibrated per configuration (spec H.3 stage 2)", "",
         f"Bisection over [{HOLD_BRACKET[0]:g}, {HOLD_BRACKET[1]:g}], {meta['n_bisect']} iterations, target "
         f"{BASELINE_TARGET_HZ:g} Hz under the M0c definition: `flymon.brain.measure.mbon_baseline_multi`, seeds "
         f"{REST_SEEDS[0]}-{REST_SEEDS[-1]}, {REST_MS:g} ms, trimmed mean over MBON cells at or below "
         f"{SAT_HZ:g} Hz. Outside {BASELINE_BAND_HZ[0]:g}-{BASELINE_BAND_HZ[1]:g} Hz the configuration fails "
         "stage 2 (the guard is still reported below, for the record).", "",
         "| config | hold @0.5 (Hz) | hold @1.0 (Hz) | accepted `mbon_hold_frac` | baseline Hz (sd over seeds) | "
         f"in {BASELINE_BAND_HZ[0]:g}-{BASELINE_BAND_HZ[1]:g} Hz? | vs default {DEFAULT_HOLD:g} |",
         "|---|---|---|---|---|---|---|"]
    for c in configs:
        k = cal[c]
        e0, e1 = k["endpoints"][f"{HOLD_BRACKET[0]:g}"], k["endpoints"][f"{HOLD_BRACKET[1]:g}"]
        L.append(f"| {c} | {e0['hz']:.2f} | {e1['hz']:.2f} | **{k['hold']:.5f}** | {k['baseline_hz']:.3f} "
                 f"({k['baseline_sd']:.3f}) | {'yes' if k['in_band'] else '**NO — stage 2 failure**'} | "
                 f"{k['delta_vs_default']:+.5f} |")
    L += ["", "Bisection traces (hold → baseline Hz):", ""]
    for c in configs:
        L.append(f"- **{c}**: " + "; ".join(f"{s['hold']:.5f} → {s['hz']:.3f}" for s in cal[c]["trace"])
                 + f". Final bracket [{cal[c]['final_bracket'][0]:.5f}, {cal[c]['final_bracket'][1]:.5f}]; "
                 f"target bracketed by the endpoints: {'yes' if cal[c]['bracketed'] else 'no'}. "
                 f"Accepted = the evaluated point closest to {BASELINE_TARGET_HZ:g} Hz.")

    L += ["", "## Configurations as run", "",
          "| config | apl_mode | apl_r_max | kc_thresh | apl_input_scale | `mbon_hold_frac` | "
          "median KC active % | median KC read-window spikes |", "|---|---|---|---|---|---|---|---|"]
    for c in configs:
        p = res[c]["params"]
        L.append(f"| {c} | {p.get('apl_mode', 'spiking')} | {p.get('apl_r_max', Params().apl_r_max):g} | "
                 f"{p.get('kc_thresh', Params().kc_thresh):g} | {res[c]['scale']:.5g} | "
                 f"{p['mbon_hold_frac']:.5f} | {100 * res[c]['median_kc_active']:.2f} | "
                 f"{res[c]['median_kc']:.0f} |")

    L += ["", "## Step 2/3 — reactivity and its uncertainty", "",
          f"H.4 reactivity rule, verbatim, per MBON *type* (counts summed over both hemispheres): median of "
          f"(read-window spikes − same-seed unstimulated resting spikes) ≥ {MED_DELTA_MIN:g} **and** share of "
          f"presentations with 0 read-window spikes ≤ {ZERO_SHARE_MAX:g}. H.4's teachability criterion is NOT "
          f"evaluated here. Stimuli: the H.3 reference set ({meta['n_odors']} odours × 2 seeds, strength "
          f"{STRENGTH}, settle {SETTLE_MS} ms, read {READ_STEPS} steps) plus a same-seed resting run for each of "
          f"the {meta['n_seeds']} seeds. CIs: 95% percentile bootstrap resampling the {meta['n_odors']} odour "
          f"clusters (2 presentations each), {N_BOOT} draws, seed {BOOT_SEED}.", "",
          f"Pools from `compartments(conn, pops, Params().core_frac)` with core_frac = {meta['core_frac']:g}: "
          f"**A ({PUNISH_TYPE} core)** = {', '.join(pools['A']) or 'none'}; "
          f"**P ({REWARD_TYPE} core)** = {', '.join(pools['P']) or 'none'}."]
    pool_block(L, f"A — {PUNISH_TYPE} core", pools["A"], res, configs)
    pool_block(L, f"P — {REWARD_TYPE} core", pools["P"], res, configs)

    L += ["", "## The G.14.8 readout: MBON05 and MBON13", ""]
    for n in CALLOUT:
        where = [k for k in ("A", "P") if n in pools[k]]
        loc = (f"pool {' and '.join(where)}" if where else
               f"**neither** pool (not a core target of {PUNISH_TYPE} or {REWARD_TYPE})")
        if n not in mbon_types:
            L.append(f"- **{n}**: not present as an MBON type in this connectome.")
            continue
        L.append(f"- **{n}** ({loc}): " + "; ".join(
            f"{c} median Δ {res[c]['types'][n]['median_delta']:.1f} "
            f"{ci(*res[c]['boot'][n]['median_delta_ci'], fmt='{:.1f}')}, zero-share "
            f"{res[c]['types'][n]['zero_share']:.3f} {ci(*res[c]['boot'][n]['zero_share_ci'])} → "
            f"{mark(res[c]['types'][n]['passes'])}" for c in configs) + ".")

    L += ["", "## Step 4 — KC activity at the recalibrated baseline", "",
          f"Median KC active % on the reference set at the recalibrated `mbon_hold_frac`, against a control: the "
          f"same configuration, same odours and seeds, re-run in this script at the default hold "
          f"{DEFAULT_HOLD:g}. Only `mbon_hold_frac` differs between the two columns.", "",
          "| config | median KC active % (recalibrated) | median KC active % (hold 0.85 control) | shift (pp) |",
          "|---|---|---|---|"]
    big = []
    for c in configs:
        d = res[c]["kc_active_shift_pp"]
        L.append(f"| {c} | {100 * res[c]['median_kc_active']:.2f} | {100 * res[c]['median_kc_active_default']:.2f} "
                 f"| {d:+.2f} |")
        if abs(d) > KC_SHIFT_CALLOUT_PP:
            big.append(f"{c} ({d:+.2f} pp)")
    L.append("")
    L.append(f"- **KC activity moves by more than {KC_SHIFT_CALLOUT_PP:g} pp in: {', '.join(big)}.**" if big else
             f"- KC activity moves by at most {KC_SHIFT_CALLOUT_PP:g} pp in every configuration "
             "(the recalibration is a readout-side change, as expected).")

    L += ["", "## Summary", "",
          "| config | `mbon_hold_frac` | baseline Hz | stage 2 | passing types in A | passing types in P | guard |",
          "|---|---|---|---|---|---|---|"]
    for c in configs:
        g, k = guard["per_config"][c], cal[c]
        L.append(f"| {c} | {k['hold']:.5f} | {k['baseline_hz']:.3f} | {'pass' if k['in_band'] else 'FAIL'} | "
                 f"{g['n_pass_A']}/{len(pools['A'])}" + (f" ({', '.join(g['pass_A'])})" if g["pass_A"] else "")
                 + f" | {g['n_pass_P']}/{len(pools['P'])}"
                 + (f" ({', '.join(g['pass_P'])})" if g["pass_P"] else "")
                 + f" | {'PASS' if g['passes'] else 'FAIL'} |")
    L += ["", f"- Recalibrated `mbon_hold_frac` vs the default {DEFAULT_HOLD:g}: "
              + "; ".join(f"{c} {cal[c]['hold']:.5f} ({cal[c]['delta_vs_default']:+.5f})" for c in configs) + ".",
          f"- **{guard['answer']}**"]
    if "C1@0.333" in guard["per_config"]:
        L.append(f"- Cap 0.333 (C1@0.333) "
                 f"{'still passes' if guard['per_config']['C1@0.333']['passes'] else 'does NOT pass'}"
                 f" the guard at its own recalibrated baseline.")
    L += ["", f"Wall time {wall_s:.0f} s ({wall_s / 60:.1f} min). Commit {meta['commit']}"
              f"{' (dirty tree)' if meta['dirty'] else ''}.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    t0 = time.time()
    conn = Connectome.load(NPZ)
    pops = Populations.from_connectome(conn)
    core_frac = Params().core_frac
    comps = compartments(conn, pops, core_frac)
    pools = dict(A=pool_types(conn, comps, PUNISH_TYPE), P=pool_types(conn, comps, REWARD_TYPE))
    mbon_types = sorted(mbon_type_index(conn, pops))
    odors_all = reference_odors(pops)
    del conn

    all_configs = {
        "C0": dict(params=dict(kc_thresh=1.5), scale=1.0),
        "C1@0.333": dict(params=dict(apl_mode="graded", apl_r_max=0.333, kc_thresh=1.5), scale=CAP0_SCALE),
        "C1@1.0": dict(params=dict(apl_mode="graded", apl_r_max=1.0, kc_thresh=CAP1_KC_THRESH), scale=CAP1_SCALE),
    }
    if smoke:
        configs, odors, n_bisect = ("C1@1.0",), odors_all[:4], 2
        chunk, workers, out_dir = 2, 8, "results/m0d/diag/smoke"
    else:
        configs, odors, n_bisect = CONFIGS, odors_all, N_BISECT
        chunk, workers, out_dir = 6, 16, "results/m0d/diag"
    seeds = [int(s) for o in odors for s in o["seeds"]]
    pool_types_all = [n for n in pools["A"] + pools["P"]]
    odor_names = [o["name"] for o in odors]

    print(f"pools: A ({PUNISH_TYPE} core) = {pools['A']}; P ({REWARD_TYPE} core) = {pools['P']}")
    print(f"{len(odors)} odours, {len(seeds)} presentations + {len(seeds)} resting runs per config; "
          f"{len(configs)} configs; {n_bisect} bisection steps + 2 endpoints x {len(REST_SEEDS)} seeds each")

    res, cal = {}, {}
    with FlyPool(NPZ, Params(), [{} for _ in range(workers)], workers=workers,
                 punish_type=PUNISH_TYPE, reward_type=REWARD_TYPE) as fp:
        for name in configs:
            c = all_configs[name]
            t1 = time.time()
            print(f"{name}: step 1 (bisect mbon_hold_frac)", flush=True)
            k = bisect_hold(fp, c["params"], c["scale"], n_bisect)
            k["calib_s"] = time.time() - t1
            cal[name] = k
            params = dict(c["params"], mbon_hold_frac=k["hold"])

            t2 = time.time()
            print(f"{name}: steps 2/4 (guard at hold {k['hold']:.5f})", flush=True)
            stim, rest = run_config(fp, params, c["scale"], odors, seeds, chunk)
            rest_by_seed = {p["seed"]: p for p in rest}
            assert set(rest_by_seed) == set(seeds), "resting runs must cover every presentation seed"

            # step 4 control: the same reference set at the default hold, for the KC active % comparison
            ctrl_params = dict(c["params"], mbon_hold_frac=DEFAULT_HOLD)
            cjobs = [dict(params=ctrl_params, scale=c["scale"], odors=odors[i:i + chunk])
                     for i in range(0, len(odors), chunk)]
            ctrl = [p for part in fp.run_jobs(job_stim_kc, cjobs) for p in part]

            rng = np.random.default_rng(BOOT_SEED)
            boot = {n: bootstrap_type(stim, rest_by_seed, n, odor_names, rng) for n in pool_types_all}
            kc_act = float(np.median([p["kc_active_frac"] for p in stim]))
            kc_act_def = float(np.median([p["kc_active_frac"] for p in ctrl]))
            res[name] = dict(params=params, scale=float(c["scale"]),
                             types={n: type_stats(stim, rest_by_seed, n) for n in mbon_types}, boot=boot,
                             median_kc=float(np.median([p["kc_spikes"] for p in stim])),
                             median_kc_rest=float(np.median([p["kc_spikes"] for p in rest])),
                             median_kc_active=kc_act, median_kc_active_default=kc_act_def,
                             kc_active_shift_pp=100.0 * (kc_act - kc_act_def),
                             stim=stim, rest=rest, ctrl=ctrl, guard_s=time.time() - t2)
            print(f"{name}: calib {k['calib_s']:.0f} s, guard {res[name]['guard_s']:.0f} s", flush=True)

    guard = dict(per_config={})
    for name in configs:
        pa = [n for n in pools["A"] if res[name]["types"][n]["passes"]]
        pp = [n for n in pools["P"] if res[name]["types"][n]["passes"]]
        guard["per_config"][name] = dict(pass_A=pa, pass_P=pp, n_pass_A=len(pa), n_pass_P=len(pp),
                                         passes=bool(pa and pp))
    cap10 = guard["per_config"].get("C1@1.0", dict(passes=False))["passes"]
    guard["cap10_passes"] = bool(cap10)
    guard["selected"] = 1.0 if cap10 else 0.333
    guard["answer"] = ("The cap-1.0 rejection does NOT survive recalibration: at its own recalibrated MBON "
                       "baseline, apl_r_max 1.0 leaves at least one passing MBON type in each pool, so the "
                       "readout-floor guard no longer rejects it."
                       if cap10 else
                       "The cap-1.0 rejection survives recalibration: even at its own recalibrated MBON "
                       "baseline, apl_r_max 1.0 leaves a pool with no passing MBON type, so the guard still "
                       "falls back to 0.333.")

    wall = time.time() - t0
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    meta = dict(npz=NPZ, smoke=smoke, commit=commit, dirty=dirty, strength=STRENGTH, settle_ms=SETTLE_MS,
                read_steps=READ_STEPS, n_odors=len(odors), n_seeds=len(seeds), core_frac=float(core_frac),
                punish_type=PUNISH_TYPE, reward_type=REWARD_TYPE, med_delta_min=MED_DELTA_MIN,
                zero_share_max=ZERO_SHARE_MAX, workers=workers, wall_s=wall, teachability_evaluated=False,
                baseline_target_hz=BASELINE_TARGET_HZ, baseline_band_hz=list(BASELINE_BAND_HZ),
                hold_bracket=list(HOLD_BRACKET), n_bisect=n_bisect, rest_seeds=list(REST_SEEDS),
                rest_ms=REST_MS, sat_hz=SAT_HZ, default_hold=float(DEFAULT_HOLD), n_boot=N_BOOT,
                boot_seed=BOOT_SEED)
    out = dict(meta=meta, pools=pools, mbon_types=mbon_types, calibration=cal, configs=res, guard=guard,
               odors=odors)
    md = report(res, cal, pools, mbon_types, guard, configs, wall, meta)
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/guard_after_baseline.json", "w") as f:
        json.dump(out, f, indent=1, default=float)
    with open(f"{out_dir}/guard_after_baseline.md", "w") as f:
        f.write(md)
    print(md)
    print(f"wall {wall:.1f} s on {workers} workers")
