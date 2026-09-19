"""M0d H.3a: the four inputs the slimmed H.3a rules need (D.4 sparsity margin, D.4 overlap margin, readout guard).

Diagnostic only: calibration data, not an H.3 selection. No M2 turns, no candidate odours, no oracle, plasticity off.
No rule is proposed here — numbers and their implications only.

The H.3a amendment is being cut to three qualifications (D.4 sparsity margin, D.4 overlap margin, readout-floor
guard). Four inputs that decide those three rules have never been measured:

1. **Jaccard operating characteristic.** The overlap clause is a comparison of two noisy numbers, and its
   seed-to-seed noise has never been quantified. No engine runs here: the per-seed design-pair rows already
   exist at settle 200 ms in `results/m0d/diag/d4_margin_protocol.json` (`configs[*].s200.per_seed`) and
   `results/m0d/diag/candidate_feasibility.json` (`step2/step3/step4 rows[*].d4.per_seed`), seeds 100-102.
   From those rows: the seed-to-seed SD of (chance - J) per config, the pooled SD, and a Monte Carlo
   (20,000 draws, seed 20260918) of the probability that a config with a given *true* overlap margin is
   measured as passing, with 3 / 8 / 16 design seeds.
2. **C0's baseline on fresh seeds.** C0 is the spiking control at its frozen `mbon_hold_frac` 0.85; the M0c
   trimmed baseline (`flymon.brain.measure.mbon_baseline_multi`, 3000 ms, 100 Hz trim) is re-measured on two
   independent 16-seed blocks (116-131, 132-147) and compared with the committed M0c record (seeds 100-107).
3. **The adopted point at its final Params.** The adopted candidate is graded APL, `apl_r_max` 0.333,
   `kc_thresh` 1.6, `apl_input_scale` 0.11863, `mbon_hold_frac` **0.84082** (the recalibrated value). Every
   1단계 number for it was measured at hold 0.85, before the baseline recalibration: reference-set median KC %
   and median APL membrane, and D.4 at the gate protocol, are re-measured at the final Params, on the ranking
   seeds (100-102) and on the proposed gate seeds (103-105).
4. **The a_i distribution at the adopted point.** On the same 96 reference presentations as item 3, the
   fraction a_i of presentations in which each KC with `kc_pn_input > 0` fired, which bounds what a
   median-based homeostasis target can do.

Nothing is reimplemented: the reference odour set and the CSC-into-APL scaling come from
`apl_input_scale_sweep.py` (8b51594), the D.4 gate protocol from `d4_margin_protocol.py` (e936434), the MBON
baseline job from `guard_after_baseline.py` (5fe7f7b), and the reference-set summary statistic from
`candidate_feasibility.py` (90c36f9). `job_ref_kc` below repeats `apl_input_scale_sweep.job`'s engine call
sequence exactly (reset / clear_drive / present / run / step x READ_STEPS, with the same per-step APL release
read) and only adds the per-KC fired mask item 4 needs, so its KC % and membrane numbers are the same
measurement as the 1단계 ones, on the same protocol.

Writes results/m0d/diag/slim_inputs.{json,md}; --smoke runs item 2 on 4+4 seeds and items 3/4 on 4 odours into
results/m0d/diag/smoke/ and prints a projection of the full run.
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
from flymon.brain.connectome import Connectome, apply_sign_override
from flymon.brain.fly_pool import FlyPool
from flymon.brain.stimuli import present

from apl_input_scale_sweep import NPZ, READ_STEPS, SETTLE_MS, STRENGTH, make_engine, reference_odors
from candidate_feasibility import point_stats
from d4_margin_protocol import GATE_SETTLE_MS, design_stats, job_design
from guard_after_baseline import BASELINE_BAND_HZ, REST_MS, SAT_HZ, job_baseline

# ---- the adopted point (candidate_feasibility.py step 5, 90c36f9) ------------------------------
ADOPTED_LABEL = "G(1.6, 0.1186)"
ADOPTED_SCALE = 0.11863011807986042          # the step-1 bisection's accepted apl_input_scale
ADOPTED_HOLD = 0.8408203125                  # the step-3 recalibrated mbon_hold_frac ("0.84082")
ADOPTED_BASE = dict(apl_mode="graded", apl_r_max=0.333, kc_thresh=1.6)
ADOPTED_PARAMS = dict(ADOPTED_BASE, mbon_hold_frac=ADOPTED_HOLD)
STAGE1_HOLD = Params().mbon_hold_frac        # 0.85 — the hold every 1단계 number was measured at
FEAS_JSON = "results/m0d/diag/candidate_feasibility.json"
D4_JSON = "results/m0d/diag/d4_margin_protocol.json"

# ---- item 1 ------------------------------------------------------------------------------------
MC_DRAWS, MC_SEED = 20_000, 20260918
TRUE_MARGINS = (0.000, 0.001, 0.002, 0.005, 0.010)
SEED_COUNTS = (3, 8, 16)
GOOD_MARGIN, BAD_MARGIN = 0.005, -0.002      # the truly-good / truly-bad configs of the (delta, seeds) search
POWER_MIN = 0.80                             # both "passes >= 0.8" and "rejected >= 0.8"
DELTA_GRID = tuple(round(0.0005 * i, 6) for i in range(0, 21))   # 0.0000 .. 0.0100 in 0.0005 steps

# ---- item 2 ------------------------------------------------------------------------------------
C0_PARAMS = dict(kc_thresh=1.5, mbon_hold_frac=0.85)   # C0 = the spiking control at its frozen hold
C0_SCALE = 1.0
BASE_BLOCKS = (("116-131", tuple(range(116, 132))), ("132-147", tuple(range(132, 148))))
M0C_SUMMARY = "results/summary/m0c.json"

# ---- item 3 ------------------------------------------------------------------------------------
RANK_SEEDS = (100, 101, 102)                 # the ranking seeds (what 1단계 used)
GATE_SEEDS = (103, 104, 105)                 # the proposed gate seeds
CHUNK = 3                                    # reference odours per job (16 workers, 48 odours)

# ---- item 4 ------------------------------------------------------------------------------------
AI_CLIP = 0.49                               # a_i at which a 4x threshold clip is hit in one step at eta = 0.5

FULL_UNITS = dict(item2=sum(len(s) for _, s in BASE_BLOCKS), item3ref=96,
                  item3d4=2 * 2 * len(RANK_SEEDS))


def fmt(x, n=3):
    return "—" if x is None else f"{x:.{n}f}"


def table(header, rows):
    L = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    L += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return L


# ---- jobs (module level so spawn can pickle them) ----------------------------------------------
def job_ref_kc(eng, pl, pops, comps, ro, params, scale, odors):
    """`apl_input_scale_sweep.job`'s reference-set protocol plus the per-KC fired mask item 4 needs.

    The engine call sequence is identical to that job (reset / clear_drive / present / run(SETTLE_MS) /
    step x READ_STEPS, reading the APL membrane and release every step), so `kc_active_frac`, `apl_v_mean`
    and `release_frac` are the same measurement; `fired` is derived from the same spike counts afterwards
    and is returned as KC indices (positions in `pops.kc`) to keep the payload small.
    """
    e, _ = make_engine(eng, pops, params, scale)
    n_apl = len(pops.apl)
    out = []
    for o in odors:
        for seed in o["seeds"]:
            e.reset(int(seed)); e.clear_drive(); present(e, pops, o["strengths"], STRENGTH); e.run(SETTLE_MS)
            counts = np.zeros(e.N, np.int32)
            apl_v = np.zeros((READ_STEPS, n_apl), np.float64)
            rel = 0.0
            for i in range(READ_STEPS):
                counts[e.step()] += 1
                va = e.v[pops.apl]
                apl_v[i] = va
                rel += float(e.apl_release(va).astype(np.float64).sum())
            kc = counts[pops.kc]
            out.append(dict(odor=o["name"], seed=int(seed), n_kc=int(kc.size),
                            kc_active_frac=float((kc > 0).mean()), kc_spikes=int(kc.sum()),
                            apl_v_mean=float(apl_v.mean()),
                            release_mean=rel / (READ_STEPS * n_apl),
                            release_frac=rel / (READ_STEPS * n_apl) / e.p.apl_r_max,
                            fired=np.flatnonzero(kc > 0).astype(np.int32).tolist()))
    return out


# ================================================================ item 1: Jaccard OC (no engine runs)
def overlap_rows():
    """Per-seed (chance - J) rows at settle 200, read out of the two committed JSON files.

    d4_margin_protocol.json: `configs[*].s200.per_seed` (7 stage-1 configs, the preserved candidate among them).
    candidate_feasibility.json: `step2/step3/step4 rows[*].d4.per_seed` (the four new kc_thresh points and the
    old candidate; steps 3 and 4 carry step 2's D.4 block forward, so identical rows are deduplicated).
    """
    rows, seen = [], set()

    def add(source, field, label, per_seed):
        if not per_seed:
            return
        m = [float(r["chance"]) - float(r["jaccard"]) for r in per_seed]
        key = (label.replace("old ", ""), tuple(round(x, 12) for x in m))
        if key in seen:
            return
        seen.add(key)
        rows.append(dict(source=source, field=field, label=label,
                         seeds=[int(r["seed"]) for r in per_seed],
                         jaccard=[float(r["jaccard"]) for r in per_seed],
                         chance=[float(r["chance"]) for r in per_seed],
                         margin=m, n=len(m), mean=float(np.mean(m)),
                         sd=float(np.std(m, ddof=1)) if len(m) > 1 else None))

    try:
        d = json.load(open(D4_JSON))
        for c in d["configs"]:
            add(D4_JSON, "configs[*].s200.per_seed", c["label"], c["s200"].get("per_seed"))
    except (OSError, KeyError, ValueError, TypeError) as exc:
        print(f"  WARNING: {D4_JSON} unusable ({exc})", flush=True)
    try:
        d = json.load(open(FEAS_JSON))
        for step in ("step2", "step3", "step4"):
            for r in d.get(step, {}).get("rows", []):
                add(FEAS_JSON, f"{step}.rows[*].d4.per_seed", r["label"], (r.get("d4") or {}).get("per_seed"))
    except (OSError, KeyError, ValueError, TypeError) as exc:
        print(f"  WARNING: {FEAS_JSON} unusable ({exc})", flush=True)
    return rows


def pooled_sd(rows):
    """Pooled within-config SD of (chance - J): sqrt(sum (n_i-1) s_i^2 / sum (n_i-1))."""
    num = sum((r["n"] - 1) * r["sd"] ** 2 for r in rows if r["sd"] is not None)
    den = sum(r["n"] - 1 for r in rows if r["sd"] is not None)
    return math.sqrt(num / den) if den else None


def mc_pass(rng, sd, true_margin, n_seeds, delta):
    """P(mean of n_seeds draws from N(true_margin, sd) > delta), by Monte Carlo over the measured noise."""
    draws = rng.normal(true_margin, sd, size=(MC_DRAWS, n_seeds)).mean(axis=1)
    return float((draws > delta).mean())


def item1():
    rows = overlap_rows()
    sd = pooled_sd(rows)
    res = dict(source_files=[D4_JSON, FEAS_JSON], settle_ms=GATE_SETTLE_MS, configs=rows,
               n_configs=len(rows), pooled_sd=sd, mc_draws=MC_DRAWS, mc_seed=MC_SEED,
               true_margins=list(TRUE_MARGINS), seed_counts=list(SEED_COUNTS))
    if sd is None or sd <= 0:
        res["oc"], res["search"] = None, None
        return res
    rng = np.random.default_rng(MC_SEED)
    res["oc"] = [dict(true_margin=m, n_seeds=n, p_pass=mc_pass(rng, sd, m, n, 0.0),
                      p_pass_analytic=float(1 - 0.5 * math.erfc((m - 0.0) / (sd / math.sqrt(n)) / math.sqrt(2))))
                 for m in TRUE_MARGINS for n in SEED_COUNTS]
    grid = []
    for n in SEED_COUNTS:
        for delta in DELTA_GRID:
            p_good = mc_pass(rng, sd, GOOD_MARGIN, n, delta)
            p_bad = mc_pass(rng, sd, BAD_MARGIN, n, delta)
            grid.append(dict(n_seeds=n, delta=delta, p_good_pass=p_good, p_bad_pass=p_bad,
                             p_bad_reject=1.0 - p_bad,
                             ok=bool(p_good >= POWER_MIN and (1.0 - p_bad) >= POWER_MIN)))
    ok = [g for g in grid if g["ok"]]
    best = min(ok, key=lambda g: (g["delta"], g["n_seeds"])) if ok else None
    res["search"] = dict(good_margin=GOOD_MARGIN, bad_margin=BAD_MARGIN, power_min=POWER_MIN,
                         delta_grid=list(DELTA_GRID), grid=grid, feasible=bool(ok), best=best,
                         best_per_seed_count={str(n): (min([g for g in ok if g["n_seeds"] == n],
                                                           key=lambda g: g["delta"]) if
                                                       any(g["n_seeds"] == n for g in ok) else None)
                                              for n in SEED_COUNTS})
    return res


# ================================================================ item 2: C0's baseline on fresh seeds
def baseline_block(pool, seeds, acc):
    jobs = [dict(params=C0_PARAMS, scale=C0_SCALE, seeds=[int(s)]) for s in seeds]
    t0 = time.time()
    rows = [r for part in pool.run_jobs(job_baseline, jobs) for r in part]
    acc("item2", len(rows), time.time() - t0)
    x = np.array([r["trimmed"] for r in rows], float)
    return dict(seeds=[int(s) for s in seeds], n=len(x), per_seed=[float(v) for v in x],
                mean_hz=float(x.mean()), sd_hz=float(x.std(ddof=1)),
                se_hz=float(x.std(ddof=1) / math.sqrt(len(x))),
                in_band=bool(BASELINE_BAND_HZ[0] <= float(x.mean()) <= BASELINE_BAND_HZ[1]),
                n_saturated=float(np.mean([r["n_saturated"] for r in rows])),
                n_types_active=float(np.mean([r["n_types_active"] for r in rows])))


def m0c_reference():
    try:
        d = json.load(open(M0C_SUMMARY))
        b = d.get("baseline") or d["sparsity"]
        return dict(source=M0C_SUMMARY, mean_hz=float(b["mbon_hz_rest_trimmed"]),
                    sd_hz=float(b["mbon_hz_rest_trimmed_sd"]), rest_ms=float(b.get("rest_ms", REST_MS)),
                    seeds=list(d["seeds"]["rest"]), hold=float(d["params_frozen"]["mbon_hold_frac"]),
                    kc_thresh=float(d["params_frozen"]["kc_thresh"]))
    except (OSError, KeyError, ValueError, TypeError):
        return None


def item2(pool, blocks, acc):
    out = dict(params=C0_PARAMS, scale=C0_SCALE, rest_ms=REST_MS, sat_hz=SAT_HZ,
               band_hz=list(BASELINE_BAND_HZ), reference=m0c_reference(), blocks=[])
    for name, seeds in blocks:
        b = baseline_block(pool, seeds, acc)
        b["name"] = name
        out["blocks"].append(b)
        print(f"  item 2: C0 seeds {name} -> {b['mean_hz']:.4f} Hz (SE {b['se_hz']:.4f}), "
              f"in 3-4 Hz: {b['in_band']}", flush=True)
    if len(out["blocks"]) == 2:
        a, b = out["blocks"]
        d = a["mean_hz"] - b["mean_hz"]
        se = math.sqrt(a["se_hz"] ** 2 + b["se_hz"] ** 2)
        out["block_diff"] = dict(diff_hz=float(d), se_hz=float(se), z=float(d / se) if se else None)
    return out


# ================================================================ items 3 & 4: the adopted point
def stage1_reference():
    """The 1단계 numbers for the adopted point, all measured at hold 0.85, read from the committed runs."""
    ref = dict(hold=STAGE1_HOLD, source=FEAS_JSON,
               ref_median_kc_pct=None, median_mv=None, d4=None, d4_seeds=list(RANK_SEEDS))
    try:
        d = json.load(open(FEAS_JSON))
        for r in d["step2"]["rows"]:
            if r["label"] == ADOPTED_LABEL:
                ref["ref_median_kc_pct"] = float(r["ref_median_kc_pct"])
                ref["median_mv"] = float(r["median_mv"])
                ref["d4"] = {k: v for k, v in r["d4"].items() if k != "per_seed"}
                ref["scale"] = float(r["scale"])
    except (OSError, KeyError, ValueError, TypeError):
        pass
    return ref


def item34(pool, pops, odors, acc):
    jobs = [dict(params=ADOPTED_PARAMS, scale=ADOPTED_SCALE, odors=odors[c:c + CHUNK])
            for c in range(0, len(odors), CHUNK)]
    t0 = time.time()
    pres = [p for part in pool.run_jobs(job_ref_kc, jobs) for p in part]
    acc("item3ref", len(pres), time.time() - t0)
    st = point_stats(pres, ADOPTED_SCALE)
    st.pop("presentations", None)
    print(f"  item 3: reference set ({len(pres)} presentations) -> median KC {st['median_kc_pct']:.4f}%, "
          f"median APL membrane {st['median_mv']:.4f} mV", flush=True)

    jobs = [dict(params=ADOPTED_PARAMS, scale=ADOPTED_SCALE, settle_ms=float(GATE_SETTLE_MS), seeds=list(s))
            for s in (RANK_SEEDS, GATE_SEEDS)]
    t0 = time.time()
    raw = pool.run_jobs(job_design, jobs)
    acc("item3d4", 2 * (len(RANK_SEEDS) + len(GATE_SEEDS)), time.time() - t0)
    d4 = {}
    for name, seeds, rows in (("rank_100_102", RANK_SEEDS, raw[0]), ("gate_103_105", GATE_SEEDS, raw[1])):
        d4[name] = design_stats(rows)
        d4[name]["seeds"] = list(seeds)
        print(f"  item 3: D.4 {name} -> A {d4[name]['pct_A']:.3f}% B {d4[name]['pct_B']:.3f}%, "
              f"margin {d4[name]['margin_pp']:+.3f} pp, chance-J {d4[name]['overlap_margin']:+.5f}", flush=True)

    item3 = dict(label=ADOPTED_LABEL, params=ADOPTED_PARAMS, scale=ADOPTED_SCALE, hold=ADOPTED_HOLD,
                 n_presentations=len(pres), reference=st, d4=d4, settle_ms=GATE_SETTLE_MS,
                 stage1=stage1_reference())
    s1 = item3["stage1"]
    item3["moves"] = dict(
        ref_median_kc_pct=(st["median_kc_pct"] - s1["ref_median_kc_pct"]) if s1["ref_median_kc_pct"] else None,
        median_mv=(st["median_mv"] - s1["median_mv"]) if s1["median_mv"] else None,
        pct_A=(d4["rank_100_102"]["pct_A"] - s1["d4"]["pct_A"]) if s1["d4"] else None,
        pct_B=(d4["rank_100_102"]["pct_B"] - s1["d4"]["pct_B"]) if s1["d4"] else None,
        margin_pp=(d4["rank_100_102"]["margin_pp"] - s1["d4"]["margin_pp"]) if s1["d4"] else None,
        overlap_margin=(d4["rank_100_102"]["overlap_margin"] - s1["d4"]["overlap_margin"]) if s1["d4"] else None)
    return item3, pres


def kc_pn_input(conn, pops, p):
    """Summed ALPN->KC synaptic weight per KC, raw and with the engine's own edge filter (min_weight, sign)."""
    sign, _ = apply_sign_override(conn, p)
    is_pn = np.zeros(conn.N, bool); is_pn[np.asarray(pops.alpn)] = True
    is_kc = np.zeros(conn.N, bool); is_kc[np.asarray(pops.kc)] = True
    m = is_pn[conn.pre] & is_kc[conn.post]
    raw = np.zeros(conn.N, float); np.add.at(raw, conn.post[m], conn.w[m].astype(float))
    mk = m & (conn.w >= p.min_weight) & (sign[conn.pre] != 0)
    kept = np.zeros(conn.N, float); np.add.at(kept, conn.post[mk], conn.w[mk].astype(float))
    kc = np.asarray(pops.kc)
    return raw[kc], kept[kc]


def item4(pres, pn_raw, pn_kept):
    n_pres = len(pres)
    n_kc = int(pres[0]["n_kc"])
    counts = np.zeros(n_kc, np.int64)
    for p in pres:
        counts[np.asarray(p["fired"], np.int64)] += 1
    a = counts / float(n_pres)

    def stats(mask, name):
        x = a[mask]
        if x.size == 0:
            return dict(name=name, n_kc=0)
        q = np.percentile(x, [25, 50, 75, 90, 99])
        return dict(name=name, n_kc=int(x.size), share_zero=float((x == 0).mean()),
                    q1=float(q[0]), median=float(q[1]), q3=float(q[2]), p90=float(q[3]), p99=float(q[4]),
                    max=float(x.max()), mean=float(x.mean()),
                    share_ge_clip=float((x >= AI_CLIP).mean()),
                    n_ge_clip=int((x >= AI_CLIP).sum()))

    out = dict(label=ADOPTED_LABEL, params=ADOPTED_PARAMS, scale=ADOPTED_SCALE,
               n_presentations=n_pres, n_kc_total=n_kc, ai_clip=AI_CLIP,
               n_kc_pn_positive_kept=int((pn_kept > 0).sum()), n_kc_pn_positive_raw=int((pn_raw > 0).sum()),
               pn_positive=stats(pn_kept > 0, "kc_pn_input > 0 (engine edge filter)"),
               pn_positive_raw=stats(pn_raw > 0, "kc_pn_input > 0 (raw connectome weights)"),
               all_kc=stats(np.ones(n_kc, bool), "all KCs"),
               histogram=dict(edges=[0.0, 1e-12, 0.05, 0.1, 0.2, 0.3, 0.49, 1.01],
                              counts=[int(c) for c in np.histogram(
                                  a[pn_kept > 0], bins=[0.0, 1e-12, 0.05, 0.1, 0.2, 0.3, 0.49, 1.01])[0]]))
    s = out["pn_positive"]
    out["median_target_reachable"] = bool(s.get("median", 0.0) > 0.0)
    return out


# ================================================================ report
def report(res, wall_s):
    meta = res["meta"]
    L = ["# M0d H.3a: the four inputs the slimmed rules need", "",
         f"Diagnostic only (calibration data, not an H.3 selection; no M2 turns, candidate odours or oracle; "
         f"plasticity off). No rule changes are proposed here — numbers and implications only. "
         f"Commit {meta['commit']} (dirty tree: {str(meta['dirty']).lower()})"
         + (", **--smoke run (reduced settings)**." if meta["smoke"] else "."), ""]

    # ---- item 1
    i1 = res["item1"]
    L += ["## 1. The Jaccard (overlap) operating characteristic", "",
          f"No engine runs: the per-seed design-pair rows at settle {i1['settle_ms']:g} ms are read out of "
          f"`{D4_JSON}` (`configs[*].s200.per_seed`) and `{FEAS_JSON}` "
          f"(`step2/step3/step4 rows[*].d4.per_seed`), design seeds "
          f"{sorted({s for c in i1['configs'] for s in c['seeds']})}. Rows that steps 3 and 4 carry forward "
          f"unchanged from step 2 are deduplicated, so each distinct measured config appears once "
          f"({i1['n_configs']} configs).", ""]
    L += table(["config", "source", "seeds", "per-seed chance − J", "mean", "SD"],
               [[c["label"], os.path.basename(c["source"]), " ".join(str(s) for s in c["seeds"]),
                 " ".join(f"{m:+.5f}" for m in c["margin"]), f"{c['mean']:+.5f}",
                 fmt(c["sd"], 5) if c["sd"] is not None else "—"] for c in i1["configs"]])
    if i1["pooled_sd"] is None:
        L += ["", "Pooled SD not computable — no config has more than one seed on record.", ""]
    else:
        sd = i1["pooled_sd"]
        L += ["", f"**Pooled seed-to-seed SD of (chance − J): {sd:.5f}** "
                  f"(within-config, {sum(c['n'] - 1 for c in i1['configs'] if c['sd'] is not None)} dof). "
                  f"For comparison, the adopted point's measured overlap margin is of order 3e-4, i.e. about "
                  f"{3e-4 / sd:.2f} SD.", "",
              f"Monte Carlo over that noise ({i1['mc_draws']:,} draws, seed {i1['mc_seed']}): probability that a "
              f"config whose **true** overlap margin is as given is measured as passing (mean chance − J > 0).", ""]
        oc = {(o["true_margin"], o["n_seeds"]): o["p_pass"] for o in i1["oc"]}
        L += table(["true margin"] + [f"{n} seeds" for n in SEED_COUNTS],
                   [[f"{m:+.3f}"] + [f"{oc[(m, n)]:.3f}" for n in SEED_COUNTS] for m in TRUE_MARGINS])
        s = i1["search"]
        L += ["", f"Declared-minimum-margin search: pass = measured mean (chance − J) > δ; the requirement is "
                  f"P(pass | true margin {s['good_margin']:+.3f}) ≥ {s['power_min']:.2f} **and** "
                  f"P(reject | true margin {s['bad_margin']:+.3f}) ≥ {s['power_min']:.2f}, over "
                  f"δ ∈ [{min(s['delta_grid']):.4f}, {max(s['delta_grid']):.4f}] in 0.0005 steps and "
                  f"seeds ∈ {list(SEED_COUNTS)}.", ""]
        L += table(["seeds", "smallest δ that works", "P(good passes)", "P(bad rejected)"],
                   [[n, (f"{s['best_per_seed_count'][str(n)]['delta']:.4f}"
                         if s["best_per_seed_count"][str(n)] else "**none in range**"),
                     (f"{s['best_per_seed_count'][str(n)]['p_good_pass']:.3f}"
                      if s["best_per_seed_count"][str(n)] else "—"),
                     (f"{s['best_per_seed_count'][str(n)]['p_bad_reject']:.3f}"
                      if s["best_per_seed_count"][str(n)] else "—")] for n in SEED_COUNTS])
        if s["feasible"]:
            b = s["best"]
            L += ["", f"Smallest feasible pair: **δ = {b['delta']:.4f} with {b['n_seeds']} design seeds** "
                      f"(good passes {b['p_good_pass']:.3f}, bad rejected {b['p_bad_reject']:.3f})."]
        else:
            L += ["", f"**No (δ, seeds) pair in the considered range does that.** With a pooled SD of "
                      f"{sd:.5f} the standard error of the mean margin is {sd / math.sqrt(3):.5f} at 3 seeds and "
                      f"{sd / math.sqrt(16):.5f} at 16 seeds, both larger than the {GOOD_MARGIN - BAD_MARGIN:.3f} "
                      f"separation between the truly-good and truly-bad configs, so no threshold in "
                      f"[{min(s['delta_grid']):.4f}, {max(s['delta_grid']):.4f}] separates them at "
                      f"{s['power_min']:.0%} in both directions."]
        L += ["", f"*Implication for the D.4 overlap rule:* the rule's discriminating power is set entirely by "
                  f"this SD ({sd:.5f}) against the seed count, so a declared minimum overlap margin is only "
                  f"meaningful at the (δ, seeds) combinations in the table above — "
                  + ("and one exists." if s["feasible"] else
                     "and none exists in the considered range, i.e. at these seed counts the clause cannot "
                     "separate a +0.005 config from a −0.002 one."), ""]

    # ---- item 2
    i2 = res["item2"]
    L += ["## 2. C0's MBON baseline on fresh seeds", "",
          f"C0 = the spiking control at its frozen `mbon_hold_frac` {C0_PARAMS['mbon_hold_frac']:g} "
          f"(`kc_thresh` {C0_PARAMS['kc_thresh']:g}, no CSC-into-APL scaling). M0c trimmed baseline: "
          f"`flymon.brain.measure.mbon_baseline_multi`, {i2['rest_ms']:g} ms, cells above {i2['sat_hz']:g} Hz "
          f"excluded; the statistic is the mean over seeds of the per-seed trimmed rate. Band "
          f"{i2['band_hz'][0]:g}-{i2['band_hz'][1]:g} Hz.", ""]
    rows = [[b["name"], b["n"], f"{b['mean_hz']:.4f}", f"{b['sd_hz']:.4f}", f"{b['se_hz']:.4f}",
             "**yes**" if b["in_band"] else "**NO**"] for b in i2["blocks"]]
    ref = i2["reference"]
    if ref:
        rows.append([f"{min(ref['seeds'])}-{max(ref['seeds'])} (committed M0c record)", len(ref["seeds"]),
                     f"{ref['mean_hz']:.4f}", f"{ref['sd_hz']:.4f}",
                     f"{ref['sd_hz'] / math.sqrt(len(ref['seeds'])):.4f}",
                     "yes" if i2["band_hz"][0] <= ref["mean_hz"] <= i2["band_hz"][1] else "NO"])
    L += table(["seed block", "n", "mean Hz", "SD", "SE", f"in {i2['band_hz'][0]:g}-{i2['band_hz'][1]:g} Hz"], rows)
    if "block_diff" in i2:
        d = i2["block_diff"]
        L += ["", f"Difference between the two fresh blocks: {d['diff_hz']:+.4f} Hz "
                  f"(SE {d['se_hz']:.4f}, z {d['z']:+.2f})."]
    inb = [b["name"] for b in i2["blocks"] if b["in_band"]]
    L += ["", f"*Implication for the readout-floor guard:* the guard is applied at a recalibrated baseline, so "
              f"C0's own baseline on seeds it was never calibrated on is the control — here "
              + (f"{len(inb)}/{len(i2['blocks'])} fresh block(s) land in the "
                 f"{i2['band_hz'][0]:g}-{i2['band_hz'][1]:g} Hz band"
                 if inb else f"neither fresh block lands in the {i2['band_hz'][0]:g}-{i2['band_hz'][1]:g} Hz band")
              + ", which is what fixes whether the control itself would clear the qualification it is used to "
                "judge against.", ""]

    # ---- item 3
    i3 = res["item3"]
    s1, mv = i3["stage1"], i3["moves"]
    L += ["## 3. The adopted point re-measured at its final Params", "",
          f"Adopted candidate {i3['label']}: `apl_mode` graded, `apl_r_max` {ADOPTED_BASE['apl_r_max']:g}, "
          f"`kc_thresh` {ADOPTED_BASE['kc_thresh']:g}, `apl_input_scale` {i3['scale']:.5f}, "
          f"`mbon_hold_frac` **{i3['hold']:.5f}** (recalibrated). Every 1단계 number was measured at hold "
          f"{s1['hold']:g}; the comparison column is that earlier measurement, read from `{s1['source']}`. "
          f"Reference set: {i3['n_presentations']} presentations (48 odours x 2 seeds, settle {SETTLE_MS} ms, "
          f"read {READ_STEPS} steps). D.4 at the gate protocol: settle {i3['settle_ms']:g} ms, read 600 ms.", ""]
    r = i3["reference"]
    L += table(["quantity", f"hold {s1['hold']:g} (1단계)", f"hold {i3['hold']:.5f} (final)", "move"],
               [["reference median KC active %", fmt(s1["ref_median_kc_pct"], 2),
                 f"{r['median_kc_pct']:.2f}", fmt(mv["ref_median_kc_pct"], 3) + " pp"],
                ["reference median APL membrane, mV", fmt(s1["median_mv"], 2),
                 f"{r['median_mv']:.2f}", fmt(mv["median_mv"], 3)]])
    L += ["", f"Reference-set IQR of the APL membrane {r['q1_mv']:.2f}-{r['q3_mv']:.2f} mV; KC active % quartiles "
              f"{r['kc_q1_pct']:.2f}-{r['kc_q3_pct']:.2f}%; median release {r['median_release_frac']:.3f} of "
              f"`apl_r_max`.", ""]
    d4rows = []
    for name, lab in (("rank_100_102", "ranking seeds 100-102"), ("gate_103_105", "gate seeds 103-105")):
        d = i3["d4"][name]
        d4rows.append([lab, f"{d['pct_A']:.2f}", f"{d['pct_B']:.2f}", f"{d['jaccard']:.5f}",
                       f"{d['chance']:.5f}", f"{d['margin_pp']:+.2f}", f"{d['overlap_margin']:+.5f}",
                       "PASS" if d["d4_ok"] else "FAIL"])
    if s1["d4"]:
        d = s1["d4"]
        d4rows.append([f"1단계, hold {s1['hold']:g}, seeds 100-102", f"{d['pct_A']:.2f}", f"{d['pct_B']:.2f}",
                       f"{d['jaccard']:.5f}", f"{d['chance']:.5f}", f"{d['margin_pp']:+.2f}",
                       f"{d['overlap_margin']:+.5f}", "PASS" if d["d4_ok"] else "FAIL"])
    L += table(["D.4 at settle 200", "A KC %", "B KC %", "Jaccard", "chance", "sparsity margin pp",
                "chance − J", "D.4"], d4rows)
    rank, gate = i3["d4"]["rank_100_102"], i3["d4"]["gate_103_105"]
    L += ["", f"Ranking seeds vs gate seeds at the same Params: sparsity margin {rank['margin_pp']:+.2f} vs "
              f"{gate['margin_pp']:+.2f} pp (Δ {gate['margin_pp'] - rank['margin_pp']:+.2f}), overlap margin "
              f"{rank['overlap_margin']:+.5f} vs {gate['overlap_margin']:+.5f} "
              f"(Δ {gate['overlap_margin'] - rank['overlap_margin']:+.5f}).", "",
          f"*Implication for the D.4 sparsity and overlap rules:* this is the size of the "
          f"\"qualification state ≠ final engine state\" gap — the hold recalibration moved the reference "
          f"median by {fmt(mv['ref_median_kc_pct'], 3)} pp and the D.4 sparsity margin by "
          f"{fmt(mv['margin_pp'], 3)} pp on the very seeds the ranking used, so any declared margin has to "
          f"absorb both that shift and the ranking-to-gate seed change above.", ""]

    # ---- item 4
    i4 = res["item4"]
    s = i4["pn_positive"]
    L += ["## 4. The a_i distribution at the adopted point", "",
          f"Same {i4['n_presentations']} reference presentations as item 3 (the identical engine runs). For every "
          f"KC, a_i = the fraction of presentations in which it fired at least one spike. The population is the "
          f"{i4['n_kc_pn_positive_kept']} KCs with `kc_pn_input > 0` out of {i4['n_kc_total']} "
          f"(summed ALPN→KC weight with the engine's own edge filter, `min_weight` {Params().min_weight:g} and a "
          f"non-zero sign; without that filter it is {i4['n_kc_pn_positive_raw']} KCs, reported as a second row).", ""]
    L += table(["population", "n KC", "share a_i = 0", "Q1", "median", "Q3", "p90", "p99", "max",
                f"share a_i ≥ {AI_CLIP:g}"],
               [[t["name"], t["n_kc"], f"{t['share_zero']:.4f}", f"{t['q1']:.4f}", f"{t['median']:.4f}",
                 f"{t['q3']:.4f}", f"{t['p90']:.4f}", f"{t['p99']:.4f}", f"{t['max']:.4f}",
                 f"{t['share_ge_clip']:.4f} ({t['n_ge_clip']})"]
                for t in (i4["pn_positive"], i4["pn_positive_raw"], i4["all_kc"])])
    h = i4["histogram"]
    bins = []
    for i, c in enumerate(h["counts"]):
        lab = "= 0" if i == 0 else "in (%g, %g]" % (h["edges"][i], min(h["edges"][i + 1], 1.0))
        bins.append(f"a_i {lab}: {c}")
    L += ["", "Distribution over the `kc_pn_input > 0` population: " + ", ".join(bins) + ".", ""]
    reach = ("**reachable**" if i4["median_target_reachable"] else "**not reachable**")
    L += [f"*Implication for the readout-floor guard / any homeostasis rule:* the median a_i is "
          f"{s['median']:.4f} with {s['share_zero']:.1%} of PN-driven KCs never firing, so a median-based "
          f"homeostasis target is {reach} — "
          + ("the median sits on a non-zero value that a per-KC rule can move."
             if i4["median_target_reachable"] else
             "with more than half the PN-driven KCs at a_i = 0 the median is pinned at exactly 0 and is "
             "insensitive to any per-KC adjustment until that majority is moved off zero, so the target has "
             "to be stated on a different statistic.")
          + f" Only {s['share_ge_clip']:.2%} ({s['n_ge_clip']}) of those KCs sit at a_i ≥ {AI_CLIP:g}, the value "
            f"that would hit a 4x threshold clip in one step at η = 0.5.", ""]

    L += [f"Wall time of the run: {wall_s:.0f} s ({wall_s / 60:.1f} min); engine work "
          + ", ".join(f"{k} {v['units']} units in {v['wall_s']:.0f} s" for k, v in meta["cost"].items())
          + f". Commit {meta['commit']}, dirty tree: {str(meta['dirty']).lower()}.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    t0 = time.time()
    p0 = Params()
    conn = Connectome.load(NPZ)
    pops = Populations.from_connectome(conn)
    pn_raw, pn_kept = kc_pn_input(conn, pops, p0)
    del conn

    odors = reference_odors(pops)
    if smoke:
        blocks = [(n, s[:4]) for n, s in BASE_BLOCKS]
        odors = odors[:4]
        workers, out_dir = 4, "results/m0d/diag/smoke"
    else:
        blocks = list(BASE_BLOCKS)
        workers, out_dir = 16, "results/m0d/diag"

    cost = {}

    def acc(step, units, wall):
        c = cost.setdefault(step, dict(units=0, wall_s=0.0))
        c["units"] += int(units)
        c["wall_s"] += float(wall)

    print(f"item 1: reading per-seed rows from the committed JSON (no engine runs)", flush=True)
    i1 = item1()
    print(f"  item 1: {i1['n_configs']} configs, pooled SD {fmt(i1['pooled_sd'], 5)}", flush=True)

    with FlyPool(NPZ, Params(), [{} for _ in range(workers)], workers=workers) as pool:
        i2 = item2(pool, blocks, acc)
        i3, pres = item34(pool, pops, odors, acc)
    i4 = item4(pres, pn_raw, pn_kept)
    print(f"  item 4: {i4['n_kc_pn_positive_kept']} PN-driven KCs, median a_i "
          f"{i4['pn_positive']['median']:.4f}, share zero {i4['pn_positive']['share_zero']:.4f}", flush=True)

    wall = time.time() - t0
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    res = dict(meta=dict(npz=NPZ, smoke=smoke, commit=commit, dirty=dirty, strength=STRENGTH,
                         settle_ms=SETTLE_MS, read_steps=READ_STEPS, gate_settle_ms=GATE_SETTLE_MS,
                         adopted_label=ADOPTED_LABEL, adopted_params=ADOPTED_PARAMS, adopted_scale=ADOPTED_SCALE,
                         rank_seeds=list(RANK_SEEDS), gate_seeds=list(GATE_SEEDS),
                         baseline_blocks={n: list(s) for n, s in blocks}, workers=workers, wall_s=wall,
                         cost=cost, plasticity=False),
               item1=i1, item2=i2, item3=i3, item4=i4)

    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/slim_inputs.json", "w") as f:
        json.dump(res, f, indent=1)
    with open(f"{out_dir}/slim_inputs.md", "w") as f:
        f.write(report(res, wall))
    print(report(res, wall))
    print(f"wrote {out_dir}/slim_inputs.json / .md in {wall:.0f} s", flush=True)

    if smoke:
        proj = 0.0
        for k, full in FULL_UNITS.items():
            c = cost.get(k)
            if not c or not c["units"]:
                continue
            per = c["wall_s"] / c["units"]
            # the smoke runs 4 workers, the full run 16
            est = per * full * (workers / 16.0)
            proj += est
            print(f"projection {k}: {c['units']} units in {c['wall_s']:.1f} s -> {full} units "
                  f"at 16 workers ≈ {est:.0f} s", flush=True)
        overhead = wall - sum(c["wall_s"] for c in cost.values())
        print(f"projected full run ≈ {proj + overhead:.0f} s ({(proj + overhead) / 60:.1f} min), "
              f"including {overhead:.0f} s of non-engine overhead measured here", flush=True)
