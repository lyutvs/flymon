"""M0d H.3a calibration: sweep a multiplier on every CSC edge into APL under graded APL. Diagnostic only.

Calibration data, not an H.3 selection: stimuli are the H.3 reference odour set (48 random 6-9 glomerulus mixtures,
rng seed 800000, DA1 and V excluded) at strength 0.35, two seeds per odour. No M2 turns, candidate odours or oracle.
Grid: kc_thresh {1.25, 1.5} x apl_input_scale geomspace(0.005, 1, 10); apl_r_max and all other Params at defaults.
Writes results/m0d/diag/apl_input_scale_sweep.json (raw) and .md (report); --smoke runs a reduced grid into
results/m0d/diag/smoke/.
"""
import json
import os
import sys
import time

import numpy as np

from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome, apply_sign_override
from flymon.brain.engine_cpu import Engine
from flymon.brain.fly_pool import FlyPool
from flymon.brain.stimuli import channel_strengths, present

NPZ = "data/malecns.npz"
STRENGTH = 0.35
SETTLE_MS, READ_STEPS = 800, 600
KC_THRESH = (1.25, 1.5)
SCALES = tuple(float(s) for s in np.geomspace(0.005, 1.0, 10))
N_ODORS, CHUNK = 48, 12
LAGS = (3, 50)
TARGET_MV, BAND_MV = 11.0, (8.5, 13.5)
REL_BINS = (0.38, 0.62)


def reference_odors(pops):
    cand = [t for t in pops.receptor_types if t not in ("ORN_DA1", "ORN_V")]   # dict order = sorted names
    assert len(cand) == 51
    rng = np.random.default_rng(800_000)
    odors = []
    for j in range(48):
        k = int(rng.integers(6, 10))                                             # 6..9 glomeruli
        types = sorted(str(t) for t in rng.choice(cand, size=k, replace=False))
        odor_j = channel_strengths(pops, types)                                  # strength ∝ 1/receptor count
        seeds_j = (800_100 + 2 * j, 800_101 + 2 * j)
        odors.append(dict(name=f"R{j:02d}", types=types, strengths=odor_j, seeds=list(seeds_j)))
    assert len({tuple(o["types"]) for o in odors}) == 48
    return odors


def make_engine(eng, pops, params, scale):
    e = Engine(eng.conn, pops, Params(**params))
    is_apl = np.zeros(e.N, bool); is_apl[pops.apl] = True
    m = is_apl[e.csc.tgt]
    if scale != 1.0:
        e.csc.w[m] *= np.float32(scale)
    return e, m


def meta_job(eng, pl, pops, comps, ro, params, scale):
    """Edge count into APL (CSC mask vs an independent count from the raw connectome) and per-cell summed in-weight."""
    p = Params(**params)
    conn = eng.conn
    sign, _ = apply_sign_override(conn, p)
    is_apl = np.zeros(conn.N, bool); is_apl[pops.apl] = True
    independent = int(((conn.w >= p.min_weight) & (sign[conn.pre] != 0) & is_apl[conn.post]).sum())
    e = Engine(eng.conn, pops, p)
    tgt, w = e.csc.tgt, e.csc.w
    pre_w = [float(w[tgt == a].sum()) for a in pops.apl]
    e, m = make_engine(eng, pops, params, scale)
    assert int(m.sum()) == independent, (int(m.sum()), independent)
    post_w = [float(e.csc.w[e.csc.tgt == a].sum()) for a in pops.apl]
    return dict(apl_cells=[int(a) for a in pops.apl], n_edges_into_apl=independent, scale=scale,
                in_weight_pre_mv=pre_w, in_weight_post_mv=post_w)


def acmax(x):
    x = np.asarray(x, np.float64) - np.mean(x)
    v = float(np.dot(x, x))
    if v == 0:
        return 0.0
    return float(max(np.dot(x[:-lag], x[lag:]) / v for lag in range(LAGS[0], LAGS[1] + 1)))


def job(eng, pl, pops, comps, ro, params, scale, odors):
    e, m = make_engine(eng, pops, params, scale)
    n_apl = len(pops.apl)
    is_kc = np.zeros(e.N, bool); is_kc[pops.kc] = True
    out = []
    for o in odors:
        for seed in o["seeds"]:
            e.reset(seed); e.clear_drive(); present(e, pops, o["strengths"], STRENGTH); e.run(SETTLE_MS)
            counts = np.zeros(e.N, np.int32)
            kc_steps = np.zeros(READ_STEPS, np.int64)
            apl_v = np.zeros((READ_STEPS, n_apl), np.float64)
            rel = 0.0
            for i in range(READ_STEPS):
                idx = e.step()
                counts[idx] += 1
                kc_steps[i] = int(is_kc[idx].sum())
                va = e.v[pops.apl]
                apl_v[i] = va
                rel += float(e.apl_release(va).astype(np.float64).sum())
            kc = counts[pops.kc]
            out.append(dict(odor=o["name"], seed=int(seed), n_edges_scaled=int(m.sum()),
                            kc_active_frac=float((kc > 0).mean()), kc_spikes=int(kc.sum()),
                            apl_v_mean=float(apl_v.mean()), apl_v_mean_per_cell=apl_v.mean(axis=0).tolist(),
                            apl_v_sd=float(apl_v.mean(axis=1).std()),
                            release_frac=rel / (READ_STEPS * n_apl) / e.p.apl_r_max,
                            kc_acmax=acmax(kc_steps)))
    return out


def med(row, key):
    return float(np.median([p[key] for p in row["presentations"]]))


def interp_cross(scales, mv, kc):
    """First adjacent pair whose median membrane brackets TARGET_MV; s log-linear, KC % linear in s."""
    for i in range(len(scales) - 1):
        a, b = mv[i] - TARGET_MV, mv[i + 1] - TARGET_MV
        if a == 0:
            return scales[i], scales[i], scales[i], kc[i]
        if a * b < 0 or b == 0:
            t = (TARGET_MV - mv[i]) / (mv[i + 1] - mv[i])
            ls = np.log(scales[i]) + t * (np.log(scales[i + 1]) - np.log(scales[i]))
            s = float(np.exp(ls))
            k = kc[i] + (s - scales[i]) / (scales[i + 1] - scales[i]) * (kc[i + 1] - kc[i])
            return scales[i], scales[i + 1], s, float(k)
    return None


def report(rows, meta, kc_list, scales, wall_s):
    n_pres = len(rows[0]["presentations"])
    L = ["# M0d H.3a calibration: APL input scale sweep on the H.3 reference odour set", "",
         f"Diagnostic only (not an H.3 selection). Reference set: {meta['n_odors']} odours (rng 800000, 6-9 glomeruli, "
         f"DA1/V excluded, strength ∝ 1/receptor count), strength {STRENGTH}, 2 seeds each, settle {SETTLE_MS} ms, "
         f"read {READ_STEPS} steps. Graded APL, apl_r_max {meta['apl_r_max']:g}, other Params at defaults. "
         f"s multiplies every CSC edge into APL ({meta['apl_edges']['n_edges_into_apl']} edges; summed in-weight per APL cell at s=1: "
         + ", ".join(f"{w:.1f}" for w in meta['apl_edges']['in_weight_pre_mv']) + " mV). "
         f"Statistics over {n_pres} presentations per config. Release shares: < {REL_BINS[0]} / in [{REL_BINS[0]}, {REL_BINS[1]}] / > {REL_BINS[1]} of apl_r_max.", ""]
    summary = {}
    for k in kc_list:
        sub = sorted([r for r in rows if r["params"]["kc_thresh"] == k], key=lambda r: r["scale"])
        L.extend([f"## kc_thresh = {k:g}", "",
                  "| s | median APL mV | IQR APL mV | median release_frac | rel < 0.38 | rel 0.38-0.62 | rel > 0.62 | median KC active % | median APL v SD | median kc_acmax |",
                  "|---|---|---|---|---|---|---|---|---|---|"])
        mv, kcp, ac = [], [], []
        for r in sub:
            ps = r["presentations"]
            v = np.array([p["apl_v_mean"] for p in ps]); rel = np.array([p["release_frac"] for p in ps])
            q1, q3 = np.percentile(v, [25, 75])
            shares = ((rel < REL_BINS[0]).mean(), ((rel >= REL_BINS[0]) & (rel <= REL_BINS[1])).mean(), (rel > REL_BINS[1]).mean())
            r["shares"] = [float(x) for x in shares]
            mv.append(float(np.median(v))); kcp.append(100 * med(r, "kc_active_frac")); ac.append(med(r, "kc_acmax"))
            L.append(f"| {r['scale']:.4g} | {mv[-1]:.2f} | {q1:.2f}-{q3:.2f} ({q3 - q1:.2f}) | {np.median(rel):.3f} | "
                     f"{shares[0]:.2f} | {shares[1]:.2f} | {shares[2]:.2f} | {kcp[-1]:.2f} | {med(r, 'apl_v_sd'):.2f} | {ac[-1]:.3f} |")
        s_list = [r["scale"] for r in sub]
        dec = sum(mv[i + 1] < mv[i] for i in range(len(mv) - 1))
        inc = sum(kcp[i + 1] > kcp[i] for i in range(len(kcp) - 1))
        cross = interp_cross(s_list, mv, kcp)
        in_band = [s for s, x in zip(s_list, mv) if BAND_MV[0] <= x <= BAND_MV[1]]
        near = int(np.argmin([abs(x - TARGET_MV) for x in mv]))
        L.extend(["", "Summary:", "",
                  f"- Monotonicity: median membrane decreases on {dec} of {len(mv) - 1} adjacent-s steps; "
                  f"median KC active % increases on {inc} of {len(kcp) - 1}.",
                  (f"- Median membrane crosses {TARGET_MV:g} mV between s={cross[0]:.4g} and s={cross[1]:.4g}; "
                   f"log-linear s at {TARGET_MV:g} mV = {cross[2]:.4g}; interpolated KC active % there = {cross[3]:.2f}."
                   if cross else f"- Median membrane does not cross {TARGET_MV:g} mV inside [{s_list[0]:g}, {s_list[-1]:g}] "
                   f"(range {min(mv):.2f}-{max(mv):.2f} mV)."),
                  f"- Grid s with median membrane in {BAND_MV[0]:g}-{BAND_MV[1]:g} mV: "
                  + (", ".join(f"{s:.4g} ({x:.2f} mV)" for s, x in zip(s_list, mv) if s in in_band) if in_band else "none") + ".",
                  f"- Grid s nearest {TARGET_MV:g} mV: {s_list[near]:.4g} ({mv[near]:.2f} mV); release shares < / in / > band: "
                  + " / ".join(f"{x:.2f}" for x in sub[near]["shares"]) + ".",
                  f"- Median kc_acmax range over s: {min(ac):.3f}-{max(ac):.3f}.", ""])
        summary[str(k)] = dict(scales=s_list, median_mv=mv, median_kc_pct=kcp, median_acmax=ac, mv_decreases=dec,
                               kc_increases=inc, cross=cross, band_scales=in_band, nearest_scale=s_list[near],
                               nearest_shares=sub[near]["shares"])
    L.extend([f"Wall time of the run: {wall_s:.0f} s ({wall_s / 60:.1f} min).", ""])
    return "\n".join(L), summary


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    t0 = time.time()
    conn = Connectome.load(NPZ); pops = Populations.from_connectome(conn); del conn
    odors = reference_odors(pops)
    if smoke:
        kc_list, scales, use, chunk, workers, out_dir = (1.5,), (SCALES[0], SCALES[-1]), odors[:2], 2, 4, "results/m0d/diag/smoke"
    else:
        kc_list, scales, use, chunk, workers, out_dir = KC_THRESH, SCALES, odors, CHUNK, 16, "results/m0d/diag"
    rows, jobs = [], []
    for k in kc_list:
        for s in scales:
            for c in range(0, len(use), chunk):
                rows.append((k, s)); jobs.append(dict(params=dict(apl_mode="graded", kc_thresh=k), scale=s, odors=use[c:c + chunk]))
    with FlyPool(NPZ, Params(), [{} for _ in range(workers)], workers=workers) as pool:
        (edge_meta,) = pool.run_jobs(meta_job, [dict(params=dict(apl_mode="graded", kc_thresh=kc_list[0]), scale=scales[0])])
        t1 = time.time()
        res = pool.run_jobs(job, jobs)
        t_sim = time.time() - t1
    merged = {}
    for (k, s), pres in zip(rows, res):
        merged.setdefault((k, s), []).extend(pres)
    out_rows = [dict(params=dict(apl_mode="graded", kc_thresh=k), scale=s, presentations=p) for (k, s), p in merged.items()]
    wall = time.time() - t0
    n_pres = sum(len(r["presentations"]) for r in out_rows)
    meta = dict(npz=NPZ, smoke=smoke, strength=STRENGTH, settle_ms=SETTLE_MS, read_steps=READ_STEPS, kc_thresh=list(kc_list),
                scales=list(scales), apl_r_max=Params().apl_r_max, n_odors=len(use), odors=use, apl_edges=edge_meta,
                acmax_lags=list(LAGS), workers=workers, wall_s=wall, sim_s=t_sim,
                s_per_presentation_worker=t_sim * workers / n_pres)
    md, summary = report(out_rows, meta, kc_list, scales, wall)
    meta["summary"] = summary
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/apl_input_scale_sweep.json", "w") as f:
        json.dump(dict(meta=meta, rows=out_rows), f, indent=1)
    with open(f"{out_dir}/apl_input_scale_sweep.md", "w") as f:
        f.write(md)
    print(md)
    print(f"sim {t_sim:.1f} s for {n_pres} presentations on {workers} workers; {meta['s_per_presentation_worker']:.2f} worker-s/presentation")
