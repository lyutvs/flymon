"""M0d H.3a calibration: APL release cap (apl_r_max) vs across-glomerulus KC-drive disparity. Diagnostic only.

Calibration data, not an H.3 selection: no M2 turns, no candidate odours, no oracle.

Stage A calibrates the cap 1.0 operating point on the H.3 reference odour set (48 odours x 2 seeds, strength 0.35,
settle 800 ms, read 600 steps; generator and CSC-into-APL scaling imported from apl_input_scale_sweep.py): a
kc_thresh x apl_input_scale grid, log-linear interpolation of the s where the median APL membrane crosses 11 mV,
then 4 log-scale bisections at the kc_thresh whose interpolated KC active % is nearest 7%. It also re-measures the
cap 0.333 point (kc_thresh 1.5, s = 0.0799) and the spiking-APL baseline C0 (kc_thresh 1.5, no scaling), recording
MBON05 and MBON13 read-window spike counts for all three.

Stage B runs the all51 protocol of lhpv3c1_ablation.py (each assignable glomerulus alone, channel strength
c_norm / receptor count, seeds 200/201, no plasticity) for C0, C1@0.333 and C1@1.0, and reports the log10 variance
of per-glomerulus mean KC read-window spikes (the decision metric), max/min, best 8x band, CV and min/median/max.

Writes results/m0d/diag/apl_cap_disparity.json (raw) and .md (report); --smoke runs a reduced version into
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
from flymon.brain.stimuli import present

from apl_input_scale_sweep import (BAND_MV, NPZ, READ_STEPS, REL_BINS, SETTLE_MS, STRENGTH, TARGET_MV,
                                   interp_cross, make_engine, reference_odors)

EXCLUDE = ("ORN_DA1", "ORN_V")          # spec 3.3: cVA and CO2 channels are not free to reassign
KC_GRID = (1.0, 1.25, 1.5)
SCALES = tuple(float(s) for s in np.geomspace(0.005, 1.0, 8))
N_BISECT = 4
TARGET_KC_PCT = 7.0
KC_BAND_PCT = (6.0, 8.0)
CAP0_SCALE = 0.0799                     # the sweep's interpolated s at 11 mV for apl_r_max 0.333, kc_thresh 1.5
MBONS = ("MBON05", "MBON13")
ALL51_SEEDS = (200, 201)
BAND_THRESHOLDS = (100, 200, 400, 600)


def mbon_index(conn, pops):
    t = np.asarray(conn.type).astype(str)
    return {n: np.array([int(i) for i in pops.mbon if t[i] == n], np.int64) for n in MBONS}


# ---- jobs (module level so spawn can pickle them) ----------------------------------------------
def job_odors(eng, pl, pops, comps, ro, params, scale, odors):
    """One reference-odour block under Params(**params) with every CSC edge into APL scaled by `scale`."""
    e, m = make_engine(eng, pops, params, scale)
    graded = e.p.apl_mode == "graded"
    n_apl = len(pops.apl)
    is_kc = np.zeros(e.N, bool); is_kc[pops.kc] = True
    mb = mbon_index(eng.conn, pops)
    out = []
    for o in odors:
        for seed in o["seeds"]:
            e.reset(seed); e.clear_drive(); present(e, pops, o["strengths"], STRENGTH); e.run(SETTLE_MS)
            counts = np.zeros(e.N, np.int32)
            apl_v = np.zeros((READ_STEPS, n_apl), np.float64)
            rel = 0.0
            for i in range(READ_STEPS):
                idx = e.step()
                counts[idx] += 1
                va = e.v[pops.apl]
                apl_v[i] = va
                if graded:
                    rel += float(e.apl_release(va).astype(np.float64).sum())
            kc = counts[pops.kc]
            out.append(dict(odor=o["name"], seed=int(seed), n_edges_scaled=int(m.sum()),
                            kc_active_frac=float((kc > 0).mean()), kc_spikes=int(kc.sum()),
                            apl_spikes=int(counts[pops.apl].sum()),
                            apl_v_mean=float(apl_v.mean()), apl_v_sd=float(apl_v.mean(axis=1).std()),
                            release_frac=(rel / (READ_STEPS * n_apl) / e.p.apl_r_max) if graded else None,
                            **{n.lower(): int(counts[mb[n]].sum()) for n in MBONS}))
    return out


def job_all51(eng, pl, pops, comps, ro, params, scale, items, c_norm):
    """all51 protocol (lhpv3c1_ablation.py): one glomerulus alone per presentation, no plasticity."""
    e, _ = make_engine(eng, pops, params, scale)
    graded = e.p.apl_mode == "graded"
    out = []
    for rtype, seed in items:
        e.reset(int(seed)); e.clear_drive()
        present(e, pops, {rtype: c_norm / len(pops.receptor_types[rtype])}, STRENGTH)
        e.run(SETTLE_MS)
        counts = np.zeros(e.N, np.int32)
        v_sum = 0.0
        for _ in range(READ_STEPS):
            counts[e.step()] += 1
            v_sum += float(e.v[pops.apl].mean())
        kc = counts[pops.kc]
        out.append(dict(g=rtype, seed=int(seed), kc=int(kc.sum()), kc_on=int((kc > 0).sum()),
                        pn=int(counts[pops.alpn].sum()), apl=int(counts[pops.apl].sum()),
                        apl_v_mean=(v_sum / READ_STEPS) if graded else None))
    return out


# ---- statistics --------------------------------------------------------------------------------
def med(ps, key):
    return float(np.median([p[key] for p in ps]))


def point_stats(ps):
    """Summary of one operating point (a list of presentations)."""
    v = np.array([p["apl_v_mean"] for p in ps])
    kc_pct = 100 * med(ps, "kc_active_frac")
    q1, q3 = np.percentile(v, [25, 75])
    d = dict(n_pres=len(ps), median_mv=float(np.median(v)), iqr_mv=[float(q1), float(q3)],
             median_kc_pct=kc_pct, median_kc_spikes=med(ps, "kc_spikes"), median_apl_v_sd=med(ps, "apl_v_sd"))
    rel = [p["release_frac"] for p in ps]
    if all(r is not None for r in rel):
        rel = np.array(rel)
        d["median_release_frac"] = float(np.median(rel))
        d["shares"] = [float((rel < REL_BINS[0]).mean()), float(((rel >= REL_BINS[0]) & (rel <= REL_BINS[1])).mean()),
                       float((rel > REL_BINS[1]).mean())]
    else:
        d["median_release_frac"], d["shares"] = None, None
    for n in MBONS:
        x = np.array([p[n.lower()] for p in ps], float)
        d[n] = dict(median=float(np.median(x)), mean=float(x.mean()), zero_share=float((x == 0).mean()),
                    max=float(x.max()))
    return d


def all51_stats(rows):
    gs = sorted({r["g"] for r in rows})
    K = np.array([np.mean([r["kc"] for r in rows if r["g"] == g]) for g in gs])
    on = np.array([np.mean([r["kc_on"] for r in rows if r["g"] == g]) for g in gs])
    pn = np.array([np.mean([r["pn"] for r in rows if r["g"] == g]) for g in gs])
    lo = max(K.min(), 1.0)
    band = int(max(int(((K >= t) & (K <= 8 * t)).sum()) for t in BAND_THRESHOLDS))
    band_t = int(max(BAND_THRESHOLDS, key=lambda t: int(((K >= t) & (K <= 8 * t)).sum())))
    return dict(n_glom=len(gs), log10_var=float(np.var(np.log10(np.maximum(K, 1)))),
                max_over_min=float(K.max() / lo), band8x=band, band8x_threshold=band_t, n_glom_total=len(gs),
                cv=float(K.std() / K.mean()), kc_min=float(K.min()), kc_median=float(np.median(K)),
                kc_max=float(K.max()), kc_mean=float(K.mean()), kc_ge_200=int((K >= 200).sum()),
                kc_on_median=float(np.median(on)), pn_median=float(np.median(pn)),
                per_glom={g: float(k) for g, k in zip(gs, K)})


# ---- driver helpers ----------------------------------------------------------------------------
def run_point(pool, params, scale, odors, chunk):
    jobs = [dict(params=params, scale=scale, odors=odors[c:c + chunk]) for c in range(0, len(odors), chunk)]
    ps = []
    for part in pool.run_jobs(job_odors, jobs):
        ps.extend(part)
    return ps


def fmt_shares(s):
    return " / ".join(f"{x:.2f}" for x in s) if s else "n/a (spiking)"


def report(res, wall_s):
    a, b = res["stage_a"], res["stage_b"]
    L = ["# M0d H.3a calibration: APL release cap vs all51 KC-drive disparity", "",
         f"Diagnostic only (not an H.3 selection; no M2 turns, candidate odours or oracle). Stage A: H.3 reference odour set "
         f"({a['n_odors']} odours, rng 800000, 6-9 glomeruli, DA1/V excluded), strength {STRENGTH}, 2 seeds each, settle "
         f"{SETTLE_MS} ms, read {READ_STEPS} steps; s multiplies every CSC edge into APL. Stage B: all51 protocol of "
         f"lhpv3c1_ablation.py (each assignable glomerulus alone, strength c_norm/receptor count with c_norm="
         f"{b['c_norm']:g}, seeds {list(ALL51_SEEDS)}, no plasticity). Release shares: < {REL_BINS[0]} / in "
         f"[{REL_BINS[0]}, {REL_BINS[1]}] / > {REL_BINS[1]} of apl_r_max.", "",
         "## Stage A — cap 1.0 grid", "",
         "| kc_thresh | s | median APL mV | median KC active % | median release_frac | rel < 0.38 | rel 0.38-0.62 | rel > 0.62 |",
         "|---|---|---|---|---|---|---|---|"]
    for k in a["kc_grid"]:
        for s, st in zip(a["scales"], a["grid"][str(k)]["points"]):
            sh = st["shares"] or [float("nan")] * 3
            L.append(f"| {k:g} | {s:.4g} | {st['median_mv']:.2f} | {st['median_kc_pct']:.2f} | "
                     f"{st['median_release_frac']:.3f} | {sh[0]:.2f} | {sh[1]:.2f} | {sh[2]:.2f} |")
    L += ["", "Interpolation at 11 mV (log-linear in s, KC % linear in s):", ""]
    for k in a["kc_grid"]:
        g = a["grid"][str(k)]
        c = g["cross"]
        L.append(f"- kc_thresh {k:g}: " + (f"crossing inside the grid between s={c[0]:.4g} and s={c[1]:.4g}; "
                                           f"s at 11 mV = {c[2]:.4g}, interpolated KC active % = {c[3]:.2f}."
                                           if c else f"no 11 mV crossing inside [{a['scales'][0]:g}, {a['scales'][-1]:g}] "
                                                     f"(median membrane {g['mv_min']:.2f}-{g['mv_max']:.2f} mV)."))
    L += ["", f"Chosen kc_thresh for the cap 1.0 bisection: {a['chosen_k']:g} (interpolated KC active % "
              f"{a['chosen_interp_kc_pct']:.2f}, nearest {TARGET_KC_PCT:g}%).", ""]
    if a["bisect"]:
        L += ["| bisection step | s | median APL mV | median KC active % |", "|---|---|---|---|"]
        for i, st in enumerate(a["bisect"], 1):
            L.append(f"| {i} | {st['scale']:.5g} | {st['median_mv']:.2f} | {st['median_kc_pct']:.2f} |")
        L.append("")
    L += ["## Stage A — accepted operating points", "",
          "| config | apl_mode | apl_r_max | kc_thresh | s | median APL mV | IQR mV | median KC active % | median release_frac | release shares |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for name in ("C0", "C1@0.333", "C1@1.0"):
        c = res["configs"][name]
        st = c["stats"]
        p = c["params"]
        L.append(f"| {name} | {p.get('apl_mode', 'spiking')} | {p.get('apl_r_max', Params().apl_r_max):g} | "
                 f"{p.get('kc_thresh', Params().kc_thresh):g} | {c['scale']:.4g} | {st['median_mv']:.2f} | "
                 f"{st['iqr_mv'][0]:.2f}-{st['iqr_mv'][1]:.2f} | {st['median_kc_pct']:.2f} | "
                 + (f"{st['median_release_frac']:.3f}" if st["median_release_frac"] is not None else "n/a")
                 + f" | {fmt_shares(st['shares'])} |")
    L += ["", "MBON read-window spike counts over the reference set (naive network, no plasticity):", "",
          "| config | MBON05 median | MBON05 mean | MBON05 zero-share | MBON13 median | MBON13 mean | MBON13 zero-share |",
          "|---|---|---|---|---|---|---|"]
    for name in ("C0", "C1@0.333", "C1@1.0"):
        st = res["configs"][name]["stats"]
        L.append(f"| {name} | {st['MBON05']['median']:.1f} | {st['MBON05']['mean']:.2f} | {st['MBON05']['zero_share']:.2f} | "
                 f"{st['MBON13']['median']:.1f} | {st['MBON13']['mean']:.2f} | {st['MBON13']['zero_share']:.2f} |")
    L += ["", "## Stage B — all51 KC-drive disparity", "",
          "| config | log10 var (decision metric) | max/min | best 8x band | CV | KC min | KC median | KC max | KC>=200 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name in ("C0", "C1@0.333", "C1@1.0"):
        s = b["stats"][name]
        L.append(f"| {name} | {s['log10_var']:.3f} | {s['max_over_min']:.0f}x | {s['band8x']}/{s['n_glom']} "
                 f"(t={s['band8x_threshold']}) | {s['cv']:.2f} | {s['kc_min']:.1f} | {s['kc_median']:.0f} | "
                 f"{s['kc_max']:.0f} | {s['kc_ge_200']} |")
    d = res["decision"]
    L += ["", "## Decision", "",
          "> Take the cap with the smaller log10 variance of per-glomerulus KC drive. "
          "If (larger - smaller) / larger < 0.10, keep 0.333.", "",
          f"- log10 var: C1@0.333 = {d['var_0333']:.3f}, C1@1.0 = {d['var_10']:.3f}; smaller = {d['smaller']}.",
          f"- margin (larger - smaller) / larger = {d['margin']:.3f} ({'<' if d['margin'] < 0.10 else '>='} 0.10).",
          f"- **The rule selects apl_r_max = {d['selected']:g}** ({d['reason']}).", "",
          "## Premises", "",
          f"- The cap 1.0 grid {'contained' if a['chosen_cross'] else 'did NOT contain'} the 11 mV crossing for the chosen "
          f"kc_thresh {a['chosen_k']:g}.",
          f"- Accepted C1@1.0 point: median membrane {res['configs']['C1@1.0']['stats']['median_mv']:.2f} mV "
          f"(band {BAND_MV[0]:g}-{BAND_MV[1]:g}), KC active {res['configs']['C1@1.0']['stats']['median_kc_pct']:.2f}% — "
          f"{'inside' if KC_BAND_PCT[0] <= res['configs']['C1@1.0']['stats']['median_kc_pct'] <= KC_BAND_PCT[1] else 'OUTSIDE'} "
          f"the {KC_BAND_PCT[0]:g}-{KC_BAND_PCT[1]:g}% band.",
          f"- Accepted C1@0.333 point (s = {CAP0_SCALE:g}, the sweep's interpolated value): median membrane "
          f"{res['configs']['C1@0.333']['stats']['median_mv']:.2f} mV, KC active "
          f"{res['configs']['C1@0.333']['stats']['median_kc_pct']:.2f}% — "
          f"{'inside' if KC_BAND_PCT[0] <= res['configs']['C1@0.333']['stats']['median_kc_pct'] <= KC_BAND_PCT[1] else 'OUTSIDE'} "
          f"the {KC_BAND_PCT[0]:g}-{KC_BAND_PCT[1]:g}% band.", "",
          f"Wall time of the run: {wall_s:.0f} s ({wall_s / 60:.1f} min). Commit {res['meta']['commit']}.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    t0 = time.time()
    conn = Connectome.load(NPZ); pops = Populations.from_connectome(conn)
    odors_all = reference_odors(pops)
    rts_all = sorted(t for t in pops.receptor_types if t not in EXCLUDE)
    c_norm = float(np.median([len(pops.receptor_types[t]) for t in rts_all]))
    del conn

    if smoke:
        kc_grid, scales, odors, rts = (1.5,), (SCALES[0], SCALES[-1]), odors_all[:4], rts_all[:4]
        n_bisect, chunk, workers, out_dir = 1, 2, 4, "results/m0d/diag/smoke"
    else:
        kc_grid, scales, odors, rts = KC_GRID, SCALES, odors_all, rts_all
        n_bisect, chunk, workers, out_dir = N_BISECT, 6, 16, "results/m0d/diag"

    raw = {}          # (config label, scale) -> presentations
    grid_sum = {}
    with FlyPool(NPZ, Params(), [{} for _ in range(workers)], workers=workers) as pool:
        # Stage A, step 1-2: cap 1.0 grid
        for k in kc_grid:
            pts, mv, kcp = [], [], []
            for s in scales:
                ps = run_point(pool, dict(apl_mode="graded", apl_r_max=1.0, kc_thresh=k), s, odors, chunk)
                raw[(f"grid_k{k:g}", s)] = ps
                st = point_stats(ps); st["scale"] = s
                pts.append(st); mv.append(st["median_mv"]); kcp.append(st["median_kc_pct"])
            cross = interp_cross(list(scales), mv, kcp)
            grid_sum[str(k)] = dict(points=pts, median_mv=mv, median_kc_pct=kcp, cross=cross,
                                    mv_min=min(mv), mv_max=max(mv))
        # step 3: pick k, then bisect s in log space
        with_cross = [k for k in kc_grid if grid_sum[str(k)]["cross"]]
        pool_k = with_cross or list(kc_grid)
        def interp_kc(k):
            g = grid_sum[str(k)]
            return g["cross"][3] if g["cross"] else g["median_kc_pct"][int(np.argmin([abs(x - TARGET_MV) for x in g["median_mv"]]))]
        chosen_k = min(pool_k, key=lambda k: abs(interp_kc(k) - TARGET_KC_PCT))
        g = grid_sum[str(chosen_k)]
        bisect_log = []
        if g["cross"]:
            lo, hi = g["cross"][0], g["cross"][1]
            evaluated = {float(p["scale"]): p for p in g["points"]}
            for _ in range(n_bisect):
                s = float(np.exp(0.5 * (np.log(lo) + np.log(hi))))
                ps = run_point(pool, dict(apl_mode="graded", apl_r_max=1.0, kc_thresh=chosen_k), s, odors, chunk)
                raw[("bisect", s)] = ps
                st = point_stats(ps); st["scale"] = s
                bisect_log.append(st); evaluated[s] = st
                if st["median_mv"] < TARGET_MV:
                    lo = s
                else:
                    hi = s
            accepted_s = min((lo, hi), key=lambda x: abs(evaluated[x]["median_mv"] - TARGET_MV))
            accepted_ps = raw.get(("bisect", accepted_s)) or raw[(f"grid_k{chosen_k:g}", accepted_s)]
        else:
            i = int(np.argmin([abs(x - TARGET_MV) for x in g["median_mv"]]))
            accepted_s = float(scales[i])
            accepted_ps = raw[(f"grid_k{chosen_k:g}", accepted_s)]

        configs = {
            "C0": dict(params=dict(kc_thresh=1.5), scale=1.0),
            "C1@0.333": dict(params=dict(apl_mode="graded", apl_r_max=0.333, kc_thresh=1.5), scale=CAP0_SCALE),
            "C1@1.0": dict(params=dict(apl_mode="graded", apl_r_max=1.0, kc_thresh=chosen_k), scale=accepted_s),
        }
        # steps 4-5: reference-set runs for C0 and C1@0.333 (C1@1.0 reuses the accepted bisection run)
        configs["C1@1.0"]["presentations"] = accepted_ps
        for name in ("C0", "C1@0.333"):
            configs[name]["presentations"] = run_point(pool, configs[name]["params"], configs[name]["scale"], odors, chunk)
        for name, c in configs.items():
            c["stats"] = point_stats(c["presentations"])

        # Stage B: all51 for the three configs
        items = [(r, seed) for r in rts for seed in ALL51_SEEDS]
        b_chunk = max(1, len(items) // workers + (1 if len(items) % workers else 0))
        b_rows, b_stats = {}, {}
        for name, c in configs.items():
            jobs = [dict(params=c["params"], scale=c["scale"], items=items[i:i + b_chunk], c_norm=c_norm)
                    for i in range(0, len(items), b_chunk)]
            rows = [r for part in pool.run_jobs(job_all51, jobs) for r in part]
            b_rows[name] = rows
            b_stats[name] = all51_stats(rows)

    v0, v1 = b_stats["C1@0.333"]["log10_var"], b_stats["C1@1.0"]["log10_var"]
    smaller = "C1@0.333" if v0 <= v1 else "C1@1.0"
    larger_v, smaller_v = max(v0, v1), min(v0, v1)
    margin = (larger_v - smaller_v) / larger_v if larger_v > 0 else 0.0
    if margin < 0.10:
        selected, reason = 0.333, f"margin {margin:.3f} < 0.10, so the rule keeps 0.333"
    else:
        selected = 0.333 if smaller == "C1@0.333" else 1.0
        reason = f"{smaller} has the smaller log10 variance and the margin {margin:.3f} >= 0.10"

    wall = time.time() - t0
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    res = dict(
        meta=dict(npz=NPZ, smoke=smoke, commit=commit, strength=STRENGTH, settle_ms=SETTLE_MS, read_steps=READ_STEPS,
                  target_mv=TARGET_MV, band_mv=list(BAND_MV), target_kc_pct=TARGET_KC_PCT, kc_band_pct=list(KC_BAND_PCT),
                  workers=workers, wall_s=wall),
        stage_a=dict(n_odors=len(odors), kc_grid=list(kc_grid), scales=list(scales), grid=grid_sum,
                     chosen_k=float(chosen_k), chosen_interp_kc_pct=float(interp_kc(chosen_k)),
                     chosen_cross=grid_sum[str(chosen_k)]["cross"], n_bisect=n_bisect, bisect=bisect_log,
                     accepted_scale=float(accepted_s), odors=odors),
        stage_b=dict(c_norm=c_norm, seeds=list(ALL51_SEEDS), n_glom=len(rts), stats=b_stats, rows=b_rows),
        configs={n: dict(params=c["params"], scale=float(c["scale"]), stats=c["stats"],
                         presentations=c["presentations"]) for n, c in configs.items()},
        decision=dict(var_0333=v0, var_10=v1, smaller=smaller, margin=float(margin), selected=selected, reason=reason),
    )
    md = report(res, wall)
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/apl_cap_disparity.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    with open(f"{out_dir}/apl_cap_disparity.md", "w") as f:
        f.write(md)
    print(md)
    n_pres = sum(len(v) for v in raw.values()) + sum(len(configs[n]["presentations"]) for n in ("C0", "C1@0.333"))
    print(f"wall {wall:.1f} s; {n_pres} reference presentations + {3 * len(items)} all51 presentations on {workers} workers")
