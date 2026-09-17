"""M0d C2 feasibility: does ORN->PN depression (orn_std) leave a 6-8% KC operating point under graded APL? Diagnostic only.

Calibration diagnostic, not an H.3 selection: stimuli are only the M0 design odour pair (design_odor_pair k=8 seed=0)
at strength 0.35, seeds 100-102. Grids: C2 (graded APL + orn_std) and C1 (graded APL, no orn_std) over apl_r_max x
kc_thresh, plus a receptor out-edge weight sweep at (0.33, 1.5) with and without orn_std.
Writes results/m0d/diag/c2_std_feasibility.json (raw) and .md (report).
"""
import json
import os

import numpy as np

from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.engine_cpu import Engine
from flymon.brain.fly_pool import FlyPool
from flymon.brain.stimuli import design_odor_pair, present

NPZ = "data/malecns.npz"
OUT_DIR = "results/m0d/diag"
STRENGTH = 0.35
SEEDS = (100, 101, 102)
SETTLE_MS, READ_STEPS = 800, 600
R_MAX = (0.1, 0.2, 0.33, 0.5, 1.0)
KC_THRESH = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
SCALES = (1, 2, 4, 8)
COMP = dict(apl_r_max=0.33, kc_thresh=1.5)
BAND = (6.0, 8.0)


def job(eng, pl, pops, comps, ro, params, scale, odors):
    e = Engine(eng.conn, pops, Params(**params))
    if scale != 1:
        e.csc.w[e.is_receptor[e.csc.pre_of_edge()]] *= np.float32(scale)
    out = []
    for name, odor in odors:
        stim = np.concatenate([pops.receptor_types[t] for t in odor])
        for seed in SEEDS:
            e.reset(seed); e.clear_drive(); present(e, pops, odor, STRENGTH); e.run(SETTLE_MS)
            counts = np.zeros(e.N, np.int32)
            apl_v = np.zeros(len(pops.apl), np.float64)
            for _ in range(READ_STEPS):
                counts[e.step()] += 1
                apl_v += e.v[pops.apl]
            kc = counts[pops.kc]
            out.append(dict(odor=name, seed=seed,
                            kc_active_frac=float((kc > 0).mean()), kc_spikes=int(kc.sum()),
                            alpn_spikes=int(counts[pops.alpn].sum()), receptor_spikes=int(counts[stim].sum()),
                            apl_v_mean=float(apl_v.mean() / READ_STEPS) if e.p.apl_mode == "graded" else None,
                            apl_v_mean_per_cell=(apl_v / READ_STEPS).tolist() if e.p.apl_mode == "graded" else None,
                            std_r_mean=float(e._std_r[stim].mean()) if e.p.orn_std else None))
    return out


def configs():
    rows = []
    for grid, std in (("C2", True), ("C1", False)):
        for r in R_MAX:
            for k in KC_THRESH:
                rows.append(dict(grid=grid, scale=1, params=dict(apl_mode="graded", orn_std=std, apl_r_max=r, kc_thresh=k)))
    for std in (True, False):
        for s in SCALES:
            rows.append(dict(grid="comp", scale=s, params=dict(apl_mode="graded", orn_std=std, **COMP)))
    return rows


def med(row, key):
    return float(np.median([p[key] for p in row["presentations"]]))


def ratio(a, b):
    return a / b if b else float("inf") if a else float("nan")


def report(rows, odors):
    cell = {(r["grid"], r["params"]["apl_r_max"], r["params"]["kc_thresh"]): r for r in rows if r["grid"] != "comp"}
    L = ["# M0d C2 feasibility: ORN->PN depression and the 6-8% KC operating point", "",
         f"Stimuli: design_odor_pair(k=8, seed=0) A = {sorted(odors[0][1])}, B = {sorted(odors[1][1])}; "
         f"strength {STRENGTH}; seeds {list(SEEDS)}; settle {SETTLE_MS} ms, read {READ_STEPS} steps. "
         "All configs graded APL; other Params at defaults. Medians are over the 6 presentations (2 odours x 3 seeds). "
         f"Cells marked ** have median KC active % in [{BAND[0]:g}, {BAND[1]:g}].", ""]
    hdr = "| apl_r_max \\ kc_thresh | " + " | ".join(f"{k:g}" for k in KC_THRESH) + " |"
    sep = "|---|" + "---|" * len(KC_THRESH)

    def table(title, fn):
        L.extend([f"### {title}", "", hdr, sep])
        for r in R_MAX:
            L.append(f"| {r:g} | " + " | ".join(fn(r, k) for k in KC_THRESH) + " |")
        L.append("")

    def kc_cell(grid):
        def f(r, k):
            v = 100 * med(cell[(grid, r, k)], "kc_active_frac")
            return f"**{v:.2f}**" if BAND[0] <= v <= BAND[1] else f"{v:.2f}"
        return f

    for grid, label in (("C2", "C2 (orn_std=True)"), ("C1", "C1 (orn_std=False)")):
        L.append(f"## {label}"); L.append("")
        table("Median KC active %", kc_cell(grid))
        table("Median APL membrane (mV above rest, mean over read window and both APL cells)",
              lambda r, k, g=grid: f"{med(cell[(g, r, k)], 'apl_v_mean'):.2f}")
        table("Median ALPN spikes (read window)", lambda r, k, g=grid: f"{med(cell[(g, r, k)], 'alpn_spikes'):.0f}")

    L.extend(["## C2 / C1 ratios (ratio of medians)", ""])
    table("ALPN spikes C2/C1", lambda r, k: f"{ratio(med(cell[('C2', r, k)], 'alpn_spikes'), med(cell[('C1', r, k)], 'alpn_spikes')):.3f}")
    table("KC active C2/C1", lambda r, k: f"{ratio(med(cell[('C2', r, k)], 'kc_active_frac'), med(cell[('C1', r, k)], 'kc_active_frac')):.3f}")
    alpn_ratios = [ratio(med(cell[("C2", r, k)], "alpn_spikes"), med(cell[("C1", r, k)], "alpn_spikes")) for r in R_MAX for k in KC_THRESH]

    c2 = [p for (g, _, _), row in cell.items() if g == "C2" for p in row["presentations"]]
    by_cell = [np.mean([p["std_r_mean"] for p in row["presentations"]]) for (g, _, _), row in cell.items() if g == "C2"]
    per_odor = {o: np.mean([p["std_r_mean"] for p in c2 if p["odor"] == o]) for o in ("A", "B")}
    rec_same = all(len({p["receptor_spikes"] for p in c2 if (p["odor"], p["seed"]) == (o, s)}) == 1 for o in ("A", "B") for s in SEEDS)
    L.extend(["## Depressed receptor gain in C2", "",
              f"Mean end-of-read `_std_r` over the stimulated receptors: {np.mean(by_cell):.4f} "
              f"(per-cell mean over presentations ranges {min(by_cell):.4f}-{max(by_cell):.4f}; odour A {per_odor['A']:.4f}, odour B {per_odor['B']:.4f}). "
              f"Stimulated-receptor spike counts identical across all C2 cells for each (odour, seed): {rec_same}.", ""])

    comp = {(r["params"]["orn_std"], r["scale"]): r for r in rows if r["grid"] == "comp"}
    c1_ref = med(comp[(False, 1)], "alpn_spikes")
    L.extend([f"## Receptor out-edge weight sweep (apl_r_max={COMP['apl_r_max']}, kc_thresh={COMP['kc_thresh']})", "",
              f"ALPN ratio is against the orn_std=False, s=1 median ({c1_ref:.0f} spikes).", "",
              "| s | orn_std | median ALPN spikes | ALPN / C1(s=1) | median KC active % | median APL mV | mean end `_std_r` |",
              "|---|---|---|---|---|---|---|"])
    for std in (True, False):
        for s in SCALES:
            row = comp[(std, s)]
            sr = f"{np.mean([p['std_r_mean'] for p in row['presentations']]):.4f}" if std else "-"
            L.append(f"| {s} | {std} | {med(row, 'alpn_spikes'):.0f} | {ratio(med(row, 'alpn_spikes'), c1_ref):.3f} | "
                     f"{100 * med(row, 'kc_active_frac'):.2f} | {med(row, 'apl_v_mean'):.2f} | {sr} |")
    s1_same = all(comp[(std, 1)]["presentations"] == cell[("C2" if std else "C1", COMP["apl_r_max"], COMP["kc_thresh"])]["presentations"]
                  for std in (True, False))
    L.extend(["", f"s=1 rows identical to the matching grid cells: {s1_same}.", ""])

    def in_band(ks):
        return [(r, k, 100 * med(cell[("C2", r, k)], "kc_active_frac")) for r in R_MAX for k in ks
                if BAND[0] <= 100 * med(cell[("C2", r, k)], "kc_active_frac") <= BAND[1]]

    def fmt(hits):
        return ", ".join(f"(apl_r_max {r:g}, kc_thresh {k:g}: {v:.2f}%)" for r, k, v in hits) or "none"

    c2_kc = [100 * med(cell[("C2", r, k)], "kc_active_frac") for r in R_MAX for k in KC_THRESH]
    hi = [100 * med(cell[("C2", r, k)], "kc_active_frac") for r in R_MAX for k in KC_THRESH if k >= 1.0]
    lo = [100 * med(cell[("C2", r, k)], "kc_active_frac") for r in R_MAX for k in KC_THRESH if k < 1.0]
    restore = [s for s in SCALES if med(comp[(True, s)], "alpn_spikes") >= c1_ref]
    closest = min(SCALES, key=lambda s: abs(ratio(med(comp[(True, s)], "alpn_spikes"), c1_ref) - 1))
    L.extend(["## Summary", "",
              f"Within the declared H.3 grid (kc_thresh >= 1.0), C2 cells with median KC active % in 6-8%: {fmt(in_band([k for k in KC_THRESH if k >= 1.0]))}; "
              f"C2 median KC active % there spans {min(hi):.2f}-{max(hi):.2f}%. "
              f"Within the extended kc_thresh 0.5-0.75, C2 cells in 6-8%: {fmt(in_band([k for k in KC_THRESH if k < 1.0]))}; "
              f"C2 median KC active % there spans {min(lo):.2f}-{max(lo):.2f}% (whole C2 grid {min(c2_kc):.2f}-{max(c2_kc):.2f}%). "
              f"Across all 35 cells the ALPN spike ratio C2/C1 spans {min(alpn_ratios):.3f}-{max(alpn_ratios):.3f}. "
              f"Under depression at (apl_r_max {COMP['apl_r_max']}, kc_thresh {COMP['kc_thresh']}), receptor weight scales s in {list(SCALES)} give ALPN / C1(s=1) = "
              + ", ".join(f"{ratio(med(comp[(True, s)], 'alpn_spikes'), c1_ref):.3f} (s={s})" for s in SCALES) + "; "
              + (f"the smallest s reaching C1-level ALPN spikes (ratio >= 1) is {restore[0]}." if restore
                 else f"no tested s reaches C1-level ALPN spikes; the closest is s={closest}."), ""])
    return "\n".join(L), alpn_ratios


if __name__ == "__main__":
    conn = Connectome.load(NPZ); pops = Populations.from_connectome(conn); del conn
    A, B = design_odor_pair(pops, k=8, seed=0)
    odors = [("A", {str(t): float(s) for t, s in A.items()}), ("B", {str(t): float(s) for t, s in B.items()})]
    rows = configs()
    with FlyPool(NPZ, Params(), [{} for _ in range(16)], workers=16) as pool:
        res = pool.run_jobs(job, [dict(params=r["params"], scale=r["scale"], odors=odors) for r in rows])
    for r, pres in zip(rows, res):
        r["presentations"] = pres
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/c2_std_feasibility.json", "w") as f:
        json.dump(dict(meta=dict(npz=NPZ, strength=STRENGTH, seeds=list(SEEDS), settle_ms=SETTLE_MS, read_steps=READ_STEPS,
                                 odors=dict(odors), band_pct=list(BAND)), rows=rows), f, indent=1)
    md, _ = report(rows, odors)
    with open(f"{OUT_DIR}/c2_std_feasibility.md", "w") as f:
        f.write(md)
    print(md)
