"""M0d H.3a: the D.4 sparsity/overlap margin measured at the gate's own protocol (settle 200 ms). Diagnostic only.

Calibration data, not an H.3 selection: no M2 turns, no candidate odours, no oracle, plasticity off.

Why: the H.3a draft ranks stage-1 candidates by their D.4 margin but measured it with settle 800 ms
(`docs/superpowers/specs/m0d-diag/two_checks.py`), while the D.4 gate calls `kc_sparsity` without a settle
argument, i.e. at its default **settle 200 ms** (`flymon/brain/measure.py`,
`scripts/reproduce_flybrain_measurements.py:cmd_sparsity`). On C0 that difference alone flips the Jaccard
clause. This script re-measures every stage-1 config on the gate's protocol and, in the same table, at
800 ms, so the protocol difference is quantified rather than assumed.

Protocol, exactly as the gate does it: `design_odor_pair(pops, k=8, seed=0)`, `kc_sparsity(engine, pops, odor,
strength=0.35, seed=s)` for s in 100-102 with `read_ms = 600` and `settle_ms` 200 (gate default) or 800 (the
M0d diagnostic window), then `jaccard` / `chance_jaccard` on the two active sets of the same seed; per-config
rows are the mean over the three seeds, as `cmd_sparsity` averages its seed rows. Those four functions are
imported from `flymon.brain.measure`, not reimplemented. Configs, the reference odour set and the CSC-into-APL
scaling are imported from `two_checks.py` / `apl_input_scale_sweep.py`.

Reference-set medians are *not* re-run here: they are read from `results/m0d/diag/two_checks.json`, which
measured them on the reference odour set (48 odours x 2 seeds, settle 800, read 600) for these same configs.

Writes results/m0d/diag/d4_margin_protocol.{json,md}; --smoke runs one config into results/m0d/diag/smoke/.
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
from flymon.brain.measure import chance_jaccard, jaccard, kc_sparsity
from flymon.brain.stimuli import design_odor_pair

from apl_input_scale_sweep import NPZ, READ_STEPS, STRENGTH, make_engine
from two_checks import CAND_LABEL, CONFIGS, D4_BAND, D4_MARGIN_PP, DESIGN_K, DESIGN_ODOR_SEED, DESIGN_SEEDS

GATE_SETTLE_MS = 200.0      # flymon.brain.measure.kc_sparsity default, what the D.4 gate uses
DIAG_SETTLE_MS = 800.0      # the M0d diagnostic window used by two_checks.py
SETTLES = (GATE_SETTLE_MS, DIAG_SETTLE_MS)
READ_MS = float(READ_STEPS)                     # Params.dt = 1 ms, so 600 read steps = read_ms 600
TWO_CHECKS = "results/m0d/diag/two_checks.json"
M0C_SPARSITY = "results/m0c/sparsity.json"
M0C_TOL_PP, M0C_TOL_J = 0.3, 0.003              # the sanity-check tolerances fixed before the run


# ---- jobs (module level so spawn can pickle them) ----------------------------------------------
def job_design(eng, pl, pops, comps, ro, params, scale, settle_ms, seeds):
    """The D.4 gate measurement on the design odour pair for one config and one settle window."""
    e, _ = make_engine(eng, pops, params, scale)
    a, b = design_odor_pair(pops, k=DESIGN_K, seed=DESIGN_ODOR_SEED)
    out = []
    for seed in seeds:
        ra = kc_sparsity(e, pops, a, STRENGTH, seed=int(seed), settle_ms=settle_ms, read_ms=READ_MS)
        rb = kc_sparsity(e, pops, b, STRENGTH, seed=int(seed), settle_ms=settle_ms, read_ms=READ_MS)
        out.append(dict(seed=int(seed), frac_active_A=ra["frac_active"], frac_active_B=rb["frac_active"],
                        jaccard=jaccard(ra["active"], rb["active"]),
                        chance=chance_jaccard(ra["frac_active"], rb["frac_active"]),
                        kc_hz_A=ra["kc_hz"], kc_hz_B=rb["kc_hz"],
                        mbon_hz_A=ra["mbon_hz"], mbon_hz_B=rb["mbon_hz"]))
    return out


# ---- statistics --------------------------------------------------------------------------------
def design_stats(rows):
    """Mean over seeds, as scripts/reproduce_flybrain_measurements.cmd_sparsity averages its seed rows."""
    keys = ("frac_active_A", "frac_active_B", "jaccard", "chance", "kc_hz_A", "kc_hz_B", "mbon_hz_A", "mbon_hz_B")
    m = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    lo, hi = 100 * D4_BAND[0], 100 * D4_BAND[1]
    pa, pb = 100 * m["frac_active_A"], 100 * m["frac_active_B"]
    m["pct_A"], m["pct_B"] = pa, pb
    m["pct_max"], m["pct_min"] = max(pa, pb), min(pa, pb)
    m["d4_sparsity_ok"] = bool(lo <= pa <= hi and lo <= pb <= hi)
    m["d4_overlap_ok"] = bool(m["jaccard"] <= m["chance"])
    m["d4_ok"] = bool(m["d4_sparsity_ok"] and m["d4_overlap_ok"])
    m["margin_pp"] = float(min(hi - m["pct_max"], m["pct_min"] - lo))     # D.4 sparsity margin, pp
    m["overlap_margin"] = float(m["chance"] - m["jaccard"])               # >= 0 means the clause holds
    m["per_seed"] = rows
    return m


def reference_medians():
    """Reference-set median KC % per config, reused from two_checks.json (not re-run here)."""
    r = json.load(open(TWO_CHECKS))
    out = {c["label"]: dict(ref_median_kc_pct=float(c["ref_median_kc_pct"]), scale=float(c["scale"]),
                            two_checks_ratio=float(c["ratio"]),
                            two_checks_pct_A=100 * float(c["design"]["frac_active_A"]),
                            two_checks_pct_B=100 * float(c["design"]["frac_active_B"]),
                            two_checks_jaccard=float(c["design"]["jaccard"]),
                            two_checks_chance=float(c["design"]["chance"]))
           for c in r["check1"]["configs"]}
    return out, dict(commit=r["meta"]["commit"], settle_ms=r["meta"]["settle_ms"], n_odors=r["meta"]["n_odors"],
                     source=TWO_CHECKS)


def m0c_check(d200):
    """C0 at the gate protocol vs the committed M0c record (kc_thresh 1.5, defaults elsewhere)."""
    try:
        rec = json.load(open(M0C_SPARSITY))
        rows = [g for g in rec["grid"] if abs(g["kc_thresh"] - 1.5) < 1e-9
                and abs(g["apl_scale"] - Params().apl_scale) < 1e-9
                and abs(g["mbon_hold_frac"] - Params().mbon_hold_frac) < 1e-9]
        if not rows:
            return None
        g = rows[0]
    except (OSError, KeyError, ValueError):
        return None
    da = 100 * abs(g["frac_active_A"] - d200["frac_active_A"])
    db = 100 * abs(g["frac_active_B"] - d200["frac_active_B"])
    dj = abs(g["jaccard"] - d200["jaccard"])
    dc = abs(g["chance"] - d200["chance"])
    return dict(source=M0C_SPARSITY, record=dict(pct_A=100 * g["frac_active_A"], pct_B=100 * g["frac_active_B"],
                                                 jaccard=g["jaccard"], chance=g["chance"],
                                                 seeds=g.get("sparsity_seeds")),
                here=dict(pct_A=d200["pct_A"], pct_B=d200["pct_B"], jaccard=d200["jaccard"], chance=d200["chance"]),
                diff_pp_A=da, diff_pp_B=db, diff_jaccard=dj, diff_chance=dc,
                tol_pp=M0C_TOL_PP, tol_jaccard=M0C_TOL_J,
                agrees=bool(da <= M0C_TOL_PP and db <= M0C_TOL_PP and dj <= M0C_TOL_J and dc <= M0C_TOL_J))


def band_from(row, ref_med):
    """Reference-set KC % band whose design-pair image keeps the 3-7% clause with D4_MARGIN_PP on both sides.

    Uses this config's two measured conversion ratios: r_max = design max odour / reference median maps the
    upper edge, r_min = design min odour / reference median maps the lower edge."""
    if not ref_med or ref_med <= 0:
        return None
    lo, hi = 100 * D4_BAND[0] + D4_MARGIN_PP, 100 * D4_BAND[1] - D4_MARGIN_PP
    r_max, r_min = row["pct_max"] / ref_med, row["pct_min"] / ref_med
    return dict(image_lo_pct=lo, image_hi_pct=hi, ratio_max=r_max, ratio_min=r_min,
                lo_pct=lo / r_min, hi_pct=hi / r_max, mid_pct=0.5 * (lo / r_min + hi / r_max),
                empty=bool(lo / r_min > hi / r_max))


# ---- report ------------------------------------------------------------------------------------
def fmt_table(res, settle):
    key = f"s{int(settle)}"
    L = [f"| config | s | reference median KC % | design A KC % | design B KC % | max/ref | Jaccard | chance | "
         f"J ≤ chance | 3-7% both | D.4 | margin pp |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in res["configs"]:
        d, ref = c[key], c["ref_median_kc_pct"]
        ratio = f"{c[key + '_ratio_max']:.3f}" if c[key + "_ratio_max"] is not None else "—"
        L.append(f"| {c['label']} | {c['scale']:.4g} | " + (f"{ref:.2f}" if ref else "—") +
                 f" | {d['pct_A']:.2f} | {d['pct_B']:.2f} | {ratio} | {d['jaccard']:.4f} | {d['chance']:.4f} | "
                 f"{'yes' if d['d4_overlap_ok'] else 'NO'} | {'yes' if d['d4_sparsity_ok'] else 'NO'} | "
                 f"{'PASS' if d['d4_ok'] else 'FAIL'} | {d['margin_pp']:+.2f} |")
    return L


def report(res, wall_s):
    meta, cand = res["meta"], res["candidate"]
    c200, c800 = cand["s200"], cand["s800"]
    L = ["# M0d H.3a: the D.4 margin at the gate's own protocol (settle 200 ms)", "",
         f"Diagnostic only (not an H.3 selection; no M2 turns, candidate odours or oracle; plasticity off). "
         f"The D.4 gate measures the design odour pair with `kc_sparsity` at its **default settle "
         f"{GATE_SETTLE_MS:g} ms** (`flymon/brain/measure.py`; `cmd_sparsity` calls it without a settle "
         f"argument), while `two_checks.py` measured the same configs at {DIAG_SETTLE_MS:g} ms. Both windows are "
         f"measured here on the same engines. Design pair `design_odor_pair(pops, k={DESIGN_K}, "
         f"seed={DESIGN_ODOR_SEED})`, strength {STRENGTH}, read {READ_MS:g} ms, seeds "
         f"{list(meta['design_seeds'])} averaged. D.4: KC active {100 * D4_BAND[0]:g}-{100 * D4_BAND[1]:g}% for "
         f"**both** odours and Jaccard ≤ chance (`scripts/write_m0_summary.gate_ok`; definitions imported from "
         f"`flymon.brain.measure`). Commit {meta['commit']} (dirty tree: {str(meta['dirty']).lower()}).", "",
         f"Reference-set medians are reused from `{meta['reference']['source']}` (48-odour H.3 reference set x 2 "
         f"seeds, settle {meta['reference']['settle_ms']} ms, commit {meta['reference']['commit']}) and were **not** "
         f"re-run here; only the design-pair numbers below are new. `max/ref` = (design-pair max odour) / "
         f"(reference median). `margin pp` = min(7 − design max, design min − 3).", "",
         f"## Settle {GATE_SETTLE_MS:g} ms — the gate's protocol", ""]
    L += fmt_table(res, GATE_SETTLE_MS)
    L += ["", f"## Settle {DIAG_SETTLE_MS:g} ms — the M0d diagnostic window (for comparison)", ""]
    L += fmt_table(res, DIAG_SETTLE_MS)

    p = res["m0c_check"]
    L += ["", "## Sanity check against the committed M0c record", ""]
    if p is None:
        L += [f"Not run: either C0 is absent from this run (--smoke) or `{M0C_SPARSITY}` has no matching row."]
    else:
        L += [f"C0 at settle {GATE_SETTLE_MS:g} ms vs `{p['source']}` (kc_thresh 1.5, apl_scale "
              f"{Params().apl_scale:g}, seeds {p['record']['seeds']}): KC % A {p['here']['pct_A']:.2f} vs "
              f"{p['record']['pct_A']:.2f} (Δ {p['diff_pp_A']:.3f} pp), B {p['here']['pct_B']:.2f} vs "
              f"{p['record']['pct_B']:.2f} (Δ {p['diff_pp_B']:.3f} pp), Jaccard {p['here']['jaccard']:.5f} vs "
              f"{p['record']['jaccard']:.5f} (Δ {p['diff_jaccard']:.5f}), chance {p['here']['chance']:.5f} vs "
              f"{p['record']['chance']:.5f} (Δ {p['diff_chance']:.5f}) — "
              f"**{'agree' if p['agrees'] else 'DO NOT agree'}** within {M0C_TOL_PP:g} pp and {M0C_TOL_J:g} Jaccard."]

    L += ["", "## Summary at the gate's protocol", "",
          f"- Preserved candidate {CAND_LABEL}: design pair A {c200['pct_A']:.2f}% / B {c200['pct_B']:.2f}%, "
          f"Jaccard {c200['jaccard']:.4f} vs chance {c200['chance']:.4f}. The 3-7% clause "
          f"**{'HOLDS' if c200['d4_sparsity_ok'] else 'FAILS'}**, the overlap clause "
          f"**{'HOLDS' if c200['d4_overlap_ok'] else 'FAILS'}** (chance − J = {c200['overlap_margin']:+.4f}); "
          f"both clauses together: **{'PASS' if c200['d4_ok'] else 'FAIL'}**, sparsity margin "
          f"{c200['margin_pp']:+.2f} pp. At settle {DIAG_SETTLE_MS:g} ms the same candidate is "
          f"{'PASS' if c800['d4_ok'] else 'FAIL'} with margin {c800['margin_pp']:+.2f} pp "
          f"(J {c800['jaccard']:.4f} vs chance {c800['chance']:.4f}).",
          f"- Overlap clause failures: {res['n_overlap_fail']['s200']}/{len(res['configs'])} configs at settle "
          f"{GATE_SETTLE_MS:g} ms, {res['n_overlap_fail']['s800']}/{len(res['configs'])} at "
          f"{DIAG_SETTLE_MS:g} ms. Configs failing at 200 ms: "
          + (", ".join(res["overlap_fail_labels"]["s200"]) or "none") + "; at 800 ms: "
          + (", ".join(res["overlap_fail_labels"]["s800"]) or "none") + ".",
          f"- Sparsity clause failures: {res['n_sparsity_fail']['s200']}/{len(res['configs'])} at "
          f"{GATE_SETTLE_MS:g} ms, {res['n_sparsity_fail']['s800']}/{len(res['configs'])} at "
          f"{DIAG_SETTLE_MS:g} ms.",
          f"- Conversion ratio (design max odour)/(reference median) at settle {GATE_SETTLE_MS:g} ms: "
          f"{res['ratio_max_range']['s200'][0]:.3f}-{res['ratio_max_range']['s200'][1]:.3f} over the configs "
          f"with a reference median on record, {cand['s200_ratio_max']:.3f} at {CAND_LABEL}; at "
          f"{DIAG_SETTLE_MS:g} ms {res['ratio_max_range']['s800'][0]:.3f}-"
          f"{res['ratio_max_range']['s800'][1]:.3f} ({cand['s800_ratio_max']:.3f} at the candidate)."]
    b = cand["band_s200"]
    if b is None:
        L += ["- Reference-set band: not computable (no reference median for the candidate)."]
    else:
        L += [f"- Reference-set KC % band whose design-pair image keeps the 3-7% clause with {D4_MARGIN_PP:g} pp "
              f"margin on both sides, using the candidate's measured ratios at settle {GATE_SETTLE_MS:g} ms "
              f"(max odour {b['ratio_max']:.3f}, min odour {b['ratio_min']:.3f}, target image "
              f"{b['image_lo_pct']:g}-{b['image_hi_pct']:g}%): "
              + ("**empty** — no reference median satisfies both edges at once "
                 f"(lower edge needs ≥ {b['lo_pct']:.2f}%, upper edge needs ≤ {b['hi_pct']:.2f}%)."
                 if b["empty"] else
                 f"**{b['lo_pct']:.2f}-{b['hi_pct']:.2f}%** (midpoint {b['mid_pct']:.2f}%).")]
        ok = res["band_overlap_note"]
        L += [f"  Overlap clause inside that band: {ok}"]
    L += ["", f"Wall time of the run: {wall_s:.0f} s ({wall_s / 60:.1f} min). Commit {meta['commit']}, "
              f"dirty tree: {str(meta['dirty']).lower()}.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    t0 = time.time()
    conn = Connectome.load(NPZ)
    pops = Populations.from_connectome(conn)
    del conn

    if smoke:
        configs, seeds = [c for c in CONFIGS if c[0] == CAND_LABEL], DESIGN_SEEDS[:1]
        workers, out_dir = 2, "results/m0d/diag/smoke"
    else:
        configs, seeds = CONFIGS, DESIGN_SEEDS
        workers, out_dir = 16, "results/m0d/diag"

    ref, ref_meta = reference_medians()

    jobs, owners = [], []
    for label, params, scale in configs:
        for settle in SETTLES:
            owners.append((label, settle))
            jobs.append(dict(params=params, scale=scale, settle_ms=float(settle), seeds=list(seeds)))
    with FlyPool(NPZ, Params(), [{} for _ in range(workers)], workers=workers) as pool:
        results = pool.run_jobs(job_design, jobs)
    raw = {o: design_stats(r) for o, r in zip(owners, results)}

    rows = []
    for label, params, scale in configs:
        r = ref.get(label)
        ref_med = r["ref_median_kc_pct"] if r else None
        row = dict(label=label, params=params, scale=float(scale), ref_median_kc_pct=ref_med,
                   ref_source=ref_meta["source"], two_checks=r)
        for settle in SETTLES:
            k = f"s{int(settle)}"
            d = raw[(label, settle)]
            row[k] = d
            row[k + "_ratio_max"] = (d["pct_max"] / ref_med) if ref_med else None
            row[k + "_ratio_min"] = (d["pct_min"] / ref_med) if ref_med else None
            row[k + "_ratio_mean"] = (0.5 * (d["pct_A"] + d["pct_B"]) / ref_med) if ref_med else None
        rows.append(row)

    cand_row = [r for r in rows if r["label"] == CAND_LABEL][0]
    cand_row["band_s200"] = band_from(cand_row["s200"], cand_row["ref_median_kc_pct"])
    cand_row["band_s800"] = band_from(cand_row["s800"], cand_row["ref_median_kc_pct"])

    n_overlap_fail, n_sparsity_fail, fail_labels, ratio_range = {}, {}, {}, {}
    for settle in SETTLES:
        k = f"s{int(settle)}"
        n_overlap_fail[k] = sum(1 for r in rows if not r[k]["d4_overlap_ok"])
        n_sparsity_fail[k] = sum(1 for r in rows if not r[k]["d4_sparsity_ok"])
        fail_labels[k] = [r["label"] for r in rows if not r[k]["d4_overlap_ok"]]
        rr = [r[k + "_ratio_max"] for r in rows if r[k + "_ratio_max"] is not None]
        ratio_range[k] = [float(min(rr)), float(max(rr))] if rr else [float("nan"), float("nan")]

    b = cand_row["band_s200"]
    if b is None or b["empty"]:
        band_note = "not applicable (the band is empty)."
    else:
        inside = [r for r in rows if r["ref_median_kc_pct"] is not None
                  and b["lo_pct"] <= r["ref_median_kc_pct"] <= b["hi_pct"]]
        bad = [r["label"] for r in inside if not r["s200"]["d4_overlap_ok"]]
        band_note = (f"{len(inside)} measured config(s) have a reference median inside the band "
                     f"({', '.join(r['label'] for r in inside) or 'none'}); overlap-clause failures among them: "
                     + (", ".join(bad) if bad else "none") + ". The band is derived from the sparsity clause "
                     "alone; the overlap clause is not a function of the reference median, so it has to be "
                     "checked per config.")

    wall = time.time() - t0
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    res = dict(
        meta=dict(npz=NPZ, smoke=smoke, commit=commit, dirty=dirty, strength=STRENGTH, read_ms=READ_MS,
                  settles_ms=list(SETTLES), gate_settle_ms=GATE_SETTLE_MS, design_seeds=list(seeds),
                  design_k=DESIGN_K, design_odor_seed=DESIGN_ODOR_SEED, d4_band=list(D4_BAND),
                  d4_margin_pp=D4_MARGIN_PP, workers=workers, wall_s=wall, reference=ref_meta),
        configs=rows, candidate=cand_row, n_overlap_fail=n_overlap_fail, n_sparsity_fail=n_sparsity_fail,
        overlap_fail_labels=fail_labels, ratio_max_range=ratio_range, band_overlap_note=band_note,
        m0c_check=m0c_check([r for r in rows if r["label"] == "C0"][0]["s200"]) if any(r["label"] == "C0" for r in rows) else None,
    )

    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/d4_margin_protocol.json", "w") as f:
        json.dump(res, f, indent=1)
    with open(f"{out_dir}/d4_margin_protocol.md", "w") as f:
        f.write(report(res, wall))
    print(report(res, wall))
    print(f"wrote {out_dir}/d4_margin_protocol.json / .md in {wall:.0f} s", flush=True)
