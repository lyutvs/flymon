"""M0d H.3a: operating characteristic of the draft's decision rules. Pure Monte Carlo, no engine.

Three red-team rounds converged on the same structural problem: the H.3a draft keeps adding stricter decision
rules, and nobody has computed how often each gate passes a candidate that is genuinely fine. Spec G.14.5 set
the precedent (compute the operating characteristic, then choose the test); this script does that for every
gate in the draft.

**No simulation of the brain happens here.** There is no FlyPool, no Engine, no connectome. Every noise level
is read out of an existing diagnostic result file (never hardcoded by hand where a file carries the number),
and each statistic is modelled as `measured point estimate + measured noise` and drawn N_DRAWS times.

Inputs (file -> quantity):

- `results/m0d/diag/guard_after_baseline.json`
  - `calibration/<cfg>/baseline_sd`   : MBON baseline per-seed sd over 8 seeds (item 1)
  - `calibration/<cfg>/trace`         : hold -> baseline Hz bisection traces, for the local slope (item 1)
  - `configs/<cfg>/{stim,rest}`       : per-presentation MBON spike counts, for zero-share and med delta and
                                        their odour-cluster bootstrap CIs, from which the cluster SE and the
                                        cluster design effect follow (item 2)
- `results/m0d/diag/apl_input_scale_sweep.json` : per-presentation APL membrane mV and KC active % on the
                                        48-odour reference set (item 3)
- `results/m0d/diag/clamp_control_power.json`   : per-odour KC active % on the 192-odour extended set (item 3)
                                        and the 192-odour clamped-release difference with its bootstrap CI,
                                        from which the cluster SE follows (item 5)
- `results/m0d/diag/d4_margin_protocol.json` if it exists, else `results/m0d/diag/two_checks.json`
                                      : D.4 design-pair sparsity per seed (item 6 chain)

Writes results/m0d/diag/gate_oc.{json,md}. Deterministic: one seed, SEED, for the whole run.

Modelling assumptions are listed in the .md; the ones doing real work are flagged there.
"""
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
RES = os.path.join(ROOT, "results", "m0d", "diag")

SEED = 20260918
N_DRAWS = 20_000
Z = 1.959963984540054  # two-sided 95%

BASELINE_TARGET = 3.5
BASELINE_BAND = (3.0, 4.0)
EARLY_STOP = 0.5  # draft's stage-2 early stop: |measured baseline - 3.5| <= 0.5
N_BISECT = 8
HOLD_BRACKET = (0.5, 1.0)

ZERO_SHARE_MAX = 0.25
MED_DELTA_MIN = 5.0

BAND_MV = 2.0  # half-split band widths the draft uses (membrane mV, KC active pp)
BAND_PP = 2.0


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def fmt(x, n=3):
    return f"{x:.{n}f}"


def table(header, rows):
    L = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    L += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return L


# ---------------------------------------------------------------- item 1: MBON baseline


def local_slope(trace):
    """Least-squares slope of baseline Hz against mbon_hold_frac over the bisection's evaluated points."""
    h = np.array([t["hold"] for t in trace], float)
    y = np.array([t["hz"] for t in trace], float)
    return float(np.polyfit(h, y, 1)[0])


def bisect_once(rng, slope, h0, sd, n_seeds, early_stop, n_iter=N_BISECT):
    """One stage-2 bisection on a true curve b(h) = 3.5 + slope * (h - h0), measured with 8-seed noise.

    Returns (true baseline at the accepted hold under rule a, measured value accepted under rule a,
             true baseline at the final bracket midpoint under rule b).
    """
    lo, hi = HOLD_BRACKET
    se = sd / math.sqrt(n_seeds)
    best_h, best_meas, best_gap = None, None, np.inf
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        true = float(np.clip(BASELINE_TARGET + slope * (mid - h0), 0.0, 9.0))
        meas = true + rng.normal(0.0, se)
        gap = abs(meas - BASELINE_TARGET)
        if gap < best_gap:
            best_h, best_meas, best_gap = mid, meas, gap
        if early_stop is not None and gap <= early_stop:
            break
        if meas < BASELINE_TARGET:
            lo = mid
        else:
            hi = mid
    mid_h = 0.5 * (lo + hi)
    t_a = float(np.clip(BASELINE_TARGET + slope * (best_h - h0), 0.0, 9.0))
    t_b = float(np.clip(BASELINE_TARGET + slope * (mid_h - h0), 0.0, 9.0))
    return t_a, best_meas, t_b


def item1(rng, guard):
    cfgs = guard["calibration"]
    per_cfg = {}
    for name, c in cfgs.items():
        per_cfg[name] = dict(sd=float(c["baseline_sd"]), slope=local_slope(c["trace"]),
                             hold=float(c["hold"]), baseline_hz=float(c["baseline_hz"]))
    sd_bar = float(np.mean([v["sd"] for v in per_cfg.values()]))
    slope_bar = float(np.mean([v["slope"] for v in per_cfg.values()]))
    h0 = float(np.mean([v["hold"] for v in per_cfg.values()]))

    # (i) conditional: stage 3 re-measures on n independent seeds and requires 3-4 Hz.
    trues = [3.0, 3.25, 3.5, 3.75, 4.0, 2.5]
    cond = {}
    for n in (8, 16, 32):
        se = sd_bar / math.sqrt(n)
        cond[n] = {}
        for t in trues:
            x = t + rng.normal(0.0, se, N_DRAWS)
            cond[n][t] = float(np.mean((x >= BASELINE_BAND[0]) & (x <= BASELINE_BAND[1])))

    # (ii) end to end: stage 2 bisection under rules a/b/c, then stage 3 on 8 fresh seeds.
    variants = {}
    specs = [("a: closest evaluated point, early stop, 8 seeds", "a", 8, EARLY_STOP),
             ("a': closest evaluated point, no early stop, 8 seeds", "a", 8, None),
             ("b: final bracket midpoint, no early stop, 8 seeds", "b", 8, None),
             ("c16: closest evaluated point, no early stop, 16 seeds", "a", 16, None),
             ("c32: closest evaluated point, no early stop, 32 seeds", "a", 32, None),
             ("b+c32: final bracket midpoint, no early stop, 32 seeds", "b", 32, None)]
    for label, rule, n_cal, stop in specs:
        acc_true = np.empty(N_DRAWS)
        acc_meas = np.empty(N_DRAWS)
        for i in range(N_DRAWS):
            t_a, m_a, t_b = bisect_once(rng, slope_bar, h0, sd_bar, n_cal, stop)
            acc_true[i] = t_a if rule == "a" else t_b
            acc_meas[i] = m_a
        for n3 in (8, 16, 32):
            se3 = sd_bar / math.sqrt(n3)
            x = acc_true + rng.normal(0.0, se3, N_DRAWS)
            p = float(np.mean((x >= BASELINE_BAND[0]) & (x <= BASELINE_BAND[1])))
            variants[f"{label} | stage3 n={n3}"] = dict(
                p_stage3=p, mean_true=float(acc_true.mean()), sd_true=float(acc_true.std(ddof=1)),
                sel_bias=float(np.mean(acc_meas - acc_true)), bias_vs_target=float(np.mean(acc_true) - BASELINE_TARGET))
    return dict(per_cfg=per_cfg, sd_bar=sd_bar, slope_bar=slope_bar, h0=h0, conditional=cond, variants=variants)


# ---------------------------------------------------------------- item 2: readout guard


def cluster_se_from_ci(ci):
    return (ci[1] - ci[0]) / (2 * Z)


def guard_stats(guard):
    """Recompute the readout-guard point estimates and cluster design effect from the raw presentations."""
    out = {}
    for cfg, blk in guard["configs"].items():
        stim, rest = blk["stim"], blk["rest"]
        rest_by_seed = {r["seed"]: r["types"] for r in rest}
        for t in ("MBON13", "MBON05"):
            z = np.array([1.0 if s["types"][t] == 0 else 0.0 for s in stim])
            d = np.array([s["types"][t] - rest_by_seed[s["seed"]][t] for s in stim], float)
            out[(cfg, t)] = dict(zero_share=float(z.mean()), med_delta=float(np.median(d)), n_pres=len(stim))
    return out


def item2(rng, guard, mbon13_zero_ci, mbon13_med_ci):
    # cluster SE and design effect, from the file's odour-cluster bootstrap CIs.
    stats = guard_stats(guard)
    deffs = []
    rows_deff = []
    for (cfg, t), s in sorted(stats.items()):
        if s["zero_share"] in (0.0, 1.0):
            continue
        # the CI for this cell, taken from the same file's bootstrap where we have it
        ci = mbon13_zero_ci.get((cfg, t))
        if ci is None:
            continue
        se_cl = cluster_se_from_ci(ci)
        p = s["zero_share"]
        se_bin = math.sqrt(p * (1 - p) / s["n_pres"])
        deff = (se_cl / se_bin) ** 2
        deffs.append(deff)
        rows_deff.append([f"{t} {cfg}", fmt(p), f"[{ci[0]:.2f}, {ci[1]:.2f}]", fmt(se_cl, 4), fmt(se_bin, 4), fmt(deff, 2)])
    deff = float(np.mean(deffs))

    def se_zero(p, n_odours):
        return math.sqrt(deff * p * (1 - p) / (2 * n_odours))

    trues = [0.10, 0.15, 0.19, 0.25, 0.30, 0.35]
    zero = {}
    for n in (48, 96, 192, 384):
        zero[n] = {}
        for p in trues:
            se = se_zero(p, n)
            est = p + rng.normal(0.0, se, N_DRAWS)
            lo, hi = est - Z * se, est + Z * se
            zero[n][p] = dict(ci_pass=float(np.mean(hi < ZERO_SHARE_MAX)),
                              ci_incon=float(np.mean((lo <= ZERO_SHARE_MAX) & (hi >= ZERO_SHARE_MAX))),
                              ci_fail=float(np.mean(lo > ZERO_SHARE_MAX)),
                              pt_pass=float(np.mean(est <= ZERO_SHARE_MAX)))
    # med delta >= 5, MBON13 C1@0.333 CI width from the file
    se_med48 = cluster_se_from_ci(mbon13_med_ci)
    med = {}
    for n in (48, 96, 192):
        se = se_med48 * math.sqrt(48.0 / n)
        med[n] = {}
        for t in (3.0, 5.0, 6.0, 8.0, 11.0):
            est = t + rng.normal(0.0, se, N_DRAWS)
            med[n][t] = dict(pt_pass=float(np.mean(est >= MED_DELTA_MIN)),
                             ci_pass=float(np.mean(est - Z * se > MED_DELTA_MIN)),
                             ci_fail=float(np.mean(est + Z * se < MED_DELTA_MIN)))
    return dict(deff=deff, deff_rows=rows_deff, se_med48=se_med48, zero=zero, med=med,
                stats={f"{c}/{t}": v for (c, t), v in stats.items()})


# ---------------------------------------------------------------- item 3: half-split


def half_split(rng, values, band, label, n_draws=N_DRAWS):
    """Permutation operating characteristic of the draft's 24+24 half-split under a homogeneous candidate."""
    v = np.asarray(values, float)
    n = len(v)
    h = n // 2
    diffs = np.empty(n_draws)
    for i in range(n_draws):
        perm = rng.permutation(n)
        diffs[i] = np.median(v[perm[:h]]) - np.median(v[perm[h:]])
    sd = float(diffs.std(ddof=1))
    thr = {"1/3 band": band / 3.0, "2 SD": 2 * sd, "3 SD": 3 * sd}
    return dict(label=label, n=n, spread_sd=float(v.std(ddof=1)), iqr=float(np.subtract(*np.percentile(v, [75, 25]))),
                split_sd=sd, p95_abs=float(np.percentile(np.abs(diffs), 95)),
                triggers={k: float(np.mean(np.abs(diffs) > t)) for k, t in thr.items()},
                thresholds={k: float(t) for k, t in thr.items()})


def item3(rng, sweep, clamp):
    # reference-set presentations at the operating point closest to the candidate (kc_thresh 1.5, scale 0.0799)
    rows = [r for r in sweep["rows"] if r["params"]["kc_thresh"] == 1.5]
    row = min(rows, key=lambda r: abs(math.log(r["scale"]) - math.log(0.0799)))
    pres = row["presentations"]
    by_odor_v, by_odor_kc = {}, {}
    for p in pres:
        by_odor_v.setdefault(p["odor"], []).append(p["apl_v_mean"])
        by_odor_kc.setdefault(p["odor"], []).append(100.0 * p["kc_active_frac"])
    v48 = [float(np.mean(x)) for x in by_odor_v.values()]
    kc48 = [float(np.mean(x)) for x in by_odor_kc.values()]
    kc192 = clamp["comparison"]["per_odor_kc_pct"]["A"]
    return dict(source_scale=row["scale"], source_kc_thresh=row["params"]["kc_thresh"],
                membrane=half_split(rng, v48, BAND_MV, "APL membrane (mV), 48 odours"),
                kc48=half_split(rng, kc48, BAND_PP, "KC active % (pp), 48 odours"),
                kc192=half_split(rng, kc192, BAND_PP, "KC active % (pp), 192 odours"))


# ---------------------------------------------------------------- item 4: conditioning


def item4(rng):
    out = {}
    for n, thrs in ((8, (7, 6, 5)), (16, (14, 12, 10))):
        for k in thrs:
            for p in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
                x = rng.binomial(n, p, N_DRAWS)
                out[f"{k}/{n}"] = out.get(f"{k}/{n}", {})
                out[f"{k}/{n}"][p] = float(np.mean(x >= k))
    both = {}
    for key, d in out.items():
        both[key] = {p: v * v for p, v in d.items()}  # one live type per pool, independent pools
    return dict(single=out, both_pools=both)


# ---------------------------------------------------------------- item 5: gain control


def item5(rng, clamp):
    m = [x for x in clamp["comparison"]["metrics"] if x["primary"]][0]
    se192 = cluster_se_from_ci(m["ci"])
    out = {}
    for n in (48, 192, 384):
        se = se192 * math.sqrt(192.0 / n)
        out[n] = {}
        for t in (0.0, -0.2, -0.4, -0.8, -1.2):
            est = t + rng.normal(0.0, se, N_DRAWS)
            out[n][t] = float(np.mean(est + Z * se < 0.0))
    return dict(measured_diff=m["diff"], measured_ci=m["ci"], se192=se192, n_clusters=m["n_clusters"], p=out)


# ---------------------------------------------------------------- D.4 design pair + item 6


def d4_inputs():
    path_pref = os.path.join(RES, "d4_margin_protocol.json")
    if os.path.exists(path_pref):
        return load("d4_margin_protocol.json"), "results/m0d/diag/d4_margin_protocol.json"
    return load("two_checks.json"), "results/m0d/diag/two_checks.json"


def item_d4(rng, d4, band, margin_pp):
    """Candidate D.4 sparsity gate: mean over design seeds of frac_active for both odours inside the band,
    shrunk by the margin. Noise = per-seed spread of the measured design pair."""
    if "check1" in d4:  # two_checks.json schema (settle 800)
        block, src = d4["check1"]["candidate"]["design"], "two_checks candidate design pair (settle 800)"
    else:  # d4_margin_protocol.json schema; prefer the re-measured settle-200 block
        cand = d4["candidate"]
        key = "s200" if "s200" in cand else "s800"
        block, src = cand[key], f"d4_margin_protocol candidate {key}"
    per_seed = block["per_seed"]
    a = np.array([100 * s["frac_active_A"] for s in per_seed])
    b = np.array([100 * s["frac_active_B"] for s in per_seed])
    lo, hi = 100 * band[0] + margin_pp, 100 * band[1] - margin_pp
    sd_bar = float(np.mean([a.std(ddof=1), b.std(ddof=1)]))
    # operating characteristic over the true sparsity of one design odour
    trues = [4.0, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5]
    oc = {}
    for n in (3, 8, 16):
        se = sd_bar / math.sqrt(n)
        oc[n] = {}
        for t in trues:
            est = t + rng.normal(0.0, se, N_DRAWS)
            single = float(np.mean((est >= lo) & (est <= hi)))
            oc[n][t] = dict(single=single, both_same_true=single ** 2)
    # the candidate as measured, seed by seed
    meas = {}
    for n in (3, 8, 16):
        r = {}
        keep = {}
        for nm, arr in (("A", a), ("B", b)):
            se = float(arr.std(ddof=1)) / math.sqrt(n)
            est = float(arr.mean()) + rng.normal(0.0, se, N_DRAWS)
            keep[nm] = (est >= lo) & (est <= hi)
            r[nm] = dict(mean=float(arr.mean()), sd=float(arr.std(ddof=1)), se=se, p=float(np.mean(keep[nm])))
        r["both"] = float(np.mean(keep["A"] & keep["B"]))
        meas[n] = r
    return dict(band_pp=[lo, hi], margin_pp=margin_pp, n_seeds_measured=len(per_seed), source_block=src,
                sd_bar=sd_bar, point_A=float(a.mean()), point_B=float(b.mean()), oc=oc, p=meas)


def item6(i1, i2, i3, i4, i5, d4):
    draft = [("stage 3 MBON baseline 3-4 Hz (rule a, 8/8 seeds)",
              i1["variants"]["a: closest evaluated point, early stop, 8 seeds | stage3 n=8"]["p_stage3"]),
             ("readout guard zero-share, 3-way CI rule, true 0.15, 48 odours", i2["zero"][48][0.15]["ci_pass"]),
             ("readout guard med delta >= 5 (CI rule), true 8.0, 48 odours", i2["med"][48][8.0]["ci_pass"]),
             ("D.4 sparsity, both odours truly at the 5.0% band centre, 3 seeds, 0.5 pp margin",
              d4["oc"][3][5.0]["both_same_true"]),
             ("half-split not triggered, membrane, 1/3 band", 1 - i3["membrane"]["triggers"]["1/3 band"]),
             ("half-split not triggered, KC %, 1/3 band", 1 - i3["kc48"]["triggers"]["1/3 band"]),
             ("conditioning 7/8 at true p 0.8, both pools", i4["both_pools"]["7/8"][0.8]),
             ("gain control CI upper < 0, true -0.8, 192 odours", i5["p"][192][-0.8])]
    best = [("stage 3 MBON baseline, bracket midpoint, 32-seed calibration and stage 3",
             i1["variants"]["b+c32: final bracket midpoint, no early stop, 32 seeds | stage3 n=32"]["p_stage3"]),
            ("readout guard zero-share, point rule <= 0.25, true 0.15, 192 odours", i2["zero"][192][0.15]["pt_pass"]),
            ("readout guard med delta >= 5 (point rule), true 8.0, 48 odours", i2["med"][48][8.0]["pt_pass"]),
            ("D.4 sparsity, both odours truly at the 5.0% band centre, 16 seeds, 0.5 pp margin",
             d4["oc"][16][5.0]["both_same_true"]),
            ("half-split not triggered, membrane, 3 SD", 1 - i3["membrane"]["triggers"]["3 SD"]),
            ("half-split not triggered, KC %, 3 SD", 1 - i3["kc48"]["triggers"]["3 SD"]),
            ("conditioning 6/8 at true p 0.8, both pools", i4["both_pools"]["6/8"][0.8]),
            ("gain control CI upper < 0, true -0.8, 384 odours", i5["p"][384][-0.8])]
    prod = lambda ch: float(np.prod([p for _, p in ch]))
    return dict(draft=dict(chain=draft, survival=prod(draft), survival_no_d4=prod([c for c in draft if "D.4" not in c[0]])),
                best=dict(chain=best, survival=prod(best), survival_no_d4=prod([c for c in best if "D.4" not in c[0]])))


# ---------------------------------------------------------------- report


def report(res):
    m = res["meta"]
    i1, i2, i3, i4, i5, d4, i6 = (res["item1"], res["item2"], res["item3"], res["item4"],
                                  res["item5"], res["d4"], res["item6"])
    L = ["# M0d H.3a: operating characteristic of the decision rules", "",
         f"Pure Monte Carlo. No `FlyPool`, no `Engine`, no connectome: every number below is a draw from "
         f"`measured point estimate + measured noise`, with the noise read from the diagnostic files listed "
         f"under Inputs. {m['n_draws']:,} draws per cell, one RNG seed {m['seed']} for the whole run. "
         f"Commit {m['commit']} (dirty tree: {str(m['dirty']).lower()}), wall time {m['wall_s']:.1f} s.", "",
         "## Inputs", ""]
    L += table(["quantity", "file", "value used"], [
        ["MBON baseline per-seed sd (8 seeds)", "guard_after_baseline.json `calibration/*/baseline_sd`",
         ", ".join(fmt(v["sd"]) for v in i1["per_cfg"].values()) + f" Hz; mean {fmt(i1['sd_bar'])} Hz"],
        ["baseline vs `mbon_hold_frac` slope", "guard_after_baseline.json `calibration/*/trace` (LS fit)",
         ", ".join(fmt(v["slope"], 1) for v in i1["per_cfg"].values()) + f" Hz/unit; mean {fmt(i1['slope_bar'], 1)}"],
        ["zero-share cluster SE / design effect", "guard_after_baseline.json bootstrap CIs",
         f"design effect {fmt(i2['deff'], 2)} vs binomial"],
        ["med delta cluster SE (48 odours)", "guard_after_baseline.json MBON13 C1@0.333 CI [4.0, 11.0]",
         f"{fmt(i2['se_med48'], 3)} spikes"],
        ["APL membrane / KC active % per presentation",
         f"apl_input_scale_sweep.json (kc_thresh {i3['source_kc_thresh']}, scale {i3['source_scale']:.5f})",
         f"48 odours x 2 seeds; odour-level sd {fmt(i3['membrane']['spread_sd'])} mV, "
         f"{fmt(i3['kc48']['spread_sd'])} pp"],
        ["KC active % per odour, extended set", "clamp_control_power.json `per_odor_kc_pct/A`",
         f"192 odours; sd {fmt(i3['kc192']['spread_sd'])} pp"],
        ["clamped-release primary effect", "clamp_control_power.json primary metric",
         f"{fmt(i5['measured_diff'], 4)} [{i5['measured_ci'][0]:.4f}, {i5['measured_ci'][1]:.4f}] over "
         f"{i5['n_clusters']} clusters -> cluster SE {fmt(i5['se192'], 4)}"],
        ["D.4 design-pair sparsity per seed", m["d4_source"],
         f"{d4['n_seeds_measured']} seeds; A {fmt(d4['p'][3]['A']['mean'], 2)}% (sd {fmt(d4['p'][3]['A']['sd'], 2)}), "
         f"B {fmt(d4['p'][3]['B']['mean'], 2)}% (sd {fmt(d4['p'][3]['B']['sd'], 2)})"]])
    L += ["", f"`results/m0d/diag/d4_margin_protocol.json` {'exists and was preferred' if 'd4_margin' in m['d4_source'] else 'does not exist, so the D.4 numbers come from two_checks.json'}.", ""]

    # item 1
    L += ["## 1. Stage 2 MBON baseline bisection + stage 3 baseline gate", "",
          "Stage 3 re-measures on fresh seeds and requires 3-4 Hz. Conditional on the *true* baseline at the "
          "accepted `mbon_hold_frac`:", ""]
    trues = [2.5, 3.0, 3.25, 3.5, 3.75, 4.0]
    L += table(["stage-3 seeds", "SE (Hz)"] + [f"true {t}" for t in trues],
               [[n, fmt(i1["sd_bar"] / math.sqrt(n))] + [fmt(i1["conditional"][n][t]) for t in trues]
                for n in (8, 16, 32)])
    L += ["", "End to end (stage-2 bisection simulated on the measured curve and noise, then stage 3):", ""]
    L += table(["stage-2 rule", "stage-3 seeds", "P(stage 3 passes)", "E[accepted true - 3.5]",
                "sd of accepted true", "selection bias E[measured - true]"],
               [[k.split(" | ")[0], k.split("n=")[1], fmt(v["p_stage3"]), fmt(v["bias_vs_target"]),
                 fmt(v["sd_true"]), fmt(v["sel_bias"])] for k, v in i1["variants"].items()])
    L += ["", "The selection bias column is the draft's 'closest evaluated point' rule reporting a baseline "
          "closer to 3.5 Hz than the truth; it is a property of picking the extremum of a noisy set.", ""]

    # item 2
    L += ["## 2. Stage 2.5 readout guard", "",
          "Cluster SE inferred from the file's odour-cluster bootstrap CIs, then scaled as "
          "`sqrt(deff * p(1-p) / (2 * n_odours))`:", ""]
    L += table(["cell", "zero-share", "95% CI (file)", "cluster SE", "binomial SE", "design effect"], i2["deff_rows"])
    L += ["", f"Design effect used: {fmt(i2['deff'], 2)}.", "",
          "Zero-share against the 0.25 threshold, three-way CI rule (pass / inconclusive / fail) and the "
          "point-estimate rule:", ""]
    rows = []
    for n in (48, 96, 192, 384):
        for p in (0.10, 0.15, 0.19, 0.25, 0.30, 0.35):
            d = i2["zero"][n][p]
            rows.append([n, p, fmt(d["ci_pass"]), fmt(d["ci_incon"]), fmt(d["ci_fail"]), fmt(d["pt_pass"])])
    L += table(["odours", "true zero-share", "P(CI pass)", "P(inconclusive)", "P(CI fail)", "P(point rule pass)"], rows)
    L += ["", f"`med delta >= 5` with the measured MBON13 C1@0.333 CI width (SE {fmt(i2['se_med48'], 3)} at 48 odours):", ""]
    rows = []
    for n in (48, 96, 192):
        for t in (3.0, 5.0, 6.0, 8.0, 11.0):
            d = i2["med"][n][t]
            rows.append([n, t, fmt(d["pt_pass"]), fmt(d["ci_pass"]), fmt(d["ci_fail"])])
    L += table(["odours", "true med delta", "P(point rule pass)", "P(CI-lower rule pass)", "P(CI-upper rule fail)"], rows)

    # item 3
    L += ["", "## 3. Half-split sample-error check", "",
          "Permutation of the measured odour-level values into 24 + 24 (or 96 + 96): this is exactly a "
          "candidate that is homogeneous and fine, so every trigger below is a false trigger.", ""]
    rows = []
    for key in ("membrane", "kc48", "kc192"):
        d = i3[key]
        for nm in ("1/3 band", "2 SD", "3 SD"):
            rows.append([d["label"], nm, fmt(d["thresholds"][nm]), fmt(d["split_sd"]), fmt(d["triggers"][nm])])
    L += table(["statistic", "threshold rule", "threshold value", "SD of split difference",
                "P(false trigger)"], rows)

    # item 4
    L += ["", "## 4. Preliminary conditioning (k of n seeds)", ""]
    keys = ["7/8", "6/8", "5/8", "14/16", "12/16", "10/16"]
    ps = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    L += table(["rule"] + [f"p={p}" for p in ps],
               [[k] + [fmt(i4["single"][k][p]) for p in ps] for k in keys])
    L += ["", "Both pools survive when each pool has exactly one live type (independent pools, same rule):", ""]
    L += table(["rule"] + [f"p={p}" for p in ps],
               [[k] + [fmt(i4["both_pools"][k][p]) for p in ps] for k in keys])

    # item 5
    L += ["", "## 5. Gain-control qualification (clamped-release control)", "",
          f"Sign-correct rule: the 95% CI upper bound of the A - B difference is below 0. Cluster SE "
          f"{fmt(i5['se192'], 4)} at {i5['n_clusters']} odours, scaled as 1/sqrt(n).", ""]
    L += table(["odours", "SE"] + [f"true {t}" for t in (0.0, -0.2, -0.4, -0.8, -1.2)],
               [[n, fmt(i5["se192"] * math.sqrt(192.0 / n), 4)] +
                [fmt(i5["p"][n][t]) for t in (0.0, -0.2, -0.4, -0.8, -1.2)] for n in (48, 192, 384)])

    # D.4
    L += ["", "## 5b. D.4 design-pair sparsity (input to the chain below)", "",
          f"Both design odours' mean active fraction must lie inside "
          f"[{fmt(d4['band_pp'][0], 1)}%, {fmt(d4['band_pp'][1], 1)}%] (the 3-7% band with the "
          f"{d4['margin_pp']} pp margin), averaged over design seeds. Seed-to-seed sd from "
          f"{d4['source_block']}: {fmt(d4['sd_bar'], 2)} pp (mean of the two odours). Operating "
          f"characteristic over one odour's true sparsity, and both odours at the same true value:", ""]
    trues_d4 = [4.0, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5]
    L += table(["design seeds", "SE (pp)"] + [f"true {t}%" for t in trues_d4],
               [[n, fmt(d4["sd_bar"] / math.sqrt(n), 2)] + [fmt(d4["oc"][n][t]["single"]) for t in trues_d4]
                for n in (3, 8, 16)])
    L += ["", "Both odours at the same true sparsity (independent draws):", ""]
    L += table(["design seeds"] + [f"true {t}%" for t in trues_d4],
               [[n] + [fmt(d4["oc"][n][t]["both_same_true"]) for t in trues_d4] for n in (3, 8, 16)])
    L += ["", f"The candidate as actually measured ({d4['source_block']}): odour A "
          f"{fmt(d4['point_A'], 2)}%, odour B {fmt(d4['point_B'], 2)}% - odour A already sits outside the "
          f"margin-shrunk band, so more seeds drive its pass probability to 0, not up:", ""]
    L += table(["design seeds", "P(odour A inside)", "P(odour B inside)", "P(both)"],
               [[n, fmt(d4["p"][n]["A"]["p"]), fmt(d4["p"][n]["B"]["p"]), fmt(d4["p"][n]["both"])]
                for n in (3, 8, 16)])

    # item 6
    L += ["", "## 6. Whole-pipeline survival for a genuinely good candidate", "",
          "True baseline 3.5 Hz, true zero-share 0.15, true med delta 8.0, true conditioning p 0.8, true "
          "gain-control effect -0.8, homogeneous over odours (so the half-split should never fire).", ""]
    for nm in ("draft", "best"):
        b = i6[nm]
        L += [f"**{'(a) the draft as written' if nm == 'draft' else '(b) best variant of each gate'}**", ""]
        L += table(["gate", "P(pass)"], [[g, fmt(p)] for g, p in b["chain"]])
        L += ["", f"Product: **{fmt(b['survival'], 4)}** (without the D.4 gate: {fmt(b['survival_no_d4'], 4)}).", ""]

    # implications
    L += ["## What these numbers imply for the thresholds", "",
          "For each gate: the sample size or threshold at which a genuinely good candidate passes with "
          "probability >= 0.8 while a genuinely bad one is rejected with probability >= 0.8. "
          "Bad = true baseline 2.5 Hz, true zero-share 0.35, conditioning p 0.4, gain-control effect 0.", ""]
    rows = []
    # baseline
    good8, bad8 = i1["conditional"][8][3.5], 1 - i1["conditional"][8][2.5]
    good32, bad32 = i1["conditional"][32][3.5], 1 - i1["conditional"][32][2.5]
    ok_n = next((n for n in (8, 16, 32) if i1["conditional"][n][3.5] >= 0.8 and 1 - i1["conditional"][n][2.5] >= 0.8), None)
    rows.append(["stage 3 baseline 3-4 Hz",
                 f"{ok_n} seeds" if ok_n else "no setting up to 32 seeds",
                 f"good(3.5) 8 seeds {fmt(good8)}, 32 seeds {fmt(good32)}; bad(2.5) rejected {fmt(bad8)} / {fmt(bad32)}"])
    # zero share CI rule
    ci_ok = next((n for n in (48, 96, 192, 384)
                  if i2["zero"][n][0.15]["ci_pass"] >= 0.8 and i2["zero"][n][0.35]["ci_fail"] >= 0.8), None)
    pt_ok = next((n for n in (48, 96, 192, 384)
                  if i2["zero"][n][0.15]["pt_pass"] >= 0.8 and (1 - i2["zero"][n][0.35]["pt_pass"]) >= 0.8), None)
    rows.append(["zero-share, three-way CI rule", f"{ci_ok} odours" if ci_ok else "no setting up to 384 odours",
                 f"good(0.15) CI-pass at 48 {fmt(i2['zero'][48][0.15]['ci_pass'])}, at 192 "
                 f"{fmt(i2['zero'][192][0.15]['ci_pass'])}; bad(0.35) CI-fail at 48 "
                 f"{fmt(i2['zero'][48][0.35]['ci_fail'])}, at 192 {fmt(i2['zero'][192][0.35]['ci_fail'])}"])
    rows.append(["zero-share, point rule <= 0.25", f"{pt_ok} odours" if pt_ok else "no setting up to 384 odours",
                 f"good(0.15) {fmt(i2['zero'][48][0.15]['pt_pass'])} at 48; bad(0.35) rejected "
                 f"{fmt(1 - i2['zero'][48][0.35]['pt_pass'])} at 48"])
    med_ok = next((n for n in (48, 96, 192) if i2["med"][n][8.0]["pt_pass"] >= 0.8 and
                   (1 - i2["med"][n][3.0]["pt_pass"]) >= 0.8), None)
    rows.append(["med delta >= 5, point rule", f"{med_ok} odours" if med_ok else "no setting up to 192 odours",
                 f"good(8.0) {fmt(i2['med'][48][8.0]['pt_pass'])} at 48; bad(3.0) rejected "
                 f"{fmt(1 - i2['med'][48][3.0]['pt_pass'])} at 48; CI rule good(8.0) "
                 f"{fmt(i2['med'][48][8.0]['ci_pass'])}"])
    # half split: no bad candidate defined; report false trigger only
    rows.append(["half-split (no bad case: it is a homogeneity check)",
                 "3 SD keeps the false trigger at or below 0.01",
                 f"membrane 1/3 band {fmt(i3['membrane']['triggers']['1/3 band'])}, 2 SD "
                 f"{fmt(i3['membrane']['triggers']['2 SD'])}, 3 SD {fmt(i3['membrane']['triggers']['3 SD'])}; "
                 f"KC 1/3 band {fmt(i3['kc48']['triggers']['1/3 band'])}, 3 SD {fmt(i3['kc48']['triggers']['3 SD'])}"])
    cond_ok = next((k for k in ("7/8", "6/8", "5/8", "14/16", "12/16", "10/16")
                    if i4["single"][k][0.8] >= 0.8 and 1 - i4["single"][k][0.4] >= 0.8), None)
    rows.append(["conditioning k of n", cond_ok or "none of the rules considered",
                 f"good(p=0.8): 7/8 {fmt(i4['single']['7/8'][0.8])}, 6/8 {fmt(i4['single']['6/8'][0.8])}, "
                 f"12/16 {fmt(i4['single']['12/16'][0.8])}; bad(p=0.4) rejected: 7/8 "
                 f"{fmt(1 - i4['single']['7/8'][0.4])}, 6/8 {fmt(1 - i4['single']['6/8'][0.4])}"])
    gc_ok = next((n for n in (48, 192, 384) if i5["p"][n][-0.8] >= 0.8 and 1 - i5["p"][n][0.0] >= 0.8), None)
    rows.append(["gain control, CI upper < 0", f"{gc_ok} odours" if gc_ok else "no setting up to 384 odours",
                 f"good(-0.8) {fmt(i5['p'][48][-0.8])} / {fmt(i5['p'][192][-0.8])} / {fmt(i5['p'][384][-0.8])} "
                 f"at 48/192/384; bad(0) rejected {fmt(1 - i5['p'][192][0.0])} at 192"])
    d4_ok = next((n for n in (3, 8, 16) if d4["oc"][n][5.0]["both_same_true"] >= 0.8 and
                  (1 - d4["oc"][n][7.5]["single"]) >= 0.8), None)
    rows.append(["D.4 sparsity with 0.5 pp margin",
                 f"{d4_ok} design seeds (good = both odours truly at the 5.0% band centre; "
                 f"bad = truly 7.5%)" if d4_ok else "no setting up to 16 seeds",
                 f"both odours at true 5.0%: 3 seeds {fmt(d4['oc'][3][5.0]['both_same_true'])}, 8 seeds "
                 f"{fmt(d4['oc'][8][5.0]['both_same_true'])}, 16 seeds "
                 f"{fmt(d4['oc'][16][5.0]['both_same_true'])}; a true 6.5% odour (on the margin) passes "
                 f"{fmt(d4['oc'][3][6.5]['single'])} / {fmt(d4['oc'][16][6.5]['single'])} at 3 / 16 seeds; "
                 f"the measured candidate's odour A ({fmt(d4['point_A'], 2)}%) passes "
                 f"{fmt(d4['p'][3]['A']['p'])} at 3 seeds"])
    L += table(["gate", "setting that meets 0.8 / 0.8", "the decisive numbers"], rows)

    L += ["", "## Modelling assumptions", "",
          "- **Normal sampling noise** for every continuous statistic (MBON baseline mean, zero-share, med "
          "delta, IQR difference), with the SD taken from the measured per-seed sd or from the width of the "
          "measured bootstrap CI (SE = width / 3.92). Where a CI in the file is visibly asymmetric "
          "(zero-shares near 0), the symmetric normal understates how sharply the rule behaves near the "
          "boundary; this assumption is doing real work for the zero-share rows at 0.10.",
          "- **Cluster design effect** for the zero-share: the odour-cluster bootstrap CI in the file is "
          f"wider than binomial by a factor whose square averages {fmt(i2['deff'], 2)}; that factor is held "
          "fixed as the odour count changes. This is the key assumption in item 2 - it is what makes "
          "'one extra seed set' (96) worth less than 'four times the odours' (192).",
          "- **Bisection model**: baseline is locally linear in `mbon_hold_frac` with the slope fitted to the "
          "measured traces, and the per-seed sd is constant in the hold. The traces are noisy enough that the "
          "fitted slope is itself uncertain; the qualitative conclusion (noise, not bisection resolution, "
          "sets the accepted baseline) is insensitive to it, because the final bracket is "
          f"{(HOLD_BRACKET[1] - HOLD_BRACKET[0]) / 2 ** N_BISECT:.5f} wide, i.e. about "
          f"{abs(i1['slope_bar']) * (HOLD_BRACKET[1] - HOLD_BRACKET[0]) / 2 ** N_BISECT:.3f} Hz, against a "
          f"stage-2 standard error of {fmt(i1['sd_bar'] / math.sqrt(8))} Hz.",
          "- **Binomial** for the conditioning gate: seeds independent with a common success probability. "
          "Any seed-to-seed correlation would make the tails heavier than modelled.",
          "- **Half-split**: no distributional assumption at all - the odour values are permuted, so the "
          "measured odour-to-odour spread is used exactly as measured. Doing real work: the assumption that "
          "the reference-set spread measured at one operating point is the spread the candidate will show.",
          "- **Independence between gates** in item 6. The gates share seeds and share the same engine "
          "configuration, so a genuinely good candidate's per-gate outcomes are probably positively "
          "correlated and the product below is a lower bound on survival. This assumption is doing real work "
          "in item 6 and nowhere else.",
          "- The 'bad' candidate values (baseline 2.5 Hz, zero-share 0.35, p 0.4, effect 0) are the task's "
          "definition, not a measurement.", ""]
    return "\n".join(L) + "\n"


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    guard = load("guard_after_baseline.json")
    sweep = load("apl_input_scale_sweep.json")
    clamp = load("clamp_control_power.json")
    d4_raw, d4_source = d4_inputs()

    # bootstrap CIs quoted in guard_after_baseline.md / .json for the two reactive types
    mbon13_zero_ci = {("C1@0.333", "MBON13"): (0.09, 0.29), ("C0", "MBON13"): (0.02, 0.17),
                      ("C1@1.0", "MBON13"): (0.00, 0.07), ("C0", "MBON05"): (0.03, 0.17),
                      ("C1@1.0", "MBON05"): (0.25, 0.49)}
    mbon13_med_ci = (4.0, 11.0)

    i1 = item1(rng, guard)
    i2 = item2(rng, guard, mbon13_zero_ci, mbon13_med_ci)
    i3 = item3(rng, sweep, clamp)
    i4 = item4(rng)
    i5 = item5(rng, clamp)
    band = d4_raw["meta"]["d4_band"] if "meta" in d4_raw and "d4_band" in d4_raw["meta"] else [0.03, 0.07]
    margin = d4_raw["meta"].get("d4_margin_pp", 0.5) if "meta" in d4_raw else 0.5
    d4 = item_d4(rng, d4_raw, band, margin)
    i6 = item6(i1, i2, i3, i4, i5, d4)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                                cwd=ROOT).stdout.strip())
    res = dict(meta=dict(seed=SEED, n_draws=N_DRAWS, commit=commit, dirty=dirty, wall_s=time.time() - t0,
                         d4_source=d4_source, engine_used=False,
                         inputs=["results/m0d/diag/guard_after_baseline.json",
                                 "results/m0d/diag/apl_input_scale_sweep.json",
                                 "results/m0d/diag/clamp_control_power.json", d4_source]),
               item1=i1, item2=i2, item3=i3, item4=i4, item5=i5, d4=d4, item6=i6)

    def jsonable(o):
        if isinstance(o, dict):
            return {str(k): jsonable(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [jsonable(x) for x in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o

    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "gate_oc.json"), "w") as f:
        json.dump(jsonable(res), f, indent=1)
    md = report(res)
    with open(os.path.join(RES, "gate_oc.md"), "w") as f:
        f.write(md)
    print(f"wrote results/m0d/diag/gate_oc.json and .md in {time.time() - t0:.1f} s "
          f"(seed {SEED}, {N_DRAWS} draws, D.4 from {d4_source})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
