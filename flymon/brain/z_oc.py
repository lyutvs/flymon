"""Z's order-0 computations on Y's code (Z.7 0b, 0c′). Y's functions (y_oc, w_oc, x_oc) are imported and called with
y_spec.SPEC or dataclasses.replace of it; nothing here assigns to a Y / X / W module or its SPEC (Z.9.2 P1-5).
- 0b gate (Z.7 0b (i)–(iii)): load_cells reads results/y/progress/oc/boot_{0..199}.npz (arrays + meta, sha256 each);
  limits_json(y_oc.limits(cells)) must equal results/y/oc.json's `limits` / `sim` (whose sha256 block Y oc pins) and
  print the gate lines; redraw(bi) = y_oc.boot_draw on Y θ̂ under Y's root (Y's tags) must equal cell bi
  (numpy.array_equal on hits, fills with NaN equal, meta as JSON). (iii) is the runner's (Y Runner._theta_b).
- 0b record (Z.0 [사후 진단], plan Reading 6): the diagnosis scripts (scratchpad z-diag/) as functions with their own
  streams — per-draw distribution (an.py), base-only counts (baseonly.py), θ* components and the ρ table / 2 × 2 /
  R² (comp.py, an.py, an2.py), near-threshold true d′ (td.py), reachability (reach.py), point-θ̂ near d′ (pt.py),
  self-base recalibration (cfC.py), per-draw thresholds (thr.py), counterfactuals (cf.py, cfA.py, cfB.py, sub50.py),
  pilot-size emulation (psize.py), drift facts. Each returns {printed name: value} (+ detail); z_rules.diag_compare
  compares at the printed digits. Record only: nothing here gates.
- 0c′ (Z.9.2 P1-3): tile_shift(θ̂, n, δ) tiles Y θ̂'s pair-level arrays to n pairs (psize.tiled), adds δ to every
  pair's MBON13(X) N1 − pre drift, keeps Y's point Σ_pair and sets n_Σ = round(5n/6); sens_job runs boot_y →
  calibrate_y (power side, d′ 1.5) → evaluate_y per g × scenario on root 86_000_000 with streams
  SeedSequence([86_000_000, tag, n, δ index, draw, …]) (tag = Y's TAG_BOOT for the draw, TAG_BOOT_SIM for the main
  simulation, "mix" for the corner stream); a failed calibration → P(PASS) 0. sens_limits gives L(δ, n) = the 5th
  percentile over draws of each draw's min over k 4–8, worst over g × scenario."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from . import w_oc
from . import w_verdict as WV
from . import y_oc
from .h3_store import sha256_file
from .y_spec import SPEC as Y_SPEC


# ================================================================ helpers
def design_index(zs, ys=Y_SPEC) -> tuple:
    p_set, q, K, F, kr = zs.diag_design
    return (ys.p_set_grid.index(p_set), ys.q_grid.index(q), ys.k_grid.index(K), F - ys.f_min), \
        [k - ys.k_min for k in range(kr[0], kr[1] + 1)]


def nan_none(a) -> list:
    a = np.asarray(a, float)
    return np.where(np.isnan(a), None, a).tolist()


def _pct(a, q, axis=0):
    return np.percentile(a, q, axis=axis)


# ================================================================ 0b gate (i) and (ii)
def load_cells(dir_, n: int) -> tuple:
    draws, shas = [], {}
    for bi in range(n):
        p = Path(dir_) / f"boot_{bi}.npz"
        shas[str(p)] = sha256_file(p)
        with np.load(p, allow_pickle=False) as f:
            draws.append(dict(arrays={k: f[k] for k in f.files if not k.startswith("__")},
                              meta=json.loads(str(f["__meta__"]))))
    return draws, shas


def limits_json(lim: dict) -> dict:
    """y_oc.limits as Y's oc.json stores it (y_runner.stage_oc's `limits` and `sim`)."""
    return dict(limits={k: (nan_none(v) if isinstance(v, np.ndarray) else v) for k, v in lim.items()
                        if k not in ("sim", "success_only")},
                sim={f"[{a}, {b}]": {k: v.tolist() for k, v in s.items()} for (a, b), s in lim["sim"].items()})


def gate_limits(draws: list, det: dict, zs, to_json, ys=Y_SPEC) -> dict:
    kr = [tuple(k) for k in ys.k_ranges]
    lim = y_oc.limits(draws, ys, kr)
    js = to_json(limits_json(lim))
    i, ks = design_index(zs, ys)
    by_k = lim["power_lo_by_k"][i][ks]
    fk = lim["false_hi_by_k"][i][ks]
    sim = float(lim["sim"][tuple(zs.diag_design[4])]["power"][i])
    d = zs.gate_digits
    printed = (np.round(by_k, d).tolist() == [round(v, d) for v in zs.gate_by_k] and round(sim, d) == zs.gate_sim
               and float(np.round(fk, d).max()) == zs.gate_false)
    eq_l, eq_s = js["limits"] == det.get("limits"), js["sim"] == det.get("sim")
    return dict(ok=bool(eq_l and eq_s and printed), equal_limits=bool(eq_l), equal_sim=bool(eq_s),
                printed=bool(printed), by_k=by_k.tolist(), sim=sim, false_by_k=fk.tolist())


def redraw(theta, z: dict, r, bi: int) -> dict:
    """Module-level (spawn pool): Y's draw bi recomputed with today's y_oc (root and tags Y's)."""
    return y_oc.boot_draw(theta, z, Y_SPEC, r, int(bi))


def redraw_equal(res: dict, cell: dict, to_json) -> dict:
    h = np.array_equal(res["arrays"]["hits"], cell["arrays"]["hits"])
    f = np.array_equal(res["arrays"]["fills"], cell["arrays"]["fills"], equal_nan=True)
    m = to_json(res["meta"]) == cell["meta"]
    return dict(hits=bool(h), fills=bool(f), meta=bool(m), ok=bool(h and f and m))


# ================================================================ 0b record: the distribution and base-only counts
def draw_power(draws: list, zs, ys=Y_SPEC) -> np.ndarray:
    """[D, g, scenario, k of the design's range]: the power side's P(PASS) of the diagnosis design."""
    i, ks = design_index(zs, ys)
    n = draws[0]["meta"]["n_rep"]
    p = np.stack([d["arrays"]["hits"][:, 0][(slice(None), slice(None)) + i] for d in draws]).astype(float) / n
    return p[..., ks]


def per_draw_worst(pw: np.ndarray) -> np.ndarray:
    return pw.min(axis=(1, 2, 3))


def rec_distribution(pw: np.ndarray, zs, ys=Y_SPEC) -> dict:
    w = per_draw_worst(pw)
    out = {f"draw_q{q:g}": float(v) for q, v in zip(zs.draw_qs, _pct(w, list(zs.draw_qs)))}
    out.update({f"n_below_{b:g}": int((w < b).sum()) for b in zs.below})
    base, near = pw[:, :, 0].min(axis=(1, 2)), pw[:, :, 1].min(axis=(1, 2))
    for q in zs.scen_qs:
        out[f"base_q{q:g}"] = float(_pct(base, q))
        out[f"near_q{q:g}"] = float(_pct(near, q))
    low = w < zs.low_power
    out["n_low"] = int(low.sum())
    for gi, g in enumerate(ys.cluster_grid):
        out[f"low_near_g{g:g}"] = float(np.median(pw[low][:, gi, 1].min(-1)))
        out[f"low_base_g{g:g}"] = float(np.median(pw[low][:, gi, 0].min(-1)))
    return out


def rec_base_only(draws: list, zs, ys=Y_SPEC) -> dict:
    """baseonly.py: simultaneous 5th / 95th percentiles per scenario set, no envelope, no fill rule (Z.0 설명용)."""
    n = draws[0]["meta"]["n_rep"]
    p = np.stack([d["arrays"]["hits"] for d in draws]).astype(float) / n
    pw, fp = p[:, :, 0], p[:, :, 1]
    lo, hi = ys.pct_scale * (1 - ys.boot_level), ys.pct_scale * ys.boot_level
    out = {}
    for name, sc in (("both", [0, 1]), ("base_only", [0]), ("near_only", [1])):
        tot = 0
        for kr in ys.k_ranges:
            ks = [k - ys.k_min for k in range(kr[0], kr[1] + 1)]
            sim = _pct(pw[:, :, sc][..., ks].min(-1), lo).reshape((-1,) + pw.shape[3:-1]).min(0)
            fs = _pct(fp[:, :, sc][..., ks].max(-1), hi).reshape((-1,) + pw.shape[3:-1]).max(0)
            ok = (sim >= ys.p_power) & (fs <= ys.p_false)
            tag = f"{kr[0]}_{kr[1]}"
            if name != "both":
                out[f"{name}_n_{tag}"] = int(ok.sum())
                out[f"{name}_max_{tag}"] = float(sim.max())
                out[f"{name}_by_pset_{tag}"] = ok.sum(axis=(1, 2, 3)).tolist()
            tot += int(ok.sum())
        if name == "near_only":
            out["near_only_n"] = tot
    i, ks = design_index(zs, ys)
    b = pw[:, :, 0][(slice(None), slice(None)) + i][..., ks]
    kw = _pct(b, lo).min(0)
    out["base_only_sim"] = float(_pct(b.min(-1), lo).min())
    out.update({f"base_only_k{k}": float(v) for k, v in zip(range(zs.diag_design[4][0], zs.diag_design[4][1] + 1),
                                                              kw)})
    return out


# ================================================================ 0b record: θ* components and the ρ table
def components(theta, r, draws: list, ys=Y_SPEC) -> list:
    """comp.py: per draw the θ* summary (boot_y on Y's draw stream) and the stored power / false corners."""
    rows = []
    J, F = len(theta["pair_means"]), theta["n_fly"]
    for bi, d in enumerate(draws):
        tb = y_oc.boot_y(theta, y_oc.rng(ys.oc_seed, y_oc.TAG_BOOT, bi), r)
        rb = y_oc.rng(ys.oc_seed, y_oc.TAG_BOOT, bi)
        rb.multivariate_normal(theta["m0"].ravel(), theta["pair_cov"], J, method="eigh")
        rb.multivariate_normal(theta["drift"].ravel(), w_oc._cov(theta["drift_pair"]), J, method="eigh")
        rb.multivariate_normal(theta["spill_pair"].mean(0).ravel(), w_oc._cov(theta["spill_pair"]), J, method="eigh")
        rb.multivariate_normal(np.zeros(4), theta["fly_cov"], J * max(F - 1, 1), method="eigh")
        pick = rb.integers(0, J, J)
        cm, cx = d["meta"]["cal"]["min"]["corners"], d["meta"]["cal"]["max"]["corners"]
        rows.append(dict(m0=tb["m0"].ravel().tolist(), m0s=tb["m0s"][..., 0].ravel().tolist(),
                         pc=np.diag(tb["pair_cov"]).tolist(), fc=np.diag(tb["fly_cov"]).tolist(),
                         drift=tb["drift"].ravel().tolist(), spill=[float(v) for v in tb["spill"]],
                         rsd_all=float(tb["resid"].std()),
                         rsd=tb["resid"].reshape(len(tb["resid"]), -1).std(0).tolist(),
                         pick=np.bincount(pick, minlength=J).tolist(),
                         a=sum(c["a"] * c["w"] for c in cm), b=sum(c["b"] * c["w"] for c in cm),
                         fa=sum(c["a"] * c["w"] for c in cx), fb=sum(c["b"] * c["w"] for c in cx)))
    return rows


def _features(rows: list) -> dict:
    X = {}
    cells = ("AX", "AY", "PX", "PY")
    for i, n in enumerate(cells):
        X[f"m0_{n}"] = [c["m0"][i] for c in rows]
        X[f"sdpair_{n}"] = [float(np.sqrt(c["pc"][i])) for c in rows]
        X[f"sdfly_{n}"] = [float(np.sqrt(c["fc"][i])) for c in rows]
    X["m0s_A"], X["m0s_P"] = [c["m0s"][0] for c in rows], [c["m0s"][1] for c in rows]
    for j, dn in enumerate(("N1pre", "N2N1", "RN2RN1")):
        for i, n in enumerate(cells):
            X[f"drift_{dn}_{n}"] = [c["drift"][len(cells) * j + i] for c in rows]
    X["spillP"], X["spillA"] = [c["spill"][0] for c in rows], [c["spill"][1] for c in rows]
    X["rsd_all"] = [c["rsd_all"] for c in rows]
    X["a_pow"], X["b_pow"] = [c["a"] for c in rows], [c["b"] for c in rows]
    X["a_false"], X["b_false"] = [c["fa"] for c in rows], [c["fb"] for c in rows]
    X["maxmult"] = [max(c["pick"]) for c in rows]
    for j in range(len(rows[0]["pick"])):
        X[f"mult_pair{j}"] = [c["pick"][j] for c in rows]
    X["n_distinct"] = [sum(1 for v in c["pick"] if v > 0) for c in rows]
    return {k: np.asarray(v, float) for k, v in X.items()}


def _r2(cols: list, y) -> float:
    M = np.column_stack([np.ones(len(y))] + [(c - c.mean()) / c.std() for c in cols])
    beta, *_ = np.linalg.lstsq(M, y, rcond=None)
    pred = M @ beta
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def rec_rho(rows: list, w: np.ndarray, zs, ys=Y_SPEC) -> dict:
    """an.py / an2.py: Spearman ρ of every θ* component with the per-draw worst power, medians low / rest, the
    a × drift 2 × 2 and the R² of the drift, a, and six columns."""
    from scipy import stats
    X = _features(rows)
    low = w < zs.low_power
    table = []
    for k, v in X.items():
        rho = stats.spearmanr(v, w).statistic
        table.append(dict(name=k, rho=float(rho), med_low=float(np.median(v[low])), med_rest=float(np.median(v[~low]))))
    table.sort(key=lambda t: -abs(t["rho"]) if np.isfinite(t["rho"]) else 0.0)
    out = {f"rho_{t['name']}": t["rho"] for t in table}
    out["rho_a_drift"] = float(stats.spearmanr(X["a_pow"], X["drift_N1pre_AX"]).statistic)
    a = X["a_pow"]
    rw = w - np.polyval(np.polyfit(a, w, 1), a)
    ry = X["sdpair_AY"] - np.polyval(np.polyfit(a, X["sdpair_AY"], 1), a)
    out["rho_sdpair_AY_partial_a"] = float(stats.spearmanr(ry, rw).statistic)     # "a 통제 뒤 부분 ρ" (Reading 6)
    out["r2_drift"] = _r2([X["drift_N1pre_AX"]], w)
    out["r2_a"] = _r2([X["a_pow"]], w)
    out["r2_six"] = _r2([X[c] for c in zs.rho_cols], w)
    qa = X["a_pow"] > np.median(X["a_pow"])
    qd = X["drift_N1pre_AX"] < np.median(X["drift_N1pre_AX"])
    cells = {}
    for i in (0, 1):
        for j in (0, 1):
            s = (qa == i) & (qd == j)
            cells[f"a_high={i}|drift_low={j}"] = dict(n=int(s.sum()), low_share=float(low[s].mean()) if s.any() else None)
    out["cell_hi_lo_n"] = cells["a_high=1|drift_low=1"]["n"]
    out["cell_hi_lo_low_pct"] = float(cells["a_high=1|drift_low=1"]["low_share"] * ys.pct_scale)
    out["_table"] = table
    out["_cells"] = cells
    return out


# ================================================================ 0b record: near-threshold true d′, reach, point θ̂
def _tb(theta, r, bi: int, ys=Y_SPEC):
    rb = y_oc.rng(ys.oc_seed, y_oc.TAG_BOOT, bi)
    tb = y_oc.boot_y(theta, rb, r)
    return tb, rb


def near_truedprimes(theta, r, z: dict, draws: list, ys=Y_SPEC) -> np.ndarray:
    """td.py: per draw, the stored power corners' weighted true d′ (w_oc.true_dprimes — already directional) at the
    near-shifted base."""
    out = []
    for bi, d in enumerate(draws):
        tb, rb = _tb(theta, r, bi, ys)
        idx = rb.integers(0, len(tb["resid"]), ys.cal_reps)
        c = d["meta"]["cal"]["min"]["corners"]
        t = dict(tb, m0s=y_oc.near_shift(tb["m0s"], ys))
        out.append(sum(k["w"] * w_oc.true_dprimes(t, k["a"], k["b"], idx, z) for k in c))
    return np.asarray(out)


def rec_near(nr: np.ndarray, w: np.ndarray, zs) -> dict:
    low = w < zs.low_power
    mn = nr.min(1)
    out = {f"near_td_q{q:g}": float(v) for q, v in zip(zs.near_qs, _pct(mn, list(zs.near_qs)))}
    s = mn < zs.near_split
    out["near_td_lt_n"], out["near_td_lt_low"] = int(s.sum()), int(low[s].sum())
    out["near_td_ge_low_share"] = float(low[~s].mean()) if (~s).any() else None
    med = np.median(nr[low], 0)
    out.update({f"low_gate_{g}": float(v) for g, v in zip(WV.GATES, med)})
    out["low_min_gate_counts"] = dict(zip(WV.GATES, np.bincount(nr[low].argmin(1), minlength=len(WV.GATES)).tolist()))
    return out


def reach(theta, r, z: dict, zs, ys=Y_SPEC) -> np.ndarray:
    """reach.py: per draw, the largest reward min-gate true d′ reachable at the near base over the a grid (b 0)."""
    lo, hi, st = zs.reach_grid
    res = []
    for bi in range(zs.y_cells_n):
        tb, rb = _tb(theta, r, bi, ys)
        idx = rb.integers(0, len(tb["resid"]), ys.cal_reps)[:zs.reach_resid]
        tn = dict(tb, m0s=y_oc.near_shift(tb["m0s"], ys))
        res.append(max(w_oc.true_dprimes(tn, a, 0.0, idx, z)[[0, 2]].min() for a in np.arange(lo, hi, st)))
    return np.asarray(res)


def rec_reach(best: np.ndarray, w: np.ndarray, zs, ys=Y_SPEC) -> dict:
    low = w < zs.low_power
    s = best < ys.d_power
    return dict(unreachable=int(s.sum()), unreachable_low=int(low[s].sum()))


def point_near(theta, z: dict, cal_min: dict, zs, ys=Y_SPEC) -> dict:
    """pt.py: at Y θ̂ with the precheck's point calibration — its first corner (label a0b0, the a 32.268 · b 6.107 the
    script used; plan Reading 6) — the near-threshold true d′ of the four gates (directional), and the largest
    reachable reward min-gate d′ over the a grid."""
    idx = y_oc.rng(ys.precheck_seed, y_oc.TAG_CAL).integers(0, len(theta["resid"]), ys.cal_reps)
    a, b = float(cal_min["corners"][0]["a"]), float(cal_min["corners"][0]["b"])
    tn = dict(theta, m0s=y_oc.near_shift(theta["m0s"], ys))
    td = w_oc.true_dprimes(tn, a, b, idx, z)
    lo, hi, st = zs.point_grid
    best = max(w_oc.true_dprimes(tn, x, 0.0, idx, z)[[0, 2]].min() for x in np.arange(lo, hi, st))
    out = {f"point_near_{g}": float(v) for g, v in zip(WV.GATES, td)}
    return dict(out, point_a=float(a), point_b=float(b), point_reach_max=float(best))


def draw_thresholds(theta, r, draws: list, w: np.ndarray, zs, ys=Y_SPEC) -> dict:
    """thr.py: Y.0's thresholds per draw on θ* with that draw's weighted power calibration."""
    ca, cp = [], []
    for bi, d in enumerate(draws):
        tb, _ = _tb(theta, r, bi, ys)
        c = d["meta"]["cal"]["min"]["corners"]
        t = y_oc.thresholds(tb, sum(k["a"] * k["w"] for k in c), sum(k["b"] * k["w"] for k in c), ys)
        ca.append(t["c_A_calc"])
        cp.append(t["c_P_calc"])
    ca, cp = np.asarray(ca), np.asarray(cp)
    low = w < zs.low_power
    out = {f"cA_q{q:g}": float(v) for q, v in zip(zs.thr_qs, _pct(ca, list(zs.thr_qs)))}
    out.update({f"cP_q{q:g}": float(v) for q, v in zip(zs.thr_qs, _pct(cp, list(zs.thr_qs)))})
    both = (ca > ys.c_a) & (cp > ys.c_p)
    return dict(out, thr_both_n=int(both.sum()), thr_both_low=int(low[both].sum()))


# ================================================================ 0b record: pooled jobs (module level for spawn)
def selfbase_job(theta, r, z: dict, bi: int, roots: tuple) -> dict:
    """cfC.py: draw bi's power calibration redone at the near-shifted base; if it calibrates, near power k 4–8."""
    ys = Y_SPEC
    tb, rb = _tb(theta, r, bi, ys)
    idx = rb.integers(0, len(tb["resid"]), ys.cal_reps)
    tn = dict(tb, m0s=y_oc.near_shift(tb["m0s"], ys))
    c = y_oc.calibrate_y(tn, ys.d_power, "min", idx, z, ys)
    meta = dict(ok=bool(c["ok"]), status=c["a"]["status"], failure=c["failure"])
    if not c["ok"]:
        return dict(arrays=dict(p=np.zeros(0)), meta=meta)
    ones = [1.0] * ys.f_max
    e = y_oc.evaluate_y(tb, y_oc.rng(roots[4], bi), y_oc.rng(roots[5], bi), ys.boot_reps,
                        [c["corners"]] * ys.k_cap, ones, ones, ys.cluster_grid[-1], z, ys, "near")
    return dict(arrays=dict(p=np.asarray(e["p"])), meta=meta)


def cf_job(theta, r, z: dict, corners: list, bi: int, kind: str, ca: float, cp: float, roots: tuple) -> dict:
    """cf.py / cfA.py / cfB.py: the near scenario (and cf.py's base at roots 3 · 4) at g 1.0 with the stored power
    corners under thresholds (ca, cp); the streams are the scripts' (root 1 / 2, draw, int(threshold))."""
    ys = Y_SPEC
    tb, _ = _tb(theta, r, bi, ys)
    yd = dataclasses.replace(ys, c_a=float(ca), c_p=float(cp))
    ones = [1.0] * ys.f_max
    if kind == "base":
        e = y_oc.evaluate_y(tb, y_oc.rng(roots[2], bi), y_oc.rng(roots[3], bi), ys.boot_reps, [corners] * ys.k_cap,
                            ones, ones, ys.cluster_grid[-1], z, ys, "base")
    else:
        t = int(cp) if kind == "p" else int(ca)
        e = y_oc.evaluate_y(tb, y_oc.rng(roots[0], bi, t), y_oc.rng(roots[1], bi, t), ys.boot_reps,
                            [corners] * ys.k_cap, ones, ones, ys.cluster_grid[-1], z, yd, "near")
    return dict(arrays=dict(p=np.asarray(e["p"])), meta=dict(kind=kind, ca=ca, cp=cp))


def sub_job(theta, r, z: dict, corners: list, bi: int, ca: float, cp: float) -> dict:
    """sub50.py: draw bi under thresholds (ca, cp) on Y's own streams (power side, both scenarios, every g)."""
    ys = Y_SPEC
    tb, _ = _tb(theta, r, bi, ys)
    yd = dataclasses.replace(ys, c_a=float(ca), c_p=float(cp))
    ones = [1.0] * ys.f_max
    out = []
    for gi, g in enumerate(ys.cluster_grid):
        row = []
        for si, sc in enumerate(y_oc.SCENARIOS):
            extra = (y_oc.tag("near"),) if sc == "near" else ()
            e = y_oc.evaluate_y(tb, y_oc.rng(ys.oc_seed, y_oc.TAG_BOOT_SIM, bi, gi, 0, *extra),
                                y_oc.rng(ys.oc_seed, y_oc.tag("mix"), y_oc.TAG_BOOT_SIM, bi, gi, 0, si),
                                ys.boot_reps, [corners] * ys.k_cap, ones, ones, g, z, yd, sc)
            row.append(np.asarray(e["p"]))
        out.append(row)
    return dict(arrays=dict(p=np.asarray(out)), meta=dict(ca=ca, cp=cp))


def tiled(theta, n: int, delta: float, share: tuple):
    """psize.tiled + Z.9.2 P1-3's shift: Y θ̂'s pair-level arrays tiled to n pairs, δ added to every pair's MBON13(X)
    N1 − pre drift (spread kept, centre moved), Y's point Σ_pair kept, n_Σ = round(n · 5 / 6)."""
    rep = int(np.ceil(n / len(theta["pair_means"])))
    pm = np.concatenate([theta["pair_means"]] * rep)[:n]
    dp = np.concatenate([theta["drift_pair"]] * rep)[:n].copy()
    dp[:, 0, WV.A, WV.X] += delta
    sp = np.concatenate([theta["spill_pair"]] * rep)[:n]
    rb = (list(theta["resid_blocks"]) * rep)[:n]
    t = w_oc.assemble(pm, dp, sp, theta["fly_cov"], rb, theta["n_fly"], theta["n_probe"])
    return dict(t, pair_cov=theta["pair_cov"], n_sigma=int(round(n * share[0] / share[1])))


def _power_job(t, r, z: dict, boot_rng, sim_rng, mix_rng, ys) -> dict:
    """boot_y → calibrate_y (power, d′ 1.5) → evaluate_y per g × scenario; P(PASS) of the full grid's design axis
    is returned whole (the caller indexes the design). A failed calibration → zeros."""
    tb = y_oc.boot_y(t, boot_rng, r)
    ib = boot_rng.integers(0, len(tb["resid"]), ys.cal_reps)
    c = y_oc.calibrate_y(tb, ys.d_power, "min", ib, z, ys)
    shp = (len(ys.cluster_grid), len(y_oc.SCENARIOS)) + y_oc.x_oc.grid_shape(ys)
    meta = dict(ok=bool(c["ok"]), status=(c["failure"] or {}).get("status"), drift_ax=float(tb["drift"][0][WV.A, WV.X]))
    if not c["ok"]:
        return dict(arrays=dict(p=np.zeros(shp)), meta=dict(meta, a=None))
    ones = [1.0] * ys.f_max
    p = np.zeros(shp)
    for gi, g in enumerate(ys.cluster_grid):
        for si, sc in enumerate(y_oc.SCENARIOS):
            e = y_oc.evaluate_y(tb, sim_rng(gi, si), mix_rng(gi, si), ys.boot_reps, [c["corners"]] * ys.k_cap, ones,
                                ones, g, z, ys, sc)
            p[gi, si] = e["p"]
    return dict(arrays=dict(p=p), meta=dict(meta, a=float(sum(k["a"] * k["w"] for k in c["corners"]))))


def psize_job(theta, r, z: dict, n: int, bi: int, root: int, share: tuple) -> dict:
    """psize.py: Y θ̂ (n 6) or tiled to n; streams (root, n, TAG_BOOT, bi), (root, n, 1 / 2, bi, g, scenario)."""
    ys = Y_SPEC
    t = theta if n == len(theta["pair_means"]) else tiled(theta, n, 0.0, share)
    return _power_job(t, r, z, y_oc.rng(root, n, y_oc.TAG_BOOT, bi),
                      lambda gi, si: y_oc.rng(root, n, 1, bi, gi, si), lambda gi, si: y_oc.rng(root, n, 2, bi, gi, si),
                      ys)


def sens_job(theta, r, z: dict, n: int, di: int, delta: float, bi: int, root: int, share: tuple) -> dict:
    """Z.9.2 P1-3's draw: streams SeedSequence([86_000_000, tag, n, δ index, draw, …])."""
    ys = Y_SPEC
    t = tiled(theta, n, delta, share)
    return _power_job(t, r, z, y_oc.rng(root, y_oc.TAG_BOOT, n, di, bi),
                      lambda gi, si: y_oc.rng(root, y_oc.TAG_BOOT_SIM, n, di, bi, gi, si),
                      lambda gi, si: y_oc.rng(root, y_oc.tag("mix"), n, di, bi, gi, si), ys)


def design_k(p: np.ndarray, zs, ys=Y_SPEC) -> np.ndarray:
    """[..., *grid (p_set, q, K, F, k)] → [..., k of the design's range]."""
    i, ks = design_index(zs, ys)
    return p[(Ellipsis,) + i + (slice(None),)][..., ks]


def sim_limit(a: np.ndarray, ys=Y_SPEC) -> dict:
    """a [D, g, scenario, k]: the simultaneous 5th-percentile limit (worst g × scenario), k-wise, base-only."""
    lo = ys.pct_scale * (1 - ys.boot_level)
    return dict(sim=float(_pct(a.min(-1), lo).min()), by_k=_pct(a, lo).reshape(-1, a.shape[-1]).min(0).tolist(),
                base_only=float(_pct(a[:, :, 0].min(-1), lo).min()))


def sens_summary(a: np.ndarray, metas: list, zs, ys=Y_SPEC) -> dict:
    w = a.min(axis=(1, 2, 3))
    out = sim_limit(a, ys)
    out["per_draw_worst_q"] = dict(zip([f"{q:g}" for q in zs.sens_qs], _pct(w, list(zs.sens_qs)).tolist()))
    out["n_below_bar"] = int((w < zs.plan_bar).sum())
    out["n_cal_fail"] = sum(not m["ok"] for m in metas)
    out["a"] = [m.get("a") for m in metas]
    out["drift_ax"] = [m["drift_ax"] for m in metas]
    return out


def drift_facts(theta_y, theta_w) -> dict:
    """Z.0: W 16 pairs' MBON13(X) N1 − pre drift (mean, SD, Pearson with the pair's naive MBON13(X) mean), W's MBON05(X)
    drift, Y 6 pairs' drift SD and SE."""
    dw = theta_w["drift_pair"][:, 0, WV.A, WV.X]
    pw = theta_w["drift_pair"][:, 0, WV.P, WV.X]
    dy = theta_y["drift_pair"][:, 0, WV.A, WV.X]
    return dict(w_drift_corr=float(np.corrcoef(dw, theta_w["pair_means"][:, WV.A, WV.X])[0, 1]),
                w_drift_mean=float(dw.mean()), w_drift_sd=float(dw.std(ddof=1)), w_p_drift_mean=float(pw.mean()),
                w_p_drift_sd=float(pw.std(ddof=1)), y_drift_sd=float(dy.std(ddof=1)),
                y_drift_se=float(dy.std(ddof=1) / np.sqrt(len(dy))), y_drift=dy.tolist())


# ================================================================ 0b record: summaries of the pooled jobs
def rec_selfbase(metas: list) -> dict:
    return dict(selfbase_floor=sum(m["status"] == y_oc.FLOOR for m in metas),
                selfbase_status=[m["status"] for m in metas])


def rec_cf(rows: list, zs, ys=Y_SPEC) -> dict:
    """rows = [(kind, ca, cp, bi, p)]: k 4–8 power at g 1.0 (the scripts' index [0, 0, 1, 0, :5]) per draw; cfB's
    count of the six draws whose k 4–8 minimum reaches each bar."""
    out = {}
    for kind, ca, cp, bi, p in rows:
        out.setdefault(f"{kind}|{ca:g}|{cp:g}", {})[str(bi)] = design_k(np.asarray(p), zs, ys).tolist()
    for (ca, cp), bar in zip(zs.cf_ab, zs.cf_ab_bars):
        got = out.get(f"ab|{ca:g}|{cp:g}", {})
        out[f"cfB_{ca:g}_{cp:g}_n_ge"] = sum(min(v) >= bar for v in got.values())
    return out


def rec_sub(arrs: dict, zs, ys=Y_SPEC) -> dict:
    """arrs = {(ca, cp): [50, g, scenario, *grid]} → sim limit, k-wise and base-only at the diagnosis design."""
    out = {}
    for (ca, cp), a in arrs.items():
        lim = sim_limit(design_k(np.asarray(a), zs, ys), ys)
        out[f"sub_{ca:g}_{cp:g}"] = lim["sim"]
        out[f"sub_{ca:g}_{cp:g}_detail"] = lim
    return out


def rec_psize(arrs: dict, zs, ys=Y_SPEC) -> dict:
    """arrs = {n: [40, g, scenario, *grid]} → sim limit, base-only, per-draw worst below p_power."""
    out = {}
    for n, a in arrs.items():
        d = design_k(np.asarray(a), zs, ys)
        lim = sim_limit(d, ys)
        out[f"psize_{n}"] = lim["sim"]
        out[f"psize_{n}_base"] = lim["base_only"]
        out[f"psize_{n}_below"] = int((d.min(axis=(1, 2, 3)) < ys.p_power).sum())
        out[f"psize_{n}_detail"] = lim
    return out


# ================================================================ 0a: unit costs for the later P2-10 reservations
def unit_timing(theta, z: dict, zs, ys=Y_SPEC) -> dict:
    """Seconds per experiment of evaluate_y and of evaluate_pre (R-pre, Z.9.2 P1-3) on the diagnosis design
    (synthetic θ, the null corner, g 0, root 85_000_000 TAG_SYNTH), and the R-pre reconfirmation estimate per design
    (rpre_reps × Y's 200 draws × g × both targets). Record only."""
    import time
    p_set, q, K, F, kr = zs.diag_design
    yd = dataclasses.replace(y_oc.x_oc.design_spec(ys, dict(p_set=p_set, q=q, K=K, F=F)), k_cap=kr[1])
    mix = [y_oc.one_corner(0.0, 0.0)] * yd.k_cap
    ones = [1.0] * F
    out = {}
    for name, fn in (("evaluate_y", y_oc.evaluate_y), ("evaluate_pre", y_oc.evaluate_pre)):
        t = time.perf_counter()
        fn(theta, y_oc.rng(zs.oc_seed, y_oc.TAG_SYNTH, 0), y_oc.rng(zs.oc_seed, y_oc.TAG_SYNTH, 1), zs.timing_reps,
           mix, ones, ones, 0.0, z, yd)
        out[f"{name}_s_per_rep"] = (time.perf_counter() - t) / zs.timing_reps
    out["rpre_reconfirm_s_per_design"] = (out["evaluate_pre_s_per_rep"] * zs.rpre_reps * ys.boot_draws
                                          * len(ys.cluster_grid) * len(y_oc.MODES))
    return out
