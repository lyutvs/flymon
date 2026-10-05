"""X's operating-characteristic pieces, phase A (X.4.2, X.4.4, X.4.6, X.9.1.1–X.9.1.4; X.9.1.2 replaces X.4.3's rule).
W's model is used as is (w_oc: fit, generator, bootstrap draw, true d′); X adds only the following.
- The p_set axis: evaluate runs w_oc.simulate (the same draws as w_oc.evaluate) and W's pair verdict (w_verdict:
  fly gates, q, 2K BAND, mechanism control, RN1 = R1), then X's set rule (x_verdict) on the first k pairs for every
  p_set at once → P(PASS) [p_set, q, K, F, k]. In simulation BAND ends at 2K, so b = 0. At p_set 1.0 this equals
  w_oc.evaluate bit for bit (P2-6).
- Calibration per status (X.9.1.2): W's bisection with the bracket's upper end × bracket_mult (4); a no_convergence
  is retried once with cal_retry_iter steps (same draws, tolerance, brackets). A failure that remains — floor,
  no_convergence after the retry, or above_at_zero on the power side — fills P(PASS) with 0 (power) / 1 (false pass).
  On the false-pass side above_at_zero is W.9.10 2's zero_floor (not a failure).
- A simulator copy (pair_bases / simulate) with a rejection-round count and an extra acceptance predicate, for the
  record-only diagnostics. At W's 200 rounds and no predicate it equals w_oc's draw for draw.
- The W calibration-failure diagnosis (X.4.3 + X.9.1.2): W's bootstrap draws re-drawn from W's root, compared with W's
  stored calibrations, every status recorded; records only.
Every stream is SeedSequence([root, tag, …]); the roots are X's (x_spec) except the W reproduction (w_spec)."""
from __future__ import annotations

import dataclasses
import math
import time

import numpy as np

from . import w_oc
from . import w_verdict as WV
from . import x_verdict as XV

TAG_CAL, TAG_POINT, TAG_RECORD, TAG_SYNTH = w_oc.TAG_CAL, w_oc.TAG_POINT, w_oc.TAG_RECORD, w_oc.TAG_SYNTH
TAG_P26 = 10
MODES = (("min", "d_power"), ("max", "d_false"))


def rng(root, *tags):
    return np.random.default_rng(np.random.SeedSequence([int(root)] + [int(t) for t in tags]))


def run_cell(tag, key, fn):
    """The default cell runner (no cache); x_runner passes a resumable one."""
    return fn()


def grid_shape(xs) -> tuple:
    return (len(xs.p_set_grid), len(xs.q_grid), len(xs.k_grid), xs.f_max - xs.f_min + 1, xs.k_cap - xs.k_min + 1)


# ================================================================ calibration per status (X.9.1.2)
def brackets(theta, mult: float) -> tuple:
    """W's hi_a / hi_b (w_oc.calibrate's formulas) × mult."""
    r = np.abs(theta["resid"]).max()
    hi_a = float((theta["m0s"][WV.P, WV.X] + theta["drift"][0][WV.P, WV.X] + r) * 2 + 1)
    hi_b = float((theta["m0s"][WV.A, WV.X] + np.abs(theta["drift"]).sum() + r) * 2 + 1)
    return hi_a * mult, hi_b * mult


def calibrate(theta, target: float, mode: str, idx, z, spec, mult: float, n_iter: int) -> dict:
    """w_oc.calibrate with brackets × mult and n_iter steps; mult 1.0 with spec.cal_iter is w_oc.calibrate exactly."""
    agg = np.min if mode == "min" else np.max
    hi_a, hi_b = brackets(theta, mult)
    ra = w_oc._bisect(lambda x: agg(w_oc.true_dprimes(theta, x, 0.0, idx, z)[[0, 2]]) - target, hi_a, spec.cal_tol,
                      n_iter)
    if ra["status"] == "above_at_zero" and mode == "max" and spec.cal_floor_rule == "zero":
        ra = dict(value=0.0, status="zero_floor", f0=ra["f0"])
    if ra["value"] is None:
        return dict(ok=False, a=ra, b=None)
    rb = w_oc._bisect(lambda x: agg(w_oc.true_dprimes(theta, ra["value"], x, idx, z)[[1, 3]]) - target, hi_b,
                      spec.cal_tol, n_iter)
    if rb["status"] == "above_at_zero" and mode == "max" and spec.cal_floor_rule == "zero":
        rb = dict(value=0.0, status="zero_floor", f0=rb["f0"])
    if rb["value"] is None:
        return dict(ok=False, a=ra, b=rb)
    td = w_oc.true_dprimes(theta, ra["value"], rb["value"], idx, z)
    return dict(ok=True, a=ra, b=rb, true_dprime=dict(zip(WV.GATES, td.tolist())))


def failure(c: dict):
    """{handle, status} of a failed calibration, None when it succeeded."""
    if c["ok"]:
        return None
    h = "a" if c["a"]["value"] is None else "b"
    return dict(handle=h, status=c[h]["status"])


def fill_value(mode: str) -> float:
    return 0.0 if mode == "min" else 1.0


def calibrate_x(theta, target: float, mode: str, idx, z, xs) -> dict:
    """X.9.1.2: brackets × xs.bracket_mult, xs.cal_iter steps; a no_convergence → once more with xs.cal_retry_iter
    steps; a remaining failure fills (power 0 / false pass 1)."""
    c = calibrate(theta, target, mode, idx, z, xs, xs.bracket_mult, xs.cal_iter)
    first = failure(c)
    retried = first is not None and first["status"] == "no_convergence"
    if retried:
        c = calibrate(theta, target, mode, idx, z, xs, xs.bracket_mult, xs.cal_retry_iter)
    return dict(c, first_failure=first, retried=retried, failure=failure(c),
                fill=None if c["ok"] else fill_value(mode))


# ================================================================ the simulator copy (diagnostics)
def pair_bases(theta, rng_, n_rep, n_pair, g, z, xs, tries: int, accept=None) -> tuple:
    """w_oc._pair_bases with `tries` rounds and an extra predicate accept(base [..., 2, 2]) -> bool [...]; returns
    (bases [n_rep, n_pair, 2, 2], filled [n_rep, n_pair] — the slots given the population pair)."""
    sdp = w_oc.sd_pre(theta, z)
    L = np.linalg.cholesky(theta["pair_cov"] + xs.chol_jitter * np.eye(4))
    w = (rng_.standard_normal((n_rep, 4)) @ L.T * math.sqrt(g)).reshape(n_rep, 1, 2, 2)
    out = np.empty((n_rep, n_pair, 2, 2))
    need = np.ones((n_rep, n_pair), bool)
    for _ in range(tries):
        u = (rng_.standard_normal((n_rep, n_pair, 4)) @ L.T).reshape(n_rep, n_pair, 2, 2)
        base = theta["m0s"] + w + u
        ok = np.abs(WV.dv(base, z)) / sdp < xs.naive_max
        if accept is not None:
            ok &= accept(base)
        take = need & ok
        out[take] = base[take]
        need &= ~ok
        if not need.any():
            return out, need
    out[need] = (theta["m0s"] + w + 0 * out)[need]
    return out, need


def simulate(theta, rng_, n_rep, n_pair, n_fly, n_probe, a_pair, b_pair, fly_a, fly_b, g, z, xs, tries: int,
             accept=None) -> tuple:
    """w_oc.simulate through pair_bases: (stages, bases, filled)."""
    base, filled = pair_bases(theta, rng_, n_rep, n_pair, g, z, xs, tries, accept)
    Lf = np.linalg.cholesky(theta["fly_cov"] + xs.chol_jitter * np.eye(4))
    v = (rng_.standard_normal((n_rep, n_pair, n_fly, 4)) @ Lf.T).reshape(n_rep, n_pair, n_fly, 2, 2)
    a = np.asarray(a_pair, float)[None, :, None] * np.asarray(fly_a, float)[None, None, :]
    b = np.asarray(b_pair, float)[None, :, None] * np.asarray(fly_b, float)[None, None, :]
    mu = w_oc.slot_means(theta, base[:, :, None], v, np.broadcast_to(a, v.shape[:3]), np.broadcast_to(b, v.shape[:3]))
    idx = rng_.integers(0, len(theta["resid"]), (n_rep, n_pair, n_fly, n_probe))
    counts = np.clip(np.rint(mu[:, :, :, None] + theta["resid"][idx]), 0, None).astype(np.int32)
    return w_oc.to_stages(counts), base, filled


# ================================================================ evaluation: W's pair verdict, X's set rule
def pair_finals(d, z, xs, qs, ks, fs):
    """Yields (qi, ki, fi, final pair codes [n, pairs]) — W's pair verdict per design, exactly as w_oc.evaluate."""
    kw = dict(bar=xs.bar, band=xs.band_width, digits=xs.round_digits)
    for ki, K in enumerate(ks):
        dK = {s: v[..., :K, :, :] for s, v in d.items()}
        d2 = {s: v[..., :2 * K, :, :] for s, v in d.items()}
        cK, c2 = WV.fly_class(WV.gate_stats(dK, z), **kw), WV.fly_class(WV.gate_stats(d2, z), **kw)
        fK, f2 = WV.mech_fractions(dK), WV.mech_fractions(d2)
        for fi, F in enumerate(fs):
            mK = WV.mech_ok(fK[..., :F, :], xs.mech_min, xs.round_digits)
            m2 = WV.mech_ok(f2[..., :F, :], xs.mech_min, xs.round_digits)
            for qi, q in enumerate(qs):
                yield qi, ki, fi, WV.pair_final(WV.pair_gate_code(cK[..., :F], q, F, xs.round_digits), mK,
                                                WV.pair_gate_code(c2[..., :F], q, F, xs.round_digits), m2)


def set_hits(fin, mach, kk, xs) -> np.ndarray:
    """[p_set, len(kk)] PASS counts of X's set rule on the first k pairs (nested in k), every p_set and k at once."""
    fin, mach = np.asarray(fin), np.asarray(mach, bool)
    j = np.asarray(kk) - 1
    m = (fin == WV.P_PASS).cumsum(-1)[:, j]
    n = m + (fin == WV.P_FAIL).cumsum(-1)[:, j]
    b = (fin == WV.P_BAND).cumsum(-1)[:, j]
    stop = ((mach | (fin == WV.P_INVALID)).cumsum(-1) > 0)[:, j]
    ps = np.asarray(xs.p_set_grid, float)[:, None, None]
    return (XV.set_code_counts(n, m, b, stop, ps, xs) == XV.S_PASS).sum(1)


def tally(d, z, xs, qs, ks, fs, kk) -> np.ndarray:
    mach = WV.rn1_mismatch(d)
    out = np.zeros((len(xs.p_set_grid), len(qs), len(ks), len(fs), len(kk)))
    for qi, ki, fi, fin in pair_finals(d, z, xs, qs, ks, fs):
        out[:, qi, ki, fi] = set_hits(fin, mach, kk, xs)
    return out


def evaluate(theta, rng_, n_rep, a_pair, b_pair, fly_a, fly_b, g, z, xs) -> np.ndarray:
    """P(PASS) [p_set, q, K, F, k] on w_oc.simulate (the draws of w_oc.evaluate) under X's set rule."""
    fs = list(range(xs.f_min, xs.f_max + 1))
    kk = list(range(xs.k_min, xs.k_cap + 1))
    hits = np.zeros(grid_shape(xs))
    done = 0
    while done < n_rep:
        n = min(xs.oc_chunk, n_rep - done)
        d = w_oc.simulate(theta, rng_, n, xs.k_cap, xs.f_max, 2 * max(xs.k_grid), a_pair, b_pair, fly_a, fly_b, g, z,
                          xs.naive_max)
        hits += tally(d, z, xs, xs.q_grid, xs.k_grid, fs, kk)
        done += n
    return hits / n_rep


# ================================================================ X.4.6 synthetic validation, P2-6, timing
def synthetic_validation(xs, z: dict, n_rep: int | None = None) -> dict:
    """X.4.6: W.9.9 P2-11's five fixtures (w_oc.synthetic_validation's structure) on X's evaluate and calibration,
    checked over every p_set: zero effect ≤ synth_null_max, big effect (d′ synth_big_dprime) ≥ synth_big_min, one
    gate only ≤, negative correlation ≤, the simple normal model not rising from F 8 to 32."""
    n_rep = xs.synth_reps if n_rep is None else int(n_rep)
    r = rng(xs.oc_seed, TAG_SYNTH)
    kw = dict(base=xs.synth_base, sd=xs.synth_sd, corr=xs.synth_corr, learn=xs.synth_learn)
    th0 = w_oc.fit(w_oc.synthetic_pilot(r, **kw))
    th_drift = w_oc.fit(w_oc.synthetic_pilot(r, drift_ax=xs.synth_drift, **kw))
    idx = r.integers(0, len(th0["resid"]), xs.cal_reps)
    lo, hi = xs.synth_null_max, xs.synth_big_min
    res = {}
    z0 = evaluate(th0, r, n_rep, *w_oc._uniform(xs, 0.0, 0.0), 0.0, z, xs)
    res["zero_effect"] = dict(max_p=float(z0.max()), limit=lo, ok=bool(z0.max() <= lo))
    big = calibrate_x(th0, xs.synth_big_dprime, "min", idx, z, xs)
    pb = (evaluate(th0, r, n_rep, *w_oc._uniform(xs, big["a"]["value"], big["b"]["value"]), 0.0, z, xs)
          if big["ok"] else np.zeros(grid_shape(xs)))
    res["big_effect"] = dict(min_p=float(pb.min()), min_p_by_p_set=pb.reshape(len(xs.p_set_grid), -1).min(1).tolist(),
                             limit=hi, ok=bool(big["ok"] and pb.min() >= hi), true_dprime=big.get("true_dprime"))
    one = evaluate(th_drift, r, n_rep, *w_oc._uniform(xs, 0.0, 0.0), 0.0, z, xs)
    td = w_oc.true_dprimes(th_drift, 0.0, 0.0, r.integers(0, len(th_drift["resid"]), xs.cal_reps), z)
    res["one_gate"] = dict(max_p=float(one.max()), limit=lo,
                           ok=bool(one.max() <= lo and td[1] >= xs.synth_drift_dprime_min),
                           true_dprime=dict(zip(WV.GATES, td.tolist())))
    if big["ok"]:
        alt = [1.0 if i % 2 == 0 else 0.0 for i in range(xs.f_max)]
        neg = evaluate(th0, r, n_rep, [big["a"]["value"]] * xs.k_cap, [big["b"]["value"]] * xs.k_cap, alt,
                       [1.0 - x for x in alt], 0.0, z, xs)
        res["negative_correlation"] = dict(max_p=float(neg.max()), limit=lo, ok=bool(neg.max() <= lo))
    else:
        res["negative_correlation"] = dict(max_p=None, ok=False)
    sn = w_oc.simple_normal(xs, r)
    p = np.array([sn[F] for F in xs.simple_normal_fs])
    res["simple_normal"] = dict(p_by_F={str(k): v for k, v in sn.items()}, diffs=np.diff(p).tolist(),
                                tol=w_oc.SIMPLE_NORMAL_TOL, ok=bool(np.all(np.diff(p) <= w_oc.SIMPLE_NORMAL_TOL)))
    res["ok"] = all(v["ok"] for v in res.values() if isinstance(v, dict))
    return res


def bit_identity(theta, z, xs, w_spec, n_rep: int, a: float, b: float) -> dict:
    """X.9.1.3 P2-6: X's evaluate at p_set 1.0 (b = 0 in simulation) equals w_oc.evaluate on the same θ and seed,
    numpy.array_equal over [q, K, F, k], at every cluster level (root oc_seed, tag TAG_P26, g index)."""
    pi = xs.p_set_grid.index(1.0)
    u = w_oc._uniform(xs, a, b)
    by_g, top = {}, 0.0
    for gi, g in enumerate(xs.cluster_grid):
        x = evaluate(theta, rng(xs.oc_seed, TAG_P26, gi), n_rep, *u, g, z, xs)[pi]
        w = w_oc.evaluate(theta, rng(xs.oc_seed, TAG_P26, gi), n_rep, *u, g, z, w_spec)
        by_g[f"g{g}"] = dict(equal=bool(np.array_equal(x, w)), max_abs_diff=float(np.abs(x - w).max()))
        top = max(top, float(w.max()))
    return dict(equal=all(v["equal"] for v in by_g.values()), by_g=by_g, max_p=top, n_rep=int(n_rep), a=a, b=b)


def oc_timing(theta, z, xs) -> dict:
    """X.5 0: one evaluate at boot_reps experiments (θ̂'s power calibration, the last cluster level), scaled to the
    precheck (g × 2 targets × precheck_reps) and to phase B's bootstrap (boot_draws × variants × g × 2 × boot_reps)."""
    idx = rng(xs.oc_seed, TAG_CAL).integers(0, len(theta["resid"]), xs.cal_reps)
    c = calibrate_x(theta, xs.d_power, "min", idx, z, xs)
    a, b = (c["a"]["value"], c["b"]["value"]) if c["ok"] else (0.0, 0.0)
    t0 = time.perf_counter()
    evaluate(theta, rng(xs.oc_seed, TAG_P26, len(xs.cluster_grid)), xs.boot_reps, *w_oc._uniform(xs, a, b),
             xs.cluster_grid[-1], z, xs)
    s = time.perf_counter() - t0
    per = len(xs.cluster_grid) * len(MODES)
    return dict(evaluate_s=s, reps=xs.boot_reps, precheck_point_s=s * per * xs.precheck_reps / xs.boot_reps,
                phase_b_boot_s=s * per * xs.boot_draws * len(xs.variants))


# ================================================================ W calibration-failure diagnosis (X.4.3 + X.9.1.2)
def _statuses(c: dict) -> dict:
    return dict(a=(c["a"] or {}).get("status"), b=(c["b"] or {}).get("status"))


def _lab(x) -> str:
    return x if isinstance(x, str) else f"{x['handle']}|{x['status']}"


def w_cal_diagnosis(theta, z, w_spec, xs, w_boot_cal: list, n_draws: int | None = None, log=None) -> dict:
    """W's bootstrap draws re-drawn from W's root (w_oc._rng(w_spec, TAG_BOOT, bi): boot, then the calibration
    draws) and calibrated: (가) as W did (brackets × w_bracket_mult, w_spec.cal_iter), compared with W's stored
    (ok, a, b); (나) a failed draw with brackets × bracket_mult; (다) a no_convergence draw with cal_retry_iter steps
    (W's brackets); (라) X's rule (calibrate_x). Records only: the rule is fixed (X.9.1.2)."""
    n = len(w_boot_cal) if n_draws is None else int(n_draws)
    rows, diffs = [], []
    for bi in range(n):
        rb = w_oc._rng(w_spec, w_oc.TAG_BOOT, bi)
        tb = w_oc.boot(theta, rb)
        ib = rb.integers(0, len(tb["resid"]), w_spec.cal_reps)
        row = dict(draw=bi)
        for m, field in MODES:
            t = getattr(w_spec, field)
            c = calibrate(tb, t, m, ib, z, w_spec, xs.w_bracket_mult, w_spec.cal_iter)
            got = dict(ok=c["ok"], a=(c["a"] or {}).get("value"), b=(c["b"] or {}).get("value"))
            if got != w_boot_cal[bi][m]:
                diffs.append(dict(draw=bi, side=m, x=got, w=w_boot_cal[bi][m]))
            e = dict(w=got, status=_statuses(c), failure=failure(c))
            if not c["ok"]:
                e["bracket_x4"] = failure(calibrate(tb, t, m, ib, z, w_spec, xs.bracket_mult, w_spec.cal_iter)) or "ok"
                if e["failure"]["status"] == "no_convergence":
                    e["retry_iter"] = failure(calibrate(tb, t, m, ib, z, w_spec, xs.w_bracket_mult,
                                                        xs.cal_retry_iter)) or "ok"
            e["x_rule"] = failure(calibrate_x(tb, t, m, ib, z, xs)) or "ok"
            row[m] = e
        rows.append(row)
        if log is not None and (bi + 1) % 10 == 0:
            log(f"x w-cal diagnosis {bi + 1}/{n}")

    def side(m):
        out = dict(n_fail=0, by_handle_status={}, bracket_x4={}, retry_iter={}, x_rule={}, statuses={})
        for r in rows:
            e = r[m]
            for h, s in e["status"].items():
                k = f"{h}|{s}"
                out["statuses"][k] = out["statuses"].get(k, 0) + 1
            if not e["w"]["ok"]:
                out["n_fail"] += 1
                for col, v in (("by_handle_status", e["failure"]), ("bracket_x4", e.get("bracket_x4")),
                               ("retry_iter", e.get("retry_iter"))):
                    if v is not None:
                        out[col][_lab(v)] = out[col].get(_lab(v), 0) + 1
            out["x_rule"][_lab(e["x_rule"])] = out["x_rule"].get(_lab(e["x_rule"]), 0) + 1
        return out
    fm = {r["draw"] for r in rows if not r["min"]["w"]["ok"]}
    fx = {r["draw"] for r in rows if not r["max"]["w"]["ok"]}
    counts = dict(total=len(fm | fx), power=len(fm), false=len(fx), both=len(fm & fx), min=side("min"),
                  max=side("max"))
    return dict(rows=rows, diffs=diffs, reproduced=not diffs, counts=counts, n_draws=n)


# ================================================================ variants (X.4.4) and the F1 filter's Σ (X.9.1.4)
def pair_cov_variant(pair_means, abs_d, name: str, xs) -> np.ndarray:
    """Σ_pair from the pilot pair means [J, 2, 2]: V0 all pairs (= w_oc's pair_cov), V1 |d′| < v1_exclude_from, V2
    weights 1 / (1 + |d′|) (numpy aweights), F1 the balanced pairs |d′| < naive_max. A variant on fewer than
    min_variant_pairs pairs keeps only its diagonal, with V0's correlations (Σ = D^½ R_V0 D^½)."""
    x = np.asarray(pair_means, float).reshape(len(pair_means), -1)
    d = np.abs(np.asarray(abs_d, float))
    v0 = np.cov(x, rowvar=False)
    if name == "V0":
        return v0
    if name == "V2":
        return np.cov(x, rowvar=False, aweights=1.0 / (1.0 + d))
    if name not in ("V1", "F1"):
        raise ValueError(f"unknown variant {name}")
    keep = d < (xs.v1_exclude_from if name == "V1" else xs.naive_max)
    c = np.cov(x[keep], rowvar=False)
    if keep.sum() >= xs.min_variant_pairs:
        return c
    s0 = np.sqrt(np.diag(v0))
    return (v0 / np.outer(s0, s0)) * np.outer(np.sqrt(np.diag(c)), np.sqrt(np.diag(c)))


def variant_thetas(theta, abs_d, xs) -> dict:
    return {v: dict(theta, pair_cov=pair_cov_variant(theta["pair_means"], abs_d, v, xs)) for v in xs.variants}


# ================================================================ the point-θ precheck (X.9.1.1)
def point_grid(theta, cal, root, n_rep, z, xs) -> dict:
    """{(g, mode): P(PASS) [p, q, K, F, k]} at θ̂ on w_oc.simulate; tag (TAG_POINT, g index, target); a failed
    calibration fills (X.9.1.2)."""
    out = {}
    for gi, g in enumerate(xs.cluster_grid):
        for m, _f in MODES:
            c = cal[m]
            out[(g, m)] = (evaluate(theta, rng(root, TAG_POINT, gi, int(m == "max")), n_rep,
                                    *w_oc._uniform(xs, c["a"]["value"], c["b"]["value"]), g, z, xs)
                           if c["ok"] else np.full(grid_shape(xs), c["fill"]))
    return out


def worst(point, xs) -> tuple:
    return (np.min([point[(g, "min")] for g in xs.cluster_grid], axis=0),
            np.max([point[(g, "max")] for g in xs.cluster_grid], axis=0))


def meets(power, false, xs) -> np.ndarray:
    d = xs.round_digits
    return (np.round(np.asarray(power) - xs.p_power, d) >= 0) & (np.round(np.asarray(false) - xs.p_false, d) <= 0)


def pick(power, false, xs, fs, require_false: bool) -> dict:
    """power / false [p, q, K, F, k]. require_false False: the largest min_k power ("점 검정력 최대 설계"). True:
    among designs whose max_k false ≤ p_false the largest min_k power, none → the smallest max_k false (then
    power). Ties: larger p_set, larger q, smaller K, smaller F (X.9.1.1; no cost in phase A)."""
    power, false = np.asarray(power), np.asarray(false)
    d = xs.round_digits
    pmin, fmax = np.round(power.min(-1), d), np.round(false.max(-1), d)
    ok = np.round(false.max(-1) - xs.p_false, d) <= 0
    cells = list(np.ndindex(pmin.shape))

    def tie(i):
        return (-xs.p_set_grid[i[0]], -xs.q_grid[i[1]], xs.k_grid[i[2]], fs[i[3]])
    if not require_false:
        rule, i = "max_power", min(cells, key=lambda i: (-pmin[i],) + tie(i))
    elif ok.any():
        rule, i = "false_ok_max_power", min((c for c in cells if ok[c]), key=lambda i: (-pmin[i],) + tie(i))
    else:
        rule, i = "min_false", min(cells, key=lambda i: (fmax[i], -pmin[i]) + tie(i))
    return dict(rule=rule, index=[int(v) for v in i], p_set=xs.p_set_grid[i[0]], q=xs.q_grid[i[1]],
                K=xs.k_grid[i[2]], F=int(fs[i[3]]), power_by_k=power[i].tolist(), false_by_k=false[i].tolist())


def table_at_f(power, false, xs, F: int) -> list:
    fi = F - xs.f_min
    ks = list(range(xs.k_min, xs.k_cap + 1))
    return [dict(p_set=p, q=q, K=K, F=F, k=ks, power=np.asarray(power)[pi, qi, ki, fi].tolist(),
                 false=np.asarray(false)[pi, qi, ki, fi].tolist())
            for pi, p in enumerate(xs.p_set_grid) for qi, q in enumerate(xs.q_grid) for ki, K in enumerate(xs.k_grid)]


def precheck(theta, z, xs, n_rep: int | None = None) -> dict:
    """X.9.1.1 on θ̂ (V0): calibration per status, P(PASS) for every design and k at every g, point power = g min,
    point false pass = g max; passed iff some design meets both targets at every k."""
    t0 = time.perf_counter()
    n_rep = xs.precheck_reps if n_rep is None else int(n_rep)
    root = xs.precheck_seed
    idx = rng(root, TAG_CAL).integers(0, len(theta["resid"]), xs.cal_reps)
    cal = {m: calibrate_x(theta, getattr(xs, f), m, idx, z, xs) for m, f in MODES}
    point = point_grid(theta, cal, root, n_rep, z, xs)
    power, false = worst(point, xs)
    fs = list(range(xs.f_min, xs.f_max + 1))
    ok = meets(power, false, xs).all(-1)
    passing = [dict(p_set=xs.p_set_grid[i[0]], q=xs.q_grid[i[1]], K=xs.k_grid[i[2]], F=fs[i[3]])
               for i in np.ndindex(ok.shape) if ok[i]]
    return dict(passed=bool(ok.any()), passing=passing, calibration=cal, power=power.tolist(), false=false.tolist(),
                point={f"g{g}|{m}": p.tolist() for (g, m), p in point.items()},
                best_power=pick(power, false, xs, fs, False), records_target=pick(power, false, xs, fs, True),
                at_f32=table_at_f(power, false, xs, xs.f_max), m_needed=XV.m_needed_table(xs),
                axes=dict(p_set=list(xs.p_set_grid), q=list(xs.q_grid), K=list(xs.k_grid), F=fs,
                          k=list(range(xs.k_min, xs.k_cap + 1)), g=list(xs.cluster_grid)),
                n_rep=n_rep, seed=int(root), timing_s=time.perf_counter() - t0)


# ================================================================ records at θ̂ (X.4.5 via X.9.1.1, precheck STOP only)
def design_spec(xs, d: dict):
    """The spec of one design (p_set, q, K, F): evaluate on it simulates F flies and 2K probes."""
    return dataclasses.replace(xs, p_set_grid=(d["p_set"],), q_grid=(d["q"],), k_grid=(d["K"],), f_min=d["F"],
                               f_max=d["F"])


def record_tags(xs) -> list:
    return ([f"t{t}" for t in xs.record_dprimes] + ["one", "half", "all1", "mixed"]
            + [f"het{s}|{m}" for m in ("p", "f") for s in xs.het_scales])


def point_records(theta, abs_d, target: dict, z, xs, n_rep: int | None = None, cell=run_cell, log=None) -> dict:
    """For the target design, P(PASS) by k on every variant (V0 · V1 · V2, records only — P3-12) × cluster level:
    the homogeneous true d′ 0.5 / 1.0 / 1.5 / 2.0 (power-style calibration), W's heterogeneous scenarios (one pair
    0.5 + rest 1.5, every other pair 0.5, all 1.0), the mixed flies (every other fly 0 at 1.5), and Σ_v × het_scales
    at 1.5 (power calibration) and 0.5 (false-pass calibration). Streams (TAG_RECORD, variant, g, scenario). A failed
    calibration → None with its status in `calibrations`."""
    n_rep = xs.oc_reps if n_rep is None else int(n_rep)
    root = xs.precheck_seed
    idx = rng(root, TAG_CAL).integers(0, len(theta["resid"]), xs.cal_reps)
    xd = design_spec(xs, target)
    cmin = {t: calibrate_x(theta, t, "min", idx, z, xs) for t in sorted(set(xs.record_dprimes) | {xs.d_power})}
    cmax = calibrate_x(theta, xs.d_false, "max", idx, z, xs)
    tags = record_tags(xs)
    thetas = variant_thetas(theta, abs_d, xs)
    K_, F_ = xs.k_cap, xd.f_max
    ones = [1.0] * F_
    half = [1.0 if i % 2 == 0 else 0.0 for i in range(F_)]

    def ab(c):
        return (c["a"]["value"], c["b"]["value"]) if c["ok"] else None

    def one(vi, v):
        th, out = thetas[v], {}
        for gi, g in enumerate(xs.cluster_grid):
            def at(th_, a_p, b_p, fa, fb, tag):
                r = rng(root, TAG_RECORD, vi, gi, tags.index(tag))
                return evaluate(th_, r, n_rep, a_p, b_p, fa, fb, g, z, xd)[0, 0, 0, 0].tolist()
            e = dict(true_dprime={str(t): (at(th, *w_oc._uniform(xd, *ab(cmin[t])), f"t{t}") if cmin[t]["ok"] else None)
                                  for t in xs.record_dprimes})
            lo, hi, mid = ab(cmin[xs.het_low_dprime]), ab(cmin[xs.d_power]), ab(cmin[xs.het_all_dprime])
            het = {}
            if lo and hi:
                het[f"one_pair_{xs.het_low_dprime}"] = at(th, [lo[0]] + [hi[0]] * (K_ - 1),
                                                          [lo[1]] + [hi[1]] * (K_ - 1), ones, ones, "one")
                het[f"half_pairs_{xs.het_low_dprime}"] = at(th, [lo[0] if i % 2 else hi[0] for i in range(K_)],
                                                            [lo[1] if i % 2 else hi[1] for i in range(K_)], ones, ones,
                                                            "half")
            if mid:
                het[f"all_{xs.het_all_dprime}"] = at(th, *w_oc._uniform(xd, *mid), "all1")
            e["heterogeneous_w"] = het
            e["mixed_flies"] = at(th, [hi[0]] * K_, [hi[1]] * K_, half, half, "mixed") if hi else None
            fa = ab(cmax)
            e["pair_cov_scaled"] = {}
            for s in xs.het_scales:
                ts = dict(th, pair_cov=th["pair_cov"] * s)
                e["pair_cov_scaled"][f"{s}|power"] = at(ts, *w_oc._uniform(xd, *hi), f"het{s}|p") if hi else None
                e["pair_cov_scaled"][f"{s}|false"] = at(ts, *w_oc._uniform(xd, *fa), f"het{s}|f") if fa else None
            out[f"g{g}"] = e
        if log is not None:
            log(f"x precheck records {v} done")
        return out
    key = dict(theta=w_oc.summary(theta), target=target, n_rep=n_rep)
    res = {v: cell(f"records_{v}", dict(key, variant=v), lambda vi=vi, v=v: one(vi, v))
           for vi, v in enumerate(xs.variants)}
    cals = {f"min|{t}": dict(ok=c["ok"], failure=c["failure"], retried=c["retried"]) for t, c in cmin.items()}
    cals[f"max|{xs.d_false}"] = dict(ok=cmax["ok"], failure=cmax["failure"], retried=cmax["retried"])
    return dict(target=target, variants=res, calibrations=cals, n_rep=n_rep, seed=int(root), tags=tags,
                note="사전 점검 STOP의 점 records(X.9.1.1) — 변형은 기록 전용(X.9.1.3 P3-12)")


# ================================================================ P1-5 residual comparison (records only)
def residual_compare(pilot: list, balanced, xs) -> dict:
    """Balanced pilot pairs vs the rest: per slot × cell × odour, the sd and quantiles of w_oc.fit's √(K/(K−1))
    residuals and the raw counts' zero share (the floor share)."""
    th = w_oc.fit(pilot)
    raw = w_oc._stack(pilot)
    J = raw.shape[0]
    res = np.stack(th["resid_blocks"])
    raw = raw.reshape(J, -1, *raw.shape[-3:])
    bal = np.asarray(balanced, bool)
    out = {}
    for name, mask in (("balanced", bal), ("rest", ~bal)):
        r, c = res[mask].reshape(-1, *res.shape[-3:]), raw[mask].reshape(-1, *raw.shape[-3:])
        grp = {}
        for si, s in enumerate(w_oc.SLOTS):
            for ci, cell in enumerate(("MBON13", "MBON05")):
                for oi, od in enumerate(("X", "Y")):
                    v = r[:, si, ci, oi]
                    grp[f"{s}|{cell}|{od}"] = dict(
                        sd=float(v.std(ddof=1)), n=int(len(v)), zero_share=float((c[:, si, ci, oi] == 0).mean()),
                        quantiles={str(q): float(x) for q, x in zip(xs.resid_quantiles,
                                                                    np.percentile(v, xs.resid_quantiles))})
        out[name] = grp
    return dict(groups=out, n_pairs=dict(balanced=int(bal.sum()), rest=int((~bal).sum())),
                note="판정·관문에 쓰지 않는다(X.9.1.3 P1-5); '균형 쌍 전용 파일럿'은 다음 결정 목록")
