"""Y's operating-characteristic pieces, phase A (Y.6.1–Y.6.3, Y.6.5 on a precheck STOP, Y.6.6, Y.3.4, Y.9.2 P1-2 ·
P1-3 · P1-4 · P1-5 · P2-10 · P2-12 · P3-14). W's model (w_oc) and X's machinery (x_oc, x_verdict) are imported and
never modified; Y adds only the following.
- Calibration (Y.6.2 + Y.9.2 P1-2): f(x) = agg(the knob's two gates' true d′)(x) − target (w_oc.true_dprimes, 2000
  fixed residual draws), reward a first, then punishment b in EVERY a state. x = 0: |f(0)| ≤ cal_tol → 0; f(0) >
  cal_tol → power above_at_zero (fill 0) / false zero_floor (0). Else the first crossing: coarse grid h = hi_W /
  grid_steps (hi_W = W's hi_a / hi_b, x_oc.brackets(θ, 1)) over ×1, then ×2, then ×4 — only the current stage —
  refining every coarse cell up to the coarse first crossing whose larger end is ≥ −refine_delta into refine_parts
  points, left to right (plan Reading 1: refinement, not breakpoint enumeration — Y.9.2 P1-2 (a), author's
  reading 9); the first refined point with f ≥ 0 starts a knob bisection keeping f(lo) < 0 ≤ f(hi) to a width ≤
  knob_tol. Two stairs per knob: w_hi = (target − d_lo) / (d_hi − d_lo), so the weighted mean true d′ is the target;
  four corners (a_s, b_t|s) with weights w_a,s · w_b,t|s. No crossing through ×4 → floor (power 0 / false 1). There is
  no no_convergence. coarse_step marks d_hi − d_lo > coarse_step_flag (mark only).
- Generator (Y.6.3, P3-14): x_oc.pair_bases with the Y accept predicate (balance stays inside pair_bases), tries
  2000; the near-threshold scenario (P1-4) shifts each accepted base per neuron to base[A, X] = c_A, base[P, X] = c_P
  (X and Y alike) and rejects a negative Y cell within the same rounds (fills stay the population pair, Reading 11).
- Simulation with the mixture: every experiment draws one corner per distinct calibration (a separate stream
  SeedSequence([root, "mix", draw, g, target, scenario]), Reading 2; the records insert their variant tags after
  draw, so they never reuse a precheck stream) and all its pairs and flies use it; the main
  stream is untouched, so a degenerate mixture with the filter off and 200 rounds equals x_oc.evaluate bit for bit.
- Fill share per k (first k pair slots); a k whose fill > fill_max in any (g, scenario, target) cell is excluded
  (P1-5). Precheck = point values, no envelope; qualify_y / select_y (limits, envelope, ×1.3 budget) are built for
  phase B and tested now (Y.6.6 mutations).
Every number comes from the YSpec passed in; every stream is SeedSequence([root, tag, …])."""
from __future__ import annotations

import dataclasses
import math
import time

import numpy as np

from . import w_oc
from . import w_verdict as WV
from . import x_oc
from . import x_verdict as XV
from . import y_rules

TAG_CAL, TAG_POINT, TAG_RECORD, TAG_SYNTH, TAG_P26 = (x_oc.TAG_CAL, x_oc.TAG_POINT, x_oc.TAG_RECORD, x_oc.TAG_SYNTH,
                                                       x_oc.TAG_P26)
TAG_BOOT, TAG_ACCEPT = w_oc.TAG_BOOT, x_oc.TAG_DIAG_ACCEPT
MODES = x_oc.MODES
SCENARIOS = ("base", "near")
OK, FLOOR, ABOVE, ZERO = "ok", "floor", "above_at_zero", "zero_floor"
rng = x_oc.rng
run_cell = x_oc.run_cell


def tag(name: str) -> int:
    """A string stream tag of Y.9.2 ("mix", "near", "thr", "R-V", "R-pre", …) as SeedSequence entropy (Reading 2)."""
    return int.from_bytes(name.encode("ascii"), "big")


# ================================================================ calibration (Y.6.2 + Y.9.2 P1-2)
def _weights(d_lo: float, d_hi: float, target: float) -> tuple:
    """(w_lo, w_hi) with w_lo · d_lo + w_hi · d_hi = target."""
    w_hi = (target - d_lo) / (d_hi - d_lo)
    return 1.0 - w_hi, w_hi


def _bisect_knob(f, lo: float, hi: float, f_lo: float, f_hi: float, ys) -> tuple:
    """Keep f(lo) < 0 ≤ f(hi) until hi − lo ≤ knob_tol (the tolerance is on the knob, not on f)."""
    while hi - lo > ys.knob_tol:
        mid = (lo + hi) / 2
        fm = f(mid)
        if fm < 0:
            lo, f_lo = mid, fm
        else:
            hi, f_hi = mid, fm
    return lo, hi, f_lo, f_hi


def _first_crossing(f, hi_w: float, f0: float, ys) -> dict:
    """Y.9.2 P1-2 (a) 1–5 (module docstring). Returns {found, lo, hi, f_lo, f_hi, stage, index (coarse cell),
    coarse_index (i* or None), refined_cells, refined_before_coarse} or {found False, refined_cells}."""
    h = hi_w / ys.grid_steps
    memo = {0: f0}

    def fx(i):
        if i not in memo:
            memo[i] = f(i * h)
        return memo[i]
    start, refined = 1, 0
    for mult in ys.widen:
        end = ys.grid_steps * mult
        i_star = next((i for i in range(start, end + 1) if fx(i) >= 0), None)
        last = end if i_star is None else i_star
        cells = [i for i in range(start, last + 1) if max(fx(i - 1), fx(i)) >= -ys.refine_delta]
        if i_star is not None and i_star not in cells:
            cells.append(i_star)
        for i in cells:
            refined += 1
            x0 = (i - 1) * h
            px, pf = x0, fx(i - 1)
            for j in range(1, ys.refine_parts + 1):
                x = i * h if j == ys.refine_parts else x0 + j * h / ys.refine_parts
                v = fx(i) if j == ys.refine_parts else f(x)
                if v >= 0:
                    return dict(found=True, lo=px, hi=x, f_lo=pf, f_hi=v, stage=mult, index=i, coarse_index=i_star,
                                refined_cells=refined, refined_before_coarse=bool(i_star is None or i < i_star))
                px, pf = x, v
        start = end + 1
    return dict(found=False, refined_cells=refined)


def calibrate_handle(f, hi_w: float, target: float, mode: str, ys) -> dict:
    """One knob: status ok · floor · above_at_zero · zero_floor and its stairs [{x, w, d}] (None on a failure)."""
    f0 = float(f(0.0))
    if abs(f0) <= ys.cal_tol:
        return dict(status=OK, stairs=[dict(x=0.0, w=1.0, d=f0 + target)], f0=f0)
    if f0 > 0:
        if mode == "max" and ys.cal_floor_rule == "zero":
            return dict(status=ZERO, stairs=[dict(x=0.0, w=1.0, d=f0 + target)], f0=f0)
        return dict(status=ABOVE, stairs=None, f0=f0)
    c = _first_crossing(f, hi_w, f0, ys)
    if not c["found"]:
        return dict(status=FLOOR, stairs=None, f0=f0, refined_cells=c["refined_cells"])
    lo, hi, f_lo, f_hi = _bisect_knob(f, c["lo"], c["hi"], c["f_lo"], c["f_hi"], ys)
    d_lo, d_hi = f_lo + target, f_hi + target
    w_lo, w_hi = _weights(d_lo, d_hi, target)
    step = d_hi - d_lo
    return dict(status=OK, stairs=[dict(x=lo, w=w_lo, d=d_lo), dict(x=hi, w=w_hi, d=d_hi)], f0=f0, stage=c["stage"],
                index=c["index"], coarse_index=c["coarse_index"], refined_cells=c["refined_cells"],
                refined_before_coarse=c["refined_before_coarse"], step=step, width=hi - lo,
                coarse_step=bool(round(step - ys.coarse_step_flag, ys.round_digits) > 0))


def _b_states(ra: dict) -> list:
    """The a states b is calibrated in: every stair of a (a_lo and a_hi, or the single x = 0 state)."""
    return ra["stairs"]


def calibrate_pair(fa, fb_of, hi_a: float, hi_b: float, target: float, mode: str, ys) -> dict:
    """a, then b in every a state (fb_of(a) is b's f in that state); corners with weights (sum 1). A failed knob in
    any state fails the whole calibration and fills (x_oc.fill_value)."""
    ra = calibrate_handle(fa, hi_a, target, mode, ys)
    out = dict(ok=False, a=ra, b=None, corners=None, fill=x_oc.fill_value(mode), failure=None)
    if ra["stairs"] is None:
        return dict(out, failure=dict(handle="a", status=ra["status"]))
    bs = []
    for s in _b_states(ra):
        rb = dict(calibrate_handle(fb_of(s["x"]), hi_b, target, mode, ys), a=s["x"])
        bs.append(rb)
        if rb["stairs"] is None:
            return dict(out, b=bs, failure=dict(handle="b", status=rb["status"], a=s["x"]))
    corners = [dict(a=s["x"], b=t["x"], w=s["w"] * t["w"], label=f"a{si}b{ti}")
               for si, (s, rb) in enumerate(zip(ra["stairs"], bs)) for ti, t in enumerate(rb["stairs"])]
    return dict(ok=True, a=ra, b=bs, corners=corners, fill=None, failure=None)


def calibrate_y(theta, target: float, mode: str, idx, z: dict, ys) -> dict:
    """calibrate_pair on θ: f_a(x) = agg(td(x, 0)[reward gates]) − target, f_b(x | a) = agg(td(a, x)[punishment
    gates]) − target, agg = min (power) / max (false pass), hi_W = W's brackets; plus each corner's four true d′."""
    agg = np.min if mode == "min" else np.max
    hi_a, hi_b = x_oc.brackets(theta, 1.0)

    def fa(x):
        return float(agg(w_oc.true_dprimes(theta, x, 0.0, idx, z)[[0, 2]]) - target)

    def fb_of(a):
        return lambda x: float(agg(w_oc.true_dprimes(theta, a, x, idx, z)[[1, 3]]) - target)
    c = calibrate_pair(fa, fb_of, hi_a, hi_b, target, mode, ys)
    if c["ok"]:
        c["true_dprime"] = [dict(zip(WV.GATES, w_oc.true_dprimes(theta, k["a"], k["b"], idx, z).tolist()), w=k["w"])
                            for k in c["corners"]]
    return c


def stair_corners(cal: dict) -> dict:
    """The record-only one-stair calibrations (Y.9.2 P1-2 (b) 기록): every knob lo (a_lo, b_lo|lo) and every knob hi
    (a_hi, b_hi|hi); each a one-corner mixture."""
    a, bs = cal["a"]["stairs"], cal["b"]
    return dict(lo=[dict(a=a[0]["x"], b=bs[0]["stairs"][0]["x"], w=1.0)],
                hi=[dict(a=a[-1]["x"], b=bs[-1]["stairs"][-1]["x"], w=1.0)])


def status_counts(rows: list) -> dict:
    """Per side: handle|status counts (b once per a state), coarse_step per handle, widening stage per handle, failed
    calibrations and the floor share (Y.6.2 기록, Y.9.2 P1-2 (c), P2-12)."""
    out = {}
    for m, _f in MODES:
        side = dict(handle_status={}, coarse_step={}, stage={}, fails=0, floor=0)
        for r in rows:
            c = r[m]
            for h, x in [("a", c["a"])] + [("b", x) for x in (c["b"] or [])]:
                k = f"{h}|{x['status']}"
                side["handle_status"][k] = side["handle_status"].get(k, 0) + 1
                if x.get("coarse_step"):
                    side["coarse_step"][h] = side["coarse_step"].get(h, 0) + 1
                if x.get("stage") is not None:
                    s = f"{h}|×{x['stage']}"
                    side["stage"][s] = side["stage"].get(s, 0) + 1
            if not c["ok"]:
                side["fails"] += 1
                side["floor"] += int(c["failure"]["status"] == FLOOR)
        side["floor_share"] = side["floor"] / len(rows) if rows else None
        out[m] = side
    return out


# ---------------------------------------------------------------- fixtures 1–5, 7, 8a (Y.6.6; P1-2 (d))
def _step(x_s, lo_v, hi_v):
    return lambda x: hi_v if x >= x_s else lo_v


def _check_two_stairs(c, f, target, ys) -> bool:
    """ok, two stairs straddling the target by re-evaluating f at the stair positions, width ≤ knob_tol, weights =
    the closed form and the weighted mean = target (fix_exact)."""
    if c["status"] != OK or c["stairs"] is None or len(c["stairs"]) != 2:
        return False
    lo, hi = c["stairs"]
    d_lo, d_hi = f(lo["x"]) + target, f(hi["x"]) + target
    w_hi = (target - d_lo) / (d_hi - d_lo)
    return bool(d_lo < target <= d_hi and hi["x"] - lo["x"] <= ys.knob_tol
                and abs(hi["w"] - w_hi) <= ys.fix_exact and abs(lo["w"] + hi["w"] - 1.0) <= ys.fix_exact
                and abs(lo["w"] * d_lo + hi["w"] * d_hi - target) <= ys.fix_exact)


def calibration_fixtures(ys) -> dict:
    """Y.6.6 fixtures 1–5 and Y.9.2 P1-2 (d) 7 · 8a on synthetic f (numbers from ys.fix_*); fixture 6 is
    w_failed_recal (record only), 8b / 9 are generator fixtures (generator_fixtures)."""
    t, hw, out = ys.fix_target, ys.fix_hi_w, {}
    x_s, lo_v, hi_v = ys.fix_step
    f1 = _step(x_s, lo_v, hi_v)
    c1, c1f = (calibrate_handle(f1, hw, t, m, ys) for m in ("min", "max"))
    out["f1_step"] = dict(ok=_check_two_stairs(c1, f1, t, ys) and _check_two_stairs(c1f, f1, t, ys)
                          and c1["stairs"] == c1f["stairs"], cal=c1)
    u1, d1, u2, d2 = ys.fix_cross
    s = ys.fix_slope

    def f2(x):
        if x < (u1 + d1) / 2:
            return s * (x - u1)
        if x < (d1 + u2) / 2:
            return s * (d1 - x)
        if x < (u2 + d2) / 2:
            return s * (x - u2)
        return s * (d2 - x)
    c2 = calibrate_handle(f2, hw, t, "min", ys)
    out["f2_nonmonotone"] = dict(ok=bool(c2["status"] == OK and c2.get("stage") == ys.widen[0]
                                        and c2.get("index") == int(round(u1 / (hw / ys.grid_steps)))
                                        and abs(c2["stairs"][-1]["x"] - u1) <= ys.knob_tol), cal=c2)
    x2 = ys.fix_x2

    def f3(x):
        return ys.fix_below if x < x2 else s * (x - x2)
    c3 = calibrate_handle(f3, hw, t, "min", ys)
    out["f3_x2_only"] = dict(ok=bool(c3["status"] == OK and c3.get("stage") == ys.widen[1]
                                    and abs(c3["stairs"][-1]["x"] - x2) <= ys.knob_tol), cal=c3)
    below = lambda x: ys.fix_below  # noqa: E731
    p4 = calibrate_pair(below, lambda a: below, hw, hw, t, "min", ys)
    q4 = calibrate_pair(below, lambda a: below, hw, hw, t, "max", ys)
    out["f4_none"] = dict(ok=bool(not p4["ok"] and p4["fill"] == 0.0 and p4["failure"]["status"] == FLOOR
                                  and not q4["ok"] and q4["fill"] == 1.0))
    above = lambda x: ys.fix_above  # noqa: E731
    p5, q5 = calibrate_handle(above, hw, t, "min", ys), calibrate_handle(above, hw, t, "max", ys)
    out["f5_above_zero"] = dict(ok=bool(p5["status"] == ABOVE and p5["stairs"] is None and q5["status"] == ZERO
                                        and q5["stairs"][0]["x"] == 0.0))
    xn, frac, v_in, v_out, xc, v_c = ys.fix_narrow
    h = hw / ys.grid_steps

    def f7(x):
        if xn <= x <= xn + h * frac:
            return v_in
        return v_c if x >= xc else v_out
    c7 = calibrate_handle(f7, hw, t, "min", ys)
    out["f7_narrow"] = dict(ok=bool(_check_two_stairs(c7, f7, t, ys) and xn - h / ys.refine_parts
                                    <= c7["stairs"][0]["x"] and c7["stairs"][-1]["x"] <= xn + h * frac
                                    and c7["refined_before_coarse"]), cal=c7)
    b_lo_s, b_hi_s, b_hi_lo, b_hi_hi = ys.fix_b_steps

    def fb_of(a):
        return _step(b_lo_s, lo_v, hi_v) if a < x_s else _step(b_hi_s, b_hi_lo, b_hi_hi)
    c8 = calibrate_pair(f1, fb_of, hw, hw, t, "min", ys)
    ok8 = c8["ok"] and abs(sum(k["w"] for k in c8["corners"]) - 1.0) <= ys.fix_exact
    if ok8:
        for s_, rb in zip(c8["a"]["stairs"], c8["b"]):
            fb = fb_of(s_["x"])
            mean_b = sum(st["w"] * (fb(st["x"]) + t) for st in rb["stairs"])
            ok8 = ok8 and abs(mean_b - t) <= ys.fix_mean
        mean_a = sum(st["w"] * (f1(st["x"]) + t) for st in c8["a"]["stairs"])
        ok8 = ok8 and abs(mean_a - t) <= ys.fix_mean
    out["f8a_two_knobs"] = dict(ok=bool(ok8), cal=c8)
    out["ok"] = all(v["ok"] for v in out.values() if isinstance(v, dict))
    return out


# ================================================================ the generator (Y.6.3, P3-14, P1-4)
def accept_y(ys, c_a=None, c_p=None):
    """Y.6.3's extra predicate on expected bases [..., 2, 2]: base[A, X] ≥ c_A ∧ base[P, X] ≥ c_P (rounded; balance
    stays inside x_oc.pair_bases)."""
    ca = ys.c_a if c_a is None else c_a
    cp = ys.c_p if c_p is None else c_p
    d = ys.round_digits

    def acc(base):
        b = np.asarray(base, float)
        return (np.round(b[..., WV.A, WV.X] - ca, d) >= 0) & (np.round(b[..., WV.P, WV.X] - cp, d) >= 0)
    return acc


def near_shift(base, ys):
    """Y.9.2 P1-4: per neuron, add c_A − base[A, X] to A's X and Y cells and c_P − base[P, X] to P's (so V's X − Y
    difference, the naive balance, is unchanged)."""
    b = np.asarray(base, float)
    out = b.copy()
    out[..., WV.A, :] = b[..., WV.A, :] + (ys.c_a - b[..., WV.A, WV.X])[..., None]
    out[..., WV.P, :] = b[..., WV.P, :] + (ys.c_p - b[..., WV.P, WV.X])[..., None]
    return out


def pair_bases_y(theta, rng_, n_rep: int, n_pair: int, g: float, z: dict, ys, scenario: str = "base",
                 accept=None) -> tuple:
    """x_oc.pair_bases (tries = ys.tries) with the Y predicate; "near" also rejects a negative shifted Y cell (same
    rounds) and shifts the accepted bases. Returns (bases [n_rep, n_pair, 2, 2], filled [n_rep, n_pair])."""
    acc = accept_y(ys) if accept is None else accept
    if scenario == "near":
        inner = acc

        def acc(base):
            s = near_shift(base, ys)
            return inner(base) & (s[..., WV.A, WV.Y] >= 0) & (s[..., WV.P, WV.Y] >= 0)
    base, filled = x_oc.pair_bases(theta, rng_, n_rep, n_pair, g, z, ys, ys.tries, acc)
    if scenario == "near":
        base = np.where(filled[..., None, None], base, near_shift(base, ys))
    return base, filled


def simulate_y(theta, rng_, n_rep, n_pair, n_fly, n_probe, a_exp, b_exp, fly_a, fly_b, g, z, ys, scenario="base",
               accept=None) -> tuple:
    """x_oc.simulate's draws in its order with per-experiment knob values a_exp / b_exp [n_rep, n_pair] (the drawn
    corners): (stages, bases, filled)."""
    base, filled = pair_bases_y(theta, rng_, n_rep, n_pair, g, z, ys, scenario, accept)
    Lf = np.linalg.cholesky(theta["fly_cov"] + ys.chol_jitter * np.eye(4))
    v = (rng_.standard_normal((n_rep, n_pair, n_fly, 4)) @ Lf.T).reshape(n_rep, n_pair, n_fly, 2, 2)
    a = np.asarray(a_exp, float)[:, :, None] * np.asarray(fly_a, float)[None, None, :]
    b = np.asarray(b_exp, float)[:, :, None] * np.asarray(fly_b, float)[None, None, :]
    mu = w_oc.slot_means(theta, base[:, :, None], v, a, b)
    idx = rng_.integers(0, len(theta["resid"]), (n_rep, n_pair, n_fly, n_probe))
    counts = np.clip(np.rint(mu[:, :, :, None] + theta["resid"][idx]), 0, None).astype(np.int32)
    return w_oc.to_stages(counts), base, filled


def one_corner(a: float, b: float) -> list:
    return [dict(a=float(a), b=float(b), w=1.0)]


def _distinct(pair_mix: list) -> tuple:
    keys, which = [], []
    for m in pair_mix:
        k = tuple((float(c["a"]), float(c["b"]), float(c["w"])) for c in m)
        if k not in keys:
            keys.append(k)
        which.append(keys.index(k))
    return keys, which


def _corner_effects(mix_rng, n: int, n_pair: int, mixes: list, which: list) -> tuple:
    """One corner per experiment per distinct mixture (Y.9.2 P1-2 (b)); a / b [n, n_pair]."""
    picks = []
    for m in mixes:
        w = np.asarray([c[2] for c in m], float)
        picks.append(mix_rng.choice(len(m), size=n, p=w / w.sum()))
    a, b = np.empty((n, n_pair)), np.empty((n, n_pair))
    for j in range(n_pair):
        m, pk = mixes[which[j]], picks[which[j]]
        a[:, j] = np.asarray([c[0] for c in m])[pk]
        b[:, j] = np.asarray([c[1] for c in m])[pk]
    return a, b


def evaluate_y(theta, rng_, mix_rng, n_rep: int, pair_mix: list, fly_a, fly_b, g: float, z: dict, ys,
               scenario: str = "base", accept=None) -> dict:
    """P(PASS) [p_set, q, K, F, k_min..k_cap] under W's pair verdict and X's set rule (x_oc.tally; b = 0 in
    simulation) on k_cap pairs × f_max flies × 2·max K probes per experiment (nested, common random numbers); pair j
    uses pair_mix[j]'s drawn corner. fill_by_k = the filled share of the first k pair slots."""
    fs = list(range(ys.f_min, ys.f_max + 1))
    kk = list(range(ys.k_min, ys.k_cap + 1))
    hits = np.zeros(x_oc.grid_shape(ys))
    filled = np.zeros(ys.k_cap)
    mixes, which = _distinct(pair_mix)
    done = 0
    while done < n_rep:
        n = min(ys.oc_chunk, n_rep - done)
        a, b = _corner_effects(mix_rng, n, ys.k_cap, mixes, which)
        d, _base, fill = simulate_y(theta, rng_, n, ys.k_cap, ys.f_max, 2 * max(ys.k_grid), a, b, fly_a, fly_b, g,
                                    z, ys, scenario, accept)
        hits += x_oc.tally(d, z, ys, ys.q_grid, ys.k_grid, fs, kk)
        filled += fill.sum(0)
        done += n
    cum = np.cumsum(filled)[ys.k_min - 1:]
    return dict(p=hits / n_rep, fill_by_k=(cum / (n_rep * np.asarray(kk, float))).tolist())


# ================================================================ pass rule, qualification, selection, picks
def _range_ks(kr, ys) -> list:
    """The k-axis positions of every k in k range kr."""
    return [k - ys.k_min for k in range(kr[0], kr[1] + 1)]


def _qual_ranges(k_ranges) -> list:
    return list(k_ranges)


def range_ok(power, false, fill_bad, ys, kr, envelope: bool) -> np.ndarray:
    """[p, q, K, F]: both targets at EVERY k of kr (x_oc.meets, rounded), no fill-excluded k (P1-5); with envelope
    (Y.5, W.9.8 H1) F … F + envelope all meet, F ≥ envelope_solo_from alone."""
    ks = _range_ks(kr, ys)
    ok = x_oc.meets(np.asarray(power)[..., ks], np.asarray(false)[..., ks], ys).all(-1)
    ok = ok & ~np.asarray(fill_bad, bool)[ks].any()
    if not envelope:
        return ok
    out = np.zeros_like(ok)
    nf = ok.shape[-1]
    for fi in range(nf):
        hi = fi + 1 if ys.f_min + fi >= ys.envelope_solo_from else min(nf, fi + ys.envelope + 1)
        out[..., fi] = ok[..., fi:hi].all(-1)
    return out


def qualify_y(power_lo, false_hi, fill_bad, ys, k_ranges) -> dict:
    """Y.5 자격 (phase B's bootstrap limits; tested now): {k range: ok [p, q, K, F]} with the envelope."""
    return {tuple(kr): range_ok(power_lo, false_hi, fill_bad, ys, kr, True) for kr in _qual_ranges(k_ranges)}


def _in_budget(r: dict, elapsed_h: float, ys) -> bool:
    return round(elapsed_h + ys.cost_margin * r["cost_h"] - ys.budget_h, ys.round_digits) <= 0


def select_y(qual: dict, cost_h, elapsed_h: float, ys) -> dict:
    """Y.5 선택 (phase B; tested now): qualifying designs in budget (elapsed + cost_margin × the design's worst total
    estimate without C ≤ budget_h; Y.9.2 P2-9, author's readings 4 · 8), ordered larger p_set → smaller cost → larger
    q → smaller K → smaller F → smaller k_lo; none qualifying → STOP_OC_UNREACHABLE, none in budget → STOP_BUDGET."""
    fs = list(range(ys.f_min, ys.f_max + 1))
    rows = []
    for kr, ok in qual.items():
        for i in np.ndindex(ok.shape):
            if ok[i]:
                dsg = dict(p_set=ys.p_set_grid[i[0]], q=ys.q_grid[i[1]], K=ys.k_grid[i[2]], F=fs[i[3]],
                           k_range=[int(kr[0]), int(kr[1])])
                rows.append(dict(dsg, cost_h=float(cost_h(dsg))))
    within = sorted((r for r in rows if _in_budget(r, elapsed_h, ys)),
                    key=lambda r: (-r["p_set"], r["cost_h"], -r["q"], r["K"], r["F"], r["k_range"][0]))
    outcome = y_rules.PASS if within else ("STOP_BUDGET" if rows else y_rules.STOP_OC_UNREACHABLE)
    return dict(outcome=outcome, selected=within[0] if within else None, ranking=within, n_qualifying=len(rows),
                n_in_budget=len(within))


def _design_rows(power, false, ys, k_ranges) -> list:
    P, Fa = np.asarray(power), np.asarray(false)
    fs = list(range(ys.f_min, ys.f_max + 1))
    d = ys.round_digits
    rows = []
    for kr in k_ranges:
        ks = _range_ks(kr, ys)
        for i in np.ndindex(P.shape[:-1]):
            pw, fp = P[i][ks], Fa[i][ks]
            rows.append(dict(index=[int(v) for v in i], p_set=ys.p_set_grid[i[0]], q=ys.q_grid[i[1]],
                             K=ys.k_grid[i[2]], F=int(fs[i[3]]), k_range=[int(kr[0]), int(kr[1])],
                             k=list(range(kr[0], kr[1] + 1)), power_by_k=pw.tolist(), false_by_k=fp.tolist(),
                             pmin=round(float(pw.min()), d), fmax=round(float(fp.max()), d)))
    return rows


def _tie(r: dict) -> tuple:
    """Y.5's order without cost: larger p_set, larger q, smaller K, smaller F, smaller k_lo (Y.8, Y.6.5)."""
    return (-r["p_set"], -r["q"], r["K"], r["F"], r["k_range"][0])


def pick_y(power, false, ys, k_ranges, require_false: bool) -> dict:
    """require_false False: Y.8's "점 검정력 최대 설계" (largest own-range min power). True: Y.6.5's records target
    without a selected design — among designs with own-range max false ≤ p_false the largest min power, none → the
    smallest max false (then power). Ties by _tie."""
    rows = _design_rows(power, false, ys, k_ranges)
    if not require_false:
        return dict(min(rows, key=lambda r: (-r["pmin"],) + _tie(r)), rule="max_power")
    ok = [r for r in rows if round(r["fmax"] - ys.p_false, ys.round_digits) <= 0]
    if ok:
        return dict(min(ok, key=lambda r: (-r["pmin"],) + _tie(r)), rule="false_ok_max_power")
    return dict(min(rows, key=lambda r: (r["fmax"], -r["pmin"]) + _tie(r)), rule="min_false")


def table_at_f_y(power, false, ys, k_ranges, F: int) -> list:
    """Y.8's table: per (p_set, q, K, k range) the k-wise point power / false at one F."""
    fi = F - ys.f_min
    out = []
    for kr in k_ranges:
        ks = _range_ks(kr, ys)
        for pi, p in enumerate(ys.p_set_grid):
            for qi, q in enumerate(ys.q_grid):
                for ki, K in enumerate(ys.k_grid):
                    out.append(dict(p_set=p, q=q, K=K, F=int(F), k_range=[int(kr[0]), int(kr[1])],
                                    k=list(range(kr[0], kr[1] + 1)),
                                    power=np.asarray(power)[pi, qi, ki, fi][ks].tolist(),
                                    false=np.asarray(false)[pi, qi, ki, fi][ks].tolist()))
    return out


# ================================================================ Y.6.6: synthetic validation, bit identity, fixtures
def _by_range(arr, ys, fn) -> dict:
    return {f"[{lo}, {hi}]": float(fn(np.asarray(arr)[..., _range_ks((lo, hi), ys)])) for lo, hi in ys.k_ranges}


def synthetic_validation(ys, z: dict, n_rep: int | None = None) -> dict:
    """W.9.9 P2-11's five fixtures on Y's evaluate and calibration, every p_set and both k ranges (k_min..k_cap):
    zero effect ≤ synth_null_max, big effect (d′ synth_big_dprime, Y's mixture) ≥ synth_big_min, one gate only ≤,
    negative fly correlation ≤, and the simple normal model not rising from F 8 to 32. Root oc_seed, tag TAG_SYNTH."""
    n_rep = ys.synth_reps if n_rep is None else int(n_rep)
    r, mr = rng(ys.oc_seed, TAG_SYNTH), rng(ys.oc_seed, tag("mix"), TAG_SYNTH)
    kw = dict(base=ys.synth_base, sd=ys.synth_sd, corr=ys.synth_corr, learn=ys.synth_learn)
    th0 = w_oc.fit(w_oc.synthetic_pilot(r, **kw))
    th_drift = w_oc.fit(w_oc.synthetic_pilot(r, drift_ax=ys.synth_drift, **kw))
    idx = r.integers(0, len(th0["resid"]), ys.cal_reps)
    ones, zero = [1.0] * ys.f_max, [one_corner(0.0, 0.0)] * ys.k_cap

    def ev(th, mix, fa=ones, fb=ones):
        return evaluate_y(th, r, mr, n_rep, mix, fa, fb, 0.0, z, ys)["p"]
    lo, hi, res = ys.synth_null_max, ys.synth_big_min, {}
    z0 = ev(th0, zero)
    res["zero_effect"] = dict(max_p=float(z0.max()), by_range=_by_range(z0, ys, np.max), limit=lo,
                              ok=bool(z0.max() <= lo))
    big = calibrate_y(th0, ys.synth_big_dprime, "min", idx, z, ys)
    pb = ev(th0, [big["corners"]] * ys.k_cap) if big["ok"] else np.zeros(x_oc.grid_shape(ys))
    res["big_effect"] = dict(min_p=float(pb.min()), by_range=_by_range(pb, ys, np.min), limit=hi,
                             ok=bool(big["ok"] and pb.min() >= hi), true_dprime=big.get("true_dprime"))
    one = ev(th_drift, zero)
    td = w_oc.true_dprimes(th_drift, 0.0, 0.0, r.integers(0, len(th_drift["resid"]), ys.cal_reps), z)
    res["one_gate"] = dict(max_p=float(one.max()), limit=lo,
                           ok=bool(one.max() <= lo and td[1] >= ys.synth_drift_dprime_min),
                           true_dprime=dict(zip(WV.GATES, td.tolist())))
    if big["ok"]:
        alt = [1.0 if i % 2 == 0 else 0.0 for i in range(ys.f_max)]
        neg = ev(th0, [big["corners"]] * ys.k_cap, alt, [1.0 - x for x in alt])
        res["negative_correlation"] = dict(max_p=float(neg.max()), limit=lo, ok=bool(neg.max() <= lo))
    else:
        res["negative_correlation"] = dict(max_p=None, ok=False)
    sn = w_oc.simple_normal(ys, r)
    p = np.array([sn[F] for F in ys.simple_normal_fs])
    res["simple_normal"] = dict(p_by_F={str(k): v for k, v in sn.items()}, diffs=np.diff(p).tolist(),
                                tol=w_oc.SIMPLE_NORMAL_TOL, ok=bool(np.all(np.diff(p) <= w_oc.SIMPLE_NORMAL_TOL)))
    res["ok"] = all(v["ok"] for v in res.values() if isinstance(v, dict))
    return res


def bit_identity(theta, z: dict, ys, xs, n_rep: int, a: float, b: float) -> dict:
    """Y.6.6 (X.9.1.3 P2-6 extended): filter off (c_A = c_P = −∞), w_rejection_tries rounds, k range [4, 8] (X's
    k_cap), knob values fixed, a one-corner mixture → Y's evaluate equals x_oc.evaluate (numpy.array_equal over
    [p_set, q, K, F, k]) at every g; same θ, same stream (root oc_seed, TAG_P26, g index)."""
    yb = dataclasses.replace(ys, k_cap=xs.k_cap, tries=ys.w_rejection_tries, c_a=-math.inf, c_p=-math.inf)
    ones, by_g = [1.0] * yb.f_max, {}
    for gi, g in enumerate(ys.cluster_grid):
        x = x_oc.evaluate(theta, rng(ys.oc_seed, TAG_P26, gi), n_rep, *w_oc._uniform(xs, a, b), g, z, xs)
        y = evaluate_y(theta, rng(ys.oc_seed, TAG_P26, gi), rng(ys.oc_seed, tag("mix"), TAG_P26, gi), n_rep,
                       [one_corner(a, b)] * yb.k_cap, ones, ones, g, z, yb)["p"]
        by_g[f"g{g}"] = dict(equal=bool(np.array_equal(x, y)), max_abs_diff=float(np.abs(x - y).max()))
    return dict(equal=all(v["equal"] for v in by_g.values()), by_g=by_g, n_rep=int(n_rep), a=float(a), b=float(b))


def _synth_theta(ys, name: str):
    return w_oc.fit(w_oc.synthetic_pilot(rng(ys.oc_seed, tag(name)), base=ys.synth_base, sd=ys.synth_sd,
                                         corr=ys.synth_corr, learn=ys.synth_learn))


def generator_fixtures(ys, z: dict) -> dict:
    """Y.6.6 + Y.9.2 P1-2 (d) 8b / 9 and the code fixtures the mutations need: 8b — a two-corner mixture's P(PASS) at
    mix_reps equals the corners' weighted P(PASS) within fix_se standard errors at every k (fix_design); 9 — the near
    shift keeps |Δd′(base)| ≤ fix_exact and sets the X levels; the accept predicate's three cases; qualify over both
    k ranges and every k of each; selection in budget; X.4.6's set fixtures and Y.5's m table on k 4–10."""
    out = {}
    th = _synth_theta(ys, "fix8")
    p_set, q, K, F = ys.fix_design
    yd = x_oc.design_spec(ys, dict(p_set=p_set, q=q, K=K, F=F))
    idx = rng(ys.oc_seed, tag("fix8"), TAG_CAL).integers(0, len(th["resid"]), ys.cal_reps)
    big = calibrate_y(th, ys.synth_big_dprime, "min", idx, z, ys)
    _x_s, lo_v, hi_v = ys.fix_step
    w_hi = (0.0 - lo_v) / (hi_v - lo_v)
    ok8 = big["ok"]
    if ok8:
        k0 = big["corners"][0]
        mix = [dict(a=0.0, b=0.0, w=1.0 - w_hi), dict(a=k0["a"], b=k0["b"], w=w_hi)]
        ones, n = [1.0] * yd.f_max, ys.mix_reps

        def ev(m, t):
            return np.asarray(evaluate_y(th, rng(ys.oc_seed, tag("fix8"), t),
                                         rng(ys.oc_seed, tag("mix"), tag("fix8"), t), n, [m] * yd.k_cap, ones, ones,
                                         0.0, z, yd)["p"][0, 0, 0, 0])
        pm, p0, p1 = ev(mix, 0), ev([dict(mix[0], w=1.0)], 1), ev([dict(mix[1], w=1.0)], 2)
        want = mix[0]["w"] * p0 + mix[1]["w"] * p1
        se = np.sqrt(pm * (1 - pm) / n + mix[0]["w"] ** 2 * p0 * (1 - p0) / n + mix[1]["w"] ** 2 * p1 * (1 - p1) / n)
        ok8 = bool(np.all(np.abs(pm - want) <= ys.fix_se * np.maximum(se, 1.0 / n)))
        out["f8b_mixture_mc"] = dict(ok=ok8, p_mix=pm.tolist(), p_weighted=want.tolist())
    else:
        out["f8b_mixture_mc"] = dict(ok=False, reason="calibration failed")
    th9 = _synth_theta(ys, "fix9")
    raw, filled = x_oc.pair_bases(th9, rng(ys.oc_seed, tag("fix9")), ys.oc_chunk, ys.k_cap, 0.0, z, ys, ys.tries,
                                  accept_y(ys))
    sh = near_shift(raw, ys)
    dd = float(np.abs(WV.dv(sh, z) - WV.dv(raw, z)).max() / w_oc.sd_pre(th9, z))
    keep = ~filled
    out["f9_near_shift"] = dict(ok=bool(dd <= ys.fix_exact and np.all(np.abs(sh[keep][:, WV.A, WV.X] - ys.c_a)
                                                                         <= ys.fix_mean)
                                        and np.all(np.abs(sh[keep][:, WV.P, WV.X] - ys.c_p) <= ys.fix_mean)),
                                max_dprime_change=dd)
    bases = np.zeros((3, 2, 2))
    bases[:, WV.A, WV.X], bases[:, WV.P, WV.X] = ys.c_a, ys.c_p
    bases[1, WV.A, WV.X], bases[2, WV.P, WV.X] = ys.c_a - 1, ys.c_p - 1
    out["accept"] = dict(ok=accept_y(ys)(bases).tolist() == [True, False, False])
    shp = x_oc.grid_shape(ys)
    pw, fp = np.ones(shp), np.zeros(shp)
    pw[0, 0, 0, 0, 0] = 0.0                                          # design 0 fails at k = k_min only
    qual = qualify_y(pw, fp, np.zeros(shp[-1], bool), ys, ys.k_ranges)
    lo_r, hi_r = (tuple(kr) for kr in ys.k_ranges)
    out["qualify_ranges"] = dict(ok=bool(set(qual) == {lo_r, hi_r} and not qual[lo_r][0, 0, 0, 0]
                                         and qual[hi_r][0, 0, 0, 0]))
    sel_q = {lo_r: np.zeros(shp[:-1], bool)}
    sel_q[lo_r][len(ys.p_set_grid) - 1, 0, 0, 0] = True              # p_set 1.0, cost over budget
    sel_q[lo_r][0, 0, 0, 0] = True                                   # p_set 0.5, cost 1 h
    s = select_y(sel_q, lambda d: ys.budget_h if d["p_set"] == ys.p_set_grid[-1] else 1.0, 0.0, ys)
    out["select_budget"] = dict(ok=bool(s["selected"] is not None and s["selected"]["p_set"] == ys.p_set_grid[0]
                                        and s["n_in_budget"] == 1))
    codes = {"PASS": XV.S_PASS, "FAIL": XV.S_FAIL, "UNDECIDED": XV.S_UNDECIDED}
    mt = XV.m_needed_table(ys)
    out["set_rule"] = dict(ok=bool(all(int(XV.set_code_counts(n_, m_, b_, False, p_, ys)) == codes[w]
                                       for (n_, m_, b_, p_), w in ys.set_fixtures)
                                   and mt["k"] == list(range(ys.k_min, ys.k_cap + 1))
                                   and all(mt["m"][str(p_)] == list(row) for p_, row in ys.m_table)),
                           m_needed=mt)
    out["ok"] = all(v["ok"] for v in out.values() if isinstance(v, dict))
    return out


# ================================================================ thresholds (Y.0 definition; Y.6.6 repro, P2-10)
def floor_threshold(theta, a: float, b: float, cell: str, ys, resid=None) -> float:
    """Y.0: the smallest naive level L on the thr_step grid (from 0) with share(rint(μ(L) + r) < 0) < floor_share,
    μ(L) = the slot mean at base level L under a, b — "A": R2 · MBON13(X) (base + drift − b), "P": R1 · MBON05(X)
    (base + drift − a); r = the residual vectors' entry (all of θ's, or `resid`). Plan Reading 3."""
    s, ci = (w_oc.SI["R2"], WV.A) if cell == "A" else (w_oc.SI["R1"], WV.P)
    res = theta["resid"] if resid is None else np.asarray(resid)
    off = float(w_oc.slot_means(theta, np.zeros((2, 2)), np.zeros((2, 2)), a, b)[s, ci, WV.X])
    r = res[:, s, ci, WV.X]
    n = 0
    while round(float((np.rint(n * ys.thr_step + off + r) < 0).mean()) - ys.floor_share, ys.round_digits) >= 0:
        n += 1
    return n * ys.thr_step


def thresholds(theta, a: float, b: float, ys, resid=None) -> dict:
    """(c_A, c_P) computed, their integers (the smallest integer above, Y.0 author's reading 1), and whether they
    match Y's declared 20 · 43 and Y.0's printed 19.25 / 42.0 — a record; the filter never changes (P0-1)."""
    ca = floor_threshold(theta, a, b, "A", ys, resid)
    cp = floor_threshold(theta, a, b, "P", ys, resid)
    ia, ip = math.floor(ca) + 1, math.floor(cp) + 1
    return dict(c_A_calc=ca, c_P_calc=cp, c_A_int=ia, c_P_int=ip, a=float(a), b=float(b),
                n_resid=int(len(theta["resid"]) if resid is None else len(resid)),
                matches_declared=bool((ia, ip) == (ys.c_a, ys.c_p)),
                matches_expected=bool((ca, cp) == tuple(ys.thr_expected)),
                note="기록 전용 — 값이 어떻든 c_A 20 · c_P 43은 바뀌지 않는다(Y.6.6, Y.9.2 P0-1)")


# ================================================================ Y.3.4 filter-candidate comparison (record only)
CANDIDATES = ("a", "b", "c", "d", "e", "f")
CANDIDATE_LABELS = dict(a="균형만", b="Y 거름", c="F2(0)", d="F2(25)", e="사용자 31 단독", f="사용자 한쪽 + 31")


def candidate(name: str, theta, z: dict, ys) -> tuple:
    """(spec, accept, label) of Y.3.4's candidate inside the simulation (on the expected base, like W / X): (a)–(e)
    keep the balance inside pair_bases and add floors; (f) turns the balance off (naive_max ∞) and accepts
    d′(base) > user_onesided_d ∧ base[A, X] ≥ user_a (Reading 18)."""
    inf = math.inf
    floors = dict(a=(-inf, -inf), b=(ys.c_a, ys.c_p), c=tuple(ys.f2_0), d=tuple(ys.f2_25), e=(ys.user_a, -inf))
    if name in floors:
        return ys, accept_y(ys, *floors[name]), CANDIDATE_LABELS[name]
    sdp, d = w_oc.sd_pre(theta, z), ys.round_digits

    def onesided(base):
        b = np.asarray(base, float)
        return ((np.round(WV.dv(b, z) / sdp - ys.user_onesided_d, d) > 0)
                & (np.round(b[..., WV.A, WV.X] - ys.user_a, d) >= 0))
    return dataclasses.replace(ys, naive_max=inf), onesided, CANDIDATE_LABELS[name]


def candidate_passes(name: str, d: float, la: float, lp: float, ys) -> bool:
    """The same candidate on measured values (W pilot naive d′ and medians): pass counts (Y.3.4 보고)."""
    r = lambda x: round(float(x), ys.round_digits)  # noqa: E731
    bal = abs(r(d)) < ys.naive_max
    if name == "a":
        return bool(bal)
    if name == "b":
        return bool(y_rules.filter_passes(d, la, lp, ys))
    if name in ("c", "d"):
        ca, cp = ys.f2_0 if name == "c" else ys.f2_25
        return bool(bal and r(la) >= ca and r(lp) >= cp)
    if name == "e":
        return bool(bal and r(la) >= ys.user_a)
    return bool(r(d) > ys.user_onesided_d and r(la) >= ys.user_a)


def compare_filters(theta, vals: dict, z: dict, ys, n_rep: int | None = None, cell=run_cell, log=None) -> dict:
    """Y.3.4 on W θ̂ (V0): Y's calibration (root compare_seed), per candidate × g × target compare_reps experiments
    (tries 2000, fill recorded), worst over g; per k range the best design (pick_y with the false target), its g 0
    values and k-wise point power / false, fill per cell (flag only), the acceptance rate on unconditioned g 0 bases
    and the pass counts among the W pilot pairs (vals {key: (naive d′, MBON13(X), MBON05(X))}) and the balanced three.
    Record only — never a gate, a STOP or a selection."""
    t0 = time.perf_counter()
    n_rep = ys.compare_reps if n_rep is None else int(n_rep)
    root = ys.compare_seed
    idx = rng(root, TAG_CAL).integers(0, len(theta["resid"]), ys.cal_reps)
    cal = {m: calibrate_y(theta, getattr(ys, f), m, idx, z, ys) for m, f in MODES}
    shp = x_oc.grid_shape(ys)
    ones = [1.0] * ys.f_max
    key0 = dict(theta=w_oc.summary(theta), n_rep=n_rep, root=root, cal={m: c["corners"] for m, c in cal.items()},
                z={k: list(v) for k, v in z.items()})                # z_V enters evaluate_y
    L = np.linalg.cholesky(theta["pair_cov"] + ys.chol_jitter * np.eye(4))
    raw = theta["m0s"] + (rng(root, TAG_ACCEPT).standard_normal((n_rep * ys.k_cap, 4)) @ L.T).reshape(-1, 2, 2)
    sdp = w_oc.sd_pre(theta, z)
    out = {}
    for ci, name in enumerate(CANDIDATES):
        spec_c, acc, label = candidate(name, theta, z, ys)
        grid, fill = {}, {}
        for gi, g in enumerate(ys.cluster_grid):
            for mi, (m, _f) in enumerate(MODES):
                c = cal[m]
                if not c["ok"]:
                    grid[(g, m)], fill[f"g{g}|{m}"] = np.full(shp, c["fill"]), None
                    continue

                def run(c=c, gi=gi, g=g, mi=mi, spec_c=spec_c, acc=acc, ci=ci):
                    r = evaluate_y(theta, rng(root, TAG_POINT, ci, gi, mi), rng(root, tag("mix"), 0, gi, mi, ci),
                                   n_rep, [c["corners"]] * ys.k_cap, ones, ones, g, z, spec_c, "base", acc)
                    return dict(p=r["p"].tolist(), fill_by_k=r["fill_by_k"])
                v = cell(f"cmp_{name}_g{g}_{m}", dict(key0, candidate=name, g=g, mode=m), run)
                grid[(g, m)], fill[f"g{g}|{m}"] = np.asarray(v["p"]), v["fill_by_k"]
                if log is not None:
                    log(f"y compare {name} g {g} {m} done ({time.perf_counter() - t0:.0f} s)")
        pw = np.min([grid[(g, "min")] for g in ys.cluster_grid], 0)
        fp = np.max([grid[(g, "max")] for g in ys.cluster_grid], 0)
        g0 = ys.cluster_grid[0]
        by_range = {}
        for kr in ys.k_ranges:
            best = pick_y(pw, fp, ys, [kr], True)
            i = tuple(best["index"])
            ks = _range_ks(kr, ys)
            by_range[f"[{kr[0]}, {kr[1]}]"] = dict(best=best, best_g0=dict(
                power_by_k=grid[(g0, "min")][i][ks].tolist(), false_by_k=grid[(g0, "max")][i][ks].tolist()))
        bal = np.abs(WV.dv(raw, z)) / sdp < spec_c.naive_max
        keys = list(vals)
        out[name] = dict(label=label, by_range=by_range, fill=fill,
                         fill_flag=any(v is not None and round(max(v) - ys.fill_max, ys.round_digits) > 0
                                       for v in fill.values()),
                         acceptance=float((bal & acc(raw)).mean()),
                         pass_counts=dict(pilot=sum(candidate_passes(name, *vals[k], ys) for k in keys),
                                          balanced=sum(candidate_passes(name, *vals[k], ys) for k in keys
                                                       if k in ys.pilot_w_pairs)),
                         arrays=dict(power_worst=pw.tolist(), false_worst=fp.tolist()))
    return dict(candidates=out, calibration=cal, n_rep=n_rep, seed=root, timing_s=time.perf_counter() - t0,
                note="모형 안의 거름은 기대값 base에 적용, 실제 거름은 측정 중앙값에 적용(Y.3.4) — 기록 전용")


def compare_summary(cmp: dict) -> dict:
    """The block's part of Y.3.4: per candidate and k range the best design, its k-wise values (worst g and g 0), the
    fill flag, acceptance and pass counts (arrays stay in the detail file)."""
    return dict(candidates={n: dict(label=c["label"], fill_flag=c["fill_flag"], acceptance=c["acceptance"],
                                    pass_counts=c["pass_counts"],
                                    by_range={r: dict(v, fill_flag=c["fill_flag"]) for r, v in c["by_range"].items()})
                            for n, c in cmp["candidates"].items()},
                calibration={m: dict(ok=c["ok"], failure=c["failure"]) for m, c in cmp["calibration"].items()},
                n_rep=cmp["n_rep"], seed=cmp["seed"], timing_s=cmp["timing_s"], note=cmp["note"])


# ================================================================ fixture 6 and the OC compute time
def w_failed_recal(theta_w, w_boot_cal: list, z: dict, w_spec, ys, log=None) -> dict:
    """Y.6.6 fixture 6 (record only): W's failed bootstrap draws (either side not ok in results/w/oc.json) re-drawn
    from W's root in W's order (w_oc._rng(w_spec, TAG_BOOT, bi) → boot → calibration draws, as
    x_oc.w_cal_diagnosis) and calibrated on both sides with Y's rule; the status table."""
    failed = [i for i, c in enumerate(w_boot_cal) if not (c["min"]["ok"] and c["max"]["ok"])]
    rows = []
    for bi in failed:
        rb = w_oc._rng(w_spec, w_oc.TAG_BOOT, bi)
        tb = w_oc.boot(theta_w, rb)
        ib = rb.integers(0, len(tb["resid"]), w_spec.cal_reps)
        row = dict(draw=bi, w=dict(min=w_boot_cal[bi]["min"]["ok"], max=w_boot_cal[bi]["max"]["ok"]))
        for m, f in MODES:
            row[m] = calibrate_y(tb, getattr(w_spec, f), m, ib, z, ys)
        rows.append(row)
        if log is not None:
            log(f"y fixture 6 draw {bi} done")
    return dict(rows=rows, failed=failed, n_failed=len(failed), counts=status_counts(rows),
                note="기록 전용(Y.6.6 보정 픽스처 6) — W θ̂의 X.10 (가) 실패 회차를 Y 규칙으로 다시 보정")


def oc_timing(theta, z: dict, ys, a: float, b: float) -> dict:
    """Y.7 0: one Y evaluate at boot_reps experiments (fixed corner, last g), scaled to the precheck (g × targets ×
    scenarios × (mixture + two stairs) × precheck_reps) and to phase B's bootstrap (boot_draws × g × targets ×
    scenarios × boot_reps) and one reconfirmation (boot_draws × … × reconfirm_reps)."""
    ones = [1.0] * ys.f_max
    t0 = time.perf_counter()
    evaluate_y(theta, rng(ys.oc_seed, TAG_P26, len(ys.cluster_grid)), rng(ys.oc_seed, tag("mix"), TAG_P26), ys.boot_reps,
               [one_corner(a, b)] * ys.k_cap, ones, ones, ys.cluster_grid[-1], z, ys)
    s = time.perf_counter() - t0
    cells = len(ys.cluster_grid) * len(MODES) * len(SCENARIOS)
    return dict(evaluate_s=s, reps=ys.boot_reps,
                precheck_point_s=s * cells * (1 + 2) * ys.precheck_reps / ys.boot_reps,
                phase_b_boot_s=s * cells * ys.boot_draws,
                reconfirm_s=s * cells * ys.boot_draws * ys.reconfirm_reps / ys.boot_reps)


# ================================================================ the pilot θ (Y.4, Y.9.2 P1-3) and its bootstrap draw
def r_v0(theta_w) -> np.ndarray:
    """W θ̂'s (16 pairs, V0, block f30ae35) pair-effect correlation matrix."""
    c = np.asarray(theta_w["pair_cov"], float)
    s = np.sqrt(np.diag(c))
    return c / np.outer(s, s)


def sigma_diag_v0(pair_means, groups: list, r) -> np.ndarray:
    """Σ_pair = D^½ R_V0 D^½, D = the variances (ddof 1) of the group-averaged pair means (Y.4 merge)."""
    x = np.asarray(pair_means, float).reshape(len(pair_means), -1)
    merged = np.stack([x[g].mean(0) for g in groups])
    s = np.sqrt(merged.var(0, ddof=1)) if len(merged) > 1 else np.zeros(x.shape[1])
    return np.asarray(r, float) * np.outer(s, s)


def fit_y(pairs: list, keys: list, r, merge: bool = True) -> dict:
    """Y.4's θ: w_oc.fit on the admitted pairs in declared order (every component from all of them), then Σ_pair
    from the X-odour-merged pairs by the diagonal + V0 rule for any n_Σ (Y.9.2 P1-3); merge False = records R-un."""
    th = w_oc.fit(list(pairs))
    groups = y_rules.merge_groups(keys) if merge else [[i] for i in range(len(keys))]
    return dict(th, pair_cov=sigma_diag_v0(th["pair_means"], groups, r), n_sigma=len(groups),
                groups=[[keys[i] for i in g] for g in groups])


def boot_y(theta, rng_, r) -> dict:
    """One Y.6.4 parametric draw (Reading 4): w_oc.boot(θ, rng) (W.9.9 P1-3), then n_Σ pair means from N(m0, Σ_pair)
    on the same generator, whose diagonal + V0 rule gives Σ*."""
    tb = w_oc.boot(theta, rng_)
    pm = rng_.multivariate_normal(theta["m0"].ravel(), theta["pair_cov"], theta["n_sigma"],
                                  method="eigh").reshape(-1, 2, 2)
    return dict(tb, pair_cov=sigma_diag_v0(pm, [[i] for i in range(len(pm))], r), n_sigma=theta["n_sigma"])


def theta_fixtures(ys) -> dict:
    """The mutations "Earthquake 묶기 생략" and "대각 + V0 대신 표본 공분산" (Y.6.6, Y.9.2 P1-3): the seven declared
    candidates merge to 6; a fitted Σ_pair's correlation is R_V0 (fix_exact)."""
    keys = list(ys.pilot_w_pairs) + list(ys.pilot_v_pairs)
    out = dict(merge=dict(ok=len(y_rules.merge_groups(keys)) == len(keys) - 1))
    pil = w_oc.synthetic_pilot(rng(ys.oc_seed, tag("fixR")), n_pair=len(keys), base=ys.synth_base, sd=ys.synth_sd,
                               corr=ys.synth_corr, learn=ys.synth_learn)
    rw = r_v0(_synth_theta(ys, "fixW"))
    th = fit_y(pil, keys, rw)
    s = np.sqrt(np.diag(th["pair_cov"]))
    out["sigma_rule"] = dict(ok=bool(np.all(np.abs(th["pair_cov"] / np.outer(s, s) - rw) <= ys.fix_exact)))
    out["ok"] = all(v["ok"] for v in out.values())
    return out


# ================================================================ P2-12 small bootstrap and P2-10 thresholds (records)
def small_bootstrap(theta, z: dict, ys, r, n_draws: int | None = None, log=None) -> dict:
    """Y.9.2 P2-12: small_boot_draws draws (boot_y, root small_boot_seed, TAG_BOOT, draw), calibration only (both
    targets, Y's rule) — status counts, floor share, coarse_step counts. Record only."""
    n = ys.small_boot_draws if n_draws is None else int(n_draws)
    rows = []
    for bi in range(n):
        rb = rng(ys.small_boot_seed, TAG_BOOT, bi)
        tb = boot_y(theta, rb, r)
        ib = rb.integers(0, len(tb["resid"]), ys.cal_reps)
        rows.append({m: calibrate_y(tb, getattr(ys, f), m, ib, z, ys) for m, f in MODES})
        if log is not None:
            log(f"y small bootstrap {bi + 1}/{n}")
    return dict(counts=status_counts(rows), n_draws=n, seed=ys.small_boot_seed,
                rows=[{m: dict(ok=c["ok"], failure=c["failure"], a=c["a"]["status"],
                               b=[x["status"] for x in (c["b"] or [])]) for m, c in row.items()} for row in rows],
                note="기록 전용(Y.9.2 P2-12) — 사전 점검 통과 조건에 쓰지 않는다")


def threshold_recompute(theta, cal_min: dict, ys) -> dict:
    """Y.9.2 P2-10: Y.0's definition on Y θ̂ under Y's power calibration, at the all-lo and all-hi stairs, on
    floor_resid_n residual vectors drawn from root records_seed, tag "thr". Record only (P0-1)."""
    if not cal_min.get("ok"):
        return dict(available=False, calibration_failure=cal_min.get("failure"))
    res = theta["resid"][rng(ys.records_seed, tag("thr")).integers(0, len(theta["resid"]), ys.floor_resid_n)]
    st = stair_corners(cal_min)
    return dict(available=True, lo=thresholds(theta, st["lo"][0]["a"], st["lo"][0]["b"], ys, res),
                hi=thresholds(theta, st["hi"][0]["a"], st["hi"][0]["b"], ys, res), seed=ys.records_seed,
                note="기록 전용(Y.9.2 P2-10) — 값이 어떻든 20 · 43은 바뀌지 않는다(P0-1)")


# ================================================================ the point-θ precheck (Y.6.1 + Y.9.2)
def precheck_y(theta, z: dict, ys, k_ranges, n_rep: int | None = None, cell=run_cell, log=None) -> dict:
    """Y.6.1 on Y θ̂: Y's calibration (root precheck_seed, TAG_CAL draws), per g × target × scenario one evaluate
    (main stream (TAG_POINT, g, target[, "near"]), mixture stream ("mix", 0, g, target, scenario)), plus the
    record-only all-lo / all-hi stairs on the same main stream; point power = min over g and scenarios, point false
    = max; fill > fill_max at a k excludes it (P1-5); passed iff a design of a KEPT k range meets both targets at
    every k of it (no envelope). Also Y.8's best-power design, Y.6.5's records target, the F = f_max table and
    X.9.1.3 P1-3's m table on k_min..k_cap. Each evaluate is a resumable cell."""
    t0 = time.perf_counter()
    n_rep = ys.precheck_reps if n_rep is None else int(n_rep)
    root = ys.precheck_seed
    k_ranges = [tuple(int(v) for v in kr) for kr in k_ranges]
    idx = rng(root, TAG_CAL).integers(0, len(theta["resid"]), ys.cal_reps)
    cal = {m: calibrate_y(theta, getattr(ys, f), m, idx, z, ys) for m, f in MODES}
    shp = x_oc.grid_shape(ys)
    ones = [1.0] * ys.f_max
    key0 = dict(theta=w_oc.summary(theta), root=root, n_rep=n_rep, cal={m: c["corners"] for m, c in cal.items()},
                z={k: list(v) for k, v in z.items()})                # z_V enters evaluate_y
    point, stairs, fills = {}, {}, {}

    def run(mix, gi, g, mi, si, sc):
        extra = (tag("near"),) if sc == "near" else ()
        r = evaluate_y(theta, rng(root, TAG_POINT, gi, mi, *extra), rng(root, tag("mix"), 0, gi, mi, si), n_rep,
                       [mix] * ys.k_cap, ones, ones, g, z, ys, sc)
        return dict(p=r["p"].tolist(), fill_by_k=r["fill_by_k"])
    for gi, g in enumerate(ys.cluster_grid):
        for mi, (m, _f) in enumerate(MODES):
            c = cal[m]
            for si, sc in enumerate(SCENARIOS):
                lab = f"g{g}|{m}|{sc}"
                if not c["ok"]:
                    point[lab] = np.full(shp, c["fill"])
                    fills[lab] = None
                    stairs[f"{lab}|lo"] = stairs[f"{lab}|hi"] = point[lab]
                    continue
                v = cell(f"point_{lab}", dict(key0, g=g, mode=m, scenario=sc),
                         lambda c=c, gi=gi, g=g, mi=mi, si=si, sc=sc: run(c["corners"], gi, g, mi, si, sc))
                point[lab], fills[lab] = np.asarray(v["p"]), v["fill_by_k"]
                st = stair_corners(c)
                for s in ("lo", "hi"):
                    if len(c["corners"]) == 1:
                        stairs[f"{lab}|{s}"] = point[lab]
                        continue
                    w = cell(f"stair_{s}_{lab}", dict(key0, g=g, mode=m, scenario=sc, stair=s),
                             lambda s=s, gi=gi, g=g, mi=mi, si=si, sc=sc, st=st: run(st[s], gi, g, mi, si, sc))
                    stairs[f"{lab}|{s}"] = np.asarray(w["p"])
                if log is not None:
                    log(f"y precheck {lab} done ({time.perf_counter() - t0:.0f} s)")
    labs = [(g, sc) for g in ys.cluster_grid for sc in SCENARIOS]
    power = np.min([point[f"g{g}|min|{sc}"] for g, sc in labs], 0)
    false = np.max([point[f"g{g}|max|{sc}"] for g, sc in labs], 0)
    fill_bad = np.zeros(shp[-1], bool)
    for v in fills.values():
        if v is not None:
            fill_bad |= np.round(np.asarray(v) - ys.fill_max, ys.round_digits) > 0
    fs = list(range(ys.f_min, ys.f_max + 1))
    passing = []
    for kr in k_ranges:
        ok = range_ok(power, false, fill_bad, ys, kr, envelope=False)
        passing += [dict(p_set=ys.p_set_grid[i[0]], q=ys.q_grid[i[1]], K=ys.k_grid[i[2]], F=fs[i[3]],
                         k_range=list(kr)) for i in np.ndindex(ok.shape) if ok[i]]
    st_worst = {s: dict(power=np.min([stairs[f"g{g}|min|{sc}|{s}"] for g, sc in labs], 0).tolist(),
                        false=np.max([stairs[f"g{g}|max|{sc}|{s}"] for g, sc in labs], 0).tolist())
                for s in ("lo", "hi")}
    return dict(passed=bool(passing), passing=passing, calibration=cal, power=power.tolist(), false=false.tolist(),
                fill_bad=fill_bad.tolist(), fills=fills, stairs=st_worst,
                point={k: v.tolist() for k, v in point.items()},
                best_power=pick_y(power, false, ys, k_ranges, False),
                records_target=pick_y(power, false, ys, k_ranges, True),
                at_f32=table_at_f_y(power, false, ys, k_ranges, ys.f_max), m_needed=XV.m_needed_table(ys),
                k_ranges=[list(kr) for kr in k_ranges],
                axes=dict(p_set=list(ys.p_set_grid), q=list(ys.q_grid), K=list(ys.k_grid), F=fs,
                          k=list(range(ys.k_min, ys.k_cap + 1)), g=list(ys.cluster_grid), scenario=list(SCENARIOS)),
                n_rep=n_rep, seed=root, timing_s=time.perf_counter() - t0)


# ================================================================ R-pre (Y.9.2 P1-5): the real filter on generated pre
def simulate_pre(theta, rng_, n_rep, n_pair, n_fly, n_probe, K, a_exp, b_exp, fly_a, fly_b, g, z, ys) -> tuple:
    """Each candidate pair's whole draw (base m0s + w + u without a base filter, fly effects, residuals) is kept only
    when its own pre (n_fly flies × the first K probes) passes y_rules.filter_values / filter_passes — the real filter's
    code — within ys.tries rounds; that same pre is the experiment's pre. A slot never accepted gets the population
    pair (last round's fly effects and residuals). Returns (stages, filled)."""
    L = np.linalg.cholesky(theta["pair_cov"] + ys.chol_jitter * np.eye(4))
    Lf = np.linalg.cholesky(theta["fly_cov"] + ys.chol_jitter * np.eye(4))
    w = (rng_.standard_normal((n_rep, 4)) @ L.T * math.sqrt(g)).reshape(n_rep, 1, 2, 2)
    a = np.asarray(a_exp, float)[:, :, None] * np.asarray(fly_a, float)[None, None, :]
    b = np.asarray(b_exp, float)[:, :, None] * np.asarray(fly_b, float)[None, None, :]
    out = np.zeros((n_rep, n_pair, n_fly, n_probe, len(w_oc.SLOTS), 2, 2), np.int32)
    need = np.ones((n_rep, n_pair), bool)

    def draw(base):
        v = (rng_.standard_normal((n_rep, n_pair, n_fly, 4)) @ Lf.T).reshape(n_rep, n_pair, n_fly, 2, 2)
        mu = w_oc.slot_means(theta, base[:, :, None], v, a, b)
        idx = rng_.integers(0, len(theta["resid"]), (n_rep, n_pair, n_fly, n_probe))
        return np.clip(np.rint(mu[:, :, :, None] + theta["resid"][idx]), 0, None).astype(np.int32)
    for _ in range(ys.tries):
        u = (rng_.standard_normal((n_rep, n_pair, 4)) @ L.T).reshape(n_rep, n_pair, 2, 2)
        c = draw(theta["m0s"] + w + u)
        d, la, lp = y_rules.filter_values(c[..., :K, w_oc.SI["pre"], :, :], z)
        ok = y_rules.filter_passes(d, la, lp, ys)
        take = need & ok
        out[take] = c[take]
        need &= ~ok
        if not need.any():
            return w_oc.to_stages(out), need
    c = draw(theta["m0s"] + w + np.zeros((n_rep, n_pair, 2, 2)))
    out[need] = c[need]
    return w_oc.to_stages(out), need


def evaluate_pre(theta, rng_, mix_rng, n_rep: int, pair_mix: list, fly_a, fly_b, g: float, z: dict, yd) -> dict:
    """evaluate_y with simulate_pre on a one-design spec yd (its F flies, 2K probes, the filter on F × K pre)."""
    fs = list(range(yd.f_min, yd.f_max + 1))
    kk = list(range(yd.k_min, yd.k_cap + 1))
    K = yd.k_grid[0]
    hits = np.zeros(x_oc.grid_shape(yd))
    filled = np.zeros(yd.k_cap)
    mixes, which = _distinct(pair_mix)
    done = 0
    while done < n_rep:
        n = min(yd.oc_chunk, n_rep - done)
        a, b = _corner_effects(mix_rng, n, yd.k_cap, mixes, which)
        d, fill = simulate_pre(theta, rng_, n, yd.k_cap, yd.f_max, 2 * K, K, a, b, fly_a, fly_b, g, z, yd)
        hits += x_oc.tally(d, z, yd, yd.q_grid, yd.k_grid, fs, kk)
        filled += fill.sum(0)
        done += n
    cum = np.cumsum(filled)[yd.k_min - 1:]
    return dict(p=hits / n_rep, fill_by_k=(cum / (n_rep * np.asarray(kk, float))).tolist())


# ================================================================ point records on a precheck STOP (Y.6.5, Y.6.4, P1-3, P1-5)
RECORD_VARIANTS = ("Y", "R-W", "R-Σ2", "R-un")          # root precheck_seed, TAG_RECORD (Reading 12)
EXTRA_VARIANTS = ("R-V", "R-pre")                       # root records_seed, their own string tag


def record_thetas(theta_y, theta_w, cands: dict, keys: list, r, ys) -> tuple:
    """The records' θ per variant: Y θ̂; R-W = W θ̂ (16 pairs, V0) under Y's filter; R-Σ2 = Σ_pair × sigma2_scale;
    R-un = Σ from the unmerged pairs (same diagonal + V0 rule); R-V = the admitted V pairs only (merge + rule), None
    when fewer than 2 remain after the merge; R-pre = Y θ̂ with the measured-pre filter. Returns (thetas, notes)."""
    vkeys = [k for k in keys if k in ys.pilot_v_pairs]
    notes = {}
    rv = None
    if len(y_rules.merge_groups(vkeys)) >= 2:
        rv = fit_y([cands[k] for k in vkeys], vkeys, r)
    else:
        notes["R-V"] = f"묶은 뒤 V 쌍 {len(y_rules.merge_groups(vkeys))} < 2(Y.9.2 P1-3) — null"
    thetas = {"Y": theta_y, "R-W": theta_w, "R-Σ2": dict(theta_y, pair_cov=theta_y["pair_cov"] * ys.sigma2_scale),
              "R-un": fit_y([cands[k] for k in keys], keys, r, merge=False), "R-V": rv, "R-pre": theta_y}
    return thetas, notes


def point_records_y(thetas: dict, target: dict, z: dict, ys, n_rep: int | None = None, cell=run_cell,
                    log=None) -> dict:
    """Y.6.5's point records on the target design (Reading 10): Y θ̂ — homogeneous true d′ record_dprimes (power
    calibration), the false-pass 0.5, mixed flies, Σ × het_scales at both targets, W's heterogeneous three — each with
    the mixture and the two stairs; every other variant — the homogeneous four and the false-pass 0.5, mixture only;
    R-pre through simulate_pre. k-wise P(PASS) (k_min..k_cap; the target's range is what Y.10 reads), worst g kept per
    g. A failed calibration → None with its status."""
    n_rep = ys.oc_reps if n_rep is None else int(n_rep)
    yd = x_oc.design_spec(ys, target)
    F_, K_ = yd.f_max, yd.k_cap
    ones, half = [1.0] * F_, [1.0 if i % 2 == 0 else 0.0 for i in range(F_)]
    out, cals, roots = {}, {}, {}
    for vi, (name, th) in enumerate(thetas.items()):
        root = ys.precheck_seed if name in RECORD_VARIANTS else ys.records_seed
        roots[name] = root
        if th is None:
            out[name] = None
            continue
        vt = (TAG_RECORD, vi) if name in RECORD_VARIANTS else (tag(name),)
        idx = (rng(root, TAG_CAL) if name in RECORD_VARIANTS else rng(root, TAG_CAL, tag(name))).integers(
            0, len(th["resid"]), ys.cal_reps)
        cmin = {t: calibrate_y(th, t, "min", idx, z, ys) for t in sorted(set(ys.record_dprimes) | {ys.d_power})}
        cmax = calibrate_y(th, ys.d_false, "max", idx, z, ys)
        cals[name] = {f"min|{t}": dict(ok=c["ok"], failure=c["failure"]) for t, c in cmin.items()}
        cals[name][f"max|{ys.d_false}"] = dict(ok=cmax["ok"], failure=cmax["failure"])
        ev = evaluate_pre if name == "R-pre" else evaluate_y

        def at(th_, mixes, fa, fb, scen, gi, g, mode_i, vt=vt, root=root, ev=ev):
            main = rng(root, *vt, gi, scen)
            mix = rng(root, tag("mix"), 0, *vt, gi, mode_i, scen)     # the variant's tags: never a precheck stream
            r = ev(th_, main, mix, n_rep, mixes, fa, fb, g, z, yd)
            return dict(p=np.asarray(r["p"])[0, 0, 0, 0].tolist(), fill=r["fill_by_k"])

        def one(name=name, th=th, cmin=cmin, cmax=cmax):
            res = {}
            for gi, g in enumerate(ys.cluster_grid):
                e, s = {}, 0
                e["true_dprime"] = {}
                for t in ys.record_dprimes:
                    s += 1
                    e["true_dprime"][str(t)] = (at(th, [cmin[t]["corners"]] * K_, ones, ones, s, gi, g, 0)
                                                if cmin[t]["ok"] else None)
                s += 1
                e["false_pass"] = at(th, [cmax["corners"]] * K_, ones, ones, s, gi, g, 1) if cmax["ok"] else None
                if name == "R-pre":
                    e["fill"] = (e["false_pass"] or {}).get("fill")
                if name != "Y":
                    res[f"g{g}"] = e
                    continue
                hi, lo, mid = cmin[ys.d_power], cmin[ys.het_low_dprime], cmin[ys.het_all_dprime]
                het = {}
                if lo["ok"] and hi["ok"]:
                    s += 1
                    het[f"one_pair_{ys.het_low_dprime}"] = at(th, [lo["corners"]] + [hi["corners"]] * (K_ - 1), ones,
                                                              ones, s, gi, g, 0)
                    s += 1
                    het[f"half_pairs_{ys.het_low_dprime}"] = at(th, [lo["corners"] if i % 2 else hi["corners"]
                                                                     for i in range(K_)], ones, ones, s, gi, g, 0)
                if mid["ok"]:
                    s += 1
                    het[f"all_{ys.het_all_dprime}"] = at(th, [mid["corners"]] * K_, ones, ones, s, gi, g, 0)
                e["heterogeneous_w"] = het
                s += 1
                e["mixed_flies"] = at(th, [hi["corners"]] * K_, half, half, s, gi, g, 0) if hi["ok"] else None
                e["pair_cov_scaled"] = {}
                for sc in ys.het_scales:
                    ts = dict(th, pair_cov=th["pair_cov"] * sc)
                    s += 1
                    e["pair_cov_scaled"][f"{sc}|power"] = (at(ts, [hi["corners"]] * K_, ones, ones, s, gi, g, 0)
                                                           if hi["ok"] else None)
                    s += 1
                    e["pair_cov_scaled"][f"{sc}|false"] = (at(ts, [cmax["corners"]] * K_, ones, ones, s, gi, g, 1)
                                                           if cmax["ok"] else None)
                e["stairs"] = {}
                for t in ys.record_dprimes:
                    if cmin[t]["ok"]:
                        stc = stair_corners(cmin[t])
                        e["stairs"][str(t)] = {k: at(th, [stc[k]] * K_, ones, ones, ys.record_dprimes.index(t) + 1,
                                                     gi, g, 0)["p"] for k in ("lo", "hi")}
                if cmax["ok"]:
                    stc = stair_corners(cmax)
                    e["stairs"][f"false|{ys.d_false}"] = {k: at(th, [stc[k]] * K_, ones, ones,
                                                                len(ys.record_dprimes) + 1, gi, g, 1)["p"]
                                                          for k in ("lo", "hi")}
                res[f"g{g}"] = e
            if log is not None:
                log(f"y records {name} done")
            return res
        out[name] = cell(f"records_{name}", dict(theta=w_oc.summary(th), target=target, n_rep=n_rep, root=root,
                                                 variant=name), one)
    return dict(target=target, variants=out, calibrations=cals, roots=roots, n_rep=n_rep,
                k=list(range(ys.k_min, ys.k_cap + 1)),
                note="사전 점검 STOP의 점 records(Y.6.5) — 변형은 기록 전용(Y.6.4, Y.9.2 P1-3 · P1-5)")
