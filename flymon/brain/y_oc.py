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
  SeedSequence([root, "mix", draw, g, target, scenario]), Reading 2) and all its pairs and flies use it; the main
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
