"""Y's calibration (Y.6.2 as amended by Y.9.2 P1-2): fixtures 1–5 and 7–8a with fixed expectations, the mutations
each one catches (Y.6.6 + P1-2 (d)), and calibrate_y on a synthetic θ (corners, weights, true d′, fills)."""
import dataclasses

import numpy as np
import pytest

from flymon.brain import w_oc
from flymon.brain import y_oc as O
from flymon.brain.y_spec import SPEC as Y

Z = {"A": (16.917, 12.484), "P": (80.167, 29.775)}


def test_fixtures_all_pass():
    fx = O.calibration_fixtures(Y)
    assert fx["ok"], {k: v for k, v in fx.items() if k != "ok" and not v["ok"]}
    assert set(fx) == {"ok", "f1_step", "f2_nonmonotone", "f3_x2_only", "f4_none", "f5_above_zero", "f7_narrow",
                       "f8a_two_knobs"}


def test_fixture1_step():
    x_s, lo_v, hi_v = Y.fix_step
    c = O.calibrate_handle(lambda x: hi_v if x >= x_s else lo_v, Y.fix_hi_w, Y.fix_target, "min", Y)
    lo, hi = c["stairs"]
    assert c["status"] == O.OK and lo["x"] < x_s <= hi["x"] and hi["x"] - lo["x"] <= Y.knob_tol
    assert abs(hi["w"] - (0.0 - lo_v) / (hi_v - lo_v)) <= 1e-12
    assert abs(lo["w"] * lo["d"] + hi["w"] * hi["d"] - Y.fix_target) <= 1e-12
    assert c["coarse_step"] is True and abs(c["step"] - (hi_v - lo_v)) <= 1e-12


def test_fixture5_and_fills():
    up = lambda x: Y.fix_above  # noqa: E731
    p = O.calibrate_handle(up, Y.fix_hi_w, Y.fix_target, "min", Y)
    f = O.calibrate_handle(up, Y.fix_hi_w, Y.fix_target, "max", Y)
    assert (p["status"], p["stairs"]) == (O.ABOVE, None)
    assert f["status"] == O.ZERO and f["stairs"] == [dict(x=0.0, w=1.0, d=Y.fix_above + Y.fix_target)]
    down = lambda x: Y.fix_below  # noqa: E731
    cp = O.calibrate_pair(down, lambda a: down, Y.fix_hi_w, Y.fix_hi_w, Y.fix_target, "min", Y)
    cf = O.calibrate_pair(down, lambda a: down, Y.fix_hi_w, Y.fix_hi_w, Y.fix_target, "max", Y)
    assert (cp["ok"], cp["fill"], cp["failure"]) == (False, 0.0, dict(handle="a", status=O.FLOOR))
    assert (cf["ok"], cf["fill"]) == (False, 1.0)


def test_no_no_convergence_status_exists():
    rng = np.random.default_rng(0)
    for _ in range(20):
        x_s = float(rng.uniform(1, 390))
        jump = float(rng.uniform(0.05, 1.0))
        c = O.calibrate_handle(lambda x: -jump / 2 if x < x_s else jump / 2, Y.fix_hi_w, Y.fix_target, "min", Y)
        assert c["status"] in (O.OK, O.FLOOR, O.ABOVE, O.ZERO)


# ================================================================ mutations (Y.6.6 + Y.9.2 P1-2 (d))
def _fails(monkeypatch, name, fn, fixture):
    """The mutant makes the fixture fail — or crash, which a stage0 run reports as INVALID."""
    monkeypatch.setattr(O, name, fn)
    try:
        ok = O.calibration_fixtures(Y)[fixture]["ok"]
    except Exception:                                   # noqa: BLE001 — a crash is a caught mutant too
        ok = False
    assert not ok


def test_mutation_one_stair_or_conservative_side(monkeypatch):
    _fails(monkeypatch, "_weights", lambda d_lo, d_hi, t: (1.0, 0.0), "f1_step")


def test_mutation_flipped_weights(monkeypatch):
    def flipped(d_lo, d_hi, t):
        w_hi = (t - d_lo) / (d_hi - d_lo)
        return w_hi, 1.0 - w_hi
    _fails(monkeypatch, "_weights", flipped, "f1_step")


def test_mutation_f_tolerance_bisection(monkeypatch):
    def f_tol(f, lo, hi, f_lo, f_hi, ys):
        for _ in range(ys.cal_iter):
            mid = (lo + hi) / 2
            fm = f(mid)
            if abs(fm) <= ys.cal_tol:
                return mid, mid, fm, fm
            lo, hi, f_lo, f_hi = (mid, hi, fm, f_hi) if fm < 0 else (lo, mid, f_lo, fm)
        return lo, lo, f_lo, f_lo                       # no convergence: one point, W-style
    _fails(monkeypatch, "_bisect_knob", f_tol, "f1_step")


def test_mutation_no_refinement(monkeypatch):
    real = O._first_crossing
    _fails(monkeypatch, "_first_crossing", lambda f, hi_w, f0, ys: real(f, hi_w, f0, dataclasses.replace(
        ys, refine_delta=-1e9)), "f7_narrow")


def test_mutation_widen_before_x1(monkeypatch):
    real = O._first_crossing
    _fails(monkeypatch, "_first_crossing", lambda f, hi_w, f0, ys: real(f, hi_w * ys.widen[-1], f0, dataclasses.replace(
        ys, widen=(1,))), "f2_nonmonotone")


def test_mutation_last_crossing(monkeypatch):
    def last(f, hi_w, f0, ys):                          # the LAST coarse up-crossing over ×4, no refinement
        h = hi_w / ys.grid_steps
        ups = [i for i in range(1, ys.grid_steps * ys.widen[-1] + 1) if f((i - 1) * h) < 0 <= f(i * h)]
        if not ups:
            return dict(found=False, refined_cells=0)
        i = ups[-1]
        return dict(found=True, lo=(i - 1) * h, hi=i * h, f_lo=f((i - 1) * h), f_hi=f(i * h), stage=ys.widen[-1],
                    index=i, coarse_index=i, refined_cells=0, refined_before_coarse=False)
    _fails(monkeypatch, "_first_crossing", last, "f2_nonmonotone")


def test_mutation_w_style_bisection(monkeypatch):
    def w_style(f, hi_w, f0, ys):
        r = w_oc._bisect(f, hi_w * ys.widen[-1], ys.cal_tol, ys.cal_iter)
        if r["value"] is None:
            return dict(found=False, refined_cells=0)
        return dict(found=True, lo=r["value"], hi=r["value"], f_lo=f(r["value"]), f_hi=f(r["value"]), stage=1,
                    index=0, coarse_index=0, refined_cells=0, refined_before_coarse=False)
    _fails(monkeypatch, "_first_crossing", w_style, "f2_nonmonotone")


def test_mutation_b_in_one_a_state(monkeypatch):
    monkeypatch.setattr(O, "_b_states", lambda ra: ra["stairs"][:1] * len(ra["stairs"]))
    assert not O.calibration_fixtures(Y)["f8a_two_knobs"]["ok"]


# ================================================================ calibrate_y on a synthetic θ
@pytest.fixture(scope="module")
def theta():
    return w_oc.fit(w_oc.synthetic_pilot(np.random.default_rng(3), base=(40.0, 90.0), sd=4.0, corr=0.8,
                                         learn=(20.0, 10.0)))


def test_calibrate_y_corners_and_true_dprime(theta):
    idx = np.random.default_rng(1).integers(0, len(theta["resid"]), Y.cal_reps)
    c = O.calibrate_y(theta, Y.d_power, "min", idx, Z, Y)
    assert c["ok"] and abs(sum(k["w"] for k in c["corners"]) - 1.0) <= 1e-12
    assert len(c["b"]) == len(c["a"]["stairs"]) and len(c["true_dprime"]) == len(c["corners"])
    mean_a = sum(s["w"] * s["d"] for s in c["a"]["stairs"])
    assert abs(mean_a - Y.d_power) <= 1e-9
    st = O.stair_corners(c)
    assert st["lo"][0]["a"] == c["a"]["stairs"][0]["x"] and st["hi"][0]["b"] == c["b"][-1]["stairs"][-1]["x"]
    bad = O.calibrate_y(theta, 400.0, "min", idx, Z, Y)
    assert not bad["ok"] and bad["fill"] == 0.0 and bad["failure"]["status"] == O.FLOOR


def test_status_counts():
    a_ok = dict(status=O.OK, stairs=[dict(x=1.0, w=1.0, d=1.5)], stage=2, coarse_step=True)
    rows = [dict(min=dict(ok=True, a=a_ok, b=[dict(a_ok, coarse_step=False)], failure=None),
                 max=dict(ok=False, a=dict(status=O.FLOOR, stairs=None), b=None, failure=dict(handle="a",
                                                                                         status=O.FLOOR)))]
    out = O.status_counts(rows)
    assert out["min"]["handle_status"] == {"a|ok": 1, "b|ok": 1} and out["min"]["coarse_step"] == {"a": 1}
    assert out["min"]["stage"] == {"a|×2": 1, "b|×2": 1} and out["max"]["floor_share"] == 1.0
