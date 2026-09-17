"""Scratch fixtures for the G.14.2 k* rule (pick_k_star / bracket), imported from the driver after the smoke."""
import importlib.util, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), "/Users/jeonsehyeon/orca/workspaces/fruit-fly/guillemot/docs/superpowers/specs/m2-calibration-g"]
spec = importlib.util.spec_from_file_location("drv", HERE / "m2_engine_probe.py")
D = importlib.util.module_from_spec(spec); spec.loader.exec_module(D)

grid = [1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
mono = {1.75: 0.080, 2.0: 0.060, 2.25: 0.040, 2.5: 0.030, 2.75: 0.020, 3.0: 0.010}
fails = 0
def check(name, cond):
    global fails
    print(("PASS " if cond else "FAIL ") + name); fails += not cond

k, br = D.pick_k_star(mono, 0.0415, grid)
check("nearest grid point within 0.5 pp -> no bisection", k == 2.25 and br is None)
k, br = D.pick_k_star(mono, 0.049, grid)
check("off by > 0.5 pp and bracketed -> nearest 2.25 and bracket (2.0, 2.25)", k == 2.25 and br == (2.0, 2.25))
pts = dict(mono); pts[2.125] = 0.0485
check("bisection midpoint wins when closest", D.pick_among(pts, 0.049, [2.0, 2.25, 2.125]) == 2.125)
pts[2.125] = 0.070
check("bisection midpoint loses when farther", D.pick_among(pts, 0.049, [2.0, 2.25, 2.125]) == 2.25)
k, br = D.pick_k_star(mono, 0.002, grid)
check("target below the grid range -> endpoint 3.0, no bracket", k == 3.0 and br is None)
k, br = D.pick_k_star(mono, 0.0845, grid)
check("target above range, within tolerance -> 1.75, no bracket", k == 1.75 and br is None)
k, br = D.pick_k_star({2.0: 0.5, 2.25: 0.25}, 0.375, [2.0, 2.25])
check("exact tie -> smaller k, bracket returned", k == 2.0 and br == (2.0, 2.25))
nm = {1.75: 0.08, 2.0: 0.02, 2.25: 0.08, 2.5: 0.02}
k, br = D.pick_k_star(nm, 0.05, [1.75, 2.0, 2.25, 2.5])
check("non-monotone medians -> first bracket in kc_thresh order", br == (1.75, 2.0))
sys.exit(1 if fails else 0)
