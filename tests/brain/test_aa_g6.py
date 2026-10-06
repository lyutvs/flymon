"""AA.9.1: the precheck wrapper replaces only d_power and precheck_seed, agrees with y_oc.precheck_y's own passing list,
reports calibration failures as statuses, and the heterogeneity wrapper is deterministic."""
import dataclasses

import numpy as np

from flymon.brain import aa_estimate as E
from flymon.brain import w_oc, y_oc
from flymon.brain.aa_spec import SPEC
from flymon.brain.y_spec import SPEC as Y

YS = dataclasses.replace(Y, precheck_reps=2, oc_chunk=2)
COSTS = dict(trial_s=0.001, presentation_s=0.0001, oracle_round_s=1.0, workers=16)
Z = {"A": (16.916666666666668, 12.483878492769072), "P": (80.16666666666667, 29.775432639827233)}


def theta():
    kw = dict(base=(40.0, 90.0), sd=4.0, corr=0.8, learn=(20.0, 10.0))
    pil = w_oc.synthetic_pilot(np.random.default_rng(3), n_pair=6, **kw)
    keys = [f"a|{i}|O{i % 4}|y{i}" for i in range(6)]
    return y_oc.fit_y(pil, keys, np.eye(4))


def test_g6_spec_replaces_two_fields():
    yd = E.g6_spec(1.2, SPEC, Y)
    diff = {k for k, v in dataclasses.asdict(yd).items() if v != dataclasses.asdict(Y)[k]}
    assert diff == {"d_power", "precheck_seed"} and yd.precheck_seed == 87_200_000 and yd.d_power == 1.2


def test_g6_row_agrees_with_precheck_passing():
    th = theta()
    row = E.g6_row(th, Z, 1.5, COSTS, 31, SPEC, YS)
    pc = y_oc.precheck_y(th, Z, E.g6_spec(1.5, SPEC, YS), [tuple(k) for k in YS.k_ranges])
    if row["status"] == "ok":
        assert row["n_pass"] == len(pc["passing"]) and set(row["target_design_power"]) == {4, 5, 6, 7, 8}
        assert row["first"] is None or row["first"] == min(
            row["designs"], key=lambda r: (-r["p_set"], r["cost_h"], -r["q"], r["K"], r["F"], r["k_range"][0]))
    else:
        assert row["status"] == pc["calibration"]["min"]["failure"]["status"]


def test_g6_unreachable_target_is_a_status():
    row = E.g6_row(theta(), Z, 40.0, COSTS, 31, SPEC, YS)
    assert row["status"] in (y_oc.FLOOR, y_oc.ABOVE)


def test_g6_hetero_deterministic():
    th = theta()
    ref = E.g6_row(th, Z, 1.5, COSTS, 31, SPEC, YS)
    if ref["status"] != "ok":
        return
    a = E.g6_hetero(th, Z, 1.5, 0.3, ref, COSTS, 31, SPEC, YS)
    assert a == E.g6_hetero(th, Z, 1.5, 0.3, ref, COSTS, 31, SPEC, YS) and "n_pass" in a
