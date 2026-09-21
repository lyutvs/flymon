"""h4_formula is G.14's verdict arithmetic with z passed in (spec H.4 step 2): checked against the frozen module."""
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from flymon.brain import h4_formula as F
from flymon.brain.h4_spec import SPEC

FROZEN = Path(__file__).resolve().parents[2] / "docs/superpowers/specs/m2-calibration-g/engine_probe_verdict.py"
FROZEN_SHA = "c9c81ab97d13e7bc163210564e1114d3caafd60c63431bbef8eb977f8fb334d7"   # spec G.14


@pytest.fixture(scope="module")
def V():
    assert hashlib.sha256(FROZEN.read_bytes()).hexdigest() == FROZEN_SHA
    s = importlib.util.spec_from_file_location("engine_probe_verdict", FROZEN)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


def _probe(rng, n, lo=0, hi=60):
    return {"A": rng.integers(lo, hi, size=(n, 2)).tolist(), "P": rng.integers(lo, hi, size=(n, 2)).tolist()}


def test_thresholds_are_g14s(V):
    assert (SPEC.testable_min, SPEC.naive_max, SPEC.t_b_min, SPEC.f_a_min) == (V.TESTABLE, V.NAIVE, V.BAR_T_B, V.BAR_F_A)


@pytest.mark.parametrize("x", [[1.0], [], [0.0, 0.0], [2.0, 2.0], [-3.0, -3.0, -3.0], [1.0, 2.0, 4.0], [0.5, -0.5]])
def test_dprime_is_g14s_including_its_limits(V, x):
    assert F.dprime(x) == V.dprime(x)


def test_pair_stats_with_f3_constants_is_g14s_on_random_and_degenerate_reports(V):
    rng = np.random.default_rng(7)
    reports = [{k: _probe(rng, 8) for k in ("pre", "R1", "R2")} for _ in range(300)]
    same = _probe(rng, 8)
    reports.append({"pre": same, "R1": same, "R2": same})                          # every d' is the 0 limit
    shifted = {"A": [[a + 5, b] for a, b in same["A"]], "P": same["P"]}
    reports.append({"pre": same, "R1": shifted, "R2": shifted})                     # r = +inf, p = 0
    reports.append({k: _probe(rng, 1) for k in ("pre", "R1", "R2")})               # undefined
    for rep in reports:
        assert F.pair_stats(rep, V.Z, SPEC.testable_min) == V.pair_stats(rep)


def test_arm_aggregate_is_g14s(V):
    rng = np.random.default_rng(3)
    for _ in range(50):
        keys = [(ax, t, f"x{j}", f"y{j}") for ax, n in (("a", 18), ("b", 21)) for j, t in enumerate(range(n))]
        stats = {k: V.pair_stats({p: _probe(rng, 8) for p in ("pre", "R1", "R2")}) for k in keys}
        assert F.arm_aggregate(stats, SPEC.naive_max, SPEC.t_b_min, SPEC.f_a_min) == V.arm_aggregate(stats)


def test_dv_uses_the_given_constants():
    probe = {"A": [[30, 10], [20, 20]], "P": [[5, 25], [40, 40]]}
    z = {"A": (100.0, 10.0), "P": (-7.0, 5.0)}
    # V(X) - V(Y) = (A_x - A_y) / sd_A - (P_x - P_y) / sd_P; the means cancel
    assert F.dv(probe, z).tolist() == [20 / 10 - (-20) / 5, 0.0]
    with pytest.raises(ValueError):
        F.dv({"A": [[1, 2, 3]], "P": [[1, 2, 3]]}, z)


def test_testable_is_inclusive_at_the_threshold():
    base = {"A": [[20, 20]] * 4, "P": [[40, 40]] * 4}
    r1 = {"A": base["A"], "P": [[40 - d, 40] for d in (2, 4, 2, 4)]}                  # dV rises by d / sd_P
    z = {"A": (0.0, 1.0), "P": (0.0, 1.0)}
    r2 = {"A": [[20 - d, 20] for d in (2, 4, 2, 4)], "P": r1["P"]}
    s = F.pair_stats({"pre": base, "R1": r1, "R2": r2}, z, testable_min=float(np.mean([2, 4]) / np.std([2, 4, 2, 4], ddof=1)))
    assert s["r"] == -s["p"] == s["m"] and s["testable"]
