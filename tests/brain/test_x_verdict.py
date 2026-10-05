"""X.3's set rule (X.4.6's boundary fixtures with their fixed verdicts, the phase-A mutations each fixture catches, the
p_set 1.0 = W overall_code identity for n ≥ 4) and X.9.1.3's m-needed table with its duplicate marks."""
import numpy as np
import pytest

from flymon.brain import w_verdict as WV
from flymon.brain import x_verdict as XV
from flymon.brain.x_spec import SPEC as XS

P, F, B, I = WV.P_PASS, WV.P_FAIL, WV.P_BAND, WV.P_INVALID
FIX = [  # (n, m, b, p_set) -> verdict, X.4.6 + one BAND-robustness case
    ((4, 2, 0, 0.5), XV.S_FAIL), ((4, 3, 0, 0.75), XV.S_PASS), ((4, 3, 0, 0.875), XV.S_FAIL),
    ((6, 3, 0, 0.625), XV.S_FAIL), ((5, 3, 1, 0.5), XV.S_PASS), ((4, 2, 2, 0.5), XV.S_UNDECIDED),
    ((3, 3, 1, 0.5), XV.S_UNDECIDED), ((4, 1, 1, 0.5), XV.S_FAIL), ((8, 8, 0, 1.0), XV.S_PASS),
    ((8, 7, 0, 1.0), XV.S_FAIL), ((4, 3, 2, 0.75), XV.S_UNDECIDED)]


@pytest.mark.parametrize("args,want", FIX)
def test_boundary_fixtures(args, want):
    n, m, b, p = args
    assert int(XV.set_code_counts(n, m, b, False, p, XS)) == want
    final = np.array([P] * m + [F] * (n - m) + [B] * b)
    assert int(XV.set_code(final, False, p, XS)) == want


def test_causes():
    assert XV.undecided_cause(4, XS) == "BAND 잔존" and XV.undecided_cause(3, XS) == "판정 가능 쌍 < 4"


def test_mutations_are_caught():
    # m ≥ 3 removed → (4, 2, 0, 0.5) would PASS; BAND robustness removed → (4, 3, 2, 0.75) would PASS;
    # the (m + b) < 3 branch removed → (4, 2, 0, 0.5) would be UNDECIDED; BAND counted in n → (3, 3, 1, 0.5) would PASS
    assert int(XV.set_code_counts(4, 2, 0, False, 0.5, XS)) == XV.S_FAIL
    assert int(XV.set_code_counts(4, 3, 2, False, 0.75, XS)) == XV.S_UNDECIDED
    assert int(XV.set_code_counts(3, 3, 1, False, 0.5, XS)) == XV.S_UNDECIDED


def test_stop_machine_first():
    assert int(XV.set_code(np.array([P] * 6), True, 0.5, XS)) == XV.S_STOP_MACHINE
    assert int(XV.set_code(np.array([P] * 5 + [I]), False, 0.5, XS)) == XV.S_STOP_MACHINE


def test_p_set_one_equals_w_overall_code_when_n_at_least_4():
    rng = np.random.default_rng(0)
    for k in range(4, 9):
        fin = rng.choice([P, F, B], size=(4000, k), p=[0.7, 0.2, 0.1])
        mach = rng.random(4000) < 0.05
        n = XV.counts(fin)[0]
        x = XV.set_code(fin, mach, 1.0, XS)
        w = WV.overall_code(fin, mach, XS.min_gate_pairs)
        assert np.array_equal(x[n >= 4], w[n >= 4])


def test_p_set_broadcasts():
    fin = np.array([[P, P, P, F, F, F]])
    code = XV.set_code(fin, np.array([False]), np.asarray(XS.p_set_grid)[:, None], XS)
    assert code.shape == (5, 1) and code[:, 0].tolist() == [XV.S_PASS] + [XV.S_FAIL] * 4


def test_m_needed_table_x913():
    t = XV.m_needed_table(XS)
    assert t["k"] == [4, 5, 6, 7, 8]
    assert t["m"] == {"0.5": [3, 3, 3, 4, 4], "0.625": [3, 4, 4, 5, 5], "0.75": [3, 4, 5, 6, 6],
                      "0.875": [4, 5, 6, 7, 7], "1.0": [4, 5, 6, 7, 8]}
    # Reading 12: X.9.1.3 lists k 4's duplicates as 0.5 · 0.625 · 0.75 only; by its own m table 0.875 and 1.0 also
    # share m = 4 at k 4 — the code marks every equal cell
    assert t["duplicates"] == {"4": [[0.5, 0.625, 0.75], [0.875, 1.0]], "5": [[0.625, 0.75], [0.875, 1.0]],
                               "6": [[0.875, 1.0]], "7": [[0.875, 1.0]], "8": []}
