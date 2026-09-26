"""Spec K.8.3–K.8.5: ranking of ADOPTED engines by N, the gate, and the closing state of every branch."""
import pytest

from flymon.brain.h3_rules import COMBO_ADOPTED, COMBO_DROPPED
from flymon.brain.j_rules import B, COMPUTE_ABORTED, INVALID, SELECTED, STOP_MULTI_TYPE
from flymon.brain.k_rules import (DROPPED_NO_READOUT, NOT_A_JUDGEMENT, SCAN_GO, STOP_NO_QUALIFIED_SETTING,
                                  STOP_NO_TARGET_GAIN, closing_state, scan_outcome, select_order)


def st(g, n=None, s=0.5, status=COMBO_ADOPTED):
    return dict(g=g, status=status, metrics=None if n is None else dict(N=n, S_median=s))


def test_order_by_n_only_adopted():
    s = [st(-0.05, 10), st(-0.1, 30), st(-0.2, status=COMBO_DROPPED), st(-0.4, 20)]
    assert select_order(s, 10.0, 0.01) == [1, 3, 0]


def test_ties_within_one_percent_go_to_larger_s_then_smaller_g():
    s = [st(-0.4, 100.0, s=0.4), st(-0.1, 99.5, s=0.6), st(-0.2, 99.5, s=0.6), st(-0.05, 50.0)]
    assert select_order(s, 10.0, 0.01) == [1, 2, 0, 3]


def test_a_non_positive_c3_drive_stops():
    with pytest.raises(ValueError, match="N_C3"):
        select_order([st(-0.1, 5)], 0.0, 0.01)


def test_scan_outcomes():
    assert scan_outcome([st(-0.1, status=COMBO_DROPPED)], [], 10.0, 1.0) == STOP_NO_QUALIFIED_SETTING
    s = [st(-0.1, 10.0), st(-0.2, 9.0)]
    assert scan_outcome(s, select_order(s, 10.0, 0.01), 10.0, 1.0) == STOP_NO_TARGET_GAIN   # ratio exactly 1.0
    s = [st(-0.1, 10.0001)]
    assert scan_outcome(s, [0], 10.0, 1.0) == SCAN_GO


@pytest.mark.parametrize("judge,state", [
    (dict(outcome=STOP_MULTI_TYPE, reading=None), "stop_multiple_types"),
    (dict(outcome=B, reading=None, reason="no reactive and teachable readout in a pool"), DROPPED_NO_READOUT),
    (dict(outcome=INVALID, reading=None), INVALID),
    (dict(outcome=COMPUTE_ABORTED), COMPUTE_ABORTED),
    (dict(outcome=SELECTED, reading=dict(band=SELECTED)), SELECTED),
    (dict(outcome=B, reading=dict(band="B_Tb")), "B_Tb"),
    (dict(outcome=B, reading=dict(band="B_NO_CONCLUSION")), "B_NO_CONCLUSION"),
    (dict(outcome=B, reading=dict(band="B_Fa")), "B_Fa"),
])
def test_every_branch_has_a_state(judge, state):
    assert closing_state(judge) == state


def test_a_bandless_reading_is_not_a_judgement():
    assert closing_state(dict(outcome=B, reading=dict(band=None))) == NOT_A_JUDGEMENT == "not_a_judgement"
    assert closing_state(dict(outcome=SELECTED, reading=dict(band=None))) == NOT_A_JUDGEMENT


@pytest.mark.parametrize("outcome", [SELECTED, "something_else"])
def test_a_readingless_outcome_other_than_b_raises(outcome):
    with pytest.raises(ValueError, match=outcome):
        closing_state(dict(outcome=outcome, reading=None))
