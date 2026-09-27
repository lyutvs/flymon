import math
import pytest
from flymon.rescope import rules as R
from flymon.rescope.spec import SPEC
from tests.rescope.rescope_fixtures import records

def test_specific_learning_passes():
    v = R.pair_verdict(records(dx=12, dy=0, noise=2), "a", SPEC, "p1000")
    assert v["status"] == R.PASS and v["spill_median"] <= 0.5

def test_presentation_drift_only_fails():
    v = R.pair_verdict(records(dx=0, dy=0, noise=2), "a", SPEC, "p1000")
    assert v["status"] == R.FAIL

def test_full_generalisation_fails_on_spill():
    v = R.pair_verdict(records(dx=12, dy=12, noise=2), "a", SPEC, "p1000")
    assert v["status"] == R.FAIL and v["spill_median"] > 0.5

def test_y_reverse_spill_fails():  # X +small, Y moves the other way: clipping must not pass it
    v = R.pair_verdict(records(dx=1, dy=-10, noise=0.5), "a", SPEC, "p1000")
    assert v["status"] == R.FAIL

def test_x_worse_but_y_reverse_fails():  # e_X < 0 while assoc is large: only the e_X <= 0 -> inf rule catches it
    v = R.pair_verdict(records(dx=-0.5, dy=-10, noise=0.5), "a", SPEC, "p1000")
    assert v["status"] == R.FAIL and v["assoc_median"] >= 1.0

def test_x_not_moving_is_infinite_spill():
    st = R.fly_stats(records(dx=-3, dy=0, noise=1), "a", SPEC, 0)
    assert st["e_x"] <= 0 and math.isinf(st["spill"])

def test_null_bigger_than_assoc_fails():
    v = R.pair_verdict(records(dx=6, null=30, noise=2), "a", SPEC, "p1000")
    assert v["status"] == R.FAIL and v["null_max"] >= v["assoc_median"]

def test_naive_negative_level_still_passes():  # no level gate: X-only learning over a negative naive dV passes
    recs = records(dx=10, noise=2)
    for r in recs:  # make X naively much more P-driven than Y (naive dV strongly negative)
        r["counts"]["a"][SPEC.p_type] += 20
    v = R.pair_verdict(recs, "a", SPEC, "p1000")
    assert v["status"] == R.PASS and v["level_median"] < 1.0

def test_x_rule_uses_mbon05_median():
    recs = records(x="b", p_naive=30)
    for r in recs:
        if r["brain"] == "Rr" and r["stage"] == "pre":
            r["counts"]["b"][SPEC.p_type] = 40
    assert R.choose_x(recs, SPEC) == "b"
    assert R.choose_x(records(x="a", x_bias=0), SPEC) == SPEC.x_tie  # equal medians -> tie rule

def test_x_mismatch_invalid():
    v = R.pair_verdict(records(x="a", dx=12, noise=2), "b", SPEC, "p1000")
    assert v["status"] == R.INVALID

def test_floor_not_constructible():
    v = R.pair_verdict(records(p_naive=3, dx=1, noise=0.5), "a", SPEC, "p1000")  # X 4, Y 3 (x_bias 1)
    assert v["status"] == R.NOT_CONSTRUCTIBLE

def test_missing_record_invalid():
    v = R.pair_verdict(records(dx=12, noise=2, drop=5), "a", SPEC, "p1000")
    assert v["status"] == R.INVALID

def test_noplast_move_stop_machine():
    v = R.pair_verdict(records(dx=12, noise=2, noplast_move=1), "a", SPEC, "p1000")
    assert v["status"] == R.STOP_MACHINE

def test_control_sign_and_qualification():
    good = records(dx=12, noise=2, pair="seed0")
    assert R.control_verdict(good, "a", SPEC, qualified=True)["status"] == R.OK
    assert R.control_verdict(good, "a", SPEC, qualified=False)["status"] == R.STOP_CONTROL_INVALID
    bad = records(dx=-12, noise=2, pair="seed0")
    assert R.control_verdict(bad, "a", SPEC, qualified=True)["status"] == R.STOP_PROTOCOL

def test_threshold_table():
    assert [R.threshold_t(m, SPEC) for m in (4, 5, 6)] == [4, 4, 5]
    assert R.false_pass(4, 4, 0.2, 0.3) == pytest.approx(0.033, abs=1e-3)
    assert R.false_pass(5, 6, 0.2, 0.3) == pytest.approx(0.052, abs=1e-3)

def _p(status): return {"status": status}

def test_overall_propagation():
    ok = {"status": R.OK}
    four_pass = {f"p{1000+i}": _p(R.PASS) for i in range(4)}
    assert R.overall(ok, four_pass, SPEC)["status"] == R.PASS
    three = dict(four_pass, p1003=_p(R.FAIL))
    assert R.overall(ok, three, SPEC)["status"] == R.FAIL          # m = 4 needs 4/4
    assert R.overall(ok, dict(four_pass, p1004=_p(R.INVALID)), SPEC)["status"] == R.INVALID
    assert R.overall(ok, dict(four_pass, p1004=_p(R.NOT_CONSTRUCTIBLE)), SPEC)["status"] == R.NOT_CONSTRUCTIBLE
    assert R.overall(ok, dict(four_pass, p1004=_p(R.STOP_MACHINE)), SPEC)["status"] == R.STOP_MACHINE
    assert R.overall({"status": R.STOP_PROTOCOL}, four_pass, SPEC)["status"] == R.STOP_PROTOCOL
    assert R.overall(ok, {k: four_pass[k] for k in list(four_pass)[:3]}, SPEC)["status"] == R.STOP_FEW_PAIRS
