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


# ---- spec 10.4 item 6: recorded items (final-review finding 1) ---------------------------------------------
def test_recorded_items_match_the_definitions():
    import numpy as np
    from flymon.brain.b_rules import dprime, v_of
    from flymon.rescope import rules
    from flymon.rescope.spec import SPEC
    from .rescope_fixtures import records
    recs = records(x="a", dx=8, dy=1, null=2, noise=3, seed=4)
    rec = rules.recorded_items(recs, "a", SPEC)
    idx = {(r["brain"], r["fly"], r["stage"], r["seed"]): r["counts"] for r in recs}
    dv = lambda b, f, st: np.array([v_of(idx[(b, f, st, s)]["a"], SPEC) - v_of(idx[(b, f, st, s)]["b"], SPEC)
                                    for s in SPEC.probe_seeds("p1000", f)])
    flies = range(SPEC.n_flies)
    assert rec["flies"] == list(flies)
    assert rec["naive_dprime"]["pooled"] == dprime(np.concatenate([dv("Rr", f, "pre") for f in flies]))
    assert rec["naive_dprime"]["n_pooled"] == SPEC.n_flies * SPEC.n_probe
    assert rec["naive_dprime"]["per_fly"][2] == dprime(dv("Rr", 2, "pre"))
    c = lambda d: float(np.mean(np.where(d > 0, 1.0, np.where(d == 0, 0.5, 0.0))))
    for b in ("Rr", "N", "N2"):
        for st in ("pre", "S1"):
            want = [c(dv(b, f, st)) for f in flies]
            assert rec["choice"][b][st]["per_fly"] == want and rec["choice"][b][st]["median"] == float(np.median(want))
    d = [c(dv("Rr", f, "S1")) - c(dv("N", f, "S1")) for f in flies]
    assert rec["choice"]["rr_minus_n_S1"]["per_fly"] == d
    for b in ("N", "N2"):
        sc = [dprime(dv(b, f, "S1") - dv(b, f, "pre")) for f in flies]
        assert rec["self_change"][b]["per_fly"] == sc
        assert rec["self_change"][b]["median"] == float(np.median(sc)) and rec["self_change"][b]["max"] == max(sc)
    assert rec["xcore"] is None


def test_recorded_choice_ties_are_half_and_xcore():
    import numpy as np
    from flymon.rescope import rules
    from flymon.rescope.spec import SPEC
    from .rescope_fixtures import records
    rec = rules.recorded_items(records(x="a", x_bias=0), "a", SPEC,                    # V(X) == V(Y) everywhere
                               xcore={"Rr": [np.array([0.1, 0.5, 1.0])] * 2, "N": [np.array([1.0, 1.0, 0.2])] * 2},
                               floor_frac=0.2)
    assert rec["choice"]["Rr"]["pre"]["median"] == 0.5 and rec["choice"]["N"]["S1"]["per_fly"] == [0.5] * SPEC.n_flies
    xc = rec["xcore"]
    assert xc["Rr"]["median"] == 0.5 and xc["N"]["median"] == 1.0 and xc["ratio_rr_over_n"] == 0.5
    assert xc["Rr"]["floor_contact"] == xc["N"]["floor_contact"] == 1 / 3 and xc["n_edges"] == 3


def test_recorded_items_do_not_change_the_verdict():
    from flymon.rescope import rules
    from flymon.rescope.spec import SPEC
    from .rescope_fixtures import records
    recs = records(x="a", dx=8, noise=3, seed=4)
    before = rules.pair_verdict(recs, "a", SPEC, "p1000")
    rules.recorded_items(recs, "a", SPEC)
    assert rules.pair_verdict(recs, "a", SPEC, "p1000") == before


def test_recorded_floor_contact_counts_float32_floored_edges():
    """The engine floors in float32 (np.maximum(w, w0 * min_weight_frac)), so a floored edge's w/w0 is 0.2 (1 +- ~6e-8);
    every one of them must count as floor contact."""
    import numpy as np
    from flymon.rescope import rules
    from flymon.rescope.spec import SPEC
    from .rescope_fixtures import records
    rng = np.random.default_rng(7)
    w0 = rng.uniform(0.1, 3.0, 2000).astype(np.float32)
    floored = np.maximum(np.float32(1e-6) * w0, w0 * np.float32(0.2)).astype(np.float32)
    r_floor = floored.astype(float) / w0.astype(float)
    assert (r_floor > 0.2).any()                                 # the float32 floor sits above 0.2 on some edges
    free = (np.float32(0.5) * w0).astype(float) / w0.astype(float)
    rec = rules.recorded_items(records(x="a"), "a", SPEC, xcore={"Rr": [r_floor, np.concatenate([r_floor, free])],
                                                                  "N": [free, free]}, floor_frac=0.2)
    assert rec["xcore"]["Rr"]["floor_contact_per_fly"] == [1.0, 0.5]
    assert rec["xcore"]["N"]["floor_contact_per_fly"] == [0.0, 0.0]


def test_silent_states_recorded_per_probe():
    """Spec 10.3 amendment 2026-09-28: every recorded probe carries its state, silent := MBON05 count < 5 for that
    odour; no verdict effect."""
    recs = records(dx=12, noise=2)
    tgt = next(r for r in recs if r["brain"] == "N" and r["stage"] == "S1" and r["fly"] == 1)
    tgt["counts"]["b"][SPEC.p_type] = 4
    edge = next(r for r in recs if r["brain"] == "Rr" and r["stage"] == "pre" and r["fly"] == 0)
    edge["counts"]["a"][SPEC.p_type] = 5
    s = R.silent_states(recs, SPEC)
    assert set(s) == {r["brain"] for r in recs} and set(s["Rr"]) == {"pre", "S1"}
    nb = s["N"]["S1"]["b"]
    i = nb["flies"].index(1)
    assert nb["state"][i][nb["seeds"][i].index(tgt["seed"])] == "silent" and nb["n_silent"] == 1
    assert nb["silent_share"] == 1 / nb["n"] and nb["n"] == sum(len(v) for v in nb["seeds"])
    assert s["Rr"]["pre"]["a"]["n_silent"] == 0                       # a count of exactly 5 is active
    assert s["N"]["S1"]["a"]["n_silent"] == 0
