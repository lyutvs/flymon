"""The H.4 procedure on a scripted measurer (spec H.4 as amended by H.3a.1, H.3a.9 and H.4a): every branch."""
import numpy as np
import pytest

from flymon.brain import h4_rules as R
from flymon.brain.h4_runner import Context, run_h4
from flymon.brain.h4_spec import SPEC

from h4_scripted import COMBOS, PAIRS, POOLS, SEEDS, Scripted, guard_of


def ctx_for(m, **kw):
    guards = {n: guard_of(m, n) for n in COMBOS}
    return Context(spec=SPEC, combos=dict(COMBOS), pools=POOLS, probe_seeds=SEEDS, expected=list(PAIRS),
                   h3_guard=guards, log=lambda s: None, **kw)


def run(m):
    return run_h4(m, ctx_for(m))


def test_equal_combinations_select_c0_and_every_combination_is_measured_in_order():
    m = Scripted()
    out = run(m)
    assert out["outcome"] == R.SELECTED and out["selection"]["winner"] == "C0" and out["selection"]["engine_unchanged"]
    assert out["selection"]["near"] == ["C0", "C1", "C3"]
    assert [c for c in m.calls if c[0] == "oracle"] == [("oracle", n, (("A", "MA1"), ("P", "MP1"))) for n in COMBOS]
    first_oracle = next(i for i, c in enumerate(m.calls) if c[0] == "oracle")
    assert {c[1] for c in m.calls[:first_oracle] if c[0] == "teach"} == set(COMBOS)      # every reselection first
    c0 = out["combos"]["C0"]
    assert c0["readout"] == {"A": "MA1", "P": "MP1"} and c0["oracle"]["aggregate"]["testable_b"] == 3
    assert set(c0["records"]) == {"naive_floor", "single_type", "report_halves"}
    assert out["selection"]["winner_below_bar"] is False
    x = np.arange(10, 18, dtype=float)
    assert c0["z"]["A"] == (float(x.mean()), float(x.std()))
    assert c0["teach"]["MA1"]["order"] == "ab" and c0["teach"]["MA1"]["teachable"]


def test_the_best_t_b_wins_outside_the_tie_band():
    m = Scripted(testable=lambda n: {"C0": (2, 2), "C1": (5, 2), "C3": (3, 2)}[n])
    out = run(m)
    assert out["selection"]["winner"] == "C1" and out["selection"]["near"] == ["C1", "C3"]


def test_a_combination_without_a_readout_is_dropped_and_gets_no_oracle():
    m = Scripted(teach=lambda n: {"MP1", "MP2"} if n == "C0" else {"MA1", "MA2", "MP1", "MP2"})
    out = run(m)
    assert out["combos"]["C0"]["status"] == R.DROPPED_NO_READOUT and "oracle" not in out["combos"]["C0"]
    assert ("oracle", "C0") not in [c[:2] for c in m.calls]
    assert out["selection"]["eligible"] == ["C1", "C3"] and out["selection"]["winner"] == "C1"


def test_two_passing_types_in_a_pool_stop_the_run():
    m = Scripted(react=lambda n: {"MA1", "MA2", "MP1"} if n == "C1" else {"MA1", "MP1"})
    out = run(m)
    assert out["outcome"] == R.STOP_MULTI_TYPE and list(out["combos"]) == ["C0", "C1"]
    assert out["selection"] is None and not [c for c in m.calls if c[0] == "oracle"]      # before any oracle


def test_invalid_rows_make_the_run_invalid_without_a_selection():
    m = Scripted(rows=lambda n, rows: rows[:-1] if n == "C3" else rows)
    out = run(m)
    assert out["outcome"] == R.INVALID and set(out["invalid"]) == {"C3"} and out["selection"] is None
    assert out["combos"]["C3"]["records"] is None and out["combos"]["C0"]["records"] is not None


def test_malformed_probes_make_the_run_invalid_and_get_no_records():
    def mangle(n, rows):                                   # one pair's R1 has 7 A probes and 8 P probes: dv would raise
        if n != "C1":
            return rows
        r0 = rows[0]
        return [dict(r0, report=dict(r0["report"], R1={"A": r0["report"]["R1"]["A"][:7], "P": r0["report"]["R1"]["P"]}))] \
            + rows[1:]
    out = run(Scripted(rows=mangle))
    assert out["outcome"] == R.INVALID and set(out["invalid"]) == {"C1"} and out["selection"] is None
    assert out["combos"]["C1"]["records"] is None


@pytest.mark.parametrize("testable, outcome", [((2, 2), R.STOP_LOW_T_B), ((5, 1), R.STOP_NO_ELIGIBLE)])
def test_the_stops_for_the_users_decision(testable, outcome):
    out = run(Scripted(testable=lambda n: testable))
    assert out["outcome"] == outcome and out["selection"]["winner"] is None


def test_reactivity_must_be_the_h3_guard():
    m = Scripted()
    ctx = ctx_for(m)
    ctx.h3_guard["C3"]["MA1"] = dict(ctx.h3_guard["C3"]["MA1"], zero_share=0.5)
    with pytest.raises(RuntimeError, match="differs from the H.3 guard"):
        run_h4(m, ctx)
    assert not [c for c in m.calls if c[0] == "oracle"]                              # found before any oracle
