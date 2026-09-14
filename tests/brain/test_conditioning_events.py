"""Training progress events: run_arm/train_block report what they do without changing the result."""
import numpy as np
import pytest

from flymon.brain.circuits import Populations, compartments
from flymon.brain.conditioning import D, Readout, run_arm, train_block
from flymon.brain.config import Params
from flymon.brain.engine_cpu import Engine
from flymon.brain.plasticity import Plasticity
from flymon.brain.stimuli import design_odor_pair


def _setup(synthetic_connectome):
    c = synthetic_connectome()
    p = Params(noise_mv=0.15, min_weight=1, balance_hemispheres=False, learn_rate=0.05, kc_thresh=0.5)
    pops = Populations.from_connectome(c)
    eng = Engine(c, pops, p, seed=0)
    comps = compartments(c, pops, p.core_frac)
    pl = Plasticity(eng, pops, comps)
    ro = Readout.from_compartments(comps)
    a, b = design_odor_pair(pops, k=2, exclude=("ORN_DA1",))
    return pops, eng, pl, ro, a, b


def _recorder():
    events = []
    return events, lambda kind, **fields: events.append((kind, fields))


def test_train_block_reports_each_presentation(synthetic_connectome):
    pops, eng, pl, ro, a, b = _setup(synthetic_connectome)
    events, on_event = _recorder()
    train_block(eng, pl, pops, a, b, 1.0, seed=2, punish="PPL105", reward=None, trials=2, present_ms=20, gap_ms=5,
                on_event=on_event)
    assert events == [("presentation", {"trial": 0, "cs": "plus", "dan": "PPL105"}),
                      ("presentation", {"trial": 0, "cs": "minus", "dan": None}),
                      ("presentation", {"trial": 1, "cs": "plus", "dan": "PPL105"}),
                      ("presentation", {"trial": 1, "cs": "minus", "dan": None})]


def test_run_arm_reports_probes_presentations_and_result(synthetic_connectome):
    pops, eng, pl, ro, a, b = _setup(synthetic_connectome)
    events, on_event = _recorder()
    r = run_arm(eng, pl, pops, ro, a, b, 1.0, seed=2, arm="both", trials=1, present_ms=50, gap_ms=10,
                settle_ms=20, read_ms=100, on_event=on_event)
    assert [k for k, _ in events] == ["arm_start", "probe", "probe", "presentation", "presentation",
                                      "probe", "probe", "arm_end"]
    assert events[0][1] == {"arm": "both", "seed": 2}
    probes = [f for k, f in events if k == "probe"]
    assert [(f["phase"], f["cs"]) for f in probes] == [("pre", "plus"), ("pre", "minus"),
                                                       ("post", "plus"), ("post", "minus")]
    assert D(ro, probes[0], probes[1]) == r["D_pre_disc"]        # events carry the raw counts both
    assert D(ro, probes[2], probes[3]) == r["D_post_disc"]       # indices are computed from
    assert events[-1] == ("arm_end", r)


def test_observing_an_arm_leaves_its_result_bit_identical(synthetic_connectome):
    pops, eng, pl, ro, a, b = _setup(synthetic_connectome)
    kw = dict(trials=2, present_ms=200, gap_ms=20, settle_ms=20, read_ms=100)
    plain = run_arm(eng, pl, pops, ro, a, b, 1.0, seed=3, arm="both", **kw)
    w_plain = eng.csc.w[pl.edges].copy()
    events, on_event = _recorder()
    seen = run_arm(eng, pl, pops, ro, a, b, 1.0, seed=3, arm="both", on_event=on_event, **kw)
    assert seen == plain
    assert plain["weights_frac"] < 1.0                 # training really moved the weights
    np.testing.assert_array_equal(eng.csc.w[pl.edges], w_plain)
