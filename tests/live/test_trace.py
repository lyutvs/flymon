"""WebSink speaks the viz.py sink interface and turns one presentation into one trace event."""
import inspect

import numpy as np

from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.engine_cpu import Engine
from flymon.brain.viz import RerunSink, SpikeTap
from flymon.live.sink import RecordingSink
from flymon.live.trace import WebSink


def test_same_method_signatures_as_rerun_sink():
    for name in ("scalar", "bars", "text"):
        ours = list(inspect.signature(getattr(WebSink, name)).parameters)
        theirs = list(inspect.signature(getattr(RerunSink, name)).parameters)
        assert ours == theirs, name


def test_flush_sends_one_trace_and_empties_the_buffer():
    rec = RecordingSink()
    web = WebSink(rec, "fm-a")
    assert web.flush("b", 1, "decide", 0) is None and rec.events == []
    web.scalar("rate_hz/kc", 100, np.float32(4.5))
    web.scalar("rate_hz/kc", 200, 5)
    web.bars("cell_hz/mbon", 200, np.array([1, 2, 3], np.int64))
    web.text("events", 150, "odour on")
    ev = web.flush("battle-gen1ou-3", 7, "decide", 1)
    assert rec.events == [ev]
    assert {k: ev[k] for k in ("type", "fly", "battle_tag", "turn", "phase", "slot")} == \
        {"type": "trace", "fly": "fm-a", "battle_tag": "battle-gen1ou-3", "turn": 7, "phase": "decide", "slot": 1}
    assert ev["series"] == [
        {"path": "rate_hz/kc", "kind": "scalar", "points": [[100.0, 4.5], [200.0, 5.0]]},
        {"path": "cell_hz/mbon", "kind": "bars", "points": [[200.0, [1.0, 2.0, 3.0]]]},
        {"path": "events", "kind": "text", "points": [[150.0, "odour on"]]},
    ]
    assert web.flush("battle-gen1ou-3", 7, "decide", 2) is None
    assert len(rec.events) == 1


def _engine(synthetic_connectome):
    c = synthetic_connectome()
    p = Params(noise_mv=0.15, min_weight=1, balance_hemispheres=False, learn_rate=0.05, kc_thresh=0.5)
    pops = Populations.from_connectome(c)
    return pops, Engine(c, pops, p, seed=0)


def test_tapping_into_a_web_sink_leaves_spikes_bit_identical(synthetic_connectome):
    runs = []
    for tapped in (False, True):
        pops, eng = _engine(synthetic_connectome)
        seen = []
        eng.on_step = lambda e, fired, seen=seen: seen.append(fired.copy())
        rec = RecordingSink()
        web = WebSink(rec, "fm-a")
        if tapped:
            SpikeTap(eng, {"kc": pops.kc, "mbon": pops.mbon}, web, every=10, per_cell={"mbon": pops.mbon})
        eng.set_ext(pops.kc[:10], 60.0)
        eng.run(50)
        runs.append((seen, web.flush("b", 1, "decide", 0)))
    (plain, none), (observed, ev) = runs
    assert none is None
    assert len(plain) == len(observed) == 50
    assert all(np.array_equal(a, b) for a, b in zip(plain, observed))
    assert [len(s["points"]) for s in ev["series"]] == [5, 5, 5]
    assert np.concatenate(observed).size > 0      # the stimulus really made spikes
