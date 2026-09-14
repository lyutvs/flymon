"""Live training view: spike-rate tap on the step hook, conditioning event logger, Rerun sink."""
from datetime import timedelta

import numpy as np
import pytest

from flymon.brain.circuits import Populations, compartments
from flymon.brain.conditioning import Readout, run_arm
from flymon.brain.config import Params
from flymon.brain.engine_cpu import Engine
from flymon.brain.plasticity import Plasticity
from flymon.brain.stimuli import design_odor_pair
from flymon.brain.viz import ConditioningViz, SpikeTap, attach_conditioning_viz


class ListSink:
    """Records every sink call as (method, path, t_ms, value)."""

    def __init__(self):
        self.calls = []

    def scalar(self, path, t_ms, value):
        self.calls.append(("scalar", path, t_ms, value))

    def bars(self, path, t_ms, values):
        self.calls.append(("bars", path, t_ms, np.array(values)))

    def text(self, path, t_ms, msg):
        self.calls.append(("text", path, t_ms, msg))

    def at(self, method, path):
        return [(t, v) for m, p, t, v in self.calls if m == method and p == path]


def _engine(synthetic_connectome, **kw):
    c = synthetic_connectome()
    p = Params(**{"noise_mv": 0.15, "min_weight": 1, "balance_hemispheres": False, "learn_rate": 0.05,
                  "kc_thresh": 0.5, **kw})
    pops = Populations.from_connectome(c)
    return c, pops, Engine(c, pops, p, seed=0)


def _plastic(synthetic_connectome, **kw):
    c, pops, eng = _engine(synthetic_connectome, **kw)
    comps = compartments(c, pops, eng.p.core_frac)
    return pops, eng, Plasticity(eng, pops, comps), comps


# ---- SpikeTap -----------------------------------------------------------------------------------
def test_tap_calls_the_hook_it_replaces(synthetic_connectome):
    c, pops, eng = _engine(synthetic_connectome)
    seen = []
    eng.on_step = lambda e, fired: seen.append(fired)
    SpikeTap(eng, {"kc": pops.kc}, ListSink(), every=10)
    eng.run(50)
    assert len(seen) == 50


def test_tap_mean_rates_are_spike_counts_per_window(synthetic_connectome):
    c, pops, eng = _engine(synthetic_connectome)
    seen = []
    eng.on_step = lambda e, fired: seen.append(fired.copy())
    eng.set_ext(pops.kc[:10], 60.0)
    sink = ListSink()
    SpikeTap(eng, {"kc": pops.kc, "mbon": pops.mbon}, sink, every=10)
    eng.run(30)
    window_s = 10 * eng.p.dt / 1000.0
    for name, cells in (("kc", pops.kc), ("mbon", pops.mbon)):
        expect = [np.isin(np.concatenate(seen[i:i + 10]), cells).sum() / (len(cells) * window_s)
                  for i in (0, 10, 20)]
        assert [v for _, v in sink.at("scalar", f"rate_hz/{name}")] == pytest.approx(expect)
    assert sum(v for _, v in sink.at("scalar", "rate_hz/kc")) > 0     # driven KCs fired: not vacuous


def test_tap_per_cell_bars_give_each_cell_its_rate(synthetic_connectome):
    c, pops, eng = _engine(synthetic_connectome)
    seen = []
    eng.on_step = lambda e, fired: seen.append(fired.copy())
    eng.set_ext(pops.kc[:10], 60.0)
    cells = pops.kc[:12]                                   # 10 driven, 2 quiet
    sink = ListSink()
    SpikeTap(eng, {"kc": pops.kc}, sink, every=10, per_cell={"kc": cells})
    eng.run(20)
    window_s = 10 * eng.p.dt / 1000.0
    bars = sink.at("bars", "cell_hz/kc")
    assert len(bars) == 2
    for w, (_, values) in enumerate(bars):
        spikes = np.concatenate(seen[10 * w:10 * w + 10])
        np.testing.assert_allclose(values, [(spikes == i).sum() / window_s for i in cells])
    assert bars[0][1][:10].sum() > 0


def test_tap_time_runs_on_across_engine_reset(synthetic_connectome):
    c, pops, eng = _engine(synthetic_connectome)
    sink = ListSink()
    tap = SpikeTap(eng, {"kc": pops.kc}, sink, every=10)
    eng.run(20)
    eng.reset(5)
    eng.run(20)
    dt = eng.p.dt
    assert [t for t, _ in sink.at("scalar", "rate_hz/kc")] == [10 * dt, 20 * dt, 30 * dt, 40 * dt]
    assert tap.sim_ms == 40 * dt


def test_tap_rejects_empty_group(synthetic_connectome):
    c, pops, eng = _engine(synthetic_connectome)
    with pytest.raises(ValueError, match="empty"):
        SpikeTap(eng, {"nobody": pops.kc[:0]}, ListSink())
    with pytest.raises(ValueError, match="empty"):
        SpikeTap(eng, {"kc": pops.kc}, ListSink(), per_cell={"nobody": pops.kc[:0]})


def test_plasticity_still_learns_under_a_tap(synthetic_connectome):
    pops, eng, pl, comps = _plastic(synthetic_connectome, noise_mv=0.0)
    SpikeTap(eng, {"kc": pops.kc}, ListSink(), every=50)     # installed after Plasticity took the hook
    eng.set_ext(pops.kc[:10], 60.0)
    pl.drive_dan("PAM08", 70.0)
    eng.run(400)
    assert pl.weights_frac_by_mbon_set(comps["PAM08"].core) < 0.99


# ---- ConditioningViz ----------------------------------------------------------------------------
def _arm_with_viz(synthetic_connectome):
    pops, eng, pl, comps = _plastic(synthetic_connectome)
    sink = ListSink()
    tap = SpikeTap(eng, {"kc": pops.kc}, sink, every=50)
    viz = ConditioningViz(sink, tap, pl, channels=("PPL105", "PAM08"))
    ro = Readout.from_compartments(comps)
    a, b = design_odor_pair(pops, k=2, exclude=("ORN_DA1",))
    r = run_arm(eng, pl, pops, ro, a, b, 1.0, seed=2, arm="both", trials=2, present_ms=200, gap_ms=20,
                settle_ms=20, read_ms=100, on_event=viz)
    return sink, pl, comps, r


def test_viz_logs_compartment_weights_and_dopamine_pulses_per_presentation(synthetic_connectome):
    sink, pl, comps, r = _arm_with_viz(synthetic_connectome)
    for name in ("PPL105", "PAM08"):
        w = sink.at("scalar", f"weights_frac/{name}")
        assert len(w) == 4                                                   # 2 trials x CS+/CS-
        assert w[-1][1] == pytest.approx(pl.weights_frac_by_mbon_set(comps[name].core))
    assert [v for _, v in sink.at("scalar", "dan_pulse/PPL105")] == [1.0, 0.0, 1.0, 0.0]
    assert [v for _, v in sink.at("scalar", "dan_pulse/PAM08")] == [0.0, 1.0, 0.0, 1.0]
    times = [t for t, _ in sink.at("scalar", "weights_frac/PAM08")]
    assert times == sorted(times) and len(set(times)) == 4


def test_viz_logs_probe_counts_and_arm_result_on_the_tap_clock(synthetic_connectome):
    sink, pl, comps, r = _arm_with_viz(synthetic_connectome)
    assert [v for _, v in sink.at("scalar", "D_pre/both")] == [r["D_pre"]]
    assert [v for _, v in sink.at("scalar", "D_post/both")] == [r["D_post"]]
    assert [v for _, v in sink.at("scalar", "dD/both")] == [r["dD"]]
    assert len(sink.at("scalar", "probe/pre/plus/A")) == 1 and len(sink.at("scalar", "probe/post/minus/P")) == 1
    texts = [m for _, m in sink.at("text", "events")]
    assert any("both" in m for m in texts)
    rate_times = [t for t, _ in sink.at("scalar", "rate_hz/kc")]
    (end_t, _), = sink.at("scalar", "dD/both")
    assert end_t >= rate_times[-1]                                  # events share the tap's sim clock


def test_viz_rejects_unknown_event(synthetic_connectome):
    pops, eng, pl, comps = _plastic(synthetic_connectome)
    sink = ListSink()
    viz = ConditioningViz(sink, SpikeTap(eng, {"kc": pops.kc}, sink), pl, channels=("PAM08",))
    with pytest.raises(ValueError, match="unknown"):
        viz("mystery")


def test_attach_conditioning_viz_taps_the_mushroom_body(synthetic_connectome):
    pops, eng, pl, comps = _plastic(synthetic_connectome)
    sink = ListSink()
    viz = attach_conditioning_viz(eng, pl, pops, sink, channels=("PPL105", "PAM08"), every=10)
    eng.run(10)
    paths = {p for _, p, _, _ in sink.calls}
    assert {"rate_hz/alpn", "rate_hz/kc", "rate_hz/mbon", "rate_hz/apl",
            "rate_hz/dan/PPL105", "rate_hz/dan/PAM08", "cell_hz/mbon"} <= paths
    assert len(sink.at("bars", "cell_hz/mbon")[0][1]) == len(pops.mbon)
    assert isinstance(viz, ConditioningViz)


# ---- RerunSink ----------------------------------------------------------------------------------
def test_rerun_sink_writes_entities_on_the_sim_timeline(tmp_path):
    rr = pytest.importorskip("rerun")
    from rerun.experimental import RrdReader

    from flymon.brain.viz import RerunSink

    path = tmp_path / "t.rrd"
    rec = rr.RecordingStream("flymon-test")
    rec.save(path)
    sink = RerunSink(rec, prefix="seed3/")
    sink.scalar("rate_hz/kc", 100.0, 2.5)
    sink.bars("cell_hz/mbon", 100.0, np.ones(4))
    sink.text("events", 100.0, "arm both")
    rec.flush()
    rec.disconnect()                                        # closes the file with its footer
    chunks = list(RrdReader(path).stream())
    assert {"/seed3/rate_hz/kc", "/seed3/cell_hz/mbon", "/seed3/events"} <= {c.entity_path for c in chunks}
    batch = next(c for c in chunks if c.entity_path == "/seed3/rate_hz/kc").to_record_batch()
    assert batch.column("sim")[0].as_py() == timedelta(milliseconds=100)
    assert batch.column("Scalars:scalars")[0].as_py() == [2.5]
