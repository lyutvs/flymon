import numpy as np
import pytest

from flymon.brain.circuits import Populations, compartments
from flymon.brain.config import Params
from flymon.brain.engine_cpu import Engine
from flymon.brain.plasticity import Plasticity
from flymon.brain.presentation import decide, reinforce

A = {"ORN_DM1": 1.0, "ORN_DA1": 1.0}
B = {"ORN_VA2": 1.0, "ORN_DM6": 1.0}


def _setup(synthetic_connectome, **kw):
    c = synthetic_connectome(disjoint_kc=True)
    p = Params(**{"noise_mv": 0.15, "min_weight": 1, "balance_hemispheres": False, "kc_thresh": 0.5, "learn_rate": 0.05, **kw})
    pops = Populations.from_connectome(c)
    eng = Engine(c, pops, p, seed=0)
    pl = Plasticity(eng, pops, compartments(c, pops, p.core_frac))
    return c, pops, eng, pl


def test_decide_pairs_noise_across_candidates(synthetic_connectome):
    c, pops, eng, pl = _setup(synthetic_connectome)
    counts = decide(eng, pl, pops, [A, A, B], 1.0, seed=5, settle_ms=50, read_ms=300)
    assert counts.shape == (3, c.N)
    np.testing.assert_array_equal(counts[0], counts[1])          # same odour, same seed -> identical spikes
    assert (counts[2] != counts[0]).any()                         # a different odour differs
    assert counts[:, pops.kc].sum() > 0


def test_decide_is_deterministic_and_leaves_weights_and_enable_flag(synthetic_connectome):
    c, pops, eng, pl = _setup(synthetic_connectome)
    pl.drive_dan("PAM08", 70.0)                                    # a stale drive must not leak into a decision
    a = decide(eng, pl, pops, [A, B], 1.0, seed=3, settle_ms=50, read_ms=300)
    b = decide(eng, pl, pops, [A, B], 1.0, seed=3, settle_ms=50, read_ms=300)
    np.testing.assert_array_equal(a, b)
    assert pl.weights_frac() == pytest.approx(1.0)
    assert pl.enabled is True
    pl.set_enabled(False)
    decide(eng, pl, pops, [A], 1.0, seed=3, settle_ms=10, read_ms=10)
    assert pl.enabled is False


def test_decide_idx_subset_and_empty(synthetic_connectome):
    c, pops, eng, pl = _setup(synthetic_connectome)
    counts = decide(eng, pl, pops, [A, B], 1.0, seed=1, settle_ms=10, read_ms=50, idx=pops.mbon)
    assert counts.shape == (2, len(pops.mbon))
    with pytest.raises(ValueError, match="at least one candidate"):
        decide(eng, pl, pops, [], 1.0, seed=1)


def test_reinforce_depresses_only_with_dan_and_when_enabled(synthetic_connectome):
    c, pops, eng, pl = _setup(synthetic_connectome)
    reinforce(eng, pl, pops, A, 1.0, "PAM08", pulse_ms=300, seed=1, settle_ms=50, gap_ms=20)
    assert pl.weights_frac() < 1.0
    pl.reset_weights()
    reinforce(eng, pl, pops, A, 1.0, None, pulse_ms=300, seed=1, settle_ms=50, gap_ms=20)
    assert pl.weights_frac() == pytest.approx(1.0)
    reinforce(eng, pl, pops, A, 1.0, "PAM08", pulse_ms=300, seed=1, settle_ms=50, gap_ms=20, enabled=False)
    assert pl.weights_frac() == pytest.approx(1.0)
    assert pl.enabled is False


def test_reinforce_recovery_and_unknown_dan(synthetic_connectome):
    c, pops, eng, pl = _setup(synthetic_connectome, recovery_per_pulse=1.0)
    reinforce(eng, pl, pops, A, 1.0, "PAM08", pulse_ms=300, seed=1, settle_ms=50, gap_ms=20)
    assert pl.weights_frac() == pytest.approx(1.0)               # full recovery after the pulse
    with pytest.raises(ValueError, match="unknown DAN"):
        reinforce(eng, pl, pops, A, 1.0, "PAM99", pulse_ms=10, seed=1)


def test_reinforce_cleans_up_when_the_pulse_fails(synthetic_connectome, monkeypatch):
    c, pops, eng, pl = _setup(synthetic_connectome)
    cells = pl.types["PAM08"][0]
    calls = {"n": 0}
    real_run = eng.run

    def failing_run(ms, count_idx=None):
        calls["n"] += 1
        if calls["n"] == 2:                     # the pulse window (1 = settle)
            raise RuntimeError("boom")
        return real_run(ms, count_idx)

    monkeypatch.setattr(eng, "run", failing_run)
    with pytest.raises(RuntimeError, match="boom"):
        reinforce(eng, pl, pops, A, 1.0, "PAM08", pulse_ms=100, seed=1, settle_ms=20, gap_ms=10)
    np.testing.assert_array_equal(eng.ext[cells], eng.ext0[cells])     # DAN quiet again
    assert eng.drive_hz.max() == 0.0                                    # odour off
    assert pl.weights_frac() == pytest.approx(1.0)
