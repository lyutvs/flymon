import numpy as np
import pytest

from flymon.brain.circuits import Populations, compartments
from flymon.brain.config import Params
from flymon.brain.engine_cpu import Engine
from flymon.brain.plasticity import Plasticity


def _setup(synthetic_connectome, **kw):
    c = synthetic_connectome()
    p = Params(noise_mv=0.0, min_weight=1, balance_hemispheres=False, **{"learn_rate": 0.05, **kw})
    pops = Populations.from_connectome(c)
    eng = Engine(c, pops, p, seed=3)
    comps = compartments(c, pops, p.core_frac)
    pl = Plasticity(eng, pops, comps)
    return c, pops, eng, pl


def _drive_kcs(eng, pops, k=10, mv=60.0):
    eng.set_ext(pops.kc[:k], mv)


def test_plastic_edges_are_exactly_kc_to_mbon(synthetic_connectome):
    c, pops, eng, pl = _setup(synthetic_connectome)
    pre = eng.csc.pre_of_edge()[pl.edges]
    post = eng.csc.tgt[pl.edges]
    assert np.isin(pre, pops.kc).all() and np.isin(post, pops.mbon).all()
    assert len(pl.edges) == 40 * 4


def test_no_change_without_dopamine(synthetic_connectome):
    c, pops, eng, pl = _setup(synthetic_connectome)
    _drive_kcs(eng, pops)
    eng.run(300)
    assert pl.weights_frac() == pytest.approx(1.0)


def test_no_change_without_kc_activity(synthetic_connectome):
    c, pops, eng, pl = _setup(synthetic_connectome)
    pl.drive_dan("PAM08", 70.0)
    eng.run(300)
    assert pl.weights_frac() == pytest.approx(1.0)


def test_coincidence_depresses_only_taught_compartment(synthetic_connectome):
    c, pops, eng, pl = _setup(synthetic_connectome)
    _drive_kcs(eng, pops)
    pl.drive_dan("PAM08", 70.0)
    eng.run(400)
    pam_core = compartments(c, pops, 0.2)["PAM08"].core
    ppl_core = compartments(c, pops, 0.2)["PPL105"].core
    assert pl.weights_frac_by_mbon_set(pam_core) < 0.99
    assert pl.weights_frac_by_mbon_set(ppl_core) == pytest.approx(1.0)


def test_floor_and_disabled(synthetic_connectome):
    c, pops, eng, pl = _setup(synthetic_connectome, learn_rate=0.5, min_weight_frac=0.2)
    _drive_kcs(eng, pops, k=40, mv=80.0)
    pl.drive_dan("PAM08", 70.0)
    eng.run(2000)
    w = eng.csc.w[pl.edges]
    assert (w / pl.w0 >= 0.2 - 1e-6).all()
    pam_core = compartments(c, pops, 0.2)["PAM08"].core
    assert pl.weights_frac_by_mbon_set(pam_core) == pytest.approx(0.2, abs=1e-3)   # taught edges hit the floor
    assert pl.weights_frac() < 0.7                                                  # untaught PPL core edges stay at 1.0
    pl.set_enabled(False)
    before = eng.csc.w[pl.edges].copy()
    eng.run(200)
    np.testing.assert_array_equal(eng.csc.w[pl.edges], before)


def test_recover_pulse_moves_toward_w0(synthetic_connectome):
    c, pops, eng, pl = _setup(synthetic_connectome, learn_rate=0.5, recovery_per_pulse=0.5)
    _drive_kcs(eng, pops, k=40, mv=80.0)
    pl.drive_dan("PAM08", 70.0)
    eng.run(1000)
    f0 = pl.weights_frac()
    pl.recover_pulse()
    f1 = pl.weights_frac()
    assert f1 == pytest.approx(f0 + (1.0 - f0) * 0.5, abs=1e-6)


def test_reset_traces_zeroes_traces_but_keeps_weights(synthetic_connectome):
    c, pops, eng, pl = _setup(synthetic_connectome, learn_rate=0.5)
    _drive_kcs(eng, pops, k=40, mv=80.0)
    pl.drive_dan("PAM08", 70.0)
    eng.run(500)
    assert pl.kc_trace.max() > 0 and pl.da.max() > 0
    w = eng.csc.w[pl.edges].copy()
    pl.reset_traces()
    assert pl.kc_trace.max() == 0 and pl.da.max() == 0 and pl.da_base.max() == 0
    np.testing.assert_array_equal(eng.csc.w[pl.edges], w)


def test_reset_weights_restores_w0(synthetic_connectome):
    c, pops, eng, pl = _setup(synthetic_connectome, learn_rate=0.5)
    _drive_kcs(eng, pops, k=40, mv=80.0)
    pl.drive_dan("PPL105", 70.0)
    eng.run(500)
    assert pl.weights_frac() < 1.0
    pl.reset_weights()
    assert pl.weights_frac() == pytest.approx(1.0)
