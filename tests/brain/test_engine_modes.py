"""M0d engine modes (spec appendix H.2): graded APL, ORN->PN depression, homeostatic KC thresholds.
Every default must keep the M0c engine bit-identical."""
import hashlib

import numpy as np
import pytest

from flymon.brain.circuits import Populations, compartments
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.engine_cpu import Engine
from flymon.brain.fly_pool import FlyPool, FlySpec
from flymon.brain.plasticity import Plasticity
from flymon.brain.presentation import decide

BASE = dict(noise_mv=0.0, min_weight=1, balance_hemispheres=False, mbon_hold_frac=0.0)


def _engine(synthetic_connectome, **kw):
    c = synthetic_connectome()
    pops = Populations.from_connectome(c)
    return Engine(c, pops, Params(**{**BASE, **kw}), seed=1), c, pops


def _reference_step(eng):
    """The M0c Engine.step, frozen verbatim before the M0d modes were added."""
    p = eng.p
    arrived = eng.delay.popleft()
    if arrived.size:
        eng.g += eng.propagate(arrived)
    dt_m = p.dt / p.tau_m
    dv = (-eng.v + eng.g + eng.ext) * dt_m
    if p.noise_mv:
        dv += eng.rng.standard_normal(eng.N, dtype=np.float32) * p.noise_mv
    free = eng.refrac <= 0
    eng.v = np.where(free, eng.v + dv, eng.v)
    eng.refrac = np.where(free, eng.refrac, eng.refrac - 1)
    np.maximum(eng.v, -p.v_thresh, out=eng.v)
    eng.g *= (1.0 - p.dt / p.tau_syn)
    spk = (eng.v >= eng.v_th) & free
    ri = eng.receptor_idx
    hz = eng.drive_hz[ri]
    pois = (eng.rng.random(ri.size) < hz * (p.dt / 1000.0)) & free[ri]
    spk[ri] = pois
    fired = np.flatnonzero(spk)
    eng.v[fired] = p.v_reset
    eng.refrac[fired] = p.refrac_steps()
    eng.last = fired
    eng.delay.append(fired)
    eng.t_ms += p.dt
    if eng.on_step is not None:
        eng.on_step(eng, fired)
    return fired


# ---- defaults -----------------------------------------------------------------------------------------------
def test_mode_defaults_are_the_m0c_engine():
    p = Params()
    assert (p.apl_mode, p.orn_std, p.kc_thresh_mode) == ("spiking", False, "pn_norm")
    assert (p.apl_v_mid, p.apl_slope, p.orn_std_f, p.orn_std_tau_ms) == (11.0, 5.0, 0.78, 893.0)


def test_default_step_is_bit_identical_to_the_m0c_step(synthetic_connectome):
    kw = dict(noise_mv=0.15, min_weight=1, balance_hemispheres=False, kc_thresh=0.5)
    c = synthetic_connectome(disjoint_kc=True)
    pops = Populations.from_connectome(c)
    a, b = Engine(c, pops, Params(**kw), seed=3), Engine(c, pops, Params(**kw), seed=3)
    orn = np.flatnonzero(c.cls == "olfactory")
    for e in (a, b):
        e.set_drive_hz(orn, 180.0)
    for _ in range(400):
        np.testing.assert_array_equal(a.step(), _reference_step(b))
    np.testing.assert_array_equal(a.v, b.v)
    np.testing.assert_array_equal(a.g, b.g)


def test_unknown_modes_are_rejected(synthetic_connectome):
    with pytest.raises(ValueError, match="apl_mode"):
        _engine(synthetic_connectome, apl_mode="local")
    with pytest.raises(ValueError, match="kc_thresh_mode"):
        _engine(synthetic_connectome, kc_thresh_mode="adaptive")
    with pytest.raises(ValueError, match="orn_std_f"):
        _engine(synthetic_connectome, orn_std=True, orn_std_f=1.5)
    with pytest.raises(ValueError, match="apl_r_max"):
        _engine(synthetic_connectome, apl_mode="graded", apl_r_max=0.0)


# ---- graded APL ---------------------------------------------------------------------------------------------
def test_graded_apl_never_spikes_and_the_spiking_control_does(synthetic_connectome):
    for mode, expect_spikes in (("spiking", True), ("graded", False)):
        eng, c, pops = _engine(synthetic_connectome, apl_mode=mode)
        eng.set_ext(pops.apl, 1000.0)
        fired_apl = any(np.isin(pops.apl, eng.step()).any() for _ in range(30))
        assert fired_apl is expect_spikes, mode


def test_apl_release_is_the_declared_sigmoid(synthetic_connectome):
    eng, c, pops = _engine(synthetic_connectome, apl_mode="graded", apl_r_max=0.4)
    v = np.array([-7.0, 0.0, 11.0, 30.0], np.float32)
    np.testing.assert_allclose(eng.apl_release(v), 0.4 / (1.0 + np.exp(-(v - 11.0) / 5.0)), rtol=1e-6)
    assert eng.apl_release(np.array([11.0], np.float32))[0] == pytest.approx(0.2)


def test_graded_release_is_queued_from_the_updated_membrane_and_delivered_after_the_delay(synthetic_connectome):
    eng, c, pops = _engine(synthetic_connectome, apl_mode="graded", apl_r_max=0.5)
    apl = int(pops.apl[0])
    kc = int(pops.kc[0])
    ptr, tgt, w = eng.csc.ptr, eng.csc.tgt, eng.csc.w
    w_apl_kc = float(w[ptr[apl]:ptr[apl + 1]][tgt[ptr[apl]:ptr[apl + 1]] == kc].sum())
    assert w_apl_kc < 0                                       # GABA, scaled by apl_scale
    eng.set_ext([apl], 100.0)
    eng.step()                                                # step 1: v_apl = 5 mV, release queued
    r1 = float(eng.apl_release(eng.v[[apl]])[0])
    assert eng._apl_release[-1][0] == pytest.approx(r1)
    assert eng.g[kc] == 0.0
    eng.step()                                                # step 2: nothing has arrived yet
    assert eng.g[kc] == 0.0
    eng.step()                                                # step 3: release of step 1 arrives, then tau_syn decay
    assert eng.g[kc] == pytest.approx(w_apl_kc * r1 * (1 - 1 / 5.0), rel=1e-5)


def test_graded_apl_rejects_repeated_out_edge_targets(synthetic_connectome):
    c0 = synthetic_connectome()
    pops = Populations.from_connectome(c0)
    apl, kc = int(pops.apl[0]), int(pops.kc[0])
    c = Connectome(bodyId=c0.bodyId, type=c0.type, cls=c0.cls, sc=c0.sc, nt=c0.nt, sign=c0.sign, side=c0.side,
                   pre=np.append(c0.pre, np.int32(apl)), post=np.append(c0.post, np.int32(kc)), w=np.append(c0.w, np.int32(5)))
    Engine(c, pops, Params(**BASE), seed=1)                   # the spiking engine sums repeated edges in propagate
    with pytest.raises(ValueError, match="unique out-edge targets"):
        Engine(c, pops, Params(**BASE, apl_mode="graded"), seed=1)


def test_graded_reset_clears_the_release_line(synthetic_connectome):
    eng, c, pops = _engine(synthetic_connectome, apl_mode="graded")
    eng.set_ext(pops.apl, 100.0)
    for _ in range(5):
        eng.step()
    eng.reset(seed=2)
    assert len(eng._apl_release) == eng.p.dly_steps()
    assert all((r == 0).all() for r in eng._apl_release)
