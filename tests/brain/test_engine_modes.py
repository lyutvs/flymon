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
