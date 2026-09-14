import numpy as np
import pytest

from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.engine_cpu import Engine


def _engine(synthetic_connectome, **kw):
    c = synthetic_connectome()
    # defaults, overridable by kw (the brief passed these positionally-fixed, which clashes with noise_mv=0.15)
    kw = {"noise_mv": 0.0, "min_weight": 1, "balance_hemispheres": False, "mbon_hold_frac": 0.0, **kw}
    p = Params(**kw)
    return Engine(c, Populations.from_connectome(c), p, seed=1), c


def test_constant_drive_follows_discrete_rc_and_fires_at_step_24(synthetic_connectome):
    eng, c = _engine(synthetic_connectome)
    i = int(np.flatnonzero(c.sc == "descending_neuron")[0])   # a DN: no inputs fire in a quiet net
    eng.set_ext([i], 10.0)
    v_prev, fired_at = 0.0, None
    for n in range(1, 40):
        spk = eng.step()
        if i in spk:
            fired_at = n
            break
        expected = v_prev + (10.0 - v_prev) * (1.0 / 20.0)   # Euler, dt=1, tau_m=20
        assert eng.v[i] == pytest.approx(expected, abs=1e-5)
        v_prev = expected
    assert fired_at == 24        # 10*(1-0.95^n) >= 7  ->  n = 24


def test_spike_arrives_after_delay_with_alpha_kernel(synthetic_connectome):
    eng, c = _engine(synthetic_connectome)
    kc = np.flatnonzero(c.cls == "Kenyon_Cell")
    mbon = np.flatnonzero(c.cls == "MBON")
    a, b = int(kc[0]), int(mbon[0])
    w_ab = c.w[(c.pre == a) & (c.post == b)].sum() * eng.p.mv_per_synapse
    eng.set_ext([a], 1000.0)     # dv = 50 mV on step 1 -> fires on step 1
    s1 = eng.step()
    assert a in s1
    assert eng.g[b] == 0.0
    eng.step()                   # delay = 2 steps: arrives on the 3rd step
    assert eng.g[b] == 0.0
    eng.step()                   # arrival is added, then the step's tau_syn decay (x0.8) applies
    assert eng.g[b] == pytest.approx(w_ab * (1 - 1 / 5.0), rel=1e-6)
    g_after_arrival = eng.g[b]
    eng.set_ext([a], 0.0)
    eng.step()
    assert eng.g[b] == pytest.approx(g_after_arrival * (1 - 1 / 5.0), rel=1e-6)   # tau_syn decay


def test_propagate_matches_dense_matrix(synthetic_connectome):
    eng, c = _engine(synthetic_connectome)
    csc = eng.csc
    M = np.zeros((c.N, c.N))
    for i in range(c.N):
        np.add.at(M[:, i], csc.tgt[csc.ptr[i]:csc.ptr[i + 1]], csc.w[csc.ptr[i]:csc.ptr[i + 1]])
    src = np.array([0, 5, 17, 33])
    np.testing.assert_allclose(eng.propagate(src), M[:, src].sum(axis=1), atol=1e-5)


def test_refractory_blocks_immediate_refire(synthetic_connectome):
    eng, c = _engine(synthetic_connectome)
    i = int(np.flatnonzero(c.sc == "descending_neuron")[1])
    eng.set_ext([i], 1000.0)
    fired = [n for n in range(6) if i in eng.step()]
    assert fired == [0, 3]       # fires, 2 refractory steps, fires again


def test_poisson_receptor_rate(synthetic_connectome):
    eng, c = _engine(synthetic_connectome)
    orn = np.flatnonzero(c.cls == "olfactory")
    eng.set_drive_hz(orn, 100.0)
    counts = eng.run(5000)
    rate = counts[orn].mean() / 5.0
    assert 75 <= rate <= 92      # p=0.1/step with 2 refractory steps -> 0.1/(1+0.2) = 83 Hz


def test_reset_with_seed_is_deterministic(synthetic_connectome):
    eng, c = _engine(synthetic_connectome, noise_mv=0.15)
    orn = np.flatnonzero(c.cls == "olfactory")
    eng.set_drive_hz(orn, 150.0)
    eng.reset(seed=7); a = eng.run(300)
    eng.reset(seed=7); b = eng.run(300)
    np.testing.assert_array_equal(a, b)


def test_mbon_hold_sets_tonic_ext(synthetic_connectome):
    c = synthetic_connectome()
    p = Params(noise_mv=0.0, min_weight=1, balance_hemispheres=False)
    eng = Engine(c, Populations.from_connectome(c), p)
    kc = np.flatnonzero(c.cls == "Kenyon_Cell")
    # `array == approx(scalar)` returns a plain bool, so no .all() here (it would AttributeError)
    assert eng.ext[np.flatnonzero(c.cls == "MBON")] == pytest.approx(0.85 * 7.0)
    assert (eng.ext[kc] == 0).all()


def test_kc_threshold_normalised_by_pn_input(synthetic_connectome):
    c = synthetic_connectome()
    p = Params(noise_mv=0.0, min_weight=1, balance_hemispheres=False, kc_thresh=1.5)
    pops = Populations.from_connectome(c)
    eng = Engine(c, pops, p)
    pn_in = np.zeros(c.N)
    m = np.isin(c.pre, pops.alpn) & np.isin(c.post, pops.kc)
    np.add.at(pn_in, c.post[m], c.w[m])
    med = np.median(pn_in[pops.kc])
    expect = 7.0 * 1.5 * np.clip(pn_in[pops.kc] / med, 0.5, 3.0)
    np.testing.assert_allclose(eng.v_th[pops.kc], expect, rtol=1e-6)
    assert (eng.v_th[pops.mbon] == 7.0).all()


def test_on_step_hook_called_after_spikes_are_queued(synthetic_connectome):
    eng, c = _engine(synthetic_connectome)
    i = int(np.flatnonzero(c.sc == "descending_neuron")[0])
    eng.set_ext([i], 1000.0)     # dv = 50 mV -> fires on the first step
    calls, seen = [], []

    def hook(engine, fired):
        seen.append(engine)
        calls.append((engine.t_ms, fired.copy(), len(engine.delay), engine.delay[-1].copy()))

    eng.on_step = hook
    for _ in range(3):
        eng.step()

    assert len(calls) == 3
    assert all(e is eng for e in seen)
    t0, fired0, dlen0, queued0 = calls[0]
    assert i in fired0
    assert t0 == 1.0                                   # the hook runs after t_ms advances
    np.testing.assert_array_equal(queued0, fired0)     # this step's spikes are already queued
    assert [dlen for _, _, dlen, _ in calls] == [eng.p.dly_steps()] * 3
