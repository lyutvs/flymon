"""M0d engine modes (spec appendix H.2): graded APL, ORN->PN depression, homeostatic KC thresholds.
Every default must keep the M0c engine bit-identical."""
import hashlib

import numpy as np
import pytest

from flymon.brain.circuits import Populations, compartments
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome, build_csc
from flymon.brain.engine_cpu import Engine
from flymon.brain.fly_pool import FlyPool, FlySpec
from flymon.brain.plasticity import Plasticity
from flymon.brain.presentation import decide
from flymon.brain.thresholds import load_kc_thresholds, save_kc_thresholds

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


# ---- apl_input_scale (spec H.3a.2) --------------------------------------------------------------------------
def _csc_of(synthetic_connectome, **kw):
    c = synthetic_connectome()
    pops = Populations.from_connectome(c)
    return build_csc(c, Params(**{**BASE, **kw}), pops.apl, pops.kc), c, pops


def test_apl_input_scale_defaults_to_one_and_keeps_the_csc_bit_identical(synthetic_connectome):
    assert Params().apl_input_scale == 1.0
    a, _, _ = _csc_of(synthetic_connectome)
    b, _, _ = _csc_of(synthetic_connectome, apl_input_scale=1.0)
    assert np.array_equal(a.w, b.w) and np.array_equal(a.tgt, b.tgt) and np.array_equal(a.ptr, b.ptr)


def test_apl_input_scale_scales_exactly_the_edges_into_apl(synthetic_connectome):
    base, c, pops = _csc_of(synthetic_connectome)
    scaled, _, _ = _csc_of(synthetic_connectome, apl_input_scale=0.25)
    into_apl = np.isin(base.tgt, np.asarray(pops.apl, np.int64))
    assert into_apl.any(), "the synthetic connectome must have edges into APL"
    assert np.array_equal(scaled.w[into_apl], (base.w[into_apl] * np.float32(0.25)).astype(np.float32))
    assert np.array_equal(scaled.w[~into_apl], base.w[~into_apl])


@pytest.mark.parametrize("bad", [0.0, -0.5, 1.5, float("nan"), float("inf")])
def test_apl_input_scale_outside_the_declared_range_is_rejected(synthetic_connectome, bad):
    c = synthetic_connectome()
    pops = Populations.from_connectome(c)
    with pytest.raises(ValueError, match="apl_input_scale"):
        Engine(c, pops, Params(**{**BASE, "apl_input_scale": bad}), seed=1)


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


def test_graded_apl_membrane_keeps_the_floor(synthetic_connectome):
    eng, c, pops = _engine(synthetic_connectome, apl_mode="graded")
    eng.set_ext(pops.apl, -1000.0)
    floor = -eng.p.v_thresh
    for _ in range(5):
        eng.step()
        np.testing.assert_array_equal(eng.v[pops.apl], np.float32(floor))
    r_floor = eng.apl_release(np.full(len(pops.apl), floor, np.float32))
    assert np.isfinite(r_floor).all() and (r_floor > 0).all()
    np.testing.assert_array_equal(eng._apl_release[-1], r_floor)


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


# ---- ORN->PN depression -------------------------------------------------------------------------------------
def test_propagate_gain_scales_each_source(synthetic_connectome):
    eng, c, pops = _engine(synthetic_connectome)
    src = np.array([0, 5, 17])
    gain = np.array([0.5, 1.0, 0.25], np.float32)
    expect = sum(g * eng.propagate(np.array([s])) for s, g in zip(src, gain))
    np.testing.assert_allclose(eng.propagate(src, gain), expect, atol=1e-5)


def test_orn_depression_follows_the_discrete_rule_and_reaches_its_steady_state(synthetic_connectome):
    f, tau = 0.78, 893.0
    eng, c, pops = _engine(synthetic_connectome, orn_std=True, orn_std_f=f, orn_std_tau_ms=tau)
    o = int(np.flatnonzero(c.cls == "olfactory")[0])
    eng.set_drive_hz([o], 1e6)                                # p = 1 whenever not refractory: every 3rd step
    k = 1.0 / tau
    r, gains = 1.0, []
    for _ in range(3000):
        fired = eng.step()
        if o in fired:
            gains.append(float(eng._std_delay[-1][fired == o][0]))
            assert gains[-1] == pytest.approx(r, rel=1e-5)
            r *= f
        r += (1.0 - r) * k
        assert eng._std_r[o] == pytest.approx(r, rel=1e-5)
    a = (1.0 - k) ** 3
    assert gains[-1] == pytest.approx((1.0 - a) / (1.0 - f * a), rel=1e-4)   # delivered gain at steady state


def test_orn_depression_leaves_other_sources_at_full_weight_and_reset_restores_it(synthetic_connectome):
    eng, c, pops = _engine(synthetic_connectome, orn_std=True)
    o = int(np.flatnonzero(c.cls == "olfactory")[0])
    kc = int(pops.kc[0])
    eng.set_drive_hz([o], 1e6)
    eng.set_ext([kc], 1000.0)
    kc_fired = 0
    for _ in range(30):
        fired = eng.step()
        if kc in fired:
            kc_fired += 1
            assert eng._std_delay[-1][fired == kc][0] == 1.0
    assert kc_fired > 0
    assert eng._std_r[o] < 1.0
    eng.reset(seed=4)
    assert (eng._std_r == 1.0).all()
    assert all(x.size == 0 for x in eng._std_delay)


# ---- homeostatic KC thresholds ------------------------------------------------------------------------------
def _theta_file(tmp_path, eng, c, pops, scale=1.2):
    v_th = eng.v_th[pops.kc].astype(np.float32).copy()
    movable = eng.kc_pn_input > 0
    v_th[movable] *= np.float32(scale)
    path = tmp_path / "theta.npz"
    return save_kc_thresholds(path, c.bodyId[pops.kc], v_th), v_th


def test_homeostatic_thresholds_load_into_the_kcs_only(synthetic_connectome, tmp_path):
    ref, c, pops = _engine(synthetic_connectome)
    (path, sha), v_th = _theta_file(tmp_path, ref, c, pops)
    eng, _, _ = _engine(synthetic_connectome, kc_thresh_mode="homeostatic", kc_thresh_file=str(path), kc_thresh_sha256=sha)
    np.testing.assert_array_equal(eng.v_th[pops.kc], v_th)
    others = np.setdiff1d(np.arange(c.N), pops.kc)
    np.testing.assert_array_equal(eng.v_th[others], ref.v_th[others])


def test_homeostatic_file_is_validated(synthetic_connectome, tmp_path):
    ref, c, pops = _engine(synthetic_connectome)
    (path, sha), v_th = _theta_file(tmp_path, ref, c, pops)
    ids = c.bodyId[pops.kc]
    mk = lambda **kw: _engine(synthetic_connectome, kc_thresh_mode="homeostatic", **kw)
    with pytest.raises(ValueError, match="kc_thresh_file"):
        mk()
    with pytest.raises(ValueError, match="sha256"):
        mk(kc_thresh_file=str(path), kc_thresh_sha256="0" * 64)
    with pytest.raises(ValueError, match="sha256"):
        mk(kc_thresh_file=str(path))
    bad = tmp_path / "bad_ids.npz"
    p2, s2 = save_kc_thresholds(bad, ids[::-1], v_th)
    with pytest.raises(ValueError, match="body ids"):
        mk(kc_thresh_file=str(p2), kc_thresh_sha256=s2)
    neg = v_th.copy(); neg[0] = -1.0
    p3, s3 = save_kc_thresholds(tmp_path / "neg.npz", ids, neg)
    with pytest.raises(ValueError, match="positive"):
        mk(kc_thresh_file=str(p3), kc_thresh_sha256=s3)
    short = tmp_path / "short.npz"
    p4, s4 = save_kc_thresholds(short, ids[:-1], v_th[:-1])
    with pytest.raises(ValueError, match="KCs"):
        mk(kc_thresh_file=str(p4), kc_thresh_sha256=s4)


def test_homeostatic_file_must_keep_the_rule_for_kcs_without_pn_input(synthetic_connectome, tmp_path):
    ref, c, pops = _engine(synthetic_connectome)
    zero = np.flatnonzero(ref.kc_pn_input == 0)
    if zero.size == 0:                                        # make one: drop every PN->KC edge onto the first KC
        c0 = synthetic_connectome()
        k0 = int(Populations.from_connectome(c0).kc[0])
        keep = ~(np.isin(c0.pre, Populations.from_connectome(c0).alpn) & (c0.post == k0))
        c = Connectome(bodyId=c0.bodyId, type=c0.type, cls=c0.cls, sc=c0.sc, nt=c0.nt, sign=c0.sign, side=c0.side,
                       pre=c0.pre[keep], post=c0.post[keep], w=c0.w[keep])
        pops = Populations.from_connectome(c)
        ref = Engine(c, pops, Params(**BASE), seed=1)
        zero = np.flatnonzero(ref.kc_pn_input == 0)
    v_th = ref.v_th[pops.kc].astype(np.float32).copy()
    v_th[zero[0]] *= np.float32(2.0)
    path, sha = save_kc_thresholds(tmp_path / "moved.npz", c.bodyId[pops.kc], v_th)
    with pytest.raises(ValueError, match="no PN input"):
        Engine(c, pops, Params(**BASE, kc_thresh_mode="homeostatic", kc_thresh_file=str(path), kc_thresh_sha256=sha))


def test_load_kc_thresholds_returns_float32_in_kc_order(synthetic_connectome, tmp_path):
    ref, c, pops = _engine(synthetic_connectome)
    (path, sha), v_th = _theta_file(tmp_path, ref, c, pops)
    got = load_kc_thresholds(str(path), sha, c.bodyId[pops.kc], ref.v_th[pops.kc], ref.kc_pn_input > 0)
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, v_th)
    assert sha == hashlib.sha256(path.read_bytes()).hexdigest()


# ---- pool = in-process in every new mode --------------------------------------------------------------------
def test_pool_decide_equals_in_process_in_every_new_mode(synthetic_npz, tmp_path):
    c = Connectome.load(synthetic_npz)
    pops = Populations.from_connectome(c)
    base = dict(noise_mv=0.15, min_weight=1, balance_hemispheres=False, kc_thresh=0.5)
    ref = Engine(c, pops, Params(**base), seed=0)
    v_th = ref.v_th[pops.kc].astype(np.float32).copy()
    v_th[ref.kc_pn_input > 0] *= np.float32(0.9)
    path, sha = save_kc_thresholds(tmp_path / "theta.npz", c.bodyId[pops.kc], v_th)
    A = {"ORN_DM1": 1.0, "ORN_DA1": 1.0}
    B = {"ORN_VA2": 1.0, "ORN_DM6": 1.0}
    for extra in (dict(apl_mode="graded", apl_r_max=0.3), dict(orn_std=True),
                  dict(kc_thresh_mode="homeostatic", kc_thresh_file=str(path), kc_thresh_sha256=sha),
                  dict(apl_mode="graded", apl_r_max=0.3, orn_std=True)):
        p = Params(**base, **extra)
        eng = Engine(c, pops, p, seed=0)
        pl = Plasticity(eng, pops, compartments(c, pops, p.core_frac))
        expected = decide(eng, pl, pops, [A, B], 1.0, seed=7, settle_ms=50, read_ms=300)
        with FlyPool(synthetic_npz, p, [FlySpec()], workers=1, timeout_s=120) as pool:
            got = pool.decide_batch([(0, [A, B], 7)], 1.0, settle_ms=50, read_ms=300)[0]
        np.testing.assert_array_equal(got, expected, err_msg=str(extra))
