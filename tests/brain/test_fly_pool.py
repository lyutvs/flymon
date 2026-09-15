import concurrent.futures

import numpy as np
import pytest

from flymon.brain.circuits import Populations, compartments
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome, shuffle_kc_mbon
from flymon.brain.engine_cpu import Engine
from flymon.brain.plasticity import Plasticity
from flymon.brain.fly_pool import FlyPool, FlySpec
from flymon.brain.pool_jobs import (baseline_job, conditioning_arm_job, phase_timing_job, rss_job,
                                    sparsity_job, weights_frac_job)
from flymon.brain.presentation import decide

P = Params(noise_mv=0.15, min_weight=1, balance_hemispheres=False, kc_thresh=0.5, learn_rate=0.05)
A = {"ORN_DM1": 1.0, "ORN_DA1": 1.0}
B = {"ORN_VA2": 1.0, "ORN_DM6": 1.0}
FLIES = [FlySpec(), FlySpec(shuffle_seed=3), FlySpec(enabled=False)]


@pytest.fixture(scope="module")
def pool(synthetic_npz):
    with FlyPool(synthetic_npz, P, FLIES, workers=2, timeout_s=120) as p:
        yield p


def _in_process(npz, shuffle_seed=None):
    c = Connectome.load(npz)
    pops = Populations.from_connectome(c)
    if shuffle_seed is not None:
        c = shuffle_kc_mbon(c, pops.kc, pops.mbon, shuffle_seed)
    eng = Engine(c, pops, P, seed=0)
    pl = Plasticity(eng, pops, compartments(c, pops, P.core_frac))
    return pops, eng, pl


def test_pool_layout(pool):
    assert pool.n_flies == 3 and pool.n_workers == 2
    assert set(pool.w0) == {None, 3} and pool.w[0].shape == pool.w0[None].shape
    assert pool.weights_frac(0) == pytest.approx(1.0)


def test_decide_batch_equals_in_process_engine(pool, synthetic_npz):
    pops, eng, pl = _in_process(synthetic_npz)
    expected = decide(eng, pl, pops, [A, B], 1.0, seed=7, settle_ms=50, read_ms=300)
    got = pool.decide_batch([(0, [A, B], 7), (0, [A], 7)], 1.0, settle_ms=50, read_ms=300)
    np.testing.assert_array_equal(got[0], expected)
    np.testing.assert_array_equal(got[1], expected[:1])
    pops, eng2, pl2 = _in_process(synthetic_npz, shuffle_seed=3)
    expected2 = decide(eng2, pl2, pops, [A, B], 1.0, seed=7, settle_ms=50, read_ms=300)
    got2 = pool.decide_batch([(1, [A, B], 7)], 1.0, settle_ms=50, read_ms=300)[0]
    np.testing.assert_array_equal(got2, expected2)
    assert (expected2[:, pops.mbon] != expected[:, pops.mbon]).any()      # scrambled wiring reaches the MBONs differently
    sub = pool.decide_batch([(0, [A], 7)], 1.0, settle_ms=50, read_ms=300, idx=pops.mbon)[0]
    np.testing.assert_array_equal(sub, expected[:1][:, pops.mbon])


def test_reinforce_batch_isolates_flies_and_respects_enabled(pool, synthetic_npz):
    pops, eng, pl = _in_process(synthetic_npz)
    naive = decide(eng, pl, pops, [A], 1.0, seed=9, settle_ms=50, read_ms=300)
    pool.reinforce_batch([(0, A, "PAM08", 300, 1), (2, A, "PAM08", 300, 1)], 1.0, settle_ms=50, gap_ms=20)
    assert pool.weights_frac(0) < 1.0
    assert pool.weights_frac(2) == pytest.approx(1.0)              # disabled fly: driven but does not learn
    assert pool.weights_frac(1) == pytest.approx(1.0)              # not in the batch
    learned = pool.decide_batch([(0, [A], 9)], 1.0, settle_ms=50, read_ms=300)[0]
    assert (learned[:, pops.mbon] != naive[:, pops.mbon]).any()    # the learned weights travelled with the request


def test_state_roundtrip_and_validation(pool):
    s = pool.state()
    w0_fly0 = pool.w[0].copy()
    pool.w[0][:] = pool.w0[None]
    pool.set_enabled(2, True)
    pool.load_state(s)
    np.testing.assert_array_equal(pool.w[0], w0_fly0)
    assert pool.flies[2].enabled is False
    bad = {"flies": s["flies"][:2]}
    with pytest.raises(ValueError, match="pool has 3"):
        pool.load_state(bad)
    bad = {"flies": [dict(e, shuffle_seed=99) if i == 1 else e for i, e in enumerate(s["flies"])]}
    with pytest.raises(ValueError, match="wiring variant"):
        pool.load_state(bad)
    np.testing.assert_array_equal(pool.w[0], w0_fly0)             # a rejected state changes nothing


def test_run_jobs_and_worker_errors_propagate(pool):
    res = pool.run_jobs(phase_timing_job, [dict(odor=A, strength=1.0, steps=5, warm=2)] * 2)
    assert len(res) == 2 and all(r["ms_decision"] > 0 and r["ms_reinforce"] > 0 for r in res)
    with pytest.raises(ValueError, match="unknown DAN"):
        pool.reinforce_batch([(0, A, "PAM99", 10, 1)], 1.0, settle_ms=5, gap_ms=5)
    assert pool.decide_batch([(0, [A], 1)], 1.0, settle_ms=5, read_ms=5)[0].shape[0] == 1   # still alive


def test_reinforce_batch_rejects_a_fly_listed_twice(pool):
    with pytest.raises(ValueError, match="more than once"):
        pool.reinforce_batch([(0, A, "PAM08", 10, 1), (0, B, "PAM08", 10, 2)], 1.0, settle_ms=5, gap_ms=5)


def test_load_state_rejects_non_finite_or_negative_weights(pool):
    s = pool.state()
    before = pool.w[0].copy()
    bad = {"flies": [dict(e, w=np.where(np.arange(e["w"].size) == 0, np.nan, e["w"])) if i == 0 else e for i, e in enumerate(s["flies"])]}
    with pytest.raises(ValueError, match="finite"):
        pool.load_state(bad)
    bad = {"flies": [dict(e, w=-e["w"]) if i == 0 else e for i, e in enumerate(s["flies"])]}
    with pytest.raises(ValueError, match="finite"):
        pool.load_state(bad)
    np.testing.assert_array_equal(pool.w[0], before)


def test_constructor_validates_before_spawning_and_caps_variants(synthetic_npz, tmp_path):
    import multiprocessing as mp
    children_before = len(mp.active_children())        # the module-scoped pool's workers may be alive
    with pytest.raises(FileNotFoundError):
        FlyPool(tmp_path / "missing.npz", P, [FlySpec()], workers=1, timeout_s=30)
    with pytest.raises(ValueError, match="max_variants"):
        FlyPool(synthetic_npz, P, [FlySpec(shuffle_seed=s) for s in range(5)], workers=1, timeout_s=30, max_variants=4)
    assert len(mp.active_children()) == children_before  # neither failure spawned (and leaked) a worker


ARM_KEYS = {"arm", "seed", "D_pre", "D_post", "dD", "counts", "weights_frac", "w_frac_a_core", "w_frac_p_core"}


def test_sparsity_job(pool):
    r = pool.run_jobs(sparsity_job, [dict(seed=100, strength=1.0, k=2, odor_seed=0)])[0]
    assert set(r) == {"frac_active_A", "frac_active_B", "jaccard", "chance", "mbon_hz_A", "mbon_hz_B"}
    assert all(np.isfinite(v) for v in r.values())
    assert 0.0 <= r["frac_active_A"] <= 1.0 and 0.0 <= r["frac_active_B"] <= 1.0 and r["jaccard"] >= 0.0


def test_baseline_job(pool):
    r = pool.run_jobs(baseline_job, [dict(seed=100, ms=200.0)])[0]
    assert set(r) == {"mbon_hz", "mbon_hz_trimmed", "n_saturated", "n_types_active", "runaway"}
    assert set(r["runaway"]) == {"sat_hz", "n_over_sat", "n_kc_over_sat", "spike_share_over_sat"}
    assert 0.0 <= r["runaway"]["spike_share_over_sat"] <= 1.0
    assert np.isfinite(r["mbon_hz"]) and np.isfinite(r["mbon_hz_trimmed"])


def test_conditioning_arm_job_leaves_the_worker_weights_reset(synthetic_npz):
    kw = dict(seed=1, strength=1.0, k=2, odor_seed=0, trials=3, present_ms=150.0, settle_ms=50.0)
    with FlyPool(synthetic_npz, P, [FlySpec()], workers=1, timeout_s=120) as one:
        both, noplast = one.run_jobs(conditioning_arm_job, [dict(kw, arm="both"), dict(kw, arm="noplast")])
        assert ARM_KEYS <= set(both) and ARM_KEYS <= set(noplast)
        assert noplast["dD"] == 0.0 and noplast["weights_frac"] == pytest.approx(1.0)
        assert both["weights_frac"] < 1.0                                    # the "both" arm learned
        assert one.run_jobs(weights_frac_job, [{}])[0] == pytest.approx(1.0)  # ... and left the worker reset


def test_batches_are_serialized_across_threads(pool):
    """Two concurrent decide_batch calls must not interleave on the pool: same answer as a sequential call."""
    def go():
        return pool.decide_batch([(0, [A], 5)], 1.0, settle_ms=20, read_ms=50)[0]

    with concurrent.futures.ThreadPoolExecutor(2) as ex:
        a, b = [f.result() for f in [ex.submit(go), ex.submit(go)]]
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, go())


def _rss_by_pid(samples):
    by = {}
    for s in samples:
        by[s["pid"]] = max(by.get(s["pid"], 0.0), s["rss_GB"])
    return by


def test_rss_job_reports_a_monotone_peak_per_worker(pool):
    first = pool.run_jobs(rss_job, [{}] * (4 * pool.n_workers))
    assert len(first) == 4 * pool.n_workers
    assert all(r["pid"] > 0 and r["rss_GB"] > 0 for r in first)
    second = pool.run_jobs(rss_job, [{}] * (4 * pool.n_workers), shuffle_seed=7)
    before, after = _rss_by_pid(first), _rss_by_pid(second)
    both = set(before) & set(after)
    assert both                                                  # some worker was sampled in both rounds
    assert all(after[pid] >= before[pid] for pid in both)         # peak RSS never falls, per process
