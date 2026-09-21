"""The H.3 worker jobs (spec H.3a): the diagnostics' protocols, the engine cache, the CSC edits, pool = in-process."""
import hashlib

import numpy as np
import pytest

from flymon.brain import h3_jobs as J
from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.engine_cpu import Engine
from flymon.brain.fly_pool import FlyPool
from flymon.brain.measure import kc_sparsity
from flymon.brain.stimuli import design_odor_pair, present

# a regime where the synthetic KCs fire and APL input is kept (the default min_weight drops the 4-synapse KC->APL edges)
BASE = dict(noise_mv=0.15, min_weight=1, balance_hemispheres=False, kc_thresh=0.5)
GRADED = dict(BASE, apl_mode="graded", apl_r_max=0.333, apl_input_scale=0.25)
RECORDS = dict(quantiles=(5.0, 50.0, 95.0), callout=("MBON01", "MBON03"))
ODORS = [dict(name="S00", types=["ORN_DM1"], strengths={"ORN_DM1": 1.0}, seeds=[11, 12]),
         dict(name="S01", types=["ORN_VA2", "ORN_DM6"], strengths={"ORN_VA2": 1.0, "ORN_DM6": 1.0}, seeds=[13, 14])]


class _Stub:
    def __init__(self, conn):
        self.conn = conn


def _diag_reference(conn, pops, params, odors, strength, settle, steps):
    """apl_input_scale_sweep.job + readout_floor_guard.job_stim (the diagnostics' call sequence), with a fresh engine."""
    e = Engine(conn, pops, params)
    out = []
    for o in odors:
        for seed in o["seeds"]:
            e.reset(seed); e.clear_drive(); present(e, pops, o["strengths"], strength); e.run(settle)
            counts = np.zeros(e.N, np.int32)
            apl_v = np.zeros((steps, len(pops.apl)))
            rel = 0.0
            for i in range(steps):
                counts[e.step()] += 1
                apl_v[i] = e.v[pops.apl]
                rel += float(e.apl_release(e.v[pops.apl]).astype(np.float64).sum())
            kc = counts[pops.kc]
            out.append(dict(kc_active_frac=float((kc > 0).mean()), kc_spikes=int(kc.sum()), apl_v_mean=float(apl_v.mean()),
                            release_frac=rel / (steps * len(pops.apl)) / e.p.apl_r_max,
                            mbon=[int(x) for x in counts[pops.mbon]]))
    return out


@pytest.fixture
def conn_pops(synthetic_connectome):
    c = synthetic_connectome(disjoint_kc=True)
    return c, Populations.from_connectome(c)


def test_reference_job_is_the_diagnostic_protocol(conn_pops):
    c, pops = conn_pops
    p = Params(**GRADED)
    J._CACHE.clear()
    got = J.reference_job(_Stub(c), None, pops, None, None, params=p, odors=ODORS, strength=3.0, settle_ms=50.0,
                          read_steps=100, **RECORDS)
    want = _diag_reference(c, pops, p, ODORS, 3.0, 50.0, 100)
    assert any(r["kc_spikes"] for r in want), "the synthetic regime must make KCs fire"
    mt = J.mbon_type_index(c, pops)
    for g, w in zip(got, want):
        for k in ("kc_active_frac", "kc_spikes", "apl_v_mean", "release_frac"):
            assert g[k] == w[k], k
        assert sum(g["types"].values()) == sum(w["mbon"]) and set(g["types"]) == set(mt)
        assert len(g["fired"]) == round(g["kc_active_frac"] * len(pops.kc))
        assert set(g["apl_to_type_input"]) == {"MBON01", "MBON03"} and len(g["apl_v_quantiles"]) == 3


def test_the_engine_cache_gives_a_fresh_engine_s_numbers(conn_pops):
    c, pops = conn_pops
    p = Params(**GRADED)
    J._CACHE.clear()
    kw = dict(params=p, odors=ODORS, strength=3.0, settle_ms=50.0, read_steps=100, **RECORDS)
    first = J.reference_job(_Stub(c), None, pops, None, None, **kw)
    J.rest_job(_Stub(c), None, pops, None, None, params=p, seeds=[1, 2], settle_ms=50.0, read_steps=100)
    again = J.reference_job(_Stub(c), None, pops, None, None, **kw)
    assert first == again and len(J._CACHE) == 1
    J.reference_job(_Stub(c), None, pops, None, None, **dict(kw, params=Params(**BASE)))
    assert len(J._CACHE) == 1 and next(iter(J._CACHE))[0] == Params(**BASE)


def test_design_job_is_kc_sparsity_on_the_design_pair(conn_pops):
    c, pops = conn_pops
    p = Params(**BASE)
    J._CACHE.clear()
    rows = J.design_job(_Stub(c), None, pops, None, None, params=p, seeds=[5], k=2, odor_seed=0, strength=3.0,
                        settle_ms=50.0, read_ms=100.0)
    a, b = design_odor_pair(pops, k=2, seed=0)
    e = Engine(c, pops, p)
    ra, rb = kc_sparsity(e, pops, a, 3.0, 5, 50.0, 100.0), kc_sparsity(e, pops, b, 3.0, 5, 50.0, 100.0)
    assert rows[0]["frac_active_A"] == ra["frac_active"] and rows[0]["frac_active_B"] == rb["frac_active"]


def test_baseline_job_counts_kcs_over_the_threshold(conn_pops):
    c, pops = conn_pops
    J._CACHE.clear()
    r = J.baseline_job(_Stub(c), None, pops, None, None, params=Params(**dict(BASE, kc_thresh=0.01)), seeds=[3],
                       ms=200.0, sat_hz=0.0)[0]
    assert r["n_kc_over_sat"] >= 0 and set(r) >= {"trimmed", "raw", "n_kc_over_sat", "n_saturated"}


def test_odor_runaway_records_the_band_below_the_judged_threshold(conn_pops):
    c, pops = conn_pops
    J._CACHE.clear()
    kw = dict(params=Params(**dict(BASE, kc_thresh=0.1)), seeds=[5], k=2, odor_seed=0, strength=3.0, settle_ms=50.0,
              read_ms=100.0)
    all_band = J.odor_runaway_job(_Stub(c), None, pops, None, None, sat_hz=1e9, record_hz=0.0, **kw)[0]
    assert all_band["n_kc_over_sat"] == 0 and all_band["n_kc_over_record"] == len(all_band["band"]) > 0
    assert {b["body_id"] for b in all_band["band"]} <= set(c.bodyId[pops.kc].tolist())
    judged = J.odor_runaway_job(_Stub(c), None, pops, None, None, sat_hz=0.0, record_hz=0.0, **kw)[0]
    assert judged["n_kc_over_sat"] == all_band["n_kc_over_record"] and judged["band"] == []


def test_apl_to_mbon_zero_edits_exactly_those_edges(conn_pops):
    c, pops = conn_pops
    p = Params(**GRADED)
    ref = Engine(c, pops, p)
    e = Engine(c, pops, p)
    sha = J.apply_csc_edit(e, pops, ("apl_to_mbon_zero",))
    src, tgt = J.edge_sources(e.csc), e.csc.tgt
    m = np.isin(src, pops.apl) & np.isin(tgt, pops.mbon)
    assert (e.csc.w[m] == 0).all() and np.array_equal(e.csc.w[~m], ref.csc.w[~m])
    assert sha == hashlib.sha256(e.csc.w.tobytes()).hexdigest()


def test_kc_to_apl_scale_matches_apl_input_scale_on_a_kc_only_input(conn_pops):
    """With every APL input from KCs, scaling KC->APL edges after construction equals apl_input_scale (the
    parameter's multiply is the last one in build_csc)."""
    c, pops = conn_pops
    src_all = J.edge_sources(Engine(c, pops, Params(**BASE)).csc)
    tgt_all = Engine(c, pops, Params(**BASE)).csc.tgt
    assert set(src_all[np.isin(tgt_all, pops.apl)]) <= set(pops.kc.tolist())
    e = Engine(c, pops, Params(**dict(GRADED, apl_input_scale=1.0)))
    J.apply_csc_edit(e, pops, ("kc_to_apl_scale", 0.25))
    assert np.array_equal(e.csc.w, Engine(c, pops, Params(**GRADED)).csc.w)


def test_graded_release_follows_an_edited_csc(conn_pops):
    """The graded-APL out-edge lists are views into the CSC, so zeroing APL->MBON reaches the release path."""
    c, pops = conn_pops
    e = Engine(c, pops, Params(**GRADED))
    J.apply_csc_edit(e, pops, ("apl_to_mbon_zero",))
    for tgt, w in e._apl_edges:
        assert (w[np.isin(tgt, pops.mbon)] == 0).all()


def test_unknown_csc_edit_is_rejected(conn_pops):
    c, pops = conn_pops
    with pytest.raises(ValueError, match="unknown csc_edit"):
        J.apply_csc_edit(Engine(c, pops, Params(**BASE)), pops, ("nope",))


def test_pool_jobs_equal_in_process(synthetic_npz):
    c = Connectome.load(synthetic_npz)
    pops = Populations.from_connectome(c)
    p = Params(**GRADED)
    kw = dict(params=p, odors=ODORS, strength=3.0, settle_ms=50.0, read_steps=100, **RECORDS)
    J._CACHE.clear()
    local = J.reference_job(_Stub(c), None, pops, None, None, **kw)
    dkw = dict(params=p, seeds=[5, 6], k=2, odor_seed=0, strength=3.0, settle_ms=50.0, read_ms=100.0)
    local_d = J.design_job(_Stub(c), None, pops, None, None, **dkw)
    with FlyPool(synthetic_npz, Params(), [{}], workers=1, timeout_s=300) as pool:
        remote = pool.run_jobs(J.reference_job, [kw])[0]
        remote_d = pool.run_jobs(J.design_job, [dkw])[0]
    assert remote == local and remote_d == local_d
