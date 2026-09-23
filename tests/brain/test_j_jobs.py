"""Spec J.11.3-J.11.4 worker jobs: the scan's all51 record and D.6 on a declared engine."""
from pathlib import Path

import numpy as np
import pytest

from flymon.brain import d6a, h3_jobs
from flymon.brain.circuits import Populations, compartments
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.engine_cpu import Engine
from flymon.brain.h4_jobs import rig_for
from flymon.brain.j_jobs import all51_uni_job, d6_job, uni_pns
from flymon.brain.j_params import StdParams

BASE = dict(noise_mv=0.15, min_weight=1, balance_hemispheres=False, kc_thresh=0.5)
KW = dict(c_norm=4.0, strength=1.0, settle_ms=50.0, read_steps=200, uni_frac=0.8, uni_min_syn=1)


def _rig(synthetic_connectome):
    c = synthetic_connectome()
    pops = Populations.from_connectome(c)
    return Engine(c, pops, Params(**BASE), seed=0), pops


def test_uni_pns_follow_the_orn_input_rule(synthetic_connectome):
    eng, pops = _rig(synthetic_connectome)
    uni = uni_pns(eng.conn, pops, 0.8, 1)
    assert sorted(uni) == sorted(pops.receptor_types)
    got = np.concatenate(list(uni.values()))
    assert got.size and np.isin(got, pops.alpn).all() and np.unique(got).size == got.size
    assert all(v.size == 0 for v in uni_pns(eng.conn, pops, 0.8, 10 ** 6).values())     # min_syn excludes every PN


def test_all51_uni_keeps_all51_job_s_counts_and_adds_the_scan_measures(synthetic_connectome):
    eng, pops = _rig(synthetic_connectome)
    items = [(t, s) for t in sorted(pops.receptor_types)[:3] for s in (1, 2)]
    ref = h3_jobs.all51_job(eng, None, pops, None, None, Params(**BASE), items, 4.0, 1.0, 50.0, 200)
    got = all51_uni_job(eng, None, pops, None, None, Params(**BASE), items, **KW)
    assert [(r["g"], r["seed"], r["kc"], r["kc_on"], r["pn"]) for r in got] == \
        [(r["g"], r["seed"], r["kc"], r["kc_on"], r["pn"]) for r in ref]
    assert all(r["std_r"] is None and r["orn_hz"] > 0 for r in got)
    dep = all51_uni_job(eng, None, pops, None, None, StdParams(**BASE, orn_std=True, receptor_scale=2.0), items, **KW)
    assert all(0 < r["std_r"] < 1 for r in dep)


def test_d6_job_is_d6a_job_on_the_declared_rig(synthetic_connectome):
    eng, pops = _rig(synthetic_connectome)
    p = StdParams(**BASE, orn_std=True, receptor_scale=2.0)
    odors = [{"ORN_DM1": 1.0}, {"ORN_VA2": 1.0}]
    got = d6_job(eng, None, pops, None, None, p, odors, 5)
    assert got["seed"] == 5 and len(got["max_win"]) == 2
    e, pl, comps = rig_for(eng.conn, pops, p)
    assert got == d6a.d6a_job(e, pl, pops, comps, None, odors, 5)          # the same presentation, G.8's defaults


@pytest.mark.skipif(not Path("data/malecns.npz").exists(), reason="MaleCNS connectome not built")
def test_the_real_unipns_are_323_and_three_all51_glomeruli_have_none():
    """The count the second red team recomputed with disparity_stages.py's rule (J.11.1): 323 over every receptor type,
    286 over the all51 glomeruli (ORN_DA1 has 30, ORN_V 7); DA4m, VM6l and VM6m have none."""
    c = Connectome.load("data/malecns.npz")
    pops = Populations.from_connectome(c)
    from flymon.brain.h3_spec import all51_glomeruli
    uni = uni_pns(c, pops, 0.8, 20)
    gl, _ = all51_glomeruli(pops)
    assert sum(v.size for v in uni.values()) == 323 and sum(uni[g].size for g in gl) == 286
    assert (uni["ORN_DA1"].size, uni["ORN_V"].size) == (30, 7)
    assert sorted(str(g) for g in gl if uni[g].size == 0) == ["ORN_DA4m", "ORN_VM6l", "ORN_VM6m"]


def test_two_receptor_scales_in_one_worker_are_two_engines(synthetic_connectome):
    """engine_for is keyed by the whole Params: a worker that measures s = 1 then s = 4 must not reuse the first CSC."""
    eng, pops = _rig(synthetic_connectome)
    items = [(t, 1) for t in sorted(pops.receptor_types)[:3]]
    kw = {**BASE, "orn_std": True}
    one = all51_uni_job(eng, None, pops, None, None, StdParams(**kw, receptor_scale=1.0), items, **KW)
    four = all51_uni_job(eng, None, pops, None, None, StdParams(**kw, receptor_scale=4.0), items, **KW)
    again = all51_uni_job(eng, None, pops, None, None, StdParams(**kw, receptor_scale=1.0), items, **KW)
    assert one == again and [r["pn"] for r in one] != [r["pn"] for r in four]
