"""Q's worker job: with edit "none" the result minus "q" is h4_jobs.oracle_job's (Reading 2); apl_to_mbon05_zero zeroes
exactly the APL out-edges onto the P readout type; apl_to_nonkc_zero is O's edit; the fixed arms are the reward-only
edit at alpha; reach is w0*fx over the reward core's P-type edges; the rig is cached per (Params, edit, type)."""
import json
from dataclasses import replace

import numpy as np
import pytest

from flymon.brain import h4_jobs as H4
from flymon.brain import o_jobs as O
from flymon.brain import q_jobs as Q
from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.engine_cpu import Engine
from flymon.brain.fly_pool import FlyPool
from flymon.brain.h3_jobs import edge_sources
from flymon.brain.stimuli import design_odor_pair

P = Params(noise_mv=0.15, min_weight=1, balance_hemispheres=False, kc_thresh=0.5, learn_rate=0.05)
TYPES = ("MBON03", "MBON04", "MBON01", "MBON02")            # synthetic PPL105 core, then PAM08 core
READOUT = {"A": "MBON03", "P": "MBON01"}
Z = {"A": (5.0, 3.0), "P": (8.0, 4.0)}
ORACLE = dict(readout=READOUT, z=Z, types=TYPES, act_seeds=(500, 501), select_seeds=(600, 601, 602),
              report_seeds=(608, 609, 610), alphas=(0.2, 0.5, 0.8), strength=3.0, settle_ms=50.0, read_ms=100.0,
              window_ms=20, punish_type="PPL105", reward_type="PAM08")
QX = dict(fixed_alphas=(0.8, 1.0), active_fx=0.5)


class _Stub:
    def __init__(self, conn):
        self.conn = conn


@pytest.fixture
def conn_pops(synthetic_connectome):
    c = synthetic_connectome(disjoint_kc=True)
    t = np.asarray(c.type).astype(str)
    apl = int(np.flatnonzero(t == "APL")[0])
    m1 = np.flatnonzero(t == "MBON01")
    c = replace(c, pre=np.append(c.pre, np.full(len(m1), apl, np.int32)), post=np.append(c.post, m1.astype(np.int32)),
                w=np.append(c.w, np.full(len(m1), 20, np.int32)))      # APL -> every MBON01 cell
    Q._RIG.clear(); H4._RIG.clear(); O._RIG.clear()
    return c, Populations.from_connectome(c)


def _canon(x):
    return json.dumps(x, sort_keys=True, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def test_none_edit_is_oracle_job(conn_pops):
    c, pops = conn_pops
    a, b = design_odor_pair(pops, k=2, seed=0)
    ref = H4.oracle_job(_Stub(c), None, pops, None, None, params=P, odor_x=a, odor_y=b, **ORACLE)
    got = Q.q_oracle_job(_Stub(c), None, pops, None, None, params=P, edit="none", odor_x=a, odor_y=b, **ORACLE, **QX)
    assert _canon({k: v for k, v in got.items() if k != "q"}) == _canon(ref)
    q = got["q"]
    assert q["edit"] == "none" and q["edit_edges"] == 0 and set(q["fixed"]) == {"0.8", "1.0"}
    assert len(q["apl_out"]["x"]) == 2 and all(np.isfinite(q["apl_out"]["x"]))
    assert any(x for row in ref["report"]["pre"]["P"] for x in row), "the synthetic regime must drive the readout"


def test_fixed_arm_at_the_selected_alpha_is_r1(conn_pops):
    c, pops = conn_pops
    a, b = design_odor_pair(pops, k=2, seed=0)
    kw = dict(ORACLE, alphas=(0.8,))
    got = Q.q_oracle_job(_Stub(c), None, pops, None, None, params=P, edit="none", odor_x=a, odor_y=b, **kw, **QX)
    assert got["alpha_reward"] == 0.8 and got["q"]["fixed"]["0.8"]["report"] == got["report"]["R1"]


def _zeroed(c, pops, edit, p_type="MBON01"):
    e0, e1 = Engine(c, pops, P, seed=0), Engine(c, pops, P, seed=0)
    _, n = Q.apply_q_edit(e1, pops, edit, p_type)
    return set(np.flatnonzero(e0.csc.w != e1.csc.w).tolist()), n, e0


def test_mbon05_edit_zeroes_exactly_apl_to_p_type(conn_pops):
    c, pops = conn_pops
    got, n, e0 = _zeroed(c, pops, Q.MBON05)
    t = np.asarray(c.type).astype(str)
    want = set(np.flatnonzero(np.isin(edge_sources(e0.csc), pops.apl)
                              & (t[e0.csc.tgt.astype(np.int64)] == "MBON01")).tolist())
    assert got == want and n == len(want) > 0


def test_nonkc_edit_is_o_jobs_edit(conn_pops):
    c, pops = conn_pops
    got, n, _ = _zeroed(c, pops, Q.NONKC)
    e0, e1 = Engine(c, pops, P, seed=0), Engine(c, pops, P, seed=0)
    O.apply_edit(e1, pops, O.NONKC)
    assert got == set(np.flatnonzero(e0.csc.w != e1.csc.w).tolist()) and n == len(got) > 0
    with pytest.raises(ValueError):
        Q.apply_q_edit(Engine(c, pops, P, seed=0), pops, "apl_all_zero", "MBON01")


def test_reach_is_w0_fx_over_the_reward_core_p_edges(conn_pops):
    c, pops = conn_pops
    a, b = design_odor_pair(pops, k=2, seed=0)
    got = Q.q_oracle_job(_Stub(c), None, pops, None, None, params=P, edit="none", odor_x=a, odor_y=b, **ORACLE, **QX)
    e, p, comps, _, _ = Q._RIG[(P, "none", "MBON01")]
    fx = np.zeros(len(pops.kc))
    fx[got["q"]["fx"]["idx"]] = got["q"]["fx"]["val"]
    t = np.asarray(c.type).astype(str)
    core = set(comps["PAM08"].core.tolist())
    W, kcs = 0.0, set()
    for i, edge in enumerate(p.edges):
        tgt = int(e.csc.tgt[edge])
        if tgt in core and t[tgt] == "MBON01":
            W += float(p.w0[i]) * fx[p.pre_kc[i]]
            kcs.add(int(p.pre_kc[i]))
    r = got["q"]["reach"]
    assert r["W_X"] == pytest.approx(W) and W > 0
    active = set(np.flatnonzero(fx >= 0.5).tolist())
    assert r["n_active_kc"] == len(active)
    assert r["f_X"] == (pytest.approx(len(active & kcs) / len(active)) if active else None)
    assert set(r["by_type"]) <= {"MBON01", "MBON02"}


def test_mbon05_edit_changes_counts_and_rig_cache(conn_pops):
    c, pops = conn_pops
    a, b = design_odor_pair(pops, k=2, seed=0)
    n0 = Q.q_oracle_job(_Stub(c), None, pops, None, None, params=P, edit="none", odor_x=a, odor_y=b, **ORACLE, **QX)
    m5 = Q.q_oracle_job(_Stub(c), None, pops, None, None, params=P, edit=Q.MBON05, odor_x=a, odor_y=b, **ORACLE, **QX)
    assert list(Q._RIG) == [(P, Q.MBON05, "MBON01")]
    assert m5["q"]["csc_sha256"] != n0["q"]["csc_sha256"] and m5["q"]["edit_edges"] > 0
    assert Q._RIG[(P, Q.MBON05, "MBON01")][1].weights_frac() == 1.0


def test_pool_equals_in_process(conn_pops, synthetic_npz, synthetic_connectome):
    c = synthetic_connectome(disjoint_kc=True)                  # the npz's connectome (no extra APL edge)
    pops = Populations.from_connectome(c)
    a, b = design_odor_pair(pops, k=2, seed=0)
    Q._RIG.clear()
    here = Q.q_oracle_job(_Stub(c), None, pops, None, None, params=P, edit="none", odor_x=a, odor_y=b, **ORACLE, **QX)
    with FlyPool(synthetic_npz, Params(), [{}], workers=1) as pool:
        there = pool.run_jobs(Q.q_oracle_job, [dict(params=P, edit="none", odor_x=a, odor_y=b, **ORACLE, **QX)])[0]
    assert _canon(here) == _canon(there)


def test_mv_scale_is_its_own_rig_and_fixed_arms_stay_out_of_selection(conn_pops):
    c, pops = conn_pops
    a, b = design_odor_pair(pops, k=2, seed=0)
    P2 = replace(P, mv_per_synapse=P.mv_per_synapse * 0.8)
    got = Q.q_oracle_job(_Stub(c), None, pops, None, None, params=P2, edit="none", odor_x=a, odor_y=b, **ORACLE, **QX)
    assert list(Q._RIG) == [(P2, "none", "MBON01")] and Q._RIG[(P2, "none", "MBON01")][0].p.mv_per_synapse == P2.mv_per_synapse
    assert set(got["select"]["reward"]) == {"0.2", "0.5", "0.8"} and got["alpha_reward"] in ORACLE["alphas"]
