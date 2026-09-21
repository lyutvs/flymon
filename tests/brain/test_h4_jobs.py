"""The H.4 worker jobs (spec H.4): the M0c arm with per-type counts, G.14's oracle with a chosen readout, the rig cache,
the weight reset and pool = in-process."""
import dataclasses

import numpy as np
import pytest

from flymon.brain import h4_jobs as J
from flymon.brain.circuits import Populations, compartments
from flymon.brain.conditioning import Readout, run_arm
from flymon.brain.config import Params
from flymon.brain.engine_cpu import Engine
from flymon.brain.fly_pool import FlyPool
from flymon.brain.h4_formula import dprime, dv
from flymon.brain.plasticity import Plasticity
from flymon.brain.presentation import decide
from flymon.brain.stimuli import design_odor_pair, present

P = Params(noise_mv=0.15, min_weight=1, balance_hemispheres=False, kc_thresh=0.5, learn_rate=0.05)
TYPES = ("MBON03", "MBON04", "MBON01", "MBON02")              # synthetic PPL105 core, then PAM08 core
TEACH = dict(types=TYPES, punish_type="PPL105", reward_type="PAM08", k=2, odor_seed=0, strength=3.0, trials=3,
             present_ms=150.0, gap_ms=50.0, settle_ms=50.0, read_ms=100.0)
READOUT = {"A": "MBON03", "P": "MBON01"}
Z = {"A": (5.0, 3.0), "P": (8.0, 4.0)}
ORACLE = dict(readout=READOUT, z=Z, types=TYPES, act_seeds=(500, 501), select_seeds=(600, 601, 602),
              report_seeds=(608, 609, 610), alphas=(0.2, 0.5, 0.8), strength=3.0, settle_ms=50.0, read_ms=100.0,
              window_ms=20, punish_type="PPL105", reward_type="PAM08")


class _Stub:
    def __init__(self, conn):
        self.conn = conn


@pytest.fixture
def conn_pops(synthetic_connectome):
    c = synthetic_connectome(disjoint_kc=True)
    return c, Populations.from_connectome(c)


def _fresh(c, pops, params=P):
    eng = Engine(c, pops, params, seed=0)
    comps = compartments(c, pops, params.core_frac)
    return eng, Plasticity(eng, pops, comps), comps


@pytest.mark.parametrize("arm", ["punish_only", "reward_only"])
@pytest.mark.parametrize("order", ["ab", "ba"])
def test_teach_job_is_run_arm_with_per_type_counts(conn_pops, arm, order):
    c, pops = conn_pops
    J._RIG.clear()
    got = J.teach_job(_Stub(c), None, pops, None, None, params=P, seed=9, arm=arm, order=order, **TEACH)
    eng, pl, comps = _fresh(c, pops)
    a, b = design_odor_pair(pops, k=2, seed=0)
    cs_plus, cs_minus = (a, b) if order == "ab" else (b, a)
    ref = run_arm(eng, pl, pops, Readout.from_compartments(comps), cs_plus, cs_minus, 3.0, 9, arm, trials=3,
                  present_ms=150.0, gap_ms=50.0, settle_ms=50.0, read_ms=100.0)["counts"]
    for ph in ("pre", "post"):
        for cs in ("plus", "minus"):
            g = got[ph][cs]
            assert (g["MBON03"] + g["MBON04"], g["MBON01"] + g["MBON02"]) == \
                (ref[f"{ph}_{cs}"]["A"], ref[f"{ph}_{cs}"]["P"]), (ph, cs)
    assert any(got["pre"][cs][t] for cs in ("plus", "minus") for t in TYPES), "the synthetic regime must drive MBONs"
    assert J._RIG[P][1].weights_frac() == 1.0 and J._RIG[P][1].enabled


def _oracle_reference(c, pops, odor_x, odor_y):
    """G.14.3's call sequence (m2_engine_probe.oracle_job) written out on a fresh rig, with this readout and z."""
    eng, pl, comps = _fresh(c, pops)
    t = np.asarray(c.type).astype(str)
    ia, ip = np.flatnonzero(t == READOUT["A"]), np.flatnonzero(t == READOUT["P"])
    idx = np.concatenate([ia, ip])
    pl.reset_weights(); pl.set_enabled(False)
    fired = np.zeros(len(pops.kc))
    for s in ORACLE["act_seeds"]:
        eng.reset(s); pl.reset_traces(); eng.clear_drive(); pl.quiet_dan(); present(eng, pops, odor_x, 3.0)
        eng.run(50.0); fired += eng.run(100.0)[pops.kc] > 0
    fx = fired / len(ORACLE["act_seeds"])

    def probe(seeds):
        out = [decide(eng, pl, pops, [odor_x, odor_y], 3.0, s, 50.0, 100.0, idx=idx) for s in seeds]
        return {"A": [o[:, :len(ia)].sum(1).tolist() for o in out], "P": [o[:, len(ia):].sum(1).tolist() for o in out]}

    rew = np.isin(pl.post_mb, pl.mb_local[comps["PAM08"].core]); pun = np.isin(pl.post_mb, pl.mb_local[comps["PPL105"].core])

    def set_w(a_r, a_p=None):
        wv = pl.w0.copy()
        if a_r is not None:
            wv[rew] = pl.w0[rew] * (1.0 - a_r * fx)[pl.pre_kc[rew]]
        if a_p is not None:
            wv[pun] = pl.w0[pun] * (1.0 - a_p * fx)[pl.pre_kc[pun]]
        eng.csc.w[pl.edges] = wv

    set_w(None); pre_sel = probe(ORACLE["select_seeds"])
    r1 = {}
    for a in ORACLE["alphas"]:
        set_w(a); r1[a] = probe(ORACLE["select_seeds"])
    a_r = max(ORACLE["alphas"], key=lambda a: (dprime(dv(r1[a], Z) - dv(pre_sel, Z)), -a))
    ch = {}
    for a in ORACLE["alphas"]:
        set_w(a_r, a); ch[a] = dprime(dv(probe(ORACLE["select_seeds"]), Z) - dv(r1[a_r], Z))
    a_p = min(ORACLE["alphas"], key=lambda a: (ch[a], a))
    set_w(None); pre = probe(ORACLE["report_seeds"])
    set_w(a_r); R1 = probe(ORACLE["report_seeds"])
    set_w(a_r, a_p); R2 = probe(ORACLE["report_seeds"])
    return a_r, a_p, {"pre": pre, "R1": R1, "R2": R2}


def test_oracle_job_is_g14s_sequence_with_the_given_readout(conn_pops):
    c, pops = conn_pops
    a, b = design_odor_pair(pops, k=2, seed=0)
    J._RIG.clear()
    got = J.oracle_job(_Stub(c), None, pops, None, None, params=P, odor_x=a, odor_y=b, **ORACLE)
    a_r, a_p, rep = _oracle_reference(c, pops, a, b)
    assert (got["alpha_reward"], got["alpha_punish"]) == (a_r, a_p)
    assert got["report"] == rep
    assert got["counts"]["pre"]["MBON03"] == rep["pre"]["A"] and set(got["counts"]["pre"]) == set(TYPES)
    assert any(x for row in rep["pre"]["P"] for x in row), "the synthetic regime must drive the readout"
    assert len(got["kc"]["x"]["frac"]) == 2 and 0.0 <= got["kc"]["jaccard"] <= 1.0
    assert J._RIG[P][1].weights_frac() == 1.0 and J._RIG[P][1].enabled


def test_the_rig_is_built_once_per_params(conn_pops):
    c, pops = conn_pops
    J._RIG.clear()
    kw = dict(seed=9, arm="punish_only", order="ab", **TEACH)
    first = J.teach_job(_Stub(c), None, pops, None, None, params=P, **kw)
    rig = J._RIG[P]
    assert J.teach_job(_Stub(c), None, pops, None, None, params=P, **kw) == first and J._RIG[P] is rig
    other = dataclasses.replace(P, kc_thresh=0.6)
    J.teach_job(_Stub(c), None, pops, None, None, params=other, **kw)
    assert list(J._RIG) == [other]


def test_pool_equals_in_process(conn_pops, synthetic_npz):
    c, pops = conn_pops
    a, b = design_odor_pair(pops, k=2, seed=0)
    J._RIG.clear()
    teach = J.teach_job(_Stub(c), None, pops, None, None, params=P, seed=9, arm="reward_only", order="ba", **TEACH)
    oracle = J.oracle_job(_Stub(c), None, pops, None, None, params=P, odor_x=a, odor_y=b, **ORACLE)
    with FlyPool(synthetic_npz, Params(), [{}], workers=1) as pool:
        t = pool.run_jobs(J.teach_job, [dict(params=P, seed=9, arm="reward_only", order="ba", **TEACH)])[0]
        o = pool.run_jobs(J.oracle_job, [dict(params=P, odor_x=a, odor_y=b, **ORACLE)])[0]
    assert t == teach and o["report"] == oracle["report"] and o["counts"] == oracle["counts"]
