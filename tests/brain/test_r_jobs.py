"""R's worker jobs on q_jobs.q_rig (R.9.6, Reading 4/5): with edit "none" kc_activity_job is k_jobs.activity_job,
r_arm_job is o_jobs.arm_job (minus wall_s) and r_oracle_job minus "r" is q_oracle_job (whose result minus "q" is
h4_jobs.oracle_job with no fixed arms); under apl_to_mbon05_zero every job records the edited edge count and a new CSC
sha; r_oracle_job's per-cell naive probe sums to the oracle's pre counts of the P type."""
import json
from dataclasses import replace

import numpy as np
import pytest

from flymon.brain import h4_jobs as H4
from flymon.brain import k_jobs as K
from flymon.brain import n_jobs as N
from flymon.brain import o_jobs as O
from flymon.brain import q_jobs as Q
from flymon.brain import r_jobs as R
from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.engine_cpu import Engine
from flymon.brain.o_spec import SPEC as O_SPEC
from flymon.brain.stimuli import design_odor_pair

P = Params(noise_mv=0.15, min_weight=1, balance_hemispheres=False, kc_thresh=0.5, learn_rate=0.05)
TYPES = ("MBON03", "MBON04", "MBON01", "MBON02")            # synthetic PPL105 core, then PAM08 core
READOUT = {"A": "MBON03", "P": "MBON01"}
Z = {"A": (5.0, 3.0), "P": (8.0, 4.0)}
W = dict(strength=3.0, settle_ms=50.0, read_ms=100.0, window_ms=20)
ORACLE = dict(readout=READOUT, z=Z, types=TYPES, act_seeds=(500, 501), select_seeds=(600, 601, 602),
              report_seeds=(608, 609, 610), alphas=(0.2, 0.5, 0.8), punish_type="PPL105", reward_type="PAM08", **W)
ARM = dict(readout=READOUT, punish_type="PPL105", reward_type="PAM08", trials=2, present_ms=300.0, gap_ms=50.0,
           train_settle_ms=100.0, seed_base=1_000_000, seed_stride=1000, **W)
FLAGS = {name: dict(punish=pu, plastic=pl, da_zero=dz) for name, pu, pl, dz in O_SPEC.o2_arms}
X, Y = {"ORN_DM1": 1.0, "ORN_DA1": 1.0}, {"ORN_VA2": 1.0, "ORN_DM6": 1.0}


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
    Q._RIG.clear(); H4._RIG.clear(); O._RIG.clear(); N._RIG.clear()
    return c, Populations.from_connectome(c)


def _canon(x):
    return json.dumps(x, sort_keys=True, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def _strip(x):
    if isinstance(x, dict):
        return {k: _strip(v) for k, v in x.items() if k != "wall_s"}
    if isinstance(x, list):
        return [_strip(v) for v in x]
    return x


def _lever_edges(c, pops) -> int:
    return Q.apply_q_edit(Engine(c, pops, P, seed=0), pops, Q.MBON05, "MBON01")[1]


def test_kc_activity_none_is_k_jobs_activity(conn_pops):
    c, pops = conn_pops
    items = [(0, X, 500), (1, Y, 501), (2, X, 502)]
    ref = K.activity_job(_Stub(c), None, pops, None, None, params=P, items=items, **W)
    got = R.kc_activity_job(_Stub(c), None, pops, None, None, params=P, edit="none", p_type="MBON01", items=items, **W)
    assert [{k: v for k, v in g.items() if k not in ("csc_sha256", "edit_edges")} for g in got] == ref
    assert {g["edit_edges"] for g in got} == {0} and len({g["csc_sha256"] for g in got}) == 1
    assert any(g["kc"] for g in got), "the synthetic regime must drive KCs"


def test_kc_activity_under_the_lever_records_the_edit(conn_pops):
    c, pops = conn_pops
    n = _lever_edges(c, pops)
    items = [(0, X, 500)]
    none = R.kc_activity_job(_Stub(c), None, pops, None, None, params=P, edit="none", p_type="MBON01", items=items, **W)
    lev = R.kc_activity_job(_Stub(c), None, pops, None, None, params=P, edit=Q.MBON05, p_type="MBON01", items=items, **W)
    assert n > 0 and lev[0]["edit_edges"] == n and lev[0]["csc_sha256"] != none[0]["csc_sha256"]


@pytest.mark.parametrize("arm", ["plastic", "frozen", "punish"])
def test_r_arm_none_is_o_arm_job(conn_pops, arm):
    c, pops = conn_pops
    ref = O.arm_job(_Stub(c), None, pops, None, None, params=P, edit="none", odor_x=X, odor_y=Y, seed=9, arm=arm,
                    **FLAGS[arm], **ARM)
    got = R.r_arm_job(_Stub(c), None, pops, None, None, params=P, edit="none", p_type="MBON01", odor_x=X, odor_y=Y,
                      seed=9, arm=arm, **FLAGS[arm], **ARM)
    assert got["r"] == {"edit_edges": 0, "p_type": "MBON01"}
    assert _canon(_strip({k: v for k, v in got.items() if k != "r"})) == _canon(_strip(ref))


def test_r_arm_under_the_lever(conn_pops):
    c, pops = conn_pops
    n = _lever_edges(c, pops)
    none = R.r_arm_job(_Stub(c), None, pops, None, None, params=P, edit="none", p_type="MBON01", odor_x=X, odor_y=Y,
                       seed=9, arm="punish", **FLAGS["punish"], **ARM)
    got = R.r_arm_job(_Stub(c), None, pops, None, None, params=P, edit=Q.MBON05, p_type="MBON01", odor_x=X, odor_y=Y,
                      seed=9, arm="punish", **FLAGS["punish"], **ARM)
    assert got["edit"] == Q.MBON05 and got["r"]["edit_edges"] == n and got["csc_sha256"] != none["csc_sha256"]


def test_q_oracle_without_fixed_arms_is_oracle_job(conn_pops):
    c, pops = conn_pops
    a, b = design_odor_pair(pops, k=2, seed=0)
    ref = H4.oracle_job(_Stub(c), None, pops, None, None, params=P, odor_x=a, odor_y=b, **ORACLE)
    got = Q.q_oracle_job(_Stub(c), None, pops, None, None, params=P, edit="none", odor_x=a, odor_y=b, fixed_alphas=(),
                         active_fx=0.5, **ORACLE)
    assert _canon({k: v for k, v in got.items() if k != "q"}) == _canon(ref) and got["q"]["fixed"] == {}


@pytest.mark.parametrize("edit", ["none", Q.MBON05])
def test_r_oracle_is_q_oracle_plus_a_per_cell_naive_probe(conn_pops, edit):
    c, pops = conn_pops
    a, b = design_odor_pair(pops, k=2, seed=0)
    kw = dict(params=P, edit=edit, odor_x=a, odor_y=b, fixed_alphas=(), active_fx=0.5, **ORACLE)
    ref = Q.q_oracle_job(_Stub(c), None, pops, None, None, **kw)
    got = R.r_oracle_job(_Stub(c), None, pops, None, None, **kw)
    assert _canon({k: v for k, v in got.items() if k != "r"}) == _canon(ref)
    r = got["r"]
    t = np.asarray(c.type).astype(str)
    assert r["p_type"] == "MBON01" and r["n_cells"] == int((t == "MBON01").sum())
    assert r["read_ms"] == 100.0 and r["dt"] == P.dt and r["refrac_steps"] == P.refrac_steps()
    pre = got["report"]["pre"]["P"]
    for j, side in enumerate(("x", "y")):
        assert len(r["p_cells"][side]) == 3
        assert [sum(cells) for cells in r["p_cells"][side]] == [row[j] for row in pre]
    assert any(any(row) for row in pre), "the synthetic regime must drive the readout"
    if edit == Q.MBON05:
        assert got["q"]["edit_edges"] == _lever_edges(c, pops)
