"""The reward-only oracle (accepted copy of h4_jobs.oracle_job's reward half) and the qualification decision."""
import numpy as np
import pytest

from flymon.brain import h4_jobs as J
from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.h4_jobs import oracle_job
from flymon.brain.stimuli import design_odor_pair
from flymon.rescope.oracle import qual_x, qualify, reward_oracle_job
from flymon.rescope.spec import SPEC

# the setup tests/brain/test_h4_jobs.py uses for oracle_job on the synthetic connectome
P = Params(noise_mv=0.15, min_weight=1, balance_hemispheres=False, kc_thresh=0.5, learn_rate=0.05)
TYPES = ("MBON03", "MBON04", "MBON01", "MBON02")
READOUT = {"A": "MBON03", "P": "MBON01"}
Z = {"A": (5.0, 3.0), "P": (8.0, 4.0)}
ORACLE = dict(readout=READOUT, z=Z, types=TYPES, act_seeds=(500, 501), select_seeds=(600, 601, 602),
              report_seeds=(608, 609, 610), alphas=(0.2, 0.5, 0.8), strength=3.0, settle_ms=50.0, read_ms=100.0,
              window_ms=20, punish_type="PPL105", reward_type="PAM08")


class _Stub:
    def __init__(self, conn):
        self.conn = conn


def _oracle_kwargs(synthetic_connectome):
    c = synthetic_connectome(disjoint_kc=True)
    pops = Populations.from_connectome(c)
    a, b = design_odor_pair(pops, k=2, seed=0)
    J._RIG.clear()
    return (_Stub(c), None, pops, None, None), dict(params=P, odor_x=a, odor_y=b, **ORACLE)


def test_reward_half_equals_oracle_job(synthetic_connectome):
    """The accepted copy: same alpha_reward, same select reward block, same report pre/R1 as oracle_job."""
    rig, kw = _oracle_kwargs(synthetic_connectome)
    full = oracle_job(*rig, **kw)
    kw_r = {k: v for k, v in kw.items() if k != "punish_type"}
    mine = reward_oracle_job(*rig, **kw_r)
    assert mine["alpha_reward"] == full["alpha_reward"]
    assert mine["select"]["pre"] == full["select"]["pre"]
    assert mine["select"]["reward"] == full["select"]["reward"]
    assert mine["report"]["pre"] == full["report"]["pre"] and mine["report"]["R1"] == full["report"]["R1"]
    assert set(mine["report"]) == {"pre", "R1"} and "punish" not in mine["select"]
    assert mine["kc"] == full["kc"]
    assert any(x for row in mine["report"]["pre"]["P"] for x in row), "the synthetic regime must drive the readout"
    assert J._RIG[P][1].weights_frac() == 1.0 and J._RIG[P][1].enabled


def test_qualify_decision():
    z = {"A": SPEC.z_a, "P": SPEC.z_p}
    pre = {"A": [[10, 10]] * 8, "P": [[30, 30]] * 8}
    r1 = {"A": [[10, 10]] * 8, "P": [[30 - 12 - (i % 2), 30] for i in range(8)]}
    naive = {"a": [30] * 16, "b": [30] * 16}
    q = qualify("b", {"pre": pre, "R1": r1}, naive, SPEC, z)
    assert q["qualified"] and q["r"] >= 2.0 and q["floor_ok"] and q["x"] == "b" and q["reasons"] == []
    q2 = qualify("b", {"pre": pre, "R1": r1}, {"a": [30] * 16, "b": [4] + [30] * 15}, SPEC, z)
    assert not q2["qualified"] and not q2["floor_ok"]      # Y (or X) floor on any seed fails
    q3 = qualify("b", {"pre": pre, "R1": r1}, {"a": [4] + [30] * 15, "b": [30] * 16}, SPEC, z)
    assert not q3["qualified"] and not q3["floor_ok"]
    assert qualify("b", {"pre": pre, "R1": r1}, {"a": [5] * 16, "b": [5] * 16}, SPEC, z)["floor_ok"]  # floor inclusive
    flat = {"A": [[10, 10]] * 8, "P": [[30, 30]] * 8}
    q4 = qualify("b", {"pre": pre, "R1": flat}, naive, SPEC, z)
    assert not q4["qualified"] and q4["floor_ok"] and q4["r"] == 0.0


def test_qual_x_rule():
    assert qual_x({"a": [40] * 16, "b": [30] * 16}, SPEC) == "a"
    assert qual_x({"a": [30] * 16, "b": [40] * 16}, SPEC) == "b"
    assert qual_x({"a": [30] * 16, "b": [30] * 16}, SPEC) == SPEC.x_tie
