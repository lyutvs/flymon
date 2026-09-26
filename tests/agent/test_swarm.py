import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from flymon.agent.config import AgentConfig
from flymon.agent.jobs import nonplastic_invariance_job
from flymon.agent.swarm import BrainSwarm, SafetyStop
from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.fly_pool import FlyPool, FlySpec
from flymon.brain.h4_jobs import type_cells

P = Params(noise_mv=0.15, min_weight=1, balance_hemispheres=False, kc_thresh=0.5, learn_rate=0.05)
A = {"ORN_DM1": 1.0, "ORN_DA1": 1.0}
B = {"ORN_VA2": 1.0, "ORN_DM6": 1.0}
CFG = AgentConfig(params=P, z={"A": (5.0, 2.0), "P": (5.0, 2.0)}, readout={"A": "MBON03", "P": "MBON01"},
                  strength=1.0, settle_ms=100.0, read_ms=100.0)  # the synthetic KCs stay silent at C3's 0.35


@pytest.fixture(scope="module")
def swarm(synthetic_npz):
    conn = Connectome.load(str(synthetic_npz))
    pops = Populations.from_connectome(conn)
    pool = FlyPool(synthetic_npz, P, [FlySpec(), FlySpec()], workers=2, timeout_s=120)
    yield BrainSwarm(pool, CFG, type_cells(conn, ["MBON03", "MBON01"]), pops.kc)
    pool.terminate()


def _req(ctx):
    return SimpleNamespace(player_id="p", battle=None, candidates=[None] * len(ctx.get("odours", [0])), context=ctx)


async def test_decide_fills_detail_and_is_reproducible(swarm):
    ctx = {"fly": 0, "battle_id": "f00-b000", "battle_index": 0, "turn": 3, "k": 0, "odours": [A, B]}
    idx = (await swarm.decide_run_batch([_req(ctx)]))[0]
    d = ctx["detail"]
    assert idx in (0, 1) and len(d["v"]) == 2 and len(d["kc_active"]) == 2 and d["tau"] == pytest.approx(1.0)
    ctx2 = dict(ctx); ctx2.pop("detail")
    assert (await swarm.decide_run_batch([_req(ctx2)]))[0] == idx and ctx2["detail"]["v"] == d["v"]


async def test_reinforce_changes_only_that_fly(swarm):
    before = {f: swarm.pool.w[f].copy() for f in (0, 1)}
    ctx = {"fly": 0, "odour": A, "dan": "PAM08", "ms": 200.0, "seed": 5}
    assert await swarm.reinforce_run_batch([_req(ctx)]) == [0]
    assert not np.array_equal(swarm.pool.w[0], before[0])
    assert np.array_equal(swarm.pool.w[1], before[1])
    swarm.pool.w[0] = before[0]


async def test_median_floor_stops_the_run(swarm):
    """Safety constraint 4: a fly whose plastic weights fall to a median below 0.5 of w0 stops the run."""
    keep = swarm.pool.w[1].copy()
    swarm.pool.w[1] = swarm.pool.w0[None] * np.float32(0.4)
    with pytest.raises(SafetyStop, match="fly 1"):
        swarm.check_median(1)
    swarm.pool.w[1] = keep


def test_nonplastic_weights_bit_identical(swarm):
    """Safety constraint 1 (mutation test): a reinforcement changes no CSC weight outside the plastic KC->MBON edges."""
    r = swarm.pool.run_jobs(nonplastic_invariance_job, [dict(odor=A, dan="PAM08", pulse_ms=400.0, seed=1, strength=1.0)])[0]
    assert r["plastic_changed"] > 0 and r["nonplastic_changed"] == 0


def test_compartment_fracs_start_at_one(swarm):
    fr = swarm.compartment_fracs(0)
    assert set(fr) == {"PAM08", "PPL105"} and all(v == pytest.approx(1.0) for v in fr.values())
