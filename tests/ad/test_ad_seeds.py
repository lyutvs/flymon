"""AD.1 / AD.5 4: every derive_seed key that names a fly uses its global number (24-47), never the pool index, while
AC's default (no mapping) is unchanged. The OS draw (Task 4) and the pulse seed (Task 2) have their own tests."""
import numpy as np
import pytest

from flymon.ac import situations as si
from flymon.ac.swarm import LVSwarm
from flymon.agent import policy
from flymon.agent.policy import derive_seed
from flymon.brain.circuits import Populations
from flymon.brain.connectome import Connectome
from flymon.brain.fly_pool import FlyPool, FlySpec
from flymon.brain.h4_jobs import type_cells
from tests.ac.test_ac_situations import PAIRS, Cfg, FakePool, _counts
from tests.ac.test_ac_swarm import CFG, P, _ctx, _req

A = {"ORN_DM1": 1.0, "ORN_DA1": 1.0}


@pytest.fixture(scope="module")
def parts(synthetic_npz):
    conn = Connectome.load(str(synthetic_npz))
    pops = Populations.from_connectome(conn)
    pool = FlyPool(synthetic_npz, P, [FlySpec(), FlySpec()], workers=2, timeout_s=120)
    yield pool, type_cells(conn, ["MBON03", "MBON01"]), pops.kc
    pool.terminate()


def _spy(monkeypatch):
    seen, real = [], policy.argmax_tiebreak
    monkeypatch.setattr(policy, "argmax_tiebreak", lambda v, s: (seen.append(s), real(v, s))[1])
    return seen


async def test_decision_and_tie_seeds_use_the_global_fly(parts, monkeypatch):
    pool, cells, kc = parts
    seen = _spy(monkeypatch)
    sw = LVSwarm(pool, CFG, cells, kc, mode="eval", seed_fly={0: 24, 1: 25})
    ctx = dict(_ctx(2, [A, A]), fly=1, battle_id="DE-f25-b003")
    await sw.decide_run_batch([_req(ctx)])
    assert ctx["detail"]["seed"] == derive_seed("decide", 25, "DE-f25-b003", 2, 0)
    assert seen[-1] == derive_seed(305, "tie", 25, "DE-f25-b003", 2, 0)


async def test_ac_swarm_without_mapping_keys_on_the_pool_index(parts, monkeypatch):
    pool, cells, kc = parts
    seen = _spy(monkeypatch)
    sw = LVSwarm(pool, CFG, cells, kc, mode="eval")
    ctx = dict(_ctx(2, [A, A]), fly=1, battle_id="AE-f01-b003")
    await sw.decide_run_batch([_req(ctx)])
    assert ctx["detail"]["seed"] == derive_seed("decide", 1, "AE-f01-b003", 2, 0)
    assert seen[-1] == derive_seed(305, "tie", 1, "AE-f01-b003", 2, 0)


SCRIPT = {(0, 0): _counts(0), (0, 1): _counts(1), (1, 0): _counts(2), (1, 1): _counts(1)}
ODS = [([{}] * 3, [{}] * 3)] * 2
IDX = (np.array([0]), np.array([1]), np.array([2, 3]))


def test_situation_seeds_use_the_global_fly_and_the_key(monkeypatch):
    seen = _spy(monkeypatch)
    pool = FakePool(SCRIPT)
    rec = si.evaluate(pool, 0, 20, PAIRS, ODS, Cfg(), *IDX, 305, seed_fly=24)
    assert [s for _, _, s in pool.calls[-1]] == [derive_seed(305, "sit", 24, i, side) for i in (0, 1) for side in (0, 1)]
    assert seen[-4:] == [derive_seed(305, "sit-tie", 24, i, side) for i in (0, 1) for side in (0, 1)]
    assert rec["fly"] == 0
    si.evaluate(pool, 0, 20, PAIRS, ODS, Cfg(), *IDX, 305, seed_fly=24, seed_key="sit-new")
    assert [s for _, _, s in pool.calls[-1]] == [derive_seed(305, "sit-new", 24, i, side) for i in (0, 1)
                                                 for side in (0, 1)]
    assert seen[-1] == derive_seed(305, "sit-new-tie", 24, 1, 1)


def test_situation_default_is_ac(monkeypatch):
    pool = FakePool(SCRIPT)
    si.evaluate(pool, 0, 20, PAIRS, ODS, Cfg(), *IDX, 305)
    assert [s for _, _, s in pool.calls[-1]] == [derive_seed(305, "sit", 0, i, side) for i in (0, 1) for side in (0, 1)]
