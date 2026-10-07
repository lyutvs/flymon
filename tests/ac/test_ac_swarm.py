"""LVSwarm (AC.2 tie rule, AC.3 shadow record) on the synthetic pool of tests/agent/test_swarm.py."""
from types import SimpleNamespace

import pytest

from flymon.ac.swarm import LVSwarm, egrid_fn, kc_ratio_exceeds
from flymon.agent.config import AgentConfig
from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.fly_pool import FlyPool, FlySpec
from flymon.brain.h4_jobs import type_cells

P = Params(noise_mv=0.15, min_weight=1, balance_hemispheres=False, kc_thresh=0.5, learn_rate=0.05)
A = {"ORN_DM1": 1.0, "ORN_DA1": 1.0}
CFG = AgentConfig(params=P, z={"A": (5.0, 2.0), "P": (5.0, 2.0)}, readout={"A": "MBON03", "P": "MBON01"},
                  strength=1.0, settle_ms=100.0, read_ms=100.0)


@pytest.fixture(scope="module")
def parts(synthetic_npz):
    conn = Connectome.load(str(synthetic_npz))
    pops = Populations.from_connectome(conn)
    pool = FlyPool(synthetic_npz, P, [FlySpec(), FlySpec()], workers=2, timeout_s=120)
    yield pool, type_cells(conn, ["MBON03", "MBON01"]), pops.kc
    pool.terminate()


def _req(ctx, battle=None, cands=None):
    return SimpleNamespace(player_id="p", battle=battle, candidates=cands or [None] * len(ctx["odours"]), context=ctx)


def _ctx(turn, odours):
    return {"fly": 0, "battle_id": "AE-f00-b000", "battle_index": 0, "turn": turn, "k": 0, "odours": odours}


async def test_eval_tie_is_broken_by_seed_and_logged(parts):
    pool, cells, kc = parts
    sw = LVSwarm(pool, CFG, cells, kc, mode="eval")
    picks = set()
    for turn in range(16):
        ctx = _ctx(turn, [A, A])                    # identical odours, one seed per decision -> identical counts
        picks.add((await sw.decide_run_batch([_req(ctx)]))[0])
        assert ctx["detail"]["tie"] is True
    assert picks == {0, 1}


async def test_learn_mode_records_the_tie_and_keeps_softmax(parts):
    pool, cells, kc = parts
    sw = LVSwarm(pool, CFG, cells, kc, mode="learn")
    ctx = _ctx(3, [A, A])
    idx = (await sw.decide_run_batch([_req(ctx)]))[0]
    assert idx in (0, 1) and ctx["detail"]["tie"] is True and ctx["detail"]["tau"] == pytest.approx(1.0)


async def test_detail_carries_the_shadow_fields(parts):
    pool, cells, kc = parts
    sw = LVSwarm(pool, CFG, cells, kc, mode="eval", egrid=lambda b, m: ["WATER", ["FIRE"]])
    ctx = _ctx(1, [A, A])
    await sw.decide_run_batch([_req(ctx)])
    d = ctx["detail"]
    for k in ("v", "a", "p", "kc_active", "tau", "seed", "tie", "kc_ratio_gt2", "egrid"):
        assert k in d
    assert d["egrid"] is None                      # no battle on the request -> no E-grid cells
    ctx = _ctx(2, [A, A])
    await sw.decide_run_batch([_req(ctx, battle=object(), cands=["m1", "m2"])])
    assert ctx["detail"]["egrid"] == [["WATER", ["FIRE"]], ["WATER", ["FIRE"]]]


def test_kc_ratio_exceeds():
    assert kc_ratio_exceeds([10, 21], 2.0) and not kc_ratio_exceeds([10, 20], 2.0)
    assert kc_ratio_exceeds([0, 3], 2.0) and not kc_ratio_exceeds([0, 0], 2.0)
    assert not kc_ratio_exceeds([5], 2.0)


def test_egrid_fn_reads_move_type_and_opponent_types():
    class Enc:
        move_info = {"Surf": ("WATER", 95)}
        move_by_id = {"surf": "Surf"}
        species_types = {"Charizard": ("FLYING", "FIRE")}

        def _species(self, mon):
            return mon.species

    battle = SimpleNamespace(opponent_active_pokemon=SimpleNamespace(species="Charizard"))
    assert egrid_fn(Enc())(battle, SimpleNamespace(id="surf")) == ["WATER", ["FIRE", "FLYING"]]
