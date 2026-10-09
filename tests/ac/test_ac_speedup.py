"""Bit-identical speedups on the AC brain path (speedup brief 2026-10-10): every new path must give exactly the old
path's numbers (np.array_equal / ==, never allclose) on the same inputs and seeds."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import pytest

NPZ = Path("data/malecns.npz")
needs_data = pytest.mark.skipif(not NPZ.exists() or not Path("results/summary/m0d.json").exists(), reason="needs data")


def _say(msg: str) -> None:
    print(f"[speedup {time.strftime('%H:%M:%S')}] {msg}", flush=True)


@pytest.fixture(scope="module")
def rig():
    from flymon.ac.config import load_lv_config, lv_codebook
    from flymon.agent import encode_grid as eg
    from flymon.brain.circuits import Populations, compartments
    from flymon.brain.connectome import Connectome
    from flymon.brain.h4_jobs import type_cells
    cfg = load_lv_config(0.002)
    conn = Connectome.load(str(NPZ)); pops = Populations.from_connectome(conn)
    cb, _ = lv_codebook()
    rc = {str(t): len(v) for t, v in pops.receptor_types.items()}
    odours = [eg.odour(rc, cb, m, ("FIRE", "FLYING"), "norm") for m in ("WATER", "ELECTRIC", "GROUND")]
    cells = type_cells(conn, ["MBON13", "MBON05"])
    idx = np.concatenate([cells["MBON13"], cells["MBON05"], np.asarray(pops.kc)])
    comps = compartments(conn, pops, cfg.params.core_frac)
    return dict(cfg=cfg, conn=conn, pops=pops, odours=odours, idx=idx, comps=comps)


def _sequence(eng, pl, pops, cfg, odours, idx) -> list:
    """decide / reinforce steps (PAM08, PPL105, no DAN) with every output and the plastic weights after each step."""
    from flymon.brain.presentation import decide, reinforce
    out = []
    for k, (dan, ms) in enumerate((("PAM08", 400.0), ("PPL105", 400.0), (None, 0.0))):
        out.append(decide(eng, pl, pops, odours, cfg.strength, 100 + k, cfg.settle_ms, cfg.read_ms, idx))
        reinforce(eng, pl, pops, odours[k], cfg.strength, dan, ms, 200 + k, cfg.settle_ms, 200.0, True)
        out += [eng.csc.w[pl.edges].copy(), pl.kc_trace.copy(), pl.da.copy(), pl.da_base.copy()]
        _say(f"sequence step {k} done")
    out.append(decide(eng, pl, pops, odours, cfg.strength, 300, cfg.settle_ms, cfg.read_ms, idx))
    return out


# ---------------------------------------------------------------- fix 1: masks instead of np.isin in Plasticity.on_step
@needs_data
def test_mask_plasticity_equals_isin_plasticity_in_process(rig):
    from flymon.ac.spec import SPEC as AC
    from flymon.brain.engine_cpu import Engine
    from flymon.brain.lv_pool import MaskPlasticity
    from flymon.brain.plasticity import Plasticity
    from flymon.brain.u_measure import apply_u_edit
    cfg, pops = rig["cfg"], rig["pops"]
    eng = Engine(rig["conn"], pops, cfg.params, seed=0)
    apply_u_edit(eng, pops, AC.lever_edit, AC.p_type)
    old = Plasticity(eng, pops, rig["comps"])
    _say("old path (np.isin)")
    t = time.perf_counter(); want = _sequence(eng, old, pops, cfg, rig["odours"], rig["idx"]); t_old = time.perf_counter() - t
    old.reset_weights()
    new = MaskPlasticity(eng, pops, rig["comps"])           # rebinds engine.on_step
    assert np.array_equal(new.edges, old.edges) and np.array_equal(new.w0, old.w0)
    _say("new path (masks)")
    t = time.perf_counter(); got = _sequence(eng, new, pops, cfg, rig["odours"], rig["idx"]); t_new = time.perf_counter() - t
    _say(f"sequence wall: isin {t_old:.2f} s, masks {t_new:.2f} s")
    assert len(got) == len(want)
    for i, (g, w) in enumerate(zip(got, want)):
        assert g.dtype == w.dtype and g.shape == w.shape, i
        assert np.array_equal(g, w), i
    assert not np.array_equal(want[1], old.w0)               # the sequence did learn (weights moved)


# ---------------------------------------------------------------- pool-level helpers
def _pool(rig, flies=2, workers=4, **kw):
    from flymon.ac.spec import SPEC as AC
    from flymon.brain.fly_pool import FlySpec
    from flymon.brain.lv_pool import LeverFlyPool
    return LeverFlyPool(NPZ, rig["cfg"].params, [FlySpec() for _ in range(flies)], edit=AC.lever_edit,
                        p_type=AC.p_type, workers=workers, timeout_s=1800, **kw)


def _same_counts(got, want) -> None:
    assert len(got) == len(want)
    for g, w in zip(got, want):
        g, w = np.asarray(g), np.asarray(w)
        assert g.dtype == w.dtype and g.shape == w.shape
        assert np.array_equal(g, w)


def _sit_inputs(rig):
    """Two synthetic confirmation pairs over the rig's three odours (situations.evaluate's record shape)."""
    o = rig["odours"]
    pairs = [dict(cands=["water", "thunder", "quake"], best1="water", best2="quake"),
             dict(cands=["water", "thunder"], best1="thunder", best2="water")]
    odours = [(o, o[::-1]), (o[:2], o[1:])]
    return pairs, odours


def _evaluate(pool, fly, rig):
    from flymon.ac import situations
    from flymon.brain.h4_jobs import type_cells
    from flymon.ac.spec import SPEC as AC
    cfg = rig["cfg"]
    cells = type_cells(rig["conn"], [cfg.readout["A"], cfg.readout["P"]])
    pairs, odours = _sit_inputs(rig)
    return situations.evaluate(pool, fly, 0, pairs, odours, cfg, cells[cfg.readout["A"]], cells[cfg.readout["P"]],
                               rig["pops"].kc, AC.tie_seed)


class _OldDecide:
    """A view of a LeverFlyPool whose decide_batch is FlyPool's (one job per fly, all its candidates serially)."""

    def __init__(self, pool):
        self._p = pool

    def __getattr__(self, k):
        return getattr(self._p, k)

    def decide_batch(self, *a, **kw):
        from flymon.brain.fly_pool import FlyPool
        return FlyPool.decide_batch(self._p, *a, **kw)


# ---------------------------------------------------------------- fix 2: one decide job per candidate
@needs_data
def test_per_candidate_decide_equals_per_fly_decide(rig):
    from flymon.brain.fly_pool import FlyPool
    from flymon.rescope.blocks import weights_sha
    cfg, o, idx = rig["cfg"], rig["odours"], rig["idx"]
    reqs = [(0, o, 11), (1, o[::-1], 12), (0, o[:2], 13)]
    with _pool(rig) as pool:
        _say("pool up")
        for step in range(2):
            t = time.perf_counter(); want = FlyPool.decide_batch(pool, reqs, cfg.strength, cfg.settle_ms, cfg.read_ms, idx)
            t_old = time.perf_counter() - t
            t = time.perf_counter(); got = pool.decide_batch(reqs, cfg.strength, cfg.settle_ms, cfg.read_ms, idx)
            t_new = time.perf_counter() - t
            _say(f"step {step}: per-fly jobs {t_old:.2f} s, per-candidate jobs {t_new:.2f} s")
            _same_counts(got, want)
            full_old = FlyPool.decide_batch(pool, reqs[:1], cfg.strength, 100.0, 100.0)
            _same_counts(pool.decide_batch(reqs[:1], cfg.strength, 100.0, 100.0), full_old)     # idx=None too
            pool.reinforce_batch([(0, o[0], "PAM08", 400.0, 21 + step), (1, o[1], "PPL105", 400.0, 31 + step)],
                                 cfg.strength, cfg.settle_ms)
        sha = [weights_sha(pool.w[f]) for f in (0, 1)]
        assert sha[0] != weights_sha(pool.w0[None])
        old_rec, new_rec = _evaluate(_OldDecide(pool), 0, rig), _evaluate(pool, 0, rig)
        assert new_rec == old_rec and new_rec["frozen"]
        assert [weights_sha(pool.w[f]) for f in (0, 1)] == sha
        with pytest.raises(ValueError, match="at least one candidate"):
            pool.decide_batch([(0, [], 1)], cfg.strength, cfg.settle_ms, cfg.read_ms, idx)
        n_jobs, real = [], pool._map
        pool._map = lambda fn, items: (n_jobs.append(len(items)), real(fn, items))[1]
        pool.decide_batch(reqs, cfg.strength, 100.0, 100.0)
        assert n_jobs == [sum(len(c) for _, c, _ in reqs)]                # one job per candidate


# ---------------------------------------------------------------- fix 3: no whole-batch waiting
import asyncio  # noqa: E402
import inspect  # noqa: E402
import threading  # noqa: E402

from tests.battle import test_barrier as _barrier_tests  # noqa: E402

_BARRIER_TESTS = [n for n, f in vars(_barrier_tests).items() if n.startswith("test_") and inspect.iscoroutinefunction(f)]


@pytest.mark.parametrize("name", _BARRIER_TESTS)
async def test_overlap_barrier_keeps_batch_barrier_behaviour(name, monkeypatch):
    from flymon.ac.barrier import OverlapBarrier
    monkeypatch.setattr(_barrier_tests, "BatchBarrier", OverlapBarrier)
    await getattr(_barrier_tests, name)()


async def test_overlap_barrier_runs_a_second_batch_while_the_first_is_in_flight():
    from flymon.ac.barrier import OverlapBarrier
    running, peak, calls = [0], [0], []
    release = asyncio.Event()

    async def run_batch(reqs):
        calls.append([r.player_id for r in reqs])
        running[0] += 1; peak[0] = max(peak[0], running[0])
        if reqs[0].player_id == "a":
            await release.wait()                         # batch "a" is the slow one
        running[0] -= 1
        return [i for i, _ in enumerate(reqs)]

    bar = OverlapBarrier(run_batch, deadline_ms=20)
    bar.register("a"); bar.register("b")
    ta = asyncio.create_task(bar.submit("a", None, ["x"], {}))
    await asyncio.sleep(0.05)                            # deadline fired: batch "a" in flight
    tb = asyncio.create_task(bar.submit("b", None, ["x"], {}))
    assert await asyncio.wait_for(tb, 1.0) == 0          # "b" is served while "a" still runs
    assert not ta.done() and peak[0] == 2
    release.set()
    assert await asyncio.wait_for(ta, 1.0) == 0
    assert calls == [["a"], ["b"]]


def _fly_sequence(pool, fly, rig, decide, out):
    """One fly's own ordered decide / reinforce steps (as a player awaits them), then a situation evaluation."""
    from flymon.rescope.blocks import weights_sha
    cfg, o, idx = rig["cfg"], rig["odours"], rig["idx"]
    rec = []
    for k, (dan, ms) in enumerate((("PAM08", 400.0), ("PPL105", 400.0))):
        rec.append(decide(pool, [(fly, o[k:] + o[:k], 40 + 10 * fly + k)], cfg.strength, cfg.settle_ms, cfg.read_ms,
                          idx)[0])
        pool.reinforce_batch([(fly, o[(k + fly) % 3], dan, ms, 60 + 10 * fly + k)], cfg.strength, cfg.settle_ms)
        rec.append(pool.w[fly].copy())
    rec.append(weights_sha(pool.w[fly]))
    rec.append(_evaluate(pool, fly, rig))
    out[fly] = rec


@needs_data
def test_overlapping_pool_calls_equal_the_serial_old_path(rig):
    from flymon.brain.fly_pool import FlyPool
    cfg, o, idx = rig["cfg"], rig["odours"], rig["idx"]
    want: dict = {}
    with _pool(rig, flies=2, workers=4, mask_plasticity=False, overlap=False) as old:
        _say("old pool up (Plasticity, per-fly jobs, lock held while waiting)")
        t = time.perf_counter()
        for f in (0, 1):
            _fly_sequence(_OldDecide(old), f, rig, FlyPool.decide_batch, want)
        t_old = time.perf_counter() - t
        lever_old = old.lever_sha()
        hold = threading.Thread(target=old.decide_batch, args=([(0, o, 1)], cfg.strength, cfg.settle_ms, cfg.read_ms, idx))
        hold.start(); time.sleep(0.5)
        assert old._lock.locked()                        # the old _map holds the lock while its batch runs
        hold.join()
    got: dict = {}
    with _pool(rig, flies=2, workers=4) as new:
        _say("new pool up (all three fixes); two flies in two threads at once")
        t = time.perf_counter()
        ths = [threading.Thread(target=_fly_sequence, args=(new, f, rig, LeverFlyPoolDecide, got)) for f in (0, 1)]
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        t_new = time.perf_counter() - t
        assert new.lever_sha() == lever_old
        hold = threading.Thread(target=new.decide_batch, args=([(0, o, 1)], cfg.strength, cfg.settle_ms, cfg.read_ms, idx))
        hold.start(); time.sleep(0.5)
        assert not new._lock.locked()                    # waiting for workers no longer holds the lock
        hold.join()
    _say(f"two-fly sequence wall: old serial {t_old:.2f} s, new overlapping {t_new:.2f} s")
    assert sorted(got) == [0, 1]
    for f in (0, 1):
        g, w = got[f], want[f]
        _same_counts([g[0], g[2]], [w[0], w[2]])          # decide counts
        _same_counts([g[1], g[3]], [w[1], w[3]])          # weights after each reinforce
        assert g[4] == w[4]                               # weights_sha
        assert g[5] == w[5] and g[5]["frozen"]            # situation-eval record


def LeverFlyPoolDecide(pool, *a, **kw):
    return pool.decide_batch(*a, **kw)


async def test_lvswarm_overlapping_batches_equal_serial_batches(synthetic_npz):
    from flymon.ac.swarm import LVSwarm
    from flymon.brain.circuits import Populations
    from flymon.brain.connectome import Connectome
    from flymon.brain.fly_pool import FlySpec
    from flymon.brain.h4_jobs import type_cells
    from flymon.brain.lv_pool import LeverFlyPool
    from tests.ac.test_ac_swarm import CFG, P, _ctx, _req
    conn = Connectome.load(str(synthetic_npz)); pops = Populations.from_connectome(conn)
    cells = type_cells(conn, ["MBON03", "MBON01"])
    B = {"ORN_VA2": 1.0, "ORN_DM6": 1.0}
    A = {"ORN_DM1": 1.0, "ORN_DA1": 1.0}

    def batches():
        return [[_req(dict(_ctx(t, [A, B]), fly=0))] for t in range(4)] + [[_req(dict(_ctx(t, [B, A]), fly=1))]
                                                                             for t in range(4)]
    with LeverFlyPool(synthetic_npz, P, [FlySpec(), FlySpec()], edit="none", workers=2, timeout_s=120,
                      overlap=False) as old:
        sw = LVSwarm(old, CFG, cells, pops.kc, mode="eval")
        assert sw._exec._max_workers == 1 and not sw.overlapping
        want_b = batches()
        want = [await sw.decide_run_batch(b) for b in want_b]
    with LeverFlyPool(synthetic_npz, P, [FlySpec(), FlySpec()], edit="none", workers=2, timeout_s=120) as new:
        sw = LVSwarm(new, CFG, cells, pops.kc, mode="eval")
        assert sw._exec._max_workers > 1 and sw.overlapping
        got_b = batches()
        got = await asyncio.gather(*(sw.decide_run_batch(b) for b in got_b))
    assert list(got) == want
    assert [b[0].context["detail"] for b in got_b] == [b[0].context["detail"] for b in want_b]


def test_acbattles_uses_the_overlap_barrier_only_for_an_overlapping_swarm(tmp_path):
    from types import SimpleNamespace
    from flymon.ac.barrier import OverlapBarrier
    from flymon.ac.battles import ACBattles
    from flymon.battle.barrier import BatchBarrier

    async def rb(reqs):
        return [0] * len(reqs)

    async def make(overlapping):
        sw = SimpleNamespace(decide_run_batch=rb, reinforce_run_batch=rb, overlapping=overlapping)
        return ACBattles({0: "FLY"}, tmp_path, None, swarm=sw).bars
    bars = asyncio.run(make(True))
    assert all(type(b) is OverlapBarrier for pair in bars.values() for b in pair)
    bars = asyncio.run(make(False))
    assert all(type(b) is BatchBarrier for pair in bars.values() for b in pair)


# ---------------------------------------------------------------- rollback race: a stale reinforce must not land
def test_rollback_during_inflight_reinforce_survives(synthetic_npz):
    from flymon.brain.fly_pool import FlySpec
    from flymon.brain.lv_pool import LeverFlyPool
    from tests.ac.test_ac_swarm import P
    A = {"ORN_DM1": 1.0, "ORN_DA1": 1.0}
    with LeverFlyPool(synthetic_npz, P, [FlySpec(), FlySpec()], edit="none", workers=2, timeout_s=120) as pool:
        snap = np.array(pool.w[0], copy=True)                 # blocks.snapshot_for
        workers_done, release, real = threading.Event(), threading.Event(), pool._map

        def held_map(fn, items):                              # jobs finished on the workers, result not yet written
            out = real(fn, items)
            workers_done.set(); release.wait(30)
            return out
        pool._map = held_map
        req = [(0, A, "PAM08", 400.0, 5), (1, A, "PAM08", 400.0, 6)]
        th = threading.Thread(target=pool.reinforce_batch, args=(req, 1.0, 100.0))
        th.start()
        assert workers_done.wait(60)
        pool.w[0] = snap.copy()                               # blocks.rollback_for while the reinforce is in flight
        release.set(); th.join(60)
        assert np.array_equal(pool.w[0], snap)                # the rolled-back weights survive
        assert not np.array_equal(pool.w[1], pool.w0[None])   # fly 1 (not rolled back) got its reinforce
        pool._map = real
        pool.reinforce_batch([req[0]], 1.0, 100.0)            # the next reinforce of fly 0 lands normally
        assert not np.array_equal(pool.w[0], snap)
