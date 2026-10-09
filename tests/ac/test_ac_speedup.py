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
