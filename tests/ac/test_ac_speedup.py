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
