"""LeverFlyPool (spec AC.2, AC.7 1 / 2b): the worker applies the CSC edit before Plasticity (u_rig's order), so a
worker's KC counts, A, P, V, choice and the weight change after one pulse equal the in-process u_rig brain's."""
from pathlib import Path

import numpy as np
import pytest

from flymon.agent import policy
from flymon.brain.config import Params
from flymon.brain.fly_pool import FlyPool, FlySpec
from flymon.brain.lv_pool import LeverFlyPool

NPZ = Path("data/malecns.npz")
needs_data = pytest.mark.skipif(not NPZ.exists() or not Path("results/summary/m0d.json").exists(), reason="needs data")
P = Params(noise_mv=0.15, min_weight=1, balance_hemispheres=False, kc_thresh=0.5, learn_rate=0.05)
A = {"ORN_DM1": 1.0, "ORN_DA1": 1.0}
B = {"ORN_VA2": 1.0, "ORN_DM6": 1.0}


def test_unknown_edit_refused_before_any_worker(synthetic_npz):
    with pytest.raises(ValueError, match="not a U edit"):
        LeverFlyPool(synthetic_npz, P, [FlySpec()], edit="bogus", workers=1)


def test_none_edit_equals_flypool(synthetic_npz):
    with FlyPool(synthetic_npz, P, [FlySpec()], workers=1, timeout_s=120) as a:
        want = a.decide_batch([(0, [A, B], 7)], 1.0, 100.0, 100.0)[0]
    with LeverFlyPool(synthetic_npz, P, [FlySpec()], edit="none", workers=1, timeout_s=120) as b:
        got = b.decide_batch([(0, [A, B], 7)], 1.0, 100.0, 100.0)[0]
        assert len(b.lever_sha()) == 64
    assert np.array_equal(got, want)


@needs_data
def test_lever_sha_matches_v_spec():
    from flymon.ac.config import load_lv_config
    from flymon.ac.spec import SPEC as AC
    cfg = load_lv_config(0.0)
    with LeverFlyPool(NPZ, cfg.params, [FlySpec()], edit=AC.lever_edit, p_type=AC.p_type, workers=1,
                      timeout_s=1800) as pool:
        assert pool.lever_sha() == AC.lever_sha


@needs_data
def test_lever_worker_equals_in_process_values():
    from flymon.ac.config import load_lv_config, lv_codebook
    from flymon.ac.spec import SPEC as AC
    from flymon.agent import encode_grid as eg
    from flymon.brain.circuits import Populations
    from flymon.brain.connectome import Connectome
    from flymon.brain.h4_jobs import type_cells
    from flymon.brain.presentation import decide, reinforce
    from flymon.brain.u_measure import u_rig
    cfg = load_lv_config(0.002)
    conn = Connectome.load(str(NPZ)); pops = Populations.from_connectome(conn)
    cb, _ = lv_codebook()
    rc = {str(t): len(v) for t, v in pops.receptor_types.items()}
    odours = [eg.odour(rc, cb, "WATER", ("FIRE", "FLYING"), "norm"),
              eg.odour(rc, cb, "ELECTRIC", ("FIRE", "FLYING"), "norm"),
              eg.odour(rc, cb, "GROUND", ("FIRE", "FLYING"), "norm")]
    cells = type_cells(conn, ["MBON13", "MBON05"])
    na, npp = len(cells["MBON13"]), len(cells["MBON05"])
    idx = np.concatenate([cells["MBON13"], cells["MBON05"], np.asarray(pops.kc)])
    eng, pl, _, _, _, _ = u_rig(conn, pops, cfg.params, AC.lever_edit, AC.p_type)
    pl.reset_weights()
    want = decide(eng, pl, pops, odours, cfg.strength, 123, cfg.settle_ms, cfg.read_ms, idx)
    reinforce(eng, pl, pops, odours[0], cfg.strength, "PAM08", 400.0, 456, cfg.settle_ms, 200.0, True)
    want_dw = eng.csc.w[pl.edges] - pl.w0
    pl.reset_weights()
    with LeverFlyPool(NPZ, cfg.params, [FlySpec()], edit=AC.lever_edit, p_type=AC.p_type, workers=1,
                      timeout_s=1800) as pool:
        got = pool.decide_batch([(0, odours, 123)], cfg.strength, cfg.settle_ms, cfg.read_ms, idx)[0]
        pool.reinforce_batch([(0, odours[0], "PAM08", 400.0, 456)], cfg.strength, cfg.settle_ms, 200.0)
        got_dw = pool.w[0] - pool.w0[None]

    def readout(c):
        a, p = c[:, :na].sum(1), c[:, na:na + npp].sum(1)
        v = policy.values(a, p, cfg.z)
        return a, p, (c[:, na + npp:] > 0).sum(1), v, int(np.argmax(v))

    (wa, wp, wk, wv, wc), (ga, gp, gk, gv, gc) = readout(want), readout(got)
    assert np.array_equal(ga, wa) and np.array_equal(gp, wp) and np.array_equal(gk, wk)
    assert np.array_equal(gv, wv) and gc == wc
    assert np.array_equal(got, want)
    assert np.array_equal(got_dw, want_dw) and np.any(got_dw != 0)
