"""M3 gate: on the C3 engine, a FlyPool worker's decision counts equal presentation.decide in-process, bit for bit."""
from pathlib import Path

import numpy as np
import pytest

from flymon.agent.config import load_c3_config
from flymon.brain import h4_pairs
from flymon.brain.circuits import Populations, compartments
from flymon.brain.connectome import Connectome
from flymon.brain.engine_cpu import Engine
from flymon.brain.fly_pool import FlyPool, FlySpec
from flymon.brain.plasticity import Plasticity
from flymon.brain.presentation import decide

NPZ = Path("data/malecns.npz")
pytestmark = pytest.mark.skipif(not NPZ.exists() or not Path("results/summary/m0d.json").exists(), reason="needs data")


def test_worker_equals_in_process_on_c3():
    cfg = load_c3_config()
    conn = Connectome.load(str(NPZ)); pops = Populations.from_connectome(conn)
    pairs = h4_pairs.even_pairs(pops)
    odours = [pairs[0]["odor_x"], pairs[0]["odor_y"]]
    eng = Engine(conn, pops, cfg.params, seed=0)
    pl = Plasticity(eng, pops, compartments(conn, pops, cfg.params.core_frac))
    want = decide(eng, pl, pops, odours, cfg.strength, 123, cfg.settle_ms, cfg.read_ms)
    with FlyPool(NPZ, cfg.params, [FlySpec()], workers=1, timeout_s=1800) as pool:
        got = pool.decide_batch([(0, odours, 123)], cfg.strength, cfg.settle_ms, cfg.read_ms)[0]
    assert np.array_equal(got, want)
