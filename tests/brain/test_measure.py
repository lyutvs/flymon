import json
from pathlib import Path

import numpy as np
import pytest

from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.engine_cpu import Engine
from flymon.brain.measure import chance_jaccard, jaccard, kc_sparsity, mbon_baseline
from flymon.brain.stimuli import design_odor_pair


def test_jaccard_and_chance():
    a = np.array([1, 1, 0, 0], bool); b = np.array([1, 0, 1, 0], bool)
    assert jaccard(a, b) == pytest.approx(1 / 3)
    assert chance_jaccard(0.05, 0.05) == pytest.approx(0.0025 / 0.0975)


def test_kc_sparsity_runs_on_synthetic(synthetic_connectome):
    c = synthetic_connectome()
    pops = Populations.from_connectome(c)
    eng = Engine(c, pops, Params(min_weight=1, balance_hemispheres=False), seed=0)
    a, b = design_odor_pair(pops, k=2, exclude=("ORN_DA1",))
    r = kc_sparsity(eng, pops, a, strength=1.0, seed=0, settle_ms=50, read_ms=100)
    assert 0.0 <= r["frac_active"] <= 1.0 and r["active"].shape == (40,)
    base = mbon_baseline(eng, pops, seed=0, ms=100)
    assert "mbon_hz" in base and "n_types_active" in base


def test_mbon_baseline_is_measured_per_params(synthetic_connectome):
    """Each grid point needs its own baseline: kc_thresh changes MBON drive, so one engine per Params."""
    c = synthetic_connectome()
    pops = Populations.from_connectome(c)
    out = []
    for kc_thresh in (1.0, 3.0):
        eng = Engine(c, pops, Params(min_weight=1, balance_hemispheres=False, kc_thresh=kc_thresh), seed=0)
        base = mbon_baseline(eng, pops, seed=0, ms=100)
        assert np.isfinite(base["mbon_hz"]) and base["mbon_hz"] >= 0.0
        assert 0 <= base["n_types_active"] <= len(set(c.type[pops.mbon].tolist()))
        out.append(base)
    assert len(out) == 2


@pytest.mark.skipif(not Path("results/m0/sparsity.json").exists(), reason="gate not yet run on real data")
def test_real_sparsity_gate_recorded():
    """The grid row that the frozen Params() defaults select must itself meet the spec 5 gate
    (KC sparsity 3-7%, overlap at or below chance, MBON baseline 3-4 Hz)."""
    d = json.loads(Path("results/m0/sparsity.json").read_text())
    p = Params()
    pick = [g for g in d["grid"] if g["kc_thresh"] == p.kc_thresh and g["apl_scale"] == p.apl_scale
            and g.get("mbon_hold_frac") == p.mbon_hold_frac]
    assert pick, "no sparsity grid row for the current Params() defaults"
    g = pick[0]
    assert 0.03 <= g["frac_active_A"] <= 0.07 and 0.03 <= g["frac_active_B"] <= 0.07
    assert g["jaccard"] <= g["chance"]
    assert 3.0 <= g["mbon_hz_rest"] <= 4.0
