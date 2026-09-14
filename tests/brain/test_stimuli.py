import numpy as np
import pytest

from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.engine_cpu import Engine
from flymon.brain.stimuli import channel_strengths, design_odor_pair, present, total_drive


def test_channel_strengths_equalise_by_receptor_count(synthetic_connectome):
    p = Populations.from_connectome(synthetic_connectome())
    od = channel_strengths(p, ["ORN_DM1", "ORN_DA1"], equalize=True)
    n = {t: len(p.receptor_types[t]) for t in od}
    assert od["ORN_DM1"] * n["ORN_DM1"] == pytest.approx(od["ORN_DA1"] * n["ORN_DA1"])
    assert np.mean(list(od.values())) == pytest.approx(1.0)


def test_present_sets_rates_and_clears_previous(synthetic_connectome):
    c = synthetic_connectome()
    pops = Populations.from_connectome(c)
    eng = Engine(c, pops, Params(noise_mv=0.0, min_weight=1, balance_hemispheres=False))
    present(eng, pops, {"ORN_DM1": 1.0}, strength=0.35)
    assert eng.drive_hz[pops.receptor_types["ORN_DM1"]] == pytest.approx(200 * 0.35)
    present(eng, pops, {"ORN_DA1": 0.5}, strength=1.0)
    assert (eng.drive_hz[pops.receptor_types["ORN_DM1"]] == 0).all()
    assert eng.drive_hz[pops.receptor_types["ORN_DA1"]] == pytest.approx(100.0)


def test_design_pair_disjoint_and_drive_matched(synthetic_connectome):
    p = Populations.from_connectome(synthetic_connectome())
    a, b = design_odor_pair(p, k=2, exclude=("ORN_DA1",), seed=0)
    assert len(a) == 2 and len(b) == 2 and not set(a) & set(b)
    assert "ORN_DA1" not in a and "ORN_DA1" not in b
    assert abs(total_drive(p, a) - total_drive(p, b)) / total_drive(p, a) < 0.25
