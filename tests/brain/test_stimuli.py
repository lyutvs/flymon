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


def test_channel_strengths_on_uneven_receptor_counts(synthetic_connectome):
    p = Populations.from_connectome(synthetic_connectome(n_orn=23))
    types = sorted(p.receptor_types)
    counts = [len(p.receptor_types[t]) for t in types]
    assert sorted(counts) == [4, 4, 5, 5, 5]
    od = channel_strengths(p, types, equalize=True)
    products = [od[t] * len(p.receptor_types[t]) for t in types]
    assert max(products) - min(products) < 1e-9
    assert np.mean([od[t] for t in types]) == pytest.approx(1.0)


def test_channel_strengths_without_equalisation(synthetic_connectome):
    p = Populations.from_connectome(synthetic_connectome(n_orn=23))
    types = sorted(p.receptor_types)
    od = channel_strengths(p, types, equalize=False)
    assert od == {t: 1.0 for t in types}


def test_design_pair_rejects_too_few_types(synthetic_connectome):
    p = Populations.from_connectome(synthetic_connectome(n_orn=23))
    with pytest.raises(ValueError, match="need 6 receptor types, have 5"):
        design_odor_pair(p, k=3, exclude=())


def test_design_pair_drive_matched_on_uneven_counts(synthetic_connectome):
    p = Populations.from_connectome(synthetic_connectome(n_orn=23))
    a, b = design_odor_pair(p, k=2, exclude=("ORN_DA1",), seed=0)
    assert len(a) == 2 and len(b) == 2 and not set(a) & set(b)
    assert "ORN_DA1" not in a and "ORN_DA1" not in b
    assert abs(total_drive(p, a) - total_drive(p, b)) / total_drive(p, a) < 0.25


def test_design_pair_seed_selects_band_offset(synthetic_connectome):
    p = Populations.from_connectome(synthetic_connectome(n_orn=23))
    cand = sorted(p.receptor_types, key=lambda t: len(p.receptor_types[t]))
    a, b = design_odor_pair(p, k=2, exclude=(), seed=0)
    assert set(a) | set(b) == set(cand[0:4])
    selections = {frozenset(a) | frozenset(b)}
    for seed in range(1, 31):
        sa, sb = design_odor_pair(p, k=2, exclude=(), seed=seed)
        selections.add(frozenset(sa) | frozenset(sb))
    assert len(selections) >= 2
