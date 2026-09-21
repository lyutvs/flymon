"""The H.3 configuration object and its odour generators (spec H.3a.3, H.3a.11)."""
import dataclasses
import hashlib
from pathlib import Path

import numpy as np
import pytest

from flymon.brain.circuits import Populations
from flymon.brain.connectome import Connectome
from flymon.brain.h3_spec import SPEC, OdorSet, all51_glomeruli, make_odors, odor_digest
from flymon.brain.stimuli import channel_strengths

NPZ = Path("data/malecns.npz")
REFERENCE_DIGEST = "2930afee11d53407ef58455ea51544b90a75cffc712dd8bad6078e62c11fb87d"     # 48 odours, rng 800000
EXTENDED_DIGEST = "f24f0ffaff9d6669e724189200d07c8cf91620a36d63cb5d6f297bb33fb2d9a1"      # 384 odours, rng 800001


def _diagnostic_rule(pops, n, seed, probe0, fmt, k_min, k_max, exclude):
    """apl_input_scale_sweep.reference_odors (8b51594), parameterised; the port must make the same draws."""
    cand = [t for t in pops.receptor_types if t not in exclude]
    rng = np.random.default_rng(seed)
    out = []
    for j in range(n):
        k = int(rng.integers(k_min, k_max + 1))
        types = sorted(str(t) for t in rng.choice(cand, size=k, replace=False))
        out.append(dict(name=fmt.format(j=j), types=types, strengths=channel_strengths(pops, types),
                        seeds=[probe0 + 2 * j, probe0 + 1 + 2 * j]))
    return out


def test_the_table_values_are_the_declared_ones():
    s = SPEC
    assert (s.reference.n, s.reference.rng_seed, s.reference.probe0, s.reference.k_min, s.reference.k_max) == \
        (48, 800_000, 800_100, 6, 9)
    assert (s.extended.n, s.extended.rng_seed, s.extended.probe0) == (384, 800_001, 810_100)
    assert s.strength == 0.35 and s.reference_window == type(s.reference_window)(800.0, 600.0)
    assert s.design_seeds == tuple(range(100, 108)) and s.design_extra_seeds == tuple(range(108, 116))
    assert (s.design_window.settle_ms, s.design_window.read_ms) == (200.0, 600.0)
    assert s.baseline_cal_seeds == tuple(range(100, 116)) and s.baseline_gate_seeds == tuple(range(116, 148))
    assert (s.boot_draws, s.boot_seed) == (10_000, 20260918) and s.all51_seeds == tuple(range(200, 208))
    assert s.runaway_rest_seeds == tuple(range(100, 108)) and s.runaway_odor_seeds == tuple(range(100, 164))
    assert (s.runaway_rest_sat_hz, s.runaway_odor_sat_hz) == (100.0, 150.0)
    assert s.kc_grid == (1.55, 1.60, 1.65, 1.70) and s.apl_r_max == 0.333 and s.scale_bracket == (0.005, 1.0)
    assert (s.homeo_eta, s.homeo_target, s.homeo_target_digits, s.homeo_median_tol, s.homeo_max_iter) == \
        (0.1, 0.062, 3, 0.0104, 40)
    assert s.runaway_odor_record_hz == 100.0 and len(s.connectome_sha256) == 64


def test_the_spec_is_frozen_and_hashable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        SPEC.strength = 1.0
    hash(SPEC)


def test_make_odors_makes_the_diagnostic_draws(synthetic_connectome):
    pops = Populations.from_connectome(synthetic_connectome())
    oset = OdorSet("S{j:02d}", 4, 11, 1000, k_min=1, k_max=2, exclude=("ORN_DA1",), n_candidates=None)
    got = make_odors(pops, oset)
    assert got == _diagnostic_rule(pops, 4, 11, 1000, "S{j:02d}", 1, 2, ("ORN_DA1",))
    assert [o["seeds"] for o in got] == [[1000, 1001], [1002, 1003], [1004, 1005], [1006, 1007]]


def test_make_odors_refuses_a_wrong_candidate_count_and_repeated_sets(synthetic_connectome):
    pops = Populations.from_connectome(synthetic_connectome())
    with pytest.raises(ValueError, match="candidate glomeruli"):
        make_odors(pops, OdorSet("S{j}", 2, 11, 0, k_min=1, k_max=2, exclude=("ORN_DA1",), n_candidates=51))
    with pytest.raises(ValueError, match="not distinct"):
        make_odors(pops, OdorSet("S{j}", 4, 12, 0, k_min=1, k_max=2, exclude=("ORN_DA1",), n_candidates=None))


def test_odor_digest_changes_with_any_field():
    o = [dict(name="R00", types=["A"], strengths={"A": 1.0}, seeds=[1, 2])]
    d = odor_digest(o)
    assert odor_digest([dict(o[0], seeds=[1, 3])]) != d and odor_digest([dict(o[0], strengths={"A": 1.5})]) != d


@pytest.mark.skipif(not NPZ.exists(), reason="MaleCNS connectome not built")
def test_the_real_odour_sets_are_the_calibrated_ones():
    pops = Populations.from_connectome(Connectome.load(NPZ))
    ref = make_odors(pops, SPEC.reference)
    ext = make_odors(pops, SPEC.extended)
    assert odor_digest(ref) == REFERENCE_DIGEST and odor_digest(ext) == EXTENDED_DIGEST
    assert ref[0]["types"] == ["ORN_DA2", "ORN_DA4l", "ORN_DP1m", "ORN_VA7m", "ORN_VL2p", "ORN_VM1", "ORN_VM5d"]
    gloms, c_norm = all51_glomeruli(pops, SPEC.reference.exclude)
    assert len(gloms) == 51 and c_norm == 34.0
    assert hashlib.sha256(NPZ.read_bytes()).hexdigest() == SPEC.connectome_sha256
