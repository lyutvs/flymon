from pathlib import Path
import pytest
from flymon.rescope.pairs import new_pairs, pairs_digest, sample_pair, pair_odors
from flymon.rescope.spec import SPEC

ROOT = Path(__file__).resolve().parents[2]
needs_npz = pytest.mark.skipif(not (ROOT / "data/malecns.npz").exists(), reason="MaleCNS connectome not built")

@pytest.fixture(scope="module")
def pops():
    from flymon.brain.connectome import Connectome
    from flymon.brain.circuits import Populations
    return Populations.from_connectome(Connectome.load(ROOT / "data/malecns.npz"))

@needs_npz
def test_digest_pinned(pops):
    pairs = new_pairs(pops, SPEC)
    assert [p["seed"] for p in pairs] == [1000, 1001, 1002, 1003, 1004, 1005]
    assert pairs_digest(pairs) == SPEC.pairs_digest

@needs_npz
def test_pair_shape(pops):
    a, b = sample_pair(pops, 1000)
    assert len(a) == len(b) == 8 and not set(a) & set(b)
    assert "ORN_DA1" not in a + b and "ORN_V" not in a + b

@needs_npz
def test_skip_rule_skips_used_band(pops, monkeypatch):
    from flymon.rescope import pairs as mod
    used = mod.used_sets(pops, SPEC)
    real = mod.sample_pair
    calls = []
    def fake(p, seed, k=8):
        calls.append(seed)
        if seed == 1000:  # pretend seed 1000 reproduced the seed-0 band
            s = sorted(next(iter(used)))
            return s[0::2], s[1::2]
        return real(p, seed, k)
    monkeypatch.setattr(mod, "sample_pair", fake)
    got = mod.new_pairs(pops, SPEC)
    assert got[0]["seed"] == 1001 and len(got) == 6 and calls[:2] == [1000, 1001]

@needs_npz
def test_pair_odors_seed0_is_design_pair(pops):
    from flymon.brain.stimuli import design_odor_pair
    a, b = design_odor_pair(pops, k=8, seed=0)
    assert pair_odors(pops, "seed0", SPEC) == {"a": a, "b": b}
