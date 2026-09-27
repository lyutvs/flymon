import json
from pathlib import Path
import pytest
from flymon.rescope.spec import SPEC

ROOT = Path(__file__).resolve().parents[2]

def test_pair_names_and_index():
    assert SPEC.pair_names() == ("seed0", "p1000", "p1001", "p1002", "p1003", "p1004", "p1005")
    assert SPEC.pair_index("seed0") == 0 and SPEC.pair_index("p1005") == 6

def test_seed_blocks_disjoint():
    probe, train, n2, qual = set(), set(), set(), set()
    for name in SPEC.pair_names():
        for f in range(SPEC.n_flies):
            probe |= set(SPEC.probe_seeds(name, f))
            for t in range(SPEC.trials):
                train.add(SPEC.train_seed(name, f, t))
                n2.add(SPEC.train_seed(name, f, t, second_null=True))
        for v in SPEC.qual_seeds(name).values():
            qual |= set(v)
    blocks = [probe, train, n2, qual]
    assert sum(map(len, blocks)) == len(set().union(*blocks))
    assert len(probe) == 7 * 8 * 8 and len(train) == 7 * 8 * 20 and len(qual) == 7 * 24
    taurec = set(range(SPEC.taurec_seed_base, SPEC.taurec_seed_base + SPEC.taurec_pulses)) | {SPEC.taurec_gen_seed}
    assert not taurec & set().union(*blocks)

@pytest.mark.skipif(not (ROOT / "results/summary/m0d.json").exists(), reason="m0d summary missing")
def test_z_matches_c3_record():
    c3 = json.loads((ROOT / "results/summary/m0d.json").read_text())["h4"]["h4"]["combos"]["C3"]
    assert tuple(c3["z"]["A"]) == SPEC.z_a and tuple(c3["z"]["P"]) == SPEC.z_p
    assert c3["readout"] == {"A": SPEC.a_type, "P": SPEC.p_type}
