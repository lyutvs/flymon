"""Spec K.8.1 / K.8.7 (iii): the odd-turn (b) pairs and their odour-level overlap with the judged pairs."""
from pathlib import Path

import pytest

from flymon.brain import h4_pairs
from flymon.brain.k_pairs import odd_pairs, odour_key, overlap_report
from flymon.brain.k_spec import SPEC

ROOT = Path(__file__).resolve().parents[2]
needs_npz = pytest.mark.skipif(not (ROOT / "data/malecns.npz").exists(), reason="MaleCNS connectome not built")


def test_odour_key_ignores_order_and_float_noise():
    assert odour_key({"DA1": 0.5, "VA1d": 1.0}) == odour_key({"VA1d": 1.0 + 1e-15, "DA1": 0.5})
    assert odour_key({"DA1": 0.5}) != odour_key({"DA1": 0.25})


def test_overlap_report_finds_shared_pairs_and_odours():
    a, b, c = {"g1": 1.0}, {"g2": 1.0}, {"g3": 1.0}
    odd = [dict(axis="b", turn=1, x="m vs P", y="m vs Q", odor_x=a, odor_y=b)]
    even = [dict(axis="b", turn=0, x="n vs R", y="n vs S", odor_x=b, odor_y=a),
            dict(axis="b", turn=2, x="k vs T", y="k vs U", odor_x=c, odor_y=b)]
    rep = overlap_report(odd, {"even": even})
    assert rep["even"]["pairs"] == [{"odd": ["b", 1, "m vs P", "m vs Q"], "other": ["b", 0, "n vs R", "n vs S"]}]
    assert rep["even"]["n_shared_odours"] == 2


@needs_npz
def test_a_missing_alternate_opponent_raises_with_the_turn(monkeypatch):
    def boom(turn_index, me, opp_types, species_types):
        raise ValueError(f"turn {turn_index}: no type-disjoint alternate opponent")
    monkeypatch.setattr("flymon.brain.k_pairs.alternate_opponent", boom)
    with pytest.raises(ValueError, match="turn 1"):
        odd_pairs(_pops())


def _pops():
    from flymon.brain.circuits import Populations
    from flymon.brain.connectome import Connectome
    return Populations.from_connectome(Connectome.load(ROOT / "data/malecns.npz"))


@needs_npz
def test_odd_pairs_are_the_even_rule_on_odd_turns_and_pinned():
    pops = _pops()
    odd = odd_pairs(pops)
    assert odd and all(p["axis"] == "b" and p["turn"] % 2 == 1 for p in odd)
    even_b = [p for p in h4_pairs.even_pairs(pops) if p["axis"] == "b"]
    assert len(even_b) == 21
    assert h4_pairs.pairs_digest(odd) == SPEC.odd_pairs_digest


@needs_npz
def test_odd_and_even_keys_are_disjoint_and_the_report_is_built():
    pops = _pops()
    odd, even = odd_pairs(pops), h4_pairs.even_pairs(pops)
    assert not {h4_pairs.pair_key(p) for p in odd} & {h4_pairs.pair_key(p) for p in even}
    rep = overlap_report(odd, {"even_b": [p for p in even if p["axis"] == "b"],
                               "even_a": [p for p in even if p["axis"] == "a"]})
    assert set(rep) == {"even_b", "even_a"} and all("n_shared_odours" in v for v in rep.values())
