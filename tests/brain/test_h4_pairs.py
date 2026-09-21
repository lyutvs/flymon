"""The oracle's pair list (spec H.4 step 2 = G.14.3): the port equals the committed calibration code and G.12's list."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from flymon.brain import h4_pairs as H
from flymon.brain.circuits import Populations
from flymon.brain.connectome import Connectome
from flymon.brain.h4_spec import SPEC

ROOT = Path(__file__).resolve().parents[2]
NPZ = ROOT / "data/malecns.npz"
CAL = ROOT / "docs/superpowers/specs/m2-calibration-g"
G12_RAW = ROOT / "results/m2/calibration/encoders/E0_even.json"      # git-excluded: G.12's E0 even-turn rows
needs_npz = pytest.mark.skipif(not NPZ.exists(), reason="data/malecns.npz not present")


@pytest.fixture(scope="module")
def pops():
    return Populations.from_connectome(Connectome.load(NPZ))


@pytest.fixture(scope="module")
def pairs(pops):
    return H.even_pairs(pops)


def _load(name):
    sys.path.insert(0, str(CAL))
    try:
        s = importlib.util.spec_from_file_location(name, CAL / f"{name}.py")
        m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
        return m
    finally:
        sys.path.remove(str(CAL))


@needs_npz
def test_the_port_equals_the_committed_calibration_code(pops, pairs):
    P, E = _load("m2_probe"), _load("m2_encoder_compare")
    st, mi, mt, vt = P.pool_vocabulary()
    turns = [t for t in P.build_turns(st, mi, 16) if t["turn"] % 2 == 0]
    assert pairs == E.pairs_for("E0", pops, E.build_encoder("E0", pops, mt, vt), turns, st)


@needs_npz
def test_the_list_is_the_declared_one(pairs):
    assert sum(p["axis"] == "a" for p in pairs) == SPEC.n_pairs_a == 18
    assert sum(p["axis"] == "b" for p in pairs) == SPEC.n_pairs_b == 21
    assert all(p["turn"] % 2 == 0 for p in pairs)
    assert H.pairs_digest(pairs) == SPEC.pairs_digest


@needs_npz
@pytest.mark.skipif(not G12_RAW.exists(), reason="G.12 raw rows not present (git-excluded)")
def test_labels_and_channels_are_g12s(pops, pairs):
    raw = json.loads(G12_RAW.read_text())
    assert [H.pair_key(p) for p in pairs] == [(r["axis"], r["turn"], r["x"], r["y"]) for r in raw["rows"]]
    _, _, mt, vt = H.pool_vocabulary()
    assert H.e0_channels(pops, mt, vt) == raw["channels"]


def test_the_digest_sees_every_field():
    p = dict(axis="a", turn=0, x="Surf", y="Earthquake", odor_x={"ORN_A": 1.0}, odor_y={"ORN_B": 1.0})
    d = H.pairs_digest([p])
    for k, v in [("axis", "b"), ("turn", 2), ("x", "Cut"), ("y", "Cut"), ("odor_x", {"ORN_A": 1.5}),
                 ("odor_y", {"ORN_C": 1.0})]:
        assert H.pairs_digest([dict(p, **{k: v})]) != d, k
    assert H.pairs_digest([dict(p, odor_x={"ORN_A": 1.0, "ORN_B": 2.0})]) == \
        H.pairs_digest([dict(p, odor_x={"ORN_B": 2.0, "ORN_A": 1.0})])


@pytest.mark.parametrize("bp, bin_", [(59, "lt60"), (60, "60to89"), (89, "60to89"), (90, "ge90")])
def test_power_bins(bp, bin_):
    assert H.power_bin(bp) == bin_


@pytest.mark.parametrize("frac, bin_", [(0.34, "low"), (0.35, "mid"), (0.67, "mid"), (0.68, "high")])
def test_hp_bins(frac, bin_):
    assert H.hp_bin(frac) == bin_
