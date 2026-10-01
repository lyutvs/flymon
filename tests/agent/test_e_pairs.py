from pathlib import Path
import pytest
from flymon.agent import e_pairs as ep, e_codebook as cb, encode_grid as eg
from flymon.agent.e_spec import SPEC, smoke

ROOT = Path(__file__).resolve().parents[2]
needs_npz = pytest.mark.skipif(not (ROOT / "data/malecns.npz").exists(), reason="no connectome")

@pytest.fixture(scope="module")
def pops():
    from flymon.brain.circuits import Populations
    from flymon.brain.connectome import Connectome
    return Populations.from_connectome(Connectome.load(ROOT / "data/malecns.npz"))

@needs_npz
def test_even_situations_reproduce_h4_digest(pops):
    rows = ep.even_situations(pops)
    assert sum(r["axis"] == "a" for r in rows) == 18 and sum(r["axis"] == "b" for r in rows) == 21

@needs_npz
def test_judgement_set_rules(pops):
    s = ep.judgement_set(pops, SPEC)
    assert s["status"] == "OK" and len(s["b"]) == 21
    assert all(SPEC.judge_first_turn <= r["turn"] <= s["last_turn"] <= SPEC.judge_last_turn for r in s["b"] + s["a"])
    used = {ep.egrid_key(r) for r in ep.used_situations(pops)}
    keys = [ep.egrid_key(r) for r in s["b"] + s["a"]]
    assert not (set(keys) & used) and len(set(keys)) == len(keys)
    assert s["n_a"] == len(s["a"])

@needs_npz
def test_overlap_zero_on_glomerulus_sets(pops):
    s = ep.judgement_set(pops, SPEC)
    rc = {str(t): len(v) for t, v in pops.receptor_types.items()}
    gl = [g for g in rc if g not in SPEC.exclude]
    drive = {g: float(i) for i, g in enumerate(gl)}
    from flymon.brain.h4_pairs import pool_vocabulary
    st, _, mon, move = pool_vocabulary()
    book = cb.build(drive, 2, smoke(SPEC))["codebook"]
    C = eg.Codebook(cb.cells(move, mon), book)
    rep = ep.overlap_report(s["b"] + s["a"], ep.used_situations(pops), rc, C, "full")
    assert rep == {"cross": 0, "within": 0}
    for r in ep.attach_odours(s["b"] + s["a"], rc, C, "full"):
        assert not (set(r["odor_x"]) & set(r["odor_y"]))

def test_egrid_key_unordered():
    r1 = dict(move_x="WATER", opp_x=("NORMAL",), move_y="WATER", opp_y=("GROUND", "ROCK"))
    r2 = dict(move_x="WATER", opp_x=("GROUND", "ROCK"), move_y="WATER", opp_y=("NORMAL",))
    assert ep.egrid_key(r1) == ep.egrid_key(r2)
