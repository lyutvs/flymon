# tests/brain/test_s_pairs.py
"""S.2: the S set generated from L turns 104-209 equals the declared set (T 177, (b) 21, (a) 43, three digests); its
keys meet none of the even / odd H.4 turns, L turns 0-63 or R's set; no key repeats inside it; every turn is in
104..T; R's set is digest-checked before its keys are used; STOP_SET_SHORT when the turns run out; a mismatch or a
smoke spec refuses judgement_rows."""
import dataclasses
import json
from pathlib import Path

import pytest

from flymon.agent import e_pairs
from flymon.agent.e_spec import SPEC as E
from flymon.brain import s_pairs as SP
from flymon.brain.s_spec import SPEC, smoke

ROOT = Path(__file__).resolve().parents[2]
ENC = json.loads((ROOT / "results/summary/encoder_grid.json").read_text())
NPZ = ROOT / "data/malecns.npz"
needs_npz = pytest.mark.skipif(not NPZ.exists(), reason="no connectome")


@pytest.fixture(scope="module")
def world():
    from flymon.brain.circuits import Populations
    from flymon.brain.connectome import Connectome
    pops = Populations.from_connectome(Connectome.load(str(NPZ)))
    return pops, {str(t): len(v) for t, v in pops.receptor_types.items()}


@pytest.fixture(scope="module")
def js(world):
    return SP.s_set(world[0], ENC, SPEC)


@needs_npz
def test_the_set_equals_the_declaration(js):
    assert SP.check_s_set(js, SPEC) == []
    assert (js["status"], js["n_b"], js["n_a"], js["last_turn"]) == ("OK", 21, 43, 177)
    assert js["digest_e0_b"].startswith("2b92f40a") and js["digest_keys"].startswith("884d49cc")
    assert set(SP.summary(js)) == {"status", "n_b", "n_a", "last_turn", "skipped", "digest_e0_b", "digest_e0_a",
                                   "digest_keys"}


@needs_npz
def test_keys_meet_no_used_situation_and_no_r_key(world, js):
    pops, _ = world
    keys = [e_pairs.egrid_key(r) for r in js["b"] + js["a"]]
    assert len(set(keys)) == len(keys)                                     # no duplicate inside the set
    used = {e_pairs.egrid_key(r) for r in e_pairs.used_situations(pops, E)}  # H.4 even + odd turns, L turns 0-63
    assert not set(keys) & used
    assert not set(keys) & SP.r_set_keys(pops, ENC)                        # R's set (turns 64-103)
    turns = [r["turn"] for r in js["b"] + js["a"]]
    assert SPEC.first_turn <= min(turns) and max(turns) <= js["last_turn"] <= SPEC.set_last_turn
    assert max(r["turn"] for r in js["b"]) == js["last_turn"]
    assert js["skipped"]["r_set"] > 0 and js["skipped"]["used"] > 0


@needs_npz
def test_r_keys_need_rs_digests(world, monkeypatch):
    pops, _ = world
    bad = json.loads(json.dumps(ENC))
    bad["set"]["digest_keys"] = "0" * 64
    with pytest.raises(ValueError, match="R's judgement set"):
        SP.r_set_keys(pops, bad)


@needs_npz
def test_stop_set_short_when_the_turns_run_out(world):
    short = dataclasses.replace(SPEC, set_last_turn=150)
    out = SP.s_set(world[0], ENC, short)
    assert out["status"] == SP.STOP_SET_SHORT and out["n_b"] < 21 and out["last_turn"] == 150
    assert SP.check_s_set(out, short)


@needs_npz
def test_judgement_rows_check_the_declaration_and_refuse_smoke(world):
    pops, rc = world
    rows = SP.judgement_rows(pops, rc, ENC, SPEC)
    assert [r["axis"] for r in rows] == ["b"] * 21 + ["a"] * 43
    assert all(set(r) >= {"odor_x", "odor_y", "odor_x_e0", "odor_y_e0"} for r in rows)
    with pytest.raises(ValueError, match="digest_keys"):
        SP.judgement_rows(pops, rc, ENC, dataclasses.replace(SPEC, digest_keys="0" * 64))
    with pytest.raises(ValueError, match="n_a"):
        SP.judgement_rows(pops, rc, ENC, dataclasses.replace(SPEC, n_a=42))
    with pytest.raises(ValueError, match="smoke"):
        SP.judgement_rows(pops, rc, ENC, smoke(SPEC))


def test_check_s_set_names_each_mismatch():
    js = dict(status="OK", n_b=21, n_a=43, last_turn=177, digest_e0_b=SPEC.digest_e0_b, digest_e0_a=SPEC.digest_e0_a,
              digest_keys=SPEC.digest_keys)
    assert SP.check_s_set(js, SPEC) == []
    for k, v in (("last_turn", 176), ("n_a", 44), ("digest_e0_a", "x")):
        assert any(k in m for m in SP.check_s_set(dict(js, **{k: v}), SPEC)), k
    assert SP.check_s_set(dict(js, status=SP.STOP_SET_SHORT, n_b=20), SPEC)
