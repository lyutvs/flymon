"""R.3 / R.9.6 / encoder 4.3, 5.1: the 112 calibration odours are encoder ③'s; the even rows are H.4's 8 even turns
((a) 18, (b) 21 — the (b) rows equal Q's); the judgement rows are the encoder's set with the three digests checked
against both R's declared values and block set, refused for a smoke spec; the encoder activity reference is the
encoder's own cache name; P's reference entries are block p's (kind and measure key)."""
import copy
import json
from pathlib import Path

import pytest

from flymon.agent.config import load_c3_config
from flymon.brain import odor_real, q_pairs
from flymon.brain import r_pairs as RP
from flymon.brain.q_spec import SPEC as Q
from flymon.brain.r_spec import SPEC, smoke

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


@needs_npz
def test_calibration_odours_are_encoder_3s_112(world):
    _, rc = world
    odours, single, dual = RP.calibration_odours(rc, q_pairs.codebook(ENC, SPEC), "norm")
    per = ENC["strength"]["configs"]["k2-norm"]["table"]["1.0"]["per_odour"]
    assert set(odours) == set(per) and len(odours) == SPEC.n_calib
    assert (len(single), len(dual)) == (32, 80)
    assert SPEC.kc_repro_odours[0] in single and set(SPEC.kc_repro_odours[1:]) <= set(dual)


@needs_npz
def test_cap_holds_at_s_1_and_not_at_1_4(world):
    _, rc = world
    cfg = load_c3_config(str(ROOT / SPEC.m0d_summary))
    cb = q_pairs.codebook(ENC, SPEC)
    args = (cfg.params.max_rate_hz, odor_real.cap_hz(cfg.params))
    assert RP.cap_at(rc, cb, "norm", 1.0, *args) is True
    assert RP.cap_at(rc, cb, "norm", 1.4, *args) is False             # encoder ③: ORN_CAP at 1.4


@needs_npz
def test_even_rows_are_h4s_and_their_b_rows_are_qs(world):
    pops, rc = world
    from flymon.agent import e_pairs
    rows = RP.even_rows(pops, rc, ENC, SPEC)
    assert [r["axis"] for r in rows].count("b") == 21 and [r["axis"] for r in rows].count("a") == 18
    assert all(0 <= r["turn"] < 16 and r["turn"] % 2 == 0 for r in rows)
    qb = q_pairs.b_rows(e_pairs.even_situations(pops), rc, ENC, Q)
    mine = [r for r in rows if r["axis"] == "b"]
    assert [RP.row_key(r) for r in mine] == [q_pairs.key_str(r) for r in qb]
    assert all(a["odor_x"] == b["odor_x"] and a["odor_y"] == b["odor_y"] for a, b in zip(mine, qb))


@needs_npz
def test_judgement_rows_check_the_digests_and_refuse_smoke(world):
    pops, rc = world
    rows = RP.judgement_rows(pops, rc, ENC, SPEC)
    assert sum(r["axis"] == "b" for r in rows) == 21 and sum(r["axis"] == "a" for r in rows) == 32
    assert all(64 <= r["turn"] <= 103 for r in rows)
    assert all("odor_x_e0" in r and "odor_x" in r for r in rows)
    bad = copy.deepcopy(ENC)
    bad["set"]["digest_keys"] = "0" * 64
    with pytest.raises(ValueError, match="digest_keys"):
        RP.judgement_rows(pops, rc, bad, SPEC)
    with pytest.raises(ValueError, match="smoke"):
        RP.judgement_rows(pops, rc, ENC, smoke(SPEC))


def _js(**kw):
    js = dict(digest_e0_b=SPEC.digest_e0_b, digest_e0_a=SPEC.digest_e0_a, digest_keys=SPEC.digest_keys, n_a=32,
              last_turn=103, status="OK", b=[0] * 21, a=[0] * 32)
    return dict(js, **kw)


def test_check_set_compares_generated_block_and_declared():
    assert RP.check_set(_js(), ENC, SPEC) == []
    assert any("digest_e0_b" in m for m in RP.check_set(_js(digest_e0_b="x"), ENC, SPEC))
    enc = copy.deepcopy(ENC)
    enc["set"]["n_a"] = 31
    assert any("n_a" in m for m in RP.check_set(_js(), enc, SPEC))
    assert RP.check_set(_js(b=[0] * 20), ENC, SPEC)
    assert RP.check_set(_js(status="STOP_SET_SHORT"), ENC, SPEC)


def test_cond_odours():
    row = dict(odor_x={"A": 1.0}, odor_y={"B": 1.0}, odor_x_e0={"C": 1.0}, odor_y_e0={"D": 1.0})
    L, C, E0 = SPEC.conditions()
    assert RP.cond_odours(row, L) == ({"A": 1.0}, {"B": 1.0}) == RP.cond_odours(row, C)
    assert RP.cond_odours(row, E0) == ({"C": 1.0}, {"D": 1.0})


def test_encoder_activity_ref_is_the_encoder_cache_name():
    from flymon.agent.e_measure import EMeasurer
    from flymon.agent.e_spec import SPEC as E
    from flymon.agent.e_store import ECache
    cfg = load_c3_config(str(ROOT / SPEC.m0d_summary))
    code = {"key": "k", "files": {}, "versions": {}}
    a = RP.encoder_activity_ref("X|Y", {"ORN_A": 1.0}, cfg.params, 100, code, SPEC)
    b = RP.encoder_activity_ref("X|Z", {"ORN_A": 1.0}, cfg.params, 100, code, SPEC)
    assert a != b and a.endswith(".json") and len(a) == 24 + len(".json")
    m = EMeasurer(None, ECache(f"{E.raw_dir}/cache", code), cfg.params, 100, E, guard_params=False)
    assert a == m.cache._path("activity", m._act_inputs("X|Y", {"ORN_A": 1.0}, 1.0, E.strength_seeds)).name


def test_p_reference_reads_only_block_ps_entries(tmp_path):
    d = tmp_path / "p_arm"
    d.mkdir()

    def put(name, kind, key, direction, arm, seed, res):
        (d / name).write_text(json.dumps(dict(kind=kind, code_key=key, inputs=dict(direction=direction, arm=arm,
                                                                                   seed=seed), result=res)))
    put("a.json", "p_arm", "K", "r1", "punish", 23_000_000, {"v": 1})
    put("b.json", "p_arm", "OTHER", "r1", "plastic", 23_000_000, {"v": 2})
    put("c.json", "o2_arm", "K", "r2", "punish", 23_000_000, {"v": 3})
    (d / "d.json").write_text("{broken")
    got = RP.p_reference(d, "K", {("r1", "punish", 23_000_000), ("r1", "plastic", 23_000_000)})
    assert got == {("r1", "punish", 23_000_000): {"v": 1}}
