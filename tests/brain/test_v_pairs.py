"""V's set generator (V.2, V.9.1, V.9.5 P2-10 / P3-12) on the real connectome and encoder summary, list only: T's set
reproduces first and joins used; the candidate odours are the full permutation's rows past the filling-independent
skips (173 odours, 53 of them U's 55); kc_input is the last skip and reads both engines; (b) 21 · (a) 43 from turn 0;
no V key or glomerulus key repeats a used one; STOP_SET_SHORT when the filter empties the pool; a missing value
refuses; the regenerated set must equal block set; smoke never reaches the set; the filter record."""
import json
from pathlib import Path

import pytest

from flymon.agent import e_pairs
from flymon.brain import t_pairs, v_pairs
from flymon.brain.r_pairs import okey
from flymon.brain.t_spec import SPEC as T
from flymon.brain.v_spec import SPEC, smoke

ROOT = Path(__file__).resolve().parents[2]
NPZ = ROOT / "data/malecns.npz"
pytestmark = pytest.mark.skipif(not NPZ.exists(), reason="no connectome")


@pytest.fixture(scope="module")
def real():
    from flymon.agent.config import load_c3_config
    from flymon.brain.circuits import Populations
    from flymon.brain.connectome import Connectome
    cfg = load_c3_config(str(ROOT / SPEC.m0d_summary))
    enc = json.loads((ROOT / SPEC.encoder_summary).read_text())
    pops = Populations.from_connectome(Connectome.load(str(NPZ)))
    cand = v_pairs.candidate_odours(pops, enc, cfg.params)
    return dict(pops=pops, enc=enc, params=cfg.params, cand=cand,
                rc={str(t): len(v) for t, v in pops.receptor_types.items()})


def _kc(real, none=None, lever=None):
    ids = list(real["cand"]["odours"])
    return {"none": {o: (none or {}).get(o, 0.05) for o in ids}, "lever": {o: (lever or {}).get(o, 0.05) for o in ids}}


def test_candidates_are_the_filling_independent_rows(real):
    c = real["cand"]
    assert len(c["odours"]) == 173 and c["n_turns"] == 1986 and c["n_rows"] == 3324
    assert c["cap_hz"] == pytest.approx(1000 / 3) and all(v <= c["cap_hz"] for v in c["cap"].values())
    u55 = json.loads((ROOT / SPEC.u_summary).read_text())["kc"]["points"]["0.6"]["record_set"]["per_odour_none"]
    assert len(set(c["odours"]) & set(u55)) == 53


def test_set_from_turn_0_with_ts_set_used(real):
    js = v_pairs.v_set(real["pops"], real["enc"], real["params"], _kc(real), SPEC)
    assert js["status"] == "OK" and (js["n_b"], js["n_a"]) == (21, 43)
    assert min(r["turn"] for r in js["b"] + js["a"]) < T.last_turn         # walked from turn 0, not after T's turns
    assert js["skipped"]["collision"] == js["skipped"]["e1"] == js["skipped"]["glom_dup"] == 0
    assert js["skipped"]["kc_input"] == 0 and js["e1_clashes"] == [] and js["cap_fails"] == [] and js["all_off_pool"]
    t = t_pairs.t_set(real["pops"], real["enc"], T, real["params"])
    t_keys = {e_pairs.egrid_key(r) for r in t["b"] + t["a"]}
    assert not t_keys & {e_pairs.egrid_key(r) for r in js["b"] + js["a"]}
    assert len({e_pairs.egrid_key(r) for r in js["b"] + js["a"]}) == 64
    assert set(js["odour_ids"]) <= set(real["cand"]["odours"])


def test_kc_input_is_the_last_skip_and_reads_both_engines(real):
    base = v_pairs.v_set(real["pops"], real["enc"], real["params"], _kc(real), SPEC)
    target = base["odour_ids"][0]
    for side in ("none", "lever"):
        bad = {target: 0.0299} if side == "none" else {target: 0.1501}
        js = v_pairs.v_set(real["pops"], real["enc"], real["params"],
                           _kc(real, **{side: bad}), SPEC)
        assert target not in js["odour_ids"] and js["skipped"]["kc_input"] > 0
    edge = v_pairs.v_set(real["pops"], real["enc"], real["params"], _kc(real, none={target: 0.03}, lever={target: 0.15}),
                         SPEC)
    assert target in edge["odour_ids"]                     # [0.03, 0.15] is closed


def test_set_short_when_the_filter_empties_the_pool(real):
    js = v_pairs.v_set(real["pops"], real["enc"], real["params"],
                       _kc(real, none={o: 0.01 for o in real["cand"]["odours"]}), SPEC)
    assert js["status"] == "STOP_SET_SHORT" and (js["n_b"], js["n_a"]) == (0, 0) and js["last_turn"] == 1985


def test_a_missing_value_refuses(real):
    kc = _kc(real)
    del kc["lever"][next(iter(kc["lever"]))]
    with pytest.raises(ValueError, match="no lever value"):
        v_pairs.v_set(real["pops"], real["enc"], real["params"], kc, SPEC)


def test_regenerated_set_must_equal_block_set_and_smoke_never_reaches_it(real):
    kc = _kc(real)
    js = v_pairs.v_set(real["pops"], real["enc"], real["params"], kc, SPEC)
    block = json.loads(json.dumps(v_pairs.set_summary(js)))
    assert v_pairs.check_v_set(js, block) == []
    rows = v_pairs.judgement_rows(real["pops"], real["rc"], real["enc"], real["params"], kc, block, SPEC)
    assert len(rows) == 64 and all("odor_x" in r and "odor_x_e0" in r for r in rows)
    od = v_pairs.set_odours(real["pops"], real["rc"], real["enc"], real["params"], kc, block, SPEC)
    assert sorted(od) == sorted(js["odour_ids"])
    e0 = v_pairs.set_e0_odours(real["pops"], real["enc"], real["params"], kc, block, SPEC)
    assert 0 < len(e0) <= 128
    tampered = dict(block, digest_keys="0" * 64)
    with pytest.raises(ValueError, match="digest_keys"):
        v_pairs.judgement_rows(real["pops"], real["rc"], real["enc"], real["params"], kc, tampered, SPEC)
    with pytest.raises(ValueError, match="smoke"):
        v_pairs.judgement_rows(real["pops"], real["rc"], real["enc"], real["params"], kc, block, smoke(SPEC))


def test_kc_record_counts_dropped_odours(real):
    u55 = json.loads((ROOT / SPEC.u_summary).read_text())["kc"]["points"]["0.6"]["record_set"]["per_odour_none"]
    kc = _kc(real, none={o: v for o, v in u55.items()})
    rec = v_pairs.kc_record(real["pops"], real["enc"], real["params"], kc, SPEC)
    low = sorted(o for o, v in u55.items() if v < 0.03 and o in real["cand"]["odours"])
    assert rec["outside"]["none"] == low and rec["outside"]["lever"] == [] and rec["dropped"] == low
    assert rec["set_rows"]["R"]["n"] == 53 and rec["set_rows"]["S"]["n"] == 64 and rec["set_rows"]["T"]["n"] == 64
    assert rec["candidate_rows_dropped"] > 0


def test_cluster_labels_are_ts():
    rows = [dict(axis="b", turn=0, x="m1", y="m2", opp_x=["FIRE"], opp_y=["WATER"]),
            dict(axis="a", turn=1, x="m1", y="m2", opp_x=["GROUND"], opp_y=["GROUND"])]
    assert v_pairs.cluster_labels(rows) == t_pairs.clusters(rows)
    assert okey("FIRE", ("GRASS",)) == "FIRE|GRASS"
