"""AB.3 / AB.7 0 생성기 자가 시험: the Gen-2 opponent rule, the Gen-1 self-test reproducing W's main set and digest,
every AB.3 declared value, the lv_odour sources, the KC-after set and c numbering. Real data (skipped without the
connectome or the summaries); nothing runs the engine."""
import json
from pathlib import Path

import pytest

from flymon.brain import ab_pairs as P
from flymon.brain import h4_pairs
from flymon.brain.ab_spec import SPEC

ROOT = Path(__file__).resolve().parents[2]
NPZ = ROOT / "data/malecns.npz"
SUMS = [ROOT / f"results/summary/{n}.json" for n in ("v_lever", "w_learning", "y_learning", "aa_learning")]


@pytest.fixture(scope="module")
def real():
    if not NPZ.exists() or not all(p.exists() for p in SUMS):
        pytest.skip("no connectome or summaries")
    from flymon.agent.config import load_c3_config
    from flymon.brain import r_pairs
    from flymon.brain.circuits import Populations
    from flymon.brain.connectome import Connectome
    from flymon.brain.v_runner import kc_values
    from flymon.brain.v_spec import SPEC as V
    pops = Populations.from_connectome(Connectome.load(str(NPZ)))
    enc = json.loads((ROOT / V.encoder_summary).read_text())
    docs = {p.stem: json.loads(p.read_text()) for p in SUMS}
    params = load_c3_config(V.m0d_summary).params
    kc = kc_values(docs["v_lever"])
    base, v_rows, js_w = P.base_used_rows(pops, enc, params, kc, docs["v_lever"]["set"]["set"])
    rc = {str(t): len(x) for t, x in pops.receptor_types.items()}
    even = r_pairs.even_rows(pops, rc, enc, V)
    lv = P.lv_sources(docs["aa_learning"], docs["w_learning"], docs["y_learning"], js_w["rows"], even, v_rows)
    return dict(pops=pops, enc=enc, kc=kc, base=base, v_rows=v_rows, w_rows=js_w["rows"], lv=lv)


def test_gen2_opponents_rule():
    _st, _mi, mon, _mv = h4_pairs.pool_vocabulary()
    assert tuple(sorted(mon)) == SPEC.pool_types
    opp, n_out = P.gen2_opponents(mon)
    names = [n for n, _ in opp]
    assert (len(opp), n_out) == (SPEC.n_opp, SPEC.n_opp_type_out)
    assert names.count("Unown") == 1 and "Entei" in names and "Celebi" in names and names[0] == "Chikorita"


def test_selftest_reproduces_w_main_set(real):
    got = P.selftest(real["base"], real["v_rows"], real["kc"])
    assert got["ok"] and (got["n_b"], got["n_a"], got["last_turn"]) == (167, 82, 1967)
    assert got["digest_keys"] == SPEC.w_digest_keys


def test_lv_sources(real):
    lv = real["lv"]
    assert lv["reasons"] == [] and lv["n"] == 108
    assert lv["per_source"] == dict(aa31=45, learned=71, v_set=66)
    assert lv["sha256"] == "d8af2a630292e92175f5b6bf120792cc0d63e174240baba6cc2fd9e9d6ed1364"   # plan-time value


def test_generate_matches_every_declared_value(real):
    gen = P.generate(real["base"], real["v_rows"], real["w_rows"], real["lv"]["odours"], real["kc"])
    assert P.declared_reasons(gen) == []
    assert gen["skipped_inorder"]["lv_odour"] == 1276 and gen["skipped_inorder"]["in_set"] == 845   # Reading 2
    assert len(gen["rows"]) == 209 and gen["digest_keys"] == P.digest_keys(gen["rows"])
    again = P.generate(real["base"], real["v_rows"], real["w_rows"], real["lv"]["odours"], real["kc"])
    assert again["keys"] == gen["keys"]


def test_ab_set_and_c(real):
    gen = P.generate(real["base"], real["v_rows"], real["w_rows"], real["lv"]["odours"], real["kc"])
    # a synthetic KC table: V's values, every new odour in band → the V-cache upper bound 184
    new = {e: {o: 0.05 for o in gen["new_odours"]} for e in ("none", "lever")}
    rows = P.ab_set(gen["rows"], P.kc_merge(real["kc"], new))
    assert len(rows) == 184 and [r["c"] for r in rows] == list(range(184))
    t = [(r["turn"], r["axis"] != "b") for r in rows]
    assert t == sorted(t)
    rec = P.set_record(rows, P.kc_merge(real["kc"], new))
    assert rec["by_axis"] == {"a": 139, "b": 45}
