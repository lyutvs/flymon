"""W's pairs (W.2, W.3 3, W.9.6 P2-11 / P3-15): the pilot = V-even L_V-testable (b) pairs + a|4 Rock Slide|Strength in
the even rows' order; the main set from V's generator with V's set as used, turns 306-1985, no axis cap, V's skips,
(b) 167 · (a) 82 (W.0's fact), declared order turn then (b) first, candidate numbers c; block-set checking."""
import json
from pathlib import Path

import pytest

from flymon.brain import w_pairs
from flymon.brain.w_spec import SPEC

ROOT = Path(__file__).resolve().parents[2]
NPZ = ROOT / "data/malecns.npz"


def test_pilot_rows_order_and_refusals():
    even = [dict(axis=a, turn=t, x=x, y="y") for a, t, x in (("b", 0, "s"), ("a", 4, "Rock Slide"), ("b", 2, "q"),
                                                              ("b", 4, "r"))]
    for r in even:
        if r["x"] == "Rock Slide":
            r["y"] = "Strength"
    v_even = [dict(key="b|0|s|y", axis="b", testable=True), dict(key="b|2|q|y", axis="b", testable=False),
              dict(key="b|4|r|y", axis="b", testable=True), dict(key="a|4|Rock Slide|Strength", axis="a",
                                                                testable=True)]
    got = w_pairs.pilot_rows(even, v_even, SPEC)
    assert [(r["axis"], r["turn"]) for r in got] == [("b", 0), ("a", 4), ("b", 4)]
    with pytest.raises(ValueError):
        w_pairs.pilot_rows(even[:1], v_even, SPEC)


def test_check_w_set():
    js = dict(n_b=1, n_a=1, n=2, first_turn=306, last_turn=307, skipped={}, clusters_b=[], clusters_a=[],
              digest_keys="k", digest_e0_b="b", digest_e0_a="a", n_odours=2, all_off_pool=True, keys=["x", "y"])
    blk = dict(w_pairs.set_summary(js), keys=["x", "y"])
    assert w_pairs.check_w_set(js, blk) == []
    assert w_pairs.check_w_set(dict(js, keys=["y", "x"]), blk) == ["W set keys differ from block set"]
    assert w_pairs.check_w_set(dict(js, n_b=2), blk)[0].startswith("W set n_b")


@pytest.fixture(scope="module")
def real():
    if not NPZ.exists() or not (ROOT / SPEC.v_summary).exists():
        pytest.skip("no connectome or V summary")
    from flymon.agent.config import load_c3_config
    from flymon.brain import r_pairs
    from flymon.brain.circuits import Populations
    from flymon.brain.connectome import Connectome
    from flymon.brain.v_runner import kc_values
    from flymon.brain.v_spec import SPEC as V
    pops = Populations.from_connectome(Connectome.load(str(NPZ)))
    enc = json.loads((ROOT / V.encoder_summary).read_text())
    v = json.loads((ROOT / SPEC.v_summary).read_text())
    params = load_c3_config(V.m0d_summary).params
    rc = {str(t): len(x) for t, x in pops.receptor_types.items()}
    return dict(pops=pops, enc=enc, v=v, params=params, rc=rc, kc=kc_values(v),
                even=r_pairs.even_rows(pops, rc, enc, V))


def test_real_pilot_and_main_set(real):
    pil = w_pairs.pilot_rows(real["even"], real["v"]["even"]["pairs"], SPEC)
    assert len(pil) == 16 and sum(r["axis"] == "b" for r in pil) == 15
    js = w_pairs.w_set(real["pops"], real["enc"], real["params"], real["kc"], real["v"]["set"]["set"], SPEC)
    assert (js["n_b"], js["n_a"], js["n"]) == (167, 82, 249) and js["all_off_pool"]
    assert js["first_turn"] == 306 and js["last_turn"] <= 1985
    t = [(r["turn"], r["axis"] != "b") for r in js["rows"]]
    assert t == sorted(t) and min(r["turn"] for r in js["rows"]) >= 306
    blk = dict(w_pairs.set_summary(js), keys=js["keys"])
    rows = w_pairs.main_rows(real["pops"], real["rc"], real["enc"], real["params"], real["kc"],
                             real["v"]["set"]["set"], blk, SPEC)
    assert [r["c"] for r in rows] == list(range(249)) and "odor_x" in rows[0]
    with pytest.raises(ValueError):
        w_pairs.main_rows(real["pops"], real["rc"], real["enc"], real["params"], real["kc"],
                          real["v"]["set"]["set"], dict(blk, digest_keys="x"), SPEC)
