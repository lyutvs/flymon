# tests/brain/test_t_pairs.py
"""T.2 / T.9.1: the T set generated from the widened Gen-1 opponent pool equals the declared set ((b) 21 last turn 17,
(a) 43 last turn 103, three digests, skip counts cap 9 · collision 0 · used 164 · glom_dup 0 · in_set 48 · pool_only
57, the cluster table, 55 odours); its keys meet no used key (H.4 turns, L turns 0-209 — R's and S's sets included) and
its glomerulus-set pairs no used row's; no key repeats inside it; every row holds an off-POOL opponent type set; no set
odour clashes with the 112 reachable odours (E1) or exceeds the ORN cap; a row whose odour exceeds the cap is dropped;
S's (and R's) set must reproduce first; STOP_SET_SHORT when the turns run out; a mismatch or a smoke spec refuses
judgement_rows / set_odours."""
import dataclasses
import json
from pathlib import Path

import pytest

from flymon.agent import e_pairs, encode_grid
from flymon.agent.e_spec import SPEC as E
from flymon.brain import l_pairs, odor_real, q_pairs
from flymon.brain import t_pairs as TP
from flymon.brain.t_spec import SPEC, smoke

ROOT = Path(__file__).resolve().parents[2]
ENC = json.loads((ROOT / "results/summary/encoder_grid.json").read_text())
NPZ = ROOT / "data/malecns.npz"
needs_npz = pytest.mark.skipif(not NPZ.exists(), reason="no connectome")


@pytest.fixture(scope="module")
def world():
    from flymon.agent.config import load_c3_config
    from flymon.brain.circuits import Populations
    from flymon.brain.connectome import Connectome
    pops = Populations.from_connectome(Connectome.load(str(NPZ)))
    return pops, {str(t): len(v) for t, v in pops.receptor_types.items()}, load_c3_config(SPEC.m0d_summary).params


@pytest.fixture(scope="module")
def js(world):
    return TP.t_set(world[0], ENC, SPEC, world[2])


def _used_rows(pops):
    from flymon.brain import h4_pairs
    st, mi, mon, move = h4_pairs.pool_vocabulary()
    chan = h4_pairs.e0_channels(pops, mon, move)
    rows = list(e_pairs.used_situations(pops, E))
    for t in l_pairs.new_turns(st, mi, 210, E.l_rng_seed):
        alt = l_pairs.alternate_from(t["opp"], t["me"], t["opp_types"], st, t["turn"])
        rows += e_pairs._rows_for_turn(pops, chan, t, alt, st, with_a=True)
    return rows


@needs_npz
def test_the_set_equals_the_declaration(js):
    assert TP.check_t_set(js, SPEC) == []
    assert (js["status"], js["n_b"], js["n_a"], js["last_turn_b"], js["last_turn"]) == ("OK", 21, 43, 17, 103)
    assert js["skipped"] == dict(cap=9, collision=0, used=164, glom_dup=0, in_set=48, pool_only=57)
    assert (js["n_opp"], js["n_combos"], js["n_odours"]) == (127, 1986, 55)
    assert js["digest_e0_b"].startswith("8c9729bf") and js["digest_keys"].startswith("37dde1ca")
    assert set(TP.summary(js)) == {"status", "n_b", "n_a", "last_turn", "last_turn_b", "last_turn_a", "skipped",
                                   "n_opp", "n_combos", "n_odours", "e1_clashes", "cap_fails", "all_off_pool",
                                   "clusters_b", "clusters_a", "digest_e0_b", "digest_e0_a", "digest_keys"}


@needs_npz
def test_keys_and_glomerulus_pairs_meet_nothing_used(world, js):
    pops, rc, _ = world
    cb = q_pairs.codebook(ENC, SPEC)
    rows = js["b"] + js["a"]
    keys = [e_pairs.egrid_key(r) for r in rows]
    assert len(set(keys)) == len(keys)                                     # no duplicate inside the set
    used = _used_rows(pops)
    assert not set(keys) & {e_pairs.egrid_key(r) for r in used}            # H.4 turns, L turns 0-209 (R, S sets)

    def gk(r):
        return frozenset({frozenset(encode_grid.glomeruli(cb, r["move_x"], tuple(r["opp_x"]))),
                          frozenset(encode_grid.glomeruli(cb, r["move_y"], tuple(r["opp_y"])))})
    assert not {gk(r) for r in rows} & {gk(r) for r in used}
    ps = TP.pool_type_sets()
    assert all(not (tuple(r["opp_x"]) in ps and tuple(r["opp_y"]) in ps) for r in rows)
    assert max(r["turn"] for r in js["b"]) == 17 and max(r["turn"] for r in js["a"]) == 103


@needs_npz
def test_e1_and_the_orn_cap_on_the_set_odours(world, js):
    pops, rc, params = world
    cb, rule = q_pairs.codebook(ENC, SPEC), E.dual_rule(SPEC.config)
    assert js["e1_clashes"] == [] and js["cap_fails"] == []
    cap = odor_real.cap_hz(params)
    for r in js["b"] + js["a"]:
        for m, o in ((r["move_x"], tuple(r["opp_x"])), (r["move_y"], tuple(r["opp_y"]))):
            assert all(params.max_rate_hz * SPEC.strength * v <= cap for v in
                       encode_grid.odour(rc, cb, m, o, rule).values())
    od = encode_grid.odour(rc, cb, "ELECTRIC", ("FIRE",), rule)            # T.9.1: the old set's odour over the cap
    assert params.max_rate_hz * SPEC.strength * max(od.values()) > cap
    assert TP._cap_fails(rc, cb, "ELECTRIC", ("FIRE",), rule, SPEC.strength, params.max_rate_hz, cap)


@needs_npz
def test_a_lower_cap_drops_the_rows_that_exceed_it(world, js):
    pops, rc, params = world
    hot = dataclasses.replace(params, max_rate_hz=params.max_rate_hz * 1.25)   # 250 Hz: more odours over the cap
    out = TP.t_set(pops, ENC, SPEC, hot)
    assert out["skipped"]["cap"] > js["skipped"]["cap"] and out["cap_fails"] == []
    assert TP.check_t_set(out, SPEC)


@needs_npz
def test_s_and_r_sets_must_reproduce_first(world, monkeypatch):
    monkeypatch.setattr(TP, "S_SPEC", dataclasses.replace(TP.S_SPEC, digest_keys="0" * 64))
    with pytest.raises(ValueError, match="S's judgement set"):
        TP.t_set(world[0], ENC, SPEC, world[2])


@needs_npz
def test_stop_set_short_when_the_turns_run_out(world):
    short = dataclasses.replace(SPEC, set_last_turn=60)
    out = TP.t_set(world[0], ENC, short, world[2])
    assert out["status"] == TP.STOP_SET_SHORT and out["last_turn"] == 60 and out["n_a"] < 43
    assert TP.check_t_set(out, short)


@needs_npz
def test_judgement_rows_set_odours_and_clusters(world):
    pops, rc, params = world
    rows = TP.judgement_rows(pops, rc, ENC, SPEC, params)
    assert [r["axis"] for r in rows] == ["b"] * 21 + ["a"] * 43
    assert all(set(r) >= {"odor_x", "odor_y", "odor_x_e0", "odor_y_e0"} and r["odor_x_e0"] and r["odor_y_e0"]
               for r in rows)
    od = TP.set_odours(pops, rc, ENC, SPEC, params)
    assert len(od) == 55 and all(isinstance(v, dict) and v for v in od.values())
    cl = TP.clusters(rows)
    assert len(cl) == 64 and cl[f"b|{rows[0]['turn']}|{rows[0]['x']}|{rows[0]['y']}"].count(" 대 ") == 1
    from collections import Counter
    assert Counter(v for k, v in cl.items() if k.startswith("a|")) == {f"{c}": n for c, n in SPEC.clusters_a}
    assert Counter(v for k, v in cl.items() if k.startswith("b|")) == {f"{x} 대 {y}": n for x, y, n in SPEC.clusters_b}
    with pytest.raises(ValueError, match="digest_keys"):
        TP.judgement_rows(pops, rc, ENC, dataclasses.replace(SPEC, digest_keys="0" * 64), params)
    with pytest.raises(ValueError, match="n_a"):
        TP.set_odours(pops, rc, ENC, dataclasses.replace(SPEC, n_a=42), params)
    with pytest.raises(ValueError, match="smoke"):
        TP.judgement_rows(pops, rc, ENC, smoke(SPEC), params)


def test_check_t_set_names_each_mismatch():
    js = dict(status="OK", n_b=21, n_a=43, last_turn=103, last_turn_b=17, last_turn_a=103, n_opp=127, n_combos=1986,
              n_odours=55, digest_e0_b=SPEC.digest_e0_b, digest_e0_a=SPEC.digest_e0_a, digest_keys=SPEC.digest_keys,
              skipped=dict(SPEC.skipped_declared), clusters_b=sorted(list(c) for c in SPEC.clusters_b),
              clusters_a=sorted(list(c) for c in SPEC.clusters_a), e1_clashes=[], cap_fails=[], all_off_pool=True)
    assert TP.check_t_set(js, SPEC) == []
    for k, v in (("last_turn", 102), ("last_turn_b", 16), ("n_a", 44), ("digest_e0_a", "x"),
                 ("skipped", dict(js["skipped"], cap=0)), ("e1_clashes", ["FIRE|GROUND"]), ("all_off_pool", False),
                 ("cap_fails", ["ELECTRIC|FIRE"]), ("n_odours", 54)):
        assert any(k in m for m in TP.check_t_set(dict(js, **{k: v}), SPEC)), k
    assert TP.check_t_set(dict(js, status=TP.STOP_SET_SHORT, n_b=20), SPEC)
