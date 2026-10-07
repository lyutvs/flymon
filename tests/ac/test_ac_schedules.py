"""AC's schedules (AC.4, 4.4) and the experiment-input manifest (AC.1 (ii)).
Controller ruling (AC.4): ONE canonical evaluation schedule (1 x eval_battles, seed 303) that every fly of every arm
plays; the learning schedule keeps per-fly rows (12 x 40, seed 302)."""
import json

import pytest

from flymon.ac import schedules as sc
from flymon.ac.spec import SPEC, bench, smoke
from flymon.rescope import blocks


def test_canonical_eval_is_one_schedule_and_disjoint():
    ev = sc.eval_canonical(SPEC)
    assert len(ev) == 20 and all(s.fly_id == 0 for s in ev) and ev == sc.canonical(1, 20, 303)
    learn = sc.learn_canonical(SPEC)
    assert len(learn) == 12 * 40 and sorted({s.fly_id for s in learn}) == list(range(12))
    blocks.assert_disjoint(learn, ev)


def test_every_fly_of_every_arm_plays_the_one_eval_schedule():
    ev = sc.eval_canonical(SPEC)
    flies = range(sum(SPEC.arm_sizes().values()))
    got = sc.assign(ev, [(g, 0) for g in flies], "AE")
    assert len(got) == len(flies) * 20 and len({s.battle_id for s in got}) == len(got)
    for g in flies:
        mine = [s for s in got if s.fly_id == g]
        assert [(s.my_team, s.opp_team) for s in mine] == [(s.my_team, s.opp_team) for s in ev]
        assert [s.battle_id for s in mine] == [f"AE-f{g:02d}-b{b:03d}" for b in range(20)]
    blocks.assert_disjoint(sc.assign(sc.learn_canonical(SPEC), [(g, g) for g in range(12)], "AL"), got)


def test_smoke_and_bench_eval_schedules():
    assert len(sc.eval_canonical(smoke())) == 2 and all(s.fly_id == 0 for s in sc.eval_canonical(smoke()))
    assert len(sc.learn_canonical(smoke())) == 2 * 2
    assert sc.eval_canonical(bench()) == [] and len(sc.learn_canonical(bench())) == 12 * 2
    assert sc.assign(sc.eval_canonical(bench()), [(0, 0)], "BE") == []
    with pytest.raises(KeyError):
        sc.assign(sc.eval_canonical(SPEC), [(0, 1)], "AE")


def test_assign_maps_local_rows_to_global_ids():
    canon = sc.canonical(2, 3, 302)
    got = sc.assign(canon, [(13, 1), (0, 0)], "AL")
    assert [s.battle_id for s in got[:3]] == ["AL-f13-b000", "AL-f13-b001", "AL-f13-b002"]
    assert all(s.fly_id == 13 for s in got[:3]) and all(s.fly_id == 0 for s in got[3:])
    assert [s.my_team for s in got[:3]] == [s.my_team for s in canon if s.fly_id == 1]
    assert [s.opp_team for s in got[3:]] == [s.opp_team for s in canon if s.fly_id == 0]


def test_battle_index_and_names():
    assert sc.battle_index("AL-f23-b039") == 39 and sc.battle_index("SE-f01-b001") == 1
    with pytest.raises(ValueError):
        sc.battle_index("PL-f00-b000")
    assert sc.account("COFF", "E", 23) == "fm-aCOE-f23" and sc.account("FLY", "L", 0) == "fm-aFLL-f00"
    assert sc.opponent_name("AL-f23-b039", 3) == "fm-h-AL-f23-b039-3" and len(sc.opponent_name("AL-f23-b039", 3)) == 18
    sc.check_names(sc.assign(sc.learn_canonical(SPEC), [(23, 11)], "AL"), SPEC.retry_max)
    sc.check_names(sc.assign(sc.eval_canonical(SPEC), [(39, 0)], "AE"), SPEC.retry_max)


def test_inputs_doc_is_deterministic_and_carries_digests():
    d = sc.inputs_doc(SPEC)
    assert d["status"] == "OK" and d["mode"] == "run" and d["seeds"] == SPEC.seeds() and d["n_pairs"] == 20
    assert d["digests"]["learn"] == blocks.schedule_digest(sc.learn_canonical(SPEC))
    assert d["digests"]["eval"] == blocks.schedule_digest(sc.canonical(1, 20, 303))
    assert len(d["eval"]) == 20 and len(d["learn"]) == 12 * 40
    assert d["digests"]["confirm"] == d["confirm"]["digest"]
    assert d["label"] == sc.LABEL
    assert sc.inputs_doc(SPEC) == d


def test_inputs_doc_stop_set_carries_label(monkeypatch):
    monkeypatch.setattr(sc.confirm, "candidate_pairs", lambda: dict(pairs=[], counts={}))
    d = sc.inputs_doc(smoke())
    assert d["status"] == "STOP_SET" and d["label"] == sc.LABEL and d["digests"]["confirm"] is None


def test_input_paths_per_mode():
    assert [str(p) for p in sc.input_paths(SPEC)] == ["results/summary/ac_inputs.json",
                                                      "results/summary/ac_inputs_manifest.json"]
    assert [str(p) for p in sc.input_paths(smoke())] == ["results/m4-smoke/inputs.json",
                                                         "results/m4-smoke/inputs_manifest.json"]
    assert str(sc.input_paths(bench())[0]) == "results/m4-bench/inputs.json"


def _write(tmp_path, spec):
    doc = sc.inputs_doc(spec)
    p = tmp_path / "inputs.json"
    p.write_text(json.dumps(doc, sort_keys=True, indent=1))
    return doc, p, sc.input_manifest(doc, p)


def test_check_inputs_clean(tmp_path):
    doc, p, man = _write(tmp_path, smoke())
    assert man["label"] == sc.LABEL
    assert sc.check_inputs(json.loads(p.read_text()), man, smoke(), p) == []


def test_check_inputs_detects_a_changed_team(tmp_path):
    doc, p, man = _write(tmp_path, smoke())
    bad = json.loads(p.read_text())
    bad["eval"][0]["opp_team"] = list(reversed(bad["eval"][0]["opp_team"]))
    p.write_text(json.dumps(bad, sort_keys=True, indent=1))
    got = sc.check_inputs(bad, man, smoke(), p)
    assert any("eval" in x for x in got) and "inputs_sha256" in got


def test_check_inputs_detects_another_spec(tmp_path):
    doc, p, man = _write(tmp_path, smoke())
    assert sc.check_inputs(json.loads(p.read_text()), man, bench(), p)
