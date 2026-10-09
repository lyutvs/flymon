"""AD's inputs (AD.1, AD.2): the reused confirmation set and evaluation schedule, the new 12 x 120 learning schedule
(seed 306) with its 80 / 40 prefix digests, the fresh 20 pairs (seed 308) from the 82 candidates left, the layout and
ids. Digests are the plan-time readings."""
import json

import pytest

from flymon.ac import schedules as ac_sched
from flymon.ad import inputs, store
from flymon.ad.spec import SPEC, smoke

DIG = dict(confirm="71a10cbcabc67d551d9db9834c3436a10ce45e4172d44aba2c1b5c829bfec038",
           eval="03e209ea65d38d461ba58befe3b893a9f54d301edd58cf4d5abb6518a91a8dfa",
           learn="4a19075eacf8a74904b5e41d1e0221d3cc7cf3d49006c883ce6aeff9f4374867",
           learn_fallback="6036ac80bae4a781c0db89d1be51cd91156535f79f32d780259ba66a4d3b1bdc",
           learn_futility="c82526d9d18ddf2e12ec4b58a785129212cd41c6dc1d38b267879aaecdc9bb13",
           new="03b861061e4d56ae03d77b2aaff4cd30def7d5bf61b152cd4dccbc9f8d874eed")


@pytest.fixture(scope="module")
def doc():
    return inputs.inputs_doc(SPEC)


def test_run_inputs_are_the_plan_time_readings(doc):
    assert doc["status"] == "OK" and doc["digests"] == DIG
    assert len(doc["learn"]) == 12 * 120 and len(doc["eval"]) == 20
    assert doc["new"]["n_candidates"] == 102 and doc["new"]["n_rest"] == 82 and len(doc["new"]["pairs"]) == 20
    assert not {inputs.pair_key(p) for p in doc["new"]["pairs"]} & {inputs.pair_key(p) for p in doc["confirm"]["pairs"]}
    assert doc["learn_overlap_302"] == 0 and doc["tags"] == ["DL", "DE"]
    assert doc["layout"][0] == ["FLYL", 0, 24] and doc["layout"][23] == ["FLYOS", 11, 47]
    assert doc["seeds"]["learn"] == 306 and doc["seeds"]["os"] == 307 and doc["seeds"]["new"] == 308


def test_schedules_assign_rows_and_global_ids(doc):
    learn, ev = inputs.schedules_for(SPEC, doc, 120)
    by = {s.battle_id: s for s in learn}
    a, b = by["DL-f24-b000"], by["DL-f36-b000"]          # FLY-L 0 and FLY-OS 0 play learning row 0
    assert (a.fly_id, b.fly_id) == (0, 12) and a.my_team == b.my_team and a.opp_team == b.opp_team
    assert len(learn) == 24 * 120 and by["DL-f47-b119"].fly_id == 23
    assert len(ev) == 24 * 20 and {tuple(s.my_team) for s in ev if s.battle_id.endswith("-b005")} == {
        tuple(doc["eval"][5]["my_team"])}
    short, _ = inputs.schedules_for(SPEC, doc, 80)
    assert len(short) == 24 * 80 and max(inputs.battle_index(s.battle_id) for s in short) == 79


def test_ids_accounts_and_names():
    assert inputs.battle_index("DL-f24-b119") == 119 and inputs.battle_index("XE-f27-b001") == 1
    with pytest.raises(ValueError):
        inputs.battle_index("AL-f00-b000")
    assert inputs.account("FLYOS", "E", 47) == "fm-dOE-f47" and inputs.account("FLYL", "L", 24) == "fm-dLL-f24"
    assert len(ac_sched.opponent_name("DL-f47-b119", 3)) == 18


def test_check_inputs_detects_a_changed_document_or_manifest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = smoke()
    doc_p, man_p = inputs.input_paths(s)
    d = inputs.inputs_doc(s)
    store.write_json(doc_p, d)
    man = inputs.input_manifest(d, doc_p)
    store.write_json(man_p, man)
    assert man["status"] == "OK" and inputs.check_inputs(json.loads(doc_p.read_text()), man, s, doc_p) == []
    bad = json.loads(doc_p.read_text())
    bad["learn"][0]["my_team"] = list(reversed(bad["learn"][0]["my_team"]))
    store.write_json(doc_p, bad)
    assert {"learn", "inputs_sha256"} <= set(inputs.check_inputs(bad, man, s, doc_p))
    store.write_json(doc_p, d)
    man2 = dict(man, inputs_sha256=store.sha256_file(doc_p), fixed=dict(man["fixed"], tags=["DL", "XX"]))
    assert inputs.check_inputs(d, man2, s, doc_p) == ["manifest_fixed"]


def test_manifest_status_follows_reuse_problems(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = smoke()
    doc_p, _ = inputs.input_paths(s)
    d = inputs.inputs_doc(s)
    store.write_json(doc_p, d)
    assert inputs.input_manifest(d, doc_p, reuse={"x": 1}, problems=["model_sha256"])["status"] == "STOP_REUSE"
