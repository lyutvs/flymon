"""scripts/run_ac_m4.py without Showdown: arguments, refusals before any pool or server, session cap, names."""
import importlib.util
import json
from pathlib import Path

import pytest

from flymon.ac import budget, schedules, store
from flymon.ac.battles import ACBattles
from flymon.ac.spec import LABEL, SPEC, smoke

ROOT = Path(__file__).resolve().parents[2]


def load():
    spec = importlib.util.spec_from_file_location("ra", ROOT / "scripts/run_ac_m4.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def write_inputs(spec):
    doc_p, man_p = schedules.input_paths(spec)
    doc = schedules.inputs_doc(spec)
    store.write_json(doc_p, doc)
    store.write_json(man_p, schedules.input_manifest(doc, doc_p))
    return doc_p


def test_parse_args():
    ra = load()
    a = ra.parse_args(["--arm", "BRAIN", "--smoke"])
    assert a.out == "results/m4-smoke/BRAIN" and a.workers == 4 and ra.spec_for(a).mode == "smoke"
    a = ra.parse_args(["--arm", "RND"])
    assert a.out == "results/m4/stage1/RND" and ra.spec_for(a) == SPEC
    a = ra.parse_args(["--arm", "BRAIN", "--bench"])
    assert a.out == "results/m4-bench/BRAIN" and a.workers == 16 and ra.spec_for(a).mode == "bench"
    with pytest.raises(SystemExit):
        ra.parse_args(["--arm", "RND", "--bench"])
    with pytest.raises(SystemExit):
        ra.parse_args(["--arm", "BRAIN", "--bench", "--resume"])


def test_refuses_missing_or_changed_inputs(tmp_path, monkeypatch):
    ra = load()
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="does not exist"):
        ra.main(["--arm", "RND", "--smoke"])
    doc_p = write_inputs(smoke())
    d = json.loads(doc_p.read_text())
    d["eval"][0]["my_team"] = list(reversed(d["eval"][0]["my_team"]))
    doc_p.write_text(json.dumps(d))
    with pytest.raises(SystemExit, match="does not match"):
        ra.main(["--arm", "RND", "--smoke"])


def test_brain_refuses_without_a_frozen_model_manifest(tmp_path, monkeypatch):
    ra = load()
    monkeypatch.chdir(tmp_path)
    write_inputs(smoke())
    with pytest.raises(SystemExit, match="ac_model_manifest"):
        ra.main(["--arm", "BRAIN", "--smoke"])
    store.write_json("results/summary/ac_model_manifest.json", {"status": "STOP_NO_RECOVERY"})
    with pytest.raises(SystemExit, match="not FROZEN"):
        ra.main(["--arm", "BRAIN", "--smoke"])


def test_run_mode_refuses_without_an_ok_budget(tmp_path, monkeypatch):
    ra = load()
    monkeypatch.chdir(tmp_path)
    write_inputs(SPEC)
    with pytest.raises(SystemExit, match="ac_budget"):
        ra.main(["--arm", "MAX"])
    store.write_json(budget.BUDGET, {"status": "STOP_BUDGET", "hours": 70.0})
    with pytest.raises(SystemExit, match="STOP_BUDGET"):
        ra.main(["--arm", "MAX"])


def test_session_cap(tmp_path, monkeypatch):
    ra = load()
    monkeypatch.chdir(tmp_path)
    a = ra.parse_args(["--arm", "RND"])
    assert ra.session_cap(SPEC, a) == 24 * 3600
    a = ra.parse_args(["--arm", "RND", "--session-hours", "2"])
    assert ra.session_cap(SPEC, a) == 2 * 3600
    p = Path("results/m4/stage1/BRAIN/wall_clock.json")
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"sessions": [{"seconds": 61 * 3600.0}]}))
    with pytest.raises(SystemExit, match="budget is spent"):
        ra.session_cap(SPEC, ra.parse_args(["--arm", "RND"]))
    assert ra.session_cap(smoke(), ra.parse_args(["--arm", "RND", "--smoke"])) == 24 * 3600


def test_result_exists_needs_resume(tmp_path, monkeypatch):
    ra = load()
    monkeypatch.chdir(tmp_path)
    write_inputs(smoke())
    store.write_json("results/m4-smoke/RND/result.json", {"complete": True})
    with pytest.raises(SystemExit, match="pass --resume"):
        ra.main(["--arm", "RND", "--smoke"])


def test_players_are_not_built_until_needed():
    b = ACBattles({0: "RND"}, Path("results/m4-smoke/RND"), None)
    assert b.players == {"L": {}, "E": {}} and b.bars == {}
    b.reset_player("E", 0)                             # no player yet: nothing to reset


def test_failed_battle_marks_the_fly_and_skips_its_later_battles(tmp_path):
    """Retries exhausted -> Stage1Hooks.after_battle (the script's mark_invalid) -> InvalidBook.mark -> skipped."""
    from flymon.ac import stage1
    from flymon.ac.invalid import InvalidBook
    from flymon.rescope import blocks
    from tests.ac.ac_fakes import run_fake
    ra = load()
    s = smoke()
    doc = {"learn": blocks.schedule_rows(schedules.learn_canonical(s)),
           "eval": blocks.schedule_rows(schedules.eval_canonical(s))}
    lay = [("RND", k) for k in range(3)]
    ev = schedules.assign(blocks.schedule_from_rows(doc["eval"]), [(k, 0) for k in range(3)], s.tags[1])
    book = InvalidBook(tmp_path / "invalid.json", {k: "RND" for k in range(3)}, {"RND": 3}, {"RND": 1})
    first = [b.battle_id for b in ev if b.fly_id == 1][0]
    fail = {(first, i) for i in range(s.retry_max + 1)}
    run, _ = run_fake(tmp_path, learn=None, eval_=ev, n=3, fail=fail, after_battle=ra.mark_invalid(s, lay, book),
                      wrap=lambda af: stage1.skipping(af, book))
    assert run["complete"]
    assert book.invalid == {1: f"{first}: unfinished after {s.retry_max} retries"} and book.stopped == {}
    saved = json.loads((tmp_path / "invalid.json").read_text())
    assert saved["invalid"] == {"1": book.invalid[1]} and saved["label"] == LABEL
    recs = {r["battle_id"]: r for r in blocks.read_jsonl(tmp_path / "logs/eval/battles.jsonl")}
    later = [b.battle_id for b in ev if b.fly_id == 1][1:]
    assert later and all(recs[b]["invalid"] and not recs[b]["finished"] for b in later)
    played = [r["battle_id"] for p in (tmp_path / "logs/eval").rglob("fly01.jsonl") for r in blocks.read_jsonl(p)]
    assert first in played and not set(later) & set(played)       # the later battles never reached a player
    assert not any(r["invalid"] for b, r in recs.items() if not b.startswith(f"{s.tags[1]}-f01-"))
    rows = stage1.nobrain_rows(tmp_path, lay, eval_=ev, run=run, book=book)
    assert [r["invalid"] for r in rows] == [False, True, False]


def test_main_closes_a_killed_session_before_the_cap(tmp_path, monkeypatch):
    ra = load()
    monkeypatch.chdir(tmp_path)
    write_inputs(smoke())
    out = Path("results/m4-smoke/RND")
    budget.open_session(out, 1000.0)                                  # never closed (hard-killed)
    store.write_json(out / "result.json", {"complete": True})
    with pytest.raises(SystemExit, match="pass --resume"):
        ra.main(["--arm", "RND", "--smoke"])
    s = json.loads((out / "wall_clock.json").read_text())["sessions"]
    assert [x["status"] for x in s] == ["killed"] and s[0]["seconds"] is not None


def test_ac_battles_names_and_copy():
    src = (ROOT / "flymon/ac/battles.py").read_text()
    assert "copied from scripts/run_rescope_battles.py:Battles" in src
    assert schedules.account("TB", "E", 23) == "fm-aTBE-f23" and len(schedules.opponent_name("AE-f23-b019", 3)) <= 18
