"""The two hooks AC adds to the shared runner (default None: rescope and M3 behave as before) and the fly-level
INVALID book (AC.8)."""
import json

import pytest

from flymon.ac.invalid import InvalidBook
from flymon.ac.spec import LABEL
from flymon.rescope import blocks
from tests.ac.ac_fakes import run_fake, snapshot

LEARN, EV = blocks.block_schedule(2, 3, 101, "PL"), blocks.block_schedule(2, 2, 102, "PE")


def test_after_battle_sees_every_settled_battle_once(tmp_path):
    seen = []
    run, _ = run_fake(tmp_path / "a", learn=LEARN, eval_=EV, n=2,
                      after_battle=lambda block, sb, rec: seen.append((block, sb.battle_id, rec["invalid"])))
    assert run["complete"]
    assert sorted(seen) == sorted([("L", s.battle_id, False) for s in LEARN] + [("E", s.battle_id, False) for s in EV])


def test_should_stop_stops_and_resume_completes(tmp_path):
    full, pfull = run_fake(tmp_path / "full", learn=LEARN, eval_=EV, n=2)
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > 3                       # three battle starts, then the session cap is reached

    run, _ = run_fake(tmp_path / "x", learn=LEARN, eval_=EV, n=2, should_stop=stop)
    assert not run["complete"] and len(run["played"]["L"]) == 3
    run, pool = run_fake(tmp_path / "x", learn=LEARN, eval_=EV, n=2, resume=True)
    assert run["complete"] and run["before"] == full["before"] and run["after"] == full["after"]
    assert snapshot(tmp_path / "x") == snapshot(tmp_path / "full")


def test_should_stop_at_the_block_boundary(tmp_path):
    played = []
    run, _ = run_fake(tmp_path / "x", learn=LEARN, eval_=EV, n=2,
                      after_battle=lambda b, sb, rec: played.append(b), should_stop=lambda: len(played) >= 6)
    assert not run["complete"] and set(run["played"]) == {"L"} and len(run["played"]["L"]) == 6
    run, _ = run_fake(tmp_path / "x", learn=LEARN, eval_=EV, n=2, resume=True)
    assert run["complete"]


def test_book_marks_skips_and_stops_an_arm(tmp_path):
    arm_of = {0: "FLY", 1: "FLY", 2: "FLY", 3: "COFF"}
    book = InvalidBook(tmp_path / "invalid.json", arm_of, {"FLY": 3, "COFF": 1}, {"FLY": 2, "COFF": 1})
    book.mark(1, "PL-f01-b000: unfinished after 3 retries")
    assert book.skip(1) and not book.skip(0) and book.n_valid("FLY") == 2 and "FLY" not in book.stopped
    book.mark(1, "again")                              # idempotent: the first reason stays
    assert book.invalid[1].startswith("PL-f01")
    book.mark(2, "x")
    assert book.stopped["FLY"] == {"status": "STOP_INFRA", "n_invalid": 2, "size": 3, "min_valid": 2}
    assert book.skip(0) and not book.skip(3)
    again = InvalidBook(tmp_path / "invalid.json", arm_of, {"FLY": 3, "COFF": 1}, {"FLY": 2, "COFF": 1})
    assert again.invalid == book.invalid and again.stopped == book.stopped
    saved = json.loads((tmp_path / "invalid.json").read_text())
    assert saved["invalid"]["2"] == "x" and saved["label"] == LABEL      # every AC result JSON carries the AC.5 label
    assert book.status() == {"invalid": {"1": book.invalid[1], "2": "x"}, "stopped": book.stopped,
                             "n_valid": {"FLY": 1, "COFF": 1}}
