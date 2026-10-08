"""Stage-1 wiring (AC.2, AC.5, AC.8) on the fake pool: situation evaluations at the declared points (C-off at 0 only),
fly-level INVALID and STOP_INFRA, resume equivalence including uncommitted situation records, result rows."""
import dataclasses
import json

from flymon.ac import schedules, stage1
from flymon.ac.invalid import InvalidBook
from flymon.ac.spec import LABEL, SPEC, smoke
from flymon.rescope import blocks
from tests.ac.ac_fakes import FakePool, run_fake

S = dataclasses.replace(smoke(), brain_arms=(("FLY", 2), ("COFF", 1), ("TB", 1)),
                        min_valid=(("FLY", 2), ("COFF", 1), ("TB", 1), ("RND", 1), ("MAX", 1)))
DOC = {"learn": blocks.schedule_rows(schedules.learn_canonical(S)),
       "eval": blocks.schedule_rows(schedules.eval_canonical(S))}
LAY = stage1.layout(S)
KEY = "h" * 64


def evaluate(g, point):
    return dict(kind="situation_eval", fly=g, point=point, rate=(g + point) / 10, switched=[1, 0], frozen=True)


def setup(out, enabled=(True, True, False, True)):
    book = InvalidBook(out / "invalid.json", {g: a for g, (a, _) in enumerate(LAY)}, dict(S.brain_arms),
                       S.min_valid_of())
    log = stage1.SituationLog(out / "logs")
    return book, log, stage1.Stage1Hooks(S, LAY, book, log, evaluate), FakePool(len(LAY), list(enabled))


def go(out, book, log, hooks, pool, learn, ev, **kw):
    if kw.get("resume"):
        log.filter(stage1.committed(out, KEY))
    hooks.initial()
    return run_fake(out, learn=learn, eval_=ev, n=len(LAY), pool=pool, after_battle=hooks.after_battle,
                    wrap=lambda af: stage1.skipping(af, book), **kw)


def test_layout_and_schedules():
    assert LAY == [("FLY", 0), ("FLY", 1), ("COFF", 0), ("TB", 0)]
    assert stage1.layout(SPEC)[:1] == [("FLY", 0)] and stage1.layout(SPEC)[12] == ("COFF", 0)
    assert stage1.layout(SPEC)[18] == ("TB", 0) and len(stage1.layout(SPEC)) == 24
    learn, ev = stage1.brain_schedules(S, DOC)
    assert [s.battle_id for s in learn if s.fly_id == 2] == ["SL-f02-b000", "SL-f02-b001"]
    t = lambda sched, g: [s.my_team for s in sched if s.fly_id == g]
    assert t(learn, 0) == t(learn, 2) == t(learn, 3) != t(learn, 1)          # FLY 0, C-off 0, TB 0 share row 0
    rnd = stage1.nobrain_schedule(S, DOC, "RND")
    assert t(rnd, 0) == t(ev, 0) and [s.battle_id for s in rnd if s.fly_id == 1][0] == "SE-f01-b000"
    assert stage1.nobrain_layout(S, "MAX") == [("MAX", 0), ("MAX", 1)]


def test_points_are_recorded_after_the_declared_battles(tmp_path):
    book, log, hooks, pool = setup(tmp_path)
    learn, ev = stage1.brain_schedules(S, DOC)
    run, _ = go(tmp_path, book, log, hooks, pool, learn, ev)
    assert run["complete"]
    recs = log.records()
    assert sorted((r["fly"], r["point"]) for r in recs) == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0),
                                                            (3, 0), (3, 1), (3, 2)]
    later = {(r["fly"], r["point"]): r["battle_id"] for r in recs if r["point"] > 0}
    assert later[(3, 2)] == "SL-f03-b001" and later[(0, 1)] == "SL-f00-b000"
    assert all(r["arm"] == LAY[r["fly"]][0] for r in recs)
    assert all(r["label"] == LABEL for r in recs)


def test_invalid_fly_is_skipped_and_arm_stops_below_min_valid(tmp_path):
    book, log, hooks, pool = setup(tmp_path)
    learn, ev = stage1.brain_schedules(S, DOC)
    fail = {("SL-f01-b000", i) for i in range(S.retry_max + 1)}
    run, _ = go(tmp_path, book, log, hooks, pool, learn, ev, fail=fail)
    assert 1 in book.invalid and book.stopped["FLY"]["status"] == "STOP_INFRA"
    assert not any(r["fly"] == 1 and r["point"] > 0 for r in log.records())
    assert {(r["fly"], r["point"]) for r in log.records() if r["fly"] == 3} == {(3, 0), (3, 1), (3, 2)}
    recs = {r["battle_id"]: r for r in blocks.read_jsonl(tmp_path / "logs/battles.jsonl")}
    assert recs["SL-f01-b001"]["invalid"] and not recs["SL-f01-b001"]["finished"]
    rows = stage1.per_fly_rows(tmp_path, LAY, eval_=ev, learn=learn, run=run, book=book,
                               w0_sha=blocks.weights_sha(pool.w0[None]))
    assert rows[1]["invalid"] and rows[0]["invalid"] and not rows[3]["invalid"] and not rows[2]["invalid"]


def test_resume_drops_uncommitted_situation_records(tmp_path):
    learn, ev = stage1.brain_schedules(S, DOC)
    b0, l0, h0, p0 = setup(tmp_path / "full")
    go(tmp_path / "full", b0, l0, h0, p0, learn, ev)
    out = tmp_path / "res"
    b1, l1, h1, p1 = setup(out)
    run, _ = go(out, b1, l1, h1, p1, learn, ev, stop_after=3)
    assert not run["complete"]
    done = stage1.committed(out, KEY)
    ghost = next(s.battle_id for s in learn if s.battle_id not in done and schedules.battle_index(s.battle_id) == 0)
    blocks.append_jsonl(out / "logs/situations.jsonl", dict(kind="situation_eval", fly=0, point=1, battle_id=ghost))
    b2, l2, h2, p2 = setup(out)
    run, _ = go(out, b2, l2, h2, p2, learn, ev, resume=True)
    assert run["complete"]
    key = lambda r: (r["fly"], r["point"], r.get("battle_id"))
    assert sorted(map(key, l2.records())) == sorted(map(key, l0.records()))
    assert len(l2.records()) == len(l0.records())


def test_coff_weight_change_makes_it_invalid(tmp_path):
    book, log, hooks, pool = setup(tmp_path, enabled=(True, True, True, True))   # C-off wrongly plastic
    learn, ev = stage1.brain_schedules(S, DOC)
    run, _ = go(tmp_path, book, log, hooks, pool, learn, ev)
    rows = stage1.per_fly_rows(tmp_path, LAY, eval_=ev, learn=learn, run=run, book=book,
                               w0_sha=blocks.weights_sha(pool.w0[None]))
    assert rows[2]["arm"] == "COFF" and rows[2]["coff_weights_unchanged"] is False and rows[2]["invalid"]
    assert book.invalid[2] == "coff_weights_changed"
    assert json.loads((tmp_path / "invalid.json").read_text())["invalid"]["2"] == "coff_weights_changed"


def test_coff_rows_when_frozen(tmp_path):
    book, log, hooks, pool = setup(tmp_path)
    learn, ev = stage1.brain_schedules(S, DOC)
    run, _ = go(tmp_path, book, log, hooks, pool, learn, ev)
    rows = stage1.per_fly_rows(tmp_path, LAY, eval_=ev, learn=learn, run=run, book=book,
                               w0_sha=blocks.weights_sha(pool.w0[None]))
    assert rows[2]["coff_weights_unchanged"] is True and not any(r["invalid"] for r in rows)
    assert [(r["arm"], r["k"]) for r in rows] == LAY


def test_all_brain_flies_share_the_evaluation_teams():
    _, ev = stage1.brain_schedules(S, DOC)
    t = lambda sched, g: [(s.my_team, s.opp_team, s.opponent) for s in sched if s.fly_id == g]
    assert len(t(ev, 0)) == S.eval_battles
    assert t(ev, 0) == t(ev, 1) == t(ev, 2) == t(ev, 3)          # FLY 0, FLY 1, C-off 0, TB 0: one schedule (AC.4)
    full = stage1.brain_schedules(SPEC, {"learn": blocks.schedule_rows(schedules.learn_canonical(SPEC)),
                                         "eval": blocks.schedule_rows(schedules.eval_canonical(SPEC))})[1]
    assert all(t(full, g) == t(full, 0) for g in range(SPEC.n_brain())) and len(t(full, 0)) == 20


def test_flies_skipped_because_their_arm_stopped_are_not_battle_failures(tmp_path):
    book, log, hooks, pool = setup(tmp_path)
    learn, ev = stage1.brain_schedules(S, DOC)
    fail = {("SL-f01-b000", i) for i in range(S.retry_max + 1)}
    run, _ = go(tmp_path, book, log, hooks, pool, learn, ev, fail=fail)
    assert book.invalid[1].startswith("SL-f01-b000: unfinished after")
    assert book.invalid[0] == "arm_stopped"                       # FLY stopped (STOP_INFRA): fly 0 was only skipped
    assert book.stopped["FLY"]["n_invalid"] == 1
    rows = stage1.per_fly_rows(tmp_path, LAY, eval_=ev, learn=learn, run=run, book=book,
                               w0_sha=blocks.weights_sha(pool.w0[None]))
    assert rows[0]["invalid"] and rows[0]["invalid_reason"] == "arm_stopped"
    assert rows[1]["invalid_reason"].startswith("SL-f01-b000")


def test_after_battle_on_a_stopped_arm_marks_arm_stopped(tmp_path):
    book, log, hooks, _ = setup(tmp_path)
    book.stopped["FLY"] = dict(status="STOP_INFRA")
    sb = stage1.brain_schedules(S, DOC)[0][0]
    assert sb.fly_id == 0 and book.skip(0)
    hooks.after_battle("L", sb, dict(invalid=True, retries=S.retry_max))
    assert book.invalid == {0: "arm_stopped"}
    hooks.after_battle("L", sb, dict(invalid=True, retries=S.retry_max))
    assert book.invalid == {0: "arm_stopped"}                     # an already-invalid fly keeps its first reason


def test_initial_repairs_a_torn_last_init_line(tmp_path):
    """Final review M4: a fly killed mid-write leaves a torn last line; initial() drops it and redoes point 0."""
    book, log, hooks, _ = setup(tmp_path)
    log.init_path.parent.mkdir(parents=True)
    good = json.dumps(dict(kind="situation_eval", fly=0, point=0, switched=[1, 0], frozen=True))
    log.init_path.write_text(good + "\n" + '{"kind": "situation_eval", "fly": 1, "po')
    hooks.initial()
    lines = log.init_path.read_text().splitlines()
    recs = [json.loads(x) for x in lines]                        # every line parses
    assert sorted(r["fly"] for r in recs) == [0, 1, 2, 3] and log.init_path.read_text().endswith("\n")


def test_initial_repairs_a_last_init_line_without_newline(tmp_path):
    book, log, hooks, _ = setup(tmp_path)
    log.init_path.parent.mkdir(parents=True)
    log.init_path.write_text(json.dumps(dict(kind="situation_eval", fly=0, point=0, switched=[1], frozen=True)))
    hooks.initial()
    recs = [json.loads(x) for x in log.init_path.read_text().splitlines()]
    assert sorted(r["fly"] for r in recs) == [0, 1, 2, 3]
