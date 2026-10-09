"""AD's budget (AD.4): sessions charged to the next start, the remaining budget and the session cap, the case-wise
extrapolation and T, the 3-valid-of-6 bench median, and the per-batch worst-case reservation."""
import datetime as dt
import json
from types import SimpleNamespace

import pytest

from flymon.ad import budget, store
from flymon.ad.spec import SPEC


def test_reference_values_choose_120_with_the_new_set():
    c = budget.choose(SPEC, 838.9, 27.2)
    assert {k: round(v, 2) for k, v in c["cases"].items()} == {"T120_new": 45.59, "T120": 44.89, "T80_new": 32.77,
                                                                "T80": 32.06}
    assert (c["status"], c["T"], c["new_set"], c["case"]) == ("OK", 120, True, "T120_new")
    assert round(c["limit_h"], 3) == 47.418


@pytest.mark.parametrize("s_batch,s_eval,want", [
    (730.0, 100.0, ("OK", 120, False, "T120")),
    (900.0, 27.2, ("OK", 80, True, "T80_new")),
    (1300.0, 27.2, ("STOP_BUDGET", None, None, None))])
def test_choose_follows_the_pre_registered_order(s_batch, s_eval, want):
    c = budget.choose(SPEC, s_batch, s_eval)
    assert (c["status"], c["T"], c["new_set"], c["case"]) == want


def test_rescope_floor():
    assert budget.hours(SPEC, 300.0, 0.0, 120, False) == pytest.approx(431.1 * 140 * 1.3 / 3600)


def test_close_stale_charges_to_the_next_start(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "results/m4/ad"
    iso = lambda t: dt.datetime.fromtimestamp(t, dt.timezone.utc).isoformat()
    store.write_json(out / "wall_clock.json", {"sessions": [
        dict(started_utc=iso(1000), ended_utc=None, seconds=None, status="running"),
        dict(started_utc=iso(5000), ended_utc=iso(5100), seconds=100.0, status="ok"),
        dict(started_utc=iso(9000), ended_utc=None, seconds=None, status="running")]})
    assert budget.close_stale(out, now=12000.0) == 2
    s = json.loads((out / "wall_clock.json").read_text())["sessions"]
    assert [x["seconds"] for x in s] == [4000.0, 100.0, 3000.0] and s[0]["status"] == "killed"
    assert s[0]["charged"] == "to_next_start" and s[2]["charged"] == "to_detection"
    assert budget.spent_s(out) == 7100.0 and budget.close_stale(out, now=13000.0) == 0


def test_remaining_and_session_cap():
    assert budget.remaining_s(SPEC, 10 * 3600) == pytest.approx((47.41797 - 10) * 3600, abs=1)
    assert budget.session_cap_s(SPEC, 10 * 3600) == 24 * 3600
    assert budget.session_cap_s(SPEC, 30 * 3600) == pytest.approx(budget.remaining_s(SPEC, 30 * 3600))


def att(n, valid, b, e):
    return dict(attempt=n, valid=valid, s_per_batch_battle=b, sit_eval_s_per_eval=e)


def test_bench_summary():
    a = [att(1, True, 800.0, 27.0), att(2, False, 2900.0, 90.0), att(3, True, 850.0, 28.0), att(4, True, 820.0, 26.0)]
    s = budget.bench_summary(a, SPEC)
    assert s["status"] == "OK" and s["used"] == [1, 3, 4] and s["s_per_batch_battle"] == 820.0
    assert s["sit_eval_s"] == 27.0 and s["values"]["s_per_batch_battle"] == [800.0, 850.0, 820.0]
    assert budget.bench_summary(a[:2], SPEC)["status"] == "MORE"
    six = [att(i, i < 3, 800.0, 27.0) for i in range(1, 7)]
    assert budget.bench_summary(six, SPEC)["status"] == "STOP_BENCH"


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def sb(bid):
    return SimpleNamespace(battle_id=bid)


def gate(tmp_path, clock, *, session_end=1e9, budget_end=1e9, new_set=True):
    return budget.BatchGate(SPEC, 120, new_set, session_end=session_end, budget_end=budget_end, bench_batch_s=100.0,
                            bench_eval_s=10.0, times_path=tmp_path / "results/m4/ad/batch_times.json", now=clock)


def test_gate_refuses_a_batch_the_budget_cannot_cover(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    clock = Clock()
    g = gate(tmp_path, clock, budget_end=1000.0)
    assert g.worst("L", 0) == 200.0
    assert g("L", sb("DL-f24-b000")) is False and g("L", sb("DL-f25-b000")) is False
    clock.t = 850.0
    assert g("L", sb("DL-f24-b001")) is True and g.refused == "STOP_BUDGET"
    assert g("L", sb("DL-f26-b000")) is False                      # batch 0 was reserved: the lagging fly plays it
    assert g("L", sb("DL-f25-b001")) is True
    assert g.refusal == dict(block="L", index=1, worst_s=200.0, left_s=150.0, reason="STOP_BUDGET")


def test_gate_session_cap_is_not_a_budget_stop(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    clock = Clock()
    g = gate(tmp_path, clock, session_end=150.0)
    assert g("L", sb("DL-f24-b000")) is True and g.refused == "session_cap"


def test_gate_uses_the_longest_batch_and_point_evaluations(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    clock = Clock()
    g = gate(tmp_path, clock)
    g("L", sb("DL-f24-b000")); clock.t = 30.0; g("L", sb("DL-f25-b000"))
    clock.t = 400.0; g.finished("L", sb("DL-f24-b000"))
    clock.t = 450.0; g.finished("L", sb("DL-f25-b000"))
    assert g.longest == 450.0
    assert g.worst("L", 5) == 900.0                                 # no point after battle 5
    assert g.worst("L", 19) == 900.0 + 10.0 * 36                    # point 20: 24 own + 12 paired
    assert g.worst("L", 39) == 900.0 + 10.0 * 60                    # point 40: + 24 new-set
    assert g.worst("E", 3) == 900.0
    again = gate(tmp_path, Clock())
    assert again.longest == 450.0                                   # persisted across sessions
