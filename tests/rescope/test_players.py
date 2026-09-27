import json

from poke_env.ps_client import AccountConfiguration

from flymon.agent import logschema
from flymon.agent.player import AgentPlayer
from flymon.agent.screen import ScreenTable
from flymon.battle.attribution import Outcome
from flymon.battle.barrier import BatchBarrier
from flymon.battle.coach import Coach
from flymon.rescope.players import YokedPlayer
from flymon.rescope.yoke import YokedQueue


class FakeEncoder:                                   # as in tests/battle/test_agent_player.py
    def odour(self, battle, move):
        return {"ORN_" + move.id: 1.0}

    def situation_key(self, battle, cands):
        return ("me", "opp", "high", "high", tuple(sorted(m.id for m in cands)))


def _kw(tmp_path, name, log="fly00.jsonl"):
    async def batch(reqs):
        return [0] * len(reqs)
    return dict(encoder=FakeEncoder(), table=ScreenTable.allow_all(), rbarrier=BatchBarrier(batch, deadline_ms=5),
                coach=Coach(), barrier=BatchBarrier(batch, deadline_ms=5), log_path=tmp_path / log,
                account_configuration=AccountConfiguration(name, None), battle_format="gen1ou", start_listening=False)


_n = [0]


def make_yoked_player(tmp_path, queue, battle_id="PL-f00-b000"):
    _n[0] += 1
    p = YokedPlayer(fly=0, queue=queue, **_kw(tmp_path, f"fm-yoke-t{_n[0]}", log="rs00.jsonl"))
    p.start_battle(battle_id, 0)
    return p


def _fly_turn(p, tag, turn, outcome=Outcome(dealt_frac=1.0, effectiveness="super")):
    p._choice[tag] = {"turn": turn, "odour": {"ORN_DM1": 1.0}, "move": "surf"}
    p._on_outcome(tag, turn, outcome)


def test_yoked_delivers_queue_not_outcome(tmp_path):
    q = YokedQueue.from_bundles([[("PPL105", 400.0)]], donor_sha256="x")
    p = make_yoked_player(tmp_path, q)
    tag, turn = "battle-gen1ou-1", 3
    _fly_turn(p, tag, turn)                          # outcome would give PAM08
    assert [(j["dan"], j["ms"]) for j in p.pending_pulses(tag)] == [("PPL105", 400.0)]
    assert p.pending_pulses(tag)[0]["odour"] == {"ORN_DM1": 1.0}


def test_miss_outcome_still_gets_the_bundle(tmp_path):
    q = YokedQueue.from_bundles([[("PAM08", 200.0), ("PPL105", 200.0)]], donor_sha256="x")
    p = make_yoked_player(tmp_path, q)
    _fly_turn(p, "battle-gen1ou-1", 1, Outcome(dealt_frac=0.0, effectiveness="neutral", missed=True))
    assert [j["dan"] for j in p.pending_pulses("battle-gen1ou-1")] == ["PAM08", "PPL105"]
    assert p.no_signal == {}


def test_empty_queue_gives_no_pulse(tmp_path):
    q = YokedQueue.from_bundles([], donor_sha256="x")
    p = make_yoked_player(tmp_path, q)
    tag = "battle-gen1ou-1"
    _fly_turn(p, tag, 1)
    assert p.pending_pulses(tag) == [] and p.no_signal[tag] == 1 and q.exhausted == 1


def test_coach_turn_does_not_consume(tmp_path):
    q = YokedQueue.from_bundles([[("PAM08", 400.0)]], donor_sha256="x")
    p = make_yoked_player(tmp_path, q)
    p._on_outcome("battle-gen1ou-1", 2, Outcome(dealt_frac=1.0, effectiveness="super"))   # no _choice: coach turn
    assert p.pending_pulses("battle-gen1ou-1") == []
    assert q.pop() == [("PAM08", 400.0)]


def test_record_is_schema_valid_and_marked_yoked(tmp_path):
    q = YokedQueue.from_bundles([[("PPL105", 400.0)]], donor_sha256="abc")
    p = make_yoked_player(tmp_path, q)
    _fly_turn(p, "battle-gen1ou-1", 3)
    recs = [json.loads(x) for x in (tmp_path / "rs00.jsonl").read_text().splitlines()]
    r = [x for x in recs if x["kind"] == "reinforce"][-1]
    logschema.validate(r)
    assert r["yoked"] is True and r["donor_sha256"] == "abc" and r["pulses"] == [["PPL105", 400.0]]
    assert r["battle_id"] == "PL-f00-b000" and r["odour_move"] == "surf" and r["turn"] == 3


def test_donor_log_to_rs_carries_over_across_battles(tmp_path):
    """End to end: a real AgentPlayer donor log -> queue; the RS fly has a different turn split across battles but
    consumes the same bundles in order, and the eval block and no-signal turns never feed the queue."""
    donor = AgentPlayer(fly=0, **_kw(tmp_path, "fm-yoke-donor"))
    hit, resisted, miss = (Outcome(dealt_frac=0.9, effectiveness="neutral"),
                           Outcome(dealt_frac=0.5, effectiveness="resisted"),
                           Outcome(dealt_frac=0.0, effectiveness="neutral", missed=True))
    plan = {"PL-f00-b000": [hit, miss, resisted], "PL-f00-b001": [miss, hit], "PE-f00-b000": [hit]}
    for bi, (bid, outs) in enumerate(plan.items()):
        donor.start_battle(bid, bi)
        for t, o in enumerate(outs, 1):
            _fly_turn(donor, f"battle-gen1ou-{bi}", t, o)
    donor_recs = [json.loads(x) for x in (tmp_path / "fly00.jsonl").read_text().splitlines()]
    expected = [[tuple(pp) for pp in r["pulses"]] for r in donor_recs
                if r["kind"] == "reinforce" and r["battle_id"].startswith("PL-") and r["pulses"]]
    q = YokedQueue.from_log(tmp_path / "fly00.jsonl", {"PL-f00-b000", "PL-f00-b001"})
    assert q.bundles == expected and len(expected) == 3

    got = []
    p = make_yoked_player(tmp_path, q, battle_id="PL-f00-b000")
    for t in (1, 2):                                   # RS battle 0: two fly turns
        _fly_turn(p, "battle-gen1ou-10", t, miss)
    got += p.pending_pulses("battle-gen1ou-10")
    p.start_battle("PL-f00-b001", 1)                    # RS battle 1: the queue carries over
    for t in (1, 2):
        _fly_turn(p, "battle-gen1ou-11", t, miss)
    got += p.pending_pulses("battle-gen1ou-11")
    assert [(g["dan"], g["ms"]) for g in got] == [pp for b in expected for pp in b]
    assert q.exhausted == 1 and q.residual_frac() == 0.0 and p.no_signal == {"battle-gen1ou-11": 1}
