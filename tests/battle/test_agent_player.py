import asyncio
import json

import pytest
from poke_env.ps_client import AccountConfiguration

from flymon.agent import logschema
from flymon.agent.player import AgentPlayer
from flymon.agent.screen import ScreenTable
from flymon.agent.swarm import SafetyStop
from flymon.battle.attribution import Outcome
from flymon.battle.barrier import BatchBarrier
from flymon.battle.coach import Coach
from flymon.battle.pool import by_species


class FakeEncoder:
    def odour(self, battle, move):
        return {"ORN_" + move.id: 1.0}

    def situation_key(self, battle, cands):
        return ("me", "opp", "high", "high", tuple(sorted(m.id for m in cands)))


def make_player(tmp_path, table=None, name="fm-agent-t", decide_exc=None, reinforce_exc=None):
    decided, reinforced = [], []

    async def decide_batch(reqs):
        if decide_exc is not None:
            raise decide_exc
        for r in reqs:
            r.context["detail"] = {"v": [0.0] * len(r.candidates), "a": [0] * len(r.candidates),
                                   "p": [0] * len(r.candidates), "kc_active": [0] * len(r.candidates),
                                   "tau": 1.0, "seed": 1}
            decided.append(r.context)
        return [0] * len(reqs)

    async def reinforce_batch(reqs):
        if reinforce_exc is not None:
            raise reinforce_exc
        reinforced.extend(r.context for r in reqs)
        return [0] * len(reqs)

    p = AgentPlayer(fly=0, encoder=FakeEncoder(), table=table or ScreenTable.allow_all(),
                    rbarrier=BatchBarrier(reinforce_batch, deadline_ms=5), coach=Coach(),
                    barrier=BatchBarrier(decide_batch, deadline_ms=5), log_path=tmp_path / "fly00.jsonl",
                    account_configuration=AccountConfiguration(name, None), battle_format="gen1ou",
                    start_listening=False)
    p.start_battle("f00-b000", 0)
    return p, decided, reinforced


async def test_fly_turn_logs_schema_valid_decision_with_values(make_battle, tmp_path):
    p, decided, _ = make_player(tmp_path)
    battle = make_battle(by_species["Blastoise"], "Charizard")
    await p.choose_move(battle)
    recs = [json.loads(x) for x in (tmp_path / "fly00.jsonl").read_text().splitlines()]
    for r in recs:
        logschema.validate(r)
    d = recs[-1]
    assert d["decider"] == "fly" and d["battle_id"] == "f00-b000" and len(d["v"]) == len(d["candidates"])
    assert decided[0]["fly"] == 0 and len(decided[0]["odours"]) == len(d["candidates"])


async def test_screen_can_hand_the_turn_to_the_coach(make_battle, tmp_path):
    p, decided, _ = make_player(tmp_path, table=ScreenTable({}, default="coach"), name="fm-agent-t2")
    await p.choose_move(make_battle(by_species["Blastoise"], "Charizard"))
    assert decided == []
    assert json.loads((tmp_path / "fly00.jsonl").read_text().splitlines()[-1])["decider"] == "coach"


async def test_fly_turn_outcome_queues_pulses_and_drain_delivers_them(make_battle, tmp_path):
    p, _, reinforced = make_player(tmp_path, name="fm-agent-t3")
    battle = make_battle(by_species["Blastoise"], "Charizard")
    await p.choose_move(battle)
    p._on_outcome(battle.battle_tag, battle.turn, Outcome(dealt_frac=0.5, effectiveness="resisted"))
    assert [q["dan"] for q in p.pending_pulses(battle.battle_tag)] == ["PAM08", "PPL105"]
    await p.drain()
    assert [r["dan"] for r in reinforced] == ["PAM08", "PPL105"] and reinforced[0]["fly"] == 0
    assert p.pending_pulses(battle.battle_tag) == []


async def test_coach_turn_never_reinforces(make_battle, tmp_path):
    """Safety constraint 3: an outcome closing on a coach-decided turn drives no DAN."""
    p, _, reinforced = make_player(tmp_path, table=ScreenTable({}, default="coach"), name="fm-agent-t4")
    battle = make_battle(by_species["Blastoise"], "Charizard")
    await p.choose_move(battle)
    p._on_outcome(battle.battle_tag, battle.turn, Outcome(dealt_frac=0.9, effectiveness="neutral"))
    await p.drain()
    assert reinforced == [] and p.pending_pulses(battle.battle_tag) == []


async def test_forced_switch_keeps_pending_fly_choice(make_battle, tmp_path):
    p, _, _ = make_player(tmp_path, name="fm-agent-t5")
    battle = make_battle(by_species["Blastoise"], "Charizard")
    await p.choose_move(battle)
    forced = make_battle(by_species["Blastoise"], "Charizard", force_switch=True)
    forced._battle_tag = battle.battle_tag
    await p.choose_move(forced)
    p._on_outcome(battle.battle_tag, battle.turn, Outcome(dealt_frac=0.2, effectiveness="neutral"))
    assert [q["dan"] for q in p.pending_pulses(battle.battle_tag)] == ["PAM08"]


def test_schema_rejects_missing_fields():
    with pytest.raises(ValueError, match="battle_id"):
        logschema.validate({"schema": 1, "kind": "decision", "fly": 0, "battle_tag": "b", "turn": 1})
    with pytest.raises(ValueError, match="kind"):
        logschema.validate({"schema": 1, "kind": "nope", "fly": 0, "battle_id": "x", "battle_tag": "b", "turn": 1})


def stub_forfeit(p, monkeypatch):
    sent = []

    async def send_message(message, room="", message_2=None):
        sent.append((message, room))
    monkeypatch.setattr(p.ps_client, "send_message", send_message)
    return sent


async def test_failing_decision_batch_records_fatal_forfeits_and_reraises(make_battle, tmp_path, monkeypatch):
    """poke-env never awaits choose_move's task: without the forfeit the battle would hang with no move sent."""
    p, _, _ = make_player(tmp_path, name="fm-agent-t6", decide_exc=KeyError("ORN_x"))
    sent = stub_forfeit(p, monkeypatch)
    battle = make_battle(by_species["Blastoise"], "Charizard")
    with pytest.raises(KeyError):
        await p.choose_move(battle)
    assert isinstance(p.fatal, KeyError)
    assert sent == [("/forfeit", battle.battle_tag)]


async def test_safety_stop_in_flush_records_fatal_forfeits_and_reraises(make_battle, tmp_path, monkeypatch):
    stop = SafetyStop("median left the band")
    p, _, _ = make_player(tmp_path, name="fm-agent-t7", reinforce_exc=stop)
    sent = stub_forfeit(p, monkeypatch)
    battle = make_battle(by_species["Blastoise"], "Charizard")
    await p.choose_move(battle)
    p._on_outcome(battle.battle_tag, battle.turn, Outcome(dealt_frac=0.5, effectiveness="resisted"))
    with pytest.raises(SafetyStop):
        await p.choose_move(battle)
    assert p.fatal is stop and sent == [("/forfeit", battle.battle_tag)]
    assert p.player_id not in p.rbarrier.active


async def test_first_fatal_wins(make_battle, tmp_path, monkeypatch):
    p, _, _ = make_player(tmp_path, name="fm-agent-t8", decide_exc=KeyError("first"))
    stub_forfeit(p, monkeypatch)
    battle = make_battle(by_species["Blastoise"], "Charizard")
    with pytest.raises(KeyError):
        await p.choose_move(battle)
    first = p.fatal
    with pytest.raises(KeyError):
        await p.choose_move(battle)
    assert p.fatal is first


async def test_flush_unregisters_from_rbarrier_when_submit_raises(make_battle, tmp_path):
    p, _, _ = make_player(tmp_path, name="fm-agent-t9", reinforce_exc=SafetyStop("stop"))
    battle = make_battle(by_species["Blastoise"], "Charizard")
    await p.choose_move(battle)
    p._on_outcome(battle.battle_tag, battle.turn, Outcome(dealt_frac=0.5, effectiveness="resisted"))
    assert len(p.pending_pulses(battle.battle_tag)) == 2
    with pytest.raises(SafetyStop):
        await p.drain()
    assert p.player_id not in p.rbarrier.active
    assert p.pending_pulses(battle.battle_tag) == []   # unsent pulses of a fatal battle are dropped
