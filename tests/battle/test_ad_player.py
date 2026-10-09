"""AD's players (AD.1, AD.5 2): the global fly in the pulse seed, the attempt token and the generation in every pulse
request, gfly on every record, FLY-OS's shadow on fly decisions, no pulse from the evaluation player."""
import json

from poke_env.ps_client import AccountConfiguration

from flymon.ad.attempts import AttemptBook
from flymon.ad.player import ADAgentPlayer, ADEvalPlayer
from flymon.agent import logschema
from flymon.agent.policy import derive_seed
from flymon.agent.screen import ScreenTable
from flymon.battle.attribution import Outcome
from flymon.battle.barrier import BatchBarrier
from flymon.battle.coach import Coach
from flymon.battle.pool import by_species


class Enc:
    def odour(self, battle, move):
        return {"ORN_" + move.id: 1.0}

    def situation_key(self, battle, cands):
        return ("me", "opp", tuple(sorted(m.id for m in cands)))


class ShadowEnc(Enc):
    def __init__(self):
        self.battle_id = None

    def set_battle(self, battle_id):
        self.battle_id = battle_id

    def shadow(self, battle):
        return dict(true="Charizard", drawn="Lapras", same_types=False)


def make(tmp_path, cls=ADAgentPlayer, enc=None, name="fm-ad-t1"):
    async def decide_batch(reqs):
        for r in reqs:
            n = len(r.candidates)
            r.context["detail"] = {"v": [0.0] * n, "a": [0] * n, "p": [0] * n, "kc_active": [0] * n, "tau": 1.0,
                                   "seed": 1}
        return [0] * len(reqs)

    async def reinforce_batch(reqs):
        return [0] * len(reqs)

    att = AttemptBook()
    p = cls(fly=3, encoder=enc or Enc(), table=ScreenTable.allow_all(),
            rbarrier=BatchBarrier(reinforce_batch, deadline_ms=5), coach=Coach(),
            barrier=BatchBarrier(decide_batch, deadline_ms=5), log_path=tmp_path / "fly03.jsonl",
            account_configuration=AccountConfiguration(name, None), battle_format="gen1ou", start_listening=False,
            gfly=27, attempts=att, gen_of=lambda f: 7)
    p.start_attempt("DL-f27-b004", 4, 1)
    return p, att


def records(tmp_path):
    return [json.loads(x) for x in (tmp_path / "fly03.jsonl").read_text().splitlines()]


async def test_pulse_request_carries_token_generation_and_global_seed(make_battle, tmp_path):
    p, att = make(tmp_path)
    assert att.current(3) == ("DL-f27-b004", 1) and p.attempt == ["DL-f27-b004", 1] and p.battle_index == 4
    battle = make_battle(by_species["Blastoise"], "Charizard")
    await p.choose_move(battle)
    p._on_outcome(battle.battle_tag, battle.turn, Outcome(dealt_frac=0.5, effectiveness="resisted"))
    q = p.pending_pulses(battle.battle_tag)
    assert [x["dan"] for x in q] == ["PAM08", "PPL105"]
    assert all(x["attempt"] == ["DL-f27-b004", 1] and x["gen"] == 7 and x["fly"] == 3 for x in q)
    assert q[1]["seed"] == derive_seed("reinforce", 27, "DL-f27-b004", battle.turn, 1)
    recs = records(tmp_path)
    for r in recs:
        logschema.validate(r)
        assert r["gfly"] == 27 and r["fly"] == 3
    rein = [r for r in recs if r["kind"] == "reinforce"][-1]
    assert rein["attempt"] == ["DL-f27-b004", 1] and rein["gen"] == 7


async def test_fly_decision_carries_the_os_shadow(make_battle, tmp_path):
    enc = ShadowEnc()
    p, _ = make(tmp_path, enc=enc, name="fm-ad-t2")
    assert enc.battle_id == "DL-f27-b004"
    await p.choose_move(make_battle(by_species["Blastoise"], "Charizard"))
    d = records(tmp_path)[-1]
    assert d["decider"] == "fly" and d["os"] == {"true": "Charizard", "drawn": "Lapras", "same_types": False}


async def test_eval_player_queues_no_pulse_and_logs_multipliers_and_shadow(make_battle, tmp_path):
    p, _ = make(tmp_path, cls=ADEvalPlayer, enc=ShadowEnc(), name="fm-ad-t3")
    battle = make_battle(by_species["Blastoise"], "Charizard")
    await p.choose_move(battle)
    p._on_outcome(battle.battle_tag, battle.turn, Outcome(dealt_frac=0.5, effectiveness="resisted"))
    assert p.pending_pulses(battle.battle_tag) == []
    d = records(tmp_path)[-1]
    assert d["gfly"] == 27 and "multipliers" in d and d["os"]["drawn"] == "Lapras"
