"""FlyCoachPlayer: real gen1ou battles against a local Showdown server, plus one offline case."""
import json

import pytest
from poke_env.player import RandomPlayer
from poke_env.ps_client import AccountConfiguration, ServerConfiguration

from flymon.battle.barrier import BatchBarrier
from flymon.battle.coach import Coach
from flymon.battle.fly_coach_player import FlyCoachPlayer
from flymon.battle.opponents import make_opponent
from flymon.battle.pool import POOL, by_species, team_export
from flymon.battle.providers import RandomProvider
from flymon.battle.server import NODE_BIN, ShowdownServer

requires_server = pytest.mark.skipif(not NODE_BIN.exists(), reason="run scripts/install_showdown.sh first")


@pytest.fixture(scope="module")
def server():
    """A local Showdown server shared by the server-backed tests in this module."""
    with ShowdownServer(port=8790) as s:
        yield s


@requires_server
async def test_two_battles_end_and_log_turns(server, tmp_path):
    cfg = ServerConfiguration(server.url_ws, "https://play.pokemonshowdown.com/action.php?")
    me = FlyCoachPlayer(provider=RandomProvider(1), coach=Coach(), barrier=None, log_path=tmp_path / "log.jsonl",
                        account_configuration=AccountConfiguration("fm-test-me", None), battle_format="gen1ou",
                        server_configuration=cfg, team=team_export(POOL[:6]), max_concurrent_battles=1)
    opp = RandomPlayer(account_configuration=AccountConfiguration("fm-test-opp", None), battle_format="gen1ou",
                       server_configuration=cfg, team=team_export(POOL[6:12]))
    try:
        await me.battle_against(opp, n_battles=2)
        assert me.n_finished_battles == 2
        lines = (tmp_path / "log.jsonl").read_text().splitlines()
        assert len(lines) > 10
        recs = [json.loads(line) for line in lines]
        assert {r["kind"] for r in recs} == {"decision", "outcome"}
        decisions = [r for r in recs if r["kind"] == "decision"]
        outcomes = [r for r in recs if r["kind"] == "outcome"]
        assert {r["decider"] for r in decisions} <= {"fly", "coach"}
        assert all(r["coach_kind"] in ("attack", "support", "switch", "default") for r in decisions)
        assert any(r["decider"] == "fly" and len(r["candidates"]) >= 2 for r in decisions)
        assert outcomes, "attribution never produced an outcome"
        assert all("outcome" not in r for r in decisions)
        stats = [me.battle_stats(t) for t in me.battles]
        assert all(s["fly_turns"] + s["coach_turns"] > 0 for s in stats)
        assert all(isinstance(s["won"], bool) for s in stats)
        assert all(s["finished"] for s in stats)
        assert all(me.n_candidates_mean(t) >= 2.0 for t in me.battles if me.battle_stats(t)["fly_turns"])
        for tag, battle in me.battles.items():
            # every battle attributed at least one of my action blocks, on the right side
            assert me.attributors[tag].my_side == battle.player_role
            assert me.outcomes.get(tag)
            mine = [r for r in outcomes if r["battle_tag"] == tag]
            assert len(mine) == len(me.outcomes[tag])
            # an outcome belongs to a turn in which I was asked to act and used a move
            move_turns = {r["turn"] for r in decisions
                          if r["battle_tag"] == tag and (r["decider"] == "fly" or r["coach_kind"] in ("attack", "support"))}
            assert all(r["turn"] in move_turns for r in mine), (mine, sorted(move_turns))
            # the last turn's outcome -- the KO/winning one -- reaches the log
            assert mine[-1]["turn"] == battle.turn
            assert mine[-1]["outcome"]["move_id"]
    finally:
        await me.ps_client.stop_listening()
        await opp.ps_client.stop_listening()


@requires_server
async def test_make_opponent_kinds(server):
    cfg = ServerConfiguration(server.url_ws, "https://play.pokemonshowdown.com/action.php?")
    team = team_export(POOL[:6])
    players = [make_opponent("random", 91, cfg, team), make_opponent("heuristic", 92, cfg, team)]
    try:
        for p in players:  # let the login finish, so stop_listening does not race the handshake
            await p.ps_client.wait_for_login()
        assert players[0].username == "fm-random-91"
        assert players[1].username == "fm-heuristic-92"
        assert type(players[0]).__name__ == "RandomPlayer"
        assert type(players[1]).__name__ == "SimpleHeuristicsPlayer"
        with pytest.raises(KeyError):
            make_opponent("nope", 93, cfg, team)
    finally:
        for p in players:
            await p.ps_client.stop_listening()


@requires_server
async def test_two_battles_through_the_barrier(server, tmp_path):
    """The barrier path end to end: the fly re-registers per battle, so no turn hangs on it."""
    calls = []

    async def run_batch(reqs):
        calls.append([r.player_id for r in reqs])
        return [0] * len(reqs)

    cfg = ServerConfiguration(server.url_ws, "https://play.pokemonshowdown.com/action.php?")
    barrier = BatchBarrier(run_batch, deadline_ms=50)
    me = FlyCoachPlayer(provider=RandomProvider(2), coach=Coach(), barrier=barrier,
                        log_path=tmp_path / "barrier.jsonl",
                        account_configuration=AccountConfiguration("fm-test-bar", None), battle_format="gen1ou",
                        server_configuration=cfg, team=team_export(POOL[:6]), max_concurrent_battles=1)
    opp = RandomPlayer(account_configuration=AccountConfiguration("fm-test-bar-opp", None), battle_format="gen1ou",
                       server_configuration=cfg, team=team_export(POOL[6:12]))
    try:
        await me.battle_against(opp, n_battles=2)
        assert me.n_finished_battles == 2
        assert all(me.battle_stats(t)["finished"] for t in me.battles)
        assert len(calls) >= 1
        recs = [json.loads(line) for line in (tmp_path / "barrier.jsonl").read_text().splitlines()]
        fly = [r for r in recs if r["kind"] == "decision" and r["decider"] == "fly"]
        assert fly, "the fly never decided a turn"
        assert all(r["chosen"] == r["candidates"][0] for r in fly)
    finally:
        await me.ps_client.stop_listening()
        await opp.ps_client.stop_listening()


async def test_out_of_range_provider_index_is_rejected(make_battle):
    """M3 feeds the index from the swarm: an out-of-range one must fail loudly, not IndexError."""

    class OutOfRangeProvider:
        async def decide(self, battle, cands, ctx):
            return 99

    player = FlyCoachPlayer(provider=OutOfRangeProvider(), coach=Coach(), barrier=None,
                            account_configuration=AccountConfiguration("fm-offline-oob", None),
                            battle_format="gen1ou", start_listening=False)
    battle = make_battle(by_species["Blastoise"], "Charizard")
    with pytest.raises(ValueError, match="index 99 for 3 candidates"):
        await player.choose_move(battle)
