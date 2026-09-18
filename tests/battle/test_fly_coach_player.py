"""FlyCoachPlayer: real gen1ou battles against a local Showdown server, plus one offline case."""
import json
import time

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
from flymon.live.sink import RecordingSink

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


# ---- live viewer events -----------------------------------------------------------------------
def _record(event):
    """A viewer event rewritten as the JSONL record it mirrors."""
    rec = {"battle_tag": event["battle_tag"], "turn": event["turn"], "kind": event["type"]}
    rec.update({k: v for k, v in event.items() if k not in ("v", "fly", "type", "battle_tag", "turn", "detail")})
    return rec


@requires_server
async def test_events_mirror_the_jsonl_log_between_start_and_end(server, tmp_path):
    cfg = ServerConfiguration(server.url_ws, "https://play.pokemonshowdown.com/action.php?")
    sink = RecordingSink()
    me = FlyCoachPlayer(provider=RandomProvider(3), coach=Coach(), barrier=None, log_path=tmp_path / "ev.jsonl",
                        event_sink=sink, account_configuration=AccountConfiguration("fm-test-ev", None),
                        battle_format="gen1ou", server_configuration=cfg, team=team_export(POOL[:6]),
                        max_concurrent_battles=1)
    opp = RandomPlayer(account_configuration=AccountConfiguration("fm-test-ev-opp", None), battle_format="gen1ou",
                       server_configuration=cfg, team=team_export(POOL[6:12]))
    try:
        await me.battle_against(opp, n_battles=2)
        recs = [json.loads(line) for line in (tmp_path / "ev.jsonl").read_text().splitlines()]
        evs = sink.events
        assert all(e["v"] == 1 and e["fly"] == "fm-test-ev" for e in evs)
        assert [_record(e) for e in evs if e["type"] in ("decision", "outcome")] == recs
        assert me.emit_errors == 0
        for tag, battle in me.battles.items():
            mine = [e for e in evs if e["battle_tag"] == tag]
            types = [e["type"] for e in mine]
            assert types.count("battle_start") == 1 and types.count("battle_end") == 1
            assert mine[0]["type"] == "battle_start"
            end = types.index("battle_end")                     # the server may still send protocol after it
            assert not ({"decision", "outcome", "trace"} & set(types[end + 1:]))
            assert mine[end]["won"] == battle.won and mine[end]["turns"] == mine[end]["turn"] == battle.turn
            assert all("detail" not in e for e in mine)           # RandomProvider reports no detail
    finally:
        await me.ps_client.stop_listening()
        await opp.ps_client.stop_listening()


class _DetailProvider:
    async def decide(self, battle, cands, ctx):
        ctx["detail"] = {"V": [0.1 * i for i in range(len(cands))]}
        return len(cands) - 1


def _offline(name, **kw):
    return FlyCoachPlayer(coach=Coach(), account_configuration=AccountConfiguration(name, None),
                          battle_format="gen1ou", start_listening=False, **kw)


async def test_without_a_sink_no_event_is_built(make_battle, monkeypatch):
    import flymon.battle.fly_coach_player as fcp

    def boom(*a, **k):
        raise AssertionError("make_event called without a sink")

    monkeypatch.setattr(fcp, "make_event", boom)
    player = _offline("fm-offline-nosink", provider=_DetailProvider())
    assert player.event_sink is None and player.turn_delay_s == 0.0
    await player.choose_move(make_battle(by_species["Blastoise"], "Charizard"))
    assert player.emit_errors == 0


async def test_provider_detail_rides_on_the_decision_event(make_battle):
    sink = RecordingSink()
    player = _offline("fm-offline-detail", provider=_DetailProvider(), event_sink=sink)
    battle = make_battle(by_species["Blastoise"], "Charizard")
    await player.choose_move(battle)
    (ev,) = sink.events
    assert ev["type"] == "decision" and ev["decider"] == "fly" and len(ev["candidates"]) == 3
    assert ev["detail"] == {"V": [0.0, 0.1, 0.2]} and ev["chosen"] == ev["candidates"][2]


async def test_detail_through_the_barrier(make_battle):
    async def run_batch(reqs):
        for r in reqs:
            r.context["detail"] = {"p": [1.0, 0.0, 0.0]}
        return [0] * len(reqs)

    sink = RecordingSink()
    player = _offline("fm-offline-bar", provider=RandomProvider(0), barrier=BatchBarrier(run_batch, deadline_ms=50),
                      event_sink=sink)
    await player.choose_move(make_battle(by_species["Blastoise"], "Charizard"))
    (ev,) = sink.events
    assert ev["detail"] == {"p": [1.0, 0.0, 0.0]} and ev["chosen"] == ev["candidates"][0]


async def test_a_raising_sink_does_not_stop_the_turn(make_battle):
    class Broken:
        def emit(self, event):
            raise RuntimeError("viewer exploded")

    player = _offline("fm-offline-broken", provider=_DetailProvider(), event_sink=Broken())
    order = await player.choose_move(make_battle(by_species["Blastoise"], "Charizard"))
    assert order.message.startswith("/choose move ")
    assert player.emit_errors == 1


async def test_turn_delay_holds_the_order_back_after_the_event(make_battle):
    sink = RecordingSink()
    player = _offline("fm-offline-delay", provider=_DetailProvider(), event_sink=sink, turn_delay_s=0.2)
    t0 = time.perf_counter()
    await player.choose_move(make_battle(by_species["Blastoise"], "Charizard"))
    assert time.perf_counter() - t0 >= 0.2
    assert [e["type"] for e in sink.events] == ["decision"]


@requires_server
async def test_protocol_events_carry_the_battle_for_the_renderer(server, tmp_path):
    """The page replays these lines in Showdown's own renderer, so they must be the raw protocol minus `|request|`."""
    cfg = ServerConfiguration(server.url_ws, "https://play.pokemonshowdown.com/action.php?")
    sink = RecordingSink()
    me = FlyCoachPlayer(provider=RandomProvider(4), coach=Coach(), barrier=None, event_sink=sink,
                        account_configuration=AccountConfiguration("fm-test-proto", None), battle_format="gen1ou",
                        server_configuration=cfg, team=team_export(POOL[:6]), max_concurrent_battles=1)
    opp = RandomPlayer(account_configuration=AccountConfiguration("fm-test-proto-opp", None), battle_format="gen1ou",
                       server_configuration=cfg, team=team_export(POOL[6:12]))
    try:
        await me.battle_against(opp, n_battles=1)
        (tag,) = list(me.battles)
        lines = [line for e in sink.events if e["type"] == "protocol" and e["battle_tag"] == tag for line in e["lines"]]
        assert lines[0] == "|init|battle"
        assert "|turn|1" in lines
        assert any(line.startswith("|move|") for line in lines)
        assert any(line.startswith("|win|") or line.startswith("|tie") for line in lines)
        assert all(not line.startswith("|request|") for line in lines)
        assert all(line.startswith("|") for line in lines)
        first_start = next(i for i, e in enumerate(sink.events) if e["type"] == "battle_start")
        first_proto = next(i for i, e in enumerate(sink.events) if e["type"] == "protocol")
        assert first_start < first_proto
    finally:
        await me.ps_client.stop_listening()
        await opp.ps_client.stop_listening()
