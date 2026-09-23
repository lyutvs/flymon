"""scripts/live_battles.py end to end: one fake-brain battle reaches a real viewer server over HTTP."""
import importlib.util
from pathlib import Path

import pytest
from poke_env.ps_client import ServerConfiguration

from flymon.battle.server import NODE_BIN, ShowdownServer
from flymon.live.server import ViewerServer
from flymon.live.sink import HttpEventSink

_spec = importlib.util.spec_from_file_location("live_battles", Path("scripts/live_battles.py"))
lb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lb)


def test_parse_args_defaults_and_names():
    a = lb.parse_args([])
    assert (a.flies, a.battles, a.provider, a.watch_fly, a.turn_delay, a.viewer) == (2, 2, "fake-brain", 0, 0.0, None)
    assert lb.fly_name(3) == "fm-live-f03"
    with pytest.raises(SystemExit):
        lb.parse_args(["--provider", "brain"])


@pytest.mark.skipif(not NODE_BIN.exists(), reason="run scripts/install_showdown.sh first")
async def test_one_fake_brain_battle_reaches_the_viewer(tmp_path):
    args = lb.parse_args(["--flies", "1", "--battles", "1", "--provider", "fake-brain", "--opponent", "random",
                          "--log-dir", str(tmp_path)])
    with ViewerServer(port=0) as viewer, ShowdownServer(port=lb.free_port()) as srv:
        sink = HttpEventSink(viewer.url)
        cfg = ServerConfiguration(srv.url_ws, "https://play.pokemonshowdown.com/action.php?")
        (result,) = await lb.run_battles(args, cfg, sink)
        sink.close(timeout_s=10)
        events = viewer.hub.state()["events"]
    assert result == {"fly": "fm-live-f00", "battles": 1, "emit_errors": 0}
    assert sink.dropped == 0 and sink.sent == len(events)
    assert {e["type"] for e in events} == {"battle_start", "protocol", "decision", "outcome", "trace", "battle_end"}
    fly_decisions = [e for e in events if e["type"] == "decision" and e["decider"] == "fly"]
    assert fly_decisions and all(len(e["detail"]["V"]) == len(e["candidates"]) for e in fly_decisions)
    assert (tmp_path / "fly00.jsonl").exists()
