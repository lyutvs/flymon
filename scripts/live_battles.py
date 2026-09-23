#!/usr/bin/env python
"""Brain-free battles streamed to the live viewer (not an experiment: nothing here is a result).

    uv run python scripts/live_viewer.py                                   # terminal 1
    uv run python scripts/live_battles.py --flies 2 --battles 2 --provider fake-brain \\
        --watch-fly 0 --turn-delay 2 --viewer http://127.0.0.1:8765        # terminal 2, open the printed URL

Only the watched fly waits `--turn-delay` seconds per order. Without `--viewer` events go nowhere.
"""
from __future__ import annotations

import argparse
import asyncio
import socket
import sys
from pathlib import Path

from poke_env.ps_client import AccountConfiguration, ServerConfiguration

from flymon.battle.coach import Coach
from flymon.battle.fly_coach_player import FlyCoachPlayer
from flymon.battle.opponents import make_opponent
from flymon.battle.pool import by_species, team_export
from flymon.battle.providers import MaxDamageProvider, RandomProvider
from flymon.battle.schedule import make_schedule
from flymon.battle.server import ShowdownServer
from flymon.live.fake_brain import FakeBrainProvider
from flymon.live.sink import HttpEventSink, NullSink

PROVIDERS = ("rnd", "max", "fake-brain")


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flies", type=int, default=2)
    ap.add_argument("--battles", type=int, default=2)
    ap.add_argument("--provider", choices=PROVIDERS, default="fake-brain")
    ap.add_argument("--opponent", choices=("heuristic", "random"), default="heuristic")
    ap.add_argument("--watch-fly", type=int, default=0)
    ap.add_argument("--turn-delay", type=float, default=0.0)
    ap.add_argument("--viewer", default=None, help="viewer base URL, e.g. http://127.0.0.1:8765")
    ap.add_argument("--showdown-port", type=int, default=0, help="0 = any free port")
    ap.add_argument("--log-dir", default=None, help="write per-fly JSONL here (default: no log)")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def fly_name(fly_id: int) -> str:
    return f"fm-live-f{fly_id:02d}"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _team(species: list[str]) -> str:
    return team_export([by_species[s] for s in species])


def make_provider(kind: str, fly_id: int, sink):
    if kind == "rnd":
        return RandomProvider(seed=fly_id)
    if kind == "max":
        return MaxDamageProvider()
    return FakeBrainProvider(RandomProvider(seed=fly_id), sink, fly_name(fly_id), seed=fly_id)


async def run_fly(args, fly_id: int, sbs: list, cfg: ServerConfiguration, sink) -> dict:
    me = FlyCoachPlayer(provider=make_provider(args.provider, fly_id, sink), coach=Coach(), barrier=None,
                        log_path=Path(args.log_dir) / f"fly{fly_id:02d}.jsonl" if args.log_dir else None,
                        event_sink=sink, turn_delay_s=args.turn_delay if fly_id == args.watch_fly else 0.0,
                        account_configuration=AccountConfiguration(fly_name(fly_id), None), battle_format="gen1ou",
                        server_configuration=cfg, team=_team(sbs[0].my_team), max_concurrent_battles=1)
    opp = make_opponent(sbs[0].opponent, fly_id, cfg, _team(sbs[0].opp_team))
    try:
        for sb in sbs:
            me.update_team(_team(sb.my_team))
            opp.update_team(_team(sb.opp_team))
            await me.battle_against(opp, n_battles=1)
    finally:
        await me.ps_client.stop_listening()
        await opp.ps_client.stop_listening()
    return {"fly": fly_name(fly_id), "battles": me.n_finished_battles, "emit_errors": me.emit_errors}


async def run_battles(args, cfg: ServerConfiguration, sink) -> list:
    sched = make_schedule(args.flies, args.battles, args.opponent, seed=args.seed)
    by_fly = {f: [s for s in sched if s.fly_id == f] for f in range(args.flies)}
    return await asyncio.gather(*(run_fly(args, f, sbs, cfg, sink) for f, sbs in by_fly.items()),
                                return_exceptions=True)


def main(argv=None) -> int:
    args = parse_args(argv)
    sink = HttpEventSink(args.viewer) if args.viewer else NullSink()
    port = args.showdown_port or free_port()
    with ShowdownServer(port=port) as srv:
        cfg = ServerConfiguration(srv.url_ws, "https://play.pokemonshowdown.com/action.php?")
        if args.viewer:
            print(f"open: {args.viewer.rstrip('/')}/?fly={fly_name(args.watch_fly)}", flush=True)
        results = asyncio.run(run_battles(args, cfg, sink))
    sink.close(timeout_s=10)
    failed = [r for r in results if isinstance(r, BaseException)]
    for r in results:
        print(f"fly failed: {r!r}" if isinstance(r, BaseException) else
              f"{r['fly']}: battles={r['battles']} emit_errors={r['emit_errors']}", flush=True)
    print(f"dropped_events={sink.dropped}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
