"""M3 smoke (spec 5 M3: F=2, 2 battles each) on the C3 engine through the local Showdown server. Infrastructure only
(appendix L.7): allow-all screen table, no judgement. Outputs under results/m3/.

    uv run python scripts/run_m3_smoke.py --flies 2 --battles 2 --workers 2 --out results/m3/smoke/<run_id>
    uv run python scripts/run_m3_smoke.py ... --stop-after 1      # then rerun with --resume
"""
from __future__ import annotations

import argparse
import asyncio
import json
import socket
import sys
from pathlib import Path

from poke_env.ps_client import AccountConfiguration, ServerConfiguration

from flymon.agent.checkpoint import CheckpointStore
from flymon.agent.config import config_hash, load_c3_config
from flymon.agent.encode import Encoder
from flymon.agent.player import AgentPlayer
from flymon.agent.runner import run_cohort
from flymon.agent.screen import ScreenTable
from flymon.agent.swarm import BrainSwarm
from flymon.battle.barrier import BatchBarrier
from flymon.battle.coach import Coach
from flymon.battle.opponents import make_opponent
from flymon.battle.pool import by_species, team_export
from flymon.battle.schedule import make_schedule
from flymon.battle.server import ShowdownServer
from flymon.brain.circuits import Populations
from flymon.brain.connectome import Connectome
from flymon.brain.fly_pool import FlyPool, FlySpec
from flymon.brain.h4_jobs import type_cells

NPZ = "data/malecns.npz"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flies", type=int, default=2)
    ap.add_argument("--battles", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stop-after", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if not Path(a.out).resolve().is_relative_to(Path("results/m3").resolve()):
        ap.error("--out must be under results/m3/")
    return a


def _team(species):
    return team_export([by_species[s] for s in species])


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); return s.getsockname()[1]


async def main_async(a, cfg, pool, swarm, enc, srv_cfg) -> dict:
    out = Path(a.out)
    store = CheckpointStore(out / "checkpoints", config_hash(cfg))
    if not a.resume and store.load() is not None:
        raise SystemExit(f"{out} already has checkpoints; pass --resume")
    ck = store.load()   # a ValueError (no usable generation, config-hash mismatch) is fatal
    if ck:
        pool.load_state(ck["pool_state"])   # before run_cohort takes its per-fly committed snapshot
    sched = make_schedule(a.flies, a.battles, "heuristic", seed=a.seed)
    dbar = BatchBarrier(swarm.decide_run_batch)
    rbar = BatchBarrier(swarm.reinforce_run_batch)
    players = {}

    async def play_one(sb):
        p = players.get(sb.fly_id)
        if p is None:
            p = players[sb.fly_id] = AgentPlayer(
                fly=sb.fly_id, encoder=enc, table=ScreenTable.allow_all(), rbarrier=rbar, coach=Coach(), barrier=dbar,
                log_path=out / "logs" / f"fly{sb.fly_id:02d}.jsonl",
                account_configuration=AccountConfiguration(f"fm-m3-f{sb.fly_id:02d}", None), battle_format="gen1ou",
                server_configuration=srv_cfg, team=_team(sb.my_team), max_concurrent_battles=1)
        opp = make_opponent(sb.opponent, sb.fly_id, srv_cfg, _team(sb.opp_team))
        p.update_team(_team(sb.my_team))
        p.start_battle(sb.battle_id, int(sb.battle_id[-3:]))
        before = set(p.battles)
        try:
            await p.battle_against(opp, n_battles=1)
            await p.drain()
            for tag in set(p.battles) - before:
                p.write_battle_summary(tag, swarm.compartment_fracs(sb.fly_id))
        finally:
            await opp.ps_client.stop_listening()

    res = await run_cohort(sched, store, out / "logs", play_one, pool.state, stop_after=a.stop_after)
    for p in players.values():
        await p.ps_client.stop_listening()
    return res


def main(argv=None) -> int:
    a = parse_args(argv)
    cfg = load_c3_config()
    conn = Connectome.load(NPZ); pops = Populations.from_connectome(conn)
    enc = Encoder(pops)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    with FlyPool(NPZ, cfg.params, [FlySpec() for _ in range(a.flies)], workers=a.workers, timeout_s=7200) as pool:
        swarm = BrainSwarm(pool, cfg, type_cells(conn, [cfg.readout["A"], cfg.readout["P"]]), pops.kc)
        try:
            with ShowdownServer(port=_port()) as srv:
                srv_cfg = ServerConfiguration(srv.url_ws, "https://play.pokemonshowdown.com/action.php?")
                res = asyncio.run(main_async(a, cfg, pool, swarm, enc, srv_cfg))
        finally:
            swarm._exec.shutdown(wait=True)   # BrainSwarm has no close(); its executor thread must not outlive the pool
    (Path(a.out) / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
