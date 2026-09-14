#!/usr/bin/env python
"""No-brain M1 pilot: run one arm (RND / MAX / WEAK-RND / WEAK-MAX), or judge the M1 gate.

One arm per process. Flies run concurrently; each fly plays its scheduled battles in order
against a fresh team, reusing one player pair via `update_team`.

    uv run python scripts/pilot_no_brain.py --arm RND --flies 16 --battles 100
    uv run python scripts/pilot_no_brain.py --gate        # MAX - RND >= 0.15 ?
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from poke_env.ps_client import AccountConfiguration, ServerConfiguration

from flymon.battle.coach import Coach
from flymon.battle.fly_coach_player import FlyCoachPlayer
from flymon.battle.opponents import make_opponent
from flymon.battle.pool import by_species, team_export
from flymon.battle.providers import MaxDamageProvider, RandomProvider
from flymon.battle.schedule import load, make_schedule, save
from flymon.battle.server import ShowdownServer

ARMS = {"RND": ("random", False), "MAX": ("max", False), "WEAK-RND": ("random", True), "WEAK-MAX": ("max", True)}
GATE_THRESHOLD = 0.15
FORMAT = "gen1ou"


def _team(species: list[str]) -> str:
    return team_export([by_species[s] for s in species])


def _mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


# ---- one arm --------------------------------------------------------------------------------


async def run_fly(arm: str, fly_id: int, sbs: list, cfg: ServerConfiguration, log_dir: Path) -> list[dict]:
    provider_kind, weak = ARMS[arm]
    provider = RandomProvider(seed=fly_id) if provider_kind == "random" else MaxDamageProvider()
    tag = arm.lower().replace("_", "-")
    me = FlyCoachPlayer(
        provider=provider, coach=Coach(weak=weak), barrier=None,
        log_path=log_dir / f"fly{fly_id:02d}.jsonl",
        account_configuration=AccountConfiguration(f"fm-{tag}-f{fly_id:02d}", None),
        battle_format=FORMAT, server_configuration=cfg, team=_team(sbs[0].my_team), max_concurrent_battles=1)
    opp = make_opponent(sbs[0].opponent, fly_id, cfg, _team(sbs[0].opp_team))
    records = []
    try:
        for sb in sbs:
            if sb.opponent != sbs[0].opponent:
                raise ValueError(f"fly {fly_id} mixes opponent kinds: {sbs[0].opponent} then {sb.opponent}")
            me.update_team(_team(sb.my_team))
            opp.update_team(_team(sb.opp_team))
            seen = set(me.battles)
            await me.battle_against(opp, n_battles=1)
            new = [t for t in me.battles if t not in seen]
            if len(new) != 1:
                raise RuntimeError(f"expected exactly one new battle for {sb.battle_id}, got {new}")
            battle, stats = me.battles[new[0]], me.battle_stats(new[0])
            records.append({"battle_id": sb.battle_id, "fly_id": fly_id, "battle_tag": new[0],
                            "won": stats["won"], "turns": battle.turn, "fly_turns": stats["fly_turns"],
                            "coach_turns": stats["coach_turns"], "n_candidates_mean": me.n_candidates_mean(new[0])})
    finally:
        await me.ps_client.stop_listening()
        await opp.ps_client.stop_listening()
    return records


def _per_fly(records: list[dict]) -> list[dict]:
    out = []
    for fly_id in sorted({r["fly_id"] for r in records}):
        rs = [r for r in records if r["fly_id"] == fly_id]
        turns = sum(r["fly_turns"] + r["coach_turns"] for r in rs)
        fly_turns = sum(r["fly_turns"] for r in rs)
        out.append({"fly_id": fly_id, "wins": sum(bool(r["won"]) for r in rs), "battles": len(rs),
                    "fly_turn_fraction": fly_turns / turns if turns else 0.0,
                    "n_candidates_mean": _mean([r["n_candidates_mean"] for r in rs if r["fly_turns"]])})
    return out


async def run_arm(args) -> dict:
    arm = args.arm
    out_dir = Path(args.out_dir)
    log_dir = out_dir / "logs" / arm
    log_dir.mkdir(parents=True, exist_ok=True)
    sched_path = Path(args.schedule) if args.schedule else out_dir / "schedule.json"
    if sched_path.exists():
        sched = load(sched_path)
    else:
        sched = make_schedule(args.flies, args.battles, "heuristic", seed=0)
        save(sched_path, sched)
    by_fly = {f: [s for s in sched if s.fly_id == f] for f in sorted({s.fly_id for s in sched})}
    cfg = ServerConfiguration(f"ws://127.0.0.1:{args.port}/showdown/websocket",
                              "https://play.pokemonshowdown.com/action.php?")
    t0 = time.time()
    results = await asyncio.gather(*(run_fly(arm, f, sbs, cfg, log_dir) for f, sbs in by_fly.items()))
    wall = time.time() - t0
    records = [r for rs in results for r in rs]
    per_fly = _per_fly(records)
    payload = {
        "arm": arm, "provider": ARMS[arm][0], "weak_coach": ARMS[arm][1],
        "n_flies": len(by_fly), "n_battles": len(records),
        "win_rate": _mean([float(bool(r["won"])) for r in records]),
        "per_fly": per_fly, "battles": records,
        "schedule_path": str(sched_path), "log_dir": str(log_dir), "wall_clock_s": round(wall, 2),
    }
    path = out_dir / f"pilot_{arm}.json"
    path.write_text(json.dumps(payload, indent=1))
    print(f"{arm}: win_rate={payload['win_rate']:.3f} battles={payload['n_battles']} "
          f"flies={payload['n_flies']} wall={wall:.1f}s -> {path}")
    return payload


# ---- gate -----------------------------------------------------------------------------------


def _arm_summary(payload: dict) -> dict:
    records, per_fly = payload["battles"], payload["per_fly"]
    turns = sum(r["fly_turns"] + r["coach_turns"] for r in records)
    rates = [p["wins"] / p["battles"] if p["battles"] else 0.0 for p in per_fly]
    return {"win_rate": payload["win_rate"], "n_battles": payload["n_battles"], "n_flies": payload["n_flies"],
            "per_fly_win_rates": rates, "sd_across_flies": float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0,
            "fly_turn_fraction": sum(r["fly_turns"] for r in records) / turns if turns else 0.0,
            "n_candidates_mean": _mean([r["n_candidates_mean"] for r in records if r["fly_turns"]])}


def _bootstrap_ci(max_rates: list[float], rnd_rates: list[float], n: int = 2000, seed: int = 0) -> list | None:
    if not max_rates or not rnd_rates:
        return None
    rng = np.random.default_rng(seed)
    a, b = np.asarray(max_rates), np.asarray(rnd_rates)
    gaps = [rng.choice(a, len(a), replace=True).mean() - rng.choice(b, len(b), replace=True).mean()
            for _ in range(n)]
    return [float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))]


def run_gate(args) -> dict:
    out_dir, summary_path = Path(args.out_dir), Path(args.summary)
    arms, schedule_path = {}, None
    for arm in ARMS:
        path = out_dir / f"pilot_{arm}.json"
        if not path.exists():
            print(f"missing arm file: {path}")
            continue
        payload = json.loads(path.read_text())
        arms[arm] = _arm_summary(payload)
        schedule_path = payload.get("schedule_path", schedule_path)
    gap = (arms["MAX"]["win_rate"] - arms["RND"]["win_rate"]) if {"MAX", "RND"} <= set(arms) else None
    ci = (_bootstrap_ci(arms["MAX"]["per_fly_win_rates"], arms["RND"]["per_fly_win_rates"])
          if {"MAX", "RND"} <= set(arms) else None)
    summary = {"arms": arms, "gap_max_minus_rnd": gap, "gap_ci95": ci,
               "gate_ok": bool(gap is not None and gap >= GATE_THRESHOLD),
               "schedule_path": str(schedule_path) if schedule_path else None,
               "generated_at": datetime.now(timezone.utc).isoformat()}
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=1))
    if schedule_path and Path(schedule_path).exists():
        shutil.copyfile(schedule_path, summary_path.parent / "schedule_m1.json")
    print(f"gap(MAX-RND)={gap} ci95={ci} gate_ok={summary['gate_ok']} -> {summary_path}")
    return summary


# ---- cli ------------------------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=sorted(ARMS))
    p.add_argument("--gate", action="store_true", help="judge MAX - RND >= 0.15 from the four arm files")
    p.add_argument("--flies", type=int, default=16)
    p.add_argument("--battles", type=int, default=100)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--server", choices=("start", "existing"), default="start")
    p.add_argument("--schedule", default=None, help="schedule JSON (default: <out-dir>/schedule.json)")
    p.add_argument("--out-dir", default="results/m1")
    p.add_argument("--summary", default="results/summary/m1_pilot.json")
    args = p.parse_args(argv)
    if not args.gate and not args.arm:
        p.error("give --arm ARM or --gate")
    if args.arm:
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        server = ShowdownServer(port=args.port) if args.server == "start" else None
        if server is not None:
            server.start()
        try:
            asyncio.run(run_arm(args))
        finally:
            if server is not None:
                server.stop()
    if args.gate:
        run_gate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
