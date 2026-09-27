#!/usr/bin/env python3
"""Re-scoped claim, stages 4-5 (spec 4.5, 10.7, 10.8): one M4 arm's battles through a local Showdown server.

    uv run python scripts/run_rescope_battles.py --phase pilot --arm FLY --flies 6 --eval 20 --learn 40 \\
        --out results/rescope/pilot/FLY
    uv run python scripts/run_rescope_battles.py --smoke --phase pilot --arm RS --out results/rescope-smoke/pilot/RS
    ... --stop-after N   then rerun with --resume

Arms: FLY (learning block with its own pulses, then the evaluation block), RS (learning block with FLY k's pulse
queue, yoked; only after FLY's whole arm is complete, then the evaluation block), COFF (plasticity off, evaluation
block only), RND / MAX (FlyCoachPlayer with RandomProvider(seed=fly) / MaxDamageProvider, evaluation block only).
Evaluation block: swarm mode "eval" (argmax), plasticity off for every fly, no pulse delivered; per-fly weight sha256
before == after is recorded (weights_bit_identical_across_eval) and a mismatch makes the fly INVALID.

Paths (derived from --out's parent, so smoke and real runs are self-contained): the eval schedule
<parent>/eval_schedule.json ({digest, schedule}, created by the first arm, digest checked by every other arm) and the
RS donor <parent>/FLY (its result.json, i.e. FLY's learning and evaluation complete, and logs/flyNN.jsonl).
Schedules: block ids L-fNN-bNNN / E-fNN-bNNN from SPEC.schedule_seeds[phase]; learn and eval must not share a
(my_team, opp_team) pair. Brain arms read recovery_per_pulse from results/summary/rescope_taurec.json
(--smoke: results/rescope-smoke/summary/rescope_taurec.json; status SELECTED). --phase judge refuses unless
rescope_power.json (same smoke rule) has status SIZED and F / E equal --flies / --eval.
All refusals run before any pool or server starts. Server errors: an unfinished battle is replayed with the same
entry up to SPEC.retry_max times (counted per battle), beyond that the fly is INVALID. --smoke forces --flies 2
--learn 2 --eval 2 --workers 2 and an --out under results/rescope-smoke/<phase>/.
Writes <out>/result.json only once the arm is complete: {phase, arm, flies, learn, eval, recovery_per_pulse,
schedule_digests, per_fly, wall_clock_s, m_overlap_note, provenance, ...}.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import glob
import json
import socket
import sys
import time
from pathlib import Path

from flymon.brain.config import Params
from flymon.rescope import blocks
from flymon.rescope.spec import SPEC
from flymon.rescope.store import git_provenance, guard, write_json

ARMS = ("FLY", "RS", "COFF", "RND", "MAX")
BRAIN = ("FLY", "RS", "COFF")
LEARNS = ("FLY", "RS")
CODE = {"FLY": "FLY", "RS": "RS", "COFF": "CO", "RND": "RN", "MAX": "MX"}
NPZ = "data/malecns.npz"
POOL_TIMEOUT_S = 7200


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=("pilot", "judge"), required=True)
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--flies", type=int, default=None)
    ap.add_argument("--eval", type=int, default=None)
    ap.add_argument("--learn", type=int, default=SPEC.learn_battles)
    ap.add_argument("--out", default=None, help="default results/rescope/<phase>/<arm>")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--stop-after", type=int, default=None, help="play N battles (both blocks) and stop (resume test)")
    ap.add_argument("--m-overlap-note", default=None, help="whether this run shared the CPU with M's runs (recorded)")
    ap.add_argument("--npz", default=NPZ)
    a = ap.parse_args(argv)
    if a.smoke:
        a.flies, a.learn, a.eval, a.workers = 2, 2, 2, 2
    if a.flies is None or a.eval is None:
        ap.error("--flies and --eval are required (or --smoke)")
    if a.flies < 1 or a.eval < 1 or a.learn < 1:
        ap.error("--flies, --eval and --learn must be >= 1")
    a.out = str(out_path(a.out, a.phase, a.arm, a.smoke))
    return a


def out_path(out, phase: str, arm: str, smoke: bool) -> Path:
    o = Path(out) if out else Path("results/rescope") / phase / arm
    if smoke and not o.as_posix().startswith("results/rescope-smoke/"):
        o = Path("results/rescope-smoke") / phase / (o.name or arm)
    return o


def summary_path(stage: str, smoke: bool) -> Path:
    return (Path("results/rescope-smoke/summary") if smoke else Path("results/summary")) / f"rescope_{stage}.json"


def schedule_seeds(phase: str, spec=SPEC) -> tuple:
    return {p: (l, e) for p, l, e in spec.schedule_seeds}[phase]


# ---- refusals (no pool, no server) ---------------------------------------------------------------
def check_power(a) -> dict:
    p = summary_path("power", a.smoke)
    if not p.exists():
        raise SystemExit(f"refusing --phase judge: {p} does not exist (size the judge run with the pilot first)")
    d = json.loads(p.read_text())
    if d.get("status") != "SIZED":
        raise SystemExit(f"refusing --phase judge: {p} status {d.get('status')!r} is not SIZED")
    if (d.get("F"), d.get("E")) != (a.flies, a.eval):
        raise SystemExit(f"refusing --phase judge: --flies/--eval {a.flies}/{a.eval} != sized F/E {d.get('F')}/{d.get('E')}")
    return dict(path=str(p), F=d["F"], E=d["E"], provenance=d.get("provenance"))


def load_taurec(smoke: bool) -> tuple:
    p = summary_path("taurec", smoke)
    if not p.exists():
        raise SystemExit(f"refusing: {p} does not exist (run scripts/run_rescope_taurec.py first)")
    d = json.loads(p.read_text())
    if d.get("status") != "SELECTED":
        raise SystemExit(f"refusing: {p} status {d.get('status')!r} is not SELECTED")
    return float(d["recovery_per_pulse"]), dict(path=str(p), provenance=d.get("provenance"))


def check_eval_schedule(path: Path, eval_) -> bool:
    """True if the file must be created; refuses a file whose digest does not match its schedule or this run's."""
    want = blocks.schedule_digest(eval_)
    if not path.exists():
        return True
    d = json.loads(path.read_text())
    try:
        got = blocks.schedule_digest(blocks.schedule_from_rows(d["schedule"]))
    except (KeyError, TypeError) as e:
        raise SystemExit(f"refusing: {path} is malformed ({e!r})")
    if d.get("digest") != got:
        raise SystemExit(f"refusing: {path} digest {str(d.get('digest'))[:12]} does not match its schedule ({got[:12]})")
    if got != want:
        raise SystemExit(f"refusing: {path} holds another eval schedule ({got[:12]}) than this run's ({want[:12]}); "
                         "every arm of a phase must use the same --flies / --eval")
    return False


def check_donor(donor: Path, a, digests: dict, r: float) -> dict:
    """RS k starts only after FLY's whole arm is complete: its result.json exists and matches this run."""
    res_p = donor / "result.json"
    if not res_p.exists():
        raise SystemExit(f"refusing RS: {res_p} does not exist (FLY's learning and evaluation blocks must be complete)")
    d = json.loads(res_p.read_text())
    want = dict(arm="FLY", phase=a.phase, flies=a.flies, learn=a.learn, eval=a.eval, recovery_per_pulse=r,
                complete=True)
    bad = {k: (d.get(k), v) for k, v in want.items() if d.get(k) != v}
    if d.get("schedule_digests") != digests:
        bad["schedule_digests"] = (d.get("schedule_digests"), digests)
    if bad:
        raise SystemExit(f"refusing RS: donor {res_p} does not match this run: {bad}")
    rows = {row["fly"]: row for row in d["per_fly"]}
    for k in range(a.flies):
        log = donor / "logs" / f"fly{k:02d}.jsonl"
        if not log.exists():
            raise SystemExit(f"refusing RS: donor log {log} does not exist")
        sha = blocks.file_sha(log)
        if rows.get(k, {}).get("learn_log_sha256") != sha:
            raise SystemExit(f"refusing RS: donor log {log} changed after FLY finished (sha256 {sha[:12]})")
    return d


# ---- Showdown plumbing (as scripts/run_m3_smoke.py) -----------------------------------------------
def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); return s.getsockname()[1]


def account(arm: str, block: str, fly: int) -> str:
    name = f"fm-r{CODE[arm]}{block}-f{fly:02d}"
    if len(name) > 18:
        raise ValueError(f"account {name!r} exceeds Showdown's 18 characters")
    return name


def opponent_name(battle_id: str, attempt: int) -> str:
    """One account per battle attempt: reusing a name can hit `nametaken` while the server still holds the login."""
    name = f"fm-h-{battle_id}" + (f"-{attempt}" if attempt else "")
    if len(name) > 18:
        raise ValueError(f"opponent name {name!r} exceeds Showdown's 18 characters")
    return name


async def on_player_loop(player, coro):
    """Await a player coroutine on the player's own loop (run_m3_smoke.on_player_loop)."""
    from poke_env.concurrency import handle_threaded_coroutines
    return await handle_threaded_coroutines(coro, player.ps_client.loop)


class Battles:
    """Per-block players and the attempt callable run_arm drives."""

    def __init__(self, a, out: Path, srv_cfg, swarm=None, enc=None, yoke=None):
        from flymon.battle.barrier import BatchBarrier
        self.a, self.out, self.srv_cfg, self.swarm, self.enc, self.yoke = a, out, srv_cfg, swarm, enc, yoke
        self.players = {"L": {}, "E": {}}
        self.bars = {}
        if swarm is not None:
            self.bars["L"] = (BatchBarrier(swarm.decide_run_batch), BatchBarrier(swarm.reinforce_run_batch))
            self.bars["E"] = (BatchBarrier(swarm.decide_run_batch), BatchBarrier(blocks.refuse_pulses))

    @staticmethod
    def team(species):
        from flymon.battle.pool import by_species, team_export
        return team_export([by_species[s] for s in species])

    def player(self, block: str, sb):
        p = self.players[block].get(sb.fly_id)
        if p is not None:
            return p
        from poke_env.ps_client import AccountConfiguration
        from flymon.agent.player import AgentPlayer
        from flymon.agent.screen import ScreenTable
        from flymon.battle.coach import Coach
        from flymon.battle.providers import MaxDamageProvider, RandomProvider
        from flymon.rescope.players import YokedPlayer
        f, arm = sb.fly_id, self.a.arm
        logs = self.out / "logs" if block == "L" else self.out / "logs" / "eval"
        common = dict(log_path=logs / f"fly{f:02d}.jsonl", coach=Coach(),
                      account_configuration=AccountConfiguration(account(arm, block, f), None), battle_format="gen1ou",
                      server_configuration=self.srv_cfg, team=self.team(sb.my_team), max_concurrent_battles=1)
        if arm in ("RND", "MAX"):
            prov = RandomProvider(seed=f) if arm == "RND" else MaxDamageProvider()
            p = blocks.NoBrainPlayer(f, provider=prov, barrier=None, **common)
        else:
            dbar, rbar = self.bars[block]
            brain = dict(fly=f, encoder=self.enc, table=ScreenTable.allow_all(), rbarrier=rbar, barrier=dbar, **common)
            if block == "E":
                p = blocks.EvalPlayer(**brain)
            elif arm == "RS":
                p = YokedPlayer(queue=self.yoke.queues[f], **brain)
            else:
                p = AgentPlayer(**brain)
        self.players[block][f] = p
        return p

    def reset_player(self, block: str, fly: int) -> None:
        """After an unfinished attempt: its queued (undelivered) pulses and pending choice must not carry over."""
        p = self.players[block].get(fly)
        if p is not None and hasattr(p, "_queue"):
            p._queue.clear()
            p._choice.clear()

    def attempt_for(self, block: str):
        async def attempt(sb, n):
            from poke_env.ps_client import AccountConfiguration
            from flymon.battle.opponents import KINDS
            p = self.player(block, sb)
            opp = KINDS[sb.opponent](account_configuration=AccountConfiguration(opponent_name(sb.battle_id, n), None),
                                     battle_format="gen1ou", server_configuration=self.srv_cfg,
                                     team=self.team(sb.opp_team), max_concurrent_battles=1)
            p.update_team(self.team(sb.my_team))
            p.start_battle(sb.battle_id, blocks.battle_index(sb.battle_id))
            before = set(p.battles)
            brain = self.swarm is not None
            try:
                await p.battle_against(opp, n_battles=1)   # returns once a fatal choose_move forfeited the battle
                if getattr(p, "fatal", None) is not None:
                    raise p.fatal
                new = sorted(set(p.battles) - before)
                if len(new) != 1:
                    return {"finished": False, "won": None, "battle_tag": None, "new_battles": new}
                tag = new[0]
                st = p.battle_stats(tag)
                res = dict(st, battle_tag=tag, turns=int(p.battles[tag].turn))
                if not st["finished"]:
                    return res                              # rolled back by run_arm: nothing is drained or summarised
                if brain:
                    await on_player_loop(p, p.drain())      # the barrier lives on POKE_LOOP, not this loop
                    if p.fatal is not None:
                        raise p.fatal
                    p.write_battle_summary(tag, self.swarm.compartment_fracs(sb.fly_id))
                return res
            finally:
                await opp.ps_client.stop_listening()
        return attempt

    async def close_block(self, block: str) -> None:
        ps = self.players[block]
        while ps:
            _, p = ps.popitem()
            await p.ps_client.stop_listening()

    async def close(self):
        for ps in self.players.values():
            for p in ps.values():
                await p.ps_client.stop_listening()


# ---- main ------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    t0 = time.time()
    a = parse_args(argv)
    spec = SPEC
    out = Path(a.out)
    guard(out / "result.json", [Params()])
    parent = out.parent
    eval_file, donor_dir = parent / "eval_schedule.json", parent / "FLY"
    guard(eval_file, [Params()])
    if a.arm == "RS" and out.resolve() == donor_dir.resolve():
        raise SystemExit("refusing: RS --out would be the FLY donor directory")
    power = check_power(a) if a.phase == "judge" else None
    lseed, eseed = schedule_seeds(a.phase, spec)
    learn_all = blocks.block_schedule(a.flies, a.learn, lseed, "L")
    eval_ = blocks.block_schedule(a.flies, a.eval, eseed, "E")
    try:
        blocks.assert_disjoint(learn_all, eval_)
    except ValueError as e:
        raise SystemExit(f"refusing: {e}")
    learn = learn_all if a.arm in LEARNS else None
    digests = dict(learn=blocks.schedule_digest(learn) if learn else None, eval=blocks.schedule_digest(eval_))
    create_eval = check_eval_schedule(eval_file, eval_)
    r, taurec = load_taurec(a.smoke) if a.arm in BRAIN else (None, None)
    donor = None
    if a.arm == "RS":
        donor = check_donor(donor_dir, a, dict(digests), r)
    if (out / "result.json").exists() and not a.resume:
        raise SystemExit(f"refusing: {out / 'result.json'} exists (the arm is complete); pass --resume to rewrite it")
    git = git_provenance(files=sorted(glob.glob("flymon/rescope/*.py")) + ["scripts/run_rescope_battles.py"])
    if git["dirty"] and not a.allow_dirty:
        raise SystemExit(f"refusing: uncommitted changes in {git['dirty_files']} (commit them or pass --allow-dirty)")
    if create_eval:
        write_json(eval_file, dict(digest=digests["eval"], schedule=blocks.schedule_rows(eval_)), [Params()])

    import asyncio
    import numpy as np
    from poke_env.ps_client import ServerConfiguration
    from flymon.agent.config import config_hash, load_c3_config
    from flymon.battle.server import ShowdownServer

    cfg = load_c3_config()
    params = cfg.params if r is None else dataclasses.replace(cfg.params, recovery_per_pulse=r)
    cfg_r = dataclasses.replace(cfg, params=params)
    gp = [params]
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    prov = dict(git=git, started_utc=started, argv=list(sys.argv[1:] if argv is None else argv), npz=a.npz,
                taurec=taurec, power=power, eval_schedule=dict(path=str(eval_file), digest=digests["eval"]),
                donor=None if donor is None else dict(path=str(donor_dir), provenance=donor.get("provenance")))
    out.mkdir(parents=True, exist_ok=True)
    key = config_hash(cfg_r) if a.arm in BRAIN else hashlib_key(a.arm)

    def srv_config(srv):
        return ServerConfiguration(srv.url_ws, "https://play.pokemonshowdown.com/action.php?")

    yoke = None
    if a.arm == "RS":
        from flymon.rescope.yoke import YokedQueue
        yoke = blocks.YokeBook(out / "yoke_state.json", {
            k: YokedQueue.from_log(donor_dir / "logs" / f"fly{k:02d}.jsonl", blocks.ids_of(learn, k))
            for k in range(a.flies)})

    if a.arm in BRAIN:
        from flymon.agent.encode import Encoder
        from flymon.agent.swarm import BrainSwarm
        from flymon.brain.circuits import Populations
        from flymon.brain.connectome import Connectome
        from flymon.brain.fly_pool import FlyPool, FlySpec
        from flymon.brain.h4_jobs import type_cells
        conn = Connectome.load(a.npz); pops = Populations.from_connectome(conn)
        enc = Encoder(pops)
        on = a.arm != "COFF"
        with FlyPool(a.npz, params, [FlySpec(enabled=on) for _ in range(a.flies)], workers=a.workers,
                     timeout_s=POOL_TIMEOUT_S) as pool:
            swarm = BrainSwarm(pool, cfg_r, type_cells(conn, [cfg.readout["A"], cfg.readout["P"]]), pops.kc,
                               mode="learn" if on else "eval")
            try:
                with ShowdownServer(port=_port()) as srv:
                    run = asyncio.run(_drive(a, out, srv_config(srv), key, spec, learn, eval_, pool, swarm, enc, yoke))
            finally:
                swarm._exec.shutdown(wait=True)   # BrainSwarm has no close(); its executor must not outlive the pool
    else:
        with ShowdownServer(port=_port()) as srv:
            run = asyncio.run(_drive(a, out, srv_config(srv), key, spec, None, eval_, None, None, None, None))

    wall = _wall_clock(out, t0, run, gp)
    if not run["complete"]:
        print(f"{a.arm}: stopped before completion (played {sum(len(v) for v in run['played'].values())}); "
              "rerun with --resume", flush=True)
        return 0
    per_fly = blocks.arm_per_fly(out, n_flies=a.flies, eval_=eval_, run=run, learn=learn, yoke=yoke, spec=spec,
                                 brain=a.arm in BRAIN,
                                 extra=None if donor is None else {row["fly"]: dict(donor_invalid=row["invalid"])
                                                                   for row in donor["per_fly"]})
    frozen = all(row["weights_bit_identical_across_eval"] is not False for row in per_fly)
    res = dict(phase=a.phase, arm=a.arm, flies=a.flies, learn=a.learn if learn else 0, eval=a.eval,
               recovery_per_pulse=r, schedule_digests=digests, per_fly=per_fly, wall_clock_s=wall,
               m_overlap_note=a.m_overlap_note, complete=True, smoke=a.smoke, retry_max=spec.retry_max,
               residual_max=spec.residual_max, eval_weights_frozen=frozen, n_invalid=sum(row["invalid"] for row in per_fly),
               logs=dict(learn="logs" if learn else None, eval="logs/eval", retries="<logs>/retries"),
               provenance=dict(prov, finished_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
    write_json(out / "result.json", res, gp)
    wins = [np.mean([b["won"] is True for b in row["eval_battles"]]) for row in per_fly]
    print(f"{a.arm}: eval win rate {float(np.mean(wins)):.3f} over {a.flies} flies, invalid {res['n_invalid']}, "
          f"weights frozen in eval {frozen}; wrote {out / 'result.json'}", flush=True)
    if not frozen:
        print("SAFETY: weights changed during the evaluation block (spec 4.5); stop and report", file=sys.stderr)
        return 3
    return 0


def hashlib_key(arm: str) -> str:
    """Checkpoint key of a no-brain arm (no AgentConfig): the arm and its provider."""
    import hashlib
    return hashlib.sha256(f"rescope-nobrain-{arm}".encode()).hexdigest()


async def _drive(a, out, srv_cfg, key, spec, learn, eval_, pool, swarm, enc, yoke) -> dict:
    b = Battles(a, out, srv_cfg, swarm=swarm, enc=enc, yoke=yoke)
    try:
        return await blocks.run_arm(out, eval_=eval_, attempt_for=b.attempt_for, cfg_hash=key,
                                    retry_max=spec.retry_max, resume=a.resume, n_flies=a.flies, learn=learn,
                                    pool=pool, swarm=swarm, yoke=yoke, reset_player=b.reset_player,
                                    stop_after=a.stop_after, on_block_end=b.close_block)
    finally:
        await b.close()


def _wall_clock(out: Path, t0: float, run: dict, gp) -> float:
    """Wall-clock over every session of this arm (resumes add up): out/wall_clock.json."""
    p = out / "wall_clock.json"
    d = json.loads(p.read_text()) if p.exists() else {"sessions": []}
    d["sessions"].append(dict(seconds=round(time.time() - t0, 2),
                              played={k: len(v) for k, v in run["played"].items()},
                              finished_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
    write_json(p, d, gp)
    return round(sum(s["seconds"] for s in d["sessions"]), 2)


if __name__ == "__main__":
    sys.exit(main())
