#!/usr/bin/env python3
"""M4 stage 1 under spec appendix AC (AC.2, AC.4-AC.8): one arm group's battles through a local Showdown server.

    uv run python scripts/run_ac_m4.py --arm BRAIN    # FLY 12 + C-off 6 + FLY-TB 6 on one L_V LeverFlyPool
    uv run python scripts/run_ac_m4.py --arm RND      # 16 flies, the shared evaluation schedule only
    uv run python scripts/run_ac_m4.py --arm MAX
    ... --stop-after N (or a session cap), then the same command with --resume
    uv run python scripts/run_ac_m4.py --smoke --arm BRAIN    # AC.7 2b smoke under results/m4-smoke/
    uv run python scripts/run_ac_m4.py --bench --arm BRAIN    # AC.6 benchmark under results/m4-bench/

BRAIN: the learning block (40 battles, softmax, own pulses; C-off with plasticity off) with situation-pair evaluations
at 0 / 10 / 20 / 30 / 40 (C-off at 0), then the evaluation block (20 battles on the shared schedule, argmax with the
seeded tie rule, plasticity off, no pulse, weight sha checked). RND / MAX: the evaluation block only. Every arm checks
the experiment-input manifest; BRAIN also the model manifest and the worker's L_V sha. Battles failing after 3 retries
make the fly INVALID; an arm that can no longer reach its minimum of valid flies stops (STOP_INFRA). A session ends at
min(24 h, the 60 h budget minus stage 1's sessions) or --session-hours; rerun with --resume. Refusals run before any
pool or server starts (see the plan's Task 15)."""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import socket
import sys
import time
from pathlib import Path

from flymon.ac import budget, schedules, stage1, store
from flymon.ac.invalid import InvalidBook
from flymon.ac.manifest import MODEL_MANIFEST
from flymon.ac.spec import LABEL, SPEC, bench, smoke
from flymon.rescope import blocks

ARMS = ("BRAIN", "RND", "MAX")
NPZ = "data/malecns.npz"
POOL_TIMEOUT_S = 7200


def spec_for(a):
    return smoke() if a.smoke else bench() if a.bench else SPEC


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=ARMS, required=True)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--smoke", action="store_true")
    g.add_argument("--bench", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stop-after", type=int, default=None)
    ap.add_argument("--session-hours", type=float, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--npz", default=NPZ)
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args(argv)
    if a.bench and a.arm != "BRAIN":
        ap.error("--bench runs the brain group only")
    if a.bench and a.resume:
        ap.error("--bench is one fresh session")
    if a.smoke:
        a.workers = 4
    if a.bench:
        a.workers = SPEC.bench_workers
    a.out = str(Path(spec_for(a).out_root) / a.arm)
    return a


def load_inputs(spec) -> tuple:
    doc_p, man_p = schedules.input_paths(spec)
    for p in (doc_p, man_p):
        if not p.exists():
            raise SystemExit(f"refusing: {p} does not exist (scripts/write_ac_inputs.py"
                             f"{'' if spec.mode == 'run' else ' --' + spec.mode})")
    doc, man = json.loads(doc_p.read_text()), json.loads(man_p.read_text())
    if doc.get("status") != "OK":
        raise SystemExit(f"refusing: {doc_p} status {doc.get('status')!r} (STOP_SET stops AC)")
    bad = schedules.check_inputs(doc, man, spec, doc_p)
    if bad:
        raise SystemExit(f"refusing: {doc_p} does not match its seeds / manifest: {bad}")
    return doc, dict(path=str(doc_p), manifest=str(man_p), manifest_sha256=store.sha256_file(man_p),
                     digests=doc["digests"])


def load_model(spec) -> tuple:
    p = Path(MODEL_MANIFEST)
    if not p.exists():
        raise SystemExit(f"refusing: {p} does not exist (AC.7 2a: scripts/write_ac_model_manifest.py)")
    d = json.loads(p.read_text())
    if d.get("status") != "FROZEN":
        raise SystemExit(f"refusing: {p} status {d.get('status')!r} is not FROZEN")
    return d, float(d["recovery_per_pulse"])


def load_budget(spec):
    if spec.mode != "run":
        return None
    p = Path(budget.BUDGET)
    if not p.exists():
        raise SystemExit(f"refusing: {p} does not exist (AC.7 3: scripts/write_ac_budget.py)")
    d = json.loads(p.read_text())
    if d.get("status") != "OK":
        raise SystemExit(f"refusing: {p} status {d.get('status')!r} (STOP_BUDGET: stage 1 does not run)")
    return d


def session_cap(spec, a) -> float:
    spent = budget.spent_s(spec.out_root) if spec.mode == "run" else 0.0
    cap = budget.session_cap_s(spec, spent)
    if a.session_hours is not None:
        cap = min(cap, a.session_hours * 3600.0)
    if cap <= 0:
        raise SystemExit(f"refusing: the {spec.budget_total_h} h budget is spent ({spent / 3600:.1f} h, AC.6)")
    return cap


def fresh_bench(out) -> None:
    """The bench is one fresh session (--resume is refused): any leftover of an earlier attempt must be removed."""
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing: {out} holds an earlier bench attempt; the bench is one fresh session "
                         f"(--resume is refused): remove {out} and rerun")


def files() -> list:
    return sorted(glob.glob("flymon/ac/*.py")) + ["flymon/brain/lv_pool.py", "flymon/rescope/blocks.py",
                                                  "flymon/agent/runner.py", "flymon/agent/policy.py",
                                                  "scripts/run_ac_m4.py"]


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); return s.getsockname()[1]


def srv_config(srv):
    from poke_env.ps_client import ServerConfiguration
    return ServerConfiguration(srv.url_ws, "https://play.pokemonshowdown.com/action.php?")


def mark_invalid(spec, lay, book):
    """RND / MAX: the Stage1Hooks invalid path (after_battle -> InvalidBook.mark; "arm_stopped" for a fly skipped only
    because its arm stopped). An evaluation-block battle never reaches the situation evaluation, so none is given."""
    return stage1.Stage1Hooks(spec, lay, book, None, None).after_battle


async def drive(a, out, key, spec, learn, eval_, pool, swarm, arm_of, encoders, book, after, srv_cfg, should_stop):
    from flymon.ac.battles import ACBattles
    b = ACBattles(arm_of, out, srv_cfg, swarm=swarm, encoders=encoders)
    try:
        return await blocks.run_arm(out, eval_=eval_, attempt_for=stage1.skipping(b.attempt_for, book), cfg_hash=key,
                                    retry_max=spec.retry_max, resume=a.resume, n_flies=len(arm_of), learn=learn,
                                    pool=pool, swarm=swarm, reset_player=b.reset_player, stop_after=a.stop_after,
                                    on_block_end=b.close_block, after_battle=after, should_stop=should_stop)
    finally:
        await b.close()


def run_brain(a, spec, out, doc, model, r, learn, eval_, lay, should_stop) -> dict:
    import asyncio
    from flymon.ac import manifest, situations
    from flymon.ac.config import load_lv_config, lv_codebook
    from flymon.ac.swarm import LVSwarm, egrid_fn
    from flymon.ac.tb import TBEncoder
    from flymon.agent.config import config_hash
    from flymon.agent.encode_grid import GridEncoder
    from flymon.battle.server import ShowdownServer
    from flymon.brain.circuits import Populations
    from flymon.brain.connectome import Connectome
    from flymon.brain.fly_pool import FlySpec
    from flymon.brain.h4_jobs import type_cells
    from flymon.brain.lv_pool import LeverFlyPool

    cfg = load_lv_config(r)
    cb, cb_digest = lv_codebook()
    bad = manifest.check_model(model, cfg=cfg, codebook_digest=cb_digest)
    if bad:
        raise SystemExit(f"refusing: this run does not match {MODEL_MANIFEST}: {bad}")
    conn = Connectome.load(a.npz)
    pops = Populations.from_connectome(conn)
    grid, tb = GridEncoder(pops, cb, spec.dual_rule), TBEncoder(pops, cb, spec.dual_rule)
    arm_of = {g: arm for g, (arm, _) in enumerate(lay)}
    encoders = {g: (tb if arm == "TB" else grid) for g, arm in arm_of.items()}
    pairs = doc["confirm"]["pairs"]
    odours = {"grid": situations.pair_odours(grid, pairs), "tb": situations.pair_odours(tb, pairs)}
    book = InvalidBook(out / "invalid.json", arm_of, dict(spec.brain_arms), spec.min_valid_of())
    key = store.digest(dict(cfg=config_hash(cfg), layout=lay, inputs=doc["digests"], mode=spec.mode,
                            lever=spec.lever_sha))
    sitlog = stage1.SituationLog(out / "logs")
    flies = [FlySpec(enabled=arm != "COFF") for arm, _ in lay]
    with LeverFlyPool(a.npz, cfg.params, flies, edit=spec.lever_edit, p_type=spec.p_type, workers=a.workers,
                      timeout_s=POOL_TIMEOUT_S) as pool:
        sha = pool.lever_sha()
        if sha != model["lever_sha"]:
            raise SystemExit(f"refusing: the workers' L_V sha {sha[:12]} != the manifest's {model['lever_sha'][:12]}")
        w0_sha = blocks.weights_sha(pool.w0[None])
        swarm = LVSwarm(pool, cfg, type_cells(conn, [cfg.readout["A"], cfg.readout["P"]]), pops.kc, mode="learn",
                        tie_seed=spec.tie_seed, kc_ratio=spec.kc_ratio, egrid=egrid_fn(grid))

        def evaluate(g, point):
            rec = situations.evaluate(pool, g, point, pairs, odours["tb" if arm_of[g] == "TB" else "grid"], cfg,
                                      swarm.a_idx, swarm.p_idx, swarm.kc, spec.tie_seed)
            rec["floor_contact"] = situations.floor_contact(pool, g, swarm._edge_masks,
                                                            (cfg.reward_type, cfg.punish_type), spec.floor_ratio)
            return rec

        hooks = stage1.Stage1Hooks(spec, lay, book, sitlog, evaluate)
        try:
            if a.resume:
                sitlog.filter(stage1.committed(out, key))
            n_init = sum(1 for g in range(len(lay)) if not book.skip(g) and not sitlog.has_init(g))
            t1 = time.time()
            hooks.initial()
            t_sit = time.time() - t1
            with ShowdownServer(port=_port()) as srv:
                t2 = time.time()
                run = asyncio.run(drive(a, out, key, spec, learn, eval_, pool, swarm, arm_of, encoders, book,
                                        hooks.after_battle, srv_config(srv), should_stop))
                t_run = time.time() - t2
            medians = {g: swarm.median_ratio(g) for g in range(len(lay))} if run["complete"] else None
        finally:
            swarm._exec.shutdown(wait=True)
    rows = (stage1.per_fly_rows(out, lay, eval_=eval_, learn=learn, run=run, book=book, w0_sha=w0_sha,
                                medians=medians) if run["complete"] else None)
    return dict(run=run, per_fly=rows, book=book.status(), lever_sha=sha, config_hash=config_hash(cfg),
                timing=dict(t_sit_init_s=t_sit, n_init=n_init, t_run_s=t_run))


def run_nobrain(a, spec, out, doc, eval_, lay, should_stop) -> dict:
    import asyncio
    from flymon.battle.server import ShowdownServer
    arm_of = {g: arm for g, (arm, _) in enumerate(lay)}
    book = InvalidBook(out / "invalid.json", arm_of, {a.arm: len(lay)}, {a.arm: spec.min_valid_of()[a.arm]})
    key = store.digest(dict(arm=a.arm, inputs=doc["digests"], mode=spec.mode))
    with ShowdownServer(port=_port()) as srv:
        run = asyncio.run(drive(a, out, key, spec, None, eval_, None, None, arm_of, None, book,
                                mark_invalid(spec, lay, book), srv_config(srv), should_stop))
    rows = stage1.nobrain_rows(out, lay, eval_=eval_, run=run, book=book) if run["complete"] else None
    return dict(run=run, per_fly=rows, book=book.status())


def main(argv=None) -> int:
    t0 = time.time()
    a = parse_args(argv)
    spec = spec_for(a)
    out = Path(a.out)
    store.guard(out / "result.json")
    if spec.mode == "bench":
        fresh_bench(out)
    doc, inputs_info = load_inputs(spec)
    model, r = load_model(spec) if a.arm == "BRAIN" else (None, None)
    bud = load_budget(spec)
    budget.close_stale(out)              # a hard-killed earlier session of this arm is charged before the cap (AC.6)
    cap = session_cap(spec, a)
    if (out / "result.json").exists() and not a.resume:
        raise SystemExit(f"refusing: {out / 'result.json'} exists (the arm is complete); pass --resume to rewrite it")
    git = store.git_provenance(files=files())
    if git["dirty"] and not a.allow_dirty:
        raise SystemExit(f"refusing: uncommitted changes in {git['dirty_files']} (commit them or pass --allow-dirty)")
    if a.arm == "BRAIN":
        learn, eval_ = stage1.brain_schedules(spec, doc)
        lay = stage1.layout(spec)
    else:
        learn, eval_ = None, stage1.nobrain_schedule(spec, doc, a.arm)
        lay = stage1.nobrain_layout(spec, a.arm)
    schedules.check_names((learn or []) + eval_, spec.retry_max)
    out.mkdir(parents=True, exist_ok=True)
    sess = budget.open_session(out, t0)
    deadline = t0 + cap
    closed = {"wall": None}
    try:
        res = (run_brain(a, spec, out, doc, model, r, learn, eval_, lay, lambda: time.time() >= deadline)
               if a.arm == "BRAIN" else run_nobrain(a, spec, out, doc, eval_, lay, lambda: time.time() >= deadline))
        run = res["run"]
        wall = closed["wall"] = budget.close_session(out, sess, t0, "ok", run)
        if spec.mode == "bench" and not run["complete"]:
            print(f"bench: stopped before completion (played {sum(len(v) for v in run['played'].values())}); "
                  f"no bench.json. The bench is one fresh session (--resume is refused): remove {out} and rerun",
                  flush=True)
            return 2
        if spec.mode == "bench":
            t = res["timing"]
            b = dict(s_per_batch_battle=t["t_run_s"] / spec.learn_battles,
                     sit_eval_s_per_fly=t["t_sit_init_s"] / max(1, t["n_init"]), flies=len(lay), workers=a.workers,
                     battles=spec.learn_battles, played=len(run["played"].get("L", [])), timing=t, git=git,
                     label=LABEL)
            store.write_json(out / "bench.json", b)
            print(f"bench: {b['s_per_batch_battle']:.1f} s/batch-battle, situation evaluation "
                  f"{b['sit_eval_s_per_fly']:.1f} s/fly; wrote {out / 'bench.json'}", flush=True)
            return 0
        if not run["complete"]:
            print(f"{a.arm}: stopped before completion (played {sum(len(v) for v in run['played'].values())}, "
                  f"invalid {res['book']['invalid']}, stopped arms {res['book']['stopped']}); rerun with --resume",
                  flush=True)
            return 0
        result = dict(arm=a.arm, mode=spec.mode, layout=lay, learn=spec.learn_battles if learn else 0,
                      eval=spec.eval_battles, per_fly=res["per_fly"], book=res["book"], inputs=inputs_info,
                      eval_digest=doc["digests"]["eval"], learn_digest=doc["digests"]["learn"],
                      confirm_digest=doc["digests"]["confirm"], recovery_per_pulse=r,
                      model=None if model is None else dict(path=MODEL_MANIFEST,
                                                            sha256=store.sha256_file(MODEL_MANIFEST)),
                      lever_sha=res.get("lever_sha"), config_hash=res.get("config_hash"), budget=bud,
                      wall_clock_s=wall, workers=a.workers, complete=True, label=LABEL,
                      provenance=dict(git=git, argv=list(sys.argv[1:] if argv is None else argv),
                                      finished_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
        store.write_json(out / "result.json", result)
        valid = sum(not row["invalid"] for row in res["per_fly"])
        print(f"{a.arm}: complete, valid flies {valid}/{len(lay)}, stopped arms {res['book']['stopped']}; "
              f"wrote {out / 'result.json'}", flush=True)
        return 0
    finally:
        if closed["wall"] is None:
            budget.close_session(out, sess, t0, "aborted", None)


if __name__ == "__main__":
    sys.exit(main())
