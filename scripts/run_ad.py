#!/usr/bin/env python3
"""AD's run (spec AD.1-AD.7): FLY-L 12 + FLY-OS 12 on one L_V LeverFlyPool through a local Showdown server.

    uv run python scripts/run_ad.py                    # the run under results/m4/ad/ (a session cap -> --resume)
    uv run python scripts/run_ad.py --resume
    uv run python scripts/run_ad.py --smoke --stop-after 3 ; ... --smoke --resume    # AD.5 1 smoke
    uv run python scripts/run_ad.py --bench N          # AD.4 re-bench attempt N under results/m4-bench/ad-N/

Learning (T from the run manifest; softmax; own pulses with attempt tokens) in two phases split at the futility check
(40), situation-pair evaluations at 0 / 20 / ... / T (FLY-L also on OS odours; the new 20 pairs at 0 / 40 / T if the
manifest includes them), then the evaluation block (the shared 20-battle schedule, argmax with the seeded tie rule,
plasticity off). Refusals run before any pool or server: inputs (STOP_MANIFEST), stage-1 reuse (STOP_REUSE), the model
manifest (FROZEN), the run manifest (STOP_MANIFEST), the budget left (STOP_BUDGET), uncommitted code. Each battle starts
only if its batch's worst case fits the session and the total budget (a total-budget refusal is STOP_BUDGET in the run:
result.json records it and the reading uses the latest common point)."""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import glob
import json
import os
import socket
import sys
import time
from pathlib import Path

from flymon.ac import budget as ac_budget
from flymon.ac import schedules as ac_sched
from flymon.ac.manifest import MODEL_MANIFEST
from flymon.ad import budget, inputs, reuse, run_manifest as rm, store
from flymon.ad.spec import LABEL, SPEC, bench, smoke
from flymon.rescope import blocks

NPZ = "data/malecns.npz"
POOL_TIMEOUT_S = 7200


def spec_for(a):
    return smoke() if a.smoke else bench(a.bench) if a.bench is not None else SPEC


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--smoke", action="store_true")
    g.add_argument("--bench", type=int, default=None, metavar="N")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stop-after", type=int, default=None)
    ap.add_argument("--session-hours", type=float, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--npz", default=NPZ)
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args(argv)
    if a.bench is not None and a.resume:
        ap.error("--bench is one fresh session")
    if a.smoke:
        a.workers = 4
    if a.bench is not None:
        a.workers = SPEC.bench_workers
    return a


def _reuse_bad(digests: dict, man: dict, spec) -> list:
    now = reuse.reuse_fields(digests["confirm"], digests["eval"], spec)
    return reuse.changed_since(man.get("reuse"), now) + reuse.reuse_problems(now, spec)


def load_inputs(spec) -> tuple:
    """AD.2: (가) before (나) - in the run, the stage-1 fields (the document's confirm / eval digests and stage 1's
    records) are checked first, and a rebuilt confirm / eval digest that left stage 1's is STOP_REUSE too, so a
    reuse change never reads as STOP_MANIFEST; then the AD-fixed fields (check_inputs) -> STOP_MANIFEST. An input
    manifest with status STOP_REUSE is STOP_REUSE; STOP_SET (fewer candidates than pairs; unreachable, 82 >= 20) maps
    to STOP_REUSE for the reused confirmation set, else (the new 20 pairs, an AD-fixed field) to STOP_MANIFEST."""
    doc_p, man_p = inputs.input_paths(spec)
    for p in (doc_p, man_p):
        if not p.exists():
            raise SystemExit(f"refusing: {p} does not exist (scripts/write_ad_inputs.py"
                             f"{'' if spec.mode == 'run' else ' --' + spec.mode})")
    doc, man = json.loads(doc_p.read_text()), json.loads(man_p.read_text())
    st = man.get("status")
    if st == "STOP_REUSE":
        raise SystemExit(f"STOP_REUSE: {man_p} status STOP_REUSE: {man.get('reuse_problems')}")
    if st == "STOP_SET":
        code = "STOP_REUSE" if (doc.get("confirm") or {}).get("status") != "OK" else "STOP_MANIFEST"
        raise SystemExit(f"{code}: {man_p} status STOP_SET (a pair set could not be drawn, AD.2)")
    if st != "OK":
        raise SystemExit(f"refusing: {man_p} status {st!r}")
    if spec.mode == "run":
        bad = _reuse_bad(doc["digests"], man, spec)
        if bad:
            raise SystemExit(f"STOP_REUSE: stage-1 fields changed or wrong since {man_p}: {bad}")
    bad = inputs.check_inputs(doc, man, spec, doc_p)
    if bad:
        if spec.mode == "run":
            rb = _reuse_bad(inputs.inputs_doc(spec)["digests"], man, spec)
            if rb:
                raise SystemExit(f"STOP_REUSE: the confirm / eval digests rebuilt from the seeds left stage 1's: {rb}")
        raise SystemExit(f"STOP_MANIFEST: {doc_p} does not match its seeds / manifest: {bad}")
    return doc, dict(path=str(doc_p), manifest=str(man_p), manifest_sha256=store.sha256_file(man_p),
                     digests=doc["digests"])


def load_model() -> tuple:
    p = Path(MODEL_MANIFEST)
    if not p.exists():
        raise SystemExit(f"refusing: {p} does not exist (AC.7 2a)")
    d = json.loads(p.read_text())
    if d.get("status") != "FROZEN":
        raise SystemExit(f"refusing: {p} status {d.get('status')!r} is not FROZEN")
    return d, float(d["recovery_per_pulse"])


def load_run_manifest(spec, a) -> tuple:
    if spec.mode == "bench":
        return None, spec.learn_battles, True
    p = rm.manifest_path(spec)
    if not p.exists():
        raise SystemExit(f"refusing: {p} does not exist (scripts/write_ad_run_manifest.py"
                         f"{' --smoke' if spec.mode == 'smoke' else ''})")
    man = json.loads(p.read_text())
    bad = rm.check(man, spec, check_code=not (spec.mode == "smoke" and a.allow_dirty))
    if bad:
        raise SystemExit(f"STOP_MANIFEST: {p} does not match this run: {bad}")
    return man, int(man["T"]), bool(man["new_set"])


def load_budget(spec):
    if spec.mode != "run":
        return None
    d = json.loads(Path(budget.BUDGET).read_text())          # the run manifest already checked it exists and is OK
    return d


def session_window(spec, out, a, t0: float) -> tuple:
    if spec.mode == "run":
        spent = budget.spent_s(out)
        cap, left = budget.session_cap_s(spec, spent), budget.remaining_s(spec, spent)
    else:
        cap, left = spec.session_cap_h * 3600.0, float("inf")
    if a.session_hours is not None:
        cap = min(cap, a.session_hours * 3600.0)
    if cap <= 0:
        raise SystemExit(f"STOP_BUDGET: the {spec.budget_limit_h():.3f} h AD budget is spent (AD.4)")
    return cap, left


def fresh_bench(out) -> None:
    out = Path(out)
    if any((out / x).exists() for x in ("checkpoints", "logs", "bench.json", "wall_clock.json")):
        raise SystemExit(f"refusing: {out} holds an earlier bench attempt; every attempt is one fresh session in its "
                         f"own directory (use the next N)")


def files() -> list:
    return (sorted(glob.glob("flymon/ad/*.py")) + sorted(glob.glob("flymon/ac/*.py"))
            + ["flymon/brain/lv_pool.py", "flymon/rescope/blocks.py", "flymon/agent/runner.py",
               "flymon/agent/policy.py", "scripts/run_ad.py"])


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); return s.getsockname()[1]


def srv_config(srv):
    from poke_env.ps_client import ServerConfiguration
    return ServerConfiguration(srv.url_ws, "https://play.pokemonshowdown.com/action.php?")


def gate_kwargs(gate) -> dict:
    """Ruling C1: no session-wide should_stop - the per-battle gate alone decides, so a batch already opened is finished
    by its lagging flies after a refusal of the next batch (a session-wide stop would drop their battle k)."""
    return dict(should_stop=None, should_stop_battle=gate)


async def drive(a, out, key, spec, learn, eval_, pool, swarm, lay, encoders, books, attempts, after, srv_cfg, gate,
                decide_futility):
    from flymon.ad import hooks
    from flymon.ad.battles import ADBattles
    from flymon.ad.runner import run_ad_arm
    b = ADBattles(lay, out, srv_cfg, swarm, encoders, attempts)
    try:
        return await run_ad_arm(out, learn=learn, eval_=eval_, attempt_for=hooks.skipping(b.attempt_for, books),
                                cfg_hash=key, retry_max=spec.retry_max, resume=a.resume, pool=pool, swarm=swarm,
                                reset_player=b.reset_player, stop_after=a.stop_after, on_block_end=b.close_block,
                                after_battle=after, **gate_kwargs(gate),
                                futility_point=spec.futility_point if spec.futility_checked() else None,
                                decide_futility=decide_futility)
    finally:
        await b.close()


def run_brain(a, spec, out, doc, model, r, learn, eval_, T, new_set, man_sha, gate_args) -> dict:
    import asyncio
    from flymon.ac import manifest
    from flymon.ac.config import load_lv_config, lv_codebook
    from flymon.ac.invalid import InvalidBook
    from flymon.ac.spec import SPEC as AC
    from flymon.ac.swarm import egrid_fn
    from flymon.ad import hooks, validity
    from flymon.ad.attempts import AttemptBook
    from flymon.ad.os_encoder import OSEncoder
    from flymon.ad.swarm import ADSwarm
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
    bad = manifest.check_model(model, cfg=cfg, codebook_digest=cb_digest, spec=AC)
    if bad:
        raise SystemExit(f"refusing: this run does not match {MODEL_MANIFEST}: {bad}")
    conn = Connectome.load(a.npz)
    pops = Populations.from_connectome(conn)
    lay = spec.layout()
    grid = GridEncoder(pops, cb, spec.dual_rule)
    encoders = {i: (OSEncoder(pops, cb, spec.dual_rule, g, spec.os_seed) if arm == "FLYOS" else grid)
                for i, (arm, _, g) in enumerate(lay)}
    sets = {"confirm": doc["confirm"]["pairs"], "new": doc["new"]["pairs"]}
    arm_of = {i: arm for i, (arm, _, _) in enumerate(lay)}
    mins = {k: v for k, v in spec.min_valid_of().items() if k in spec.arm_sizes()}
    books = {"L": InvalidBook(out / "invalid.json", arm_of, spec.arm_sizes(), mins),
             "E": InvalidBook(out / "invalid_eval.json", arm_of, spec.arm_sizes(), mins)}
    key = rm.checkpoint_key(cfg_hash=config_hash(cfg), layout=lay, input_digests=doc["digests"], mode=spec.mode,
                            lever_sha=spec.lever_sha, run_manifest_sha256=man_sha)
    sitlog = hooks.ADSituationLog(out / "logs")
    with LeverFlyPool(a.npz, cfg.params, [FlySpec() for _ in lay], edit=spec.lever_edit, p_type=spec.p_type,
                      workers=a.workers, timeout_s=POOL_TIMEOUT_S) as pool:
        sha = pool.lever_sha()
        if sha != model["lever_sha"]:
            raise SystemExit(f"refusing: the workers' L_V sha {sha[:12]} != the manifest's {model['lever_sha'][:12]}")
        attempts = AttemptBook()
        swarm = ADSwarm(pool, cfg, type_cells(conn, [cfg.readout["A"], cfg.readout["P"]]), pops.kc, attempts=attempts,
                        mode="learn", tie_seed=spec.tie_seed, kc_ratio=spec.kc_ratio, egrid=egrid_fn(grid),
                        seed_fly={i: g for i, (_, _, g) in enumerate(lay)},
                        stale_path=out / "logs" / "stale_pulses.jsonl")
        evaluate = hooks.make_evaluate(pool, swarm, cfg, spec, lay, sets, grid)
        hk = hooks.ADHooks(spec, T, new_set, lay, books, sitlog, evaluate)
        gate = budget.BatchGate(spec, T, new_set, times_path=out / "batch_times.json", **gate_args)

        def after(block, sb, rec):
            gate.finished(block, sb)                         # the battle's span ends before its point evaluations
            hk.after_battle(block, sb, rec)

        def decide_futility():
            fut = validity.persist_futility(out / "futility.json",
                                            validity.futility(sitlog.records(), sets["confirm"], spec))
            print(validity.futility_line(fut), flush=True)   # AD.2: only these two numbers and the decision
            return fut

        try:
            if a.resume:
                done = hooks.committed_all(out, key)         # STOP_INFRA (checksum) | STOP_MANIFEST (key)
                sitlog.filter(done)
                for blk, tag in zip("LE", spec.tags):        # a half-played batch is finished, never refused
                    gate.reopen(blk, [b for b in done if b.startswith(tag + "-")])
            t1 = time.time()
            n_init = hk.initial()
            t_sit = time.time() - t1
            with ShowdownServer(port=_port()) as srv:
                t2 = time.time()
                run = asyncio.run(drive(a, out, key, spec, learn, eval_, pool, swarm, lay, encoders, books, attempts,
                                        after, srv_config(srv), gate, decide_futility))
                t_run = time.time() - t2
            medians = {i: swarm.median_ratio(i) for i in range(len(lay))} if run["complete"] else None
        finally:
            swarm._exec.shutdown(wait=True)
    rows = None
    if run["complete"] and eval_:
        ids = set(run["learn_ids"])
        rows = blocks.arm_per_fly(out, n_flies=len(lay), eval_=eval_, run=run,
                                  learn=[sb for sb in learn if sb.battle_id in ids], spec=None, brain=True,
                                  weight_medians=medians)
        for i, row in enumerate(rows):
            arm, k, g = lay[i]
            row.update(arm=arm, k=k, gfly=g, learn_invalid=books["L"].invalid.get(i),
                       eval_invalid=books["E"].invalid.get(i))
    return dict(run=run, per_fly=rows, books={b: bk.status() for b, bk in books.items()}, lever_sha=sha,
                config_hash=config_hash(cfg), stale=len(swarm.stale),
                gate=dict(refused=gate.refused, refusal=gate.refusal, longest_s=gate.longest),
                timing=dict(t_sit_init_s=t_sit, n_init=n_init, t_run_s=t_run))


def bench_doc(spec, t: dict, *, t0: float, t_ended: float, workers: int, git) -> dict:
    """bench.json (AD.4, read by scripts/write_ad_budget.py): pid is this python's own (the load log's process, not
    uv's), [t_started, t_ended] the attempt's window; s/evaluation = the point-0 situation time over EVERY point-0
    evaluation of the session (n_init: 24 own + 12 paired OS + 24 new-set = 60 on a fresh bench)."""
    return dict(s_per_batch_battle=t["t_run_s"] / spec.learn_battles,
                sit_eval_s_per_eval=t["t_sit_init_s"] / max(1, t["n_init"]), n_init=t["n_init"],
                flies=spec.n_flies(), workers=workers, battles=spec.learn_battles, pid=os.getpid(),
                t_started=t0, t_ended=t_ended, attempt=spec.bench_attempt, timing=t, git=git, label=LABEL)


def main(argv=None) -> int:
    t0 = time.time()
    a = parse_args(argv)
    spec = spec_for(a)
    out = Path(spec.out_root)
    store.guard(out / "result.json")
    if spec.mode == "bench":
        fresh_bench(out)
    doc, info = load_inputs(spec)
    model, r = load_model()
    man, T, new_set = load_run_manifest(spec, a)
    spec = dataclasses.replace(spec, learn_battles=int(T))
    man_sha = store.sha256_file(rm.manifest_path(spec)) if man is not None else None
    bud = load_budget(spec)
    budget.close_stale(out, t0)                    # AD.4: a killed AD session is charged to this session's start
    cap, left = session_window(spec, out, a, t0)
    if (out / "result.json").exists() and not a.resume:
        raise SystemExit(f"refusing: {out / 'result.json'} exists; pass --resume to rewrite it")
    git = store.git_provenance(files=files())
    if git["dirty"] and not a.allow_dirty:
        raise SystemExit(f"refusing: uncommitted changes in {git['dirty_files']} (commit them or pass --allow-dirty)")
    learn, eval_ = inputs.schedules_for(spec, doc, T)
    ac_sched.check_names(learn + eval_, spec.retry_max)
    out.mkdir(parents=True, exist_ok=True)
    sess = ac_budget.open_session(out, t0)
    gate_args = dict(session_end=t0 + cap, budget_end=t0 + left,
                     bench_batch_s=bud["bench"]["s_per_batch_battle"] if bud else 0.0,
                     bench_eval_s=bud["bench"]["sit_eval_s"] if bud else 0.0)
    closed = {"wall": None}
    try:
        res = run_brain(a, spec, out, doc, model, r, learn, eval_, T, new_set, man_sha, gate_args)
        run = res["run"]
        wall = closed["wall"] = ac_budget.close_session(out, sess, t0, "ok", run)
        if spec.mode == "bench":
            if not run["complete"]:
                print("bench: stopped before completion; no bench.json (use the next attempt N)", flush=True)
                return 2
            t = res["timing"]
            b = bench_doc(spec, t, t0=t0, t_ended=time.time(), workers=a.workers, git=git)
            store.write_json(out / "bench.json", b)
            print(f"bench {spec.bench_attempt}: {b['s_per_batch_battle']:.1f} s/batch-battle, "
                  f"{b['sit_eval_s_per_eval']:.1f} s/evaluation ({t['n_init']}); wrote {out / 'bench.json'}", flush=True)
            return 0
        fut = run.get("futility") or {}
        base = dict(mode=spec.mode, T=T, new_set=new_set, futility=fut or None,
                    T_effective=spec.futility_point if fut.get("status") == "STOP_FUTILE" else T,
                    layout=[list(x) for x in spec.layout()], books=res["books"], inputs=info,
                    eval_digest=doc["digests"]["eval"], learn_prefix_digest=None if man is None else man["learn_prefix_digest"],
                    run_manifest=dict(path=str(rm.manifest_path(spec)), sha256=man_sha),
                    model=dict(path=MODEL_MANIFEST, sha256=store.sha256_file(MODEL_MANIFEST)), recovery_per_pulse=r,
                    lever_sha=res["lever_sha"], config_hash=res["config_hash"], gate=res["gate"],
                    stale_pulses=res["stale"], wall_clock_s=wall, workers=a.workers, label=LABEL,
                    provenance=dict(git=git, argv=list(sys.argv[1:] if argv is None else argv),
                                    finished_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
        if not run["complete"]:
            if res["gate"]["refused"] == "STOP_BUDGET":
                store.write_json(out / "result.json", dict(base, status="STOP_BUDGET", complete=False, per_fly=None))
                print(f"AD: STOP_BUDGET (in run) at {res['gate']['refusal']}; wrote {out / 'result.json'}", flush=True)
                return 0
            print(f"AD: stopped before completion (played L {len(run['played']['L'])}, E {len(run['played']['E'])}; "
                  f"gate {res['gate']['refused']}); rerun with --resume", flush=True)
            return 0
        store.write_json(out / "result.json", dict(base, status="COMPLETE", complete=True, per_fly=res["per_fly"]))
        valid = sum(not row["invalid"] for row in res["per_fly"])
        print(f"AD: complete, valid flies {valid}/{len(res['per_fly'])}, futility {fut.get('status')}, T_effective "
              f"{base['T_effective']}; wrote {out / 'result.json'}", flush=True)
        return 0
    finally:
        if closed["wall"] is None:
            ac_budget.close_session(out, sess, t0, "aborted", None)


if __name__ == "__main__":
    sys.exit(main())
