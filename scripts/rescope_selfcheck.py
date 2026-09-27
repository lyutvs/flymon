#!/usr/bin/env python3
"""Re-scoped claim, self-checks on the real connectome before any stage (plan Task 11).

    uv run python scripts/rescope_selfcheck.py [--allow-dirty] [--npz data/malecns.npz]

Four checks, each {"ok": bool, ...}; a check that raises is recorded as {"ok": False, "error": ...}:
  1. pairs_digest         pairs.new_pairs on the real connectome reproduces SPEC.pairs_digest.
  2. oracle_copy_real     on C3 and the seed0 pair (qualification seeds with 2 act / select / report seeds, X by
                          oracle.qual_x as in the qualification), reward_oracle_job's alpha_reward, select["pre"],
                          select["reward"] and report pre / R1 equal h4_jobs.oracle_job's.
  3. pool_equivalence_c3  one FlyPool.decide_batch on the C3 pool equals presentation.decide on an in-process
                          engine for the same seed (the method of tests/agent/test_swarm_c3.py).
  4. naive_vs_oracle_pre  the naive MBON05 counts the qualification computes (run_rescope_qualify.naive_counts, the
                          FlyPool.decide_batch path) on seed0's report seeds equal reward_oracle_job's report["pre"]["P"]
                          (h4_jobs.decide in the job's rig), per seed as [X, Y]; a match on all-zero counts is not ok.
Writes results/rescope/selfcheck.json ({checks, all_ok, provenance}) and exits 1 unless all_ok. Refuses (SystemExit)
without --allow-dirty when flymon/rescope/, this script or scripts/run_rescope_qualify.py has uncommitted changes.
Every pool is built inside main() (spawn re-imports this module in the workers).
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import functools
import glob
import importlib.util
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np

from flymon.brain.config import Params
from flymon.rescope.spec import SPEC
from flymon.rescope.store import git_provenance, guard, write_json

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = "scripts/rescope_selfcheck.py"
QUALIFY = "scripts/run_rescope_qualify.py"
OUT = Path("results/rescope/selfcheck.json")
POOL_TIMEOUT_S = 1800
NPZ = "data/malecns.npz"          # set by main(--npz)
CHECK_SPEC = dataclasses.replace(SPEC, n_qual_seeds=2)     # 2 act / select / report seeds per pair
PAIR = SPEC.control


def _qualify_module():
    spec = importlib.util.spec_from_file_location("run_rescope_qualify", ROOT / QUALIFY)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


@functools.lru_cache(maxsize=None)
def _real():
    from flymon.agent.config import load_c3_config
    from flymon.brain.circuits import Populations
    from flymon.brain.connectome import Connectome
    conn = Connectome.load(NPZ)
    return conn, Populations.from_connectome(conn), load_c3_config()


@functools.lru_cache(maxsize=None)
def _seed0_naive():
    """(naive, x, odors): the qualification's naive MBON05 counts of seed0 (select then report seeds) and its X."""
    from flymon.brain.fly_pool import FlyPool, FlySpec
    from flymon.brain.h4_jobs import type_cells
    from flymon.rescope.oracle import qual_x
    from flymon.rescope.pairs import pair_odors
    conn, pops, cfg = _real()
    q = _qualify_module()
    odors = pair_odors(pops, PAIR, CHECK_SPEC)
    idx_p = type_cells(conn, [CHECK_SPEC.p_type])[CHECK_SPEC.p_type]
    with FlyPool(NPZ, cfg.params, [FlySpec()], workers=1, timeout_s=POOL_TIMEOUT_S) as pool:
        naive = q.naive_counts(pool, CHECK_SPEC, PAIR, odors, idx_p)
    return naive, qual_x(naive, CHECK_SPEC), odors


@functools.lru_cache(maxsize=None)
def _seed0_oracles():
    """(oracle_job row, reward_oracle_job row) on the qualification's oracle pool (default Params, the job gets C3)."""
    from flymon.brain.fly_pool import FlyPool
    from flymon.brain.h4_jobs import oracle_job
    from flymon.rescope.oracle import reward_oracle_job
    _, _, cfg = _real()
    q = _qualify_module()
    _, x, odors = _seed0_naive()
    kw = q.oracle_kwargs(CHECK_SPEC, cfg.params, PAIR, odors, x)
    with FlyPool(NPZ, Params(), [{}], workers=1, punish_type=CHECK_SPEC.punish_dan, reward_type=CHECK_SPEC.reward_dan,
                 timeout_s=POOL_TIMEOUT_S) as pool:
        full = pool.run_jobs(oracle_job, [dict(kw, punish_type=CHECK_SPEC.punish_dan)])[0]
        mine = pool.run_jobs(reward_oracle_job, [kw])[0]
    return full, mine


def check_pairs_digest() -> dict:
    from flymon.rescope.pairs import new_pairs, pairs_digest
    _, pops, _ = _real()
    pairs = new_pairs(pops, SPEC)
    got = pairs_digest(pairs)
    return dict(ok=got == SPEC.pairs_digest, digest=got, pinned=SPEC.pairs_digest, seeds=[p["seed"] for p in pairs])


def check_oracle_copy() -> dict:
    full, mine = _seed0_oracles()
    _, x, _ = _seed0_naive()
    same = {"alpha_reward": mine["alpha_reward"] == full["alpha_reward"],
            "select_pre": mine["select"]["pre"] == full["select"]["pre"],
            "select_reward": mine["select"]["reward"] == full["select"]["reward"],
            "report_pre": mine["report"]["pre"] == full["report"]["pre"],
            "report_R1": mine["report"]["R1"] == full["report"]["R1"]}
    return dict(ok=all(same.values()), same=same, pair=PAIR, x=x, alpha_reward=[mine["alpha_reward"],
                full["alpha_reward"]], seeds=CHECK_SPEC.qual_seeds(PAIR))


def check_pool_equivalence() -> dict:
    from flymon.brain import h4_pairs
    from flymon.brain.circuits import compartments
    from flymon.brain.engine_cpu import Engine
    from flymon.brain.fly_pool import FlyPool, FlySpec
    from flymon.brain.plasticity import Plasticity
    from flymon.brain.presentation import decide
    conn, pops, cfg = _real()
    pair = h4_pairs.even_pairs(pops)[0]
    odours = [pair["odor_x"], pair["odor_y"]]
    eng = Engine(conn, pops, cfg.params, seed=0)
    pl = Plasticity(eng, pops, compartments(conn, pops, cfg.params.core_frac))
    want = decide(eng, pl, pops, odours, cfg.strength, 123, cfg.settle_ms, cfg.read_ms)
    with FlyPool(NPZ, cfg.params, [FlySpec()], workers=1, timeout_s=POOL_TIMEOUT_S) as pool:
        got = pool.decide_batch([(0, odours, 123)], cfg.strength, cfg.settle_ms, cfg.read_ms)[0]
    return dict(ok=bool(np.array_equal(got, want)), seed=123, shape=list(np.shape(want)),
                spikes_in_process=int(np.sum(want)), spikes_pool=int(np.sum(got)))


def check_naive_vs_oracle_pre() -> dict:
    naive, x, _ = _seed0_naive()
    _, mine = _seed0_oracles()
    y = "b" if x == "a" else "a"
    n_sel = len(CHECK_SPEC.qual_seeds(PAIR)["select"])
    naive_report = [[naive[x][n_sel + i], naive[y][n_sel + i]] for i in range(len(naive[x]) - n_sel)]
    oracle_pre = [[int(v) for v in row] for row in mine["report"]["pre"]["P"]]
    nonzero = any(v for row in naive_report for v in row)
    return dict(ok=bool(naive_report == oracle_pre and nonzero), naive_report_xy=naive_report,
                oracle_report_pre_P=oracle_pre, nonzero=nonzero, x=x, report_seeds=CHECK_SPEC.qual_seeds(PAIR)["report"])


CHECKS = (("pairs_digest", "check_pairs_digest"), ("oracle_copy_real", "check_oracle_copy"),
          ("pool_equivalence_c3", "check_pool_equivalence"), ("naive_vs_oracle_pre", "check_naive_vs_oracle_pre"))


def _provenance() -> dict:
    try:
        return git_provenance(files=sorted(glob.glob("flymon/rescope/*.py")) + [SCRIPT, QUALIFY])
    except (subprocess.CalledProcessError, OSError) as e:
        return {"commit": None, "dirty": None, "dirty_files": [], "error": repr(e)}


def main(argv=None) -> int:
    global NPZ
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default=NPZ)
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args(argv)
    NPZ = a.npz
    gp = [Params()]
    guard(OUT, gp)
    git = _provenance()
    if (git["dirty"] or git["dirty"] is None) and not a.allow_dirty:
        raise SystemExit(f"refusing: uncommitted changes (or no git) {git.get('dirty_files')} {git.get('error', '')} "
                         f"(commit them or pass --allow-dirty)")
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    checks = {}
    for key, fn in CHECKS:
        try:
            checks[key] = globals()[fn]()
        except Exception as e:           # a check that cannot run is a failed check, recorded
            checks[key] = {"ok": False, "error": repr(e), "traceback": traceback.format_exc()}
        print(f"{key}: {'ok' if checks[key].get('ok') else 'FAILED'}", flush=True)
    all_ok = all(c.get("ok") is True for c in checks.values())
    write_json(OUT, dict(checks=checks, all_ok=all_ok, npz=a.npz,
                         provenance=dict(git=git, started_utc=started,
                                         finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                                         argv=list(sys.argv[1:] if argv is None else argv))), gp)
    print(f"all_ok {all_ok}; wrote {OUT}", flush=True)
    if not all_ok:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
