#!/usr/bin/env python3
"""Re-scoped claim, pair qualification (spec 10.3): for seed0 and every new designed pair, the naive MBON05 counts on
every qualification select and report seed (both odours; C3 engine), X = oracle.qual_x, the reward-only oracle on C3
(oracle.reward_oracle_job, z = the C3 z {"A": SPEC.z_a, "P": SPEC.z_p}) and oracle.qualify (floor rule B, spec 10.3
amendment 2026-09-28: per odour, median >= floor_spikes and silent-seed share <= floor_silent_max).

    uv run python scripts/run_rescope_qualify.py --npz data/malecns.npz --workers 16 --out results/rescope/qualify
    uv run python scripts/run_rescope_qualify.py --smoke --allow-dirty --workers 4      # 2 qualification seeds

Writes <out>/oracle_rows.json (raw) and results/summary/rescope_qualify.json ({pairs (each with the rule-B items per
odour in `floor` and the recorded silent-seed share in `silent_share`), qualified, m, stop,
control_qualified, oc_table, pairs_digest, recovery_per_pulse, provenance}); stop is STOP_CONTROL_INVALID when seed0 is
not qualified (precedence), else STOP_FEW_PAIRS when m < 4, else None; --smoke writes under results/rescope-smoke/ only (its summary
at results/rescope-smoke/summary/rescope_qualify.json). Refuses (SystemExit) an --out outside the re-scope trees and,
without --allow-dirty, a dirty flymon/rescope/ or this script, and a C3 recovery_per_pulse other than 0 (spec 10.2).
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import glob
import sys
from pathlib import Path

from flymon.brain.config import Params
from flymon.brain.fly_pool import FlyPool, FlySpec
from flymon.rescope.oracle import qual_x, qualify, reward_oracle_job
from flymon.rescope.primary import check_recovery
from flymon.rescope.rules import oc_table
from flymon.rescope.spec import SPEC
from flymon.rescope.store import git_provenance, guard, write_json

STAGE = "qualify"
POOL_TIMEOUT_S = 1800


def smoke_spec(spec=SPEC):
    return dataclasses.replace(spec, n_flies=2, n_probe=2, trials=2, valid_min=1, n_qual_seeds=2)


def paths(out: str, smoke: bool, stage: str) -> tuple:
    """(out dir, summary path). --smoke forces both under results/rescope-smoke/."""
    if not smoke:
        return Path(out), Path("results/summary") / f"rescope_{stage}.json"
    o = Path(out)
    rel = o.as_posix()
    if not rel.startswith("results/rescope-smoke/"):
        o = Path("results/rescope-smoke") / (o.name or stage)
    return o, Path("results/rescope-smoke/summary") / f"rescope_{stage}.json"


def tracked_files(script: str) -> list:
    return sorted(glob.glob("flymon/rescope/*.py")) + [script]


def naive_counts(pool, spec, name: str, odors: dict, idx_p) -> dict:
    """Naive MBON05 spike sums of both odours on every select and report qualification seed (in that order)."""
    q = spec.qual_seeds(name)
    seeds = list(q["select"]) + list(q["report"])
    reqs = [(k % pool.n_flies, [odors["a"], odors["b"]], int(s)) for k, s in enumerate(seeds)]
    res = pool.decide_batch(reqs, spec.strength, settle_ms=spec.probe_settle_ms, read_ms=spec.probe_read_ms, idx=idx_p)
    return {o: [int(c[j].sum()) for c in res] for j, o in enumerate(("a", "b"))}


def oracle_kwargs(spec, params, name: str, odors: dict, x: str) -> dict:
    q = spec.qual_seeds(name)
    y = "b" if x == "a" else "a"
    return dict(params=params, odor_x=odors[x], odor_y=odors[y], readout={"A": spec.a_type, "P": spec.p_type},
                z={"A": spec.z_a, "P": spec.z_p}, types=[spec.a_type, spec.p_type], act_seeds=list(q["act"]),
                select_seeds=list(q["select"]), report_seeds=list(q["report"]), alphas=tuple(spec.oracle_alphas),
                strength=spec.strength, settle_ms=spec.oracle_settle_ms, read_ms=spec.oracle_read_ms,
                window_ms=spec.kc_window_ms, reward_type=spec.reward_dan)


def naive_all(pool, spec, odors_of: dict, idx_p, log=print) -> dict:
    out = {}
    for name in spec.pair_names():
        out[name] = naive_counts(pool, spec, name, odors_of[name], idx_p)
        log(f"{name}: naive MBON05 a {out[name]['a']} b {out[name]['b']}")
    return out


def oracle_all(pool, spec, params, odors_of: dict, naive: dict, log=print) -> dict:
    """{name: dict(qualify(...), naive, alpha_reward, row)}; one run_jobs call for every pair."""
    names = list(spec.pair_names())
    xs = {n: qual_x(naive[n], spec) for n in names}
    kws = [oracle_kwargs(spec, params, n, odors_of[n], xs[n]) for n in names]
    rows = pool.run_jobs(reward_oracle_job, kws)
    z = {"A": spec.z_a, "P": spec.z_p}
    out = {}
    for n, row in zip(names, rows):
        out[n] = dict(qualify(xs[n], row["report"], naive[n], spec, z), naive=naive[n],
                      alpha_reward=row["alpha_reward"], row=row)
        log(f"{n}: X {xs[n]}, r {out[n]['r']}, floor_ok {out[n]['floor_ok']} "
            f"(silent share a {out[n]['silent_share']['a']}, b {out[n]['silent_share']['b']}) -> "
            f"{'qualified' if out[n]['qualified'] else 'not qualified ' + str(out[n]['reasons'])}")
    return out


def measure(naive_pool, oracle_pool, spec, params, odors_of: dict, idx_p, log=print) -> dict:
    """Both phases on given pools (the tests' fakes); the pair entries without the raw rows."""
    res = oracle_all(oracle_pool, spec, params, odors_of, naive_all(naive_pool, spec, odors_of, idx_p, log), log)
    return {n: {k: v for k, v in e.items() if k != "row"} for n, e in res.items()}


def summarize(spec, pairs: dict) -> dict:
    names = spec.pair_names()
    qualified = [n for n in names[1:] if pairs[n]["qualified"]]
    m = len(qualified)
    control = bool(pairs[spec.control]["qualified"])
    # spec 10.3: an unqualified seed0 stops first (STOP_CONTROL_INVALID), before the pair count (STOP_FEW_PAIRS)
    stop = "STOP_CONTROL_INVALID" if not control else ("STOP_FEW_PAIRS" if m < spec.min_pairs else None)
    return dict(pairs=pairs, qualified=qualified, m=m, stop=stop, control_qualified=control)



def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default="data/malecns.npz")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="results/rescope/qualify")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args(argv)
    spec = smoke_spec() if a.smoke else SPEC
    out, summary = paths(a.out, a.smoke, STAGE)
    guard(out / "oracle_rows.json", [Params()])
    guard(summary, [Params()])
    git = git_provenance(files=tracked_files("scripts/run_rescope_qualify.py"))
    if git["dirty"] and not a.allow_dirty:
        raise SystemExit(f"refusing: uncommitted changes in {git['dirty_files']} (commit them or pass --allow-dirty)")

    from flymon.agent.config import load_c3_config
    from flymon.brain.circuits import Populations
    from flymon.brain.connectome import Connectome
    from flymon.brain.h4_jobs import type_cells
    from flymon.rescope.pairs import new_pairs, pair_odors, pairs_digest

    cfg = load_c3_config()
    recovery = check_recovery(cfg.params)
    conn = Connectome.load(a.npz)
    pops = Populations.from_connectome(conn)
    cells = type_cells(conn, [spec.a_type, spec.p_type])
    digest = pairs_digest(new_pairs(pops, spec))
    if digest != spec.pairs_digest:
        raise SystemExit(f"refusing: the designed pairs' digest {digest[:12]} is not the pinned {spec.pairs_digest[:12]}")
    odors_of = {n: pair_odors(pops, n, spec) for n in spec.pair_names()}
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    log = lambda s: print(s, flush=True)
    with FlyPool(a.npz, cfg.params, [FlySpec() for _ in range(a.workers)], workers=a.workers,
                 timeout_s=POOL_TIMEOUT_S) as pool:
        naive = naive_all(pool, spec, odors_of, cells[spec.p_type], log)
    with FlyPool(a.npz, Params(), [{} for _ in range(a.workers)], workers=a.workers, punish_type=spec.punish_dan,
                 reward_type=spec.reward_dan, timeout_s=POOL_TIMEOUT_S) as pool:
        res = oracle_all(pool, spec, cfg.params, odors_of, naive, log)
    gp = [Params(), cfg.params]
    write_json(out / "oracle_rows.json", {n: e["row"] for n, e in res.items()}, gp)
    pairs = {n: {k: v for k, v in e.items() if k != "row"} for n, e in res.items()}
    summ = dict(summarize(spec, pairs), oc_table=oc_table(spec), pairs_digest=digest, smoke=a.smoke,
                recovery_per_pulse=recovery,
                spec=dataclasses.asdict(spec),
                provenance=dict(git=git, started_utc=started, finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                                argv=list(sys.argv[1:] if argv is None else argv), npz=a.npz))
    write_json(summary, summ, gp)
    log(f"qualified {summ['qualified']} (m = {summ['m']}), control qualified {summ['control_qualified']}, "
        f"stop {summ['stop']}; wrote {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
