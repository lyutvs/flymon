#!/usr/bin/env python3
"""Re-scoped claim, primary test (spec 10.2 / 10.4): seed0 (the control) and then every qualified pair, each on the
Rr / N / N' / noplast layout from naive (flymon.rescope.primary.run_pair, resumable per step), judged by
rules.control_verdict / rules.pair_verdict and rules.overall.

    uv run python scripts/run_rescope_primary.py --npz data/malecns.npz --workers 16 --out results/rescope/primary
    uv run python scripts/run_rescope_primary.py --smoke --allow-dirty --workers 4

Refuses (SystemExit) unless results/summary/rescope_qualify.json (--smoke: results/rescope-smoke/summary/
rescope_qualify.json) exists with stop None and control_qualified; refuses an --out outside the re-scope trees,
without --allow-dirty a dirty flymon/rescope/ or this script, a C3 recovery_per_pulse other than 0 (spec 10.2), and a
designed-pairs digest that is not SPEC.pairs_digest or not the qualification summary's. Writes <out>/<pair>/
checkpoint.npz and records.json and results/summary/rescope_primary.json ({control, pairs, overall, oc_table,
recorded: {<pair>: {level, sign, naive, qual_x_vs_x, silent, naive_dprime, choice, self_change, xcore, xcore_mask,
flies}},
recovery_per_pulse, pairs_digest, min_weight_frac, provenance}); --smoke writes under results/rescope-smoke/ only.
The recorded items (spec 10.4 item 6, rules.recorded_items) have no verdict effect.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import glob
import json
import sys
from pathlib import Path

from flymon.brain.config import Params
from flymon.rescope import primary, rules
from flymon.rescope.spec import SPEC
from flymon.rescope.store import Checkpoint, git_provenance, guard, write_json

STAGE = "primary"
POOL_TIMEOUT_S = 1800


def smoke_spec(spec=SPEC):
    return dataclasses.replace(spec, n_flies=2, n_probe=2, trials=2, valid_min=1, n_qual_seeds=2)


def paths(out: str, smoke: bool, stage: str) -> tuple:
    """(out dir, summary path). --smoke forces both under results/rescope-smoke/."""
    if not smoke:
        return Path(out), Path("results/summary") / f"rescope_{stage}.json"
    o = Path(out)
    if not o.as_posix().startswith("results/rescope-smoke/"):
        o = Path("results/rescope-smoke") / (o.name or stage)
    return o, Path("results/rescope-smoke/summary") / f"rescope_{stage}.json"


def load_qualification(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"refusing: {path} does not exist (run scripts/run_rescope_qualify.py first)")
    q = json.loads(path.read_text())
    if q.get("stop") is not None:
        raise SystemExit(f"refusing: the qualification stopped ({q['stop']})")
    if not q.get("control_qualified"):
        raise SystemExit("refusing: the control pair seed0 is not qualified (STOP_CONTROL_INVALID)")
    return q


def recorded(records, x, spec, verdict, qual_x, xcore=None, floor_frac=None) -> dict:
    """Recorded, not judged: the level / sign medians, the naive MBON05 medians (Rr pre), X vs the qualification X and
    spec 10.4 item 6 (rules.recorded_items: naive d', choice ratios, N / N' self change, X-core MBON05 w/w0 Rr vs N
    and floor contact) and the per-seed probe state (rules.silent_states: active / silent, silent := MBON05 count
    < floor_spikes; spec 10.3 amendment 2026-09-28)."""
    return dict(level=verdict.get("level_median"), sign=verdict.get("sign_median"),
                naive={o: rules.naive_p(records, o, spec) for o in ("a", "b")},
                qual_x_vs_x=None if qual_x is None else bool(qual_x == x),
                silent=rules.silent_states(records, spec),
                **rules.recorded_items(records, x, spec, xcore=xcore, floor_frac=floor_frac))


def check_pairs_digest(digest: str, spec, qual: dict) -> str:
    """The designed pairs rebuilt from the connectome must be the pinned ones and the ones the qualification ran on."""
    if digest != spec.pairs_digest:
        raise SystemExit(f"refusing: the designed pairs' digest {digest[:12]} is not the pinned {spec.pairs_digest[:12]}")
    if qual.get("pairs_digest") != digest:
        raise SystemExit(f"refusing: the qualification's pairs digest {str(qual.get('pairs_digest'))[:12]} is not this "
                         f"run's {digest[:12]}")
    return digest


def run_all(pool, spec, qual: dict, odors_of: dict, cells: dict, out: Path, key: str, params_list, provenance,
            log=print, floor_frac=None) -> dict:
    """seed0 then every qualified pair; the pool is reset to its initial (naive) state before each pair. The X-core
    mask (both odours) is built on the naive pool before run_pair and its weights read right after it."""
    initial = pool.state()
    names = [spec.control] + list(qual["qualified"])
    control, pairs, rec = None, {}, {}
    for name in names:
        pool.load_state(initial)
        masks = primary.xcore_masks(pool, spec, name, odors_of[name], cells)
        ck = Checkpoint(out / name, f"{key}-{name}", params_list)
        r = primary.run_pair(pool, spec, name, odors_of[name], cells, checkpoint=ck, log=log, provenance=provenance)
        xw = primary.xcore_weights(pool, r["layout"], masks[r["x"]])
        write_json(out / name / "records.json", dict(pair=name, x=r["x"], layout=r["layout"], records=r["records"],
                                                     provenance=r["provenance"], replayed=r["replayed"]), params_list)
        if name == spec.control:
            control = rules.control_verdict(r["records"], r["x"], spec, qualified=True)
            full = rules.pair_verdict(r["records"], r["x"], spec, name)
            log(f"{name}: control {control['status']} (sign median {control['sign_median']})")
        else:
            full = pairs[name] = rules.pair_verdict(r["records"], r["x"], spec, name)
            log(f"{name}: {full['status']} {full['reasons']}")
        qx = (qual.get("pairs") or {}).get(name, {}).get("x")
        rec[name] = recorded(r["records"], r["x"], spec, full, qx, xcore=xw, floor_frac=floor_frac)
        rec[name]["xcore_mask"] = dict(seed=masks["seed"], n_active_kc=masks["n_active"],
                                       n_edges={o: int(masks[o].sum()) for o in ("a", "b")})
    return dict(control=control, pairs=pairs, overall=rules.overall(control, pairs, spec), oc_table=rules.oc_table(spec),
                recorded=rec)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default="data/malecns.npz")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="results/rescope/primary")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args(argv)
    spec = smoke_spec() if a.smoke else SPEC
    out, summary = paths(a.out, a.smoke, STAGE)
    guard(out / "x.json", [Params()])
    guard(summary, [Params()])
    _, qual_path = paths("results/rescope/qualify", a.smoke, "qualify")
    qual = load_qualification(qual_path)
    git = git_provenance(files=sorted(glob.glob("flymon/rescope/*.py")) + ["scripts/run_rescope_primary.py"])
    if git["dirty"] and not a.allow_dirty:
        raise SystemExit(f"refusing: uncommitted changes in {git['dirty_files']} (commit them or pass --allow-dirty)")

    from flymon.agent.config import load_c3_config
    from flymon.brain.b_runner import fly_specs
    from flymon.brain.circuits import Populations
    from flymon.brain.connectome import Connectome
    from flymon.brain.fly_pool import FlyPool, FlySpec
    from flymon.brain.h4_jobs import type_cells
    from flymon.rescope.pairs import new_pairs, pair_odors, pairs_digest

    cfg = load_c3_config()
    recovery = primary.check_recovery(cfg.params)
    conn = Connectome.load(a.npz)
    pops = Populations.from_connectome(conn)
    digest = check_pairs_digest(pairs_digest(new_pairs(pops, spec)), spec, qual)
    cells = type_cells(conn, [spec.a_type, spec.p_type])
    odors_of = {n: pair_odors(pops, n, spec) for n in [spec.control] + list(qual["qualified"])}
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    prov = dict(git=git, started_utc=started, argv=list(sys.argv[1:] if argv is None else argv), npz=a.npz,
                qualify=dict(path=str(qual_path), provenance=qual.get("provenance")))
    log = lambda s: print(s, flush=True)
    gp = [cfg.params]
    with FlyPool(a.npz, cfg.params, [FlySpec(**d) for d in fly_specs(primary.layout(spec))], workers=a.workers,
                 timeout_s=POOL_TIMEOUT_S) as pool:
        res = run_all(pool, spec, qual, odors_of, cells, out, git["commit"] + ("-smoke" if a.smoke else ""), gp, prov,
                      log, floor_frac=float(cfg.params.min_weight_frac))
    summ = dict(res, smoke=a.smoke, spec=dataclasses.asdict(spec), recovery_per_pulse=recovery, pairs_digest=digest,
                min_weight_frac=float(cfg.params.min_weight_frac),
                provenance=dict(prov, finished_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
    write_json(summary, summ, gp)
    log(f"overall {res['overall']['status']} (m {res['overall']['m']}, t {res['overall']['t']}, "
        f"pass {res['overall']['n_pass']}); wrote {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
