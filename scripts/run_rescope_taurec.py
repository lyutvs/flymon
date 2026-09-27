#!/usr/bin/env python3
"""Re-scoped claim, stage 1 (spec 10.6): pick recovery_per_pulse on synthetic pulses (flymon.rescope.taurec).

    uv run python scripts/run_rescope_taurec.py --npz data/malecns.npz --out results/rescope/taurec
    uv run python scripts/run_rescope_taurec.py --smoke --allow-dirty

taurec_odours battle-encoder odours (generator seed taurec_gen_seed); the taught edges = plastic edges in the reward or
punish DAN core compartment whose KC fires on a naive read of odours[0]; then per recovery value a fresh two-fly pool
(C3 Params with recovery_per_pulse replaced) runs taurec.trajectory, and taurec.select picks the smallest value whose
alt path minimum is >= median_floor (else STOP_NO_RECOVERY). Refuses an --out outside the re-scope trees and, without
--allow-dirty, a dirty flymon/rescope/ or this script. Writes <out>/r_<value>.json per recovery value and
results/summary/rescope_taurec.json ({status, recovery_per_pulse, path_min, trajectories, odours_seed, n_taught_edges,
n_active_kc, spec, smoke, provenance}); --smoke (20 pulses every 5, 4 odours, grid (0.0, 0.02)) writes under
results/rescope-smoke/ only.

    uv run python scripts/run_rescope_taurec.py --reselect --out results/rescope/taurec

--reselect (spec 10.6 amendment 2026-09-28) runs no trajectory: it reads <out>/r_<value>.json for every grid value
(refuses if any is missing or names another value), applies the amended taurec.select (alt median path min >=
median_floor and alt floor_frac_taught path max <= taurec_taught_floor_max) and rewrites the summary with
rule "10.6 amendment 2026-09-28", the per-r files' sha256, and the previous old-rule summary's selection (if one exists)
under superseded_rule_v1. Run it only after the grid run has exited (the grid run writes its own summary at the end).
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import glob
import hashlib
import json
import sys
from pathlib import Path

from flymon.brain.config import Params
from flymon.rescope.spec import SPEC
from flymon.rescope.store import git_provenance, guard, write_json

STAGE = "taurec"
POOL_TIMEOUT_S = 1800


def smoke_spec(spec=SPEC):
    return dataclasses.replace(spec, taurec_pulses=20, taurec_sample_every=5, taurec_odours=4,
                               recovery_grid=(0.0, 0.02))


def paths(out: str, smoke: bool, stage: str = STAGE) -> tuple:
    """(out dir, summary path). --smoke forces both under results/rescope-smoke/."""
    if not smoke:
        return Path(out), Path("results/summary") / f"rescope_{stage}.json"
    o = Path(out)
    if not o.as_posix().startswith("results/rescope-smoke/"):
        o = Path("results/rescope-smoke") / (o.name or stage)
    return o, Path("results/rescope-smoke/summary") / f"rescope_{stage}.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default="data/malecns.npz")
    ap.add_argument("--out", default="results/rescope/taurec")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--reselect", action="store_true",
                    help="re-apply the amended selection (spec 10.6, 2026-09-28) to the per-r files; no trajectories")
    a = ap.parse_args(argv)
    spec = smoke_spec() if a.smoke else SPEC
    out, summary = paths(a.out, a.smoke)
    guard(out / "x.json", [Params()])
    guard(summary, [Params()])
    git = git_provenance(files=sorted(glob.glob("flymon/rescope/*.py")) + ["scripts/run_rescope_taurec.py"])
    if git["dirty"] and not a.allow_dirty:
        raise SystemExit(f"refusing: uncommitted changes in {git['dirty_files']} (commit them or pass --allow-dirty)")
    if a.reselect:
        return reselect(out, summary, spec, a.smoke, git, list(sys.argv[1:] if argv is None else argv))

    from flymon.agent.config import load_c3_config
    from flymon.brain.circuits import Populations
    from flymon.brain.connectome import Connectome
    from flymon.brain.fly_pool import FlyPool, FlySpec
    from flymon.rescope import taurec

    cfg = load_c3_config()
    pops = Populations.from_connectome(Connectome.load(a.npz))
    odours = taurec.synthetic_odours(pops, spec.taurec_odours, spec.taurec_gen_seed)
    plan = taurec.pulse_plan(spec)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    prov = dict(git=git, started_utc=started, argv=list(sys.argv[1:] if argv is None else argv), npz=a.npz)
    log = lambda s: print(s, flush=True)
    with FlyPool(a.npz, cfg.params, [FlySpec()], workers=1, timeout_s=POOL_TIMEOUT_S) as pool:
        mask, n_active = taurec.taught_mask(pool, odours[0], spec)
    log(f"taught edges {int(mask.sum())} (active KCs {n_active})")
    results, gp = {}, [cfg.params]
    for r in spec.recovery_grid:
        params = dataclasses.replace(cfg.params, recovery_per_pulse=float(r))
        with FlyPool(a.npz, params, [FlySpec(), FlySpec()], workers=2, timeout_s=POOL_TIMEOUT_S) as pool:
            results[float(r)] = taurec.trajectory(pool, odours, plan, spec, mask)
        write_json(out / f"r_{r}.json", dict(recovery_per_pulse=float(r), trajectory=results[float(r)]), gp)
        log(f"r={r}: alt path min {min(results[float(r)]['alt']['ratio']):.3f}, "
            f"same path min {min(results[float(r)]['same']['ratio']):.3f}")
    sel = taurec.select(results, spec)
    summ = dict(sel, trajectories={str(r): t for r, t in results.items()}, odours_seed=spec.taurec_gen_seed,
                n_taught_edges=int(mask.sum()), n_active_kc=n_active, smoke=a.smoke, spec=dataclasses.asdict(spec),
                provenance=dict(prov, finished_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
    write_json(summary, summ, gp)
    log(f"{sel['recovery_per_pulse'] if sel['status'] == 'SELECTED' else sel['status']}; wrote {summary}")
    return 0


SUPERSEDED_KEYS = ("status", "recovery_per_pulse", "path_min", "provenance")
CARRY_KEYS = ("odours_seed", "n_taught_edges", "n_active_kc")


def load_grid(out: Path, spec) -> tuple:
    """({r: trajectory}, {file name: sha256}) from <out>/r_<value>.json for every value of spec.recovery_grid."""
    missing = [f"r_{r}.json" for r in spec.recovery_grid if not (out / f"r_{r}.json").exists()]
    if missing:
        raise SystemExit(f"refusing --reselect: {out} lacks {missing} (the grid run is not complete)")
    results, sha = {}, {}
    for r in spec.recovery_grid:
        p = out / f"r_{r}.json"
        raw = p.read_bytes()
        d = json.loads(raw)
        if float(d.get("recovery_per_pulse", float("nan"))) != float(r):
            raise SystemExit(f"refusing --reselect: {p} records recovery_per_pulse {d.get('recovery_per_pulse')!r}, not {r}")
        results[float(r)], sha[p.name] = d["trajectory"], hashlib.sha256(raw).hexdigest()
    return results, sha


def reselect(out: Path, summary: Path, spec, smoke: bool, git: dict, argv: list) -> int:
    """Spec 10.6 amendment 2026-09-28: the amended selection on the recorded grid; no trajectory is re-run."""
    from flymon.agent.config import load_c3_config
    from flymon.rescope import taurec
    results, sha = load_grid(out, spec)
    old = json.loads(summary.read_text()) if summary.exists() else None
    if old is not None and old.get("rule") == taurec.RULE:
        superseded = old.get("superseded_rule_v1")          # a re-run of --reselect keeps the original old-rule record
    elif old is not None:
        superseded = {k: old.get(k) for k in SUPERSEDED_KEYS}
    else:
        superseded = None
    sel = taurec.select(results, spec)
    summ = dict(sel, superseded_rule_v1=superseded, trajectories={str(r): t for r, t in results.items()},
                **{k: (old or {}).get(k) for k in CARRY_KEYS}, smoke=smoke, spec=dataclasses.asdict(spec),
                provenance=dict(git=git, argv=argv, reselect=True, per_r_sha256=sha, source_dir=out.as_posix(),
                                finished_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
    write_json(summary, summ, [load_c3_config().params])
    for r in spec.recovery_grid:
        print(f"r={r}: alt path min {sel['path_min'][str(float(r))]:.3f}, alt floor_frac_taught path max "
              f"{sel['floor_frac_taught_path_max'][str(float(r))]}, fails {sel['failed'][str(float(r))] or 'none'}", flush=True)
    print(f"{sel['recovery_per_pulse'] if sel['status'] == 'SELECTED' else sel['status']} ({taurec.RULE}); wrote {summary}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
