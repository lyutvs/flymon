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
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import glob
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
    a = ap.parse_args(argv)
    spec = smoke_spec() if a.smoke else SPEC
    out, summary = paths(a.out, a.smoke)
    guard(out / "x.json", [Params()])
    guard(summary, [Params()])
    git = git_provenance(files=sorted(glob.glob("flymon/rescope/*.py")) + ["scripts/run_rescope_taurec.py"])
    if git["dirty"] and not a.allow_dirty:
        raise SystemExit(f"refusing: uncommitted changes in {git['dirty_files']} (commit them or pass --allow-dirty)")

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


if __name__ == "__main__":
    sys.exit(main())
