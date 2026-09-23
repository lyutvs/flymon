#!/usr/bin/env python3
"""Spec J.12.2: run one pair's B protocol on C0 and write its raw probe records (git-excluded).

    uv run python scripts/run_b_test.py --pair calibration          # the pilot: R, N, N2, noplast (~15-20 min)
    uv run python scripts/run_b_test.py --pair exploration          # a test pair: R, N, noplast (only after stage 2 reads B)
    uv run python scripts/run_b_test.py --pair calibration --smoke --allow-dirty    # 2 flies, 2 seeds, 2 trials

Run it from the repository root. The run checkpoints after every step under <out>/<pair>/checkpoint.npz and resumes
from it (same code key only). The raw record goes to <out>/<pair>/<run id>.json; scripts/write_b_summary.py derives
the summaries from it. This script is itself a measurement file (flymon/brain/b_files.py): what it measures is part of
the measurement key.

The raw record carries the provenance of the run that MEASURED (the checkpoint keeps the git state and start time of the
run that measured the first step): measured_git, measured_started_utc; written_git is this invocation's git state, and
replayed is true when every step was already done in the checkpoint at the start (nothing measured by this invocation).
scripts/write_b_summary.py refuses a raw whose measured_git has dirty hashed files. Exit codes: 0 done, 2 refused before
measuring (not at the root, dirty hashed files, not the declared connectome, unknown pair).
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import sys
import time
import uuid
from pathlib import Path

import numpy as np

from flymon.brain.b_files import HASHED_FILES, MEASURE_FILES
from flymon.brain.b_runner import fly_specs, layout, probe, run_pair, screen_calibration
from flymon.brain.b_spec import PAIRS, SPEC, BSpec
from flymon.brain.b_store import Checkpoint, write_json
from flymon.brain.circuits import Populations
from flymon.brain.config import Params
from flymon.brain.connectome import Connectome
from flymon.brain.fly_pool import FlyPool, FlySpec
from flymon.brain.h3_spec import SPEC as H3_SPEC
from flymon.brain.h3_store import ROOT, code_key, git_state
from flymon.brain.stimuli import design_odor_pair

POOL_TIMEOUT_S = 1800


def smoke_spec(spec: BSpec) -> BSpec:
    return dataclasses.replace(spec, n_flies=2, n_probe=2, trials=2, valid_min=1)


def refuse(why: str) -> int:
    print(f"refused: {why}", file=sys.stderr)
    return 2


def type_cells(conn, types) -> dict:
    t = np.asarray(conn.type).astype(str)
    return {n: np.flatnonzero(t == n) for n in types}


def main(argv=None, spec: BSpec | None = None, require_root: bool = True) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", required=True)
    ap.add_argument("--npz", default="data/malecns.npz")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args(argv)
    if a.pair not in PAIRS:
        return refuse(f"--pair must be one of {PAIRS}, got {a.pair!r}")
    if require_root and Path.cwd().resolve() != ROOT:
        return refuse(f"not at the repository root {ROOT}")
    spec = spec or (smoke_spec(SPEC) if a.smoke else SPEC)
    out = Path(a.out or ("results/b-smoke" if a.smoke else "results/b"))
    git = git_state(files=HASHED_FILES)
    if git["dirty_hashed"] and not a.allow_dirty:
        return refuse(f"hashed files are dirty {git['dirty_hashed']} (commit them, or --allow-dirty)")
    code, manifest = code_key(a.npz, files=MEASURE_FILES), code_key(a.npz, files=HASHED_FILES)
    npz_sha = code["files"]["npz:" + Path(a.npz).name]
    if npz_sha != H3_SPEC.connectome_sha256:
        return refuse(f"{a.npz} is not the declared connectome ({npz_sha[:12]})")
    now = dt.datetime.now(dt.timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    started_utc = now.isoformat(timespec="seconds")
    conn = Connectome.load(a.npz)
    pops = Populations.from_connectome(conn)
    pair_odors = lambda seed: dict(zip(("a", "b"), design_odor_pair(pops, k=spec.k, seed=seed)))
    cells = type_cells(conn, (spec.a_type, spec.p_type))
    del conn
    with_n2 = a.pair == "calibration"
    lay = layout(spec, with_n2)
    t0 = time.time()
    with FlyPool(a.npz, Params(), [FlySpec(**f) for f in fly_specs(lay)], workers=a.workers,
                 timeout_s=POOL_TIMEOUT_S) as pool:
        screening = None
        if a.pair == "calibration":                     # J.12.7: naive probes of the reward-arm brains only
            screening = screen_calibration(spec, lambda seed: probe(pool, spec, a.pair, lay[:spec.n_flies],
                                                                    pair_odors(seed), cells, "pre"))
            print(f"calibration pair: seed {screening['selected']} ({screening['rows']})", flush=True)
        pair_seed = screening["selected"] if screening else spec.pair_seed(a.pair)
        odors = pair_odors(pair_seed)
        ck = Checkpoint(out / a.pair, f"{code['key']}-seed{pair_seed}" + ("-smoke" if a.smoke else ""), [Params()])
        res = run_pair(pool, spec, a.pair, odors, cells, with_n2, dict(spec.fixed_x).get(a.pair), checkpoint=ck,
                       log=lambda s: print(s, flush=True), provenance=dict(git=git, started_utc=started_utc))
    measured = res["provenance"] or {}
    raw = dict(run_id=run_id, pair=a.pair, pair_seed=pair_seed, screening=screening, smoke=a.smoke,
               measured_git=measured.get("git"), measured_started_utc=measured.get("started_utc"), written_git=git,
               replayed=bool(res["replayed"]), code=manifest, measure_key=code["key"], spec=spec, odors=odors,
               x=res["x"], layout=res["layout"],
               wall_s=round(time.time() - t0, 1), records=res["records"])
    path = write_json(out / a.pair / f"{run_id}.json", raw, [Params()])
    if res["replayed"]:
        print(f"note: every step was already in {ck.path}; nothing was measured now (replayed)", flush=True)
    print(f"wrote {path}: X = odour {res['x']}, {len(res['records'])} probe records, {raw['wall_s'] / 60:.1f} min",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
