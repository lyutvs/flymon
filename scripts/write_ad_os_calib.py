#!/usr/bin/env python3
"""Spec AD.5 3 / AD.6 2: the KC-activity calibration of FLY-OS on the naive L_V brain (one LeverFlyPool fly at w0),
written to results/summary/ad_os_kc_calib.json BEFORE any gate reads it (commit it, then run check_ad_smoke.py). Also
records the total-strength check |sum - k| over every (move type, species) OS odour (<= 1e-9; a failure is a code
defect, STOP_SMOKE, judged by the smoke check).

    uv run python scripts/write_ad_os_calib.py [--workers 16]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from flymon.ad import calib, os_encoder, store
from flymon.ad.spec import LABEL, SPEC


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--npz", default="data/malecns.npz")
    ap.add_argument("--inputs", default="results/summary/ac_inputs.json", help="the stage-1 confirmation set (AC.4)")
    a = ap.parse_args(argv)
    from flymon.ac.config import load_lv_config, lv_codebook
    from flymon.ac.spec import SPEC as AC
    from flymon.agent.encode_grid import GridEncoder
    from flymon.brain.circuits import Populations
    from flymon.brain.connectome import Connectome
    from flymon.brain.fly_pool import FlySpec
    from flymon.brain.h4_jobs import type_cells
    from flymon.brain.lv_pool import LeverFlyPool
    model = json.loads(Path("results/summary/ac_model_manifest.json").read_text())
    if model.get("status") != "FROZEN":
        raise SystemExit("refusing: the model manifest is not FROZEN")
    cfg = load_lv_config(float(model["recovery_per_pulse"]))
    cb, _ = lv_codebook()
    conn = Connectome.load(a.npz)
    pops = Populations.from_connectome(conn)
    grid = GridEncoder(pops, cb, SPEC.dual_rule)
    pairs = json.loads(Path(a.inputs).read_text())["confirm"]["pairs"]
    cells = type_cells(conn, [cfg.readout["A"], cfg.readout["P"]])
    na, npp = len(cells[cfg.readout["A"]]), len(cells[cfg.readout["P"]])
    idx = np.concatenate([cells[cfg.readout["A"]], cells[cfg.readout["P"]], np.asarray(pops.kc)])
    rows = []
    with LeverFlyPool(a.npz, cfg.params, [FlySpec()], edit=AC.lever_edit, p_type=AC.p_type, workers=a.workers,
                      timeout_s=7200) as pool:
        for rep in SPEC.calib_reps:
            meta, reqs = calib.requests(grid, pairs, rep, SPEC.tie_seed)
            counts = pool.decide_batch(reqs, cfg.strength, cfg.settle_ms, cfg.read_ms, idx)
            rows += calib.kc_rows(meta, counts, na + npp)
            print(f"rep {rep}: {len(reqs)} requests", flush=True)
    meas = calib.measure(rows, grid.move_info, grid.species_types)
    strength = os_encoder.total_strength_error(grid)
    doc = dict(meas, strength_error=strength, strength_k=int(cb.k), n_pairs=len(pairs), seeds_key="os-calib",
               tie_seed=SPEC.tie_seed, rows=rows, label=LABEL)
    store.write_json(calib.CALIB, doc)
    print(f"R by rep {[round(x, 4) for x in meas['R_by_rep']]}, R {meas['R']:.4f}, SD_R {meas['sd_R']:.4f}, "
          f"KS {meas['ks']:.3f}, strength error {strength:.2e}; wrote {calib.CALIB} (no gate judged here)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
