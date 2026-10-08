#!/usr/bin/env python3
"""Spec AC.7 2b: read the smoke run (results/m4-smoke/) and run the gate tests, then write
results/summary/ac_smoke.json (status OK, or STOP_SMOKE with exit 2).

    uv run pytest -q -o addopts="" > .superpowers/sdd/2026-10-07-ac-m3-m4/suite.log 2>&1; \
        echo $? > results/m4-smoke/suite_exit.txt
    uv run python scripts/check_ac_smoke.py --suite-exit results/m4-smoke/suite_exit.txt

The full suite is the controller's (it runs ~20 min); this script runs only the gate subsets and the naive-L_V POOL
odour response on the 79 candidate situations (one LeverFlyPool fly, --workers workers)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from flymon.ac import smoke, stage1, store
from flymon.ac.spec import LABEL, SPEC
from flymon.ac.spec import smoke as smoke_spec


def run_pytest(ids) -> int:
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-o", "addopts=", *ids]).returncode


def pool_response(a) -> dict:
    from flymon.ac import confirm, situations
    from flymon.ac.config import grid_encoder, load_lv_config, load_recovery
    from flymon.agent.policy import derive_seed
    from flymon.brain.circuits import Populations
    from flymon.brain.connectome import Connectome
    from flymon.brain.fly_pool import FlySpec
    from flymon.brain.h4_jobs import type_cells
    from flymon.brain.lv_pool import LeverFlyPool
    r, _ = load_recovery()
    cfg = load_lv_config(r)
    conn = Connectome.load(a.npz)
    pops = Populations.from_connectome(conn)
    grid = grid_encoder(pops)
    sits = confirm.situations_of(confirm.candidate_pairs()["pairs"])
    cells = type_cells(conn, [cfg.readout["A"], cfg.readout["P"]])
    idx = np.concatenate([cells[cfg.readout["A"]], cells[cfg.readout["P"]], np.asarray(pops.kc)])
    reqs = [(0, [situations.offline_odour(grid, m, opp) for m in cands], derive_seed(SPEC.tie_seed, "resp", i))
            for i, (_, opp, cands) in enumerate(sits)]
    with LeverFlyPool(a.npz, cfg.params, [FlySpec()], edit=SPEC.lever_edit, p_type=SPEC.p_type, workers=a.workers,
                      timeout_s=3600) as pool:
        counts = pool.decide_batch(reqs, cfg.strength, cfg.settle_ms, cfg.read_ms, idx)
    return smoke.pool_response(counts, len(cells[cfg.readout["A"]]), len(cells[cfg.readout["P"]]), cfg.z)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite-exit", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--npz", default="data/malecns.npz")
    a = ap.parse_args(argv)
    root = Path(smoke_spec().out_root)
    res = {arm: (json.loads((root / arm / "result.json").read_text()) if (root / arm / "result.json").exists()
                 else None) for arm in ("BRAIN", "RND", "MAX")}
    brain = res["BRAIN"] or {}
    sits = stage1.SituationLog(root / "BRAIN" / "logs").records()
    paths = smoke.log_paths(root / "BRAIN")
    errs = smoke.schema_errors(paths)
    gates = dict(
        full_suite=Path(a.suite_exit).read_text().strip() == "0",
        safety_tests=run_pytest(smoke.SAFETY_TESTS) == 0,
        worker_equals_in_process=run_pytest(smoke.EQUALITY_TESTS) == 0,
        tb_odour_tests=run_pytest(smoke.TB_TESTS) == 0,
        log_schema=bool(paths) and not errs,
        brain_complete_after_resume=bool(brain.get("complete")) and smoke.resumed(root / "BRAIN"),
        nobrain_complete=all(res[x] and res[x].get("complete") for x in ("RND", "MAX")),
        coff_weights_unchanged=bool(brain.get("per_fly")) and all(
            row["coff_weights_unchanged"] for row in brain["per_fly"] if row["arm"] == "COFF"),
        eval_digest_shared=all(res.values()) and len({r["eval_digest"] for r in res.values()}) == 1,
        situation_evals_frozen=bool(sits) and all(r["frozen"] for r in sits))
    try:
        resp = pool_response(a)
        gates.update(smoke.response_gates(resp))
    except (SystemExit, Exception) as e:       # e.g. load_recovery refusing: POOL gates fail, summary still written
        resp = dict(error=f"{type(e).__name__}: {e}")
        gates.update(pool_tie=False, pool_a0=False, pool_p0=False)
    doc = dict(status=smoke.status(gates), gates=gates, pool_response=resp, kc_ratio_gt2=smoke.kc_ratio_frac(paths),
               schema_errors=errs[:20], n_situation_records=len(sits), label=LABEL,
               provenance=dict(git=store.git_provenance(), suite_exit=a.suite_exit))
    store.write_json(smoke.SMOKE, doc)
    for k, v in gates.items():
        print(f"{'ok ' if v else 'FAILED'} {k}", flush=True)
    print(f"{doc['status']}; POOL response {resp}; KC ratio > 2 {doc['kc_ratio_gt2']}; wrote {smoke.SMOKE}")
    return 0 if doc["status"] == "OK" else 2


if __name__ == "__main__":
    sys.exit(main())
