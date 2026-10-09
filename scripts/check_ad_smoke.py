#!/usr/bin/env python3
"""Spec AD.5 / AD.6 2: read the AD smoke run (results/m4-smoke/ad/) and the committed KC calibration, run the gate
test subsets, reproduce stage 1's point-0 records of flies 0 and 12, re-measure the POOL odour response, check the
reuse fields (가), then write results/summary/ad_smoke.json.

    uv run pytest -q -o addopts="" > $L/suite.log 2>&1; echo $? > results/m4-smoke/ad/suite_exit.txt
    uv run python scripts/check_ad_smoke.py --suite-exit results/m4-smoke/ad/suite_exit.txt

Exit 0 OK, 2 STOP_SMOKE, 4 STOP_REUSE, 5 STOP_BALANCE. A missing input fails its gate (no traceback); the summary is
always written."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from flymon.ac import smoke as ac_smoke
from flymon.ac.manifest import MODEL_MANIFEST
from flymon.ad import calib, hooks, inputs, os_encoder, reuse, smoke, store
from flymon.ad.spec import LABEL, SPEC
from flymon.ad.spec import smoke as smoke_spec

EXIT = {"OK": 0, "STOP_SMOKE": 2, "STOP_REUSE": 4, "STOP_BALANCE": 5}
AC_INPUTS = "results/summary/ac_inputs.json"


def run_pytest(ids) -> int:
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-o", "addopts=", *ids]).returncode


def _json(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def brain_measures(a, pairs) -> dict:
    """Stage-1 reproduction (flies 0 and 12 at point 0, stage-1 seeds) and the naive-L_V POOL odour response."""
    from flymon.ac import confirm, situations
    from flymon.ac.config import load_lv_config, lv_codebook
    from flymon.ac.spec import SPEC as AC
    from flymon.agent.encode_grid import GridEncoder
    from flymon.agent.policy import derive_seed
    from flymon.brain.circuits import Populations
    from flymon.brain.connectome import Connectome
    from flymon.brain.fly_pool import FlySpec
    from flymon.brain.h4_jobs import type_cells
    from flymon.brain.lv_pool import LeverFlyPool
    model = json.loads(Path(MODEL_MANIFEST).read_text())
    cfg = load_lv_config(float(model["recovery_per_pulse"]))
    cb, _ = lv_codebook()
    conn = Connectome.load(a.npz)
    pops = Populations.from_connectome(conn)
    grid = GridEncoder(pops, cb, SPEC.dual_rule)
    cells = type_cells(conn, [cfg.readout["A"], cfg.readout["P"]])
    a_idx, p_idx = cells[cfg.readout["A"]], cells[cfg.readout["P"]]
    idx = np.concatenate([a_idx, p_idx, np.asarray(pops.kc)])
    ods = situations.pair_odours(grid, pairs)
    sits = confirm.situations_of(confirm.candidate_pairs()["pairs"])
    reqs = [(0, [situations.offline_odour(grid, m, opp) for m in cands], derive_seed(AC.tie_seed, "resp", i))
            for i, (_, opp, cands) in enumerate(sits)]
    with LeverFlyPool(a.npz, cfg.params, [FlySpec(), FlySpec()], edit=AC.lever_edit, p_type=AC.p_type,
                      workers=a.workers, timeout_s=7200) as pool:
        got = [dict(situations.evaluate(pool, i, 0, pairs, ods, cfg, a_idx, p_idx, pops.kc, AC.tie_seed, seed_fly=g),
                    fly=g) for i, g in enumerate((0, 12))]
        counts = pool.decide_batch(reqs, cfg.strength, cfg.settle_ms, cfg.read_ms, idx)
    resp = ac_smoke.pool_response(counts, len(a_idx), len(p_idx), cfg.z)
    return dict(reproduced=got, response=resp, strength_error=os_encoder.total_strength_error(grid))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite-exit", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--npz", default="data/malecns.npz")
    a = ap.parse_args(argv)
    s = smoke_spec()
    root = Path(s.out_root)
    res = _json(root / "result.json") or {}
    doc = _json(inputs.input_paths(s)[0]) or {}
    lay = s.layout()
    sits = hooks.ADSituationLog(root / "logs").records()
    paths = ac_smoke.log_paths(root)
    errs = ac_smoke.schema_errors(paths)
    os_flies = {i for i, (arm, _, _) in enumerate(lay) if arm == "FLYOS"}
    shadow = smoke.os_shadow_errors(paths, os_flies, os_seed=s.os_seed)
    played = smoke.eval_played(root / "logs" / "eval" / "battles.jsonl", doc.get("eval") or [], lay, s.tags[1])
    doc_eval = (doc.get("digests") or {}).get("eval")
    run_doc, disjoint_err = smoke.run_inputs_rebuilt(SPEC)          # AD.2: assert_disjoint as a gate, not a traceback
    ac_inputs = _json(AC_INPUTS) or {}
    pairs = (ac_inputs.get("confirm") or {}).get("pairs") or []
    suite = Path(a.suite_exit)
    gates = dict(
        full_suite=suite.exists() and suite.read_text().strip() == "0",
        safety_tests=run_pytest(ac_smoke.SAFETY_TESTS) == 0,
        worker_equals_in_process=run_pytest(ac_smoke.EQUALITY_TESTS) == 0,
        stale_interleaving_tests=run_pytest(smoke.STALE_TESTS) == 0,
        os_draw_tests=run_pytest(smoke.OS_TESTS) == 0,
        global_fly_tests=run_pytest(smoke.SEED_TESTS) == 0,
        manifest_refusal_tests=run_pytest(smoke.MANIFEST_TESTS) == 0,
        learn_eval_disjoint=disjoint_err is None,
        log_schema=bool(paths) and not errs,
        os_shadow_schema=bool(paths) and not shadow,
        complete_after_resume=bool(res.get("complete")) and ac_smoke.resumed(root),
        situation_evals_frozen=bool(sits) and all(r["frozen"] for r in sits),
        eval_digest_shared=bool(doc_eval) and smoke.eval_shared(played, doc_eval, res.get("eval_digest"), len(lay)))
    diff = None
    try:
        if not pairs:
            raise SystemExit(f"{AC_INPUTS} missing or without confirm.pairs")
        bm = brain_measures(a, pairs)
        p = reuse.stage1_paths(SPEC)
        want = reuse.stage1_fly_records(p["init"], (0, 12), 0)  # point-0 records of stage-1 flies 0 (FLY), 12 (C-off)
        diff = smoke.reproduction_diff(want, bm["reproduced"])
        ac_resp = json.loads(Path(ac_smoke.SMOKE).read_text())["pool_response"]
        gates.update(stage1_reproduction=not diff, os_total_strength=bm["strength_error"] <= SPEC.strength_tol,
                     pool_response_equals_stage1=smoke.same_response(bm["response"], ac_resp),
                     **ac_smoke.response_gates(bm["response"], SPEC))
    except (SystemExit, Exception) as e:
        bm, diff = dict(error=f"{type(e).__name__}: {e}"), (diff or []) + ["brain measures failed"]
        gates.update(stage1_reproduction=False, os_total_strength=False, pool_response_equals_stage1=False,
                     pool_tie=False, pool_a0=False, pool_p0=False)
    # AD.2 (가): the run-mode digests rebuilt from the seeds (stage 1's recorded ones if the rebuild failed), checked
    # against stage 1 and the run's input manifest if written - before any (나) check, so a drift is STOP_REUSE.
    digests = (run_doc or ac_inputs or {}).get("digests") or {}
    try:
        f, problems = smoke.reuse_check(dict(confirm=digests.get("confirm"), eval=digests.get("eval")), SPEC,
                                        _json(inputs.input_paths(SPEC)[1]))
        if (ac_inputs.get("digests") or {}).get("confirm") != reuse.CONFIRM_DIGEST or \
                (ac_inputs.get("digests") or {}).get("eval") != reuse.EVAL_DIGEST:
            problems.append("ac_inputs_digests")
    except (SystemExit, Exception) as e:
        f, problems = dict(error=f"{type(e).__name__}: {e}"), ["stage-1 files unreadable"]
    gates["reuse"] = not problems
    cal = _json(calib.CALIB)
    try:
        bal = calib.band(cal, SPEC) if cal else dict(status="STOP_BALANCE", error=f"{calib.CALIB} missing")
    except (KeyError, TypeError, ValueError) as e:
        bal = dict(status="STOP_BALANCE", error=f"{calib.CALIB} unreadable: {type(e).__name__}: {e}")
    gates["kc_balance"] = bal["status"] == "OK"
    groups = dict(smoke=[k for k in gates if k not in ("reuse", "kc_balance")], reuse=["reuse"], balance=["kc_balance"])
    st = smoke.status(gates, groups)
    out = dict(status=st, gates=gates, groups=groups, schema_errors=errs[:20], os_shadow_errors=shadow[:20],
               os_draws=smoke.os_draw_record(paths, os_flies), kc_ratio_gt2=ac_smoke.kc_ratio_frac(paths),
               eval_played=dict(played, doc_digest=doc_eval, result_digest=res.get("eval_digest"),
                                problems=played["problems"][:20]),
               learn_eval_disjoint_error=disjoint_err,
               reproduction_diff=diff[:20], brain={k: v for k, v in bm.items() if k != "reproduced"},
               reuse=f, reuse_problems=problems, kc_band=bal,
               calib_sha256=store.sha256_file(calib.CALIB) if cal else None,
               n_situation_records=len(sits), futility=res.get("futility"), label=LABEL,
               provenance=dict(git=store.git_provenance(), suite_exit=a.suite_exit))
    store.write_json(smoke.SMOKE, out)
    for k, v in gates.items():
        print(f"{'ok ' if v else 'FAILED'} {k}", flush=True)
    print(f"{st}; KC band {bal}; wrote {smoke.SMOKE}; {LABEL}", flush=True)
    return EXIT[st]


if __name__ == "__main__":
    sys.exit(main())
