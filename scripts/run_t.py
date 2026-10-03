#!/usr/bin/env python3
"""Spec appendix T (T.9 wins over T.0-T.8): each engine variant's own reference-set z for the oracle, the same lever
(APL->MBON05 removal), one M2 judgement on a new set from the widened Gen-1 opponent pool. The controller runs every
stage; commit each block before the next.

    uv run python scripts/run_t.py --stage reuse                     # T.3 1 R's repro and gate ① reused (no pool)
    uv run python scripts/run_t.py --stage set                       # T.9.1 the set, list only (no pool)
    uv run python scripts/run_t.py --stage z                         # T.1 / T.9.4 reference-set z, unedited then lever
    uv run python scripts/run_t.py --stage gate1s                    # T.9.1 KC band on the T set's 55 odours
    uv run python scripts/run_t.py --stage smoke --workers 4         # 24_409_xxx / 25_209_xxx, even (b) 0 and 20 only
    uv run python scripts/run_t.py --stage oc                        # T.6 both operating characteristics (no pool)
    uv run python scripts/run_t.py --stage gate2_oc                  # T.6 gate ② operating characteristic (no pool)
    uv run python scripts/run_t.py --stage gate2                     # ② P_L and P_C on 25_200_000+i, h4 z
    uv run python scripts/run_t.py --stage gate2 --rerun-after-invalid   # once, after a fixed INVALID (T.3 6)
    uv run python scripts/run_t.py --stage gate3                     # T.9.2 C from R's even raw, L on z_lever
    uv run python scripts/run_t.py --stage jm --condition L          # then C, E0: the judgement set, one per run
    uv run python scripts/run_t.py --stage seal                      # pre-read validity, manifest, archive (no pool)
    uv run python scripts/run_t.py --stage judge                     # bands, sentence, records (no pool)
    uv run python scripts/run_t.py --stage recompute --note "..."    # after judge: analysis defect (T.5)
    uv run python scripts/run_t.py --stage invalid_run --note "..."  # after judge: measurement defect (T.5)
    uv run python scripts/run_t.py --stage rs_reread                 # after judge: R / S raw under T's z (record)

Exit 0 recorded (PASS / SEALED / READ / a record stage), 3 a gate STOP (STOP_REUSE, STOP_SET_SHORT, STOP_Z_REPRO,
STOP_Z_DEGENERATE, STOP_STRENGTH_LEVER, gate ②'s three, STOP_EVEN_REPRO, STOP_EVEN_LOW_LEVER, STOP_C_EVEN_MISMATCH;
recorded, T stops), 5 a gate INVALID (z, gate1s, gate2, gate3), 6 seal not SEALED, judge NOT_READ or a smoke with
problems, 7 R's reuse condition or the T measurement key broke (T stops; no block), 2 a refusal (arguments, cwd,
connectome sha256, chain, uncommitted summary, dirty hashed file, data pins, R's even raw missing).
After a recorded STOP_REUSE (exit 3) every later stage refuses with exit 2 (the chain stops at a non-PASS gate);
exit 7 means the reuse condition (or the T measurement key) broke after a PASS was recorded."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

NPZ = "data/malecns.npz"
EXIT_STOP, EXIT_INVALID, EXIT_NOT_READ = 3, 5, 6
STAGES = ("reuse", "set", "z", "gate1s", "smoke", "oc", "gate2_oc", "gate2", "gate3", "jm", "seal", "judge",
          "recompute", "invalid_run", "rs_reread")
POOL_STAGES = ("z", "gate1s", "smoke", "gate2", "gate3", "jm")
GATE_STAGES = ("reuse", "set", "z", "gate1s", "gate2", "gate3")
QUIET = ("pairs", "manifest", "records", "p_judgement_L", "p_judgement_C", "conditions", "rows", "archive", "checks",
         "g_fail", "set", "none", "lever", "record", "independent", "cluster", "C", "L", "L_h4_R")


def check_args(a) -> str | None:
    if not a.stage:
        return "--stage is required"
    if (a.stage == "jm") != (a.condition is not None):
        return "--condition goes with --stage jm only"
    if a.stage in ("recompute", "invalid_run") and not a.note:
        return "--note is required for recompute / invalid_run (T.5)"
    if a.note and a.stage not in ("recompute", "invalid_run"):
        return "--note goes with recompute / invalid_run only"
    if a.rerun_after_invalid and a.stage != "gate2":
        return "--rerun-after-invalid goes with --stage gate2 only"
    return None


def make_measure(stage: str, code: dict, pool, ctx: dict):
    """measure(z): one RMeasurer per z (plan Reading 4), every one with T's spec over one TCache rooted at T's cache
    (results/t/cache, results/t/smoke/cache for smoke), so no T entry — gate ③'s L even pairs included — lands in R's
    or S's tree (TCache.put writes only through t_store's guard)."""
    from flymon.brain.h3_store import canonical
    from flymon.brain.r_measure import RMeasurer
    from flymon.brain.t_spec import SPEC
    from flymon.brain.t_store import TCache
    cache = TCache(SPEC.smoke_cache_dir if stage == "smoke" else SPEC.cache_dir, code)
    measurers = {}

    def measure(z):
        k = canonical({t: list(v) for t, v in z.items()})
        if k not in measurers:
            measurers[k] = RMeasurer(pool, cache, SPEC, ctx["params"], ctx["readout"], z, ctx["types"], ctx["n_kc"])
        return measurers[k]
    return measure


def exit_code(stage: str, out: dict) -> int:
    from flymon.brain import t_rules as R
    if stage in GATE_STAGES:
        o = out.get("outcome")
        return 0 if o == R.PASS else (EXIT_INVALID if o == R.INVALID else EXIT_STOP)
    if stage == "smoke":
        return EXIT_NOT_READ if out.get("problems") else 0
    if stage == "seal":
        return 0 if out.get("status") == R.SEALED else EXIT_NOT_READ
    if stage == "judge":
        return 0 if out.get("status") == R.READ else EXIT_NOT_READ
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=STAGES)
    ap.add_argument("--condition", choices=("L", "C", "E0"))
    ap.add_argument("--workers", type=int)
    ap.add_argument("--note")
    ap.add_argument("--rerun-after-invalid", action="store_true")
    a = ap.parse_args(argv)
    why = check_args(a)
    if why:
        print(f"refusing: {why}", file=sys.stderr)
        return 2

    from flymon.brain.h3_spec import SPEC as H3
    from flymon.brain.h3_store import ROOT, code_key, sha256_file
    if Path.cwd().resolve() != ROOT:
        print(f"refusing: run from the repository root {ROOT}", file=sys.stderr)
        return 2
    if not Path(NPZ).exists() or sha256_file(NPZ) != H3.connectome_sha256:
        print(f"refusing: {NPZ} is missing or its sha256 is not {H3.connectome_sha256}", file=sys.stderr)
        return 2

    from flymon.brain import t_runner
    from flymon.brain.fly_pool import FlyPool
    from flymon.brain.odor_real import DataMismatch
    from flymon.brain.r_measure import R_MEASURE_FILES
    from flymon.brain.t_measure import ZMeasurer, t_measure_key
    from flymon.brain.t_spec import SPEC

    try:
        ctx = t_runner.build_ctx(SPEC, NPZ)
        code = code_key(NPZ, files=R_MEASURE_FILES)
        workers = a.workers or SPEC.workers
        pool = None
        if a.stage in POOL_STAGES:
            pool = FlyPool(NPZ, ctx["params"], flies=[{}] * workers, workers=workers, punish_type=SPEC.punish_type,
                           reward_type=SPEC.reward_type, timeout_s=SPEC.pool_timeout_s)
        try:
            measure = make_measure(a.stage, code, pool, ctx)
            zm = ZMeasurer(pool, ctx["params"], SPEC.p_type, ctx["types"])
            r = t_runner.Runner(measure, zm, ctx, SPEC, code=code, tcode=t_measure_key(NPZ),
                                pipeline=t_runner.pipeline_key())
            if a.stage == "jm":
                out = r.stage_jm(a.condition)
            elif a.stage == "gate2":
                out = r.stage_gate2(rerun=a.rerun_after_invalid)
            elif a.stage in ("recompute", "invalid_run"):
                out = getattr(r, f"stage_{a.stage}")(a.note)
            else:
                out = getattr(r, f"stage_{a.stage}")()
        finally:
            if pool is not None:
                pool.close()
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 2
    except DataMismatch as e:
        print(f"refusing: the Hallem data do not match their pins: {e}", file=sys.stderr)
        return 2
    if out.get("sentence"):
        print(out["sentence"])
    print(json.dumps({k: v for k, v in out.items() if k not in QUIET}, ensure_ascii=False, default=str)[:2000])
    return exit_code(a.stage, out)


if __name__ == "__main__":
    sys.exit(main())
