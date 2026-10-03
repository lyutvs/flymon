#!/usr/bin/env python3
"""Spec appendix S (S.9 wins over S.0-S.8): the same lever (APL->MBON05 removal) on the last unused judgement set (L
generator turns 104-209), one M2 judgement. The controller runs every stage; commit each block before the next.

    uv run python scripts/run_s.py --stage reuse                     # S.3 ① / S.9.5 R's gates reused (no pool)
    uv run python scripts/run_s.py --stage set                       # S.2 the set, list only (no pool)
    uv run python scripts/run_s.py --stage smoke --workers 4         # 24_309_xxx / 25_109_xxx, even (b) 0 and 20 only
    uv run python scripts/run_s.py --stage oc                        # S.6 operating characteristic (no pool)
    uv run python scripts/run_s.py --stage gate2_oc                  # S.9.7 gate ② operating characteristic (no pool)
    uv run python scripts/run_s.py --stage gate2                     # ② P_L and P_C on 25_100_000+i
    uv run python scripts/run_s.py --stage gate2 --rerun-after-invalid   # once, after a fixed INVALID (S.3 ④)
    uv run python scripts/run_s.py --stage jm --condition L          # then C, E0: the judgement set, one per run
    uv run python scripts/run_s.py --stage seal                      # pre-read validity, manifest, archive (no pool)
    uv run python scripts/run_s.py --stage judge                     # bands, sentence, records (no pool)
    uv run python scripts/run_s.py --stage recompute --note "..."    # after judge: analysis defect (S.5)
    uv run python scripts/run_s.py --stage invalid_run --note "..."  # after judge: measurement defect (S.5)

Exit 0 recorded (PASS / SEALED / READ / a record stage), 3 a gate STOP (STOP_REUSE, STOP_SET_SHORT, gate ②'s three
STOPs; recorded, S stops), 5 gate ② INVALID, 6 seal not SEALED, judge NOT_READ or a smoke with problems, 7 R's reuse
condition broke after `reuse` (S stops; no block), 2 a refusal (arguments, cwd, connectome sha256, chain, uncommitted
summary, dirty hashed file, data pins)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

NPZ = "data/malecns.npz"
EXIT_STOP, EXIT_INVALID, EXIT_NOT_READ = 3, 5, 6
STAGES = ("reuse", "set", "smoke", "oc", "gate2_oc", "gate2", "jm", "seal", "judge", "recompute", "invalid_run")
POOL_STAGES = ("smoke", "gate2", "jm")
QUIET = ("pairs", "manifest", "records", "p_judgement_L", "p_judgement_C", "conditions", "rows", "archive", "checks",
         "g_fail", "set")


def check_args(a) -> str | None:
    if not a.stage:
        return "--stage is required"
    if (a.stage == "jm") != (a.condition is not None):
        return "--condition goes with --stage jm only"
    if a.stage in ("recompute", "invalid_run") and not a.note:
        return "--note is required for recompute / invalid_run (S.5)"
    if a.note and a.stage not in ("recompute", "invalid_run"):
        return "--note goes with recompute / invalid_run only"
    if a.rerun_after_invalid and a.stage != "gate2":
        return "--rerun-after-invalid goes with --stage gate2 only"
    return None


def exit_code(stage: str, out: dict) -> int:
    from flymon.brain import s_rules as R
    if stage in ("reuse", "set", "gate2"):
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

    from flymon.brain import s_runner
    from flymon.brain.fly_pool import FlyPool
    from flymon.brain.odor_real import DataMismatch
    from flymon.brain.r_measure import R_MEASURE_FILES, RMeasurer
    from flymon.brain.s_spec import SPEC
    from flymon.brain.s_store import SCache

    try:
        ctx = s_runner.build_ctx(SPEC, NPZ)
        code = code_key(NPZ, files=R_MEASURE_FILES)
        workers = a.workers or SPEC.workers
        root = SPEC.smoke_cache_dir if a.stage == "smoke" else SPEC.cache_dir
        pool = None
        if a.stage in POOL_STAGES:
            pool = FlyPool(NPZ, ctx["params"], flies=[{}] * workers, workers=workers, punish_type=SPEC.punish_type,
                           reward_type=SPEC.reward_type, timeout_s=SPEC.pool_timeout_s)
        try:
            m = RMeasurer(pool, SCache(root, code), SPEC, ctx["params"], ctx["readout"], ctx["z"], ctx["types"],
                          ctx["n_kc"])
            r = s_runner.Runner(m, ctx, SPEC, code=code, pipeline=s_runner.pipeline_key())
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
