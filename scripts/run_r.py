#!/usr/bin/env python3
"""Spec appendix R (R.9 wins over R.0-R.8): the lever APL->MBON05 removal and its one M2 judgement on the encoder
track's unused judgement set. The controller runs every stage; commit each block before the next.

    uv run python scripts/run_r.py --list-refs                       # encoder ③ entries to copy (no pool, no block)
    uv run python scripts/run_r.py --stage repro                     # R.9.6 no-edit reproduction gate
    uv run python scripts/run_r.py --stage smoke --workers 4         # 25_008_xxx / 25_009_xxx, even (b) 0 and 20 only
    uv run python scripts/run_r.py --stage oc                        # R.6 / R.9.5 operating characteristic (no pool)
    uv run python scripts/run_r.py --stage gate1                     # ① validity under the lever (R.9.2)
    uv run python scripts/run_r.py --stage gate2                     # ② P replication with the lever
    uv run python scripts/run_r.py --stage gate2 --rerun-after-invalid   # once, after a fixed INVALID (R.5)
    uv run python scripts/run_r.py --stage even                      # ③ L, C, E0 on H.4 seeds (one pool, resumable)
    uv run python scripts/run_r.py --stage gate3                     # ③ read (no pool)
    uv run python scripts/run_r.py --stage jm --condition L          # then C, E0: the judgement set, one per run
    uv run python scripts/run_r.py --stage seal                      # pre-read validity, manifest, archive (no pool)
    uv run python scripts/run_r.py --stage judge                     # bands, sentence, records (no pool)
    uv run python scripts/run_r.py --stage recompute --note "..."    # after judge: analysis defect (R.5)
    uv run python scripts/run_r.py --stage invalid_run --note "..."  # after judge: measurement defect (R.5)

Exit 0 recorded (PASS / SEALED / READ / a record stage), 3 a gate STOP (recorded; R stops), 4 repro failed (or a
later stage after a failed repro), 5 a gate INVALID, 6 seal not SEALED, judge NOT_READ or a smoke with problems, 2 a
refusal (arguments, cwd, connectome sha256, chain, uncommitted
summary, dirty hashed file, data pins)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

NPZ = "data/malecns.npz"
EXIT_STOP, EXIT_REPRO_FAIL, EXIT_INVALID, EXIT_NOT_READ = 3, 4, 5, 6
STAGES = ("repro", "smoke", "oc", "gate1", "gate2", "even", "gate3", "jm", "seal", "judge", "recompute", "invalid_run")
POOL_STAGES = ("repro", "smoke", "gate1", "gate2", "even", "jm")
QUIET = ("pairs", "manifest", "records", "p_judgement", "conditions", "rows", "odours", "archive", "checks", "record",
         "g_fail", "rows_detail")


def check_args(a) -> str | None:
    if a.list_refs:
        return "--list-refs runs alone" if a.stage else None
    if not a.stage:
        return "--stage is required"
    if (a.stage == "jm") != (a.condition is not None):
        return "--condition goes with --stage jm only"
    if a.stage in ("recompute", "invalid_run") and not a.note:
        return "--note is required for recompute / invalid_run (R.5)"
    if a.note and a.stage not in ("recompute", "invalid_run"):
        return "--note goes with recompute / invalid_run only"
    if a.rerun_after_invalid and a.stage != "gate2":
        return "--rerun-after-invalid goes with --stage gate2 only"
    return None


def exit_code(stage: str, out: dict) -> int:
    from flymon.brain import r_rules as R
    if stage == "repro":
        return 0 if out.get("passed") else EXIT_REPRO_FAIL
    if stage in ("gate1", "gate2", "gate3"):
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
    ap.add_argument("--list-refs", action="store_true")
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

    from flymon.brain import r_runner
    from flymon.brain.fly_pool import FlyPool
    from flymon.brain.odor_real import DataMismatch
    from flymon.brain.r_measure import R_MEASURE_FILES, RMeasurer
    from flymon.brain.r_spec import SPEC
    from flymon.brain.r_store import RCache

    try:
        ctx = r_runner.build_ctx(SPEC, NPZ)
        if a.list_refs:
            for oid in SPEC.kc_repro_odours:
                print(f"{oid}\t{ctx['kc_ref'](oid)}")
            return 0
        code = code_key(NPZ, files=R_MEASURE_FILES)
        workers = a.workers or SPEC.workers
        root = SPEC.smoke_cache_dir if a.stage == "smoke" else SPEC.cache_dir
        pool = None
        if a.stage in POOL_STAGES:
            pool = FlyPool(NPZ, ctx["params"], flies=[{}] * workers, workers=workers, punish_type=SPEC.punish_type,
                           reward_type=SPEC.reward_type, timeout_s=SPEC.pool_timeout_s)
        try:
            m = RMeasurer(pool, RCache(root, code), SPEC, ctx["params"], ctx["readout"], ctx["z"], ctx["types"],
                          ctx["n_kc"])
            r = r_runner.Runner(m, ctx, SPEC, code=code)
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
