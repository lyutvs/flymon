#!/usr/bin/env python3
"""Spec appendix U (U.9 wins over U.0-U.8): the partial lever APL->MBON05 × f, each engine variant's own reference-set z
for the oracle, one M2 judgement on T's unused set. The controller runs every stage; commit each block before the next.

    uv run python scripts/run_u.py --stage reuse                     # U.3 1 R's repro and T's unedited z reused (no pool)
    uv run python scripts/run_u.py --stage path                      # U.9.1 endpoints f = 1 / f = 0 bit for bit
    uv run python scripts/run_u.py --stage set                       # U.2 T's set, list only (no pool)
    uv run python scripts/run_u.py --stage scan                      # U.3 3 (a) 9-point guard scan + U.9.3 contrast
    uv run python scripts/run_u.py --stage kc                        # U.3 3 (c) KC band for at most 3 smallest f
    uv run python scripts/run_u.py --stage even                      # U.3 4 C from R's even raw, L_f per qualified f
    uv run python scripts/run_u.py --stage choose                    # U.3 5 / U.9.2 the f choice (no pool)
    uv run python scripts/run_u.py --stage smoke --workers 4         # 24_509_xxx / 25_309_xxx, even (b) 0 and 20 only
    uv run python scripts/run_u.py --stage oc                        # U.6 both operating characteristics (no pool)
    uv run python scripts/run_u.py --stage gate2_oc                  # U.6 gate ② operating characteristic (no pool)
    uv run python scripts/run_u.py --stage gate2                     # ② P_L(f*) and P_C on 25_300_000+i, h4 z
    uv run python scripts/run_u.py --stage gate2 --rerun-after-invalid   # once, after a fixed INVALID (U.3 8)
    uv run python scripts/run_u.py --stage jm --condition L          # then C, E0: the judgement set, one per run
    uv run python scripts/run_u.py --stage seal                      # pre-read validity, manifest, archive (no pool)
    uv run python scripts/run_u.py --stage judge                     # bands, sentence, records (no pool)
    uv run python scripts/run_u.py --stage recompute --note "..."    # after judge: analysis defect (U.5)
    uv run python scripts/run_u.py --stage invalid_run --note "..."  # after judge: measurement defect (U.5)

Exit 0 recorded (PASS / SEALED / READ / a record stage), 3 a gate STOP (STOP_REUSE, STOP_U_PATH_REPRO, STOP_SET_SHORT,
STOP_NO_QUALIFIED_F, STOP_EVEN_REPRO, STOP_EVEN_PUNISH, STOP_EVEN_LOW_LEVER, gate ②'s three; recorded, U stops), 5 a
gate INVALID (path, scan, kc, even, choose, gate2), 6 seal not SEALED, judge NOT_READ or a smoke with problems, 7 the
reuse condition or a measurement key broke (U stops; no block), 2 a refusal (arguments, cwd, connectome sha256, chain,
uncommitted summary, dirty hashed file, data pins, R's even raw or T's z rows missing)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

NPZ = "data/malecns.npz"
EXIT_STOP, EXIT_INVALID, EXIT_NOT_READ = 3, 5, 6
STAGES = ("reuse", "path", "set", "scan", "kc", "even", "choose", "smoke", "oc", "gate2_oc", "gate2", "jm", "seal",
          "judge", "recompute", "invalid_run")
POOL_STAGES = ("path", "scan", "kc", "even", "smoke", "gate2", "jm")
GATE_STAGES = ("reuse", "path", "set", "scan", "kc", "even", "choose", "gate2")
QUIET = ("pairs", "manifest", "records", "p_judgement_L", "p_judgement_C", "conditions", "rows", "archive", "checks",
         "g_fail", "set", "sides", "points", "contrast", "per_f", "C", "L", "independent", "cluster")


def check_args(a) -> str | None:
    if not a.stage:
        return "--stage is required"
    if (a.stage == "jm") != (a.condition is not None):
        return "--condition goes with --stage jm only"
    if a.stage in ("recompute", "invalid_run") and not a.note:
        return "--note is required for recompute / invalid_run (U.5)"
    if a.note and a.stage not in ("recompute", "invalid_run"):
        return "--note goes with recompute / invalid_run only"
    if a.rerun_after_invalid and a.stage != "gate2":
        return "--rerun-after-invalid goes with --stage gate2 only"
    return None


def make_measure(stage: str, ucode: dict, pool, ctx: dict):
    """measure(z): one RMeasurer per z, every one with U's spec over one UPool (U's job copies) and one UCache keyed by
    the U measurement key and rooted at U's cache (results/u/cache, results/u/smoke/cache for smoke), so no U entry lands
    in R's or T's tree (UCache.put writes only through u_store's guard)."""
    from flymon.brain.h3_store import canonical
    from flymon.brain.r_measure import RMeasurer
    from flymon.brain.u_measure import UPool
    from flymon.brain.u_spec import SPEC
    from flymon.brain.u_store import UCache
    cache = UCache(SPEC.smoke_cache_dir if stage == "smoke" else SPEC.cache_dir, ucode)
    upool = UPool(pool)
    measurers = {}

    def measure(z):
        k = canonical({t: list(v) for t, v in z.items()})
        if k not in measurers:
            measurers[k] = RMeasurer(upool, cache, SPEC, ctx["params"], ctx["readout"], z, ctx["types"], ctx["n_kc"])
        return measurers[k]
    return measure


def exit_code(stage: str, out: dict) -> int:
    from flymon.brain import u_rules as R
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

    from flymon.brain import u_runner
    from flymon.brain.fly_pool import FlyPool
    from flymon.brain.odor_real import DataMismatch
    from flymon.brain.r_measure import R_MEASURE_FILES
    from flymon.brain.t_measure import t_measure_key
    from flymon.brain.u_measure import UZMeasurer, u_measure_key
    from flymon.brain.u_spec import SPEC

    try:
        ctx = u_runner.build_ctx(SPEC, NPZ)
        code, ucode = code_key(NPZ, files=R_MEASURE_FILES), u_measure_key(NPZ)
        workers = a.workers or SPEC.workers
        pool = None
        if a.stage in POOL_STAGES:
            pool = FlyPool(NPZ, ctx["params"], flies=[{}] * workers, workers=workers, punish_type=SPEC.punish_type,
                           reward_type=SPEC.reward_type, timeout_s=SPEC.pool_timeout_s)
        try:
            measure = make_measure(a.stage, ucode, pool, ctx)
            zm = UZMeasurer(pool, ctx["params"], SPEC.p_type, ctx["rec_types"], SPEC.kc_types)
            r = u_runner.Runner(measure, zm, ctx, SPEC, code=code, tcode=t_measure_key(NPZ), ucode=ucode,
                                pipeline=u_runner.pipeline_key())
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
