#!/usr/bin/env python3
"""Spec appendix V (V.9 wins over V.0-V.8): the combined lever L_V (APL->MBON05 removed + MBON05->MBON09 / MBON11 /
MBON01 chain entry cut, 13 CSC edges), each engine variant's own reference-set z for the oracle, a new set from T's
widened-pool generator filtered by both engines' per-odour KC input, one M2 judgement. The controller runs every stage;
commit each block before the next.

    uv run python scripts/run_v.py --stage reuse                     # V.3 1 R / T / U reused (no pool)
    uv run python scripts/run_v.py --stage path                      # V.3 2 three engines + 3 + 3 even pairs, no cache
    uv run python scripts/run_v.py --stage kc_input                  # V.3 3 candidate odours on both engines
    uv run python scripts/run_v.py --stage set                       # V.2 the set, list only (no pool)
    uv run python scripts/run_v.py --stage z                         # V.3 5 z_V from block path (no pool)
    uv run python scripts/run_v.py --stage kc_band                   # V.3 6 112 calibration odours + set re-check
    uv run python scripts/run_v.py --stage even                      # V.3 7 C from R's even raw, L_V on z_V
    uv run python scripts/run_v.py --stage smoke --workers 4         # 24_609_xxx / 25_409_xxx, even (b) 0 and 20 only
    uv run python scripts/run_v.py --stage oc                        # V.6 both operating characteristics (no pool)
    uv run python scripts/run_v.py --stage gate2_oc                  # V.6 gate ② operating characteristic (no pool)
    uv run python scripts/run_v.py --stage gate2                     # ② P_L(V) and P_C on 25_400_000+i (h4 z, z_V ratio)
    uv run python scripts/run_v.py --stage gate2 --rerun-after-invalid   # once, after a fixed INVALID (V.3 10)
    uv run python scripts/run_v.py --stage measurement_started       # V.9.5 P2-8 the set-use marker (no pool)
    uv run python scripts/run_v.py --stage jm --condition L          # then C, E0: the judgement set, one per run
    uv run python scripts/run_v.py --stage seal                      # pre-read validity, manifest, archive (no pool)
    uv run python scripts/run_v.py --stage judge                     # bands, sentence, records (no pool)
    uv run python scripts/run_v.py --stage recompute --note "..."    # after judge: analysis defect (V.5)
    uv run python scripts/run_v.py --stage invalid_run --note "..."  # after judge: measurement defect (V.5)

Exit 0 recorded (PASS / SEALED / READ / a record stage), 3 a gate STOP (STOP_REUSE, STOP_V_PATH_REPRO, STOP_SET_SHORT,
STOP_Z_DEGENERATE, STOP_STRENGTH_LEVER, STOP_EVEN_REPRO, STOP_EVEN_PUNISH, STOP_EVEN_LOW_LEVER, gate ②'s three;
recorded, V stops), 5 a gate INVALID (path, kc_input, z, kc_band, even, gate2), 6 seal not SEALED, judge NOT_READ or a
smoke with problems, 7 the reuse condition or a measurement key broke (V stops; no block), 2 a refusal (arguments, cwd,
connectome sha256, chain, uncommitted summary, dirty hashed file, data pins, R's even raw, T's z rows or U's rows
missing, the set not reproducing block set)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

NPZ = "data/malecns.npz"
EXIT_STOP, EXIT_INVALID, EXIT_NOT_READ = 3, 5, 6
STAGES = ("reuse", "path", "kc_input", "set", "z", "kc_band", "even", "smoke", "oc", "gate2_oc", "gate2",
          "measurement_started", "jm", "seal", "judge", "recompute", "invalid_run")
POOL_STAGES = ("path", "kc_input", "kc_band", "even", "smoke", "gate2", "jm")
GATE_STAGES = ("reuse", "path", "kc_input", "set", "z", "kc_band", "even", "gate2")
QUIET = ("pairs", "manifest", "records", "record", "p_judgement_L", "p_judgement_C", "conditions", "rows", "archive",
         "checks", "g_fail", "set", "sides", "C", "L", "independent", "cluster", "record_calib", "record_set",
         "record_e0", "compare", "kc_record", "odour_ids", "u_compare", "csc_facts", "ratio", "ratio_z_V", "guard")


def check_args(a) -> str | None:
    if not a.stage:
        return "--stage is required"
    if (a.stage == "jm") != (a.condition is not None):
        return "--condition goes with --stage jm only"
    if a.stage in ("recompute", "invalid_run") and not a.note:
        return "--note is required for recompute / invalid_run (V.5)"
    if a.note and a.stage not in ("recompute", "invalid_run"):
        return "--note goes with recompute / invalid_run only"
    if a.rerun_after_invalid and a.stage != "gate2":
        return "--rerun-after-invalid goes with --stage gate2 only"
    return None


def make_measure(stage: str, ucode: dict, pool, ctx: dict):
    """measure(z): one RMeasurer per z, every one with V's spec over one UPool (U's job copies, unchanged) and one
    VCache keyed by the V measurement key (= U's) and rooted at V's cache (results/v/cache, results/v/smoke/cache for
    smoke), so no V entry lands in R's, T's or U's tree (VCache.put writes only through v_store's guard)."""
    from flymon.brain.h3_store import canonical
    from flymon.brain.r_measure import RMeasurer
    from flymon.brain.u_measure import UPool
    from flymon.brain.v_spec import SPEC
    from flymon.brain.v_store import VCache
    cache = VCache(SPEC.smoke_cache_dir if stage == "smoke" else SPEC.cache_dir, ucode)
    upool = UPool(pool)
    measurers = {}

    def measure(z):
        k = canonical({t: list(v) for t, v in z.items()})
        if k not in measurers:
            measurers[k] = RMeasurer(upool, cache, SPEC, ctx["params"], ctx["readout"], z, ctx["types"], ctx["n_kc"])
        return measurers[k]
    return measure


def exit_code(stage: str, out: dict) -> int:
    from flymon.brain import v_rules as R
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

    from flymon.brain import v_runner
    from flymon.brain.fly_pool import FlyPool
    from flymon.brain.odor_real import DataMismatch
    from flymon.brain.r_measure import R_MEASURE_FILES
    from flymon.brain.t_measure import t_measure_key
    from flymon.brain.u_measure import UZMeasurer, u_measure_key
    from flymon.brain.v_spec import SPEC

    try:
        ctx = v_runner.build_ctx(SPEC, NPZ)
        code, ucode = code_key(NPZ, files=R_MEASURE_FILES), u_measure_key(NPZ)
        workers = a.workers or SPEC.workers
        pool = None
        if a.stage in POOL_STAGES:
            pool = FlyPool(NPZ, ctx["params"], flies=[{}] * workers, workers=workers, punish_type=SPEC.punish_type,
                           reward_type=SPEC.reward_type, timeout_s=SPEC.pool_timeout_s)
        try:
            measure = make_measure(a.stage, ucode, pool, ctx)
            zm = UZMeasurer(pool, ctx["params"], SPEC.p_type, ctx["rec_types"], SPEC.kc_types)
            r = v_runner.Runner(measure, zm, ctx, SPEC, code=code, tcode=t_measure_key(NPZ), ucode=ucode,
                                pipeline=v_runner.pipeline_key())
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
