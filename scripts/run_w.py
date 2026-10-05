#!/usr/bin/env python3
"""Spec appendix W (W.9.10 > W.9.9 > W.9.8 > W.9.1-W.9.7 > W.0-W.8): the F v4 learning test on V's combined lever
L_V — F.2's sequential R / N brains and G.5's RN on u_rig, read with z_V, judged by per-fly joint satisfaction with a
design (q, K, F; k cap 8) chosen by W's own operating characteristic from a POOL even pilot. The controller runs every
stage; commit each block before the next.

    uv run python scripts/run_w.py --stage stage0                    # 0 verdict code, w_oc, tables, synthetic checks
    uv run python scripts/run_w.py --stage reuse                     # 1 V's blocks, R / T / U keys (no pool)
    uv run python scripts/run_w.py --stage path                      # 2 V gate ② rows, V oracle naive counts, reward
    uv run python scripts/run_w.py --stage pilot                     # 3 / 3a POOL even pilot (resumes), H6 stop
    uv run python scripts/run_w.py --stage oc                        # 4 the OC, the design and its ranking (no pool)
    uv run python scripts/run_w.py --stage smoke --workers 4         # 5 smoke 42_1xx_xxx, the cost ledger
    uv run python scripts/run_w.py --stage budget                    # 5a worst-case budget gate (no pool)
    uv run python scripts/run_w.py --stage set                       # 6 the main set's digests (no pool)
    uv run python scripts/run_w.py --stage oracle                    # 7 the oracle screen (24_700_xxx; the set is used)
    uv run python scripts/run_w.py --stage estimate                  # 8 the estimate on the testable count (no pool)
    uv run python scripts/run_w.py --stage naive                     # 9 naive probes until 8 gate pairs (resumes)
    uv run python scripts/run_w.py --stage gates                     # 10 the gate pairs, STOP_FEW_PAIRS (no pool)
    uv run python scripts/run_w.py --stage learn                     # 11 R / N / RN on the gate pairs (resumes)
    uv run python scripts/run_w.py --stage band                      # 11 the 2K probes for every gate pair (resumes)
    uv run python scripts/run_w.py --stage records                   # 12 C and the plasticity-off control (resumes)
    uv run python scripts/run_w.py --stage seal                      # 13 validity, manifest, archive (no pool)
    uv run python scripts/run_w.py --stage judge                     # 14 the verdict, once (no pool)
    uv run python scripts/run_w.py --stage recompute --note "..."    # after judge: analysis defect (W.5)
    uv run python scripts/run_w.py --stage invalid_run --note "..."  # after judge: measurement defect (W.5, H8)

Exit 0 recorded (PASS / SEALED / a judgement was read / a record stage), 3 a gate STOP (STOP_REUSE,
STOP_W_PATH_REPRO, STOP_PILOT_NO_EFFECT, STOP_OC_UNREACHABLE, STOP_BUDGET, STOP_FEW_PAIRS; recorded, W stops), 5 a gate
INVALID, 6 seal not SEALED or a smoke with problems, 7 the reuse condition or a measurement key broke (no block), 2 a
refusal (arguments, cwd, connectome sha256, chain, uncommitted summary, dirty hashed file, V's raw data, the set not
reproducing block set)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

NPZ = "data/malecns.npz"
EXIT_STOP, EXIT_INVALID, EXIT_NOT_READ = 3, 5, 6
STAGES = ("stage0", "reuse", "path", "pilot", "oc", "smoke", "budget", "set", "oracle", "estimate", "naive", "gates",
          "learn", "band", "records", "seal", "judge", "recompute", "invalid_run")
POOL_STAGES = ("path", "pilot", "smoke", "oracle", "naive", "learn", "band", "records")
GATE_STAGES = ("stage0", "reuse", "path", "pilot", "oc", "budget", "set", "oracle", "estimate", "naive", "gates",
               "learn", "band", "records")
QUIET = ("manifest", "checks", "record", "records", "screened", "pairs", "judgement", "ranking", "calibration", "theta",
         "synthetic", "tables", "set", "archive", "oc_timing", "decision_files", "costs", "cost", "exploratory")


def check_args(a) -> str | None:
    if not a.stage:
        return "--stage is required"
    if a.stage in ("recompute", "invalid_run") and not a.note:
        return "--note is required for recompute / invalid_run (W.5)"
    if a.note and a.stage not in ("recompute", "invalid_run"):
        return "--note goes with recompute / invalid_run only"
    return None


def make_measurers(ucode: dict, wcode: dict, pool, ctx: dict):
    """measure(z, smoke): one RMeasurer per (z, smoke) over U's job copies (u_measure.UPool, unchanged) behind a WCache
    keyed by the W measurement key (results/w/cache or results/w/smoke/cache); learner(smoke): a WMeasurer behind the
    same caches."""
    from flymon.brain.r_measure import RMeasurer
    from flymon.brain.u_measure import UPool
    from flymon.brain.v_spec import SPEC as V
    from flymon.brain.w_measure import WMeasurer
    from flymon.brain.w_spec import SPEC
    from flymon.brain.w_store import WCache
    caches = {s: WCache(SPEC.smoke_cache_dir if s else SPEC.cache_dir, wcode) for s in (False, True)}
    upool = UPool(pool)
    windows = dict(strength=SPEC.strength, settle_ms=V.settle_ms, read_ms=V.read_ms, window_ms=V.window_ms)
    timing = dict(present_ms=SPEC.pulse_ms, gap_ms=SPEC.gap_ms, train_settle_ms=SPEC.train_settle_ms,
                  seed_stride=SPEC.fly_train_stride)

    def measure(z, smoke=False):
        return RMeasurer(upool, caches[smoke], V, ctx["params"], ctx["readout"], z, ctx["types"], ctx["n_kc"])

    def learner(smoke=False):
        return WMeasurer(pool, caches[smoke], ctx["params"], ctx["readout"], V.p_type, SPEC.reward_dan,
                         SPEC.punish_dan, windows, timing)
    return measure, learner


def exit_code(stage: str, out: dict) -> int:
    from flymon.brain import w_rules as R
    if stage in GATE_STAGES:
        o = out.get("outcome")
        return 0 if o == R.PASS else (EXIT_INVALID if o == R.INVALID else EXIT_STOP)
    if stage == "smoke":
        return EXIT_NOT_READ if out.get("problems") else 0
    if stage == "seal":
        return 0 if out.get("status") == R.SEALED else EXIT_NOT_READ
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=STAGES)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--note")
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

    from flymon.brain import w_runner
    from flymon.brain.fly_pool import FlyPool
    from flymon.brain.odor_real import DataMismatch
    from flymon.brain.r_measure import R_MEASURE_FILES
    from flymon.brain.t_measure import t_measure_key
    from flymon.brain.u_measure import u_measure_key
    from flymon.brain.w_measure import w_measure_key
    from flymon.brain.w_spec import SPEC, smoke

    spec = smoke(SPEC) if a.stage == "smoke" else SPEC
    try:
        ctx = w_runner.build_ctx(spec, NPZ)
        code, ucode, wcode = code_key(NPZ, files=R_MEASURE_FILES), u_measure_key(NPZ), w_measure_key(NPZ)
        workers = a.workers or spec.workers
        pool = None
        if a.stage in POOL_STAGES:
            pool = FlyPool(NPZ, ctx["params"], flies=[{}] * workers, workers=workers, punish_type=SPEC.punish_dan,
                           reward_type=SPEC.reward_dan, timeout_s=SPEC.pool_timeout_s)
        try:
            measure, learner = make_measurers(ucode, wcode, pool, ctx)
            r = w_runner.Runner(measure, learner, ctx, spec, code=code, tcode=t_measure_key(NPZ), ucode=ucode,
                                wcode=wcode, pipeline=w_runner.pipeline_key())
            if a.stage in ("recompute", "invalid_run"):
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
