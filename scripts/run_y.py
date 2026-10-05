#!/usr/bin/env python3
"""Spec appendix Y, step 0p (Y.7 0p, in parallel with the red-team): the controller runs each stage and commits its block
before the next.

    .venv/bin/python scripts/run_y.py --stage digest                 # 0p-b keys, V blocks, main-set digest (no pool)
    .venv/bin/python scripts/run_y.py --stage oracle --workers 16    # 0p-c 249-pair oracle (24_700_xxx; resumes)
    .venv/bin/python scripts/run_y.py --stage archive --name oracle  # re-archive a committed block (idempotent)

Exit 0 PASS, 3 a gate STOP (STOP_REUSE, STOP_FEW_PAIRS; recorded, Y stops), 5 INVALID (a code / machine defect: do not
commit), 2 a refusal (arguments, cwd, connectome sha256, chain, uncommitted summary, dirty hashed file).
Output (Y.7 0p-c read rule, Y red-team P0-1): the oracle stage prints only its outcome label and exit code on stdout —
no per-pair value and no count table (the controller reads the table from the block after Y.9.2 is committed); the
measurer's progress goes to stderr. The digest stage (no per-pair value) prints its sentence and the block minus QUIET.
The archive stage (Y.9.2 P3-13) prints the archived paths and sha256 only; exit 0, or 2 on a refusal."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

NPZ = "data/malecns.npz"
EXIT_STOP, EXIT_INVALID = 3, 5
STAGES = ("digest", "oracle")
ARCHIVE = "archive"
POOL_STAGES = ("oracle",)
LABEL_ONLY = ("oracle",)
QUIET = ("set", "reuse", "v_reuse", "keys", "git", "seeds", "z_V", "detail_path", "archive")


def exit_code(out: dict) -> int:
    from flymon.brain import y_rules as R
    o = out.get("outcome")
    return 0 if o == R.PASS else (EXIT_INVALID if o == R.INVALID else EXIT_STOP)


def report_lines(stage: str, out: dict, limit: int) -> list:
    """The stdout lines for a finished stage: the label line always; for a LABEL_ONLY stage nothing else."""
    lines = [f"stage {stage}: outcome {out.get('outcome')} (exit {exit_code(out)})"]
    if stage in LABEL_ONLY:
        return lines
    if out.get("sentence"):
        lines.append(out["sentence"])
    lines.append(json.dumps({k: v for k, v in out.items() if k not in QUIET}, ensure_ascii=False, default=str)[:limit])
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=STAGES + (ARCHIVE,), required=True)
    ap.add_argument("--name", choices=STAGES, help="the stage to archive (with --stage archive)")
    ap.add_argument("--workers", type=int)
    a = ap.parse_args(argv)
    if (a.stage == ARCHIVE) != (a.name is not None):
        print("refusing: --name goes with --stage archive only (and it needs one)", file=sys.stderr)
        return 2
    from flymon.brain.h3_spec import SPEC as H3
    from flymon.brain.h3_store import ROOT, sha256_file
    if Path.cwd().resolve() != ROOT:
        print(f"refusing: run from the repository root {ROOT}", file=sys.stderr)
        return 2
    if not Path(NPZ).exists() or sha256_file(NPZ) != H3.connectome_sha256:
        print(f"refusing: {NPZ} is missing or its sha256 is not {H3.connectome_sha256}", file=sys.stderr)
        return 2
    from flymon.brain import y_runner
    from flymon.brain.y_spec import SPEC
    if a.stage == ARCHIVE:
        try:
            man = y_runner.Runner({}, SPEC).archive(a.name)
        except SystemExit as e:
            return int(e.code) if isinstance(e.code, int) else 2
        for e in man:
            print(f"archived {e['dst']} sha256 {e['sha256']}")
        print(f"stage archive {a.name}: {len(man)} file(s) (exit 0)")
        return 0
    from flymon.brain.fly_pool import FlyPool
    from flymon.brain.odor_real import DataMismatch
    from flymon.brain.w_spec import SPEC as W
    pool = None
    try:
        ctx = y_runner.build_ctx(NPZ)
        if a.stage in POOL_STAGES:
            workers = a.workers or SPEC.workers
            pool = FlyPool(NPZ, ctx["params"](), flies=[{}] * workers, workers=workers, punish_type=W.punish_dan,
                           reward_type=W.reward_dan, timeout_s=SPEC.pool_timeout_s)
        try:
            out = getattr(y_runner.Runner(ctx, SPEC, measure=lambda: ctx["measurer"](pool)), f"stage_{a.stage}")()
        finally:
            if pool is not None:
                pool.close()
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 2
    except DataMismatch as e:
        print(f"refusing: the Hallem data do not match their pins: {e}", file=sys.stderr)
        return 2
    for ln in report_lines(a.stage, out, SPEC.cli_print_chars):
        print(ln)
    return exit_code(out)


if __name__ == "__main__":
    sys.exit(main())
