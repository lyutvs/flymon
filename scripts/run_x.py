#!/usr/bin/env python3
"""Spec appendix X, phase A (X.9.1 > X.0–X.9; X.9.1.1 구현 단계화): the controller runs each stage and commits its
block before the next.

    .venv/bin/python scripts/run_x.py --stage stage0     # 0  X code checks, X.4.6, P2-6 on θ̂, W calibration diagnosis
    .venv/bin/python scripts/run_x.py --stage precheck   # 0a point-θ precheck + record-only diagnostics (resumes)

Exit 0 PASS, 3 a gate STOP (STOP_REUSE, STOP_OC_UNREACHABLE; recorded, X stops), 5 INVALID (a code defect: do not
commit), 2 a refusal (arguments, cwd, connectome sha256, chain, uncommitted summary, dirty hashed file)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

NPZ = "data/malecns.npz"
EXIT_STOP, EXIT_INVALID = 3, 5
STAGES = ("stage0", "precheck")
QUIET = ("synthetic", "w_cal_diagnosis", "tables", "decision_files", "w_files", "calibration", "theta", "at_f32",
         "m_needed", "records", "diag", "residuals", "env", "p26", "oc_timing", "git")


def exit_code(out: dict) -> int:
    from flymon.brain import x_rules as R
    o = out.get("outcome")
    return 0 if o == R.PASS else (EXIT_INVALID if o == R.INVALID else EXIT_STOP)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=STAGES, required=True)
    a = ap.parse_args(argv)
    from flymon.brain.h3_spec import SPEC as H3
    from flymon.brain.h3_store import ROOT, sha256_file
    if Path.cwd().resolve() != ROOT:
        print(f"refusing: run from the repository root {ROOT}", file=sys.stderr)
        return 2
    if not Path(NPZ).exists() or sha256_file(NPZ) != H3.connectome_sha256:
        print(f"refusing: {NPZ} is missing or its sha256 is not {H3.connectome_sha256}", file=sys.stderr)
        return 2
    from flymon.brain import x_runner
    from flymon.brain.odor_real import DataMismatch
    from flymon.brain.x_spec import SPEC
    try:
        out = getattr(x_runner.Runner(x_runner.build_ctx(NPZ), SPEC), f"stage_{a.stage}")()
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 2
    except DataMismatch as e:
        print(f"refusing: the Hallem data do not match their pins: {e}", file=sys.stderr)
        return 2
    if out.get("sentence"):
        print(out["sentence"])
    print(json.dumps({k: v for k, v in out.items() if k not in QUIET}, ensure_ascii=False,
                     default=str)[:SPEC.cli_print_chars])
    return exit_code(out)


if __name__ == "__main__":
    sys.exit(main())
