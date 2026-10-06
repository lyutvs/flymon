#!/usr/bin/env python3
"""Spec appendix AB (AB.7 6784–6870): the controller runs each stage and commits its block before the next.

    .venv/bin/python scripts/run_ab.py --stage stage0                  # 0a: tests log, bench, AA differential, env
    .venv/bin/python scripts/run_ab.py --stage reuse                   # 0b: AB.2 reuse conditions, lv_odour, 0f source
    .venv/bin/python scripts/run_ab.py --stage generate                # 0c: the KC-pre set vs every AB.3 declared value
    .venv/bin/python scripts/run_ab.py --stage seal_code               # 0d: the seal (before any 2nd-gen measurement)
    .venv/bin/python scripts/run_ab.py --stage cal_gate --workers 16   # 0e: the pre-measurement calibration gate
    .venv/bin/python scripts/run_ab.py --stage futility --workers 16   # 0f: the futility gate (AB.9.3)
    .venv/bin/python scripts/run_ab.py --stage kc_input --workers 16   # 3 … 10: Stage II (plan "Staged build")
    .venv/bin/python scripts/run_ab.py --stage archive --name stage0   # re-archive a committed block (idempotent)

Pre-measurement restart (AB.7 0d, plan Reading 15; refused once kc_input or any 2nd-gen raw exists, or after a STOP):
    .venv/bin/python scripts/run_ab.py --stage restart_preseal --note note.json   # {symptom, clause, cause}
    (results/ab, the archive → *.invalid-<n>; summary rewritten to restart_preseal_<n> + budget, or removed); then
    commit, §2 (a new tests log) and stage0.

Defect procedure after the first measurement (AB.7 0d, plan Reading 16; Stage II):
    --stage defect --note note.json · --stage reseal --name <n> · --stage recompute --name <stage>

Exit 0 PASS, 3 STOP (recorded, AB stops), 5 INVALID (do not commit), 6 environment mismatch (the ledger entry is
written, the user decides), 7 seal mismatch, 2 a refusal (arguments, cwd, connectome sha256, chain, dirty files, a
stage not built yet). Output: the outcome line, the sentence and the block minus the top-level QUIET fields — bulky
records (env, git, …) and every container that may hold a per-pair or per-cell value (pairs, manifest, per_pair,
detail, the calibration counts and truths, the 0e structures, the 0f model / rows / grid), so the operator reads only
the outcome and sentence lines."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

NPZ = "data/malecns.npz"
EXIT_STOP, EXIT_INVALID = 3, 5
ARCHIVE = "archive"
DEFECT, RESEAL, RECOMPUTE, RESTART = "defect", "reseal", "recompute", "restart_preseal"
QUIET = ("archive", "env", "git", "decision_files", "numbers", "bench", "candidates", "differential", "pairs",
         "manifest", "per_pair", "detail", "cal", "truth", "sel_counts", "ver_counts", "structures", "model", "rows",
         "grid", "seal", "stream_vectors", "lv_odour", "lv_keys", "keys", "odour_ids", "new_odours", "v_cache_odours",
         "opponents", "selftest", "invalidated", "restart")


def exit_code(out: dict) -> int:
    from flymon.brain import ab_rules as R
    o = out.get("outcome")
    return 0 if o == R.PASS else (EXIT_INVALID if o == R.INVALID else EXIT_STOP)


def report_lines(stage: str, out: dict, limit: int) -> list:
    lines = [f"stage {stage}: outcome {out.get('outcome')} (exit {exit_code(out)})"]
    if out.get("sentence"):
        lines.append(out["sentence"])
    lines.append(json.dumps({k: v for k, v in out.items() if k not in QUIET and k != "sentence"}, ensure_ascii=False,
                            default=str)[:limit])
    return lines


def arg_reasons(a, spec) -> list:
    """--name / --note by stage: archive <stage or stage_v<n>>, defect / restart_preseal --note, reseal <n>,
    recompute <stage of plan Reading 16>."""
    from flymon.brain.ab_runner import RECOMPUTE_STAGES
    why = []
    if (a.stage in (DEFECT, RESTART)) != (a.note is not None):
        why.append("--note goes with --stage defect / restart_preseal only (and they need one)")
    if a.stage == ARCHIVE:
        if a.name is None or not re.fullmatch(rf"({'|'.join(spec.stages)})(_v\d+)?", a.name):
            why.append("--stage archive needs --name <stage> (or <stage>_v<n>)")
    elif a.stage == RESEAL:
        if a.name is None or not a.name.isdigit():
            why.append("--stage reseal needs --name <n> (the defect number)")
    elif a.stage == RECOMPUTE:
        if a.name not in RECOMPUTE_STAGES:
            why.append(f"--stage recompute needs --name one of {list(RECOMPUTE_STAGES)}")
    elif a.name is not None:
        why.append("--name goes with --stage archive / reseal / recompute only")
    return why


def main(argv=None) -> int:
    from flymon.brain.ab_spec import SPEC
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=SPEC.stages + (ARCHIVE, DEFECT, RESEAL, RECOMPUTE, RESTART), required=True)
    ap.add_argument("--name", help="archive: the block; reseal: the defect number; recompute: the stage")
    ap.add_argument("--note", help="defect / restart_preseal: a JSON file {symptom, clause, cause[, affected]}")
    ap.add_argument("--workers", type=int)
    a = ap.parse_args(argv)
    why = arg_reasons(a, SPEC)
    if why:
        print(f"refusing: {'; '.join(why)}", file=sys.stderr)
        return 2
    from flymon.brain.h3_spec import SPEC as H3
    from flymon.brain.h3_store import ROOT, sha256_file
    if Path.cwd().resolve() != ROOT:
        print(f"refusing: run from the repository root {ROOT}", file=sys.stderr)
        return 2
    if not Path(NPZ).exists() or sha256_file(NPZ) != H3.connectome_sha256:
        print(f"refusing: {NPZ} is missing or its sha256 is not {H3.connectome_sha256}", file=sys.stderr)
        return 2
    from flymon.brain import ab_runner
    from flymon.brain.odor_real import DataMismatch
    try:
        ctx = ab_runner.build_ctx(NPZ, SPEC)
        runner = ab_runner.Runner(ctx, SPEC)
        runner.workers = a.workers
        if a.stage == ARCHIVE:
            man = runner.archive(a.name)
            for e in man:
                print(f"archived {e['dst']} sha256 {e['sha256']}")
            print(f"stage archive {a.name}: {len(man)} file(s) (exit 0)")
            return 0
        if a.stage in (DEFECT, RESEAL, RECOMPUTE, RESTART):   # no FlyPool: these never measure
            runner.workers = a.workers or SPEC.workers
            fn = getattr(runner, a.stage, None)
            if fn is None:
                print(f"refusing: --stage {a.stage} is not implemented yet (Stage II)", file=sys.stderr)
                return 2
            out = (fn(a.note) if a.stage in (DEFECT, RESTART) else fn(int(a.name)) if a.stage == RESEAL
                   else fn(a.name))
            for ln in report_lines(a.stage, out, SPEC.cli_print_chars):
                print(ln)
            return exit_code(out)
        if getattr(runner, f"stage_{a.stage}", None) is None:   # refused before any FlyPool starts
            runner.run(a.stage)                                # refuses: SystemExit 2, "not implemented yet"
            return 2
        if a.stage in SPEC.fly_stages:                     # cal_gate / calibrate / records spawn their own pool
            from flymon.brain.fly_pool import FlyPool
            from flymon.brain.w_spec import SPEC as W
            workers = a.workers or SPEC.workers
            runner.workers = workers
            runner.pool = FlyPool(NPZ, ctx["params"](), flies=[{}] * workers, workers=workers,
                                  punish_type=W.punish_dan, reward_type=W.reward_dan, timeout_s=SPEC.pool_timeout_s)
        try:
            out = runner.run(a.stage)
        finally:
            if getattr(runner, "pool", None) is not None:
                runner.pool.close()
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
