#!/usr/bin/env python3
"""Spec appendix AA (AA.7 6245–6264): the controller runs each stage and commits its block before the next.

    .venv/bin/python scripts/run_aa.py --stage stage0                  # 0a: tests log, synth validation, differential
    .venv/bin/python scripts/run_aa.py --stage reuse                   # 1: reuse conditions
    .venv/bin/python scripts/run_aa.py --stage smoke --workers 16      # 2: smoke + budget check
    .venv/bin/python scripts/run_aa.py --stage seal_code               # 3: seal the estimation code
    .venv/bin/python scripts/run_aa.py --stage screen --workers 16     # 4: naive filter on the 31 lenient pairs
    .venv/bin/python scripts/run_aa.py --stage coverage --workers 16   # 4b: structure-matched coverage
    .venv/bin/python scripts/run_aa.py --stage learn --workers 16      # 5: learning measurement (resumes)
    .venv/bin/python scripts/run_aa.py --stage estimate                # 6: the primary estimate
    .venv/bin/python scripts/run_aa.py --stage records --workers 16    # 7: records + G.6 recompute record
    .venv/bin/python scripts/run_aa.py --stage archive --name screen   # re-archive a committed block (idempotent)

Defect procedure (AA.7 3, plan Reading 19), only for a sealed stage (coverage / estimate / records):
    .venv/bin/python scripts/run_aa.py --stage defect --note note.json  # ① defect_<n> {symptom, clause, cause, affected}
    (② patch + regression test, merge, rerun the order-0 tests log — Operator §2)
    .venv/bin/python scripts/run_aa.py --stage reseal --name <n>        # ③ reseal_<n>: new seal hashes, new tests log
    .venv/bin/python scripts/run_aa.py --stage recompute --name estimate --workers 16   # ④ estimate_v<n> (cache only)

Pre-seal restart (INVALID at stage0 / smoke, or any aa_* fix before seal_code; refused once seal_code exists):
    .venv/bin/python scripts/run_aa.py --stage restart_preseal   # results/aa, the archive → *.invalid-<n>; summary
    (removed if HEAD does not track it, else rewritten to a preseal_restarts record to commit); then §2 and stage0

Exit 0 PASS, 3 STOP (recorded, AA stops), 5 INVALID (do not commit), 6 environment mismatch (the ledger entry is
written, the user decides), 7 seal mismatch (defect procedure first), 2 a refusal (arguments, cwd, connectome sha256,
chain, dirty files). Output: the outcome line, the sentence and the block minus the top-level QUIET fields — bulky
records (env, git, …) and every container that holds a per-pair main-set value: `pairs` (screen, records), `primary`
(estimate: per-pair means / CIs and the pair range), `s1_records` and `sets` (records: pair ranges, drop-one rows,
per-pair SD) — so nothing printed carries a main-set pair value beyond the AA.8 sentence itself."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

NPZ = "data/malecns.npz"
EXIT_STOP, EXIT_INVALID = 3, 5
ARCHIVE = "archive"
DEFECT, RESEAL, RECOMPUTE, RESTART = "defect", "reseal", "recompute", "restart_preseal"
QUIET = ("archive", "env", "git", "decision_files", "numbers", "synth", "candidates", "differential", "pairs", "manifest",
         "primary", "s1_records", "sets")


def exit_code(out: dict) -> int:
    from flymon.brain import aa_rules as R
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
    """--name / --note by stage: archive <stage or stage_v<n>>, defect --note, reseal <n>, recompute <sealed stage>."""
    import re
    why = []
    if (a.stage == DEFECT) != (a.note is not None):
        why.append("--note goes with --stage defect only (and it needs one)")
    if a.stage == ARCHIVE:
        if a.name is None or not re.fullmatch(rf"({'|'.join(spec.stages)})(_v\d+)?", a.name):
            why.append("--stage archive needs --name <stage> (or <stage>_v<n>)")
    elif a.stage == RESEAL:
        if a.name is None or not a.name.isdigit():
            why.append("--stage reseal needs --name <n> (the defect number)")
    elif a.stage == RECOMPUTE:
        if a.name not in spec.sealed_stages:
            why.append(f"--stage recompute needs --name one of {list(spec.sealed_stages)}")
    elif a.name is not None:
        why.append("--name goes with --stage archive / reseal / recompute only")
    return why


def main(argv=None) -> int:
    from flymon.brain.aa_spec import SPEC
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=SPEC.stages + (ARCHIVE, DEFECT, RESEAL, RECOMPUTE, RESTART), required=True)
    ap.add_argument("--name", help="archive: the block; reseal: the defect number; recompute: the sealed stage")
    ap.add_argument("--note", help="defect: a JSON file {symptom, clause, cause, affected}")
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
    from flymon.brain import aa_runner
    from flymon.brain.odor_real import DataMismatch
    try:
        ctx = aa_runner.build_ctx(NPZ, SPEC)
        runner = aa_runner.Runner(ctx, SPEC)
        runner.workers = a.workers
        if a.stage == ARCHIVE:
            man = runner.archive(a.name)
            for e in man:
                print(f"archived {e['dst']} sha256 {e['sha256']}")
            print(f"stage archive {a.name}: {len(man)} file(s) (exit 0)")
            return 0
        if a.stage in (DEFECT, RESEAL, RECOMPUTE, RESTART):   # no FlyPool: these never measure
            runner.workers = a.workers or SPEC.workers
            out = (runner.defect(a.note) if a.stage == DEFECT else runner.reseal(int(a.name)) if a.stage == RESEAL
                   else runner.restart_preseal() if a.stage == RESTART else runner.recompute(a.name))
            for ln in report_lines(a.stage, out, SPEC.cli_print_chars):
                print(ln)
            return exit_code(out)
        if a.stage in SPEC.pool_stages:
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
