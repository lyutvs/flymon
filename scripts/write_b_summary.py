#!/usr/bin/env python3
"""Spec J.12.3 / J.12.5: derive the B summaries from raw records (scripts/run_b_test.py) by flymon/brain/b_rules.py.

    uv run python scripts/write_b_summary.py calibration results/b/calibration/<run id>.json
    uv run python scripts/write_b_summary.py test results/b/exploration/<run id>.json results/b/confirmation/<run id>.json

`calibration` writes results/summary/b_calibration.json (the pilot's status and, when CALIBRATED, t_R, t_P, t_C with
each threshold's null pass rate and power). `test` needs a CALIBRATED summary and writes results/summary/b_test.json
(both pairs' verdicts and the B verdict). Both refuse (exit 2) outside the repository root, and a raw record that is a
smoke run, has no measured_git (the provenance of the run that measured it, scripts/run_b_test.py) or was measured with
dirty hashed files (measured_git.dirty_hashed), was measured under another spec or under another measurement key than
this code's, or is not the pair named. `test` also refuses a calibration summary that is not CALIBRATED, was made under
another measurement key, or was made with another flymon/brain/b_rules.py (rules_sha256: recalibrate). Nothing is
written on a refusal.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from flymon.brain import b_rules
from flymon.brain.b_files import MEASURE_FILES
from flymon.brain.b_spec import SPEC
from flymon.brain.b_store import write_json
from flymon.brain.config import Params
from flymon.brain.h3_store import ROOT, canonical, code_key, sha256_file

CAL = "results/summary/b_calibration.json"
TEST = "results/summary/b_test.json"


def refuse(why: str) -> int:
    print(f"refused: {why}", file=sys.stderr)
    return 2


def load(path: str, pair: str, key: str):
    raw = json.loads(Path(path).read_text())
    problems = []
    if raw.get("pair") != pair:
        problems.append(f"{path} is pair {raw.get('pair')!r}, not {pair!r}")
    if raw.get("smoke"):
        problems.append(f"{path} is a smoke run")
    mg = raw.get("measured_git")
    if not isinstance(mg, dict):
        problems.append(f"{path} has no measured_git (not a raw of this run script)")
    elif mg.get("dirty_hashed"):
        problems.append(f"{path} was measured with dirty hashed files {mg['dirty_hashed']}")
    if canonical(raw.get("spec")) != canonical(SPEC):
        problems.append(f"{path} was measured under another spec")
    if raw.get("measure_key") != key:
        problems.append(f"{path} was measured under key {str(raw.get('measure_key'))[:12]}, this code's is {key[:12]}")
    return raw, problems


def main(argv=None, npz: str = "data/malecns.npz", require_root: bool = True) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("calibration", "test") or len(argv) != (2 if argv[0] == "calibration" else 3):
        return refuse("usage: calibration <raw> | test <exploration raw> <confirmation raw>")
    if require_root and Path.cwd().resolve() != ROOT:
        return refuse(f"not at the repository root {ROOT}")
    key = code_key(npz, files=MEASURE_FILES)["key"]
    rules = {"flymon/brain/b_rules.py": sha256_file(ROOT / "flymon/brain/b_rules.py")}
    if argv[0] == "calibration":
        raw, problems = load(argv[1], "calibration", key)
        if problems:
            return refuse("; ".join(problems))
        if raw["x"] != b_rules.choose_x(raw["records"], SPEC, None):
            return refuse(f"{argv[1]}: the pilot's X is not the rule's")    # the thresholds depend on the pilot's X
        cal = b_rules.calibrate(raw["records"], raw["x"], SPEC, "calibration")
        out = dict(what="spec J.12.5 calibration pilot", run_id=raw["run_id"], raw=argv[1], raw_sha256=sha256_file(argv[1]),
                   measure_key=key, rules_sha256=rules, x=raw["x"], **cal)
        write_json(CAL, out, [Params()])
        print(f"wrote {CAL}: {cal['status']} {cal.get('values')}")
        return 0
    try:
        cal = json.loads(Path(CAL).read_text())
    except (OSError, ValueError) as e:
        return refuse(f"no usable {CAL}: {e}")
    if cal.get("status") != b_rules.CALIBRATED or cal.get("measure_key") != key:
        return refuse(f"{CAL} is {cal.get('status')} under key {str(cal.get('measure_key'))[:12]}: not a calibration of this code")
    if cal.get("rules_sha256") != rules:
        return refuse(f"recalibrate: {CAL} was made with another b_rules.py")
    (e_raw, pe), (c_raw, pc) = load(argv[1], "exploration", key), load(argv[2], "confirmation", key)
    if pe or pc:
        return refuse("; ".join(pe + pc))
    th = cal["values"]
    e, c = (b_rules.pair_verdict(r["records"], r["x"], th, SPEC, p) for r, p in ((e_raw, "exploration"),
                                                                                  (c_raw, "confirmation")))
    out = dict(what="spec J.12.3 B test", thresholds=th, calibration_run_id=cal["run_id"], measure_key=key,
               rules_sha256=rules, raws={p: dict(path=a, sha256=sha256_file(a)) for p, a in
                                         (("exploration", argv[1]), ("confirmation", argv[2]))},
               exploration=e, confirmation=c, verdict=b_rules.b_verdict(e, c))
    write_json(TEST, out, [Params()])
    print(f"wrote {TEST}: {out['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
