#!/usr/bin/env python3
"""Re-scoped claim, M4 verdict (spec 4.5, 10.9): (2a) FLY - RND, (2b) FLY - C-off and FLY - FLY-RS from the five arms'
evaluation blocks (flymon.rescope.stats: paired fly -> battle bootstrap, SPEC.boot_draws draws, seed SPEC.boot_seed).

    uv run python scripts/write_rescope_m4.py --phase judge
    uv run python scripts/write_rescope_m4.py --smoke --phase pilot --allow-dirty

Reads <root>/<phase>/<ARM>/result.json for ARM in FLY RS COFF RND MAX (root results/rescope, --smoke
results/rescope-smoke) and writes results/summary/rescope_m4.json (--smoke: results/rescope-smoke/summary/
rescope_m4.json; never results/summary/). Without --smoke only --phase judge is accepted (the pilot is not judged).

Any input problem - a missing or unreadable arm file, an arm of another phase / arm name / smoke flag, eval schedule
digests that differ between arms (or from <root>/<phase>/eval_schedule.json), a malformed row, NaN - writes the
summary with status INVALID (both verdicts INVALID, never PASS / FAIL) and exits 2. Too few valid pairs in a
comparison makes only the verdict using it INVALID (plan R10); the summary is written and the exit code is 2.
Refusals (SystemExit, nothing written): a dirty flymon/rescope/ or this script without --allow-dirty, and an existing
non-smoke summary without --force (the judged file is not overwritten silently).

Also recorded, never judged (spec 4.5): per-arm n_flies / n_eval / invalid flies, RS residuals and donor sha256s;
the information-turn type match of each brain arm's eval-block fly decisions (stats.info_turn_stats over the valid
flies' <arm>/logs/eval/flyNN.jsonl, whose decision records carry `multipliers` from blocks.EvalPlayer); the fly
decision fraction (eval only, learn only, both blocks); the learning-block win curve (FLY, RS); RS turn / pulse
mismatch (dropped bundles, exhausted turns); the final plastic weight median / w0 per fly (brain arms).
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import glob
import json
import sys
from pathlib import Path

from flymon.brain.config import Params
from flymon.rescope import stats
from flymon.rescope.blocks import file_sha, read_jsonl
from flymon.rescope.spec import SPEC
from flymon.rescope.store import git_provenance, guard, write_json

ARMS = stats.ARMS
BRAIN = ("FLY", "RS", "COFF")
SCRIPT = "scripts/write_rescope_m4.py"


def paths(phase: str, smoke: bool) -> tuple:
    root = Path("results/rescope-smoke" if smoke else "results/rescope") / phase
    summ = Path("results/rescope-smoke/summary" if smoke else "results/summary") / "rescope_m4.json"
    return root, summ


def load_arms(root: Path, phase: str, smoke: bool) -> tuple:
    """({arm: result dict or None}, reasons, inputs)."""
    arms, reasons, inputs = {}, [], {}
    for arm in ARMS:
        p = root / arm / "result.json"
        inputs[arm] = dict(path=str(p), sha256=file_sha(p))
        if not p.exists():
            arms[arm] = None
            reasons.append(f"{arm}: {p} does not exist")
            continue
        try:
            d = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError) as e:
            arms[arm] = None
            reasons.append(f"{arm}: {p} unreadable ({e!r})")
            continue
        if not isinstance(d, dict):
            arms[arm] = None
            reasons.append(f"{arm}: {p} is not a JSON object")
            continue
        want = dict(arm=arm, phase=phase, smoke=smoke, complete=True)
        bad = {k: (d.get(k), v) for k, v in want.items() if d.get(k) != v}
        if bad:
            reasons.append(f"{arm}: result.json does not match (got, want) {bad}")
        arms[arm] = d
        inputs[arm]["commit"] = ((d.get("provenance") or {}).get("git") or {}).get("commit")
    return arms, reasons, inputs


def eval_digests(root: Path, arms: dict) -> tuple:
    """({arm: eval digest}, reasons): every arm's eval schedule digest must be one value, and the phase's
    eval_schedule.json (when present) must hold it."""
    dig = {a: ((d or {}).get("schedule_digests") or {}).get("eval") for a, d in arms.items() if d is not None}
    reasons = []
    vals = set(dig.values())
    if None in vals:
        reasons.append(f"eval schedule digest missing: {sorted(a for a, v in dig.items() if v is None)}")
    if len(vals - {None}) > 1:
        reasons.append(f"eval schedule digests differ between arms: { {a: (v or '')[:12] for a, v in dig.items()} }")
    f = root / "eval_schedule.json"
    if f.exists():
        try:
            fd = json.loads(f.read_text()).get("digest")
        except (OSError, json.JSONDecodeError, AttributeError) as e:
            fd = None
            reasons.append(f"{f} unreadable ({e!r})")
        if fd is not None and vals - {None} and vals - {None} != {fd}:
            reasons.append(f"{f} digest {fd[:12]} differs from the arms'")
    return dig, reasons


def arm_record(arm: str, d: dict, spec=SPEC) -> dict:
    rows = d.get("per_fly") or []
    out = dict(n_flies=d.get("flies"), n_eval=d.get("eval"), learn=d.get("learn"),
               recovery_per_pulse=d.get("recovery_per_pulse"), schedule_digests=d.get("schedule_digests"),
               invalid_flies=sorted(int(r["fly"]) for r in rows if isinstance(r, dict) and "fly" in r
                                    and stats.row_invalid(r, spec)),
               eval_weights_frozen=d.get("eval_weights_frozen"))
    if arm == "RS":
        out["rs"] = {int(r["fly"]): dict(residual_frac=r.get("residual_frac"), donor_sha256=r.get("donor_sha256"),
                                         donor_invalid=r.get("donor_invalid"), invalid=r.get("invalid"))
                     for r in rows if isinstance(r, dict) and "fly" in r}
    return out


def _valid_logs(root: Path, arm: str, d: dict, sub: str, spec=SPEC) -> tuple:
    """(records, missing log paths) of the valid flies' fly logs in <arm>/<sub>."""
    recs, missing = [], []
    for r in d.get("per_fly") or []:
        if not isinstance(r, dict) or "fly" not in r or stats.row_invalid(r, spec):
            continue
        p = root / arm / sub / f"fly{int(r['fly']):02d}.jsonl"
        if not p.exists():
            missing.append(str(p))
        recs += read_jsonl(p)
    return recs, missing


def info_turns(root: Path, arm: str, d: dict, spec=SPEC) -> dict:
    """Recorded only: stats.info_turn_stats over the valid flies' eval-block decision records."""
    recs, missing = _valid_logs(root, arm, d, "logs/eval", spec)
    return dict(stats.info_turn_stats(recs), missing_logs=missing)


def decision_fractions(root: Path, arm: str, d: dict, spec=SPEC) -> dict:
    """Recorded only: the fly-decided share of decisions in the eval block, the learning block and both."""
    ev, m1 = _valid_logs(root, arm, d, "logs/eval", spec)
    le, m2 = ([], []) if not d.get("learn") else _valid_logs(root, arm, d, "logs", spec)
    return dict(eval_only=stats.decision_fraction(ev), learn_only=stats.decision_fraction(le) if d.get("learn") else None,
                both_blocks=stats.decision_fraction(ev + le), missing_logs=m1 + m2)


def summarize(root: Path, phase: str, smoke: bool, spec=SPEC) -> dict:
    arms, reasons, inputs = load_arms(root, phase, smoke)
    dig, r2 = eval_digests(root, arms)
    reasons += r2
    v = stats.m4_verdict(arms, spec)
    if reasons:
        v = dict(v, status=stats.INVALID, reasons=reasons + list(v.get("reasons", [])))
        for key in ("2a", "2b"):
            v[key] = dict(v[key], verdict=stats.INVALID, **{"pass": None})
    present = {a: d for a, d in arms.items() if isinstance(d, dict)}
    v["arms"] = {a: arm_record(a, d, spec) for a, d in present.items()}
    v["eval_schedule_digests"] = dig
    v["inputs"] = inputs
    brain = [a for a in BRAIN if a in present]
    v["recorded"] = dict(v.get("recorded", {}),
                         info_turn_match={a: info_turns(root, a, present[a], spec) for a in brain},
                         fly_decision_fraction={a: decision_fractions(root, a, present[a], spec) for a in brain},
                         learn_curve={a: stats.learn_curve(present[a], spec) for a in ("FLY", "RS") if a in present},
                         rs_mismatch=stats.rs_mismatch(present["RS"]) if "RS" in present else None,
                         final_weight_median_ratio={a: stats.weight_medians(present[a]) for a in brain})
    return v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=("pilot", "judge"), required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite an existing results/summary/rescope_m4.json")
    a = ap.parse_args(argv)
    if a.phase != "judge" and not a.smoke:
        raise SystemExit("refusing: the M4 verdict is written for --phase judge only (pilot arms are not judged)")
    root, summary = paths(a.phase, a.smoke)
    gp = [Params()]
    guard(summary, gp)
    if summary.exists() and not a.smoke and not a.force:
        raise SystemExit(f"refusing: {summary} exists (a judged file); pass --force to rewrite it")
    git = git_provenance(files=sorted(glob.glob("flymon/rescope/*.py")) + [SCRIPT])
    if git["dirty"] and not a.allow_dirty:
        raise SystemExit(f"refusing: uncommitted changes in {git['dirty_files']} (commit them or pass --allow-dirty)")
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    v = summarize(root, a.phase, a.smoke, SPEC)
    v.update(phase=a.phase, smoke=a.smoke, root=str(root), boot_draws=SPEC.boot_draws, boot_seed=SPEC.boot_seed,
             spec=dataclasses.asdict(SPEC),
             provenance=dict(git=git, started_utc=started, finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                             argv=list(sys.argv[1:] if argv is None else argv)))
    write_json(summary, v, gp)
    if v["status"] == stats.INVALID:
        print(f"M4 INVALID: {v['reasons']}; wrote {summary}", file=sys.stderr, flush=True)
        return 2
    c2a, c2b = v["2a"], v["2b"]
    ci = lambda c: "n/a" if c.get("lo") is None else f"{c['diff']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}]"
    print(f"(2a) FLY - RND {ci(c2a)} -> {c2a['verdict']}; (2b) FLY - COFF {ci(c2b['vs_coff'])}, "
          f"FLY - RS {ci(c2b['vs_rs'])} -> {c2b['verdict']}; wrote {summary}", flush=True)
    if stats.INVALID in (c2a["verdict"], c2b["verdict"]):
        print(f"INVALID verdict(s): {c2a['reasons'] + c2b['reasons']}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
