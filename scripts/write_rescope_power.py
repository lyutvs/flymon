#!/usr/bin/env python3
"""Re-scoped claim, stage 4 sizing (spec 10.7, plan R8 / R9): pilot variance components -> joint (2b) power -> the
cheapest (F, E) on the grid -> the 60-hour budget.

    uv run python scripts/write_rescope_power.py --m-overlap no [--workers 16]
    uv run python scripts/write_rescope_power.py --smoke --m-overlap no --allow-dirty

Reads results/rescope/pilot/<ARM>/result.json for ARM in FLY RS COFF RND (--smoke: results/rescope-smoke/pilot/...)
and writes results/summary/rescope_power.json (--smoke: results/rescope-smoke/summary/rescope_power.json; never
results/summary/). status is SIZED | STOP_BUDGET | STOP_POWER (flymon.rescope.power.choose); the judge battle CLI
requires SIZED and F / E equal to its --flies / --eval. Exit 0 on SIZED, 2 on a STOP.

Per-arm components come from the evaluation-block win tables (flymon.rescope.stats.win_table: unfinished = loss,
INVALID flies - incl. RS donor_invalid / residual > max / eval weights changed - excluded). Rates (batch model,
plan R11): sec_per_batch_battle_brain = FLY wall_clock_s / (ceil(flies / W_FLY) x (learn + eval)),
sec_per_batch_battle_nobrain = RND wall_clock_s / (ceil(flies / W_RND) x eval), W_arm = the pilot result.json's
`workers`; power.wall_hours scales them by ceil(F / --workers) (the judge's workers, default 16, recorded). RS / COFF
rates are recorded only. Fly decisions per battle (decision records with decider "fly" in
the logs named by result.json["logs"]) are recorded with the decisions implied by the chosen (F, E).

Refusals (SystemExit, nothing written): a pilot arm file missing / unreadable / of another arm, phase or smoke flag /
not complete / malformed (stats.check_arm); fly or eval counts that differ between arms, or (non-smoke) a pilot size
other than R9 (6 flies, learn 40, eval 20); schedule digests that are not the pilot ones (SPEC.schedule_seeds
"pilot", recomputed from the arm's flies / learn / eval) or an eval_schedule.json that differs; fewer than two valid
flies in an arm; an RS fly whose donor trace fails (stats.donor_mismatches: RS donor_sha256, FLY learn_log_sha256,
the sha256 of pilot/FLY/logs/flyNN.jsonl and the two arms' learn schedule digests must agree); for FLY and RND (the rate arms): a wall_clock_s that is missing, None, non-finite or <= 0, a
`workers` that is not an integer >= 1, a session log <arm>/wall_clock.json that is missing / unreadable / empty, any
session without an end record (status ok / aborted, ended_utc, seconds; a hard-killed session's time is unknown), or
session seconds that do not sum to wall_clock_s (never sized on an undercounted or guessed rate); a dirty
flymon/rescope/ or this script without --allow-dirty; an existing non-smoke summary without --force.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import glob
import json
import math
import sys
from pathlib import Path

from flymon.brain.config import Params
from flymon.rescope import blocks, power, stats
from flymon.rescope.blocks import file_sha, read_jsonl
from flymon.rescope.spec import SPEC
from flymon.rescope.store import git_provenance, guard, write_json

PILOT_ARMS = ("FLY", "RS", "COFF", "RND")
LEARNS = ("FLY", "RS")
BRAIN = ("FLY", "RS", "COFF")
R9 = dict(flies=6, learn=40, eval=20)
SCRIPT = "scripts/write_rescope_power.py"
PRE_ESTIMATE = (                                  # spec 10.7's bullet, verbatim (markdown kept; lines joined by \n)
    '**사전 추정(기록, 2026-09-28)**: p ≈ 0.3에서 차이 0.05를 두 비교 동시 0.8로 잡으면 팔마다 평가 배틀이 약 1,700–1,800개다. 초파리 결정은 학습 블록을 더해 약 12만–13만 회이고,'
    '\n'
    'M0b 처리량 환산으로 **약 40–50시간 이상**이다(부분 배치 손실·Showdown 시간·RS 직렬화 제외). 그래서 `STOP_BUDGET`(60시간)이 걸릴 수 있다. 걸리면 멈추고,'
    '\n'
    '최소 관심 효과나 예산의 조정은 **어떤 판정 데이터보다 먼저** 날짜 붙은 개정으로 한다. 최소 관심 효과 0.05는 RND → MAX 폭(0.32, 원 스펙 B.2)의 16%다.')
PRE_ESTIMATE_SOURCE = "docs/superpowers/specs/2026-09-28-rescoped-claim-design.md section 10.7"
RATE_ARMS = ("FLY", "RND")
WALL_TOL_S = 0.05


def paths(smoke: bool) -> tuple:
    root = Path("results/rescope-smoke" if smoke else "results/rescope") / "pilot"
    summ = Path("results/rescope-smoke/summary" if smoke else "results/summary") / "rescope_power.json"
    return root, summ


def _int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _wall(d: dict):
    w = d.get("wall_clock_s")
    ok = isinstance(w, (int, float)) and not isinstance(w, bool) and math.isfinite(w) and w > 0
    return float(w) if ok else None


def expected_digests(flies: int, learn: int, eval_: int, spec=SPEC) -> dict:
    lseed, eseed = {p: (l, e) for p, l, e in spec.schedule_seeds}["pilot"]
    return dict(learn=blocks.schedule_digest(blocks.block_schedule(flies, learn, lseed, blocks.block_tag("pilot", "L"))),
                eval=blocks.schedule_digest(blocks.block_schedule(flies, eval_, eseed, blocks.block_tag("pilot", "E"))))


def load_pilot(root: Path, smoke: bool, spec=SPEC) -> tuple:
    """({arm: result}, inputs); SystemExit listing every reason when the pilot cannot be sized from."""
    arms, reasons, inputs = {}, [], {}
    for arm in PILOT_ARMS:
        p = root / arm / "result.json"
        inputs[arm] = dict(path=str(p), sha256=file_sha(p))
        if not p.exists():
            reasons.append(f"{arm}: {p} does not exist")
            continue
        try:
            d = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError) as e:
            reasons.append(f"{arm}: {p} unreadable ({e!r})")
            continue
        if not isinstance(d, dict):
            reasons.append(f"{arm}: {p} is not a JSON object")
            continue
        want = dict(arm=arm, phase="pilot", smoke=smoke, complete=True)
        bad = {k: (d.get(k), v) for k, v in want.items() if d.get(k) != v}
        if bad:
            reasons.append(f"{arm}: result.json does not match (got, want) {bad}")
        reasons += stats.check_arm(arm, d, spec)
        for k in ("flies", "eval", "learn"):
            if not _int(d.get(k)):
                reasons.append(f"{arm}: {k} {d.get(k)!r} is not an integer")
        inputs[arm]["commit"] = ((d.get("provenance") or {}).get("git") or {}).get("commit")
        arms[arm] = d
    if reasons:
        raise SystemExit("refusing: the pilot cannot be sized from:\n  " + "\n  ".join(reasons))

    sizes = {a: (d["flies"], d["eval"]) for a, d in arms.items()}
    if len(set(sizes.values())) != 1:
        reasons.append(f"pilot arms differ in (flies, eval): {sizes}")
    learns = {a: d["learn"] for a, d in arms.items()}
    if learns["FLY"] != learns["RS"] or learns["FLY"] < 1 or learns["COFF"] or learns["RND"]:
        reasons.append(f"learn counts {learns}: FLY and RS must share one learning block, COFF / RND have none")
    if not smoke:
        for a, d in arms.items():
            got = dict(flies=d["flies"], eval=d["eval"], **({"learn": d["learn"]} if a in LEARNS else {}))
            want = {k: v for k, v in R9.items() if k in got}
            if got != want:
                reasons.append(f"{a}: pilot size {got} is not R9's {want}")
    if not reasons:
        F0, E0 = sizes["FLY"]
        want = expected_digests(F0, learns["FLY"], E0, spec)
        for a, d in arms.items():
            got = d.get("schedule_digests") or {}
            if got.get("eval") != want["eval"]:
                reasons.append(f"{a}: eval schedule digest {str(got.get('eval'))[:12]} is not the pilot one "
                               f"({want['eval'][:12]})")
            wl = want["learn"] if a in LEARNS else None
            if got.get("learn") != wl:
                reasons.append(f"{a}: learn schedule digest {str(got.get('learn'))[:12]} is not the pilot one "
                               f"({str(wl)[:12]})")
        f = root / "eval_schedule.json"
        if f.exists():
            try:
                fd = json.loads(f.read_text()).get("digest")
            except (OSError, json.JSONDecodeError, AttributeError) as e:
                fd = f"unreadable ({e!r})"
            if fd != want["eval"]:
                reasons.append(f"{f} digest {str(fd)[:12]} is not the pilot one ({want['eval'][:12]})")
    for k, why in sorted(stats.donor_mismatches(arms["FLY"], arms["RS"], root / "FLY" / "logs").items()):
        reasons.append(f"RS fly {k}: {why} (spec 10.8: FLY {k} / RS {k} must be rerun together)")
    for a, d in arms.items():
        n = len(stats.win_table(d, spec))
        if n < 2:
            reasons.append(f"{a}: {n} valid flies (a between-fly variance needs >= 2)")
    for a in RATE_ARMS:
        if _wall(arms[a]) is None:
            reasons.append(f"{a}: wall_clock_s {arms[a].get('wall_clock_s')!r} is missing / not a positive number "
                           "(the rate cannot be measured; rerun or record the arm's wall clock)")
        w = arms[a].get("workers")
        if not _int(w) or w < 1:
            reasons.append(f"{a}: workers {w!r} is not an integer >= 1 (the batch model needs the pilot's workers)")
        reasons += session_reasons(a, root / a / "wall_clock.json", arms[a].get("wall_clock_s"))
    if reasons:
        raise SystemExit("refusing: the pilot cannot be sized from:\n  " + "\n  ".join(reasons))
    return arms, inputs


def _sessions(p: Path):
    """The session list of <arm>/wall_clock.json, or a reason string."""
    if not p.exists():
        return f"{p} does not exist"
    try:
        ss = json.loads(p.read_text())["sessions"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as e:
        return f"{p} unreadable ({e!r})"
    if not isinstance(ss, list) or not ss or not all(isinstance(x, dict) for x in ss):
        return f"{p} has no session list"
    return ss


def _ended(x: dict) -> bool:
    sec = x.get("seconds")
    return (x.get("status") in ("ok", "aborted") and isinstance(x.get("ended_utc"), str)
            and isinstance(sec, (int, float)) and not isinstance(sec, bool) and math.isfinite(sec) and sec >= 0)


def session_reasons(arm: str, p: Path, wall) -> list:
    """Why this arm's session log cannot back its wall_clock_s (empty = it can)."""
    ss = _sessions(p)
    if isinstance(ss, str):
        return [f"{arm}: session log {ss} (wall_clock_s cannot be checked for undercounting)"]
    open_ = [i for i, x in enumerate(ss) if not _ended(x)]
    if open_:
        return [f"{arm}: {p} sessions {open_} have no end record (killed hard; their time is unknown, so "
                "wall_clock_s undercounts)"]
    total = sum(x["seconds"] for x in ss)
    if isinstance(wall, (int, float)) and not isinstance(wall, bool) and abs(total - wall) > WALL_TOL_S:
        return [f"{arm}: {p} sessions sum to {total:.2f} s but wall_clock_s is {wall}"]
    return []


def rates_of(arms: dict, root: Path) -> tuple:
    """(rates for power.wall_hours: seconds per battle-per-fly of one batch, per-arm rate record incl. which arms
    supplied the rates). Batch model (plan R11): an arm's wall clock / (ceil(flies / workers) x battles per fly)."""
    rec = {}
    for a, d in arms.items():
        per_fly = d["learn"] + d["eval"]
        w, W = _wall(d), d.get("workers")
        B = power.batches(d["flies"], W) if _int(W) and W >= 1 else None
        ss = _sessions(root / a / "wall_clock.json")
        ss = [] if isinstance(ss, str) else ss
        rec[a] = dict(wall_clock_s=d.get("wall_clock_s"), workers=W, batches=B, battles_per_fly=per_fly,
                      sec_per_batch_battle=None if w is None or B is None else w / (B * per_fly),
                      sessions=len(ss), aborted_sessions=sum(1 for x in ss if x.get("status") == "aborted"),
                      m_overlap_note=d.get("m_overlap_note"))
    rates = dict(sec_per_batch_battle_brain=rec["FLY"]["sec_per_batch_battle"],
                 sec_per_batch_battle_nobrain=rec["RND"]["sec_per_batch_battle"])
    src = dict(sec_per_batch_battle_brain="FLY", sec_per_batch_battle_nobrain="RND", recorded_only=["RS", "COFF"],
               model="batch (plan R11): hours scale with ceil(F / workers)")
    return rates, dict(per_arm=rec, sources=src)


def _log_dir(root: Path, arm: str, d: dict, key: str, default: str) -> Path:
    logs = d.get("logs") if isinstance(d.get("logs"), dict) else {}
    sub = logs.get(key, default)
    return root / arm / (sub if isinstance(sub, str) and sub else default)


def decisions(root: Path, arm: str, d: dict, spec=SPEC) -> dict:
    """Recorded: fly-decided decisions per battle of the valid flies, eval block and (FLY / RS) learning block."""
    out, missing = {}, []
    blocks_ = [("eval", "logs/eval")] + ([("learn", "logs")] if d.get("learn") else [])
    for key, default in blocks_:
        n_dec = n_bat = 0
        for k, r in stats.fly_rows(d):
            if stats.row_invalid(r, spec):
                continue
            p = _log_dir(root, arm, d, key, default) / f"fly{k:02d}.jsonl"
            if not p.exists():
                missing.append(str(p))
                continue
            n_dec += sum(1 for x in read_jsonl(p) if isinstance(x, dict) and x.get("kind") == "decision"
                         and x.get("decider") == "fly")
            n_bat += d["eval"] if key == "eval" else d["learn"]
        out[key] = dict(fly_decisions=n_dec, battles=n_bat, per_battle=n_dec / n_bat if n_bat else None)
    return dict(out, missing_logs=missing)


def projected_decisions(dec: dict, F, E, spec=SPEC):
    """Fly decisions implied by (F, E): FLY and RS learn + eval, COFF eval, at FLY's measured per-battle rates."""
    fl = dec.get("FLY", {})
    le, ev = (fl.get("learn") or {}).get("per_battle"), (fl.get("eval") or {}).get("per_battle")
    if F is None or le is None or ev is None:
        return None
    return 2 * F * spec.learn_battles * le + 3 * F * E * ev


def summarize(root: Path, smoke: bool, m_overlap: str, workers: int = 16, spec=SPEC) -> dict:
    arms, inputs = load_pilot(root, smoke, spec)
    comp = {a: power.components(stats.win_table(d, spec)) for a, d in arms.items()}
    rates, rate_rec = rates_of(arms, root)
    out = power.choose(comp, rates, spec, workers=workers)
    dec = {a: decisions(root, a, arms[a], spec) for a in BRAIN}
    pilot = {a: dict(flies=d["flies"], learn=d["learn"], eval=d["eval"],
                     invalid_flies=sorted(k for k, r in stats.fly_rows(d) if stats.row_invalid(r, spec)),
                     schedule_digests=d.get("schedule_digests")) for a, d in arms.items()}
    note = ("the pilot shared the CPU with M's runs: the rates and the extrapolated hours carry that load"
            if m_overlap == "yes" else None)
    return dict(out, components=comp, rates=rates, rate_record=rate_rec, pilot=pilot, inputs=inputs,
                m_overlap=m_overlap, m_overlap_hours_note=note, workers=workers,
                fly_decisions=dict(per_arm=dec, projected_at_choice=projected_decisions(dec, out["F"], out["E"], spec)),
                grid=dict(F=list(power.GRID_F), E=list(power.GRID_E)),
                pre_estimate=dict(text=PRE_ESTIMATE, source=PRE_ESTIMATE_SOURCE))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m-overlap", choices=("yes", "no"), required=True,
                    help="whether the pilot shared the CPU with M's runs (recorded verbatim)")
    ap.add_argument("--workers", type=int, default=16, help="the judge run's FlyPool workers (batch model, R11)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite an existing results/summary/rescope_power.json")
    a = ap.parse_args(argv)
    if a.workers < 1:
        ap.error("--workers must be >= 1")
    root, summary = paths(a.smoke)
    gp = [Params()]
    guard(summary, gp)
    if summary.exists() and not a.smoke and not a.force:
        raise SystemExit(f"refusing: {summary} exists; pass --force to rewrite it")
    git = git_provenance(files=sorted(glob.glob("flymon/rescope/*.py")) + [SCRIPT])
    if git["dirty"] and not a.allow_dirty:
        raise SystemExit(f"refusing: uncommitted changes in {git['dirty_files']} (commit them or pass --allow-dirty)")
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    v = summarize(root, a.smoke, a.m_overlap, a.workers, SPEC)
    v.update(smoke=a.smoke, root=str(root), power_draws=SPEC.power_draws, seed=SPEC.boot_seed,
             spec=dataclasses.asdict(SPEC),
             provenance=dict(git=git, started_utc=started, finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                             argv=list(sys.argv[1:] if argv is None else argv)))
    write_json(summary, v, gp)
    if v["status"] == "SIZED":
        print(f"SIZED: F={v['F']} E={v['E']} hours={v['hours']:.1f} p_2b={v['p_2b']:.3f}; wrote {summary}", flush=True)
        return 0
    extra = "" if v["hours"] is None else f" (cheapest powered F={v['F']} E={v['E']} hours={v['hours']:.1f})"
    print(f"{v['status']}{extra}; wrote {summary}", file=sys.stderr, flush=True)
    return 2


if __name__ == "__main__":
    sys.exit(main())
