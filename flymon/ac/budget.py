"""Spec AC.6: stages 1 and 2 together <= 60 h (M0b gate). Stage 1's extrapolation = the larger of the representative
benchmark's s/batch-battle (24 flies, 16 workers, 2 battles, AC.7 2b) and rescope's measured 431.1 s, times stage 1's
batch count (learn 40 + eval 20), plus the serial situation evaluations (benchmark s/fly x 96), times 1.3; above 48 h
is STOP_BUDGET (no reduced path: battle 40 defines criterion 1). Each session is capped at min(24 h, 60 h minus
every stage-1 session so far, aborted ones included); a resume reserves nothing again. Sessions are kept per arm in
<out>/wall_clock.json (rescope's format, plus the AC.5 label). A session hard-killed before its own close (SIGKILL, OOM)
is left "running"; close_stale() closes it as "killed" before the next session opens, charged from its start to the
latest mtime under <out>/checkpoints or <out>/logs (at most the next session's start; 0 if no file)."""
from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

from .spec import LABEL, SPEC
from .store import write_json

BUDGET = "results/summary/ac_budget.json"
BENCH = "results/m4-bench/BRAIN/bench.json"


def batches(spec=SPEC) -> int:
    return int(spec.learn_battles + spec.eval_battles)


def n_sit_evals(spec=SPEC) -> int:
    return sum(n * (1 if arm == "COFF" else len(spec.sit_points)) for arm, n in spec.brain_arms)


def extrapolate(bench: dict, spec=SPEC) -> dict:
    used = max(float(bench["s_per_batch_battle"]), spec.rescope_s_per_batch_battle)
    sit = float(bench["sit_eval_s_per_fly"]) * n_sit_evals(spec)
    hours = (used * batches(spec) + sit) * spec.margin / 3600.0
    return dict(status="OK" if hours <= spec.budget_stage1_h else "STOP_BUDGET", hours=hours,
                limit_h=spec.budget_stage1_h, s_per_batch_battle_used=used,
                s_per_batch_battle_bench=float(bench["s_per_batch_battle"]),
                rescope_s_per_batch_battle=spec.rescope_s_per_batch_battle, batches=batches(spec),
                sit_evals=n_sit_evals(spec), sit_eval_s=sit, margin=spec.margin)


def spent_s(root) -> float:
    total = 0.0
    for p in sorted(Path(root).glob("*/wall_clock.json")):
        total += sum(float(s["seconds"]) for s in json.loads(p.read_text())["sessions"] if s.get("seconds") is not None)
    return total


def session_cap_s(spec, spent: float) -> float:
    return min(spec.session_cap_h * 3600.0, spec.budget_total_h * 3600.0 - float(spent))


def open_session(out, t0: float) -> int:
    p = Path(out) / "wall_clock.json"
    d = json.loads(p.read_text()) if p.exists() else {"sessions": []}
    d["label"] = LABEL
    d["sessions"].append(dict(started_utc=dt.datetime.fromtimestamp(t0, dt.timezone.utc).isoformat(), ended_utc=None,
                              seconds=None, status="running", played=None, complete=None))
    write_json(p, d)
    return len(d["sessions"]) - 1


def close_session(out, idx: int, t0: float, status: str, run) -> float:
    p = Path(out) / "wall_clock.json"
    d = json.loads(p.read_text())
    d["label"] = LABEL
    d["sessions"][idx].update(ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                              seconds=round(time.time() - t0, 2), status=status,
                              played=None if run is None else {k: len(v) for k, v in run["played"].items()},
                              complete=None if run is None else bool(run["complete"]))
    write_json(p, d)
    return round(sum(x["seconds"] for x in d["sessions"] if x.get("seconds") is not None), 2)


def close_stale(out) -> int:
    """Close every session of `out` still "running" (a hard-killed one) as "killed"; returns how many were closed."""
    out = Path(out)
    p = out / "wall_clock.json"
    if not p.exists():
        return 0
    d = json.loads(p.read_text())
    mtimes = [f.stat().st_mtime for sub in ("checkpoints", "logs") for f in (out / sub).rglob("*") if f.is_file()]
    last = max(mtimes, default=None)
    starts = [dt.datetime.fromisoformat(s["started_utc"]).timestamp() for s in d["sessions"]]
    n = 0
    for i, s in enumerate(d["sessions"]):
        if s.get("status") != "running":
            continue
        end = last if last is not None else starts[i]
        if i + 1 < len(starts):
            end = min(end, starts[i + 1])
        sec = max(0.0, end - starts[i])
        s.update(status="killed", seconds=round(sec, 2),
                 ended_utc=dt.datetime.fromtimestamp(starts[i] + sec, dt.timezone.utc).isoformat())
        n += 1
    if n:
        d["label"] = LABEL
        write_json(p, d)
    return n
