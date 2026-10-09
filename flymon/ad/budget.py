"""AD's budget (spec AD.4, AD.8 11). Total 60 h; stage 1 spent 41 695.3 s; AD's limit is 60 h - 11.582 h - 1 h reserve =
47.418 h. Bench and smoke time are not charged.
- Sessions live in <out>/wall_clock.json (AC's open_session / close_session format). A session hard-killed before its
  own close is left "running"; close_stale() (called at the next session's start) closes it as "killed" and charges it
  from its start to the next recorded session's start, or to `now` (the session opening now = detection) - AD's rule,
  not stage 1's last-mtime rule.
- Session cap = min(24 h, the limit minus every AD session so far); a resume reserves nothing again.
- Extrapolation: hours = (max(bench median s/batch-battle, 431.1) x (T + 20) + bench median s/evaluation x the
  situation evaluations of (T, new set)) x 1.3 / 3600. T = 120 with the new set if it fits, else 120 without it, else 80
  (with the set, else without), else STOP_BUDGET. Futility shortening is not counted.
- Bench median: the first 3 valid attempts (validity: Task 11), each of s/batch-battle and s/evaluation; fewer than 3
  valid after 6 attempts is STOP_BENCH.
- BatchGate (AD.4 '상한 근처의 배치'): a batch-battle (block, schedule index k) starts only if both the session and the
  total budget left cover its worst case = worst_factor x the longest batch-battle so far in the run (first: x the bench
  median; persisted in batch_times.json) + the bench median s/evaluation x the situation evaluations at point k + 1 (a
  learning batch ending on a point). Once a batch is open its other flies' battles are not checked again. A refusal by
  the total budget is STOP_BUDGET (the run stops; the reading uses the latest common point); by the session cap only, the
  session closes at a batch boundary for --resume. A batch-battle's span runs from its first start to its last finish
  (run_ad.py calls finished() before the battle's point evaluations, so a span excludes its own point's evaluations -
  worst() adds them from the bench). On a resume, reopen() marks every batch with a committed battle as opened."""
from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import numpy as np

from .inputs import battle_index
from .spec import LABEL
from .store import write_json

BUDGET = "results/summary/ad_budget.json"


def _ts(iso: str) -> float:
    return dt.datetime.fromisoformat(iso).timestamp()


def close_stale(out, now: float) -> int:
    p = Path(out) / "wall_clock.json"
    if not p.exists():
        return 0
    d = json.loads(p.read_text())
    starts = [_ts(s["started_utc"]) for s in d["sessions"]]
    n = 0
    for i, s in enumerate(d["sessions"]):
        if s.get("status") != "running":
            continue
        nxt = i + 1 < len(starts)
        end = starts[i + 1] if nxt else float(now)
        sec = max(0.0, end - starts[i])
        s.update(status="killed", seconds=round(sec, 2), charged="to_next_start" if nxt else "to_detection",
                 ended_utc=dt.datetime.fromtimestamp(starts[i] + sec, dt.timezone.utc).isoformat())
        n += 1
    if n:
        d["label"] = LABEL
        write_json(p, d)
    return n


def spent_s(out) -> float:
    p = Path(out) / "wall_clock.json"
    if not p.exists():
        return 0.0
    return float(sum(float(s["seconds"]) for s in json.loads(p.read_text())["sessions"] if s.get("seconds") is not None))


def remaining_s(spec, spent: float) -> float:
    return spec.budget_limit_h() * 3600.0 - float(spent)


def session_cap_s(spec, spent: float) -> float:
    return min(spec.session_cap_h * 3600.0, remaining_s(spec, spent))


def hours(spec, s_batch: float, s_eval: float, T: int, new_set: bool) -> float:
    used = max(float(s_batch), spec.rescope_s_per_batch_battle)
    return (used * spec.batches(T) + float(s_eval) * spec.sit_evals(T, new_set)) * spec.margin / 3600.0


def choose(spec, s_batch: float, s_eval: float) -> dict:
    T, F = spec.learn_battles, spec.fallback_battles
    order = ((f"T{T}_new", T, True), (f"T{T}", T, False), (f"T{F}_new", F, True), (f"T{F}", F, False))
    cases = {name: hours(spec, s_batch, s_eval, t, new) for name, t, new in order}
    limit = spec.budget_limit_h()
    base = dict(cases=cases, limit_h=limit, s_per_batch_battle_used=max(float(s_batch), spec.rescope_s_per_batch_battle),
                s_per_batch_battle_bench=float(s_batch), sit_eval_s=float(s_eval),
                rescope_s_per_batch_battle=spec.rescope_s_per_batch_battle, margin=spec.margin,
                sit_evals={name: spec.sit_evals(t, new) for name, t, new in order},
                batches={name: spec.batches(t) for name, t, _ in order})
    for name, t, new in order:
        if cases[name] <= limit:
            return dict(base, status="OK", T=t, new_set=new, case=name, hours=cases[name])
    return dict(base, status="STOP_BUDGET", T=None, new_set=None, case=None, hours=None)


def bench_summary(attempts: list, spec) -> dict:
    valid = [a for a in attempts if a["valid"]][: spec.bench_valid]
    vals = dict(s_per_batch_battle=[float(a["s_per_batch_battle"]) for a in valid],
                sit_eval_s=[float(a["sit_eval_s_per_eval"]) for a in valid])
    if len(valid) == spec.bench_valid:
        return dict(status="OK", used=[a["attempt"] for a in valid], values=vals,
                    s_per_batch_battle=float(np.median(vals["s_per_batch_battle"])),
                    sit_eval_s=float(np.median(vals["sit_eval_s"])), n_attempts=len(attempts))
    status = "STOP_BENCH" if len(attempts) >= spec.bench_max_attempts else "MORE"
    return dict(status=status, used=[a["attempt"] for a in valid], values=vals, s_per_batch_battle=None,
                sit_eval_s=None, n_attempts=len(attempts))


class BatchGate:
    def __init__(self, spec, T: int, new_set: bool, *, session_end: float, budget_end: float, bench_batch_s: float,
                 bench_eval_s: float, times_path, now=time.time):
        self.spec, self.T, self.new_set = spec, int(T), bool(new_set)
        self.session_end, self.budget_end = float(session_end), float(budget_end)
        self.bench_batch_s, self.bench_eval_s = float(bench_batch_s), float(bench_eval_s)
        self.times_path, self.now = Path(times_path), now
        d = json.loads(self.times_path.read_text()) if self.times_path.exists() else {}
        self.longest = float(d.get("longest_s", 0.0))
        self.opened: set = set()
        self.spans: dict = {}
        self.refused, self.refusal = None, None

    def reopen(self, block: str, battle_ids) -> int:
        """A resume: every batch with a committed battle counts as opened (opened is in memory only), so a batch half
        played in an earlier session is finished by its lagging flies, never refused. Returns how many were added."""
        keys = {(block, battle_index(b)) for b in battle_ids} - self.opened
        self.opened |= keys
        return len(keys)

    def worst(self, block: str, index: int) -> float:
        base = self.spec.worst_factor * (self.longest if self.longest > 0 else self.bench_batch_s)
        n = self.spec.sit_evals_at(int(index) + 1, self.T, self.new_set) if block == "L" else 0
        return base + self.bench_eval_s * n

    def __call__(self, block: str, sb) -> bool:
        key = (block, battle_index(sb.battle_id))
        if key in self.opened:
            return False
        if self.refused is not None:
            return True
        t, w = self.now(), self.worst(*key)
        for reason, end in (("STOP_BUDGET", self.budget_end), ("session_cap", self.session_end)):
            if end - t < w:
                self.refused = reason
                self.refusal = dict(block=block, index=key[1], worst_s=w, left_s=end - t, reason=reason)
                return True
        self.opened.add(key)
        self.spans[key] = [t, None]
        return False

    def finished(self, block: str, sb) -> None:
        key = (block, battle_index(sb.battle_id))
        span = self.spans.get(key)
        if span is None:
            return
        t = self.now()
        span[1] = t if span[1] is None else max(span[1], t)
        dur = span[1] - span[0]
        if dur > self.longest:
            self.longest = dur
            write_json(self.times_path, dict(longest_s=dur, block=block, index=key[1], label=LABEL))
