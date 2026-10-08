"""Stage 1 of M4 under AC (spec AC.2, AC.5, AC.8): the brain group's layout (global flies FLY 0-11, C-off 12-17,
FLY-TB 18-23 on one pool; RND / MAX use arm-local ids 0..n-1 in their own directories), its schedules (arm-local fly k
plays learning row k; every fly of every arm plays the one evaluation schedule, AC.4 ruling), the skip of INVALID flies
and stopped arms, the situation-pair evaluations (each record carries LABEL, AC.5), and the result rows.
- Point 0: Stage1Hooks.initial() evaluates every fly lacking a point-0 record on its naive weights (before run_arm
  loads any checkpoint), into logs/situations_init.jsonl, kept across resumes (deterministic from w0).
- Points 10 / 20 / 30 / 40: after_battle evaluates a FLY / FLY-TB fly right after its learning battle with index
  point - 1 settles and before that battle's checkpoint commit, into logs/situations.jsonl keyed by that battle id; a
  resume filters the file to committed battles first, so an evaluation whose battle was not committed is redone.
  C-off is evaluated at 0 only (AC.2: it is deterministic; no column is duplicated).
- An invalid battle (retries exhausted) marks its fly in the InvalidBook (AC.8); a battle skipped only because the
  fly's arm already stopped (STOP_INFRA) marks it with the distinct reason "arm_stopped", not a battle failure."""
from __future__ import annotations

import json
from pathlib import Path

from ..agent.checkpoint import CheckpointStore, filter_log
from ..rescope import blocks
from . import schedules
from .spec import LABEL


def layout(spec) -> list:
    return [(arm, k) for arm, n in spec.brain_arms for k in range(n)]


def brain_schedules(spec, doc) -> tuple:
    lay = layout(spec)
    learn = schedules.assign(blocks.schedule_from_rows(doc["learn"]), [(g, k) for g, (_, k) in enumerate(lay)],
                             spec.tags[0])
    # AC.4 ruling: ONE evaluation schedule (canonical row 0) played by every fly of every arm
    ev = schedules.assign(blocks.schedule_from_rows(doc["eval"]), [(g, 0) for g in range(len(lay))], spec.tags[1])
    return learn, ev


def nobrain_layout(spec, arm: str) -> list:
    return [(arm, k) for k in range(spec.arm_sizes()[arm])]


def nobrain_schedule(spec, doc, arm: str) -> list:
    members = [(k, 0) for _, k in nobrain_layout(spec, arm)]   # arm-local ids; the one evaluation schedule (AC.4)
    return schedules.assign(blocks.schedule_from_rows(doc["eval"]), members, spec.tags[1])


def skipping(attempt_for, book):
    def wrapped(block):
        attempt = attempt_for(block)

        async def att(sb, n):
            if book.skip(sb.fly_id):
                return {"finished": False, "won": None, "skipped": True}
            return await attempt(sb, n)
        return att
    return wrapped


def committed(out, key: str) -> set:
    ck = CheckpointStore(Path(out) / "checkpoints" / "learn", key).load()
    return set(ck["completed"]) if ck else set()


class SituationLog:
    def __init__(self, logs_dir):
        self.init_path = Path(logs_dir) / "situations_init.jsonl"
        self.path = Path(logs_dir) / "situations.jsonl"

    def repair_init(self) -> int:
        """Drop unparsable lines (a torn last line from a kill mid-write) and end the file with a newline, so the next
        append is not glued onto a torn record; returns the number of lines dropped."""
        if not self.init_path.exists():
            return 0
        text = self.init_path.read_text()
        keep = []
        for line in text.splitlines():
            try:
                json.loads(line)
            except json.JSONDecodeError:
                continue
            keep.append(line)
        fixed = "".join(x + "\n" for x in keep)
        if fixed != text:
            blocks._atomic_text(self.init_path, fixed)
        return len(text.splitlines()) - len(keep)

    def has_init(self, fly: int) -> bool:
        return any(r.get("fly") == int(fly) for r in blocks.read_jsonl(self.init_path))

    def add_init(self, rec: dict) -> None:
        blocks.append_jsonl(self.init_path, rec)

    def add(self, rec: dict) -> None:
        blocks.append_jsonl(self.path, rec)

    def filter(self, completed: set) -> int:
        return filter_log(self.path, set(completed))

    def records(self) -> list:
        return blocks.read_jsonl(self.init_path) + blocks.read_jsonl(self.path)


class Stage1Hooks:
    def __init__(self, spec, lay, book, sitlog: SituationLog, evaluate):
        self.spec, self.lay, self.book, self.sitlog, self.evaluate = spec, list(lay), book, sitlog, evaluate

    def _record(self, g: int, point: int, battle_id) -> dict:
        rec = self.evaluate(g, point)
        if not rec.get("frozen", False):
            raise SystemExit(f"fly {g}: weights changed during the point-{point} situation evaluation (AC.5)")
        return dict(rec, arm=self.lay[g][0], battle_id=battle_id, label=LABEL)

    def initial(self) -> None:
        self.sitlog.repair_init()
        for g in range(len(self.lay)):
            if not self.book.skip(g) and not self.sitlog.has_init(g):
                self.sitlog.add_init(self._record(g, 0, None))

    def after_battle(self, block: str, sb, rec: dict) -> None:
        g = int(sb.fly_id)
        if rec["invalid"]:
            # a fly skipped only because its arm already stopped (STOP_INFRA) is not a battle failure; an already
            # INVALID fly keeps its first reason (mark is a no-op)
            self.book.mark(g, "arm_stopped" if self.book.skip(g) else
                           f"{sb.battle_id}: unfinished after {rec['retries']} retries")
            return
        if block != "L" or self.book.skip(g) or self.lay[g][0] == "COFF":
            return
        point = schedules.battle_index(sb.battle_id) + 1
        if point in self.spec.sit_points:
            self.sitlog.add(self._record(g, point, sb.battle_id))


def per_fly_rows(out, lay, *, eval_, learn, run, book, w0_sha: str, medians=None) -> list:
    rows = blocks.arm_per_fly(out, n_flies=len(lay), eval_=eval_, run=run, learn=learn, spec=None, brain=True,
                              weight_medians=medians)
    for g, row in enumerate(rows):
        arm, k = lay[g]
        row.update(arm=arm, k=k)
        if arm == "COFF":
            row["coff_weights_unchanged"] = run["before"].get(g) == w0_sha
            if not row["coff_weights_unchanged"]:
                book.mark(g, "coff_weights_changed")
        if row["invalid"] and g not in book.invalid:
            book.mark(g, "battle_invalid" if row.get("battle_invalid") else "weights_changed_in_eval")
        row["invalid"] = bool(row["invalid"] or g in book.invalid)
        row["invalid_reason"] = book.invalid.get(g)
    return rows


def nobrain_rows(out, lay, *, eval_, run, book) -> list:
    rows = blocks.arm_per_fly(out, n_flies=len(lay), eval_=eval_, run=run, brain=False)
    for g, row in enumerate(rows):
        row.update(arm=lay[g][0], k=lay[g][1])
        if row["invalid"] and g not in book.invalid:
            book.mark(g, "battle_invalid")
        row["invalid"] = bool(row["invalid"] or g in book.invalid)
        row["invalid_reason"] = book.invalid.get(g)
    return rows
