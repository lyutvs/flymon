"""Fly-level INVALID and STOP_INFRA (spec AC.8). A battle still unfinished after retry_max retries makes its whole fly
INVALID (rescope blocks' rule; a fly that learned fewer than 40 battles is not analysed); its remaining battles are
skipped. The moment an arm can no longer reach its minimum number of valid flies, the arm stops (STOP_INFRA) and its
remaining flies are skipped too. The book is saved atomically at every change, before the battle's checkpoint commit,
so a resume never plays a battle of a fly the book has already given up. The saved file carries LABEL (AC.5)."""
from __future__ import annotations

import json
from pathlib import Path

from ..rescope.blocks import _atomic_text
from .spec import LABEL


class InvalidBook:
    def __init__(self, path, arm_of: dict, sizes: dict, min_valid: dict):
        self.path = Path(path)
        self.arm_of = {int(k): v for k, v in arm_of.items()}
        self.sizes, self.min_valid = dict(sizes), dict(min_valid)
        d = json.loads(self.path.read_text()) if self.path.exists() else {"invalid": {}, "stopped": {}}
        self.invalid = {int(k): v for k, v in d["invalid"].items()}
        self.stopped = dict(d["stopped"])

    def n_valid(self, arm: str) -> int:
        return self.sizes[arm] - sum(1 for f in self.invalid if self.arm_of[f] == arm)

    def mark(self, fly: int, reason: str) -> None:
        fly = int(fly)
        if fly in self.invalid:
            return
        self.invalid[fly] = str(reason)
        arm = self.arm_of[fly]
        if arm not in self.stopped and self.n_valid(arm) < self.min_valid[arm]:
            self.stopped[arm] = dict(status="STOP_INFRA", n_invalid=self.sizes[arm] - self.n_valid(arm),
                                     size=self.sizes[arm], min_valid=self.min_valid[arm])
        self.save()

    def skip(self, fly: int) -> bool:
        return int(fly) in self.invalid or self.arm_of[int(fly)] in self.stopped

    def status(self) -> dict:
        return dict(invalid={str(k): v for k, v in sorted(self.invalid.items())}, stopped=dict(self.stopped),
                    n_valid={a: self.n_valid(a) for a in self.sizes})

    def save(self) -> None:
        _atomic_text(self.path, json.dumps(dict(invalid={str(k): v for k, v in sorted(self.invalid.items())},
                                                stopped=self.stopped, label=LABEL), sort_keys=True, ensure_ascii=False))
