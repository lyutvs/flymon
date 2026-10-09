"""The current battle attempt of every fly (spec AD.5 2). ADBattles begins an attempt before the player plays it and
retires the fly's token in the rollback of an unfinished attempt (before its weights are restored), so a pulse request
made in an attempt that is no longer current - before or after the rollback - fails is_current. Flies are pool
indices. Thread-safe: the pool's write-back asks is_current from an executor thread."""
from __future__ import annotations

import threading


class AttemptBook:
    def __init__(self):
        self._cur: dict = {}
        self._lock = threading.Lock()

    def begin(self, fly: int, battle_id: str, attempt: int) -> list:
        tok = (str(battle_id), int(attempt))
        with self._lock:
            self._cur[int(fly)] = tok
        return list(tok)

    def retire(self, fly: int) -> None:
        with self._lock:
            self._cur.pop(int(fly), None)

    def current(self, fly: int):
        with self._lock:
            return self._cur.get(int(fly))

    def is_current(self, fly: int, token) -> bool:
        if token is None:
            return False
        return self.current(fly) == (str(token[0]), int(token[1]))
