"""FLY-RS pulse queue (spec 10.8): the donor FLY k's non-empty learning-block pulse bundles in log order.

RS fly k pops one bundle per fly-decided turn of its own learning block; the queue carries over across battles.
Turns whose donor outcome gave no pulse never enter the queue. When the queue runs dry the remaining RS fly turns
get no pulse (counted in `exhausted`); bundles left at the end of the learning block are dropped and their share of
the donor's total pulse ms is `residual_frac()`. A share above `residual_max` (0.05) makes the pair INVALID.
The queue state is JSON-serialisable so the battle checkpoint can persist it; `load_state` refuses a state taken
from a different donor log (sha256 mismatch: FLY k was rerun, so RS k must be rerun too)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class YokedQueue:
    def __init__(self, bundles: list, donor_sha256: str):
        self.bundles = [[(str(d), float(ms)) for d, ms in b] for b in bundles]
        if any(not b for b in self.bundles):
            raise ValueError("a yoked queue holds only non-empty pulse bundles")
        self.donor_sha256 = str(donor_sha256)
        self.i = 0                # next bundle to pop
        self.exhausted = 0        # pops on an empty queue (RS fly turns that got no pulse)

    @classmethod
    def from_bundles(cls, bundles, donor_sha256) -> "YokedQueue":
        return cls(bundles, donor_sha256)

    @classmethod
    def from_log(cls, path, battle_ids) -> "YokedQueue":
        raw = Path(path).read_bytes()
        ids = set(battle_ids)
        bundles = []
        for line in raw.decode().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("kind") == "reinforce" and r.get("battle_id") in ids and r.get("pulses"):
                bundles.append(r["pulses"])
        return cls.from_bundles(bundles, hashlib.sha256(raw).hexdigest())

    # ---- consumption ----------------------------------------------------------------------
    def pop(self):
        if self.i >= len(self.bundles):
            self.exhausted += 1
            return None
        self.i += 1
        return list(self.bundles[self.i - 1])

    # ---- accounting -----------------------------------------------------------------------
    def total_ms(self) -> float:
        return float(sum(ms for b in self.bundles for _, ms in b))

    def delivered_ms(self) -> float:
        return float(sum(ms for b in self.bundles[:self.i] for _, ms in b))

    def remaining_ms(self) -> float:
        return float(sum(ms for b in self.bundles[self.i:] for _, ms in b))

    def residual_frac(self) -> float:
        tot = self.total_ms()
        return 0.0 if tot == 0 else self.remaining_ms() / tot

    def dropped_bundles(self) -> int:
        return len(self.bundles) - self.i

    def residual_ok(self, residual_max: float) -> bool:
        """Spec 10.8: the pair is INVALID when the dropped share exceeds residual_max (equal is still valid)."""
        return self.residual_frac() <= residual_max

    def summary(self, residual_max: float) -> dict:
        return {"donor_sha256": self.donor_sha256, "n_bundles": len(self.bundles), "delivered_bundles": self.i,
                "dropped_bundles": self.dropped_bundles(), "exhausted_turns": self.exhausted,
                "total_ms": self.total_ms(), "delivered_ms": self.delivered_ms(),
                "residual_frac": self.residual_frac(), "residual_max": float(residual_max),
                "valid": self.residual_ok(residual_max)}

    # ---- checkpoint -----------------------------------------------------------------------
    def state(self) -> dict:
        return {"i": self.i, "exhausted": self.exhausted, "n_bundles": len(self.bundles),
                "donor_sha256": self.donor_sha256}

    def load_state(self, d: dict) -> None:
        if d["donor_sha256"] != self.donor_sha256:
            raise ValueError("donor log changed since this RS fly started (rerun FLY k and RS k together)")
        if int(d.get("n_bundles", len(self.bundles))) != len(self.bundles) or not 0 <= int(d["i"]) <= len(self.bundles):
            raise ValueError(f"queue state {d} does not fit this donor queue of {len(self.bundles)} bundles")
        self.i = int(d["i"])
        self.exhausted = int(d.get("exhausted", 0))
