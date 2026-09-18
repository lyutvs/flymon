"""A stand-in for the M3 brain on the viewer path only. Never use it in an experiment arm.

It keeps the inner provider's choice, then does what the brain will do for the viewer: fills
`context["detail"]` and sends one trace per candidate presentation through a WebSink. The numbers are
made up; what is real is the path they travel.
"""
from __future__ import annotations

import numpy as np

from .trace import WebSink

POPULATIONS = ("alpn", "kc", "mbon", "apl")


class FakeBrainProvider:
    def __init__(self, inner, event_sink, fly: str, seed: int = 0, settle_ms: float = 800.0,
                 read_ms: float = 600.0, every_ms: float = 100.0, n_mbon: int = 6):
        self.inner, self.fly = inner, fly
        self.web = WebSink(event_sink, fly)
        self.rng = np.random.default_rng(seed)
        self.settle_ms, self.read_ms, self.every_ms, self.n_mbon = settle_ms, read_ms, every_ms, n_mbon
        self.clock_ms = 0.0                 # runs on across presentations, like SpikeTap's clock

    async def decide(self, battle, candidates, context) -> int:
        idx = await self.inner.decide(battle, candidates, context)
        n = len(candidates)
        V = self.rng.normal(0.0, 0.5, n)
        V[idx] = V.max() + 0.25             # the made-up values agree with the choice
        p = np.exp(V) / np.exp(V).sum()
        kc = self.rng.uniform(0.04, 0.08, n)
        context["detail"] = {"V": np.round(V, 3), "p": np.round(p, 3), "kc_active_frac": np.round(kc, 4)}
        for slot in range(n):
            self._present(slot, kc[slot])
            self.web.flush(context["battle_tag"], context["turn"], "decide", slot)
        return idx

    def _present(self, slot: int, kc_frac: float) -> None:
        on = self.clock_ms
        steps = int(round((self.settle_ms + self.read_ms) / self.every_ms))
        self.web.text("events", on, f"candidate {slot} on")
        cells = np.zeros(self.n_mbon)
        for k in range(1, steps + 1):
            t = on + k * self.every_ms
            rise = min(1.0, k * self.every_ms / 300.0)
            base = {"alpn": 12.0, "kc": 40.0 * kc_frac, "mbon": 3.5, "apl": 150.0}
            for pop in POPULATIONS:
                self.web.scalar(f"rate_hz/{pop}", t, base[pop] * (0.4 + 0.6 * rise) * self.rng.uniform(0.85, 1.15))
            cells = np.clip(3.5 + self.rng.normal(0.0, 1.5, self.n_mbon) + 4.0 * rise * (slot % 2), 0.0, None)
        self.web.bars("cell_hz/mbon", on + steps * self.every_ms, cells)
        self.clock_ms = on + steps * self.every_ms
