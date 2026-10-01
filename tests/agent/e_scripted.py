"""ScriptedMeasurer: EMeasurer's three methods with deterministic fake values and a .calls log, for runner tests.

activity: frac 0.06 for every odour at every strength (controller ruling R3 — the scripted world has an eligible
strength); oracle: report = tests.brain.h4_scripted.report(testable=self.testable.get((axis, turn, x, y), False)),
kc {"jaccard": 0.2}; drive: mean_spikes from self.spikes (default 1600.0), kc_frac_mean 0.06."""
from __future__ import annotations

from tests.brain.h4_scripted import report

FRAC = 0.06


class ScriptedMeasurer:
    def __init__(self, testable: dict | None = None, spikes: dict | None = None, n_report: int = 8):
        self.testable = dict(testable or {})
        self.spikes = dict(spikes or {})
        self.n_report = int(n_report)
        self.calls: list = []

    def drive(self, glomeruli: list, c_norm: float, receptor_counts: dict) -> dict:
        self.calls.append(("drive", list(glomeruli), float(c_norm)))
        return {g: dict(mean_spikes=float(self.spikes.get(g, 1600.0)), kc_frac_mean=FRAC) for g in glomeruli}

    def activity(self, odours: dict, s: float, seeds) -> dict:
        seeds = list(seeds)
        self.calls.append(("activity", sorted(odours), float(s), len(seeds)))
        return {oid: dict(frac=[FRAC] * len(seeds), max_win=[3] * len(seeds)) for oid in odours}

    def oracle(self, rows: list, strength: float, readout: dict, z: dict, types, seeds: dict, tag: str) -> list:
        self.calls.append(("oracle", tag, len(rows), float(strength)))
        out = []
        for r in rows:
            t = bool(self.testable.get((r["axis"], r["turn"], r["x"], r["y"]), False))
            out.append({**r, "report": report(testable=t, n=self.n_report), "kc": {"jaccard": 0.2}})
        return out
