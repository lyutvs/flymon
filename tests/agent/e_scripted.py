"""ScriptedMeasurer: EMeasurer's three methods with deterministic fake values and a .calls log, for runner tests.

activity: frac 0.06 for every odour at every strength (controller ruling R3 — the scripted world has an eligible
strength); oracle: report = tests.brain.h4_scripted.report(testable=...), testable looked up as (tag, axis, turn, x, y) first,
then (axis, turn, x, y) (default False); kc {"jaccard": 0.2}; fail_oracle = set of tags whose oracle call raises once
(an interrupted run); oracle_key: a content key of the call, no cache file; drive: mean_spikes from self.spikes (default 1600.0), kc_frac_mean 0.06."""
from __future__ import annotations

import hashlib
import json

from tests.brain.h4_scripted import report

FRAC = 0.06


class ScriptedMeasurer:
    def __init__(self, testable: dict | None = None, spikes: dict | None = None, n_report: int = 8):
        self.testable = dict(testable or {})
        self.spikes = dict(spikes or {})
        self.n_report = int(n_report)
        self.calls: list = []
        self.fail_oracle: set = set()

    def drive(self, glomeruli: list, c_norm: float, receptor_counts: dict) -> dict:
        self.calls.append(("drive", list(glomeruli), float(c_norm)))
        return {g: dict(mean_spikes=float(self.spikes.get(g, 1600.0)), kc_frac_mean=FRAC) for g in glomeruli}

    def activity(self, odours: dict, s: float, seeds) -> dict:
        seeds = list(seeds)
        self.calls.append(("activity", sorted(odours), float(s), len(seeds)))
        return {oid: dict(frac=[FRAC] * len(seeds), max_win=[3] * len(seeds)) for oid in odours}

    def oracle(self, rows: list, strength: float, readout: dict, z: dict, types, seeds: dict, tag: str) -> list:
        self.calls.append(("oracle", tag, len(rows), float(strength)))
        if tag in self.fail_oracle:
            self.fail_oracle.discard(tag)
            raise RuntimeError(f"oracle {tag}: scripted interruption")
        out = []
        for r in rows:
            key = (r["axis"], r["turn"], r["x"], r["y"])
            t = bool(self.testable.get((tag, *key), self.testable.get(key, False)))
            out.append({**r, "report": report(testable=t, n=self.n_report), "kc": {"jaccard": 0.2}})
        return out

    def oracle_key(self, row: dict, strength: float, readout: dict, z: dict, types, seeds: dict) -> tuple:
        blob = json.dumps(dict(x=row["odor_x"], y=row["odor_y"], s=float(strength),
                               seeds={k: [int(v) for v in seeds[k]] for k in seeds}), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest(), None
