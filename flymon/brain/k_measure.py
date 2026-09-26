"""The measurement layer of spec appendix K: H.3's and H.4's measurers run unchanged (through J's files), plus the
per-KC activity on the odd-turn (b) pairs and one raster (K.8.2).

Resume (H.3a.12's form): every measurement is cached under a content key — `h3_store.MeasureCache` keyed by the files a
result depends on (`MEASURE_FILES`), the NPZ and the inputs (the full Params, which carry g). The procedure modules
(metrics, rules, runner, store, scripts) are in `HASHED_FILES` (dirty check and manifest), not in the key."""
from __future__ import annotations

import numpy as np

from . import k_jobs
from .h3_measure import chunks
from .j_measure import HASHED_FILES as J_HASHED_FILES, MEASURE_FILES as J_MEASURE_FILES
from .k_pairs import odour_key

MEASURE_FILES = tuple(dict.fromkeys(J_MEASURE_FILES + (
    "flymon/brain/k_params.py", "flymon/brain/k_pairs.py", "flymon/brain/k_jobs.py", "flymon/brain/k_measure.py")))
HASHED_FILES = tuple(dict.fromkeys(MEASURE_FILES + J_HASHED_FILES + (
    "flymon/brain/k_spec.py", "flymon/brain/k_metrics.py", "flymon/brain/k_rules.py", "flymon/brain/k_runner.py",
    "flymon/brain/k_store.py", "scripts/run_k_scan.py", "scripts/run_k_judge.py")))


class KMeasurer:
    def __init__(self, pool, spec, pairs: list, n_kc: int, cache):
        """pairs: k_pairs.odd_pairs; identical odours (odour_key) are measured once and shared."""
        self.pool, self.spec, self.pairs, self.n_kc, self.cache = pool, spec, pairs, int(n_kc), cache
        self.params_seen: list = []
        self.odours, self.index, seen = [], [], {}
        for p in pairs:
            ij = []
            for k in ("odor_x", "odor_y"):
                key = odour_key(p[k])
                if key not in seen:
                    seen[key] = len(self.odours)
                    self.odours.append({str(g): float(v) for g, v in p[k].items()})
                ij.append(seen[key])
            self.index.append(tuple(ij))

    def _seen(self, params):
        if params not in self.params_seen:
            self.params_seen.append(params)

    def _common(self, params) -> dict:
        h4 = self.spec.j.h4
        return dict(params=params, strength=h4.h3.strength, settle_ms=h4.oracle_window.settle_ms,
                    read_ms=h4.oracle_window.read_ms, window_ms=int(h4.kc_window_ms))

    def activity(self, params) -> list:
        """[{fx, fy, cx}] per pair: fire fraction over act seeds and mean read-window count (cx for X)."""
        self._seen(params)
        seeds = [int(s) for s in self.spec.j.h4.act_seeds]
        common = self._common(params)
        items = [(i, o, s) for i, o in enumerate(self.odours) for s in seeds]
        jobs = [dict(common, items=c) for c in chunks(items, self.pool.n_workers)]
        rows = self.cache.get_or_compute("k_act", dict(common, odours=self.odours, seeds=seeds),
                                         lambda: [r for part in self.pool.run_jobs(k_jobs.activity_job, jobs)
                                                  for r in part], [params])
        fired = np.zeros((len(self.odours), self.n_kc))
        counts = np.zeros((len(self.odours), self.n_kc))
        for r in rows:
            fired[r["i"], r["kc"]] += 1
            counts[r["i"], r["kc"]] += r["n"]
        fired /= len(seeds)
        counts /= len(seeds)
        return [dict(fx=fired[ix], fy=fired[iy], cx=counts[ix]) for ix, iy in self.index]

    def raster(self, params, odor: dict, seed: int, probe) -> dict:
        self._seen(params)
        c = self._common(params)
        kw = dict(params=params, odor=odor, seed=int(seed), strength=c["strength"], settle_ms=c["settle_ms"],
                  read_ms=c["read_ms"], probe=[int(x) for x in probe])
        return self.cache.get_or_compute("k_raster", kw, lambda: self.pool.run_jobs(k_jobs.raster_job, [kw])[0],
                                         [params])
