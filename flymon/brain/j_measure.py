"""The measurement layer of spec appendix J: H.3's measurer (reference, rest, design, baseline, runaway) and H.4's
(teaching arms, oracle) run unchanged against J's cache, plus the scan's all51 record and D.6 on the judged engine.

Resume (H.3a.12's form): every measurement is cached under a content key — `h3_store.MeasureCache` keyed by the files a
measurement's result depends on (`MEASURE_FILES`), the NPZ, the versions and the inputs (the full Params, which for a
StdParams carries the depression and the receptor scale). The procedure modules are in `HASHED_FILES` (dirty check and
manifest), not in the key.
"""
from __future__ import annotations

from . import d6a, j_jobs
from .h3_measure import chunks
from .h3_store import MEASURE_FILES as H3_MEASURE_FILES
from .h4_measure import HASHED_FILES as H4_HASHED_FILES, MEASURE_FILES as H4_MEASURE_FILES

MEASURE_FILES = tuple(dict.fromkeys(H4_MEASURE_FILES + H3_MEASURE_FILES + (
    "flymon/brain/d6a.py", "flymon/brain/h4_pairs.py", "flymon/brain/j_params.py", "flymon/brain/j_jobs.py",
    "flymon/brain/j_measure.py")))
HASHED_FILES = tuple(dict.fromkeys(MEASURE_FILES + H4_HASHED_FILES + (
    "flymon/brain/h3_c3.py", "flymon/brain/h3_runner.py", "flymon/brain/j_spec.py", "flymon/brain/j_rules.py",
    "flymon/brain/j_runner.py", "flymon/brain/j_setup.py", "flymon/brain/j_store.py", "scripts/run_j_std_scan.py",
    "scripts/run_j_std_judge.py")))


class JMeasurer:
    def __init__(self, pool, spec, all51: tuple, cache):
        """all51: (glomeruli, c_norm) from h3_spec.all51_glomeruli."""
        self.pool, self.spec, self.all51_items, self.cache = pool, spec, all51, cache
        self.params_seen: list = []

    def _seen(self, params):
        if params not in self.params_seen:
            self.params_seen.append(params)

    def all51_uni(self, params) -> list:
        s, h3 = self.spec, self.spec.h4.h3
        self._seen(params)
        gloms, c_norm = self.all51_items
        items = [(g, int(seed)) for g in gloms for seed in h3.all51_seeds]
        common = dict(params=params, c_norm=float(c_norm), strength=h3.strength, settle_ms=h3.all51_window.settle_ms,
                      read_steps=int(h3.all51_window.read_ms), uni_frac=s.uni_frac, uni_min_syn=s.uni_min_syn)
        jobs = [dict(common, items=c) for c in chunks(items, self.pool.n_workers)]
        return self.cache.get_or_compute("j_all51", dict(common, items=items),
                                         lambda: [r for part in self.pool.run_jobs(j_jobs.all51_uni_job, jobs)
                                                  for r in part], [params])

    def d6(self, params, by_turn: dict, seeds) -> list:
        """d6a rows ({turn, seed, max_win, kc_active_frac, kc_spikes}) for every turn x seed, one cache entry per block
        of d6_seed_block seeds (an interruption loses at most one block)."""
        self._seen(params)
        turns, seeds, rows = sorted(by_turn), [int(x) for x in seeds], []
        n = self.spec.d6_seed_block
        for i in range(0, len(seeds), n):
            block = seeds[i:i + n]
            cells = [(t, sd) for t in turns for sd in block]
            jobs = [dict(params=params, odors=by_turn[t], seed=sd) for t, sd in cells]
            inputs = dict(params=params, odors={str(t): by_turn[t] for t in turns}, seeds=block,
                          strength=d6a.STRENGTH, settle_ms=d6a.SETTLE_MS, read_ms=d6a.READ_MS, window_ms=d6a.WINDOW_MS)
            res = self.cache.get_or_compute("j_d6", inputs, lambda: self.pool.run_jobs(j_jobs.d6_job, jobs), [params])
            rows += [{"turn": t, **r} for (t, _), r in zip(cells, res)]
        return rows
