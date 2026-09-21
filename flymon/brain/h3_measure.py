"""The measurement layer of the M0d H.3 runner: one method per H.3a.3 table row, each served from the cache or
computed on the FlyPool. The procedures in h3_runner / h3_c3 / h3_records only see this interface, so their tests
substitute a scripted measurer and never run the engine."""
from __future__ import annotations

from . import h3_jobs


def chunks(items: list, n_jobs: int) -> list:
    n = max(1, min(int(n_jobs), len(items)))
    size = -(-len(items) // n)
    return [items[i:i + size] for i in range(0, len(items), size)]


class PoolMeasurer:
    def __init__(self, pool, spec, odors: dict, all51: tuple, cache, diag_cache):
        """odors: {"reference": [...], "extended": [...]} from h3_spec.make_odors; all51: (glomeruli, c_norm)."""
        self.pool, self.spec, self.odors, self.all51_items = pool, spec, odors, all51
        self.cache, self.diag_cache = cache, diag_cache
        self.n_jobs = pool.n_workers
        self.params_seen: list = []          # every Params measured: the run report is guarded with all of them

    def _run(self, kind, inputs, fn, jobs, params_list, diag=False):
        for p in params_list:
            if p not in self.params_seen:
                self.params_seen.append(p)
        store = self.diag_cache if diag else self.cache
        return store.get_or_compute(kind, inputs, lambda: [r for part in self.pool.run_jobs(fn, jobs) for r in part],
                                    params_list)

    def reference(self, params, which: str = "reference", csc_edit=None) -> list:
        s, odors = self.spec, self.odors[which]
        common = dict(params=params, strength=s.strength, settle_ms=s.reference_window.settle_ms,
                      read_steps=int(s.reference_window.read_ms), quantiles=tuple(s.apl_v_quantiles),
                      callout=tuple(s.callout_types), csc_edit=csc_edit)
        jobs = [dict(common, odors=c) for c in chunks(odors, self.n_jobs)]
        inputs = dict(common, odors=odors, which=which)
        return self._run("reference", inputs, h3_jobs.reference_job, jobs, [params], diag=csc_edit is not None)

    def rest(self, params, seeds, csc_edit=None) -> list:
        s = self.spec
        common = dict(params=params, settle_ms=s.reference_window.settle_ms,
                      read_steps=int(s.reference_window.read_ms), csc_edit=csc_edit)
        jobs = [dict(common, seeds=c) for c in chunks([int(x) for x in seeds], self.n_jobs)]
        return self._run("rest", dict(common, seeds=[int(x) for x in seeds]), h3_jobs.rest_job, jobs, [params],
                         diag=csc_edit is not None)

    def design(self, params, seeds) -> list:
        s = self.spec
        common = dict(params=params, k=s.design_k, odor_seed=s.design_odor_seed, strength=s.strength,
                      settle_ms=s.design_window.settle_ms, read_ms=s.design_window.read_ms)
        jobs = [dict(common, seeds=[int(x)]) for x in seeds]
        return self._run("design", dict(common, seeds=[int(x) for x in seeds]), h3_jobs.design_job, jobs, [params])

    def baseline(self, params, seeds, sat_hz: float | None = None) -> list:
        s = self.spec
        common = dict(params=params, ms=s.baseline_ms, sat_hz=float(s.baseline_sat_hz if sat_hz is None else sat_hz))
        jobs = [dict(common, seeds=[int(x)]) for x in seeds]
        return self._run("baseline", dict(common, seeds=[int(x) for x in seeds]), h3_jobs.baseline_job, jobs,
                         [params])

    def odor_runaway(self, params, seeds) -> list:
        s = self.spec
        common = dict(params=params, k=s.design_k, odor_seed=s.design_odor_seed, strength=s.strength,
                      settle_ms=s.runaway_odor_window.settle_ms, read_ms=s.runaway_odor_window.read_ms,
                      sat_hz=s.runaway_odor_sat_hz, record_hz=s.runaway_odor_record_hz)
        jobs = [dict(common, seeds=[int(x)]) for x in seeds]
        return self._run("odor_runaway", dict(common, seeds=[int(x) for x in seeds]), h3_jobs.odor_runaway_job,
                         jobs, [params])

    def all51(self, params) -> list:
        s = self.spec
        gloms, c_norm = self.all51_items
        items = [(g, int(seed)) for g in gloms for seed in s.all51_seeds]
        common = dict(params=params, c_norm=float(c_norm), strength=s.strength,
                      settle_ms=s.all51_window.settle_ms, read_steps=int(s.all51_window.read_ms))
        jobs = [dict(common, items=c) for c in chunks(items, self.n_jobs)]
        return self._run("all51", dict(common, items=items), h3_jobs.all51_job, jobs, [params])
